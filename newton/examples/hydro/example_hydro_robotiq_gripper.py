# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Robotiq Gripper
#
# Simple self-contained example that loads the Robotiq 2F-85 gripper
# (V1 or V4) on a D6 joint base above a table, with a small box to
# grasp. Supports multiple collision pipelines: MuJoCo native, Newton
# GJK/MPR, Newton SDF, and Newton hydroelastic.
#
# Collision mode and gripper version are set as instance variables at
# the top of __init__ for quick switching. An automatic state machine
# drives the gripper through approach → close → lift → hold, or a
# manual mode checkbox enables GUI sliders for direct control.
#
# The state machine, rate limiting, and target setting all run as GPU
# kernels — no numpy round-trips in the control loop.
#
# Command: python -m newton.examples hydro_robotiq_gripper
#
###########################################################################

import os
import shutil
import xml.etree.ElementTree as ET
from enum import Enum, IntEnum

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton._src.utils.download_assets import download_git_folder
from newton.geometry import HydroelasticSDF


class TaskType(IntEnum):
    """State-machine tasks for the automatic grasp sequence."""

    APPROACH = 0
    CLOSE_GRIPPER = 1
    LIFT = 2
    HOLD = 3


# ---- Warp kernels ----


@wp.func
def s_curve_profile(t: float, T: float, ramp_fraction: float) -> float:
    """S-curve trapezoidal position profile, normalized to [0, 1].

    Three phases:
      1. Accel ramp  [0, t_r]:       smooth 0 → v_max using half-cosine
      2. Cruise       [t_r, T-t_r]:  constant v_max
      3. Decel ramp  [T-t_r, T]:     smooth v_max → 0 using half-cosine

    The profile integrates to exactly 1.0 over [0, T], so
    ``start + s_curve_profile(t, T, f) * (end - start)`` interpolates
    from ``start`` to ``end`` with zero velocity at both endpoints and
    bounded acceleration throughout.

    Args:
        t: Current time [s].
        T: Total duration [s].
        ramp_fraction: Fraction of T used for each ramp (0, 0.5).
            E.g., 0.25 → 25% accel, 50% cruise, 25% decel.
    """
    t = wp.clamp(t, 0.0, T)
    f = wp.clamp(ramp_fraction, 0.01, 0.5)
    t_r = f * T
    # v_max chosen so total displacement = 1.0:  v_max = 1 / (T - t_r)
    v_max = 1.0 / (T - t_r)

    if t < t_r:
        # Phase 1: accel ramp — integral of v_max * 0.5*(1 - cos(π*t/t_r))
        return v_max * (t * 0.5 - t_r / (2.0 * wp.pi) * wp.sin(wp.pi * t / t_r))
    elif t < T - t_r:
        # Phase 2: cruise — displacement from phase 1 end + linear cruise
        p1_end = v_max * t_r * 0.5  # integral of phase 1 at t=t_r
        return p1_end + v_max * (t - t_r)
    else:
        # Phase 3: decel ramp — mirror of phase 1
        t_decel = t - (T - t_r)  # time into decel phase [0, t_r]
        p12_end = v_max * t_r * 0.5 + v_max * (T - 2.0 * t_r)  # end of cruise
        # Integral of v_max * 0.5*(1 + cos(π*t_d/t_r)) from 0 to t_decel
        return p12_end + v_max * (t_decel * 0.5 + t_r / (2.0 * wp.pi) * wp.sin(wp.pi * t_decel / t_r))


def create_rounded_box_mesh(hx: float, hy: float, hz: float, radius: float, subdivisions: int = 3):
    """Create a rounded box mesh using convex hull of corner spheres.

    Places icospheres at the 8 inner corners of the box and takes their
    convex hull, producing a watertight mesh with smooth edges and corners.

    Args:
        hx, hy, hz: Half-extents of the box [m].
        radius: Rounding radius [m]. Must be < min(hx, hy, hz).
        subdivisions: Icosphere subdivisions (higher = smoother). Default 3.

    Returns:
        Newton Mesh with smooth edges, or None if trimesh is not available.
    """
    try:
        import trimesh  # noqa: PLC0415
    except ImportError:
        print("Warning: trimesh not available, falling back to regular box")
        return None

    inner = np.array([hx - radius, hy - radius, hz - radius])
    corners = np.array([[sx, sy, sz] for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]]) * inner
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

    all_verts = np.vstack([sphere.vertices + c for c in corners])
    hull = trimesh.convex.convex_hull(all_verts)

    return newton.Mesh(
        vertices=hull.vertices.astype(np.float32),
        indices=hull.faces.astype(np.int32).flatten(),
        compute_inertia=True,
    )


@wp.kernel(enable_backward=False)
def control_pipeline_kernel(
    # Mode (GPU buffer, written from CPU on toggle)
    manual_mode: wp.array(dtype=int),
    # GUI inputs (manual mode, GPU buffers)
    gui_pos: wp.array(dtype=wp.vec3),
    gui_rot: wp.array(dtype=wp.vec3),
    gui_ctrl: wp.array(dtype=float),
    # State machine state (auto mode, read/write)
    task: wp.array(dtype=int),
    task_timer: wp.array(dtype=float),
    # State machine config
    grasp_z: float,
    lift_z: float,
    grasp_ctrl: float,
    task_durations: wp.array(dtype=float),
    frame_dt: float,
    hold_rot: wp.vec3,
    # Rate limiter state (read/write)
    prev_base_target: wp.array2d(dtype=float),
    prev_gripper_ctrl: wp.array(dtype=float),
    # Rate limit config (GPU buffers, written from CPU on slider change)
    vel_limits: wp.array(dtype=wp.vec3),  # (1,): [pos_max_delta, rot_max_delta, gripper_max_delta]
    # Outputs
    joint_target_pos: wp.array(dtype=float),
    direct_control: wp.array2d(dtype=float),
    dofs_per_world: int,
    gripper_ctrl_idx: int,
):
    tid = wp.tid()

    # ---- Compute desired targets ----
    if manual_mode[0] == 1:
        d_pos = gui_pos[0]
        d_rot = gui_rot[0]
        d_ctrl = gui_ctrl[0]
    else:
        p = task[tid]
        task_timer[tid] = task_timer[tid] + frame_dt

        desired_z = grasp_z
        if p == TaskType.LIFT.value:
            # S-curve trapezoidal profile: smooth accel → cruise → smooth decel
            # ramp_fraction=0.25 → 25% accel, 50% cruise, 25% decel
            alpha = s_curve_profile(task_timer[tid], task_durations[TaskType.LIFT.value], 0.25)
            desired_z = grasp_z + alpha * (lift_z - grasp_z)
        elif p == TaskType.HOLD.value:
            desired_z = lift_z

        d_ctrl = 0.0
        if p == TaskType.CLOSE_GRIPPER.value or p == TaskType.LIFT.value or p == TaskType.HOLD.value:
            d_ctrl = grasp_ctrl

        if task_timer[tid] > task_durations[p] and p < TaskType.HOLD.value:
            task[tid] = p + 1
            task_timer[tid] = 0.0

        d_pos = wp.vec3(0.0, 0.0, desired_z)
        d_rot = hold_rot

    # ---- Rate-limit and write outputs ----
    limits = vel_limits[0]
    pos_max_delta = limits[0]
    rot_max_delta = limits[1]
    gripper_max_delta = limits[2]

    offset = tid * dofs_per_world

    for i in range(3):
        delta = wp.clamp(d_pos[i] - prev_base_target[tid, i], -pos_max_delta, pos_max_delta)
        clamped = prev_base_target[tid, i] + delta
        prev_base_target[tid, i] = clamped
        joint_target_pos[offset + i] = clamped

    for i in range(3):
        delta = wp.clamp(d_rot[i] - prev_base_target[tid, 3 + i], -rot_max_delta, rot_max_delta)
        clamped = prev_base_target[tid, 3 + i] + delta
        prev_base_target[tid, 3 + i] = clamped
        joint_target_pos[offset + 3 + i] = clamped

    prev = prev_gripper_ctrl[tid]
    gdelta = wp.clamp(d_ctrl - prev, -gripper_max_delta, gripper_max_delta)
    clamped_ctrl = prev + gdelta
    prev_gripper_ctrl[tid] = clamped_ctrl
    direct_control[tid, gripper_ctrl_idx] = clamped_ctrl


def _patch_solimp(tree: ET.ElementTree) -> None:
    """Patch equality constraint solimp[0] (dmin) from 0.95 to 0.5 in-place.

    solimp[0] controls impedance at zero constraint violation. The default 0.95
    causes aggressive enforcement that amplifies small perturbations in the
    coupled finger linkage, leading to oscillation. Lowering dmin to 0.5 makes
    the constraint more compliant near zero violation, damping oscillatory modes
    while preserving grasp accuracy. Must be done pre-parse: equality params are
    baked during MJCF loading. MuJoCo issue 906.
    Ref: https://github.com/google-deepmind/mujoco/issues/906#issuecomment-1849032881
    """
    for eq_elem in tree.iter("equality"):
        for child in eq_elem:
            solimp = child.get("solimp")
            if solimp:
                parts = solimp.split()
                parts[0] = "0.5"
                child.set("solimp", " ".join(parts))


def _patch_v1_mjcf(mjcf_path: str) -> str:
    """Patch V1 MJCF: apply solimp fix.

    Returns the path to the patched file.
    """
    tree = ET.parse(mjcf_path)
    _patch_solimp(tree)
    patched_path = mjcf_path.replace(".xml", "_v1_patched.xml")
    tree.write(patched_path)
    return patched_path


def _patch_v4_mjcf(mjcf_path: str) -> str:
    """Patch V4 MJCF to fix known issues vs V3.

    1. Add missing coupler joints (left_coupler, right_coupler bodies have no
       joint element in V4, giving 6 DOFs instead of V3's 8).
    2. Fix base collision geom transform — V4 visual base has
       pos/quat but collision base has none, causing misalignment.
    3. Patch equality constraint solimp[0] (dmin) from 0.95 to 0.5.

    Returns the path to the patched file.
    """
    tree = ET.parse(mjcf_path)

    # 1. Add missing coupler joints
    coupler_names = {
        "left_coupler": "left_coupler_joint",
        "right_coupler": "right_coupler_joint",
    }

    for body in tree.iter("body"):
        name = body.get("name")
        if name in coupler_names and body.find("joint") is None:
            joint_elem = ET.Element("joint")
            joint_elem.set("name", coupler_names[name])
            joint_elem.set("class", "coupler")
            # Insert after <inertial> to match V3 element order
            inertial = body.find("inertial")
            idx = list(body).index(inertial) + 1 if inertial is not None else 0
            body.insert(idx, joint_elem)

    # 2. Fix base collision geom: copy pos/quat from the visual base geom
    for body in tree.iter("body"):
        if body.get("name") != "base":
            continue
        visual_base = None
        for geom in body.findall("geom"):
            if geom.get("class") == "visual" and geom.get("mesh") == "base":
                visual_base = geom
                break
        if visual_base is None:
            break
        for geom in body.findall("geom"):
            if geom.get("class") == "collision" and geom.get("mesh") == "base":
                if visual_base.get("pos") and not geom.get("pos"):
                    geom.set("pos", visual_base.get("pos"))
                if visual_base.get("quat") and not geom.get("quat"):
                    geom.set("quat", visual_base.get("quat"))
        break

    # 3. Patch solimp
    _patch_solimp(tree)

    patched_path = mjcf_path.replace(".xml", "_v4_patched.xml")
    tree.write(patched_path)
    return patched_path


class SubstepRecorder:
    """GPU ring buffer that snapshots body state, joint state, contact info,
    and MuJoCo Warp internal forces at each substep.

    All buffers are pre-allocated so ``record()`` only issues ``wp.copy``
    kernel launches — safe inside a CUDA graph capture.
    """

    def __init__(
        self,
        num_substeps: int,
        body_count: int,
        dof_count: int,
        joint_coord_count: int,
        contact_max: int,
        mjw_nv: int = 0,
        mjw_nu: int = 0,
        mjw_naconmax: int = 0,
    ):
        self.num_substeps = num_substeps
        self.body_count = body_count
        self.dof_count = dof_count
        self.joint_coord_count = joint_coord_count
        self.mjw_nv = mjw_nv
        self.mjw_nu = mjw_nu

        # Body state snapshots
        self.body_q = wp.zeros((num_substeps, body_count), dtype=wp.transform)
        self.body_qd = wp.zeros((num_substeps, body_count), dtype=wp.spatial_vector)
        self.body_f = wp.zeros((num_substeps, body_count), dtype=wp.spatial_vector)

        # Newton joint state
        self.joint_q = wp.zeros((num_substeps, joint_coord_count), dtype=wp.float32)
        self.joint_qd = wp.zeros((num_substeps, dof_count), dtype=wp.float32)

        # Contact info
        self.contact_count = wp.zeros(num_substeps, dtype=wp.int32)
        self._contact_stiffness_sample_max = min(contact_max, 8192)
        self.contact_stiffness = wp.zeros((num_substeps, self._contact_stiffness_sample_max), dtype=wp.float32)

        # MuJoCo Warp internal forces (flattened across worlds: nworld * nv)
        if mjw_nv > 0:
            self.mjw_qfrc_actuator = wp.zeros((num_substeps, mjw_nv), dtype=wp.float32)
            self.mjw_qfrc_constraint = wp.zeros((num_substeps, mjw_nv), dtype=wp.float32)
            self.mjw_qacc = wp.zeros((num_substeps, mjw_nv), dtype=wp.float32)
        else:
            self.mjw_qfrc_actuator = None
            self.mjw_qfrc_constraint = None
            self.mjw_qacc = None

        if mjw_nu > 0:
            self.mjw_actuator_force = wp.zeros((num_substeps, mjw_nu), dtype=wp.float32)
        else:
            self.mjw_actuator_force = None

        # MuJoCo Warp contact distance per substep (from mjw_data.contact.dist)
        # negative = penetrating; compute max pen during readback
        self._mjw_naconmax = mjw_naconmax
        if mjw_naconmax > 0:
            self.mjw_contact_dist = wp.zeros((num_substeps, mjw_naconmax), dtype=wp.float32)
            self.mjw_nacon = wp.zeros(num_substeps, dtype=wp.int32)
        else:
            self.mjw_contact_dist = None
            self.mjw_nacon = None

        # Contact normals snapshot (for variance analysis)
        self._normal_sample_max = min(contact_max, 8192)
        self.contact_normals = wp.zeros((num_substeps, self._normal_sample_max), dtype=wp.vec3)

    def record(self, substep_idx: int, state, contacts, mjw_data=None):
        """Snapshot state into substep slot. Graph-safe (wp.copy only)."""
        s = substep_idx
        wp.copy(self.body_q, state.body_q, dest_offset=s * self.body_count, count=self.body_count)
        wp.copy(self.body_qd, state.body_qd, dest_offset=s * self.body_count, count=self.body_count)
        if state.body_f is not None:
            wp.copy(self.body_f, state.body_f, dest_offset=s * self.body_count, count=self.body_count)

        # Joint state
        nq = self.joint_coord_count
        wp.copy(self.joint_q, state.joint_q, dest_offset=s * nq, count=nq)
        nv = self.dof_count
        wp.copy(self.joint_qd, state.joint_qd, dest_offset=s * nv, count=nv)

        # Contacts
        if contacts is not None:
            wp.copy(self.contact_count, contacts.rigid_contact_count, dest_offset=s, count=1)
            if contacts.rigid_contact_stiffness is not None:
                n = self._contact_stiffness_sample_max
                wp.copy(self.contact_stiffness, contacts.rigid_contact_stiffness, dest_offset=s * n, count=n)
            nn = self._normal_sample_max
            wp.copy(self.contact_normals, contacts.rigid_contact_normal, dest_offset=s * nn, count=nn)

        # MuJoCo Warp internals
        if mjw_data is not None and self.mjw_qfrc_actuator is not None:
            mjw_nv = self.mjw_nv
            # qfrc_actuator, qfrc_constraint, qacc are (nworld, nv) → flatten
            wp.copy(self.mjw_qfrc_actuator, mjw_data.qfrc_actuator, dest_offset=s * mjw_nv, count=mjw_nv)
            wp.copy(self.mjw_qfrc_constraint, mjw_data.qfrc_constraint, dest_offset=s * mjw_nv, count=mjw_nv)
            wp.copy(self.mjw_qacc, mjw_data.qacc, dest_offset=s * mjw_nv, count=mjw_nv)
        if mjw_data is not None and self.mjw_actuator_force is not None:
            mjw_nu = self.mjw_nu
            wp.copy(self.mjw_actuator_force, mjw_data.actuator_force, dest_offset=s * mjw_nu, count=mjw_nu)

        # MuJoCo contact distance (graph-safe: wp.copy only)
        if mjw_data is not None and self.mjw_contact_dist is not None:
            n = self._mjw_naconmax
            wp.copy(self.mjw_contact_dist, mjw_data.contact.dist, dest_offset=s * n, count=n)
            wp.copy(self.mjw_nacon, mjw_data.nacon, dest_offset=s, count=1)

    def readback(self):
        """Read all GPU buffers to CPU. Call OUTSIDE the graph."""
        result = {
            "body_q": self.body_q.numpy().reshape(self.num_substeps, self.body_count, -1),
            "body_qd": self.body_qd.numpy().reshape(self.num_substeps, self.body_count, -1),
            "body_f": self.body_f.numpy().reshape(self.num_substeps, self.body_count, -1),
            "joint_q": self.joint_q.numpy().reshape(self.num_substeps, self.joint_coord_count),
            "joint_qd": self.joint_qd.numpy().reshape(self.num_substeps, self.dof_count),
            "contact_count": self.contact_count.numpy(),
            "contact_stiffness": self.contact_stiffness.numpy().reshape(
                self.num_substeps, self._contact_stiffness_sample_max
            ),
            "contact_normals": self.contact_normals.numpy().reshape(self.num_substeps, self._normal_sample_max, 3),
        }
        if self.mjw_qfrc_actuator is not None:
            result["mjw_qfrc_actuator"] = self.mjw_qfrc_actuator.numpy().reshape(self.num_substeps, self.mjw_nv)
            result["mjw_qfrc_constraint"] = self.mjw_qfrc_constraint.numpy().reshape(self.num_substeps, self.mjw_nv)
            result["mjw_qacc"] = self.mjw_qacc.numpy().reshape(self.num_substeps, self.mjw_nv)
        if self.mjw_actuator_force is not None:
            result["mjw_actuator_force"] = self.mjw_actuator_force.numpy().reshape(self.num_substeps, self.mjw_nu)
        if self.mjw_contact_dist is not None:
            dist_all = self.mjw_contact_dist.numpy().reshape(self.num_substeps, self._mjw_naconmax)
            nacon_all = self.mjw_nacon.numpy()
            # Compute max penetration per substep from contact.dist (negative = penetrating)
            mjw_max_pen = np.zeros(self.num_substeps)
            for s in range(self.num_substeps):
                nc = int(nacon_all[s])
                if nc > 0:
                    mjw_max_pen[s] = max(0.0, -float(np.min(dist_all[s, :nc])))
            result["mjw_max_pen"] = mjw_max_pen
        return result


class DiagnosticsLogger:
    """Per-frame physics telemetry for debugging grasp failures."""

    def __init__(self, num_worlds: int, sample_interval: int = 1):
        self.num_worlds = num_worlds
        self.sample_interval = sample_interval
        self.records = []
        self.substep_records = []  # per-frame substep snapshots

    def sample(
        self,
        frame,
        state,
        contacts,
        model,
        task_array,
        bodies_per_world,
        object_body_idx,
        gripper_joints_start,
        surface_pen=0.0,
        surface_contact_count=0,
        rigid_contact_count=0,
    ):
        if frame % self.sample_interval != 0:
            return

        contact_count = 0
        contact_stiffness_max = 0.0
        contact_stiffness_mean = 0.0
        contact_force_max = 0.0
        if contacts is not None:
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
        if contact_count > 0 and contacts is not None and contacts.rigid_contact_stiffness is not None:
            stiffness = contacts.rigid_contact_stiffness.numpy()[:contact_count]
            contact_stiffness_max = float(np.max(stiffness))
            contact_stiffness_mean = float(np.mean(stiffness))
        if contact_count > 0 and contacts is not None:
            forces = contacts.rigid_contact_force.numpy()[:contact_count]
            force_mags = np.linalg.norm(forces, axis=1)
            contact_force_max = float(np.max(force_mags))

        body_q = state.body_q.numpy().reshape(self.num_worlds, bodies_per_world, 7)
        body_qd = state.body_qd.numpy().reshape(self.num_worlds, bodies_per_world, 6)
        obj_pos = body_q[:, object_body_idx, :3]
        obj_vel = body_qd[:, object_body_idx, :3]
        obj_angvel = body_qd[:, object_body_idx, 3:]
        obj_speed = np.linalg.norm(obj_vel, axis=1)
        obj_angspeed = np.linalg.norm(obj_angvel, axis=1)

        joint_qd = state.joint_qd.numpy()
        dofs_per_world = len(joint_qd) // self.num_worlds
        joint_qd = joint_qd.reshape(self.num_worlds, dofs_per_world)
        gripper_qd = joint_qd[:, gripper_joints_start : gripper_joints_start + 8]
        gripper_speed_max = float(np.max(np.abs(gripper_qd)))

        tasks = task_array.numpy()

        for w in range(self.num_worlds):
            self.records.append(
                {
                    "frame": frame,
                    "world": w,
                    "task": int(tasks[w]),
                    "obj_z": float(obj_pos[w, 2]),
                    "obj_speed": float(obj_speed[w]),
                    "obj_angspeed": float(obj_angspeed[w]),
                    "contact_count": contact_count,
                    "contact_stiffness_max": contact_stiffness_max,
                    "contact_stiffness_mean": contact_stiffness_mean,
                    "contact_force_max": contact_force_max,
                    "gripper_speed_max": gripper_speed_max,
                    "surface_contact_count": surface_contact_count,
                    "rigid_contact_count": rigid_contact_count,
                    "surface_pen_mm": surface_pen * 1000.0,
                    "has_nan": bool(np.any(np.isnan(body_q[w]))),
                }
            )

    def sample_substeps(
        self,
        frame,
        substep_data,
        bodies_per_world,
        object_body_idx,
        gripper_joints_start,
        task_array,
        num_worlds_mjw=None,
    ):
        """Process substep GPU readback into per-substep records."""
        body_q = substep_data["body_q"]
        body_qd = substep_data["body_qd"]
        body_f = substep_data["body_f"]
        joint_q_all = substep_data["joint_q"]
        joint_qd_all = substep_data["joint_qd"]
        contact_count = substep_data["contact_count"]
        contact_stiffness = substep_data["contact_stiffness"]

        mjw_max_pen = substep_data.get("mjw_max_pen")

        contact_normals = substep_data.get("contact_normals")  # (num_substeps, N, 3)

        has_mjw = "mjw_qfrc_actuator" in substep_data
        if has_mjw:
            mjw_qfrc_act = substep_data["mjw_qfrc_actuator"]
            mjw_qfrc_con = substep_data["mjw_qfrc_constraint"]
            mjw_qacc = substep_data["mjw_qacc"]
            _mjw_act_force = substep_data.get("mjw_actuator_force")
            mjw_nv_total = mjw_qfrc_act.shape[1]
            mjw_nv_per_world = mjw_nv_total // (num_worlds_mjw or self.num_worlds)

        num_substeps = body_q.shape[0]
        tasks = task_array.numpy()
        dofs_per_world = joint_qd_all.shape[1] // self.num_worlds
        coords_per_world = joint_q_all.shape[1] // self.num_worlds

        for s in range(num_substeps):
            bq = body_q[s].reshape(self.num_worlds, bodies_per_world, -1)
            bqd = body_qd[s].reshape(self.num_worlds, bodies_per_world, -1)
            bf = body_f[s].reshape(self.num_worlds, bodies_per_world, -1)
            jq = joint_q_all[s].reshape(self.num_worlds, coords_per_world)
            jqd = joint_qd_all[s].reshape(self.num_worlds, dofs_per_world)
            nc = int(contact_count[s])
            stiff = contact_stiffness[s]

            stiff_max = float(np.max(stiff[:nc])) if nc > 0 else 0.0
            pen_max = float(mjw_max_pen[s]) if mjw_max_pen is not None else 0.0

            # Contact normal analysis
            normal_mean_mag = 0.0
            normal_variance = 0.0
            normal_max_angle_deg = 0.0
            if nc > 0 and contact_normals is not None:
                normals = contact_normals[s, :nc]  # (nc, 3)
                # Mean normal direction
                mean_normal = np.mean(normals, axis=0)
                mean_mag = np.linalg.norm(mean_normal)
                normal_mean_mag = float(mean_mag)
                if mean_mag > 1e-8:
                    mean_dir = mean_normal / mean_mag
                    # Dot product of each normal with mean direction
                    dots = np.clip(np.sum(normals * mean_dir, axis=1), -1.0, 1.0)
                    angles = np.arccos(dots) * (180.0 / np.pi)
                    normal_variance = float(np.var(dots))
                    normal_max_angle_deg = float(np.max(angles))

            for w in range(self.num_worlds):
                obj_pos = bq[w, object_body_idx, :3]
                obj_vel = bqd[w, object_body_idx, :3]
                obj_f = bf[w, object_body_idx, :]
                gripper_q_w = jq[w, gripper_joints_start : gripper_joints_start + 8]
                gripper_qd_w = jqd[w, gripper_joints_start : gripper_joints_start + 8]

                rec = {
                    "frame": frame,
                    "substep": s,
                    "world": w,
                    "task": int(tasks[w]),
                    "obj_z": float(obj_pos[2]),
                    "obj_speed": float(np.linalg.norm(obj_vel)),
                    "obj_force_mag": float(np.linalg.norm(obj_f[:3])),
                    "obj_torque_mag": float(np.linalg.norm(obj_f[3:])),
                    "contact_count": nc,
                    "contact_stiffness_max": stiff_max,
                    "max_penetration": pen_max,
                    "normal_mean_mag": normal_mean_mag,
                    "normal_variance": normal_variance,
                    "normal_max_angle_deg": normal_max_angle_deg,
                    "gripper_q_max": float(np.max(gripper_q_w)),
                    "gripper_q_min": float(np.min(gripper_q_w)),
                    "gripper_speed_max": float(np.max(np.abs(gripper_qd_w))),
                    "has_nan": bool(np.any(np.isnan(bq[w]))),
                }

                # MuJoCo Warp forces for this world
                if has_mjw:
                    mjw_w_start = w * mjw_nv_per_world
                    mjw_w_end = mjw_w_start + mjw_nv_per_world
                    qfrc_act_w = mjw_qfrc_act[s, mjw_w_start:mjw_w_end]
                    qfrc_con_w = mjw_qfrc_con[s, mjw_w_start:mjw_w_end]
                    qacc_w = mjw_qacc[s, mjw_w_start:mjw_w_end]
                    rec["mjw_qfrc_actuator_max"] = float(np.max(np.abs(qfrc_act_w)))
                    rec["mjw_qfrc_constraint_max"] = float(np.max(np.abs(qfrc_con_w)))
                    rec["mjw_qacc_max"] = float(np.max(np.abs(qacc_w)))
                    rec["mjw_qfrc_constraint_has_nan"] = bool(np.any(np.isnan(qfrc_con_w)))
                    rec["mjw_qacc_has_nan"] = bool(np.any(np.isnan(qacc_w)))

                self.substep_records.append(rec)

    def write_substep_csv(self, path):
        if not self.substep_records:
            return
        import csv  # noqa: PLC0415

        keys = self.substep_records[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.substep_records)
        print(f"Substep diagnostics CSV written to: {path}")

    def print_substep_nan_analysis(self):
        """Find the exact substep where NaN first appears."""
        nan_recs = [r for r in self.substep_records if r["has_nan"]]
        if not nan_recs:
            print("  No NaN detected in substep records.")
            return

        first = min(nan_recs, key=lambda r: (r["frame"], r["substep"]))
        print("\n  SUBSTEP NaN ANALYSIS:")
        print(f"    First NaN: frame {first['frame']}, substep {first['substep']}, world {first['world']}")

        # Show the substep progression leading up to NaN for that world
        world = first["world"]
        frame = first["frame"]
        frame_recs = [r for r in self.substep_records if r["frame"] == frame and r["world"] == world]
        frame_recs.sort(key=lambda r: r["substep"])

        has_mjw = "mjw_qfrc_constraint_max" in frame_recs[0]
        hdr = (
            f"    {'sub':>3} {'obj_z':>8} {'obj_spd':>8} {'stiff_max':>12} {'pen_max':>10}"
            f" {'n_var':>8} {'n_ang°':>6} {'grip_q':>8} {'grip_spd':>8}"
        )
        if has_mjw:
            hdr += f" {'act_frc':>10} {'con_frc':>10} {'qacc':>10} {'con_nan':>7} {'acc_nan':>7}"
        hdr += f" {'nan':>4}"
        print(f"    Substep progression (frame {frame}, world {world}):")
        print(hdr)
        for r in frame_recs:
            line = (
                f"    {r['substep']:>3} {r['obj_z']:>8.4f} {r['obj_speed']:>8.4f} "
                f"{r['contact_stiffness_max']:>12.2e} {r.get('max_penetration', 0):>10.6f} "
                f"{r.get('normal_variance', 0):>8.5f} {r.get('normal_max_angle_deg', 0):>6.1f} "
                f"{r.get('gripper_q_max', 0):>8.4f} {r['gripper_speed_max']:>8.4f}"
            )
            if has_mjw:
                line += (
                    f" {r['mjw_qfrc_actuator_max']:>10.2e} {r['mjw_qfrc_constraint_max']:>10.2e}"
                    f" {r['mjw_qacc_max']:>10.2e}"
                    f" {'NaN' if r.get('mjw_qfrc_constraint_has_nan') else 'ok':>7}"
                    f" {'NaN' if r.get('mjw_qacc_has_nan') else 'ok':>7}"
                )
            line += f" {'NaN' if r['has_nan'] else 'ok':>4}"
            print(line)

        # Also show previous frame's last substep for context
        prev_recs = [r for r in self.substep_records if r["frame"] == frame - 1 and r["world"] == world]
        if prev_recs:
            last_prev = max(prev_recs, key=lambda r: r["substep"])
            print(f"\n    Previous frame {frame - 1}, substep {last_prev['substep']}:")
            print(
                f"      obj_z={last_prev['obj_z']:.4f} obj_speed={last_prev['obj_speed']:.4f}"
                f" stiff_max={last_prev['contact_stiffness_max']:.2e}"
                f" pen_max={last_prev.get('max_penetration', 0):.6f}"
                f" grip_q_max={last_prev.get('gripper_q_max', 0):.4f}"
                f" grip_speed={last_prev['gripper_speed_max']:.4f}"
            )
            if has_mjw:
                print(
                    f"      mjw_act_frc={last_prev['mjw_qfrc_actuator_max']:.2e}"
                    f" mjw_con_frc={last_prev['mjw_qfrc_constraint_max']:.2e}"
                    f" mjw_qacc={last_prev['mjw_qacc_max']:.2e}"
                )

        # Show surface vs contacts penetration from per-frame records
        frame_rec = [r for r in self.records if r["frame"] == frame and r["world"] == world]
        if frame_rec:
            fr = frame_rec[0]
            print(f"\n    Contact surface stats (frame {frame}):")
            print(
                f"      Surface faces: {fr.get('surface_contact_count', 0)}"
                f"  Rigid contacts: {fr.get('rigid_contact_count', 0)}"
                f"  Reduction ratio: {fr.get('rigid_contact_count', 0)}/{fr.get('surface_contact_count', 1)}"
            )
            print(f"      Surface pen (SDF): {fr.get('surface_pen_mm', 0):.3f} mm")

    def print_summary(self):
        if not self.records:
            print("No diagnostic records.")
            return

        task_names = {0: "APPROACH", 1: "CLOSE", 2: "LIFT", 3: "HOLD"}

        print(f"\n{'=' * 70}")
        print("DIAGNOSTICS SUMMARY")
        print(f"{'=' * 70}")

        for task_id, task_name in task_names.items():
            phase_records = [r for r in self.records if r["task"] == task_id and not r["has_nan"]]
            if not phase_records:
                continue

            stiffness_max = max(r["contact_stiffness_max"] for r in phase_records)
            stiffness_mean = np.mean([r["contact_stiffness_mean"] for r in phase_records])
            force_max = max(r["contact_force_max"] for r in phase_records)
            obj_speed_max = max(r["obj_speed"] for r in phase_records)
            obj_angspeed_max = max(r["obj_angspeed"] for r in phase_records)
            contact_count_max = max(r["contact_count"] for r in phase_records)
            gripper_speed_max = max(r["gripper_speed_max"] for r in phase_records)

            print(f"\n  Phase: {task_name} ({len(phase_records)} samples)")
            print(f"    Contact count (max):        {contact_count_max}")
            print(f"    Contact stiffness (max):     {stiffness_max:.2e}")
            print(f"    Contact stiffness (mean):    {stiffness_mean:.2e}")
            print(f"    Contact force (max):         {force_max:.2e} N")
            print(f"    Object speed (max):          {obj_speed_max:.4f} m/s")
            print(f"    Object angular speed (max):  {obj_angspeed_max:.4f} rad/s")
            print(f"    Gripper joint speed (max):   {gripper_speed_max:.4f} rad/s")

        nan_records = [r for r in self.records if r["has_nan"]]
        if nan_records:
            print(f"\n  NaN EVENTS ({len(nan_records)} samples):")
            first_nan = min(nan_records, key=lambda r: r["frame"])
            print(
                f"    First NaN: frame {first_nan['frame']}, world {first_nan['world']}, task={task_names.get(first_nan['task'], '?')}"
            )
            pre_nan = [
                r
                for r in self.records
                if r["world"] == first_nan["world"] and r["frame"] < first_nan["frame"] and not r["has_nan"]
            ]
            if pre_nan:
                last = max(pre_nan, key=lambda r: r["frame"])
                print(f"    Last healthy frame {last['frame']}:")
                print(f"      contact_stiffness_max: {last['contact_stiffness_max']:.2e}")
                print(f"      contact_force_max:     {last['contact_force_max']:.2e}")
                print(f"      obj_speed:             {last['obj_speed']:.4f}")
                print(f"      obj_angspeed:          {last['obj_angspeed']:.4f}")

        print(f"\n{'=' * 70}")

    def write_csv(self, path):
        if not self.records:
            return
        import csv  # noqa: PLC0415

        keys = self.records[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"Diagnostics CSV written to: {path}")


class CollisionMode(Enum):
    """Collision pipeline modes.

    MUJOCO: MuJoCo native contacts.
    NEWTON_DEFAULT: Newton GJK/MPR.
    NEWTON_SDF: Newton with SDF.
    NEWTON_HYDROELASTIC: Newton hydroelastic.
    """

    MUJOCO = "mujoco"
    NEWTON_DEFAULT = "newton_default"
    NEWTON_SDF = "newton_sdf"
    NEWTON_HYDROELASTIC = "newton_hydroelastic"


class ObjectShape(Enum):
    """Shape of the grasp object on the table."""

    BOX = "box"
    ROUNDED_BOX = "rounded_box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"


class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        # ---- Configuration (change these to switch modes) ----
        self.use_v4 = False  # True = V4 gripper, False = V1
        # self.object_shape = ObjectShape.BOX
        # self.collision_mode = CollisionMode.NEWTON_DEFAULT
        # self.object_armature = 0.01  # artificial inertia on grasp object [kg*m^2]

        # ---- Simulation parameters ----
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 64
        self.collide_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.viewer = viewer
        self.test_mode = args.test if args else False
        self.diagnostics = args.diagnostics if args and hasattr(args, "diagnostics") else False
        self.diag_logger = DiagnosticsLogger(self.num_worlds) if self.diagnostics else None

        # self.viewer._paused = True

        # Contact budget: hydroelastic reduction yields ~240 contacts per shape pair
        # (20 normal bins x 7 spatial + 100 voxel slots). With 4 finger pads x 1 object
        # = 4 pairs → ~960 contacts/world. Use 2000/world for headroom.
        self.rigid_contact_max = 2_000 * self.num_worlds

        # ---- Initial base pose (single source of truth) ----
        # base_target_pos/rot are D6 joint targets (relative to parent_xform).
        # The static orientation (e.g. gripper pointing down) is baked into
        # the D6 parent_xform so the joint stays near zero — avoiding gimbal lock.
        self.base_parent_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        self.base_target_pos = [0.0, 0.0, 0.5]  # [x, y, z] in meters
        self.base_target_rot = [0.0, 0.0, 0.0]  # [rx, ry, rz] in radians

        # ---- GUI state ----
        self.manual_mode = False
        # Contact stats (cached for GUI, updated periodically)
        self._gui_hydro_contact_count = 0
        self._gui_hydro_max_pen_surface = 0.0  # from contact surface depth (SDF-based)
        self._max_pen_surface_ever = 0.0
        self._gui_hydro_max_pen_contacts = 0.0  # kept for hydroelastic GUI display
        self._gui_mjw_max_pen = 0.0  # from mjw_data.contact.dist (unified, all modes)
        self._mjw_max_pen_ever = 0.0
        self.gripper_ctrl_value = 0.0
        self.show_isosurface = False

        # ---- Override from CLI args ----
        if args:
            if hasattr(args, "object_shape") and args.object_shape:
                self.object_shape = ObjectShape(args.object_shape)
            if hasattr(args, "no_manual") and args.no_manual:
                self.manual_mode = False
            if hasattr(args, "object_armature") and args.object_armature is not None:
                self.object_armature = args.object_armature
            if hasattr(args, "collision_mode") and args.collision_mode:
                self.collision_mode = CollisionMode(args.collision_mode)

        # Max velocity limits (per-frame deltas, derived from target physical rates).
        # target_velocity / fps = per_frame_delta
        self.base_pos_max_vel = 0.3 / self.fps  # 0.3 m/s / 100 fps = 0.003 m/frame
        self.base_rot_max_vel = 1.2 / self.fps  # 1.2 rad/s / 100 fps = 0.012 rad/frame
        self.gripper_max_delta = 255 / self.fps  # 255 units/s / 100 fps = 2.55 units/frame

        # GUI cache (avoid GPU sync every frame)
        self._gui_task_val = 0
        self._gui_task_timer_val = 0.0
        self._gui_read_interval = 10
        self._frame_count = 0

        # ---- Scene geometry ----
        self.table_height = 0.1
        self.object_half_size = 0.03
        self.object_init_z = self.table_height + self.object_half_size + 0.001

        # ---- Shape config ----
        self.shape_cfg = newton.ModelBuilder.ShapeConfig(
            gap=0.005,
            mu=1.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        self.kh = 2e11  # hydroelastic stiffness [Pa]
        if args and hasattr(args, "kh") and args.kh is not None:
            self.kh = args.kh
        self._impratio = 10.0
        if args and hasattr(args, "impratio") and args.impratio is not None:
            self._impratio = args.impratio
        self.sdf_params = {
            "max_resolution": 64,
            "narrow_band_range": (-0.005, 0.005),
        }

        # ---- Build model ----
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        self._add_gripper(builder)

        self._add_scene_shapes(builder)
        builder.add_ground_plane()

        # Count bodies per world before replication
        self.bodies_per_world = builder.body_count
        self.dofs_per_world = builder.joint_dof_count

        # ---- Replicate and finalize ----
        scene = newton.ModelBuilder()
        scene.replicate(builder, self.num_worlds, spacing=(0.5, 0.5, 0.0))
        self.model = scene.finalize()

        # ---- GPU state arrays for state machine ----
        self.task = wp.zeros(self.num_worlds, dtype=int)
        self.task_timer = wp.zeros(self.num_worlds, dtype=float)

        # GPU state for rate limiting (persistent across frames)
        init_6dof = [self.base_target_pos + self.base_target_rot]
        self.prev_base_target = wp.array(
            init_6dof * self.num_worlds,
            dtype=float,
            shape=(self.num_worlds, 6),
        )
        self.prev_gripper_ctrl = wp.zeros(self.num_worlds, dtype=float)

        # Task config
        # Task durations [s]: APPROACH, CLOSE_GRIPPER, LIFT, HOLD
        self.task_durations = wp.array([0.5, 1.0, 1.0, 1.0], dtype=float)

        # Precompute Z targets for state machine
        self.gripper_base_to_tcp_dist = 0.155
        self.grasp_z = self.table_height + self.object_half_size + self.gripper_base_to_tcp_dist
        self.lift_z = self.grasp_z + 0.10

        # Gripper ctrl target: close past the object by a shape-dependent margin [mm].
        # Flat shapes (box) need a small margin; round shapes need more to develop
        # sufficient contact area for friction to hold during lift.
        if args and hasattr(args, "grasp_margin") and args.grasp_margin is not None:
            grasp_margin_mm = args.grasp_margin
        else:
            grasp_margin_mm = {
                ObjectShape.BOX: 3.0,
                ObjectShape.ROUNDED_BOX: 5.0,
                ObjectShape.SPHERE: 13.0,
                ObjectShape.CYLINDER: 13.0,
                ObjectShape.CAPSULE: 13.0,
            }[self.object_shape]
        self.grasp_ctrl = self._mm_to_ctrl(self.object_half_size * 2 * 1000.0, grasp_margin_mm)
        print(
            f"Grasp ctrl target: {self.grasp_ctrl:.1f}/255 "
            f"(object={self.object_half_size * 2 * 1000:.0f}mm, margin={grasp_margin_mm:.0f}mm)"
        )

        # ---- GPU buffers for GUI-driven values (written from CPU, read by kernel) ----
        self._gpu_manual_mode = wp.array([int(self.manual_mode)], dtype=int)
        self._gpu_gui_pos = wp.array([wp.vec3(*self.base_target_pos)], dtype=wp.vec3)
        self._gpu_gui_rot = wp.array([wp.vec3(*self.base_target_rot)], dtype=wp.vec3)
        self._gpu_gui_ctrl = wp.array([self.gripper_ctrl_value], dtype=float)
        self._gpu_vel_limits = wp.array(
            [wp.vec3(self.base_pos_max_vel, self.base_rot_max_vel, self.gripper_max_delta)],
            dtype=wp.vec3,
        )
        self._gpu_dirty = True  # Force initial sync

        # ---- Collision pipeline ----
        self.collision_pipeline = self._create_collision_pipeline()
        self.contacts = self.collision_pipeline.contacts() if self.collision_pipeline else None

        # ---- Solver ----
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=(self.collision_mode == CollisionMode.MUJOCO),
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=self.rigid_contact_max // self.num_worlds,
            nconmax=self.rigid_contact_max // self.num_worlds,
            iterations=50,
            ls_iterations=100,
            impratio=self._impratio,
        )

        # ---- State and control ----
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.has_mujoco_ctrl = hasattr(self.control, "mujoco")
        if self.has_mujoco_ctrl:
            self.direct_control = wp.zeros_like(self.control.mujoco.ctrl)
            self.mujoco_ctrl_gripper_idx = 0
            print(f"MJCF ctrl size: {self.direct_control.shape}")

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Per-world test diagnostics
        if self.test_mode:
            self.object_max_z = np.full(self.num_worlds, self.object_init_z)
            self.world_nan_frame = np.full(self.num_worlds, -1, dtype=int)
            self.world_nan_task = np.full(self.num_worlds, -1, dtype=int)
            # Vertical slip tracking: record object z at start of LIFT,
            # then compare with gripper lift to measure how much the object slips down.
            self.object_z_at_lift_start = np.full(self.num_worlds, np.nan)
            self.object_z_at_hold = np.full(self.num_worlds, np.nan)
        else:
            self.object_max_z = None

        # ---- Viewer setup ----
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets(wp.vec3(0.5, 0.5, 0.0))
        if hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = self.show_isosurface

        # ---- Substep recorder (diagnostics) ----
        if self.diagnostics:
            # Get MuJoCo Warp dimensions if available
            mjw_nv = 0
            mjw_nu = 0
            if hasattr(self.solver, "mjw_data"):
                mjw_data = self.solver.mjw_data
                mjw_nv = mjw_data.qfrc_actuator.size
                mjw_nu = mjw_data.actuator_force.size
            mjw_naconmax = int(mjw_data.naconmax) if hasattr(self.solver, "mjw_data") else 0
            self.substep_recorder = SubstepRecorder(
                num_substeps=self.sim_substeps,
                body_count=self.model.body_count,
                dof_count=self.model.joint_dof_count,
                joint_coord_count=self.model.joint_coord_count,
                contact_max=self.rigid_contact_max,
                mjw_nv=mjw_nv,
                mjw_nu=mjw_nu,
                mjw_naconmax=mjw_naconmax,
            )
        else:
            self.substep_recorder = None

        # Pre-cache the reshaped direct_control view for the control graph
        self._direct_control_2d = self.direct_control.reshape((self.num_worlds, -1))

        # Diagnostic: disable gravity if requested
        if args and hasattr(args, "no_gravity") and args.no_gravity:
            self.model.gravity = wp.vec3(0.0, 0.0, 0.0)

        self.capture()

        version = "V4" if self.use_v4 else "V1"
        print(f"Robotiq 2F-85 {version} | Collision: {self.collision_mode.value} | Worlds: {self.num_worlds}")

    def _download_gripper_assets(self) -> str:
        """Download gripper MJCF and return the path to the XML file."""
        repo_url = "https://github.com/google-deepmind/mujoco_menagerie.git"
        if self.use_v4:
            asset_path = download_git_folder(repo_url, "robotiq_2f85_v4")
            mjcf_path = f"{asset_path}/2f85.xml"
            # Copy corrected inertia XML from busbar_assets if available
            corrected = os.path.join("busbar_assets", "2f85_corrected_inertia.xml")
            if os.path.exists(corrected):
                dest = f"{asset_path}/2f85_corrected_inertia.xml"
                shutil.copy2(corrected, dest)
                mjcf_path = dest
            # Add missing coupler joints so V4 has 8 DOFs like V3
            mjcf_path = _patch_v4_mjcf(mjcf_path)
        else:
            asset_path = download_git_folder(repo_url, "robotiq_2f85")
            mjcf_path = _patch_v1_mjcf(f"{asset_path}/2f85.xml")
        return mjcf_path

    def _add_gripper(self, builder):
        """Load the Robotiq 2F-85 MJCF, attach it via a D6 joint, and tune armature."""
        mjcf_path = self._download_gripper_assets()

        # D6 joint: 6 scalar DOFs (3 pos + 3 rot radians)
        pos_ke, pos_kd = 5000.0, 500.0
        rot_ke, rot_kd = 5000.0, 500.0
        Dof = newton.ModelBuilder.JointDofConfig

        builder.add_mjcf(
            mjcf_path,
            xform=wp.transform(
                wp.vec3(0, 0, 0),
                self.base_parent_rot,
            ),
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    Dof(axis=newton.Axis.X, target_ke=pos_ke, target_kd=pos_kd),
                    Dof(axis=newton.Axis.Y, target_ke=pos_ke, target_kd=pos_kd),
                    Dof(axis=newton.Axis.Z, target_ke=pos_ke, target_kd=pos_kd),
                ],
                "angular_axes": [
                    Dof(axis=newton.Axis.X, target_ke=rot_ke, target_kd=rot_kd),
                    Dof(axis=newton.Axis.Y, target_ke=rot_ke, target_kd=rot_kd),
                    Dof(axis=newton.Axis.Z, target_ke=rot_ke, target_kd=rot_kd),
                ],
            },
            enable_self_collisions=False,
            ignore_inertial_definitions=False,
        )

        # Joint layout: D6 = 6 coords (3 pos + 3 rot)
        self.base_joint_idx = 0
        self.gripper_joints_start = 6

        # Initialize joint_q and joint_target_pos from base_target_pos/rot
        for i in range(3):
            builder.joint_q[i] = self.base_target_pos[i]
            builder.joint_q[3 + i] = self.base_target_rot[i]
            builder.joint_target_pos[i] = self.base_target_pos[i]
            builder.joint_target_pos[3 + i] = self.base_target_rot[i]

        # Override tendon coefficients and force range
        builder.custom_attributes["mujoco:tendon_coef"].values = [0.485, 0.485]
        builder.custom_attributes["mujoco:actuator_forcerange"].values[-1] = [-150, 150]

        # Enable gravity compensation on the D6 base DOFs so the PD controller
        # only corrects positioning error, not the static gravitational load.
        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(self.gripper_joints_start):
            gravcomp_attr.values[dof_idx] = True

        # Enable body-level gravity compensation on all gripper bodies.
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(builder.body_count):
            gravcomp_body.values[body_idx] = 1.0

        # Gripper joint armature derivation (PhysX Joint Parameter Tuning Guide methodology):
        #
        # The hydroelastic contact stiffness k_contact = kh * area_eff / (2 * kh) ≈ area_eff / 2
        # for equal-stiffness shapes, but the MuJoCo solver sees the per-contact stiffness
        # directly. With kh=1e9, k_contact ≈ 5e6 N/m (from diagnostics).
        #
        # The finger link inertia is very small (~3.4e-6 kg·m²), giving:
        #   ω_n = sqrt(k_contact / I_link) ≈ 1.2e6 rad/s (way too high)
        #
        # For stability with implicit integrator at dt=0.001s, we need ω_n * dt < ~10:
        #   ω_n_max = 10 / dt = 10,000 rad/s
        #   I_needed = k_contact / ω_n_max² = 5e6 / 1e8 = 0.05 kg·m²
        #   armature = I_needed - I_link ≈ 0.05 kg·m²
        #
        # The MJCF default armature (driver=0.005, others=0.001) is ~1000x too small
        # for hydroelastic stiffness. We scale based on kh to maintain stability:
        #   armature_scale = max(1.0, kh / 1e9)  [unitless]
        #
        # Ref: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html
        # Ref: https://github.com/google-deepmind/mujoco/issues/906#issuecomment-1849032881
        gripper_dof_offset = self.gripper_joints_start
        mjcf_armature = np.array([0.005, 0.001, 0.001, 0.001, 0.005, 0.001, 0.001, 0.001])
        armature_scale = 2.0
        gripper_armature = (armature_scale * mjcf_armature).tolist()
        builder.joint_armature[gripper_dof_offset : gripper_dof_offset + 8] = gripper_armature
        print(f"Gripper armature (scale={armature_scale:.1f}x): {[f'{a:.4f}' for a in gripper_armature]}")

    def _add_scene_shapes(self, builder):
        """Add table, grasp object, and configure hydroelastic finger shapes."""
        # Table as kinematic body (mass=0) so it replicates correctly per world.
        # Using body=-1 (static world body) causes MuJoCo to miss collisions
        # for replicated worlds since all tables share the same worldbody.
        table_half = (0.2, 0.2, self.table_height / 2)
        table_xform = wp.transform(wp.vec3(0.0, 0.0, self.table_height / 2), wp.quat_identity())
        table_body = builder.add_body(xform=table_xform, label="table")
        builder.body_mass[table_body] = 0.0
        self._add_object_shape(
            builder,
            body=table_body,
            shape=ObjectShape.BOX,
            size=table_half,
        )

        # Grasp object on table
        object_xform = wp.transform(wp.vec3(0.0, 0.0, self.object_init_z), wp.quat_identity())
        self.object_body_idx = builder.add_body(xform=object_xform, label="grasp_object", armature=self.object_armature)
        s = self.object_half_size
        size = {
            ObjectShape.BOX: (s, s, s),
            ObjectShape.ROUNDED_BOX: (s, s, s),
            ObjectShape.SPHERE: (s,),
            ObjectShape.CYLINDER: (s, s),
            ObjectShape.CAPSULE: (s, s),
        }[self.object_shape]
        self._add_object_shape(builder, self.object_body_idx, shape=self.object_shape, size=size)

        # Build SDFs on finger shapes (required for SDF and hydroelastic modes)
        if self.collision_mode in (CollisionMode.NEWTON_SDF, CollisionMode.NEWTON_HYDROELASTIC):
            self._setup_finger_sdf(builder)

    def _add_object_shape(self, builder, body, shape: ObjectShape, size: tuple[float, ...], xform=None):
        """Add a mesh shape, with SDF for SDF/hydroelastic collision modes.

        Args:
            builder: model builder to add the shape to.
            body: body index (-1 for static/world body).
            shape: which :class:`ObjectShape` to create.
            size: shape-specific dimensions — ``(hx, hy, hz)`` for BOX,
                ``(radius,)`` for SPHERE, ``(radius, half_height)`` for
                CYLINDER/CAPSULE.
            xform: optional transform passed to ``add_shape_mesh``.
        """
        use_sdf = self.collision_mode in (CollisionMode.NEWTON_SDF, CollisionMode.NEWTON_HYDROELASTIC)
        use_hydro = self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC

        if shape == ObjectShape.BOX:
            mesh = newton.Mesh.create_box(
                *size, duplicate_vertices=True, compute_normals=False, compute_uvs=False, compute_inertia=True
            )
        elif shape == ObjectShape.ROUNDED_BOX:
            radius = min(size) * 0.3  # 30% of smallest half-extent
            mesh = create_rounded_box_mesh(*size, radius=radius)
            if mesh is None:
                # Fallback if scipy/skimage not available
                mesh = newton.Mesh.create_box(
                    *size, duplicate_vertices=True, compute_normals=False, compute_uvs=False, compute_inertia=True
                )
        elif shape == ObjectShape.SPHERE:
            mesh = newton.Mesh.create_sphere(*size, compute_inertia=True)
        elif shape == ObjectShape.CYLINDER:
            mesh = newton.Mesh.create_cylinder(*size, compute_inertia=True)
        elif shape == ObjectShape.CAPSULE:
            mesh = newton.Mesh.create_capsule(*size, compute_inertia=True)
        else:
            raise ValueError(f"Unknown object shape: {shape}")

        if use_sdf:
            mesh.build_sdf(
                max_resolution=self.sdf_params["max_resolution"],
                narrow_band_range=self.sdf_params["narrow_band_range"],
                margin=self.shape_cfg.gap,
            )

        builder.add_shape_mesh(
            body=body,
            mesh=mesh,
            xform=xform,
        )

        # Apply material properties for all collision pipelines
        cfg = self.shape_cfg
        shape_idx = len(builder.shape_material_kh) - 1
        builder.shape_gap[shape_idx] = cfg.gap
        builder.shape_material_mu[shape_idx] = cfg.mu
        builder.shape_material_mu_torsional[shape_idx] = cfg.mu_torsional
        builder.shape_material_mu_rolling[shape_idx] = cfg.mu_rolling

        if use_hydro:
            builder.shape_material_kh[shape_idx] = self.kh
            builder.shape_flags[-1] |= newton.ShapeFlags.HYDROELASTIC

    def _setup_finger_sdf(self, builder):
        """Build SDFs on finger contact shapes; enable hydroelastic flag if in hydroelastic mode."""
        use_hydro = self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC
        cfg = self.shape_cfg

        # For V1 and V4: find pad collision shapes by exact label suffix
        pad_names = {"left_pad1", "left_pad2", "right_pad1", "right_pad2"}
        pad_shape_indices = set()
        for shape_idx, lbl in enumerate(builder.shape_label):
            if lbl and lbl.split("/")[-1] in pad_names:
                pad_shape_indices.add(shape_idx)

        for shape_idx in pad_shape_indices:
            if builder.shape_type[shape_idx] == newton.GeoType.BOX:
                # Convert BOX to MESH + SDF for SDF-based contact
                hx, hy, hz = builder.shape_scale[shape_idx]
                mesh = newton.Mesh.create_box(
                    hx,
                    hy,
                    hz,
                    duplicate_vertices=True,
                    compute_normals=False,
                    compute_uvs=False,
                    compute_inertia=True,
                )
                mesh.build_sdf(
                    max_resolution=self.sdf_params["max_resolution"],
                    narrow_band_range=self.sdf_params["narrow_band_range"],
                    margin=cfg.gap,
                )
                builder.shape_type[shape_idx] = newton.GeoType.MESH
                builder.shape_source[shape_idx] = mesh
                builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)

            # Apply material properties
            builder.shape_gap[shape_idx] = cfg.gap
            builder.shape_material_mu[shape_idx] = cfg.mu
            builder.shape_material_mu_torsional[shape_idx] = cfg.mu_torsional
            builder.shape_material_mu_rolling[shape_idx] = cfg.mu_rolling

            builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.VISIBLE
            if use_hydro:
                builder.shape_material_kh[shape_idx] = self.kh
                builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC

        if self.use_v4:
            # V4: Enable tongue meshes on follower bodies
            follower_shape_indices = {
                i
                for i, lbl in enumerate(builder.shape_label)
                if lbl.split("/")[-1] in ("left_follower_geom_1", "right_follower_geom_0")
            }

            for shape_idx in follower_shape_indices:
                if builder.shape_type[shape_idx] == newton.GeoType.MESH:
                    self._build_shape_sdf(builder, shape_idx)
                    builder.shape_gap[shape_idx] = cfg.gap
                    builder.shape_material_mu[shape_idx] = cfg.mu
                    builder.shape_material_mu_torsional[shape_idx] = cfg.mu_torsional
                    builder.shape_material_mu_rolling[shape_idx] = cfg.mu_rolling
                    if use_hydro:
                        builder.shape_material_kh[shape_idx] = self.kh
                        builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC

    def _build_shape_sdf(self, builder, shape_idx):
        """Build SDF for a mesh shape, baking scale if needed."""
        mesh = builder.shape_source[shape_idx]
        if mesh is None:
            return
        shape_scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
        if not np.allclose(shape_scale, 1.0):
            mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
            builder.shape_source[shape_idx] = mesh
            builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
        mesh.clear_sdf()
        mesh.build_sdf(
            max_resolution=self.sdf_params["max_resolution"],
            narrow_band_range=self.sdf_params["narrow_band_range"],
            margin=self.shape_cfg.gap,
        )

    @staticmethod
    def _mm_to_ctrl(object_width_mm: float, margin_mm: float) -> float:
        """Convert a grasp margin [mm] to Robotiq 2F-85 ctrl value [0-255].

        Args:
            object_width_mm: Object width between the fingers [mm].
            margin_mm: Extra closure past first contact [mm].

        Returns:
            Ctrl value in [0, 255].
        """
        stroke_mm = 85.0  # Robotiq 2F-85 full stroke
        finger_gap_mm = object_width_mm - 2.0 * margin_mm
        return min(255.0, max(0.0, 255.0 * (1.0 - finger_gap_mm / stroke_mm)))

    def _create_collision_pipeline(self):
        """Create collision pipeline based on collision mode."""
        if self.collision_mode == CollisionMode.MUJOCO:
            return None
        elif self.collision_mode == CollisionMode.NEWTON_DEFAULT:
            return newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
            )
        elif self.collision_mode == CollisionMode.NEWTON_SDF:
            return newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                reduce_contacts=True,
                broad_phase="explicit",
            )
        elif self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            return newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                reduce_contacts=True,
                broad_phase="explicit",
                sdf_hydroelastic_config=HydroelasticSDF.Config(
                    output_contact_surface=True,
                    buffer_fraction=1.0,
                    buffer_mult_iso=2,
                    buffer_mult_contact=2,
                    anchor_contact=True,
                ),
            )
        else:
            raise ValueError(f"Unknown collision mode: {self.collision_mode}")

    # ---- Simulation ----

    def capture(self):
        self.control_graph = None
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._control_step()
            self.control_graph = cap.graph
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def _control_step(self):
        """Run control pipeline kernel + copy ctrl."""
        wp.launch(
            control_pipeline_kernel,
            dim=self.num_worlds,
            inputs=[
                self._gpu_manual_mode,
                self._gpu_gui_pos,
                self._gpu_gui_rot,
                self._gpu_gui_ctrl,
                self.task,
                self.task_timer,
                self.grasp_z,
                self.lift_z,
                self.grasp_ctrl,
                self.task_durations,
                self.frame_dt,
                wp.vec3(*self.base_target_rot),
                self.prev_base_target,
                self.prev_gripper_ctrl,
                self._gpu_vel_limits,
                self.control.joint_target_pos,
                self._direct_control_2d,
                self.dofs_per_world,
                self.mujoco_ctrl_gripper_idx,
            ],
        )
        if self.has_mujoco_ctrl:
            wp.copy(self.control.mujoco.ctrl, self.direct_control)

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if self.collision_pipeline and i % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            # Substep snapshot (graph-safe: wp.copy only)
            if self.substep_recorder is not None:
                mjw_data = self.solver.mjw_data if hasattr(self.solver, "mjw_data") else None
                self.substep_recorder.record(i, self.state_0, self.contacts, mjw_data)

    def _sync_gpu_buffers(self):
        """Write GUI-driven Python values to GPU buffers (only when dirty)."""
        if not self._gpu_dirty:
            return
        self._gpu_manual_mode.assign(wp.array([int(self.manual_mode)], dtype=int))
        self._gpu_gui_pos.assign(wp.array([wp.vec3(*self.base_target_pos)], dtype=wp.vec3))
        self._gpu_gui_rot.assign(wp.array([wp.vec3(*self.base_target_rot)], dtype=wp.vec3))
        self._gpu_gui_ctrl.assign(wp.array([self.gripper_ctrl_value], dtype=float))
        self._gpu_vel_limits.assign(
            wp.array(
                [wp.vec3(self.base_pos_max_vel, self.base_rot_max_vel, self.gripper_max_delta)],
                dtype=wp.vec3,
            )
        )
        self._gpu_dirty = False

    def step(self):
        # Sync GUI values to GPU (cheap — only on change)
        self._sync_gpu_buffers()

        # Control pipeline (graph-captured)
        if self.control_graph:
            wp.capture_launch(self.control_graph)
        else:
            self._control_step()

        # Simulate (graph-captured)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self._frame_count += 1

        # Substep readback (outside graph — safe to call .numpy())
        if self.substep_recorder is not None and self.diag_logger is not None:
            if self._frame_count % self.diag_logger.sample_interval == 0:
                substep_data = self.substep_recorder.readback()
                self.diag_logger.sample_substeps(
                    frame=self._frame_count,
                    substep_data=substep_data,
                    bodies_per_world=self.bodies_per_world,
                    object_body_idx=self.object_body_idx,
                    gripper_joints_start=self.gripper_joints_start,
                    task_array=self.task,
                )

        # Periodic GPU read for GUI cache
        if self._frame_count % self._gui_read_interval == 0:
            self._gui_task_val = int(self.task.numpy()[0])
            self._gui_task_timer_val = float(self.task_timer.numpy()[0])

            # Hydro contact surface penetration stats
            if self.collision_pipeline is not None and self.collision_pipeline.hydroelastic_sdf is not None:
                surface_data = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
                if surface_data is not None:
                    nc = int(surface_data.face_contact_count.numpy()[0])
                    self._gui_hydro_contact_count = nc
                    if nc > 0:
                        depths = surface_data.contact_surface_depth.numpy()[:nc]
                        # depth is negative for penetrating (SDF convention)
                        min_depth = float(np.min(depths))
                        self._gui_hydro_max_pen_surface = -min_depth  # positive = penetration
                        self._max_pen_surface_ever = max(self._max_pen_surface_ever, self._gui_hydro_max_pen_surface)
                    else:
                        self._gui_hydro_max_pen_surface = 0.0

            # Unified penetration from MuJoCo Warp contact.dist (works for ALL collision modes)
            if hasattr(self.solver, "mjw_data"):
                mjw_data = self.solver.mjw_data
                nc = int(mjw_data.nacon.numpy()[0])
                if nc > 0:
                    dist = mjw_data.contact.dist.numpy()[:nc]
                    min_dist = float(np.min(dist))
                    self._gui_mjw_max_pen = max(0.0, -min_dist)
                    self._mjw_max_pen_ever = max(self._mjw_max_pen_ever, self._gui_mjw_max_pen)
                else:
                    self._gui_mjw_max_pen = 0.0

        # Per-world diagnostics for testing
        if self.test_mode:
            body_q = self.state_0.body_q.numpy()
            obj_indices = np.arange(self.num_worlds) * self.bodies_per_world + self.object_body_idx
            obj_z = body_q[obj_indices, 2]

            # Detect first NaN per world (check all bodies, not just object)
            body_q_reshaped = body_q.reshape(self.num_worlds, self.bodies_per_world, 7)
            world_has_nan = np.any(np.isnan(body_q_reshaped), axis=(1, 2))
            newly_nan = world_has_nan & (self.world_nan_frame < 0)
            if np.any(newly_nan):
                tasks = self.task.numpy()
                self.world_nan_frame[newly_nan] = self._frame_count
                self.world_nan_task[newly_nan] = tasks[newly_nan]

            # Only update max_z for worlds that haven't gone NaN
            healthy = ~world_has_nan
            np.maximum(self.object_max_z, np.where(healthy, obj_z, self.object_max_z), out=self.object_max_z)

            # Vertical slip: record object z at LIFT start and at HOLD
            tasks_np = self.task.numpy()
            for w in range(self.num_worlds):
                if world_has_nan[w]:
                    continue
                t = int(tasks_np[w])
                if t == int(TaskType.LIFT) and np.isnan(self.object_z_at_lift_start[w]):
                    self.object_z_at_lift_start[w] = obj_z[w]
                if t == int(TaskType.HOLD):
                    self.object_z_at_hold[w] = obj_z[w]

        if self.diag_logger:
            # Get surface data (post-graph, outside CUDA capture)
            surface_pen = 0.0
            surface_count = 0
            if self.collision_pipeline is not None and self.collision_pipeline.hydroelastic_sdf is not None:
                sd = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
                if sd is not None:
                    surface_count = int(sd.face_contact_count.numpy()[0])
                    if surface_count > 0:
                        depths = sd.contact_surface_depth.numpy()[:surface_count]
                        surface_pen = float(-np.min(depths))  # positive = penetrating

            rigid_count = 0
            if self.contacts is not None:
                rigid_count = int(self.contacts.rigid_contact_count.numpy()[0])

            self.diag_logger.sample(
                frame=self._frame_count,
                state=self.state_0,
                contacts=self.contacts,
                model=self.model,
                task_array=self.task,
                bodies_per_world=self.bodies_per_world,
                object_body_idx=self.object_body_idx,
                gripper_joints_start=self.gripper_joints_start,
                surface_pen=surface_pen,
                surface_contact_count=surface_count,
                rigid_contact_count=rigid_count,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
            self.viewer.log_hydro_contact_surface(
                (
                    self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
                    if self.collision_pipeline.hydroelastic_sdf is not None
                    else None
                ),
                penetrating_only=True,
            )
        self.viewer.end_frame()

    def gui(self, imgui):
        imgui.text(f"Frame: {self._frame_count}  Sim time: {self.sim_time:.2f}s")
        imgui.text(f"Collision: {self.collision_mode.value}")
        imgui.text(f"Object: {self.object_shape.value} (armature={self.object_armature:.0e})")
        imgui.separator()

        # Isosurface toggle (hydroelastic only)
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC and hasattr(self.viewer, "renderer"):
            changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
            if changed:
                self.viewer.show_hydro_contact_surface = self.show_isosurface

        # Contact penetration (unified, all collision modes)
        mjw_pen_mm = self._gui_mjw_max_pen * 1000.0
        mjw_pen_ever_mm = self._mjw_max_pen_ever * 1000.0
        imgui.text(f"Pen (mjw): {mjw_pen_mm:.3f} mm  (max: {mjw_pen_ever_mm:.3f} mm)")

        # Hydroelastic-specific contact stats
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            rigid_count = 0
            if self.contacts is not None:
                rigid_count = (
                    int(self.contacts.rigid_contact_count.numpy()[0])
                    if self._frame_count % self._gui_read_interval == 0
                    else self._gui_hydro_contact_count
                )
            imgui.text(f"Surface faces: {self._gui_hydro_contact_count}  Rigid: {rigid_count}")
            pen_surface_mm = self._gui_hydro_max_pen_surface * 1000.0
            pen_surface_ever_mm = self._max_pen_surface_ever * 1000.0
            imgui.text(f"Pen (surface): {pen_surface_mm:.3f} mm  (max: {pen_surface_ever_mm:.3f} mm)")
        imgui.separator()

        # Manual mode toggle
        changed, new_manual = imgui.checkbox("Manual Mode", self.manual_mode)
        if changed:
            self.manual_mode = new_manual
            self._gpu_dirty = True

        # Task display (auto mode only)
        if not self.manual_mode:
            imgui.text(f"Task: {TaskType(self._gui_task_val).name.lower()}")
            imgui.text(f"Timer: {self._gui_task_timer_val:.2f}s")

        imgui.separator()

        # Control
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Control"):
            imgui.indent()
            # Gripper ctrl — slider + input
            changed, val = imgui.slider_float(
                "Gripper Ctrl (0=open, 255=closed)", self.gripper_ctrl_value, 0.0, 255.0, format="%.1f"
            )
            if changed and self.manual_mode:
                self.gripper_ctrl_value = val
                self._gpu_dirty = True
            changed, val = imgui.input_float("Gripper Ctrl##input", self.gripper_ctrl_value, format="%.1f")
            if changed and self.manual_mode:
                self.gripper_ctrl_value = min(max(val, 0.0), 255.0)
                self._gpu_dirty = True
            imgui.unindent()

        # Base position — slider + input for each axis
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Base Position"):
            imgui.indent()
            for i, (label, lo, hi) in enumerate([("Base X", -0.3, 0.3), ("Base Y", -0.3, 0.3), ("Base Z", 0.1, 0.6)]):
                changed, val = imgui.slider_float(label, self.base_target_pos[i], lo, hi, format="%.4f")
                if changed and self.manual_mode:
                    self.base_target_pos[i] = val
                    self._gpu_dirty = True
                changed, val = imgui.input_float(f"{label}##input", self.base_target_pos[i], format="%.4f")
                if changed and self.manual_mode:
                    self.base_target_pos[i] = min(max(val, lo), hi)
                    self._gpu_dirty = True
            imgui.unindent()

        # Base rotation — slider + input for roll/pitch/yaw
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.collapsing_header("Base Rotation"):
            imgui.indent()
            for i, label in enumerate(["Roll", "Pitch", "Yaw"]):
                changed, val = imgui.slider_float(label, self.base_target_rot[i], -wp.pi, wp.pi, format="%.3f")
                if changed and self.manual_mode:
                    self.base_target_rot[i] = val
                    self._gpu_dirty = True
                changed, val = imgui.input_float(f"{label}##input", self.base_target_rot[i], format="%.3f")
                if changed and self.manual_mode:
                    self.base_target_rot[i] = min(max(val, -pi), pi)
                    self._gpu_dirty = True
            imgui.unindent()

        # Velocity limits
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.collapsing_header("Velocity Limits"):
            imgui.indent()
            changed, val = imgui.slider_float(
                "Base Pos Max Vel (m/frame)", self.base_pos_max_vel, 0.001, 0.05, format="%.4f"
            )
            if changed:
                self.base_pos_max_vel = val
                self._gpu_dirty = True
            changed, val = imgui.slider_float(
                "Base Rot Max Vel (rad/frame)", self.base_rot_max_vel, 0.005, 0.1, format="%.4f"
            )
            if changed:
                self.base_rot_max_vel = val
                self._gpu_dirty = True
            changed, val = imgui.slider_float(
                "Gripper Max Delta (/frame)", self.gripper_max_delta, 0.5, 20.0, format="%.1f"
            )
            if changed:
                self.gripper_max_delta = val
                self._gpu_dirty = True
            imgui.unindent()

    def test_final(self):
        version = "V4" if self.use_v4 else "V1"

        if not self.test_mode:
            print(f"Robotiq 2F-85 {version} basic gripper example completed successfully")
            return

        min_lift = 0.05  # Object should lift at least 5cm [m]
        nan_worlds = []
        success_worlds = []
        fail_worlds = []

        for w in range(self.num_worlds):
            if self.world_nan_frame[w] >= 0:
                task_name = TaskType(self.world_nan_task[w]).name.lower()
                nan_worlds.append(f"  World {w}: NaN at frame {self.world_nan_frame[w]} (task={task_name})")
            else:
                lift = self.object_max_z[w] - self.object_init_z
                if lift > min_lift:
                    success_worlds.append(w)
                else:
                    fail_worlds.append(f"  World {w}: FAIL lift={lift:.3f}m")

        n_nan = len(nan_worlds)
        n_success = len(success_worlds)
        n_fail = len(fail_worlds)
        n_total = self.num_worlds

        print(f"\n{'=' * 50}")
        print(f"Grasp Test Report — {version} | {self.object_shape.value}")
        print(f"{'=' * 50}")
        print(f"  Worlds: {n_total} | Success: {n_success} | Fail: {n_fail} | NaN: {n_nan}")

        if nan_worlds:
            print(f"\nNaN worlds ({n_nan}):")
            for line in nan_worlds:
                print(line)

        if fail_worlds:
            print(f"\nFailed worlds ({n_fail}):")
            for line in fail_worlds:
                print(line)

        rate = n_success / n_total
        print(f"\nGrasp success rate: {n_success}/{n_total} ({rate:.0%})")
        print(f"  shape={self.object_shape.value}, armature={self.object_armature}")

        # Vertical slip report
        # Expected lift = lift_z - grasp_z = 0.10m
        # Actual lift per world = z_at_hold - z_at_lift_start
        # Slip = expected_lift - actual_lift
        expected_lift = self.lift_z - self.grasp_z
        for w in range(self.num_worlds):
            z0 = self.object_z_at_lift_start[w]
            z1 = self.object_z_at_hold[w]
            if not np.isnan(z0) and not np.isnan(z1):
                actual_lift = z1 - z0
                slip = expected_lift - actual_lift
                print(f"  World {w}: lift={actual_lift * 1000:.1f}mm, slip={slip * 1000:.1f}mm")

        # Contact force report from MuJoCo constraint forces
        if hasattr(self.solver, "mjw_data"):
            mjw_data = self.solver.mjw_data
            qfrc_con = mjw_data.qfrc_constraint.numpy()  # (nworld, nv)
            # mjw_data arrays are 2D: (nworld, nv)
            if qfrc_con.ndim == 1:
                nv_per_world = len(qfrc_con) // self.num_worlds
                qfrc_con = qfrc_con.reshape(self.num_worlds, nv_per_world)
            print("\n  Grip constraint forces (last frame):")
            for w in range(min(self.num_worlds, qfrc_con.shape[0])):
                gripper_frc = qfrc_con[w, self.gripper_joints_start : self.gripper_joints_start + 8]
                print(
                    f"    World {w}: max={np.max(np.abs(gripper_frc)):.2f} Nm, "
                    f"driver=[{gripper_frc[0]:.2f}, {gripper_frc[4]:.2f}] Nm"
                )

        print(f"{'=' * 50}\n")

        if self.diag_logger:
            self.diag_logger.print_summary()
            self.diag_logger.print_substep_nan_analysis()
            self.diag_logger.write_csv(f"diagnostics_{self.collision_mode.value}_{self.object_shape.value}.csv")
            self.diag_logger.write_substep_csv(
                f"diagnostics_substeps_{self.collision_mode.value}_{self.object_shape.value}.csv"
            )

        # Unified penetration report (all collision modes, from mjw_data.contact.dist)
        mjw_pen_mm = self._mjw_max_pen_ever * 1000.0
        print(f"  Max penetration (mjw, all time): {mjw_pen_mm:.3f} mm")
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            surface_pen_mm = self._max_pen_surface_ever * 1000.0
            print(f"  Max penetration (surface, all time): {surface_pen_mm:.3f} mm")

        assert n_nan == 0, f"{n_nan}/{n_total} worlds diverged to NaN"

        assert rate > 0.5, f"Grasp success rate too low: {rate:.0%} ({n_success}/{n_total})"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=4, help="Number of simulated worlds.")
    parser.add_argument(
        "--object-shape",
        type=str,
        default="box",
        choices=[s.value for s in ObjectShape],
        help="Shape of the grasp object.",
    )
    parser.add_argument(
        "--no-manual",
        action="store_true",
        help="Start in auto mode (disable manual mode).",
    )
    parser.add_argument(
        "--object-armature",
        type=float,
        default=1e-2,
        help="Artificial inertia added to the grasp object body [kg*m^2].",
    )
    parser.add_argument(
        "--collision-mode",
        type=str,
        default="newton_hydroelastic",
        choices=[m.value for m in CollisionMode],
        help="Collision pipeline to use.",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Enable per-frame physics diagnostics logging.")
    parser.add_argument("--kh", type=float, default=None, help="Override hydroelastic stiffness [Pa]. Default: 1e11.")
    parser.add_argument("--impratio", type=float, default=None, help="Override MuJoCo impratio. Default: 10.")
    parser.add_argument("--grasp-margin", type=float, default=None, help="Override grasp margin [mm] for all shapes.")
    parser.add_argument("--no-gravity", action="store_true", help="Disable gravity (diagnostic mode).")
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_worlds=args.num_worlds, args=args)

    newton.examples.run(example, args)
