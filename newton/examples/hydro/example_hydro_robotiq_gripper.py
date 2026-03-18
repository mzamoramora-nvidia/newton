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
        if p == TaskType.LIFT.value or p == TaskType.HOLD.value:
            desired_z = lift_z

        d_ctrl = 0.0
        if p == TaskType.CLOSE_GRIPPER.value or p == TaskType.LIFT.value or p == TaskType.HOLD.value:
            d_ctrl = 255.0

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
        self.sim_substeps = 10
        self.collide_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.viewer = viewer
        self.test_mode = args.test if args else False

        # self.viewer._paused = True

        self.rigid_contact_max = 10_000 * self.num_worlds

        # ---- Initial base pose (single source of truth) ----
        # base_target_pos/rot are D6 joint targets (relative to parent_xform).
        # The static orientation (e.g. gripper pointing down) is baked into
        # the D6 parent_xform so the joint stays near zero — avoiding gimbal lock.
        self.base_parent_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        self.base_target_pos = [0.0, 0.0, 0.5]  # [x, y, z] in meters
        self.base_target_rot = [0.0, 0.0, 0.0]  # [rx, ry, rz] in radians

        # ---- GUI state ----
        self.manual_mode = False
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
            gap=0.01,
            mu=1.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        self.kh = 1e11  # hydroelastic stiffness
        self.sdf_params = {
            "max_resolution": 64,
            "narrow_band_range": (-0.01, 0.01),
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
        self.task_durations = wp.array([0.5, 1.0, 0.5, 1.0], dtype=float)

        # Precompute Z targets for state machine
        self.gripper_base_to_tcp_dist = 0.155
        self.grasp_z = self.table_height + self.object_half_size + self.gripper_base_to_tcp_dist
        self.lift_z = self.grasp_z + 0.10

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
            impratio=1000.0,
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
        else:
            self.object_max_z = None

        # ---- Viewer setup ----
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets(wp.vec3(0.5, 0.5, 0.0))
        if hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = self.show_isosurface

        # Pre-cache the reshaped direct_control view for the control graph
        self._direct_control_2d = self.direct_control.reshape((self.num_worlds, -1))

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

        # Scale gripper joint armature by 2x for stability.
        # The Robotiq 2F-85 MJCF defines small armature values for its 8 joints
        # (driver=0.005, coupler/spring_link/follower=0.001) to damp high-frequency oscillation.
        # Ref: https://github.com/google-deepmind/mujoco/issues/906#issuecomment-1849032881
        # Scaling by 2x seems to help stabilize the gripper.
        gripper_dof_offset = self.gripper_joints_start
        gripper_armature = (2.0 * np.array([0.005, 0.001, 0.001, 0.001, 0.005, 0.001, 0.001, 0.001])).tolist()
        builder.joint_armature[gripper_dof_offset : gripper_dof_offset + 8] = gripper_armature

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

        # Periodic GPU read for GUI cache
        if self._frame_count % self._gui_read_interval == 0:
            self._gui_task_val = int(self.task.numpy()[0])
            self._gui_task_timer_val = float(self.task_timer.numpy()[0])

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
        # Configuration info
        imgui.text(f"Collision: {self.collision_mode.value}")
        imgui.text(f"Object: {self.object_shape.value} (armature={self.object_armature:.0e})")
        imgui.separator()

        # Isosurface toggle (hydroelastic only)
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC and hasattr(self.viewer, "renderer"):
            changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
            if changed:
                self.viewer.show_hydro_contact_surface = self.show_isosurface

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
        print(f"{'=' * 50}\n")

        assert n_nan == 0, f"{n_nan}/{n_total} worlds diverged to NaN"
        assert rate > 0.5, f"Grasp success rate too low: {rate:.0%} ({n_success}/{n_total})"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=4, help="Number of simulated worlds.")
    parser.add_argument(
        "--object-shape",
        type=str,
        default="box",
        choices=["box", "sphere", "cylinder", "capsule"],
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
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_worlds=args.num_worlds, args=args)

    newton.examples.run(example, args)
