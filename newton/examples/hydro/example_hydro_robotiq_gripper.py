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

import copy
import os
import shutil
import xml.etree.ElementTree as ET
from enum import Enum

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton._src.utils.download_assets import download_git_folder
from newton.geometry import HydroelasticSDF

# ---- Phase constants (usable in Warp kernels) ----
PHASE_GO_TO_TARGET = 0
PHASE_APPROACH = 1
PHASE_CLOSE_GRIPPER = 2
PHASE_LIFT = 3
PHASE_HOLD = 4
NUM_PHASES = 5


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
    phase: wp.array(dtype=int),
    phase_timer: wp.array(dtype=float),
    # State machine config
    above_z: float,
    grasp_z: float,
    lift_z: float,
    phase_durations: wp.array(dtype=float),
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
        p = phase[tid]
        phase_timer[tid] = phase_timer[tid] + frame_dt

        desired_z = above_z
        if p == PHASE_APPROACH or p == PHASE_CLOSE_GRIPPER:
            desired_z = grasp_z
        elif p == PHASE_LIFT or p == PHASE_HOLD:
            desired_z = lift_z

        d_ctrl = 0.0
        if p == PHASE_CLOSE_GRIPPER:
            d_ctrl = wp.min(phase_timer[tid] / 1.0, 1.0) * 255.0
        elif p == PHASE_LIFT or p == PHASE_HOLD:
            d_ctrl = 255.0

        if phase_timer[tid] > phase_durations[p] and p < PHASE_HOLD:
            phase[tid] = p + 1
            phase_timer[tid] = 0.0

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

    # 3. Patch equality constraint solimp[0] from 0.95 to 0.5
    for eq_elem in tree.iter("equality"):
        for child in eq_elem:
            solimp = child.get("solimp")
            if solimp:
                parts = solimp.split()
                parts[0] = "0.5"
                child.set("solimp", " ".join(parts))

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
        self.collision_mode = CollisionMode.NEWTON_HYDROELASTIC
        self.object_shape = ObjectShape.BOX

        # ---- Simulation parameters ----
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.collide_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.viewer = viewer
        self.test_mode = args.test if args else False

        # self.viewer._paused = True

        # ---- Initial base pose (single source of truth) ----
        # base_target_pos/rot are D6 joint targets (relative to parent_xform).
        # The static orientation (e.g. gripper pointing down) is baked into
        # the D6 parent_xform so the joint stays near zero — avoiding gimbal lock.
        self.base_parent_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        self.base_target_pos = [0.0, 0.0, 0.5]  # [x, y, z] in meters
        self.base_target_rot = [0.0, 0.0, 0.0]  # [rx, ry, rz] in radians

        # ---- GUI state ----
        self.manual_mode = True
        self.gripper_ctrl_value = 0.0
        self.show_isosurface = False

        # Max velocity limits (per frame)
        self.base_pos_max_vel = 0.005  # ~0.3 m/s at 60fps
        self.base_rot_max_vel = 0.02  # ~1.2 rad/s at 60fps
        self.gripper_max_delta = 5.0  # ~300/s at 60fps (ctrl units)

        # GUI cache (avoid GPU sync every frame)
        self._gui_phase_val = 0
        self._gui_timer_val = 0.0
        self._gui_read_interval = 10
        self._frame_count = 0

        # ---- Scene geometry ----
        self.table_height = 0.1
        self.box_size = 0.03
        self.object_init_z = self.table_height + self.box_size + 0.001

        # ---- Hydroelastic shape config ----
        self.hydro_shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=1e11,
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.01, 0.01),
            gap=0.01,
        )
        self.mesh_shape_cfg = copy.deepcopy(self.hydro_shape_cfg)
        self.mesh_shape_cfg.sdf_max_resolution = None
        self.mesh_shape_cfg.sdf_target_voxel_size = None
        self.mesh_shape_cfg.sdf_narrow_band_range = (-0.1, 0.1)
        self.hydro_mesh_sdf_max_resolution = self.hydro_shape_cfg.sdf_max_resolution

        # ---- Build model ----
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

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

        # Count bodies per world before replication
        self.bodies_per_world = builder.body_count

        # ---- Add table (static mesh) ----
        table_half = (0.2, 0.2, self.table_height / 2)
        table_mesh = newton.Mesh.create_box(
            table_half[0],
            table_half[1],
            table_half[2],
            duplicate_vertices=True,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=True,
        )
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            table_mesh.build_sdf(
                max_resolution=self.hydro_mesh_sdf_max_resolution,
                narrow_band_range=self.hydro_shape_cfg.sdf_narrow_band_range,
                margin=self.hydro_shape_cfg.gap,
            )
        builder.add_shape_mesh(
            body=-1,
            mesh=table_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, self.table_height / 2), wp.quat_identity()),
            # cfg=self.mesh_shape_cfg if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC else None,
        )

        # ---- Add grasp object on table ----
        object_xform = wp.transform(wp.vec3(0.0, 0.0, self.object_init_z), wp.quat_identity())
        self.object_body_local = builder.add_body(xform=object_xform, label="grasp_object")
        self._add_object_shape(builder, self.object_body_local)

        # ---- Configure hydroelastic on finger shapes ----
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            self._setup_finger_hydroelastic(builder)

        # ---- Replicate and finalize ----
        scene = newton.ModelBuilder()
        scene.replicate(builder, self.num_worlds, spacing=(0.5, 0.5, 0.0))
        scene.add_ground_plane()
        self.model = scene.finalize()

        self.dofs_per_world = self.model.joint_dof_count // self.num_worlds

        # ---- GPU state arrays for state machine ----
        self.phase = wp.zeros(self.num_worlds, dtype=int)
        self.phase_timer = wp.zeros(self.num_worlds, dtype=float)

        # GPU state for rate limiting (persistent across frames)
        init_6dof = [self.base_target_pos + self.base_target_rot]
        self.prev_base_target = wp.array(
            init_6dof * self.num_worlds,
            dtype=float,
            shape=(self.num_worlds, 6),
        )
        self.prev_gripper_ctrl = wp.zeros(self.num_worlds, dtype=float)

        # Phase config
        self.phase_durations = wp.array([1.5, 1.5, 1.5, 2.0, 1e10], dtype=float)

        # Precompute Z targets for state machine
        self.above_z = self.base_target_pos[2]
        self.grasp_z = self.table_height + self.box_size + 0.04
        self.lift_z = self.above_z + 0.1

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
            njmax=2000,
            nconmax=2000,
            iterations=15,
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

        # Track max object height for testing
        self.object_max_z = [self.object_init_z] * self.num_worlds if self.test_mode else None

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
            mjcf_path = f"{asset_path}/2f85.xml"
        return mjcf_path

    def _add_object_shape(self, builder, body):
        """Add grasp object shape, with SDF mesh for hydroelastic if needed."""
        s = self.box_size
        use_hydro = self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC

        if self.object_shape == ObjectShape.BOX:
            mesh = newton.Mesh.create_box(
                s, s, s, duplicate_vertices=True, compute_normals=False, compute_uvs=False, compute_inertia=True
            )
        elif self.object_shape == ObjectShape.SPHERE:
            mesh = newton.Mesh.create_sphere(s, compute_inertia=True)
        elif self.object_shape == ObjectShape.CYLINDER:
            mesh = newton.Mesh.create_cylinder(s, s, compute_inertia=True)
        elif self.object_shape == ObjectShape.CAPSULE:
            mesh = newton.Mesh.create_capsule(s, s, compute_inertia=True)
        else:
            raise ValueError(f"Unknown object shape: {self.object_shape}")

        if use_hydro:
            mesh.build_sdf(
                max_resolution=self.hydro_mesh_sdf_max_resolution,
                narrow_band_range=self.hydro_shape_cfg.sdf_narrow_band_range,
                margin=self.hydro_shape_cfg.gap,
            )

        builder.add_shape_mesh(
            body=body,
            mesh=mesh,
        )

    def _setup_finger_hydroelastic(self, builder):
        """Enable hydroelastic on finger contact shapes, disable on everything else."""
        if self.use_v4:
            # V4: disable pad box shapes, enable tongue meshes on follower bodies
            pad_body_indices = {
                i
                for i, lbl in enumerate(builder.body_label)
                if lbl.split("/")[-1] in ("left_pad", "right_pad", "left_silicone_pad", "right_silicone_pad")
            }
            follower_body_indices = {
                i
                for i, lbl in enumerate(builder.body_label)
                if lbl.split("/")[-1] in ("left_follower", "right_follower")
            }

            for shape_idx, body_idx in enumerate(builder.shape_body):
                if body_idx in pad_body_indices:
                    builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.HYDROELASTIC
                    builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.COLLIDE_SHAPES
                    builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.VISIBLE
                elif body_idx in follower_body_indices:
                    shape_lbl = builder.shape_label[shape_idx] or ""
                    is_tongue_collision = (
                        builder.shape_type[shape_idx] == newton.GeoType.MESH
                        and shape_lbl.split("/")[-1] in ("left_follower_geom_1", "right_follower_geom_0")
                        and "visual" not in shape_lbl
                    )
                    if is_tongue_collision:
                        self._build_shape_sdf(builder, shape_idx)
                        builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
        else:
            # V1: find pad collision shapes by exact label suffix
            pad_names = {"left_pad1", "left_pad2", "right_pad1", "right_pad2"}
            pad_shape_indices = set()
            for shape_idx, lbl in enumerate(builder.shape_label):
                if lbl and lbl.split("/")[-1] in pad_names:
                    pad_shape_indices.add(shape_idx)

            for shape_idx in pad_shape_indices:
                if builder.shape_type[shape_idx] == newton.GeoType.BOX:
                    # Convert BOX to MESH + SDF for hydroelastic contact
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
                        max_resolution=self.hydro_mesh_sdf_max_resolution,
                        narrow_band_range=self.hydro_shape_cfg.sdf_narrow_band_range,
                        margin=self.hydro_shape_cfg.gap,
                    )
                    builder.shape_type[shape_idx] = newton.GeoType.MESH
                    builder.shape_source[shape_idx] = mesh
                    builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
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
            max_resolution=self.hydro_mesh_sdf_max_resolution,
            narrow_band_range=self.hydro_shape_cfg.sdf_narrow_band_range,
            margin=self.hydro_shape_cfg.gap,
        )

    def _create_collision_pipeline(self):
        """Create collision pipeline based on collision mode."""
        if self.collision_mode == CollisionMode.MUJOCO:
            return None
        elif self.collision_mode == CollisionMode.NEWTON_DEFAULT:
            return newton.CollisionPipeline(self.model)
        elif self.collision_mode == CollisionMode.NEWTON_SDF:
            return newton.CollisionPipeline(
                self.model,
                reduce_contacts=True,
                broad_phase="explicit",
            )
        elif self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            return newton.CollisionPipeline(
                self.model,
                reduce_contacts=True,
                broad_phase="explicit",
                sdf_hydroelastic_config=HydroelasticSDF.Config(
                    output_contact_surface=hasattr(self.viewer, "renderer"),
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
                self.phase,
                self.phase_timer,
                self.above_z,
                self.grasp_z,
                self.lift_z,
                self.phase_durations,
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
            self._gui_phase_val = int(self.phase.numpy()[0])
            self._gui_timer_val = float(self.phase_timer.numpy()[0])

        # Track max object height for testing
        if self.test_mode:
            body_q = self.state_0.body_q.numpy()
            for w in range(self.num_worlds):
                obj_idx = w * self.bodies_per_world + self.object_body_local
                z = float(body_q[obj_idx][2])
                self.object_max_z[w] = max(self.object_max_z[w], z)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        if (
            self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC
            and self.collision_pipeline
            and self.collision_pipeline.hydroelastic_sdf is not None
        ):
            self.viewer.log_hydro_contact_surface(
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
                penetrating_only=True,
            )
        self.viewer.end_frame()

    def gui(self, imgui):
        # Phase display (cached — updated every _gui_read_interval frames)
        phase_names = ["go_to_target", "approach", "close", "lift", "hold"]
        imgui.text(f"Phase: {phase_names[self._gui_phase_val]}")
        imgui.text(f"Timer: {self._gui_timer_val:.2f}s")
        imgui.separator()

        # Manual mode toggle
        changed, new_manual = imgui.checkbox("Manual Mode", self.manual_mode)
        if changed:
            self.manual_mode = new_manual
            self._gpu_dirty = True
        imgui.separator()

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

        imgui.separator()

        # Base position — slider + input for each axis
        for i, (label, lo, hi) in enumerate([("Base X", -0.3, 0.3), ("Base Y", -0.3, 0.3), ("Base Z", 0.1, 0.6)]):
            changed, val = imgui.slider_float(label, self.base_target_pos[i], lo, hi, format="%.4f")
            if changed and self.manual_mode:
                self.base_target_pos[i] = val
                self._gpu_dirty = True
            changed, val = imgui.input_float(f"{label}##input", self.base_target_pos[i], format="%.4f")
            if changed and self.manual_mode:
                self.base_target_pos[i] = min(max(val, lo), hi)
                self._gpu_dirty = True

        imgui.separator()

        # Base rotation — slider + input for roll/pitch/yaw
        pi = 3.14159
        for i, label in enumerate(["Roll", "Pitch", "Yaw"]):
            changed, val = imgui.slider_float(label, self.base_target_rot[i], -pi, pi, format="%.3f")
            if changed and self.manual_mode:
                self.base_target_rot[i] = val
                self._gpu_dirty = True
            changed, val = imgui.input_float(f"{label}##input", self.base_target_rot[i], format="%.3f")
            if changed and self.manual_mode:
                self.base_target_rot[i] = min(max(val, -pi), pi)
                self._gpu_dirty = True

        # Velocity limits
        imgui.separator()
        imgui.text("Velocity Limits")
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

        # Isosurface toggle (hydroelastic only)
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC and hasattr(self.viewer, "renderer"):
            imgui.separator()
            changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
            if changed:
                self.viewer.show_hydro_contact_surface = self.show_isosurface

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert not np.any(np.isnan(body_q)), "Body positions contain NaN values"

        if self.test_mode:
            min_lift = 0.05  # Object should lift at least 5cm
            for w in range(self.num_worlds):
                max_z = self.object_max_z[w]
                lift = max_z - self.object_init_z
                assert lift > min_lift, (
                    f"World {w}: Object not lifted enough. "
                    f"init_z={self.object_init_z:.3f}, max_z={max_z:.3f}, lift={lift:.3f}"
                )

        version = "V4" if self.use_v4 else "V1"
        print(f"Robotiq 2F-85 {version} basic gripper example completed successfully")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=2, help="Number of simulated worlds.")
    parser.set_defaults(num_frames=480)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_worlds=args.num_worlds, args=args)

    newton.examples.run(example, args)
