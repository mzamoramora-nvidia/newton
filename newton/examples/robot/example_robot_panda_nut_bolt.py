# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Panda Nut Bolt
#
# Demonstrates a Franka Panda robot picking up an M20 nut and placing it
# on a bolt using hydroelastic contacts, gravity compensation, and
# IK-based control. The nut threads onto the bolt under gravity after
# release.
#
# Command: python -m newton.examples robot_panda_nut_bolt --world-count 4
#
###########################################################################

import argparse
import copy
import enum
import time
from dataclasses import replace

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact

# IsaacGymEnvs nut/bolt assets
ASSEMBLY_STR = "m20_loose"
ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

# SDF parameters for nut/bolt meshes (high resolution for threaded geometry)
SDF_MAX_RESOLUTION_NUT_BOLT = 256
SDF_NARROW_BAND_NUT_BOLT = (-0.005, 0.005)

# SDF parameters for gripper/table meshes
SDF_MAX_RESOLUTION_GRIPPER = 64
SDF_NARROW_BAND_GRIPPER = (-0.01, 0.01)


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    STABILIZE = 3
    LIFT = 4
    MOVE_TO_BOLT = 5
    REFINE_PLACE = 6
    RELEASE = 7
    RETRACT = 8
    HOME = 9


def load_mesh_with_sdf(
    mesh_file: str,
    gap: float = 0.005,
    scale: float = 1.0,
    max_resolution: int = 256,
    narrow_band_range: tuple[float, float] = (-0.005, 0.005),
) -> tuple[newton.Mesh, wp.vec3, np.ndarray]:
    """Load a triangle mesh, center it, and build an SDF.

    Args:
        mesh_file: Mesh file path.
        gap: Contact margin [m].
        scale: Uniform mesh scale [unitless].
        max_resolution: SDF grid resolution.
        narrow_band_range: SDF narrow band range [m].

    Returns:
        Tuple of ``(mesh, center_vec, half_extents)`` where ``center_vec``
        is the recenter offset [m] and ``half_extents`` is the mesh
        half-extents [m] after scaling.
    """
    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)

    min_extent = vertices.min(axis=0)
    max_extent = vertices.max(axis=0)
    center = (min_extent + max_extent) / 2
    half_extents = (max_extent - min_extent) / 2 * scale

    vertices = vertices - center
    center_vec = wp.vec3(center) * float(scale)

    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=max_resolution,
        narrow_band_range=narrow_band_range,
        margin=gap,
        scale=(scale, scale, scale),
    )
    return mesh, center_vec, half_extents


def add_mesh_object(
    builder: newton.ModelBuilder,
    mesh: newton.Mesh,
    transform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    label: str | None = None,
    center_vec: wp.vec3 | None = None,
    scale: float = 1.0,
    floating: bool = True,
) -> int:
    """Add a mesh shape, optionally as a new floating body.

    Args:
        builder: Model builder.
        mesh: Mesh geometry with SDF data.
        transform: Body/shape transform with position [m] and orientation.
        shape_cfg: Shape configuration.
        label: Body/shape label.
        center_vec: Mesh center offset [m] (from :func:`load_mesh_with_sdf`).
        scale: Uniform mesh scale [unitless].
        floating: If ``True`` create a new dynamic body, otherwise attach
            to the world body (``body=-1``).

    Returns:
        Created body index, or ``-1`` if fixed.
    """
    if center_vec is not None:
        center_world = wp.quat_rotate(transform.q, center_vec)
        transform = wp.transform(transform.p + center_world, transform.q)

    if floating:
        body = builder.add_body(label=label, xform=transform)
        builder.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale), cfg=shape_cfg)
        return body
    else:
        builder.add_shape_mesh(
            body=-1, mesh=mesh, scale=(scale, scale, scale), xform=transform, cfg=shape_cfg, label=label
        )
        return -1


def _transform_points_body_to_world(p_local: np.ndarray, body_idx: np.ndarray, body_q: np.ndarray) -> np.ndarray:
    """Transform body-local points to world frame using body transforms.

    Shapes on body=-1 (world body) pass their points through unchanged.

    Args:
        p_local: (N, 3) body-local points.
        body_idx: (N,) body index per point. Use -1 for world-attached shapes.
        body_q: (body_count, 7) body transforms as [px, py, pz, qx, qy, qz, qw].
    """
    out = p_local.copy()
    has_body = body_idx >= 0
    if not np.any(has_body):
        return out
    tf = body_q[body_idx[has_body]]
    pos = tf[:, :3]
    qx, qy, qz, qw = tf[:, 3], tf[:, 4], tf[:, 5], tf[:, 6]
    v = p_local[has_body]
    # Rodrigues-style quat rotation: rotated = v + 2*qw*cross(q_vec,v) + 2*cross(q_vec, cross(q_vec, v))
    tx = 2.0 * (qy * v[:, 2] - qz * v[:, 1])
    ty = 2.0 * (qz * v[:, 0] - qx * v[:, 2])
    tz = 2.0 * (qx * v[:, 1] - qy * v[:, 0])
    rx = v[:, 0] + qw * tx + (qy * tz - qz * ty)
    ry = v[:, 1] + qw * ty + (qz * tx - qx * tz)
    rz = v[:, 2] + qw * tz + (qx * ty - qy * tx)
    out[has_body, 0] = pos[:, 0] + rx
    out[has_body, 1] = pos[:, 1] + ry
    out[has_body, 2] = pos[:, 2] + rz
    return out


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_soft_limits: wp.array[float],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    task_offset_approach: wp.vec3,
    task_offset_lift: wp.vec3,
    task_offset_bolt_approach: wp.vec3,
    task_offset_place: wp.vec3,
    task_offset_retract: wp.vec3,
    grasping_z_offset: float,
    grasp_yaw_offset: float,
    gripper_open_pos: float,
    gripper_closed_pos: float,
    bolt_place_pos: wp.vec3,
    home_pos: wp.vec3,
    task_init_body_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ee_index: int,
    nut_body_index: int,
    num_bodies_per_world: int,
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_target_interpolated: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_target_interpolated: wp.array[wp.vec4],
    gripper_target: wp.array2d[wp.float32],
):
    tid = wp.tid()

    idx = task_idx[tid]
    task = task_schedule[idx]
    task_time_soft_limit = task_time_soft_limits[idx]

    task_time_elapsed[tid] += task_dt

    # Interpolation parameter t between 0 and 1
    t = wp.min(1.0, task_time_elapsed[tid] / task_time_soft_limit)

    # Get the EE position and rotation at the start of this task
    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_id])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_id])
    ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    # Get the current nut position and rotation
    nut_body_id = tid * num_bodies_per_world + nut_body_index
    nut_pos = wp.transform_get_translation(body_q[nut_body_id])
    nut_quat = wp.transform_get_rotation(body_q[nut_body_id])

    t_gripper = 0.0

    if task == TaskType.APPROACH.value:
        # Move above the nut, align gripper offset by grasp_yaw_offset about Z
        ee_pos_target[tid] = nut_pos + task_offset_approach
        yaw_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), grasp_yaw_offset)
        ee_quat_target = yaw_rot * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi) * wp.quat_inverse(nut_quat)
    elif task == TaskType.REFINE_APPROACH.value:
        # Descend to nut grasping height
        ee_pos_target[tid] = nut_pos + wp.vec3(0.0, 0.0, grasping_z_offset)
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value:
        # Close gripper around nut
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = t
    elif task == TaskType.STABILIZE.value:
        # Hold position with gripper closed to let contact forces settle
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.LIFT.value:
        # Lift nut upward
        ee_pos_target[tid] = ee_pos_prev + task_offset_lift
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.MOVE_TO_BOLT.value:
        # Move above the bolt
        ee_pos_target[tid] = bolt_place_pos + task_offset_bolt_approach
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
        t_gripper = 1.0
    elif task == TaskType.REFINE_PLACE.value:
        # Lower nut onto bolt top
        ee_pos_target[tid] = bolt_place_pos + task_offset_place
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        # Open gripper to release nut onto bolt threads
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0 - t
    elif task == TaskType.RETRACT.value:
        # Move upward away from bolt
        ee_pos_target[tid] = ee_pos_prev + task_offset_retract
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
    elif task == TaskType.HOME.value:
        ee_pos_target[tid] = home_pos
    else:
        ee_pos_target[tid] = home_pos

    ee_pos_target_interpolated[tid] = ee_pos_prev * (1.0 - t) + ee_pos_target[tid] * t
    ee_quat_interpolated = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)

    ee_rot_target[tid] = ee_quat_target[:4]
    ee_rot_target_interpolated[tid] = ee_quat_interpolated[:4]

    # Interpolate gripper between open and closed positions
    gripper_pos = gripper_open_pos * (1.0 - t_gripper) + gripper_closed_pos * t_gripper
    gripper_target[tid, 0] = gripper_pos
    gripper_target[tid, 1] = gripper_pos


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_soft_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    num_bodies_per_world: int,
    ee_index: int,
    pos_threshold_xy: float,
    pos_threshold_z: float,
    rot_threshold_deg: float,
    # outputs
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_init_body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    idx = task_idx[tid]
    task_time_soft_limit = task_time_soft_limits[idx]

    # Get the current position of the end-effector
    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_current = wp.transform_get_translation(body_q[ee_body_id])
    ee_quat_current = wp.transform_get_rotation(body_q[ee_body_id])

    # Per-axis position error
    pos_diff = ee_pos_target[tid] - ee_pos_current
    err_x = wp.abs(pos_diff[0])
    err_y = wp.abs(pos_diff[1])
    err_z = wp.abs(pos_diff[2])
    pos_settled = err_x < pos_threshold_xy and err_y < pos_threshold_xy and err_z < pos_threshold_z

    # Rotation error
    ee_quat_target = wp.quaternion(ee_rot_target[tid][:3], ee_rot_target[tid][3])
    quat_rel = ee_quat_current * wp.quat_inverse(ee_quat_target)
    rot_err = wp.abs(wp.degrees(2.0 * wp.atan2(wp.length(quat_rel[:3]), quat_rel[3])))
    rot_settled = rot_err < rot_threshold_deg

    # Advance when time elapsed, position settled, rotation settled, and not last task
    if (
        task_time_elapsed[tid] >= task_time_soft_limit
        and pos_settled
        and rot_settled
        and task_idx[tid] < wp.len(task_time_soft_limits) - 1
    ):
        task_idx[tid] += 1
        task_time_elapsed[tid] = 0.0

        body_id_start = tid * num_bodies_per_world
        for i in range(num_bodies_per_world):
            body_id = body_id_start + i
            task_init_body_q[body_id] = body_q[body_id]


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.collide_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.test_mode = args.test
        # Single flag that gates the expensive per-readback tracking path
        # (forces, friction, slip, penetration). --test implies tracking since
        # test_final consumes those metrics. When False, the sim runs lean.
        self._track_stats = args.test
        self.finger_kh = args.finger_kh
        self.nut_kh = args.nut_kh
        self.viewer = viewer

        # Scene layout
        self.table_height = 0.1
        self.table_pos = wp.vec3(0.0, -0.5, 0.5 * self.table_height)
        self.table_top_center = self.table_pos + wp.vec3(0.0, 0.0, 0.5 * self.table_height)
        self.robot_base_pos = self.table_top_center + wp.vec3(-0.5, 0.0, 0.0)

        # Bolt and nut positions on the table
        self.bolt_base_pos = self.table_top_center + wp.vec3(0.1, 0.0, 0.0)
        self.nut_start_pos = self.table_top_center + wp.vec3(0.05, 0.15, 0.0)

        # Task offsets
        self.task_offset_approach = wp.vec3(0.0, 0.0, 0.04)
        self.task_offset_lift = wp.vec3(0.0, 0.0, 0.15)
        self.task_offset_bolt_approach = wp.vec3(0.0, 0.0, 0.06)
        self.task_offset_place = wp.vec3(0.0, 0.0, 0.001)
        self.task_offset_retract = wp.vec3(0.0, 0.0, 0.10)
        self.grasping_z_offset = 0.001
        self.grasp_yaw_offset = 30.0 * wp.pi / 180.0  # rotate gripper 30 deg about Z wrt nut

        # Per-axis advance thresholds
        self.pos_threshold_xy = 0.0005  # 0.5 mm
        self.pos_threshold_z = 0.00075  # 0.75 mm (tighter for vertical precision)
        self.rot_threshold_deg = 0.5  # degrees

        # Gripper open/closed positions [m per finger]
        # Panda gripper: each finger travels 0 (closed) to 0.04m (open), we use 0.06 as open target
        self.gripper_open_pos = 0.06
        # Grasp margin: close to nut half-width minus a margin to avoid over-squeezing.
        # For a regular hex nut, the grasp width depends on the angle relative to the flats.
        # width(theta) = across_flats / cos(theta_eff), where theta_eff is the angle from
        # the nearest flat normal (0 to 30 deg range due to 6-fold symmetry).
        nut_across_flats = 0.030  # M20 nut, 30mm
        # theta_eff: angle from the nearest flat normal (0-30 deg, 6-fold symmetry)
        rem = self.grasp_yaw_offset % (np.pi / 3)  # remainder in [0, 60 deg)
        theta_eff = rem if rem <= np.pi / 6 else np.pi / 3 - rem
        nut_grasp_width = nut_across_flats / np.cos(theta_eff)
        grasp_margin = args.grasp_margin  # extra closure past first contact [m]
        self.gripper_closed_pos = max(0.0, nut_grasp_width / 2.0 - grasp_margin)
        gripper_ke = 100.0  # from joint_target_ke for finger joints
        expected_force_per_finger = gripper_ke * grasp_margin
        print(
            f"Grasp: yaw={np.degrees(self.grasp_yaw_offset):.0f}deg, "
            f"nut width={nut_grasp_width * 1000:.1f}mm, "
            f"margin={grasp_margin * 1000:.1f}mm, "
            f"finger target={self.gripper_closed_pos * 1000:.1f}mm, "
            f"expected force/finger={expected_force_per_finger:.3f}N"
        )

        # Download nut/bolt assets
        print("Downloading nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")

        # Load nut/bolt meshes with SDF
        bolt_mesh, bolt_center, bolt_half_extents = load_mesh_with_sdf(
            bolt_file,
            gap=0.005,
            max_resolution=SDF_MAX_RESOLUTION_NUT_BOLT,
            narrow_band_range=SDF_NARROW_BAND_NUT_BOLT,
        )
        nut_mesh, nut_center, _nut_half_extents = load_mesh_with_sdf(
            nut_file,
            gap=0.005,
            max_resolution=SDF_MAX_RESOLUTION_NUT_BOLT,
            narrow_band_range=SDF_NARROW_BAND_NUT_BOLT,
        )

        # Compute placement position above bolt top
        bolt_effective_center = wp.vec3(
            self.bolt_base_pos[0] + float(bolt_center[0]),
            self.bolt_base_pos[1] + float(bolt_center[1]),
            self.bolt_base_pos[2] + float(bolt_center[2]),
        )
        self.bolt_place_pos = wp.vec3(
            bolt_effective_center[0],
            bolt_effective_center[1],
            bolt_effective_center[2] + float(bolt_half_extents[2]) + 0.005,
        )

        # Store bolt frame for debug visualization (static, identity rotation)
        self.bolt_frame_pos = bolt_effective_center

        # Build robot with table, gravity compensation, and hydroelastic fingers
        robot_builder = self.build_franka_with_table()
        self.robot_body_count = robot_builder.body_count

        # Model for IK (robot only, no nut/bolt)
        self.model_single = copy.deepcopy(robot_builder).finalize()

        # Add bolt (fixed to ground) and nut (floating body).
        # Both bolt and nut use the MuJoCo minimum mu (1e-5, MJ_MINMU floor) so
        # the nut threads onto the bolt with the least possible friction.
        # MuJoCo combines pair friction as max(mu_a, mu_b), so:
        #   effective nut-bolt friction  = max(1e-5, 1e-5) = 1e-5 (smooth threading)
        #   effective nut-finger friction = max(1e-5, 1.0) = 1.0  (strong grip)
        bolt_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            mu=1e-5,
            ke=1e7,
            kd=1e4,
            gap=0.005,
            density=8000.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
            is_hydroelastic=True,
        )
        nut_cfg = replace(bolt_cfg, kh=self.nut_kh)

        bolt_xform = wp.transform(self.bolt_base_pos, wp.quat_identity())
        add_mesh_object(
            robot_builder,
            bolt_mesh,
            bolt_xform,
            bolt_cfg,
            label="bolt",
            center_vec=bolt_center,
            floating=False,
        )

        self.nut_body_index = robot_builder.body_count
        nut_xform = wp.transform(
            self.nut_start_pos,
            # wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
        )
        add_mesh_object(
            robot_builder,
            nut_mesh,
            nut_xform,
            nut_cfg,
            label="nut",
            center_vec=nut_center,
            floating=True,
        )

        # Build multi-world scene
        scene = newton.ModelBuilder()
        scene.replicate(robot_builder, self.world_count)
        ground_shape_idx = scene.add_ground_plane()

        # Filter out collisions between the robot base and the ground plane.
        # Robot sits on the table above ground; ground contacts are spurious.
        base_link_suffixes = ("/fr3_link0", "/fr3_link1")
        for shape_idx, body_idx in enumerate(scene.shape_body):
            if body_idx < 0:
                continue
            label = scene.body_label[body_idx]
            if label.endswith(base_link_suffixes):
                scene.add_shape_collision_filter_pair(shape_idx, ground_shape_idx)

        self.model = scene.finalize()
        self.num_bodies_per_world = self.model.body_count // self.world_count

        # Randomize the nut's initial XY position per world by patching
        # model.joint_q before the first eval_fk. The nut is a free body with
        # 7 joint coords (x, y, z, qx, qy, qz, qw) placed after the 9 robot
        # arm/gripper DOFs within each world's joint_q slice.
        self.nut_xy_jitter = args.nut_xy_jitter
        joint_q_per_world = self.model.joint_coord_count // self.world_count
        nut_joint_q_offset = 9
        rng = np.random.default_rng(args.seed)
        xy_jitter_np = rng.uniform(-self.nut_xy_jitter, self.nut_xy_jitter, size=(self.world_count, 2)).astype(
            np.float32
        )
        joint_q_view = self.model.joint_q.reshape((self.world_count, joint_q_per_world))
        current = joint_q_view[:, nut_joint_q_offset : nut_joint_q_offset + 2].numpy()
        updated = wp.array(current + xy_jitter_np, dtype=wp.float32)
        wp.copy(dest=joint_q_view[:, nut_joint_q_offset : nut_joint_q_offset + 2], src=updated)

        # Contact sensor: only needed when tracking is on. Must be created
        # BEFORE Contacts so the "force" attribute is requested on the model.
        if self._track_stats:
            self.contact_sensor = SensorContact(
                self.model,
                sensing_obj_bodies="nut",
                counterpart_bodies="*finger*",
            )
        else:
            self.contact_sensor = None

        # Collision pipeline with hydroelastic SDF.
        # output_contact_surface is required for the imgui Show Isosurface
        # toggle (needs a renderer) and for hydro penetration tracking.
        self.rigid_contact_max = 1000 * self.world_count
        has_renderer = hasattr(viewer, "renderer")
        sdf_hydroelastic_config = HydroelasticSDF.Config(
            output_contact_surface=self._track_stats or has_renderer,
            buffer_mult_iso=2,
            anchor_contact=args.anchor_contact,
            moment_matching=args.moment_matching,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=self.rigid_contact_max,
            broad_phase="explicit",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )
        self.contacts = self.collision_pipeline.contacts()

        # MuJoCo solver with Newton contacts
        num_contacts_per_world = self.rigid_contact_max // self.world_count
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=num_contacts_per_world,
            nconmax=num_contacts_per_world,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        # State and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        # IK and tasks
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.setup_ik()
        self.setup_tasks()

        # Viewer
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self.show_isosurface = False
        self.show_debug_frames = True
        self.show_single_world = False

        # GUI state (updated periodically in step, displayed in render_ui)
        self._gui_selected_world = 0
        self._gui_task_name = "—"
        self._gui_task_timer = 0.0
        self._gui_pos_err = np.zeros(3)
        self._gui_rot_err = 0.0
        self._gui_frame = 0
        self._gui_read_interval = 4  # read GPU state every N frames
        self._gui_contact_count = 0

        # Per-world contact force tracking
        self._cur_force = np.zeros(self.world_count)  # current total force on nut [N]
        self._max_force = np.zeros(self.world_count)  # all-time max total force [N]
        self._cur_friction = np.zeros(self.world_count)  # current friction force [N]
        self._max_friction = np.zeros(self.world_count)  # all-time max friction force [N]
        # Per-counterpart (left/right finger) forces for selected world
        self._gui_finger_forces = np.zeros(2)  # [left, right] magnitudes [N]
        self._nut_z_at_lift_start = np.full(self.world_count, np.nan)
        self._ee_z_at_lift_start = np.full(self.world_count, np.nan)
        self._nut_z_at_lift_end = np.full(self.world_count, np.nan)
        self._ee_z_at_lift_end = np.full(self.world_count, np.nan)
        self._slip = np.full(self.world_count, np.nan)  # computed after lift [m]

        # Per-state and per-penetration tracking arrays are only allocated when
        # the expensive tracking path is enabled.
        n_states = len(TaskType)
        if self._track_stats:
            self._state_force_max = np.zeros((self.world_count, n_states))
            self._state_friction_max = np.zeros((self.world_count, n_states))
            self._state_nut_z_start = np.full((self.world_count, n_states), np.nan)
            self._state_nut_z_end = np.full((self.world_count, n_states), np.nan)
            self._state_ee_z_start = np.full((self.world_count, n_states), np.nan)
            self._state_ee_z_end = np.full((self.world_count, n_states), np.nan)
            self._state_pen_rigid_nut_finger = np.zeros((self.world_count, n_states))
            self._state_pen_rigid_nut_bolt = np.zeros((self.world_count, n_states))
            self._state_pen_rigid_nut_table = np.zeros((self.world_count, n_states))
            self._state_pen_hydro_nut_finger = np.zeros((self.world_count, n_states))
            self._state_pen_hydro_nut_bolt = np.zeros((self.world_count, n_states))
            self._state_pen_hydro_nut_table = np.zeros((self.world_count, n_states))
            self._classify_shapes()

        # SPS (steps per second) tracker — lightweight, always on.
        self._sps_frame_count = 0
        self._sps_last_time = time.perf_counter()
        self._sps_samples: list[float] = []
        self._sps_warmup_done = False

        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(wp.vec3(0.5, 0.0, 0.5), -15, -140)
            self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))
            self.viewer.show_hydro_contact_surface = self.show_isosurface
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.capture()

    def build_franka_with_table(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=self.finger_kh,
            gap=0.01,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        shape_cfg_meshes = replace(shape_cfg, is_hydroelastic=True)

        builder.default_shape_cfg = shape_cfg

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )

        # Initial joint configuration
        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        builder.joint_q[:9] = [*init_q, 0.05, 0.05]
        builder.joint_target_pos[:9] = [*init_q, 1.0, 1.0]

        # Joint gains (high values needed with gravity compensation)
        builder.joint_target_ke[:9] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
        builder.joint_target_kd[:9] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
        builder.joint_effort_limit[:9] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        # Gravity compensation on arm joints
        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(7):
            gravcomp_attr.values[dof_idx] = True

        # Gravity compensation on arm and hand bodies
        # Body 0 = base, 1 = fr3_link0, 2-8 = fr3_link1-7,
        # 9-11 = fr3_link8/fr3_hand/fr3_hand_tcp, 12-13 = fingers
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(2, 14):
            gravcomp_body.values[body_idx] = 1.0

        # Find finger and hand bodies for hydroelastic SDF
        def find_body(name):
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        finger_body_indices = {
            find_body("fr3_leftfinger"),
            find_body("fr3_rightfinger"),
            find_body("fr3_hand"),
        }
        self.ee_index = find_body("fr3_hand_tcp")

        # Enable hydroelastic SDF on finger/hand mesh shapes
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx in finger_body_indices and builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = builder.shape_source[shape_idx]
                if mesh is not None and mesh.sdf is None:
                    shape_scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
                    if not np.allclose(shape_scale, 1.0):
                        mesh = mesh.copy(vertices=mesh.vertices * shape_scale, recompute_inertia=True)
                        builder.shape_source[shape_idx] = mesh
                        builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    mesh.build_sdf(
                        max_resolution=SDF_MAX_RESOLUTION_GRIPPER,
                        narrow_band_range=SDF_NARROW_BAND_GRIPPER,
                        margin=shape_cfg.gap,
                    )
                builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
            elif body_idx not in finger_body_indices:
                non_finger_shape_indices.append(shape_idx)

        # Convert non-finger shapes to convex hulls
        builder.approximate_meshes(
            method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True
        )

        # Table (hydroelastic mesh on world body)
        table_mesh = newton.Mesh.create_box(
            0.4,
            0.4,
            0.5 * self.table_height,
            duplicate_vertices=True,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=True,
        )
        table_mesh.build_sdf(
            max_resolution=SDF_MAX_RESOLUTION_GRIPPER,
            narrow_band_range=SDF_NARROW_BAND_GRIPPER,
            margin=shape_cfg.gap,
        )
        builder.add_shape_mesh(
            body=-1,
            mesh=table_mesh,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=shape_cfg_meshes,
            label="table",
        )

        return builder

    def setup_ik(self):
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        init_ee_pos = body_q_np[self.ee_index][:3]
        self.home_pos = wp.vec3(init_ee_pos)

        # Position objective (per-world)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([self.home_pos] * self.world_count, dtype=wp.vec3),
        )

        # Rotation objective (per-world)
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.transform_get_rotation(self.ee_tf)[:4]] * self.world_count, dtype=wp.vec4),
        )

        ik_dofs = self.model_single.joint_coord_count

        # Joint limit objective
        self.joint_limit_lower = wp.clone(self.model.joint_limit_lower.reshape((self.world_count, -1))[:, :ik_dofs])
        self.joint_limit_upper = wp.clone(self.model.joint_limit_upper.reshape((self.world_count, -1))[:, :ik_dofs])
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.joint_limit_lower.flatten(),
            joint_limit_upper=self.joint_limit_upper.flatten(),
        )

        # Variables the solver will update
        self.joint_q_ik = wp.clone(self.model.joint_q.reshape((self.world_count, -1))[:, :ik_dofs])

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def setup_tasks(self):
        task_schedule = [
            TaskType.APPROACH,
            TaskType.REFINE_APPROACH,
            TaskType.GRASP,
            TaskType.STABILIZE,
            TaskType.LIFT,
            TaskType.MOVE_TO_BOLT,
            TaskType.REFINE_PLACE,
            TaskType.RELEASE,
            TaskType.RETRACT,
            TaskType.HOME,
        ]
        task_time_soft_limits = [1.5, 1.0, 0.5, 1.0, 1.5, 2.0, 1.5, 0.5, 1.0, 2.0]

        self.task_counter = len(task_schedule)
        self.task_schedule = wp.array(task_schedule, dtype=wp.int32)
        self.task_time_soft_limits = wp.array(task_time_soft_limits, dtype=float)

        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.task_idx = wp.zeros(self.world_count, dtype=wp.int32)
        self.task_dt = self.frame_dt
        self.task_time_elapsed = wp.zeros(self.world_count, dtype=wp.float32)

        # Target arrays
        self.ee_pos_target = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_pos_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(self.world_count, dtype=wp.vec4)
        self.ee_rot_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec4)
        self.gripper_target = wp.zeros(shape=(self.world_count, 2), dtype=wp.float32)

    def set_joint_targets(self):
        wp.launch(
            set_target_pose_kernel,
            dim=self.world_count,
            inputs=[
                self.task_schedule,
                self.task_time_soft_limits,
                self.task_idx,
                self.task_time_elapsed,
                self.task_dt,
                self.task_offset_approach,
                self.task_offset_lift,
                self.task_offset_bolt_approach,
                self.task_offset_place,
                self.task_offset_retract,
                self.grasping_z_offset,
                self.grasp_yaw_offset,
                self.gripper_open_pos,
                self.gripper_closed_pos,
                self.bolt_place_pos,
                self.home_pos,
                self.task_init_body_q,
                self.state_0.body_q,
                self.ee_index,
                self.nut_body_index,
                self.num_bodies_per_world,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_target_interpolated,
                self.ee_rot_target,
                self.ee_rot_target_interpolated,
                self.gripper_target,
            ],
        )

        # Set IK targets
        self.pos_obj.set_target_positions(self.ee_pos_target_interpolated)
        self.rot_obj.set_target_rotations(self.ee_rot_target_interpolated)

        # Solve IK
        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        # Set joint target positions from IK solution + gripper targets
        joint_target_view = self.control.joint_target_pos.reshape((self.world_count, -1))
        wp.copy(dest=joint_target_view[:, :7], src=self.joint_q_ik[:, :7])
        wp.copy(dest=joint_target_view[:, 7:9], src=self.gripper_target[:, :2])

        # Advance tasks when conditions are met
        wp.launch(
            advance_task_kernel,
            dim=self.world_count,
            inputs=[
                self.task_time_soft_limits,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.num_bodies_per_world,
                self.ee_index,
                self.pos_threshold_xy,
                self.pos_threshold_z,
                self.rot_threshold_deg,
            ],
            outputs=[
                self.task_idx,
                self.task_time_elapsed,
                self.task_init_body_q,
            ],
        )

    def capture(self):
        self.capture_sim()
        self.capture_ik()

    def capture_sim(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def capture_ik(self):
        self.graph_ik = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if i % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.set_joint_targets()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self._gui_frame += 1

        # Periodic readback (avoid every-frame GPU sync). The heavy tracking
        # path is only entered when explicitly enabled via --test.
        if self._gui_frame % self._gui_read_interval == 0:
            if self._track_stats:
                self._update_gui_state()
                self._update_contact_and_slip()
            elif hasattr(self.viewer, "renderer"):
                # Lightweight GUI updates only: task name, timer, pos/rot error.
                self._update_gui_state()

        self._update_sps()

    def _update_sps(self):
        """Print SPS every ~1s with running average and std deviation."""
        self._sps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._sps_last_time
        if elapsed < 1.0:
            return
        sps = (self._sps_frame_count / elapsed) * self.world_count
        self._sps_frame_count = 0
        self._sps_last_time = now
        sps_per_env = sps / self.world_count

        if not self._sps_warmup_done:
            self._sps_warmup_done = True
            print(
                f"[SPS] sim_time={self.sim_time:.2f}s  {sps:.1f} steps/s  "
                f"({sps_per_env:.1f}/env, {self.world_count} worlds) (warmup)"
            )
            return

        self._sps_samples.append(sps)
        n = len(self._sps_samples)
        avg = sum(self._sps_samples) / n
        std = ((sum((s - avg) ** 2 for s in self._sps_samples) / (n - 1)) ** 0.5) if n > 1 else 0.0
        print(
            f"[SPS] sim_time={self.sim_time:.2f}s  {sps:.1f} steps/s  "
            f"({sps_per_env:.1f}/env, {self.world_count} worlds)  "
            f"avg={avg:.1f}  std={std:.1f}  n={n}"
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_hydro_contact_surface(
            (
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
                if self.collision_pipeline.hydroelastic_sdf is not None
                else None
            ),
            penetrating_only=True,
        )
        if self.show_debug_frames:
            self._log_debug_frames()
            self._log_test_zone()
            self._log_sampling_zone()
        else:
            self.viewer.log_lines("/debug_frames", None, None, None)
            self.viewer.log_lines("/test_zone", None, None, None)
            self.viewer.log_lines("/sampling_zone", None, None, None)
        self.viewer.end_frame()

    def _log_debug_frames(self):
        """Draw bolt, EE IK target, and nut frames as RGB axis lines per world."""
        axis_len = 0.05
        axes = [
            (wp.vec3(1.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)),  # X axis, red
            (wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 1.0, 0.0)),  # Y axis, green
            (wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 1.0)),  # Z axis, blue
        ]

        starts = []
        ends = []
        colors = []

        def add_frame(pos, q):
            for axis_dir, color in axes:
                starts.append(pos)
                ends.append(pos + wp.quat_rotate(q, axis_dir) * axis_len)
                colors.append(color)

        # Read world offsets
        offsets = self.viewer.world_offsets.numpy() if self.viewer.world_offsets is not None else None

        # Read EE targets and body transforms (GPU -> CPU)
        ee_pos_np = self.ee_pos_target_interpolated.numpy()
        ee_rot_np = self.ee_rot_target_interpolated.numpy()
        body_q_np = self.state_0.body_q.numpy()

        for w in range(self.world_count):
            off = (
                wp.vec3(float(offsets[w][0]), float(offsets[w][1]), float(offsets[w][2]))
                if offsets is not None
                else wp.vec3()
            )

            # Bolt frame (static, identity rotation)
            add_frame(self.bolt_frame_pos + off, wp.quat_identity())

            # EE IK target frame
            e = ee_pos_np[w]
            eq = ee_rot_np[w]
            add_frame(
                wp.vec3(float(e[0]), float(e[1]), float(e[2])) + off,
                wp.quat(float(eq[0]), float(eq[1]), float(eq[2]), float(eq[3])),
            )

            # Nut frame (dynamic body)
            nq = body_q_np[w * self.num_bodies_per_world + self.nut_body_index]
            add_frame(
                wp.vec3(float(nq[0]), float(nq[1]), float(nq[2])) + off,
                wp.quat(float(nq[3]), float(nq[4]), float(nq[5]), float(nq[6])),
            )

        self.viewer.log_lines(
            "/debug_frames",
            wp.array(starts, dtype=wp.vec3),
            wp.array(ends, dtype=wp.vec3),
            wp.array(colors, dtype=wp.vec3),
        )

    def _log_test_zone(self):
        """Draw two square outlines at the min and max Z levels that would pass the
        'nut reached the bottom of the bolt' test, centered on the bolt XY per world."""
        bolt_center_z = float(self.bolt_frame_pos[2])
        z_min = bolt_center_z - 0.010
        z_max = bolt_center_z + 0.010
        half = 0.03  # 60mm square side
        cx = float(self.bolt_frame_pos[0])
        cy = float(self.bolt_frame_pos[1])
        green = wp.vec3(0.0, 1.0, 0.0)  # pass-zone color

        offsets = self.viewer.world_offsets.numpy() if self.viewer.world_offsets is not None else None

        starts = []
        ends = []
        colors = []

        def add_square(z, off):
            corners = [
                wp.vec3(cx - half, cy - half, z) + off,
                wp.vec3(cx + half, cy - half, z) + off,
                wp.vec3(cx + half, cy + half, z) + off,
                wp.vec3(cx - half, cy + half, z) + off,
            ]
            for i in range(4):
                starts.append(corners[i])
                ends.append(corners[(i + 1) % 4])
                colors.append(green)

        for w in range(self.world_count):
            off = (
                wp.vec3(float(offsets[w][0]), float(offsets[w][1]), float(offsets[w][2]))
                if offsets is not None
                else wp.vec3()
            )
            add_square(z_min, off)
            add_square(z_max, off)

        self.viewer.log_lines(
            "/test_zone",
            wp.array(starts, dtype=wp.vec3),
            wp.array(ends, dtype=wp.vec3),
            wp.array(colors, dtype=wp.vec3),
        )

    def _log_sampling_zone(self):
        """Draw a yellow square outline on the table surface showing the XY region
        from which the nut's initial position is sampled."""
        if self.nut_xy_jitter <= 0.0:
            self.viewer.log_lines("/sampling_zone", None, None, None)
            return

        cx = float(self.nut_start_pos[0])
        cy = float(self.nut_start_pos[1])
        z = float(self.table_top_center[2]) + 0.001  # just above the table
        half = self.nut_xy_jitter
        yellow = wp.vec3(1.0, 1.0, 0.0)

        offsets = self.viewer.world_offsets.numpy() if self.viewer.world_offsets is not None else None

        starts = []
        ends = []
        colors = []

        for w in range(self.world_count):
            off = (
                wp.vec3(float(offsets[w][0]), float(offsets[w][1]), float(offsets[w][2]))
                if offsets is not None
                else wp.vec3()
            )
            corners = [
                wp.vec3(cx - half, cy - half, z) + off,
                wp.vec3(cx + half, cy - half, z) + off,
                wp.vec3(cx + half, cy + half, z) + off,
                wp.vec3(cx - half, cy + half, z) + off,
            ]
            for i in range(4):
                starts.append(corners[i])
                ends.append(corners[(i + 1) % 4])
                colors.append(yellow)

        self.viewer.log_lines(
            "/sampling_zone",
            wp.array(starts, dtype=wp.vec3),
            wp.array(ends, dtype=wp.vec3),
            wp.array(colors, dtype=wp.vec3),
        )

    def _update_gui_state(self):
        """Read GPU state for the selected world (called periodically from step)."""
        w = self._gui_selected_world

        # Current task name and timer
        task_val = int(self.task_idx.numpy()[w])
        self._gui_task_name = TaskType(self.task_schedule.numpy()[task_val]).name
        self._gui_task_timer = float(self.task_time_elapsed.numpy()[w])

        # EE position error (target vs actual)
        body_q = self.state_0.body_q.numpy()
        ee_body_id = w * self.num_bodies_per_world + self.ee_index
        ee_pos = body_q[ee_body_id][:3]
        ee_target = self.ee_pos_target.numpy()[w]
        self._gui_pos_err = ee_target - ee_pos

        # EE rotation error (degrees)
        ee_quat = body_q[ee_body_id][3:7]
        target_vec4 = self.ee_rot_target.numpy()[w]
        # q_rel = q_current * inv(q_target)
        q_cur = ee_quat
        q_tgt = target_vec4
        dot = abs(np.dot(q_cur, q_tgt))
        dot = min(dot, 1.0)
        self._gui_rot_err = np.degrees(2.0 * np.arccos(dot))

    def _classify_shapes(self):
        """Pre-compute shape->category boolean maps and per-shape world indices
        so penetration contacts can be filtered per category and per world."""
        n_shapes = self.model.shape_count
        shape_body = self.model.shape_body.numpy()
        body_labels = self.model.body_label
        shape_labels = list(self.model.shape_label)  # may contain None
        shape_world = (
            self.model.shape_world.numpy()
            if self.model.shape_world is not None
            else np.full(n_shapes, -1, dtype=np.int32)
        )

        self._is_nut_shape = np.zeros(n_shapes, dtype=bool)
        self._is_finger_shape = np.zeros(n_shapes, dtype=bool)
        self._is_bolt_shape = np.zeros(n_shapes, dtype=bool)
        self._is_table_shape = np.zeros(n_shapes, dtype=bool)

        for s in range(n_shapes):
            body = int(shape_body[s])
            body_label = body_labels[body] if body >= 0 else ""
            shape_label = shape_labels[s] if shape_labels[s] is not None else ""
            combined = f"{body_label}/{shape_label}"

            if "nut" in combined:
                self._is_nut_shape[s] = True
            elif "finger" in combined:
                self._is_finger_shape[s] = True
            elif "bolt" in combined:
                self._is_bolt_shape[s] = True
            elif "table" in combined:
                self._is_table_shape[s] = True

        self._shape_world = shape_world.astype(np.int32)

    def _compute_rigid_penetration_per_world(self):
        """Compute current max penetration per world per category from rigid contacts.

        Returns a dict mapping category name ('nut_finger', 'nut_bolt',
        'nut_table') to a numpy array of shape (world_count,) in metres.
        """
        out = {
            "nut_finger": np.zeros(self.world_count),
            "nut_bolt": np.zeros(self.world_count),
            "nut_table": np.zeros(self.world_count),
        }
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if count == 0:
            return out

        p0_local = self.contacts.rigid_contact_point0.numpy()[:count]
        p1_local = self.contacts.rigid_contact_point1.numpy()[:count]
        n = self.contacts.rigid_contact_normal.numpy()[:count]
        s0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        s1 = self.contacts.rigid_contact_shape1.numpy()[:count]

        valid = (s0 >= 0) & (s1 >= 0)
        if not np.any(valid):
            return out

        # rigid_contact_point0/1 are in body-local frame; transform to world
        shape_body = self.model.shape_body.numpy()
        body_q = self.state_0.body_q.numpy()
        b0 = shape_body[s0]
        b1 = shape_body[s1]
        p0 = _transform_points_body_to_world(p0_local, b0, body_q)
        p1 = _transform_points_body_to_world(p1_local, b1, body_q)

        depths = np.sum((p0 - p1) * n, axis=1)
        self._scatter_pen_per_world(s0, s1, depths, valid, out)
        return out

    def _compute_hydro_penetration_per_world(self):
        """Compute current max penetration per world per category from the
        hydroelastic contact surface (per-face centroid depths).
        """
        out = {
            "nut_finger": np.zeros(self.world_count),
            "nut_bolt": np.zeros(self.world_count),
            "nut_table": np.zeros(self.world_count),
        }
        hydro = self.collision_pipeline.hydroelastic_sdf
        if hydro is None:
            return out
        surface = hydro.get_contact_surface()
        if surface is None:
            return out

        face_count = int(surface.face_contact_count.numpy()[0])
        if face_count == 0:
            return out

        depths = surface.contact_surface_depth.numpy()[:face_count]
        shape_pair = surface.contact_surface_shape_pair.numpy()[:face_count]
        s0 = shape_pair[:, 0]
        s1 = shape_pair[:, 1]

        valid = (s0 >= 0) & (s1 >= 0)
        # hydro depths are negative inside (penetrating); flip sign to match rigid convention
        pen = -depths
        self._scatter_pen_per_world(s0, s1, pen, valid, out)
        return out

    def _scatter_pen_per_world(self, s0, s1, depths, valid, out):
        """Classify each contact by pair category and scatter max depth per world.

        Positive depth = penetrating. Only depths > 0 and valid (shape >=0) count.
        """
        nut0 = self._is_nut_shape[s0] & valid
        nut1 = self._is_nut_shape[s1] & valid
        fin0 = self._is_finger_shape[s0]
        fin1 = self._is_finger_shape[s1]
        bol0 = self._is_bolt_shape[s0]
        bol1 = self._is_bolt_shape[s1]
        tbl0 = self._is_table_shape[s0]
        tbl1 = self._is_table_shape[s1]

        is_nut_finger = (nut0 & fin1) | (nut1 & fin0)
        is_nut_bolt = (nut0 & bol1) | (nut1 & bol0)
        is_nut_table = (nut0 & tbl1) | (nut1 & tbl0)

        # Derive per-contact world index: prefer the nut side's world; fall back
        # to the counterpart's world when the nut shape is global (shape_world < 0).
        nut_shape = np.where(nut0, s0, np.where(nut1, s1, 0))
        other_shape = np.where(nut0, s1, s0)
        nut_world = self._shape_world[nut_shape]
        other_world = self._shape_world[other_shape]
        world = np.where(nut_world >= 0, nut_world, other_world)

        penetrating = depths > 0
        for mask, key in (
            (is_nut_finger & penetrating, "nut_finger"),
            (is_nut_bolt & penetrating, "nut_bolt"),
            (is_nut_table & penetrating, "nut_table"),
        ):
            if not np.any(mask):
                continue
            sel_world = world[mask]
            sel_depth = depths[mask]
            in_range = (sel_world >= 0) & (sel_world < self.world_count)
            if not np.any(in_range):
                continue
            np.maximum.at(out[key], sel_world[in_range], sel_depth[in_range])

    def _update_contact_and_slip(self):
        """Track contact forces and vertical slippage (called periodically from step)."""
        # Update contact forces via the solver and sensor
        self._gui_contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        self.solver.update_contacts(self.contacts, self.state_0)
        self.contact_sensor.update(self.state_0, self.contacts)

        # Read per-world total force and friction on the nut from finger contacts
        total_force_np = self.contact_sensor.total_force.numpy()  # (world_count, 3)
        friction_force_np = self.contact_sensor.total_force_friction.numpy()  # (world_count, 3)
        for w in range(self.world_count):
            f_mag = float(np.linalg.norm(total_force_np[w]))
            fr_mag = float(np.linalg.norm(friction_force_np[w]))
            self._cur_force[w] = f_mag
            self._cur_friction[w] = fr_mag
            self._max_force[w] = max(self._max_force[w], f_mag)
            self._max_friction[w] = max(self._max_friction[w], fr_mag)

        # Read per-finger forces for the selected world (from force_matrix)
        w = self._gui_selected_world
        if self.contact_sensor.force_matrix is not None:
            fm = self.contact_sensor.force_matrix.numpy()  # (world_count, n_counterparts, 3)
            n_cols = fm.shape[1]
            for c in range(min(n_cols, 2)):
                self._gui_finger_forces[c] = float(np.linalg.norm(fm[w, c]))

        # Slippage + per-state tracking: snapshot Z positions at state entry/exit
        tasks_np = self.task_idx.numpy()
        schedule_np = self.task_schedule.numpy()
        body_q = self.state_0.body_q.numpy()

        # Compute per-world per-category max penetration from both sources.
        cur_pen_rigid = self._compute_rigid_penetration_per_world()  # dict category -> (world_count,)
        cur_pen_hydro = self._compute_hydro_penetration_per_world()

        for w in range(self.world_count):
            task_type = int(schedule_np[int(tasks_np[w])])
            ee_body_id = w * self.num_bodies_per_world + self.ee_index
            nut_body_id = w * self.num_bodies_per_world + self.nut_body_index
            ee_z = float(body_q[ee_body_id][2])
            nut_z = float(body_q[nut_body_id][2])

            # Per-state tracking: record start Z on first observation, update end Z
            # and max force/friction while in this state.
            if np.isnan(self._state_nut_z_start[w, task_type]):
                self._state_nut_z_start[w, task_type] = nut_z
                self._state_ee_z_start[w, task_type] = ee_z
            self._state_nut_z_end[w, task_type] = nut_z
            self._state_ee_z_end[w, task_type] = ee_z
            self._state_force_max[w, task_type] = max(self._state_force_max[w, task_type], self._cur_force[w])
            self._state_friction_max[w, task_type] = max(self._state_friction_max[w, task_type], self._cur_friction[w])

            # Per-state max penetration per category per source
            self._state_pen_rigid_nut_finger[w, task_type] = max(
                self._state_pen_rigid_nut_finger[w, task_type], cur_pen_rigid["nut_finger"][w]
            )
            self._state_pen_rigid_nut_bolt[w, task_type] = max(
                self._state_pen_rigid_nut_bolt[w, task_type], cur_pen_rigid["nut_bolt"][w]
            )
            self._state_pen_rigid_nut_table[w, task_type] = max(
                self._state_pen_rigid_nut_table[w, task_type], cur_pen_rigid["nut_table"][w]
            )
            self._state_pen_hydro_nut_finger[w, task_type] = max(
                self._state_pen_hydro_nut_finger[w, task_type], cur_pen_hydro["nut_finger"][w]
            )
            self._state_pen_hydro_nut_bolt[w, task_type] = max(
                self._state_pen_hydro_nut_bolt[w, task_type], cur_pen_hydro["nut_bolt"][w]
            )
            self._state_pen_hydro_nut_table[w, task_type] = max(
                self._state_pen_hydro_nut_table[w, task_type], cur_pen_hydro["nut_table"][w]
            )

            # Record at LIFT start (first frame we see LIFT)
            if task_type == TaskType.LIFT and np.isnan(self._nut_z_at_lift_start[w]):
                self._nut_z_at_lift_start[w] = nut_z
                self._ee_z_at_lift_start[w] = ee_z

            # Record at MOVE_TO_BOLT (lift finished)
            if task_type == TaskType.MOVE_TO_BOLT:
                self._nut_z_at_lift_end[w] = nut_z
                self._ee_z_at_lift_end[w] = ee_z
                # Compute slip: how much more the EE lifted than the nut
                if not np.isnan(self._nut_z_at_lift_start[w]):
                    ee_lift = ee_z - self._ee_z_at_lift_start[w]
                    nut_lift = nut_z - self._nut_z_at_lift_start[w]
                    self._slip[w] = ee_lift - nut_lift

    def render_ui(self, imgui):
        imgui.separator()

        # World selector
        changed, val = imgui.slider_int("World", self._gui_selected_world, 0, self.world_count - 1)
        if changed:
            self._gui_selected_world = val
            if self.show_single_world:
                self.viewer.set_visible_worlds([val])

        changed, self.show_single_world = imgui.checkbox("Show Single World", self.show_single_world)
        if changed:
            if self.show_single_world:
                self.viewer.set_visible_worlds([self._gui_selected_world])
            else:
                self.viewer.set_visible_worlds(None)

        imgui.separator()

        # Task state machine info
        imgui.text(f"Task: {self._gui_task_name}")
        imgui.text(f"Timer: {self._gui_task_timer:.2f}s")

        # Per-axis position error with threshold check
        err = self._gui_pos_err * 1000.0  # mm
        limit_xy = self.pos_threshold_xy * 1000.0
        limit_z = self.pos_threshold_z * 1000.0
        ok_x = "*" if abs(err[0]) < limit_xy else " "
        ok_y = "*" if abs(err[1]) < limit_xy else " "
        ok_z = "*" if abs(err[2]) < limit_z else " "
        imgui.text(f"Pos x:{err[0]:+.2f}mm {ok_x}  y:{err[1]:+.2f}mm {ok_y}  z:{err[2]:+.2f}mm {ok_z}")
        imgui.text(f"  limit: xy<{limit_xy:.2f}mm  z<{limit_z:.2f}mm")

        # Rotation error with threshold check
        ok_r = "*" if self._gui_rot_err < self.rot_threshold_deg else " "
        imgui.text(f"Rot err: {self._gui_rot_err:.2f} deg {ok_r}  limit<{self.rot_threshold_deg:.1f}")

        imgui.text(f"Frame: {self._gui_frame}  Sim: {self.sim_time:.2f}s")

        imgui.separator()

        # Contact / force / penetration metrics are only available when tracking
        # is enabled (via --test). Otherwise show a hint so the user knows.
        w = self._gui_selected_world
        if self._track_stats:
            imgui.text(f"Contacts: {self._gui_contact_count}")
            imgui.text(f"Total force:    {self._cur_force[w]:.3f} N  (max: {self._max_force[w]:.3f} N)")
            imgui.text(f"Friction force: {self._cur_friction[w]:.3f} N  (max: {self._max_friction[w]:.3f} N)")
            imgui.text(f"Finger L: {self._gui_finger_forces[0]:.3f} N  R: {self._gui_finger_forces[1]:.3f} N")

            if not np.isnan(self._slip[w]):
                imgui.text(f"Slip: {self._slip[w] * 1000:.2f} mm")
            else:
                imgui.text("Slip: —")

            pen_rf = self._state_pen_rigid_nut_finger[w].max() * 1000.0
            pen_rb = self._state_pen_rigid_nut_bolt[w].max() * 1000.0
            pen_rt = self._state_pen_rigid_nut_table[w].max() * 1000.0
            pen_hf = self._state_pen_hydro_nut_finger[w].max() * 1000.0
            pen_hb = self._state_pen_hydro_nut_bolt[w].max() * 1000.0
            pen_ht = self._state_pen_hydro_nut_table[w].max() * 1000.0
            imgui.text("Max pen [mm]  (rigid | hydro)")
            imgui.text(f"  nut-finger: {pen_rf:.3f} | {pen_hf:.3f}")
            imgui.text(f"  nut-bolt:   {pen_rb:.3f} | {pen_hb:.3f}")
            imgui.text(f"  nut-table:  {pen_rt:.3f} | {pen_ht:.3f}")
        else:
            imgui.text("Stats tracking disabled. Run with --test to enable.")

        imgui.separator()

        # Visualization toggles
        changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
        if changed:
            self.viewer.show_hydro_contact_surface = self.show_isosurface
        _, self.show_debug_frames = imgui.checkbox("Show Debug Frames", self.show_debug_frames)

    def test_final(self):
        body_q = self.state_0.body_q.numpy()

        bolt_xy = np.array([float(self.bolt_place_pos[0]), float(self.bolt_place_pos[1])])
        table_top_z = float(self.table_top_center[2])

        # Print stats for all worlds first (before assertions)
        print("\nPer-world report:")
        errors = []
        # Tolerances
        xy_tol = 0.02  # 20mm
        # "Nut at the bottom" reference: from nut_bolt_hydro pre-fix data, when
        # the nut threads fully down the bolt it ends up with its body center
        # approximately at the bolt body center (within a few mm). Require our
        # nut's body center to be within +/-10mm of the bolt center to count
        # as "reached the bottom".
        bolt_center_z = float(self.bolt_frame_pos[2])
        nut_at_bottom_z_ref = bolt_center_z
        nut_at_bottom_z_min = nut_at_bottom_z_ref - 0.010
        nut_at_bottom_z_max = nut_at_bottom_z_ref + 0.010
        print(
            f"  bolt_center_z={bolt_center_z:.4f}  "
            f"nut_at_bottom_z_ref={nut_at_bottom_z_ref:.4f}  "
            f"valid range=[{nut_at_bottom_z_min:.4f}, {nut_at_bottom_z_max:.4f}]"
        )

        for world_id in range(self.world_count):
            nut_body_id = world_id * self.num_bodies_per_world + self.nut_body_index
            nut_pos = body_q[nut_body_id][:3]
            nut_xy = nut_pos[:2]
            xy_error = float(np.linalg.norm(nut_xy - bolt_xy))

            if self._track_stats:
                f_max = self._max_force[world_id]
                fr_max = self._max_friction[world_id]
                slip = self._slip[world_id]
                slip_str = f"{slip * 1000:.2f}mm" if not np.isnan(slip) else "n/a"
                lift_str = ""
                if not np.isnan(slip):
                    nut_lift = self._nut_z_at_lift_end[world_id] - self._nut_z_at_lift_start[world_id]
                    lift_str = f"  lift={nut_lift * 1000:.1f}mm"
                z_after_release = self._state_nut_z_end[world_id, TaskType.RELEASE.value]
                descent = z_after_release - nut_pos[2] if not np.isnan(z_after_release) else np.nan
                descent_str = f"{descent * 1000:+.2f}mm" if not np.isnan(descent) else "n/a"
                print(
                    f"  World {world_id}: "
                    f"nut=({nut_pos[0]:.4f}, {nut_pos[1]:.4f}, {nut_pos[2]:.4f})  "
                    f"xy_err={xy_error * 1000:.1f}mm  "
                    f"descent={descent_str}  "
                    f"force_max={f_max:.3f}N  friction_max={fr_max:.3f}N  "
                    f"slip={slip_str}{lift_str}"
                )
            else:
                print(
                    f"  World {world_id}: "
                    f"nut=({nut_pos[0]:.4f}, {nut_pos[1]:.4f}, {nut_pos[2]:.4f})  "
                    f"xy_err={xy_error * 1000:.1f}mm"
                )

            # Collect errors instead of asserting immediately
            if not np.all(np.isfinite(nut_pos)):
                errors.append(f"World {world_id}: Nut has non-finite position {nut_pos}")
                continue
            # Phase 1: nut XY aligned with bolt
            if xy_error >= xy_tol:
                errors.append(f"World {world_id}: XY placement error {xy_error * 1000:.1f}mm > {xy_tol * 1000:.0f}mm")
            # Phase 2: nut must be above the table (didn't fall through)
            if nut_pos[2] <= table_top_z:
                errors.append(f"World {world_id}: Nut below table. z={nut_pos[2]:.4f}")
            # Phase 3: nut must reach the bottom of the bolt (threaded down).
            if not (nut_at_bottom_z_min < nut_pos[2] < nut_at_bottom_z_max):
                errors.append(
                    f"World {world_id}: Nut did not reach the bottom of the bolt. "
                    f"nut_z={nut_pos[2]:.4f} outside expected range "
                    f"[{nut_at_bottom_z_min:.4f}, {nut_at_bottom_z_max:.4f}] "
                    f"(bolt_center={bolt_center_z:.4f}, target={nut_at_bottom_z_ref:.4f})"
                )

        # Per-state report and penetration aggregate are only printed when
        # tracking is enabled. Skip both cleanly otherwise.
        if not self._track_stats:
            if errors:
                raise AssertionError("\n".join(errors))
            print(f"test_final passed for all {self.world_count} worlds")
            return

        # Per-state report: tabular format, one row per (state, world), with
        # force/slip metrics and rigid+hydro penetration side-by-side.
        print("\nPer-state report:")
        header_line1 = (
            f"  {'State':<16}{'World':>6}  "
            f"{'force':>8}  {'friction':>9}  "
            f"{'nut_dz':>10}  {'slip':>10}  "
            f"{'rigid pen_max [mm]':^34}  "
            f"{'hydro pen_max [mm]':^34}"
        )
        header_line2 = (
            f"  {'':<16}{'':>6}  "
            f"{'[N]':>8}  {'[N]':>9}  "
            f"{'[mm]':>10}  {'[mm]':>10}  "
            f"{'nut-finger':>10} {'nut-bolt':>10} {'nut-table':>10}  "
            f"{'nut-finger':>10} {'nut-bolt':>10} {'nut-table':>10}"
        )
        print(header_line1)
        print(header_line2)
        print("  " + "-" * (len(header_line1) - 2))

        for state in TaskType:
            visited = ~np.isnan(self._state_nut_z_start[:, state.value])
            if not np.any(visited):
                continue
            for world_id in range(self.world_count):
                if not visited[world_id]:
                    continue
                s = state.value
                nut_dz = self._state_nut_z_end[world_id, s] - self._state_nut_z_start[world_id, s]
                ee_dz = self._state_ee_z_end[world_id, s] - self._state_ee_z_start[world_id, s]
                slip_dz = ee_dz - nut_dz
                f_max = self._state_force_max[world_id, s]
                fr_max = self._state_friction_max[world_id, s]
                pen_rf = self._state_pen_rigid_nut_finger[world_id, s] * 1000.0
                pen_rb = self._state_pen_rigid_nut_bolt[world_id, s] * 1000.0
                pen_rt = self._state_pen_rigid_nut_table[world_id, s] * 1000.0
                pen_hf = self._state_pen_hydro_nut_finger[world_id, s] * 1000.0
                pen_hb = self._state_pen_hydro_nut_bolt[world_id, s] * 1000.0
                pen_ht = self._state_pen_hydro_nut_table[world_id, s] * 1000.0
                print(
                    f"  {state.name:<16}{world_id:>6}  "
                    f"{f_max:>8.3f}  {fr_max:>9.3f}  "
                    f"{nut_dz * 1000:>+10.2f}  {slip_dz * 1000:>+10.2f}  "
                    f"{pen_rf:>10.3f} {pen_rb:>10.3f} {pen_rt:>10.3f}  "
                    f"{pen_hf:>10.3f} {pen_hb:>10.3f} {pen_ht:>10.3f}"
                )

        # Informational aggregate: rigid vs hydro penetration by category
        print("\nPenetration sources (rigid vs hydro), aggregated over all worlds and states:")
        for name, rigid_arr, hydro_arr in (
            ("nut-finger", self._state_pen_rigid_nut_finger, self._state_pen_hydro_nut_finger),
            ("nut-bolt", self._state_pen_rigid_nut_bolt, self._state_pen_hydro_nut_bolt),
            ("nut-table", self._state_pen_rigid_nut_table, self._state_pen_hydro_nut_table),
        ):
            r_max = float(rigid_arr.max()) * 1000.0
            h_max = float(hydro_arr.max()) * 1000.0
            max_abs_diff = float(np.abs(rigid_arr - hydro_arr).max()) * 1000.0
            ratio = (r_max / h_max) if h_max > 1e-6 else float("nan")
            print(
                f"  {name:<10}  rigid_max={r_max:>6.3f}mm  hydro_max={h_max:>6.3f}mm  "
                f"rigid/hydro={ratio:>5.2f}  max_abs_diff={max_abs_diff:>6.3f}mm"
            )

        if errors:
            raise AssertionError("\n".join(errors))

        print(f"test_final passed for all {self.world_count} worlds")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=1800)
        parser.set_defaults(world_count=4)
        parser.add_argument(
            "--grasp-margin",
            type=float,
            default=0.018,
            help="Extra gripper closure past first contact with the nut [m].",
        )
        parser.add_argument(
            "--nut-xy-jitter",
            type=float,
            default=0.03,
            help="Half-width of the uniform XY sampling region for the nut's initial position [m].",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="RNG seed for nut XY randomization.",
        )
        parser.add_argument(
            "--finger-kh",
            type=float,
            default=1e11,
            help="Hydroelastic stiffness for finger/hand/table shapes [Pa/m].",
        )
        parser.add_argument(
            "--nut-kh",
            type=float,
            default=1e11,
            help="Hydroelastic stiffness for the nut shape [Pa/m].",
        )
        parser.add_argument(
            "--anchor-contact",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable HydroelasticSDF.Config.anchor_contact to preserve moment balance.",
        )
        parser.add_argument(
            "--moment-matching",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Enable HydroelasticSDF.Config.moment_matching to preserve max friction moment "
                "after contact reduction (implicitly enables anchor_contact)."
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
