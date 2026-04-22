# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Panda Nut Bolt
#
# Demonstrates a Franka Panda robot picking up a nut and placing it
# on a bolt using hydroelastic contacts, gravity compensation, and
# IK-based control. The nut threads onto the bolt under gravity after
# release. Supports M20 (default) and M16 assemblies via --assembly.
#
# Command: python -m newton.examples robot_panda_nut_bolt --world-count 4
# Command: python -m newton.examples robot_panda_nut_bolt --assembly m16_loose
#
###########################################################################

import argparse
import copy
import enum
import time
from dataclasses import dataclass, replace

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
ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

# SDF parameters for nut/bolt meshes (high resolution for threaded geometry)
SDF_MAX_RESOLUTION_NUT_BOLT = 256
SDF_NARROW_BAND_NUT_BOLT = (-0.005, 0.005)

# SDF parameters for gripper/table meshes
SDF_MAX_RESOLUTION_GRIPPER = 64
SDF_NARROW_BAND_GRIPPER = (-0.01, 0.01)

# Scene layout (metres, world frame)
TABLE_HEIGHT = 0.1
TABLE_HALF_EXTENT = 0.4
TABLE_POS = wp.vec3(0.0, -0.5, 0.5 * TABLE_HEIGHT)
TABLE_TOP_CENTER = TABLE_POS + wp.vec3(0.0, 0.0, 0.5 * TABLE_HEIGHT)
ROBOT_BASE_POS = TABLE_TOP_CENTER + wp.vec3(-0.5, 0.0, 0.0)
BOLT_BASE_POS = TABLE_TOP_CENTER + wp.vec3(0.1, 0.0, 0.0)
NUT_START_POS = TABLE_TOP_CENTER + wp.vec3(0.05, 0.15, 0.0)

# EE task offsets — added to per-task anchor poses.
TASK_OFFSET_APPROACH = wp.vec3(0.0, 0.0, 0.04)
TASK_OFFSET_LIFT = wp.vec3(0.0, 0.0, 0.15)
TASK_OFFSET_PLACE = wp.vec3(0.0, 0.0, 0.001)
TASK_OFFSET_RETRACT = wp.vec3(0.0, 0.0, 0.10)

# Grasp geometry
GRASP_YAW_OFFSET_DEG = 30.0  # gripper yaw relative to nut, about Z
GRIPPER_OPEN_POS = 0.06  # per-finger full-open position [m]
GRIPPER_KE = 100.0  # must match joint_target_ke for finger joints


@dataclass(frozen=True)
class AssemblyParams:
    """Per-assembly geometry and tuning defaults [m].

    ``test_zone_z_{min,max}_offset`` are the Z bounds (relative to
    ``bolt_center_z``) of the nut-final-position acceptance band used by
    :meth:`Example.test_final` and drawn as squares in the viewer. They
    are tunable at runtime via the ImGui panel ("Test zone Z"); these
    defaults are the values found via that tuning.
    """

    assembly_str: str
    nut_across_flats: float
    task_offset_bolt_approach_z: float
    grasp_margin: float
    screw_grip_margin: float
    screw_regrip_clearance: float
    grasping_z_offset: float
    test_zone_z_min_offset: float
    test_zone_z_max_offset: float


ASSEMBLIES: dict[str, AssemblyParams] = {
    "m20_loose": AssemblyParams(
        assembly_str="m20_loose",
        nut_across_flats=0.030,
        task_offset_bolt_approach_z=0.06,
        grasp_margin=0.018,
        screw_grip_margin=0.018,
        screw_regrip_clearance=0.003,
        grasping_z_offset=0.001,
        test_zone_z_min_offset=-0.008,
        test_zone_z_max_offset=0.025,
    ),
    "m16_loose": AssemblyParams(
        assembly_str="m16_loose",
        nut_across_flats=0.024,
        task_offset_bolt_approach_z=0.04,
        grasp_margin=0.014,
        screw_grip_margin=0.014,
        screw_regrip_clearance=0.002,
        grasping_z_offset=0.004,
        test_zone_z_min_offset=-0.002,
        test_zone_z_max_offset=0.012,
    ),
}

# IK convergence thresholds for the task-FSM advance condition.
POS_THRESHOLD_XY = 0.0005  # 0.5 mm
POS_THRESHOLD_Z = 0.00075  # 0.75 mm
ROT_THRESHOLD_DEG = 0.5

# Collision pipeline buffer sizing (per world).
RIGID_CONTACT_MAX_PER_WORLD = 1000

# Task-FSM soft time budgets [seconds].
#   Pre-screw covers APPROACH, REFINE_APPROACH, GRASP, STABILIZE, LIFT,
#   MOVE_TO_BOLT, REFINE_PLACE. Screw cycles repeat (rotate, regrip).
#   Post-screw covers RELEASE, RETRACT, HOME.
PRE_SCREW_LIMITS = [0.6, 0.4, 0.5, 1.0, 1.5, 2.0, 1.5]
SCREW_CYCLE_LIMITS = [0.8, 0.35]
POST_SCREW_LIMITS = [0.5, 0.4, 0.8]


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    STABILIZE = 3
    LIFT = 4
    MOVE_TO_BOLT = 5
    REFINE_PLACE = 6
    SCREW_ROTATE = 7
    SCREW_REGRIP = 8
    RELEASE = 9
    RETRACT = 10
    HOME = 11


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
    gripper_screw_grip_pos: float,
    gripper_screw_regrip_pos: float,
    screw_angle: float,
    screw_regrip_z_offset: float,
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
    elif task == TaskType.SCREW_ROTATE.value:
        # XY stays locked on the bolt axis (don't follow nut XY drift — that
        # would let the nut keep walking off the bolt). Z follows the nut's
        # current height so the arm descends with the threading.
        ee_pos_target[tid] = wp.vec3(bolt_place_pos[0], bolt_place_pos[1], nut_pos[2])
        # Clockwise-from-above (negative about +Z) drives a right-handed thread
        # downward. The previous sign was counterclockwise — actively unscrewing.
        yaw_step = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -screw_angle)
        ee_quat_target = yaw_step * ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.SCREW_REGRIP.value:
        # Fully open gripper, lift slightly, rotate yaw back to the pre-rotate
        # angle. XY stays locked on the bolt axis.
        ee_pos_target[tid] = wp.vec3(bolt_place_pos[0], bolt_place_pos[1], nut_pos[2] + screw_regrip_z_offset)
        yaw_step = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), screw_angle)
        ee_quat_target = yaw_step * ee_quat_prev
        t_gripper = 0.0
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

    # Pick the "closed" finger target per task: tight full closure for the
    # initial grasp/lift/place, but a gentler closure during the screw motion
    # so the rotating gripper doesn't bind against the nut.
    if task == TaskType.SCREW_ROTATE.value:
        closed_pos = gripper_screw_grip_pos
    else:
        closed_pos = gripper_closed_pos

    # The "open" target during SCREW_REGRIP is partial — just enough to clear
    # the rotating hex corners — so the fingers don't have to complete a full
    # stroke to the 6cm open position between every screw cycle.
    if task == TaskType.SCREW_REGRIP.value:
        open_pos = gripper_screw_regrip_pos
    else:
        open_pos = gripper_open_pos

    # Interpolate gripper between open and closed positions
    gripper_pos = open_pos * (1.0 - t_gripper) + closed_pos * t_gripper
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


# Shape categories used by the penetration kernels. Pair categories are the
# output slot index: 0=nut_finger, 1=nut_bolt, 2=nut_table.
SHAPE_CAT_OTHER = 0
SHAPE_CAT_NUT = 1
SHAPE_CAT_FINGER = 2
SHAPE_CAT_BOLT = 3
SHAPE_CAT_TABLE = 4


@wp.func
def _classify_pair_nut_x(cat0: int, cat1: int) -> int:
    """Return pair category index (0=nut_finger, 1=nut_bolt, 2=nut_table) or -1
    if the pair is not a tracked nut-vs-X contact."""
    # Put the nut on the left.
    a = cat0
    b = cat1
    if b == SHAPE_CAT_NUT:
        a = cat1
        b = cat0
    if a != SHAPE_CAT_NUT:
        return -1
    if b == SHAPE_CAT_FINGER:
        return 0
    if b == SHAPE_CAT_BOLT:
        return 1
    if b == SHAPE_CAT_TABLE:
        return 2
    return -1


@wp.kernel(enable_backward=False)
def update_rigid_penetration_kernel(
    contact_count: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_category: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    # output (wc, 3) — pair categories: 0=nut_finger, 1=nut_bolt, 2=nut_table
    pen_rigid: wp.array2d[float],
):
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    s0 = contact_shape0[tid]
    s1 = contact_shape1[tid]
    if s0 < 0 or s1 < 0:
        return
    pair_cat = _classify_pair_nut_x(shape_category[s0], shape_category[s1])
    if pair_cat < 0:
        return

    # Transform contact points from body-local to world and compute depth.
    b0 = shape_body[s0]
    b1 = shape_body[s1]
    p0 = contact_point0[tid]
    p1 = contact_point1[tid]
    if b0 >= 0:
        p0 = wp.transform_point(body_q[b0], p0)
    if b1 >= 0:
        p1 = wp.transform_point(body_q[b1], p1)
    depth = wp.dot(p0 - p1, contact_normal[tid])
    if depth <= 0.0:
        return

    world = shape_world[s0]
    if world < 0:
        world = shape_world[s1]
    if world < 0 or world >= pen_rigid.shape[0]:
        return
    wp.atomic_max(pen_rigid, world, pair_cat, depth)


@wp.kernel(enable_backward=False)
def update_hydro_penetration_kernel(
    face_count: wp.array[wp.int32],
    face_depth: wp.array[float],
    face_shape_pair: wp.array[wp.vec2i],
    shape_world: wp.array[wp.int32],
    shape_category: wp.array[wp.int32],
    # output (wc, 3)
    pen_hydro: wp.array2d[float],
):
    tid = wp.tid()
    if tid >= face_count[0]:
        return
    pair = face_shape_pair[tid]
    s0 = pair[0]
    s1 = pair[1]
    if s0 < 0 or s1 < 0:
        return
    pair_cat = _classify_pair_nut_x(shape_category[s0], shape_category[s1])
    if pair_cat < 0:
        return
    # Hydro depth is negative when penetrating; flip sign to match rigid convention.
    depth = -face_depth[tid]
    if depth <= 0.0:
        return
    world = shape_world[s0]
    if world < 0:
        world = shape_world[s1]
    if world < 0 or world >= pen_hydro.shape[0]:
        return
    wp.atomic_max(pen_hydro, world, pair_cat, depth)


@wp.kernel(enable_backward=False)
def update_debug_frame_lines_kernel(
    world_offsets: wp.array[wp.vec3],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    bolt_frame_pos: wp.vec3,
    nut_body_index: int,
    num_bodies_per_world: int,
    axis_len: float,
    # output: 9 * world_count vec3s — per world, 3 frames (bolt, EE, nut), 3 axes each.
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    w = wp.tid()
    off = world_offsets[w]
    base = w * 9

    # Bolt frame (identity rotation, static position).
    bolt_pos = bolt_frame_pos + off
    starts[base + 0] = bolt_pos
    ends[base + 0] = bolt_pos + wp.vec3(axis_len, 0.0, 0.0)
    starts[base + 1] = bolt_pos
    ends[base + 1] = bolt_pos + wp.vec3(0.0, axis_len, 0.0)
    starts[base + 2] = bolt_pos
    ends[base + 2] = bolt_pos + wp.vec3(0.0, 0.0, axis_len)

    # EE target frame.
    ee_pos = ee_pos_target[w] + off
    ee_q = wp.quaternion(ee_rot_target[w][:3], ee_rot_target[w][3])
    starts[base + 3] = ee_pos
    ends[base + 3] = ee_pos + wp.quat_rotate(ee_q, wp.vec3(axis_len, 0.0, 0.0))
    starts[base + 4] = ee_pos
    ends[base + 4] = ee_pos + wp.quat_rotate(ee_q, wp.vec3(0.0, axis_len, 0.0))
    starts[base + 5] = ee_pos
    ends[base + 5] = ee_pos + wp.quat_rotate(ee_q, wp.vec3(0.0, 0.0, axis_len))

    # Nut frame.
    nut_xform = body_q[w * num_bodies_per_world + nut_body_index]
    nut_pos = wp.transform_get_translation(nut_xform) + off
    nut_q = wp.transform_get_rotation(nut_xform)
    starts[base + 6] = nut_pos
    ends[base + 6] = nut_pos + wp.quat_rotate(nut_q, wp.vec3(axis_len, 0.0, 0.0))
    starts[base + 7] = nut_pos
    ends[base + 7] = nut_pos + wp.quat_rotate(nut_q, wp.vec3(0.0, axis_len, 0.0))
    starts[base + 8] = nut_pos
    ends[base + 8] = nut_pos + wp.quat_rotate(nut_q, wp.vec3(0.0, 0.0, axis_len))


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
        self.nut_bolt_mu = float(args.nut_bolt_mu)
        self.viewer = viewer

        # Scene layout — see module-level TABLE_/ROBOT_/BOLT_/NUT_* constants.
        self.table_height = TABLE_HEIGHT
        self.table_pos = TABLE_POS
        self.table_top_center = TABLE_TOP_CENTER
        self.robot_base_pos = ROBOT_BASE_POS
        self.bolt_base_pos = BOLT_BASE_POS
        self.nut_start_pos = NUT_START_POS

        # Assembly-specific parameters (M20 or M16).
        self.assembly = ASSEMBLIES[args.assembly]
        grasp_margin = args.grasp_margin if args.grasp_margin is not None else self.assembly.grasp_margin
        screw_grip_margin = (
            args.screw_grip_margin if args.screw_grip_margin is not None else self.assembly.screw_grip_margin
        )
        screw_regrip_clearance = (
            args.screw_regrip_clearance
            if args.screw_regrip_clearance is not None
            else self.assembly.screw_regrip_clearance
        )

        # Task offsets — see module-level TASK_OFFSET_* constants.
        self.task_offset_approach = TASK_OFFSET_APPROACH
        self.task_offset_lift = TASK_OFFSET_LIFT
        self.task_offset_bolt_approach = wp.vec3(0.0, 0.0, self.assembly.task_offset_bolt_approach_z)
        self.task_offset_place = TASK_OFFSET_PLACE
        self.task_offset_retract = TASK_OFFSET_RETRACT
        self.grasping_z_offset = self.assembly.grasping_z_offset
        self.grasp_yaw_offset = float(wp.radians(GRASP_YAW_OFFSET_DEG))
        self.screw_cycles = int(args.screw_cycles)
        self.screw_angle = float(wp.radians(float(args.screw_angle_deg)))
        self.screw_regrip_z_offset = float(args.screw_regrip_z_offset)
        self.max_nut_tilt_deg = float(args.max_nut_tilt_deg)

        # Per-axis IK-settle thresholds (module constants).
        self.pos_threshold_xy = POS_THRESHOLD_XY
        self.pos_threshold_z = POS_THRESHOLD_Z
        self.rot_threshold_deg = ROT_THRESHOLD_DEG

        # Gripper geometry. Panda fingers travel 0 (closed) to 0.04 m (open);
        # GRIPPER_OPEN_POS=0.06 is the slightly-overcommanded open target.
        self.gripper_open_pos = GRIPPER_OPEN_POS
        nut_across_flats = self.assembly.nut_across_flats
        rem = self.grasp_yaw_offset % (wp.pi / 3.0)
        theta_eff = rem if rem <= wp.pi / 6.0 else wp.pi / 3.0 - rem
        nut_grasp_width = nut_across_flats / float(wp.cos(theta_eff))
        self.gripper_closed_pos = max(0.0, nut_grasp_width / 2.0 - grasp_margin)
        self.gripper_screw_grip_pos = max(0.0, nut_grasp_width / 2.0 - screw_grip_margin)
        nut_corner_radius = nut_across_flats / (2.0 * float(wp.cos(wp.pi / 6.0)))
        self.gripper_screw_regrip_pos = nut_corner_radius + screw_regrip_clearance
        expected_force_per_finger = GRIPPER_KE * grasp_margin
        expected_screw_force_per_finger = GRIPPER_KE * screw_grip_margin
        print(f"Assembly: {self.assembly.assembly_str}, nut across flats={nut_across_flats * 1000:.0f}mm")
        print(
            f"Grasp: yaw={float(wp.degrees(self.grasp_yaw_offset)):.0f}deg, "
            f"nut width={nut_grasp_width * 1000:.1f}mm, "
            f"margin={grasp_margin * 1000:.1f}mm, "
            f"finger target={self.gripper_closed_pos * 1000:.1f}mm, "
            f"expected force/finger={expected_force_per_finger:.3f}N"
        )
        print(
            f"Screw grip: margin={screw_grip_margin * 1000:.1f}mm  "
            f"finger target={self.gripper_screw_grip_pos * 1000:.1f}mm  "
            f"expected force/finger={expected_screw_force_per_finger:.3f}N"
        )

        # Download nut/bolt assets
        self.build_scene(args)

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
        # output_contact_surface is expensive (per-frame triangulation of the
        # pressure field). It's required for the imgui Show Isosurface toggle
        # and for hydro penetration tracking (--test). Default off even when a
        # viewer is present — pass --show-isosurface to opt in at startup.
        self.rigid_contact_max = RIGID_CONTACT_MAX_PER_WORLD * self.world_count
        want_iso = self._track_stats or args.show_isosurface
        sdf_hydroelastic_config = HydroelasticSDF.Config(
            output_contact_surface=want_iso,
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
        # The GUI toggle only governs rendering. Computation of the contact
        # surface is gated on --show-isosurface at init (above) because the
        # collision pipeline buffers are allocated there.
        self.show_isosurface = args.show_isosurface
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
            # Nut Z-axis tilt relative to world Z [rad] — max per state per world.
            # 0 = nut is perfectly upright (its Z axis aligned with world Z).
            # A growing tilt during SCREW_* states indicates the gripper is
            # pushing the nut off-axis and threading will not work well.
            self._state_nut_tilt_max = np.zeros((self.world_count, n_states))
            self._classify_shapes()
            # GPU output buffers for the penetration-reduction kernels.
            # Columns: 0=nut_finger, 1=nut_bolt, 2=nut_table.
            self._pen_rigid_current_wp = wp.zeros((self.world_count, 3), dtype=float)
            self._pen_hydro_current_wp = wp.zeros((self.world_count, 3), dtype=float)

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

        self._setup_line_buffers()
        self.capture()

    def build_scene(self, args):
        """Assemble the scene: download nut/bolt assets, build the robot +
        table builder (via :meth:`build_franka_with_table`), add nut and bolt
        shapes, replicate across worlds, add the ground plane, and finalize
        ``self.model``. Also patches the nut's initial XY per world.
        """
        # Download IsaacGymEnvs mesh assets.
        assembly_str = self.assembly.assembly_str
        print(f"Downloading nut/bolt assets ({assembly_str})...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        bolt_file = str(asset_path / f"factory_bolt_{assembly_str}.obj")
        nut_file = str(asset_path / f"factory_nut_{assembly_str}_subdiv_3x.obj")

        # Load nut/bolt meshes with SDF (high resolution for threaded geometry).
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

        # Cached bolt-top pose for IK target + debug-frame drawing.
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
        self.bolt_frame_pos = bolt_effective_center

        # Test-zone Z bounds (offsets from bolt_center_z [m]) come from the
        # assembly defaults and are tunable at runtime via the ImGui panel.
        # The /test_zone squares drawn in the viewer follow these values.
        self.test_zone_z_min_offset = self.assembly.test_zone_z_min_offset
        self.test_zone_z_max_offset = self.assembly.test_zone_z_max_offset
        print(
            f"Bolt: center_z={float(bolt_effective_center[2]):.4f}  "
            f"top_z={float(bolt_effective_center[2]) + float(bolt_half_extents[2]):.4f}  "
            f"test_zone min={float(bolt_effective_center[2]) + self.test_zone_z_min_offset:.4f}  "
            f"max={float(bolt_effective_center[2]) + self.test_zone_z_max_offset:.4f}"
        )

        # Robot + table (gravity compensation, hydroelastic fingers, coarse SDFs on arm).
        robot_builder = self.build_franka_with_table()
        self.robot_body_count = robot_builder.body_count

        # Arm-only model for IK (no nut/bolt).
        self.model_single = copy.deepcopy(robot_builder).finalize()

        # Nut + bolt shape config. Both use the same mu (default 0.2, matches
        # devel-example-nut-bolt). MuJoCo combines pair friction as
        # max(mu_a, mu_b), so:
        #   effective nut-bolt friction  = max(mu, mu)  = mu   (holds nut under gravity between screw cycles)
        #   effective nut-finger friction = max(mu, 1.0) = 1.0 (strong grip)
        # pure mu=1e-5 let gravity free-spin the nut after release, making Z
        # unreliable. Higher mu resists sliding (slower descent per cycle).
        bolt_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            mu=self.nut_bolt_mu,
            ke=1e7,
            kd=1e4,
            gap=0.005,
            density=8000.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
            is_hydroelastic=True,
        )
        nut_cfg = replace(bolt_cfg, kh=self.nut_kh)

        add_mesh_object(
            robot_builder,
            bolt_mesh,
            wp.transform(self.bolt_base_pos, wp.quat_identity()),
            bolt_cfg,
            label="bolt",
            center_vec=bolt_center,
            floating=False,
        )
        self.nut_body_index = robot_builder.body_count
        add_mesh_object(
            robot_builder,
            nut_mesh,
            wp.transform(self.nut_start_pos),
            nut_cfg,
            label="nut",
            center_vec=nut_center,
            floating=True,
        )

        # Replicate into a multi-world scene + ground plane, then filter out
        # spurious ground contacts on the robot base (the robot sits on the table).
        scene = newton.ModelBuilder()
        scene.replicate(robot_builder, self.world_count)
        ground_shape_idx = scene.add_ground_plane()
        base_link_suffixes = ("/fr3_link0", "/fr3_link1")
        for shape_idx, body_idx in enumerate(scene.shape_body):
            if body_idx < 0:
                continue
            if scene.body_label[body_idx].endswith(base_link_suffixes):
                scene.add_shape_collision_filter_pair(shape_idx, ground_shape_idx)

        self.model = scene.finalize()
        self.num_bodies_per_world = self.model.body_count // self.world_count

        # Randomize the nut's initial XY per world. The nut is a free body with
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
        # builder.approximate_meshes(
        #     method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True
        # )

        # Attach coarse (non-hydroelastic) SDFs to every non-finger mesh shape.
        # Without an SDF, mesh-vs-{mesh,primitive} collision must walk the BVH
        # for every query point, which dominates frame time. A low-res SDF
        # gives O(1) distance lookups at ~1 MB per shape — see the Newton
        # collisions docs.
        for shape_idx in non_finger_shape_indices:
            if builder.shape_type[shape_idx] != newton.GeoType.MESH:
                continue
            mesh = builder.shape_source[shape_idx]
            if mesh is None or mesh.sdf is not None:
                continue
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

        # Table (hydroelastic mesh on world body)
        table_mesh = newton.Mesh.create_box(
            TABLE_HALF_EXTENT,
            TABLE_HALF_EXTENT,
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
        pre_screw = [
            TaskType.APPROACH,
            TaskType.REFINE_APPROACH,
            TaskType.GRASP,
            TaskType.STABILIZE,
            TaskType.LIFT,
            TaskType.MOVE_TO_BOLT,
            TaskType.REFINE_PLACE,
        ]
        # Time budgets live in module-level PRE_SCREW_LIMITS, SCREW_CYCLE_LIMITS,
        # POST_SCREW_LIMITS. Screw cycles repeat (rotate, regrip) per cycle.
        pre_screw_limits = PRE_SCREW_LIMITS
        screw_block = [TaskType.SCREW_ROTATE, TaskType.SCREW_REGRIP] * self.screw_cycles
        screw_block_limits = SCREW_CYCLE_LIMITS * self.screw_cycles

        post_screw = [
            TaskType.RELEASE,
            TaskType.RETRACT,
            TaskType.HOME,
        ]
        post_screw_limits = POST_SCREW_LIMITS

        task_schedule = pre_screw + screw_block + post_screw
        task_time_soft_limits = pre_screw_limits + screw_block_limits + post_screw_limits

        self.task_counter = len(task_schedule)
        self.task_schedule = wp.array(task_schedule, dtype=wp.int32)
        # Cache the host copy — task_schedule is immutable after setup, so
        # per-frame tracking can read from here instead of syncing each frame.
        self._task_schedule_np = np.asarray(task_schedule, dtype=np.int32)
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
                self.gripper_screw_grip_pos,
                self.gripper_screw_regrip_pos,
                self.screw_angle,
                self.screw_regrip_z_offset,
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

    def _setup_line_buffers(self):
        """Allocate the line buffers used by :meth:`_log_debug_frames`,
        :meth:`_log_test_zone`, and :meth:`_log_sampling_zone` once. The
        static buffers (test/sampling zones, all colors) are filled here;
        :meth:`_log_debug_frames` refreshes its starts/ends via a kernel
        each frame so no per-frame host-side allocation is needed.
        """
        wc = self.world_count
        # Cache world offsets on the GPU. set_world_offsets was called once
        # during viewer setup, so this is static for the run.
        if self.viewer.world_offsets is not None:
            self._world_offsets_wp = wp.clone(self.viewer.world_offsets)
        else:
            self._world_offsets_wp = wp.zeros(wc, dtype=wp.vec3)
        offsets_np = self._world_offsets_wp.numpy()

        # ---- /debug_frames: 9 segments/world (bolt + EE + nut, each 3 axes).
        self._lines_debug_axis_len = 0.05
        self._lines_debug_starts = wp.zeros(9 * wc, dtype=wp.vec3)
        self._lines_debug_ends = wp.zeros(9 * wc, dtype=wp.vec3)
        # Colors are static: R, G, B per frame, 3 frames per world.
        rgb = [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)]
        self._lines_debug_colors = wp.array(rgb * (3 * wc), dtype=wp.vec3)

        # ---- /test_zone: 8 segments/world (two squares x 4 edges).
        # Z levels follow the tunable test_zone_z_{min,max}_offset so the
        # on-screen squares match what test_final actually checks.
        self._test_zone_world_offsets_np = offsets_np
        self._rebuild_test_zone_lines()
        self._lines_test_colors = wp.array([wp.vec3(0.0, 1.0, 0.0)] * (8 * wc), dtype=wp.vec3)

        # ---- /sampling_zone: 4 segments/world, only when jitter > 0.
        if self.nut_xy_jitter > 0.0:
            sx = float(self.nut_start_pos[0])
            sy = float(self.nut_start_pos[1])
            sz = float(self.table_top_center[2]) + 0.001
            sh = float(self.nut_xy_jitter)
            yellow = wp.vec3(1.0, 1.0, 0.0)
            sample_starts, sample_ends = [], []
            for w in range(wc):
                off = wp.vec3(float(offsets_np[w][0]), float(offsets_np[w][1]), float(offsets_np[w][2]))
                corners = [
                    wp.vec3(sx - sh, sy - sh, sz) + off,
                    wp.vec3(sx + sh, sy - sh, sz) + off,
                    wp.vec3(sx + sh, sy + sh, sz) + off,
                    wp.vec3(sx - sh, sy + sh, sz) + off,
                ]
                for i in range(4):
                    sample_starts.append(corners[i])
                    sample_ends.append(corners[(i + 1) % 4])
            self._lines_sampling_starts = wp.array(sample_starts, dtype=wp.vec3)
            self._lines_sampling_ends = wp.array(sample_ends, dtype=wp.vec3)
            self._lines_sampling_colors = wp.array([yellow] * (4 * wc), dtype=wp.vec3)
        else:
            self._lines_sampling_starts = None
            self._lines_sampling_ends = None
            self._lines_sampling_colors = None

    def _log_debug_frames(self):
        """Refresh the debug-frame line buffers via a kernel and log them."""
        wp.launch(
            update_debug_frame_lines_kernel,
            dim=self.world_count,
            inputs=[
                self._world_offsets_wp,
                self.ee_pos_target_interpolated,
                self.ee_rot_target_interpolated,
                self.state_0.body_q,
                self.bolt_frame_pos,
                self.nut_body_index,
                self.num_bodies_per_world,
                self._lines_debug_axis_len,
            ],
            outputs=[self._lines_debug_starts, self._lines_debug_ends],
        )
        self.viewer.log_lines(
            "/debug_frames",
            self._lines_debug_starts,
            self._lines_debug_ends,
            self._lines_debug_colors,
        )

    def _log_test_zone(self):
        """Draw two square outlines at the min and max Z levels that would pass
        the 'nut reached the bottom of the bolt' test. Buffers are rebuilt
        only when :attr:`test_zone_z_min_offset` / :attr:`test_zone_z_max_offset`
        change via the ImGui panel."""
        self.viewer.log_lines(
            "/test_zone",
            self._lines_test_starts,
            self._lines_test_ends,
            self._lines_test_colors,
        )

    def _rebuild_test_zone_lines(self):
        """Rebuild /test_zone line buffers from the current Z-offset settings."""
        bolt_center_z = float(self.bolt_frame_pos[2])
        cx = float(self.bolt_frame_pos[0])
        cy = float(self.bolt_frame_pos[1])
        half = 0.03  # 60 mm square side
        z_lo = bolt_center_z + self.test_zone_z_min_offset
        z_hi = bolt_center_z + self.test_zone_z_max_offset
        starts, ends = [], []
        for w in range(self.world_count):
            off = wp.vec3(
                float(self._test_zone_world_offsets_np[w][0]),
                float(self._test_zone_world_offsets_np[w][1]),
                float(self._test_zone_world_offsets_np[w][2]),
            )
            for z in (z_lo, z_hi):
                corners = [
                    wp.vec3(cx - half, cy - half, z) + off,
                    wp.vec3(cx + half, cy - half, z) + off,
                    wp.vec3(cx + half, cy + half, z) + off,
                    wp.vec3(cx - half, cy + half, z) + off,
                ]
                for i in range(4):
                    starts.append(corners[i])
                    ends.append(corners[(i + 1) % 4])
        self._lines_test_starts = wp.array(starts, dtype=wp.vec3)
        self._lines_test_ends = wp.array(ends, dtype=wp.vec3)

    def _log_sampling_zone(self):
        """Draw a yellow square showing the XY region the nut is sampled from.
        Entirely static; omitted when jitter is zero."""
        if self._lines_sampling_starts is None:
            self.viewer.log_lines("/sampling_zone", None, None, None)
            return
        self.viewer.log_lines(
            "/sampling_zone",
            self._lines_sampling_starts,
            self._lines_sampling_ends,
            self._lines_sampling_colors,
        )

    def _update_gui_state(self):
        """Read GPU state for the selected world (called periodically from step)."""
        w = self._gui_selected_world

        # Current task name and timer
        task_val = int(self.task_idx.numpy()[w])
        self._gui_task_name = TaskType(self._task_schedule_np[task_val]).name
        self._gui_task_timer = float(self.task_time_elapsed.numpy()[w])

        # EE position error (target vs actual)
        body_q = self.state_0.body_q.numpy()
        ee_body_id = w * self.num_bodies_per_world + self.ee_index
        ee_pos = body_q[ee_body_id][:3]
        ee_target = self.ee_pos_target.numpy()[w]
        self._gui_pos_err = ee_target - ee_pos

        # EE rotation error via Warp quaternion math:
        #   q_err = q_current * inv(q_target); angle = 2 * acos(|q_err.w|).
        q_cur = wp.quat(body_q[ee_body_id][3:7])
        q_tgt = wp.quat(self.ee_rot_target.numpy()[w])
        q_err = q_cur * wp.quat_inverse(q_tgt)
        w_abs = min(1.0, abs(float(q_err[3])))
        self._gui_rot_err = float(wp.degrees(2.0 * wp.acos(w_abs)))

    def _classify_shapes(self):
        """Pre-compute the per-shape category (nut / finger / bolt / table /
        other) and the per-shape world index on the GPU, so penetration
        kernels can classify and route contacts without host-side work."""
        n_shapes = self.model.shape_count
        shape_body_np = self.model.shape_body.numpy()
        body_labels = self.model.body_label
        shape_labels = list(self.model.shape_label)  # may contain None
        shape_world_np = (
            self.model.shape_world.numpy()
            if self.model.shape_world is not None
            else np.full(n_shapes, -1, dtype=np.int32)
        ).astype(np.int32)

        category = np.full(n_shapes, SHAPE_CAT_OTHER, dtype=np.int32)
        for s in range(n_shapes):
            body = int(shape_body_np[s])
            body_label = body_labels[body] if body >= 0 else ""
            shape_label = shape_labels[s] if shape_labels[s] is not None else ""
            combined = f"{body_label}/{shape_label}"

            if "nut" in combined:
                category[s] = SHAPE_CAT_NUT
            elif "finger" in combined:
                category[s] = SHAPE_CAT_FINGER
            elif "bolt" in combined:
                category[s] = SHAPE_CAT_BOLT
            elif "table" in combined:
                category[s] = SHAPE_CAT_TABLE

        # Upload category and world maps to the GPU for the penetration kernels.
        self._shape_category_wp = wp.array(category, dtype=wp.int32)
        self._shape_world_wp = wp.array(shape_world_np, dtype=wp.int32)

    def _pen_result_dict(self, pen_wc3_np):
        """Convert a (world_count, 3) numpy array into the per-category dict
        expected by the per-state tracking loop. Column order matches
        SHAPE_CAT_* pair indices: 0=nut_finger, 1=nut_bolt, 2=nut_table.
        """
        return {
            "nut_finger": pen_wc3_np[:, 0],
            "nut_bolt": pen_wc3_np[:, 1],
            "nut_table": pen_wc3_np[:, 2],
        }

    def _compute_rigid_penetration_per_world(self):
        """Reduce rigid contacts on the GPU into a (world_count, 3) array of
        per-world per-category max penetration depth, then sync once.
        """
        self._pen_rigid_current_wp.zero_()
        wp.launch(
            update_rigid_penetration_kernel,
            dim=self.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.contacts.rigid_contact_normal,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.model.shape_body,
                self._shape_world_wp,
                self._shape_category_wp,
                self.state_0.body_q,
            ],
            outputs=[self._pen_rigid_current_wp],
        )
        return self._pen_result_dict(self._pen_rigid_current_wp.numpy())

    def _compute_hydro_penetration_per_world(self):
        """Reduce hydroelastic face depths on the GPU into a per-world
        per-category max-penetration dict, same layout as the rigid helper.
        """
        empty = np.zeros((self.world_count, 3), dtype=np.float32)
        hydro = self.collision_pipeline.hydroelastic_sdf
        if hydro is None:
            return self._pen_result_dict(empty)
        surface = hydro.get_contact_surface()
        if surface is None:
            return self._pen_result_dict(empty)

        self._pen_hydro_current_wp.zero_()
        wp.launch(
            update_hydro_penetration_kernel,
            dim=surface.contact_surface_depth.shape[0],
            inputs=[
                surface.face_contact_count,
                surface.contact_surface_depth,
                surface.contact_surface_shape_pair,
                self._shape_world_wp,
                self._shape_category_wp,
            ],
            outputs=[self._pen_hydro_current_wp],
        )
        return self._pen_result_dict(self._pen_hydro_current_wp.numpy())

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

        # Slippage + per-state tracking: snapshot Z positions at state entry/exit.
        # task_schedule is static — read once from the cached host copy. body_q is
        # fetched once per tracking call and passed into the penetration helpers.
        tasks_np = self.task_idx.numpy()
        schedule_np = self._task_schedule_np
        body_q = self.state_0.body_q.numpy()

        # Per-world per-category max penetration — reduced on the GPU, synced once each.
        cur_pen_rigid = self._compute_rigid_penetration_per_world()
        cur_pen_hydro = self._compute_hydro_penetration_per_world()

        for w in range(self.world_count):
            task_type = int(schedule_np[int(tasks_np[w])])
            ee_body_id = w * self.num_bodies_per_world + self.ee_index
            nut_body_id = w * self.num_bodies_per_world + self.nut_body_index
            ee_z = float(body_q[ee_body_id][2])
            nut_z = float(body_q[nut_body_id][2])

            # Nut tilt: angle between the nut's body-Z axis and world Z.
            nut_quat = wp.quat(body_q[nut_body_id][3:7])
            z_axis_world = wp.quat_rotate(nut_quat, wp.vec3(0.0, 0.0, 1.0))
            nut_tilt = float(wp.acos(max(-1.0, min(1.0, float(z_axis_world[2])))))

            # Per-state tracking: record start Z on first observation, update end Z
            # and max force/friction while in this state.
            if np.isnan(self._state_nut_z_start[w, task_type]):
                self._state_nut_z_start[w, task_type] = nut_z
                self._state_ee_z_start[w, task_type] = ee_z
            self._state_nut_z_end[w, task_type] = nut_z
            self._state_ee_z_end[w, task_type] = ee_z
            self._state_force_max[w, task_type] = max(self._state_force_max[w, task_type], self._cur_force[w])
            self._state_friction_max[w, task_type] = max(self._state_friction_max[w, task_type], self._cur_friction[w])
            self._state_nut_tilt_max[w, task_type] = max(self._state_nut_tilt_max[w, task_type], nut_tilt)

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
        imgui.separator()
        imgui.text("Pos err:")
        imgui.text(f" x:{err[0]:+.2f}mm {ok_x}|y:{err[1]:+.2f}mm {ok_y}|z:{err[2]:+.2f}mm {ok_z}")
        imgui.text(f" limits: xy<{limit_xy:.2f}mm  z<{limit_z:.2f}mm")

        # Rotation error with threshold check
        ok_r = "*" if self._gui_rot_err < self.rot_threshold_deg else " "
        imgui.separator()
        imgui.text("Rot err:")
        imgui.text(f"{self._gui_rot_err:.2f} deg {ok_r}  limit<{self.rot_threshold_deg:.1f}")
        imgui.separator()

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

        imgui.separator()
        imgui.text("Test zone Z (mm, rel. bolt center):")
        min_mm = self.test_zone_z_min_offset * 1000.0
        max_mm = self.test_zone_z_max_offset * 1000.0
        changed_min, new_min_mm = imgui.slider_float("Z min", min_mm, -30.0, 30.0, "%.1f")
        changed_max, new_max_mm = imgui.slider_float("Z max", max_mm, -30.0, 60.0, "%.1f")
        if changed_min:
            self.test_zone_z_min_offset = new_min_mm / 1000.0
        if changed_max:
            self.test_zone_z_max_offset = new_max_mm / 1000.0
        if changed_min or changed_max:
            self._rebuild_test_zone_lines()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()

        bolt_xy = np.array([float(self.bolt_place_pos[0]), float(self.bolt_place_pos[1])])
        table_top_z = float(self.table_top_center[2])

        # Print stats for all worlds first (before assertions)
        print("\nPer-world report:")
        errors = []
        # Tolerances
        xy_tol = 0.02  # 20mm
        # Nut-z acceptance band. Offsets relative to bolt_center_z come
        # from the assembly defaults and are user-tunable from the ImGui
        # panel.
        bolt_center_z = float(self.bolt_frame_pos[2])
        nut_at_bottom_z_min = bolt_center_z + float(self.test_zone_z_min_offset)
        nut_at_bottom_z_max = bolt_center_z + float(self.test_zone_z_max_offset)
        print(
            f"  bolt_center_z={bolt_center_z:.4f}  "
            f"valid range=[{nut_at_bottom_z_min:.4f}, {nut_at_bottom_z_max:.4f}] "
            f"(offsets min={self.test_zone_z_min_offset * 1000:+.1f}mm  "
            f"max={self.test_zone_z_max_offset * 1000:+.1f}mm)"
        )

        max_tilt_deg = float(self.max_nut_tilt_deg)

        for world_id in range(self.world_count):
            nut_body_id = world_id * self.num_bodies_per_world + self.nut_body_index
            nut_pos = body_q[nut_body_id][:3]
            nut_xy = nut_pos[:2]
            dx = float(nut_xy[0] - bolt_xy[0])
            dy = float(nut_xy[1] - bolt_xy[1])
            xy_error = float(wp.sqrt(dx * dx + dy * dy))
            # Final-frame nut tilt via Warp quaternion rotation: rotate world-Z
            # by the nut quaternion and read the z component.
            nut_quat = wp.quat(body_q[nut_body_id][3:7])
            z_axis_world = wp.quat_rotate(nut_quat, wp.vec3(0.0, 0.0, 1.0))
            nut_tilt_deg = float(wp.degrees(wp.acos(max(-1.0, min(1.0, float(z_axis_world[2]))))))

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
                    f"tilt={nut_tilt_deg:.2f}deg  "
                    f"descent={descent_str}  "
                    f"force_max={f_max:.3f}N  friction_max={fr_max:.3f}N  "
                    f"slip={slip_str}{lift_str}"
                )
            else:
                print(
                    f"  World {world_id}: "
                    f"nut=({nut_pos[0]:.4f}, {nut_pos[1]:.4f}, {nut_pos[2]:.4f})  "
                    f"xy_err={xy_error * 1000:.1f}mm  "
                    f"tilt={nut_tilt_deg:.2f}deg"
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
                    f"(bolt_center={bolt_center_z:.4f})"
                )
            # Phase 4: nut body-Z axis must stay close to world Z (upright).
            # A threaded nut that ends up tipped means the screw motion got
            # weird — e.g. stepped off the threads or rocked sideways.
            if nut_tilt_deg > max_tilt_deg:
                errors.append(
                    f"World {world_id}: Nut tilt {nut_tilt_deg:.2f}deg exceeds "
                    f"limit {max_tilt_deg:.2f}deg (body-Z drifted from world-Z)."
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
            f"{'nut_dz':>10}  {'slip':>10}  {'tilt':>8}  "
            f"{'rigid pen_max [mm]':^34}  "
            f"{'hydro pen_max [mm]':^34}"
        )
        header_line2 = (
            f"  {'':<16}{'':>6}  "
            f"{'[N]':>8}  {'[N]':>9}  "
            f"{'[mm]':>10}  {'[mm]':>10}  {'[deg]':>8}  "
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
                tilt_deg = float(wp.degrees(float(self._state_nut_tilt_max[world_id, s])))
                print(
                    f"  {state.name:<16}{world_id:>6}  "
                    f"{f_max:>8.3f}  {fr_max:>9.3f}  "
                    f"{nut_dz * 1000:>+10.2f}  {slip_dz * 1000:>+10.2f}  {tilt_deg:>8.2f}  "
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
        parser.set_defaults(num_frames=3000)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--assembly",
            type=str,
            default="m20_loose",
            choices=list(ASSEMBLIES.keys()),
            help="Nut-bolt assembly type. m20_loose (default, original) or m16_loose (matches Isaac Lab Factory).",
        )
        parser.add_argument(
            "--grasp-margin",
            type=float,
            default=None,
            help="Extra gripper closure past first contact with the nut [m]. Default depends on assembly.",
        )
        parser.add_argument(
            "--nut-xy-jitter",
            type=float,
            default=0.03,
            help="Half-width of the uniform XY sampling region for the nut's initial position [m].",
        )
        parser.add_argument(
            "--screw-cycles",
            type=int,
            default=40,
            help="Number of (rotate, regrip) screw cycles inserted between REFINE_PLACE and RELEASE (0 disables screwing).",
        )
        parser.add_argument(
            "--screw-angle-deg",
            type=float,
            default=120.0,
            help="Yaw step per screw cycle [degrees]. Matches devel-example-nut-bolt reference (π/1.5).",
        )
        parser.add_argument(
            "--screw-regrip-z-offset",
            type=float,
            default=0.001,
            help="Z offset applied during SCREW_REGRIP to disengage the fingers [m].",
        )
        parser.add_argument(
            "--screw-regrip-clearance",
            type=float,
            default=None,
            help="Radial clearance over the hex across-corners radius for the SCREW_REGRIP finger target [m]. Default depends on assembly.",
        )
        parser.add_argument(
            "--screw-grip-margin",
            type=float,
            default=None,
            help="Grasp margin used during SCREW_ROTATE [m]. Default depends on assembly.",
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
            "--nut-bolt-mu",
            type=float,
            default=0.2,
            help="Friction coefficient for nut and bolt shapes (matches devel-example-nut-bolt reference). Combined as max(mu_a, mu_b) — finger grip unaffected.",
        )
        parser.add_argument(
            "--max-nut-tilt-deg",
            type=float,
            default=2.0,
            help="Fail test_final if the nut's body-Z axis is tilted more than this many degrees from world Z at the end of the simulation. Catches runs where the nut stepped off the threads and ended up rocked.",
        )
        parser.add_argument(
            "--show-isosurface",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Compute and render the hydroelastic contact surface (isosurface). Off by default — the triangulation runs every step and is the single most expensive optional feature. --test forces it on for hydro penetration tracking.",
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
