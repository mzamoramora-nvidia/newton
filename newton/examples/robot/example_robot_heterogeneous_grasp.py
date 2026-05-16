# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Heterogeneous Grasp
#
# Demonstrates heterogeneous grasping environments: each world contains a
# different object (shape, mass, size), all grasped and lifted by a Franka
# Panda + Robotiq 2F-85 gripper. Supports 4 collision pipelines.
#
# Command: python -m newton.examples robot_heterogeneous_grasp
#
###########################################################################

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from enum import IntEnum

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
import newton.viewer
from newton import Contacts
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact


_NUT_BOLT_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
_NUT_BOLT_ASSEMBLY = "m20_loose"

_LEGO_PITCH = 0.008
_LEGO_BRICK_HEIGHT = 0.0096
_LEGO_STUD_RADIUS = 0.0024
_LEGO_STUD_HEIGHT = 0.0017
_LEGO_WALL_THICKNESS = 0.0012
_LEGO_TOP_THICKNESS = 0.001
_LEGO_TUBE_OUTER_RADIUS = 0.003255
_LEGO_CYLINDER_SEGMENTS = 24


class ObjectShape(IntEnum):
    BOX = 0
    SPHERE = 1
    CYLINDER = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CUP = 5
    RUBBER_DUCK = 6
    LEGO_BRICK = 7
    RJ45_PLUG = 8
    BEAR = 9
    NUT = 10
    BOLT = 11


SHAPE_NAMES = [s.name for s in ObjectShape]
NUM_SHAPES = len(ObjectShape)

_MESH_SHAPES = frozenset({
    ObjectShape.CUP, ObjectShape.RUBBER_DUCK, ObjectShape.LEGO_BRICK,
    ObjectShape.RJ45_PLUG, ObjectShape.BEAR, ObjectShape.NUT, ObjectShape.BOLT,
})


class CollisionMode(IntEnum):
    MUJOCO = 0
    NEWTON_DEFAULT = 1
    NEWTON_SDF = 2
    NEWTON_HYDROELASTIC = 3


class TaskType(IntEnum):
    APPROACH = 0
    CLOSE_GRIPPER = 1
    SETTLE = 2  # hold pose + full closure; lets contact forces stabilize before LIFT
    LIFT = 3
    HOLD = 4
    DONE = 5


NUM_TASKS = len(TaskType)
TASK_NAMES = [t.name for t in TaskType]


@wp.func
def s_curve_profile(t: float, T: float, ramp_fraction: float) -> float:
    """S-curve trapezoidal position profile, normalized to [0, 1]."""
    t = wp.clamp(t, 0.0, T)
    f = wp.clamp(ramp_fraction, 0.01, 0.5)
    t_r = f * T
    v_max = 1.0 / (T - t_r)

    if t < t_r:
        return v_max * (t * 0.5 - t_r / (2.0 * wp.pi) * wp.sin(wp.pi * t / t_r))
    elif t < T - t_r:
        p1_end = v_max * t_r * 0.5
        return p1_end + v_max * (t - t_r)
    else:
        t_decel = t - (T - t_r)
        p12_end = v_max * t_r * 0.5 + v_max * (T - 2.0 * t_r)
        return p12_end + v_max * (t_decel * 0.5 + t_r / (2.0 * wp.pi) * wp.sin(wp.pi * t_decel / t_r))


def _patch_v4_mjcf(mjcf_path: str) -> str:
    """Patch V4 MJCF: fix base collision geom transform and pad masses."""
    tree = ET.parse(mjcf_path)

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

    for body in tree.iter("body"):
        if body.get("name") in ("left_pad", "right_pad"):
            for geom in body.findall("geom"):
                geom.set("mass", "0.00175")

    patched_path = mjcf_path.replace(".xml", "_patched.xml")
    tree.write(patched_path)
    return patched_path


def _load_mesh_asset_no_sdf(asset_path, prim_path):
    """Load a USD mesh asset with scale baked in. SDF built later per-world."""
    stage = Usd.Stage.Open(str(asset_path / "model.usda"))
    prim = stage.GetPrimAtPath(prim_path)
    mesh = newton.usd.get_mesh(prim, load_normals=True, face_varying_normal_conversion="vertex_splitting")
    parent_prim = stage.GetPrimAtPath("/root/Model")
    scale = np.asarray(newton.usd.get_scale(parent_prim), dtype=np.float32)
    if not np.allclose(scale, 1.0):
        mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
    return mesh


# ---- LEGO brick mesh generation ----


def _lego_cylinder_mesh(radius, height, segments, cx=0.0, cy=0.0, cz=0.0, bottom_cap=True):
    n = segments
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    ring_x = cx + radius * cos_a
    ring_y = cy + radius * sin_a
    verts = []
    faces = []
    side_bot = np.column_stack([ring_x, ring_y, np.full(n, cz)]).astype(np.float32)
    side_top = np.column_stack([ring_x, ring_y, np.full(n, cz + height)]).astype(np.float32)
    verts.append(side_bot)
    verts.append(side_top)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + j, n + i])
        faces.append([i, j, n + j])
    off_top = 2 * n
    cap_top_ring = np.column_stack([ring_x, ring_y, np.full(n, cz + height)]).astype(np.float32)
    cap_top_center = np.array([[cx, cy, cz + height]], dtype=np.float32)
    verts.append(cap_top_ring)
    verts.append(cap_top_center)
    tc = off_top + n
    for i in range(n):
        j = (i + 1) % n
        faces.append([tc, off_top + i, off_top + j])
    if bottom_cap:
        off_bot = off_top + n + 1
        cap_bot_ring = np.column_stack([ring_x, ring_y, np.full(n, cz)]).astype(np.float32)
        cap_bot_center = np.array([[cx, cy, cz]], dtype=np.float32)
        verts.append(cap_bot_ring)
        verts.append(cap_bot_center)
        bc = off_bot + n
        for i in range(n):
            j = (i + 1) % n
            faces.append([bc, off_bot + j, off_bot + i])
    return np.vstack(verts), np.array(faces, dtype=np.int32)


def _lego_combine_meshes(mesh_list):
    all_v, all_f, off = [], [], 0
    for v, f in mesh_list:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    return np.vstack(all_v).astype(np.float32), np.vstack(all_f).astype(np.int32)


def _lego_make_shell_mesh(nx, ny):
    ox = nx * _LEGO_PITCH / 2.0
    oy = ny * _LEGO_PITCH / 2.0
    inx = ox - _LEGO_WALL_THICKNESS
    iny = oy - _LEGO_WALL_THICKNESS
    H = _LEGO_BRICK_HEIGHT
    T = _LEGO_TOP_THICKNESS
    v = np.array(
        [
            [-ox, -oy, 0], [+ox, -oy, 0], [+ox, +oy, 0], [-ox, +oy, 0],
            [-ox, -oy, H], [+ox, -oy, H], [+ox, +oy, H], [-ox, +oy, H],
            [-inx, -iny, 0], [+inx, -iny, 0], [+inx, +iny, 0], [-inx, +iny, 0],
            [-inx, -iny, H - T], [+inx, -iny, H - T], [+inx, +iny, H - T], [-inx, +iny, H - T],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
            [1, 2, 6], [1, 6, 5],
            [0, 8, 9], [0, 9, 1],
            [1, 9, 10], [1, 10, 2],
            [2, 10, 11], [2, 11, 3],
            [3, 11, 8], [3, 8, 0],
            [9, 8, 12], [9, 12, 13],
            [11, 10, 14], [11, 14, 15],
            [8, 11, 15], [8, 15, 12],
            [10, 9, 13], [10, 13, 14],
            [12, 15, 14], [12, 14, 13],
        ],
        dtype=np.int32,
    )
    return v, f


def _lego_make_brick_mesh(nx=4, ny=2):
    shell_v, shell_f = _lego_make_shell_mesh(nx, ny)
    seg = _LEGO_CYLINDER_SEGMENTS
    stud_meshes = []
    for i in range(nx):
        for j in range(ny):
            sx = (i - (nx - 1) / 2.0) * _LEGO_PITCH
            sy = (j - (ny - 1) / 2.0) * _LEGO_PITCH
            stud_meshes.append(
                _lego_cylinder_mesh(
                    _LEGO_STUD_RADIUS, _LEGO_STUD_HEIGHT, seg,
                    cx=sx, cy=sy, cz=_LEGO_BRICK_HEIGHT, bottom_cap=False,
                )
            )
    tube_meshes = []
    if ny == 2:
        tube_height = _LEGO_BRICK_HEIGHT - _LEGO_TOP_THICKNESS
        for i in range(nx - 1):
            tx = (i - (nx - 2) / 2.0) * _LEGO_PITCH
            tube_meshes.append(
                _lego_cylinder_mesh(_LEGO_TUBE_OUTER_RADIUS, tube_height, seg, cx=tx, cy=0.0, cz=0.0)
            )
    v, f = _lego_combine_meshes([(shell_v, shell_f), *stud_meshes, *tube_meshes])
    center = (v.min(axis=0) + v.max(axis=0)) / 2.0
    v -= center
    return newton.Mesh(v, f.flatten())


# ---- Mesh loading helpers for local USD and external OBJ assets ----


def _load_usd_mesh_local(usd_path, prim_path):
    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(prim_path)
    mesh = newton.usd.get_mesh(prim, load_normals=False)
    verts = mesh.vertices
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    if not np.allclose(center, 0.0, atol=1e-4):
        mesh = mesh.copy(vertices=verts - center, recompute_inertia=True)
    return mesh


def _load_obj_mesh_trimesh(obj_path):
    import trimesh  # noqa: PLC0415

    tm = trimesh.load(obj_path, force="mesh")
    vertices = np.array(tm.vertices, dtype=np.float32)
    indices = np.array(tm.faces.flatten(), dtype=np.int32)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices -= center
    return newton.Mesh(vertices, indices)


# ---- Warp kernels for IK + state machine ----


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_idx: wp.array[wp.int32],
    task_timer: wp.array[wp.float32],
    task_durations: wp.array[wp.float32],
    grasp_z: wp.array[wp.float32],
    lift_z: wp.array[wp.float32],
    grasp_ctrl: wp.array[wp.float32],
    object_xy: wp.vec2,
    ee_rot_down: wp.vec4,
    task_init_body_q: wp.array[wp.transform],
    ee_body_global_indices: wp.array[wp.int32],
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_pos_target_interp: wp.array[wp.vec3],
    ee_rot_target_interp: wp.array[wp.vec4],
    gripper_target: wp.array[wp.float32],
):
    """Compute IK target pose with interpolation from task start pose."""
    tid = wp.tid()
    state = task_idx[tid]
    timer = task_timer[tid]

    # Desired rotation: keep initial EE orientation (already pointing down)
    rot_down = ee_rot_down

    gz = grasp_z[tid]
    lz = lift_z[tid]
    gc = grasp_ctrl[tid]
    ox = object_xy[0]
    oy = object_xy[1]

    target_z = gz
    ctrl = 0.0

    if state == wp.static(int(TaskType.APPROACH)):
        target_z = gz
        ctrl = 0.0
    elif state == wp.static(int(TaskType.CLOSE_GRIPPER)):
        target_z = gz
        # Gradually close gripper over the CLOSE_GRIPPER duration
        dur_close = task_durations[wp.static(int(TaskType.CLOSE_GRIPPER))]
        alpha = wp.clamp(timer / dur_close, 0.0, 1.0)
        ctrl = alpha * gc
    elif state == wp.static(int(TaskType.SETTLE)):
        target_z = gz
        ctrl = gc
    elif state == wp.static(int(TaskType.LIFT)):
        target_z = lz
        ctrl = gc
    elif state == wp.static(int(TaskType.HOLD)):
        target_z = lz
        ctrl = gc
    else:
        target_z = lz
        ctrl = gc

    target_pos = wp.vec3(ox, oy, target_z)
    ee_pos_target[tid] = target_pos
    ee_rot_target[tid] = rot_down
    gripper_target[tid] = ctrl

    # Interpolate from task-start TCP pose to target pose.
    # ee_pos_prev is the Robotiq base body position; we compute the TCP position
    # from it so interpolation is in the same space as the IK target (TCP frame).
    dur = task_durations[state] if state < wp.static(int(TaskType.DONE)) else 1.0
    t = wp.clamp(timer / dur, 0.0, 1.0)

    ee_body_idx = ee_body_global_indices[tid]
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_idx])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_idx])
    tcp_offset_local = wp.vec3(0.0, 0.0, 0.174)
    tcp_pos_prev = ee_pos_prev + wp.quat_rotate(ee_quat_prev, tcp_offset_local)

    ee_pos_target_interp[tid] = tcp_pos_prev * (1.0 - t) + target_pos * t

    ee_quat_target = wp.quaternion(rot_down[0], rot_down[1], rot_down[2], rot_down[3])
    ee_quat_interp = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)
    ee_rot_target_interp[tid] = wp.vec4(ee_quat_interp[0], ee_quat_interp[1], ee_quat_interp[2], ee_quat_interp[3])


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_idx: wp.array[wp.int32],
    task_timer: wp.array[wp.float32],
    task_durations: wp.array[wp.float32],
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_actual: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    task_init_body_q: wp.array[wp.transform],
    num_bodies: int,
    frame_dt: float,
):
    """Advance the per-world state machine. On transition, snapshot body_q."""
    tid = wp.tid()
    state = task_idx[tid]

    if state >= wp.static(int(TaskType.DONE)):
        return

    task_timer[tid] = task_timer[tid] + frame_dt

    dur = task_durations[state]
    timer = task_timer[tid]

    if timer < dur:
        return

    # For APPROACH and LIFT, also require EE position settling
    if state == wp.static(int(TaskType.APPROACH)) or state == wp.static(int(TaskType.LIFT)):
        target = ee_pos_target[tid]
        actual = ee_pos_actual[tid]
        err_x = wp.abs(target[0] - actual[0])
        err_y = wp.abs(target[1] - actual[1])
        err_z = wp.abs(target[2] - actual[2])
        settled = err_x < 0.001 and err_y < 0.001 and err_z < 0.00075
        if not settled:
            return

    # Advance to next state
    task_idx[tid] = state + 1
    task_timer[tid] = 0.0

    # Snapshot current body_q as start pose for the next task's interpolation
    body_start = tid * num_bodies
    for i in range(num_bodies):
        task_init_body_q[body_start + i] = body_q[body_start + i]


@wp.kernel(enable_backward=False)
def extract_ee_pos_kernel(
    body_q: wp.array[wp.transform],
    ee_body_global_indices: wp.array[wp.int32],
    # output
    ee_pos_actual: wp.array[wp.vec3],
):
    """Extract world-frame TCP position per world from body_q."""
    tid = wp.tid()
    body_idx = ee_body_global_indices[tid]
    ee_pos = wp.transform_get_translation(body_q[body_idx])
    ee_quat = wp.transform_get_rotation(body_q[body_idx])
    tcp_offset_local = wp.vec3(0.0, 0.0, 0.174)
    ee_pos_actual[tid] = ee_pos + wp.quat_rotate(ee_quat, tcp_offset_local)


@wp.kernel(enable_backward=False)
def reset_cur_penetration_kernel(cur_penetration: wp.array[wp.float32]):
    tid = wp.tid()
    cur_penetration[tid] = 0.0


@wp.kernel(enable_backward=False)
def update_penetration_kernel(
    contact_dist: wp.array[wp.float32],  # mjw_data.contact.dist, shape (naconmax,)
    contact_worldid: wp.array[wp.int32],  # mjw_data.contact.worldid, shape (naconmax,)
    nacon: wp.array[wp.int32],  # mjw_data.nacon, shape (1,)
    world_count: wp.int32,
    cur_penetration: wp.array[wp.float32],  # shape (wc,)
    max_penetration: wp.array[wp.float32],  # shape (wc,)
):
    tid = wp.tid()
    if tid >= nacon[0]:
        return
    w = contact_worldid[tid]
    if w < 0 or w >= world_count:
        return
    pen = wp.max(-contact_dist[tid], 0.0)
    wp.atomic_max(cur_penetration, w, pen)
    wp.atomic_max(max_penetration, w, pen)


@wp.kernel(enable_backward=False)
def update_metrics_kernel(
    pad_force_matrix: wp.array2d[wp.vec3],
    pad_force_matrix_friction: wp.array2d[wp.vec3],
    table_force_matrix: wp.array2d[wp.vec3],
    table_force_matrix_friction: wp.array2d[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    task_idx: wp.array[wp.int32],
    prev_task_idx: wp.array[wp.int32],
    num_states: wp.int32,
    hold_state: wp.int32,
    frame_count: wp.int32,
    cur_pad_force: wp.array[wp.float32],
    cur_pad_friction: wp.array[wp.float32],
    cur_table_force: wp.array[wp.float32],
    cur_table_friction: wp.array[wp.float32],
    max_pad_force: wp.array[wp.float32],
    max_pad_friction: wp.array[wp.float32],
    max_table_force: wp.array[wp.float32],
    max_table_friction: wp.array[wp.float32],
    max_object_z: wp.array[wp.float32],
    object_z_at_hold_start: wp.array[wp.float32],
    object_z_at_hold_end: wp.array[wp.float32],
    object_vel_sum: wp.array[wp.float32],
    object_vel_count: wp.array[wp.int32],
    world_nan_frame: wp.array[wp.int32],
    state_pad_force_sum: wp.array2d[wp.float32],
    state_pad_force_max: wp.array2d[wp.float32],
    state_table_force_sum: wp.array2d[wp.float32],
    state_table_force_max: wp.array2d[wp.float32],
    state_pen_sum: wp.array2d[wp.float32],
    state_pen_max: wp.array2d[wp.float32],
    state_vel_sum: wp.array2d[wp.float32],
    state_count: wp.array2d[wp.int32],
    cur_penetration: wp.array[wp.float32],
):
    w = wp.tid()
    n_counterparts = pad_force_matrix.shape[1]

    # Pad force + friction (sum over counterparts, then magnitude)
    pad_f_sum = wp.vec3(0.0, 0.0, 0.0)
    pad_fr_sum = wp.vec3(0.0, 0.0, 0.0)
    for c in range(n_counterparts):
        pad_f_sum = pad_f_sum + pad_force_matrix[w, c]
        pad_fr_sum = pad_fr_sum + pad_force_matrix_friction[w, c]
    pf = wp.length(pad_f_sum)
    pfr = wp.length(pad_fr_sum)

    # Table force + friction
    n_table_counterparts = table_force_matrix.shape[1]
    tbl_f_sum = wp.vec3(0.0, 0.0, 0.0)
    tbl_fr_sum = wp.vec3(0.0, 0.0, 0.0)
    for c in range(n_table_counterparts):
        tbl_f_sum = tbl_f_sum + table_force_matrix[w, c]
        tbl_fr_sum = tbl_fr_sum + table_force_matrix_friction[w, c]
    tf = wp.length(tbl_f_sum)
    tfr = wp.length(tbl_fr_sum)

    cur_pad_force[w] = pf
    cur_pad_friction[w] = pfr
    cur_table_force[w] = tf
    cur_table_friction[w] = tfr
    max_pad_force[w] = wp.max(max_pad_force[w], pf)
    max_pad_friction[w] = wp.max(max_pad_friction[w], pfr)
    max_table_force[w] = wp.max(max_table_force[w], tf)
    max_table_friction[w] = wp.max(max_table_friction[w], tfr)

    # Object pose + velocity
    obj_global = body_world_start[w] + object_body_offset
    obj_q = body_q[obj_global]
    obj_pos = wp.transform_get_translation(obj_q)
    obj_z = obj_pos[2]
    obj_vel = wp.spatial_bottom(body_qd[obj_global])
    vel_mag = wp.length(obj_vel)

    # NaN detection
    obj_z_is_nan = wp.isnan(obj_z)
    if obj_z_is_nan and world_nan_frame[w] < 0:
        world_nan_frame[w] = frame_count

    if not obj_z_is_nan:
        max_object_z[w] = wp.max(max_object_z[w], obj_z)

    vel_is_nan = wp.isnan(vel_mag)
    if not vel_is_nan:
        object_vel_sum[w] = object_vel_sum[w] + vel_mag
        object_vel_count[w] = object_vel_count[w] + 1

    # State-bucketed accumulation
    cur_task = task_idx[w]
    prev_task = prev_task_idx[w]
    if cur_task < num_states:
        t = cur_task
        pen_mm = cur_penetration[w] * 1000.0
        state_pad_force_sum[w, t] = state_pad_force_sum[w, t] + pf
        state_pad_force_max[w, t] = wp.max(state_pad_force_max[w, t], pf)
        state_table_force_sum[w, t] = state_table_force_sum[w, t] + tf
        state_table_force_max[w, t] = wp.max(state_table_force_max[w, t], tf)
        state_pen_sum[w, t] = state_pen_sum[w, t] + pen_mm
        state_pen_max[w, t] = wp.max(state_pen_max[w, t], pen_mm)
        if not vel_is_nan:
            state_vel_sum[w, t] = state_vel_sum[w, t] + vel_mag * 1000.0
        state_count[w, t] = state_count[w, t] + 1

    # HOLD transition
    if cur_task == hold_state:
        if prev_task != hold_state:
            object_z_at_hold_start[w] = obj_z
        object_z_at_hold_end[w] = obj_z


@wp.kernel(enable_backward=False)
def copy_prev_task_kernel(
    task_idx: wp.array[wp.int32],
    prev_task_idx: wp.array[wp.int32],
):
    tid = wp.tid()
    prev_task_idx[tid] = task_idx[tid]


# GUI staging buffer indices (keep in sync with stage_gui_metrics_kernel below).
_GUI_STAGE_FIELDS = [
    "cur_pad_force",
    "max_pad_force",
    "cur_pad_friction",
    "max_pad_friction",
    "cur_table_force",
    "max_table_force",
    "cur_table_friction",
    "max_table_friction",
    "cur_penetration",
    "max_penetration",
    "avg_vel",
    "object_z",
    "object_initial_z",
    "max_object_z",
    "task_idx",  # stored as float for uniform array
    "world_nan_frame",  # -1 sentinel -> NaN semantics preserved via float cast
    "task_timer",  # task_timer[selected_world]
    "task_dur",  # task_durations[cur_task], or 0 if cur_task >= num_task_durations
    "ee_target_x",
    "ee_target_y",
    "ee_target_z",
    "ee_actual_x",
    "ee_actual_y",
    "ee_actual_z",
]
_GUI_STAGE_SIZE = len(_GUI_STAGE_FIELDS)


@wp.kernel(enable_backward=False)
def stage_gui_metrics_kernel(
    selected_world: wp.int32,
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    cur_pad_force: wp.array[wp.float32],
    max_pad_force: wp.array[wp.float32],
    cur_pad_friction: wp.array[wp.float32],
    max_pad_friction: wp.array[wp.float32],
    cur_table_force: wp.array[wp.float32],
    max_table_force: wp.array[wp.float32],
    cur_table_friction: wp.array[wp.float32],
    max_table_friction: wp.array[wp.float32],
    cur_penetration: wp.array[wp.float32],
    max_penetration: wp.array[wp.float32],
    object_vel_sum: wp.array[wp.float32],
    object_vel_count: wp.array[wp.int32],
    object_initial_z: wp.array[wp.float32],
    max_object_z: wp.array[wp.float32],
    task_idx: wp.array[wp.int32],
    world_nan_frame: wp.array[wp.int32],
    task_timer: wp.array[wp.float32],
    task_durations: wp.array[wp.float32],
    num_task_durations: wp.int32,
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_actual: wp.array[wp.vec3],
    out: wp.array[wp.float32],
):
    # dim=1 -- single-thread pack.
    # Output indices must match _GUI_STAGE_FIELDS order.
    w = selected_world
    obj_global = body_world_start[w] + object_body_offset
    obj_pos = wp.transform_get_translation(body_q[obj_global])
    out[0] = cur_pad_force[w]
    out[1] = max_pad_force[w]
    out[2] = cur_pad_friction[w]
    out[3] = max_pad_friction[w]
    out[4] = cur_table_force[w]
    out[5] = max_table_force[w]
    out[6] = cur_table_friction[w]
    out[7] = max_table_friction[w]
    out[8] = cur_penetration[w]
    out[9] = max_penetration[w]
    vel_count = object_vel_count[w]
    if vel_count > 0:
        out[10] = object_vel_sum[w] / wp.float32(vel_count)
    else:
        out[10] = 0.0
    out[11] = obj_pos[2]
    out[12] = object_initial_z[w]
    out[13] = max_object_z[w]
    out[14] = wp.float32(task_idx[w])
    # world_nan_frame is an int32 frame index; float32 exactly represents
    # integers up to 2**24 (~16.7M frames), well beyond any realistic episode.
    out[15] = wp.float32(world_nan_frame[w])
    out[16] = task_timer[w]
    cur_task = task_idx[w]
    # Guard: task_idx can be >= num_task_durations (DONE state); pack 0.0 then.
    if cur_task < num_task_durations:
        out[17] = task_durations[cur_task]
    else:
        out[17] = 0.0
    ee_t = ee_pos_target[w]
    ee_a = ee_pos_actual[w]
    out[18] = ee_t[0]
    out[19] = ee_t[1]
    out[20] = ee_t[2]
    out[21] = ee_a[0]
    out[22] = ee_a[1]
    out[23] = ee_a[2]


@wp.kernel(enable_backward=False)
def compute_debug_frame_lines_kernel(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    world_offsets: wp.array[wp.vec3],
    object_body_offset: wp.int32,
    ee_body_offset: wp.int32,
    tcp_offset: wp.float32,
    axis_len: wp.float32,
    # Outputs: flat arrays of length wc * 9.
    begins: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    w = wp.tid()
    base = w * 9
    wo = world_offsets[w]

    # Object frame (bright RGB).
    obj_global = body_world_start[w] + object_body_offset
    obj_pos = wp.transform_get_translation(body_q[obj_global]) + wo
    for axis in range(3):
        begins[base + axis] = obj_pos
        if axis == 0:
            ends[base + axis] = obj_pos + wp.vec3(axis_len, 0.0, 0.0)
            colors[base + axis] = wp.vec3(1.0, 0.0, 0.0)
        elif axis == 1:
            ends[base + axis] = obj_pos + wp.vec3(0.0, axis_len, 0.0)
            colors[base + axis] = wp.vec3(0.0, 1.0, 0.0)
        else:
            ends[base + axis] = obj_pos + wp.vec3(0.0, 0.0, axis_len)
            colors[base + axis] = wp.vec3(0.0, 0.0, 1.0)

    # EE base frame (muted RGB).
    ee_global = body_world_start[w] + ee_body_offset
    ee_tf = body_q[ee_global]
    ee_pos = wp.transform_get_translation(ee_tf) + wo
    ee_quat = wp.transform_get_rotation(ee_tf)

    for axis in range(3):
        idx = base + 3 + axis
        begins[idx] = ee_pos
        if axis == 0:
            ends[idx] = ee_pos + wp.vec3(axis_len, 0.0, 0.0)
            colors[idx] = wp.vec3(0.8, 0.4, 0.4)
        elif axis == 1:
            ends[idx] = ee_pos + wp.vec3(0.0, axis_len, 0.0)
            colors[idx] = wp.vec3(0.4, 0.8, 0.4)
        else:
            ends[idx] = ee_pos + wp.vec3(0.0, 0.0, axis_len)
            colors[idx] = wp.vec3(0.4, 0.4, 0.8)

    # TCP frame (saturated RGB) -- offset along EE local Z by tcp_offset.
    tcp_pos = ee_pos + wp.quat_rotate(ee_quat, wp.vec3(0.0, 0.0, tcp_offset))
    for axis in range(3):
        idx = base + 6 + axis
        begins[idx] = tcp_pos
        if axis == 0:
            ends[idx] = tcp_pos + wp.vec3(axis_len * 0.5, 0.0, 0.0)
            colors[idx] = wp.vec3(1.0, 0.4, 0.4)
        elif axis == 1:
            ends[idx] = tcp_pos + wp.vec3(0.0, axis_len * 0.5, 0.0)
            colors[idx] = wp.vec3(0.4, 1.0, 0.4)
        else:
            ends[idx] = tcp_pos + wp.vec3(0.0, 0.0, axis_len * 0.5)
            colors[idx] = wp.vec3(0.4, 0.4, 1.0)


@wp.kernel(enable_backward=False)
def _seed_object_initial_z_kernel(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    object_initial_z: wp.array[wp.float32],
):
    w = wp.tid()
    obj_global = body_world_start[w] + object_body_offset
    object_initial_z[w] = wp.transform_get_translation(body_q[obj_global])[2]


class GraspMetrics:
    """GPU-resident per-world + per-state metric accumulators for the grasp example.

    All state lives in Warp arrays. ``update()`` launches Warp kernels only --
    no Python-level branches on GPU data, no ``.numpy()`` calls. CPU readbacks
    happen only in ``stage_gui()`` (commit 2) and ``readback_all()`` (commit 3).
    """

    def __init__(self, world_count: int, num_states: int, hold_state: int, object_body_offset: int):
        wc = world_count
        self.world_count = wc
        self.num_states = num_states
        self.hold_state = hold_state
        self.object_body_offset = object_body_offset

        # Per-world running scalars
        self.cur_pad_force = wp.zeros(wc, dtype=wp.float32)
        self.cur_pad_friction = wp.zeros(wc, dtype=wp.float32)
        self.cur_table_force = wp.zeros(wc, dtype=wp.float32)
        self.cur_table_friction = wp.zeros(wc, dtype=wp.float32)
        self.cur_penetration = wp.zeros(wc, dtype=wp.float32)

        self.max_pad_force = wp.zeros(wc, dtype=wp.float32)
        self.max_pad_friction = wp.zeros(wc, dtype=wp.float32)
        self.max_table_force = wp.zeros(wc, dtype=wp.float32)
        self.max_table_friction = wp.zeros(wc, dtype=wp.float32)
        self.max_penetration = wp.zeros(wc, dtype=wp.float32)

        self.max_object_z = wp.full(wc, -float("inf"), dtype=wp.float32)
        self.object_initial_z = wp.zeros(wc, dtype=wp.float32)
        self.object_z_at_hold_start = wp.full(wc, float("nan"), dtype=wp.float32)
        self.object_z_at_hold_end = wp.full(wc, float("nan"), dtype=wp.float32)
        self.object_vel_sum = wp.zeros(wc, dtype=wp.float32)
        self.object_vel_count = wp.zeros(wc, dtype=wp.int32)

        self.world_nan_frame = wp.full(wc, -1, dtype=wp.int32)
        self.prev_task_idx = wp.zeros(wc, dtype=wp.int32)

        # GUI staging (pre-allocated; populated on demand).
        self.gui_staging = wp.zeros(_GUI_STAGE_SIZE, dtype=wp.float32)

        nt = num_states
        self.state_pad_force_sum = wp.zeros((wc, nt), dtype=wp.float32)
        self.state_pad_force_max = wp.zeros((wc, nt), dtype=wp.float32)
        self.state_table_force_sum = wp.zeros((wc, nt), dtype=wp.float32)
        self.state_table_force_max = wp.zeros((wc, nt), dtype=wp.float32)
        self.state_pen_sum = wp.zeros((wc, nt), dtype=wp.float32)
        self.state_pen_max = wp.zeros((wc, nt), dtype=wp.float32)
        self.state_vel_sum = wp.zeros((wc, nt), dtype=wp.float32)
        self.state_count = wp.zeros((wc, nt), dtype=wp.int32)

    def capture_initial_object_z(self, state_0, body_world_start_array):
        wp.launch(
            _seed_object_initial_z_kernel,
            dim=self.world_count,
            inputs=[state_0.body_q, body_world_start_array, self.object_body_offset],
            outputs=[self.object_initial_z],
        )

    def update(
        self,
        state_0,
        pad_sensor,
        table_sensor,
        mjw_data,
        task_idx,
        body_world_start_array,
        frame_count: int,
    ):
        wc = self.world_count

        wp.launch(reset_cur_penetration_kernel, dim=wc, inputs=[], outputs=[self.cur_penetration])
        naconmax = mjw_data.naconmax
        if naconmax > 0:
            wp.launch(
                update_penetration_kernel,
                dim=naconmax,
                inputs=[
                    mjw_data.contact.dist,
                    mjw_data.contact.worldid,
                    mjw_data.nacon,
                    wc,
                ],
                outputs=[self.cur_penetration, self.max_penetration],
            )

        wp.launch(
            update_metrics_kernel,
            dim=wc,
            inputs=[
                pad_sensor.force_matrix,
                pad_sensor.force_matrix_friction,
                table_sensor.force_matrix,
                table_sensor.force_matrix_friction,
                state_0.body_q,
                state_0.body_qd,
                body_world_start_array,
                self.object_body_offset,
                task_idx,
                self.prev_task_idx,
                self.num_states,
                self.hold_state,
                frame_count,
            ],
            outputs=[
                self.cur_pad_force,
                self.cur_pad_friction,
                self.cur_table_force,
                self.cur_table_friction,
                self.max_pad_force,
                self.max_pad_friction,
                self.max_table_force,
                self.max_table_friction,
                self.max_object_z,
                self.object_z_at_hold_start,
                self.object_z_at_hold_end,
                self.object_vel_sum,
                self.object_vel_count,
                self.world_nan_frame,
                self.state_pad_force_sum,
                self.state_pad_force_max,
                self.state_table_force_sum,
                self.state_table_force_max,
                self.state_pen_sum,
                self.state_pen_max,
                self.state_vel_sum,
                self.state_count,
                self.cur_penetration,
            ],
        )

        wp.launch(copy_prev_task_kernel, dim=wc, inputs=[task_idx], outputs=[self.prev_task_idx])

    def stage_gui(
        self,
        selected_world: int,
        state_0,
        body_world_start_array,
        task_idx,
        task_timer,
        task_durations,
        ee_pos_target,
        ee_pos_actual,
    ):
        """Pack selected-world metrics into a small staging buffer. Returns numpy view."""
        wp.launch(
            stage_gui_metrics_kernel,
            dim=1,
            inputs=[
                int(selected_world),
                state_0.body_q,
                body_world_start_array,
                self.object_body_offset,
                self.cur_pad_force,
                self.max_pad_force,
                self.cur_pad_friction,
                self.max_pad_friction,
                self.cur_table_force,
                self.max_table_force,
                self.cur_table_friction,
                self.max_table_friction,
                self.cur_penetration,
                self.max_penetration,
                self.object_vel_sum,
                self.object_vel_count,
                self.object_initial_z,
                self.max_object_z,
                task_idx,
                self.world_nan_frame,
                task_timer,
                task_durations,
                int(task_durations.shape[0]),
                ee_pos_target,
                ee_pos_actual,
            ],
            outputs=[self.gui_staging],
        )
        return self.gui_staging.numpy()  # single small array readback (24 floats)

    def readback_all(self) -> dict[str, np.ndarray]:
        """Single batched GPU -> CPU transfer at episode end. Use in test_final only."""
        return {
            "max_pad_force": self.max_pad_force.numpy(),
            "max_pad_friction": self.max_pad_friction.numpy(),
            "max_table_force": self.max_table_force.numpy(),
            "max_table_friction": self.max_table_friction.numpy(),
            "max_penetration": self.max_penetration.numpy(),
            "max_object_z": self.max_object_z.numpy(),
            "object_initial_z": self.object_initial_z.numpy(),
            "object_z_at_hold_start": self.object_z_at_hold_start.numpy(),
            "object_z_at_hold_end": self.object_z_at_hold_end.numpy(),
            "object_vel_sum": self.object_vel_sum.numpy(),
            "object_vel_count": self.object_vel_count.numpy(),
            "world_nan_frame": self.world_nan_frame.numpy(),
            "state_pad_force_sum": self.state_pad_force_sum.numpy(),
            "state_pad_force_max": self.state_pad_force_max.numpy(),
            "state_table_force_sum": self.state_table_force_sum.numpy(),
            "state_table_force_max": self.state_table_force_max.numpy(),
            "state_pen_sum": self.state_pen_sum.numpy(),
            "state_pen_max": self.state_pen_max.numpy(),
            "state_vel_sum": self.state_vel_sum.numpy(),
            "state_count": self.state_count.numpy(),
        }


@wp.kernel(enable_backward=False)
def write_gripper_targets_kernel(
    ik_solution: wp.array2d[wp.float32],
    gripper_target: wp.array[wp.float32],
    joint_targets: wp.array2d[wp.float32],
    num_arm_dofs: int,
    gripper_dof_start: int,
    num_gripper_dofs: int,
):
    """Copy IK arm solution + gripper target to joint_targets 2D array."""
    tid = wp.tid()

    # Copy arm DOFs from IK solution
    for j in range(num_arm_dofs):
        joint_targets[tid, j] = ik_solution[tid, j]

    # Write gripper DOFs: all gripper joints get the same control value
    ctrl = gripper_target[tid]
    for j in range(num_gripper_dofs):
        joint_targets[tid, gripper_dof_start + j] = ctrl


@wp.kernel(enable_backward=False)
def write_mujoco_ctrl_kernel(
    ik_solution: wp.array2d[wp.float32],
    gripper_target: wp.array[wp.float32],
    mujoco_ctrl: wp.array2d[wp.float32],
    num_arm_actuators: int,
    gripper_actuator_idx: int,
):
    """Write IK arm solution + gripper target to MuJoCo ctrl."""
    tid = wp.tid()
    for j in range(num_arm_actuators):
        mujoco_ctrl[tid, j] = ik_solution[tid, j]
    mujoco_ctrl[tid, gripper_actuator_idx] = gripper_target[tid]


class Example:
    def __init__(self, viewer, args):
        self.test_mode = args.test
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.collide_substeps = args.collide_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = args.world_count
        self.seed = args.seed
        self.collision_mode = CollisionMode[args.collision_mode.upper()]
        self.kh = args.kh
        self.verbose = args.verbose
        self.table_half_xy = args.table_half_xy
        self.viewer = viewer
        self.episode_steps = 0

        self._generate_world_params()
        robot_builder, arm_only_builder = self._build_robot()
        # IK model: arm-only (no gripper DOFs) for cleaner IK solving
        self.model_arm_only = arm_only_builder.finalize()
        self._load_mesh_objects()
        self._world_z_half = np.full(self.world_count, -1.0)
        self._world_y_half = np.full(self.world_count, -1.0)
        scene = self._build_scene(robot_builder)
        self._setup_collision_sdf(scene)
        self.model = scene.finalize()

        # Contact budget: 2000 per world (matching example_hydro_robotiq_gripper)
        self.rigid_contact_max = 2_000 * self.world_count
        print(
            f"Bodies: {self.model.body_count}, Joints: {self.model.joint_count}, DOFs: {self.model.joint_coord_count}"
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # IK and state machine setup
        self._setup_ik()
        self._setup_state_machine()
        self.control = self.model.control()
        self.joint_target_shape = self.control.joint_target_pos.reshape((self.world_count, -1)).shape
        self.joint_targets_2d = wp.zeros(self.joint_target_shape, dtype=wp.float32)
        self.graph_ik = None

        # MuJoCo ctrl for MJCF general actuators (CTRL_DIRECT mode)
        self.has_mujoco_ctrl = hasattr(self.control, "mujoco") and self.control.mujoco is not None
        if self.has_mujoco_ctrl:
            ctrl = self.control.mujoco.ctrl
            num_actuators = ctrl.shape[0] // self.world_count
            self.mujoco_ctrl_2d = ctrl.reshape((self.world_count, num_actuators))
            # Gripper actuator is the last one (after 7 arm actuators)
            self.gripper_actuator_idx = self.num_franka_arm_dofs
            print(f"MuJoCo ctrl: {num_actuators} actuators/world, gripper at idx {self.gripper_actuator_idx}")

            # Initialize ctrl to current arm joint positions so arm holds its pose
            init_q = self.model.joint_q.numpy()
            ctrl_np = ctrl.numpy().reshape(self.world_count, num_actuators)
            dofs_per_world = self.model.joint_coord_count // self.world_count
            for w in range(self.world_count):
                q_start = w * dofs_per_world
                for j in range(self.num_franka_arm_dofs):
                    ctrl_np[w, j] = init_q[q_start + j]
            wp.copy(ctrl, wp.array(ctrl_np.flatten(), dtype=wp.float32))

        self._setup_contact_sensor()
        self._create_collision_pipeline()
        self._create_solver()
        self._create_sensor_contacts()
        self._setup_metrics()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.5, 0.0, 0.5), -15, -140)
        self.viewer.set_world_offsets(wp.vec3(2.0, 2.0, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True
        self._setup_gui()
        self.capture()

    def _download_assets(self):
        """Download menagerie assets once and cache paths."""
        if hasattr(self, "_franka_dir"):
            return
        from newton._src.utils.download_assets import download_git_folder

        self._franka_dir = download_git_folder(
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            "franka_emika_panda",
        )
        self._robotiq_dir = download_git_folder(
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            "robotiq_2f85_v4",
        )

    def _build_robot(self):
        """Build a ModelBuilder containing Franka Panda (no hand) + Robotiq 2F-85 V4."""
        self._download_assets()

        self.shape_cfg = newton.ModelBuilder.ShapeConfig(
            kh=self.kh,
            gap=0.0005,
            mu=1.0,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg = self.shape_cfg
        builder.rigid_gap = self.shape_cfg.gap
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # --- Franka Panda (no hand) from MuJoCo Menagerie ---
        # Layout matching ik_cube_stacking: table at (0, -0.5), robot 0.5m behind it.
        self.table_pos = wp.vec3(0.0, -0.5, 0.05)  # table body center
        self.table_top = wp.vec3(0.0, -0.5, 0.1)  # table surface
        self.robot_base_pos = wp.vec3(-0.5, -0.5, 0.1)  # on table, 0.5m behind center

        builder.add_mjcf(
            str(self._franka_dir / "panda_nohand.xml"),
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )

        # Initial arm pose (same as ik_cube_stacking — reach-forward config)
        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        builder.joint_q[0:7] = init_q
        builder.joint_target_pos[0:7] = init_q

        # Save arm-only builder for IK (before attaching gripper)
        # The IK only needs the arm kinematic chain — link_offset handles TCP.
        arm_only_builder = copy.deepcopy(builder)

        def find_body(name):
            return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

        # --- Robotiq 2F-85 V4 ---
        robotiq_mjcf = _patch_v4_mjcf(str(self._robotiq_dir / "2f85.xml"))

        # Attach Robotiq to link7 (last link with mass), applying the attachment
        # body's transform (z=0.107m, quat from panda_nohand.xml) plus an extra
        # 90° Z rotation for better jaw alignment with the workspace.
        link7_idx = find_body("link7")
        # attachment body in panda_nohand.xml: pos="0 0 0.107" quat="0.3826834 0 0 0.9238795"
        # MJCF quat (w,x,y,z) → Warp quat (x,y,z,w)
        q_attach = wp.quat(0.0, 0.0, 0.9238795, 0.3826834)
        q_90z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
        robotiq_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.107),
            wp.mul(q_attach, q_90z),
        )
        builder.add_mjcf(
            robotiq_mjcf,
            parent_body=link7_idx,
            xform=robotiq_xform,
            enable_self_collisions=False,
        )

        # Track key body indices.
        # The attachment body from panda_nohand.xml has zero mass/inertia;
        # give it a tiny mass so the inertia validator doesn't warn.
        self.attachment_body_idx = find_body("attachment")
        builder.body_mass[self.attachment_body_idx] = 1e-6
        builder.body_inertia[self.attachment_body_idx] = wp.mat33(np.eye(3, dtype=np.float32) * 1e-9)
        franka_body_count = self.attachment_body_idx + 1
        self.robotiq_base_body_idx = next(
            i for i in range(franka_body_count, builder.body_count) if builder.body_label[i].endswith("/base")
        )

        # --- Joint configuration ---
        # panda_nohand: 7 arm DOFs, no finger joints
        self.num_franka_arm_dofs = 7
        self.gripper_dof_start = self.num_franka_arm_dofs  # Robotiq DOFs start right after arm
        self.num_gripper_dofs = builder.joint_coord_count - self.num_franka_arm_dofs

        print(
            f"Joint layout: {self.num_franka_arm_dofs} arm + "
            f"{self.num_gripper_dofs} gripper = {builder.joint_coord_count} total DOFs"
        )
        print(f"Gripper DOF range: [{self.gripper_dof_start}, {self.gripper_dof_start + self.num_gripper_dofs})")

        # Gripper armature scaling for hydroelastic stability (from example_hydro_robotiq_gripper)
        # The MJCF default armature is too small for hydroelastic contact stiffness.
        if self.num_gripper_dofs == 6:
            mjcf_armature = np.array([0.005, 0.001, 0.001, 0.005, 0.001, 0.001])
        else:
            mjcf_armature = np.array([0.005, 0.001, 0.001, 0.001, 0.005, 0.001, 0.001, 0.001])
        armature_scale = 2.0
        gripper_armature = (armature_scale * mjcf_armature).tolist()
        builder.joint_armature[self.gripper_dof_start : self.gripper_dof_start + len(gripper_armature)] = (
            gripper_armature
        )

        # Gravity compensation on all bodies (arm + gripper)
        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(self.num_franka_arm_dofs):
            gravcomp_attr.values[dof_idx] = True

        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(1, builder.body_count):
            gravcomp_body.values[body_idx] = 1.0

        self.bodies_per_world = builder.body_count
        self.joints_per_world = builder.joint_count
        self.dofs_per_world = builder.joint_coord_count

        return builder, arm_only_builder

    def _generate_world_params(self):
        """Generate per-world object parameters: shape, mass, and size."""
        rng = np.random.default_rng(self.seed)
        wc = self.world_count

        # Round-robin shape assignment across 8 object types
        self.world_shapes = [ObjectShape(i % NUM_SHAPES) for i in range(wc)]

        # Fixed density (1000 kg/m³ ≈ water) — mass varies with shape volume and size.
        # This gives a natural mass range (~4g–244g) without unrealistic density combos.
        self.object_density = 1000.0

        # Base half-size 25 mm with uniform variation factor.
        # Larger base than the robotiq gripper example (15mm) to improve SDF quality
        # on mesh objects and give better contact area for grasping.
        base_hs = 0.025
        self.world_half_sizes = base_hs * rng.uniform(0.75, 1.25, size=wc)

        if self.verbose:
            for i in range(wc):
                print(
                    f"  World {i:3d}: shape={SHAPE_NAMES[self.world_shapes[i]]:>12s}  "
                    f"hs={self.world_half_sizes[i] * 1000:.1f} mm"
                )

    def _load_mesh_objects(self):
        """Load mesh assets for all mesh-based shapes (only those actually needed).

        SDFs are NOT built here — they are built at the correct scale in
        ``_setup_collision_sdf`` after scale-baking per world.
        """
        needed = set(self.world_shapes)
        self.mesh_objects = {}

        if ObjectShape.CUP in needed:
            cup_path = newton.utils.download_asset("manipulation_objects/cup")
            self.mesh_objects[ObjectShape.CUP] = _load_mesh_asset_no_sdf(cup_path, "/root/Model/Model")

        if ObjectShape.RUBBER_DUCK in needed:
            duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
            self.mesh_objects[ObjectShape.RUBBER_DUCK] = _load_mesh_asset_no_sdf(duck_path, "/root/Model/SurfaceMesh")

        if ObjectShape.LEGO_BRICK in needed:
            self.mesh_objects[ObjectShape.LEGO_BRICK] = _lego_make_brick_mesh(4, 2)

        if ObjectShape.RJ45_PLUG in needed:
            rj45_path = newton.examples.get_asset("rj45_plug.usd")
            self.mesh_objects[ObjectShape.RJ45_PLUG] = _load_usd_mesh_local(rj45_path, "/World/Plug")

        if ObjectShape.BEAR in needed:
            bear_path = newton.examples.get_asset("bear.usd")
            self.mesh_objects[ObjectShape.BEAR] = _load_usd_mesh_local(bear_path, "/root/bear/bear")

        if ObjectShape.NUT in needed or ObjectShape.BOLT in needed:
            asset_path = newton.examples.download_external_git_folder(
                _NUT_BOLT_REPO_URL, _NUT_BOLT_FOLDER
            )
            if ObjectShape.NUT in needed:
                nut_file = str(asset_path / f"factory_nut_{_NUT_BOLT_ASSEMBLY}_subdiv_3x.obj")
                self.mesh_objects[ObjectShape.NUT] = _load_obj_mesh_trimesh(nut_file)
            if ObjectShape.BOLT in needed:
                bolt_file = str(asset_path / f"factory_bolt_{_NUT_BOLT_ASSEMBLY}.obj")
                self.mesh_objects[ObjectShape.BOLT] = _load_obj_mesh_trimesh(bolt_file)

    def _add_object(self, builder: newton.ModelBuilder, world_id: int):
        """Add a grasp object to the builder for one world. All shapes as meshes.

        Converting all primitive shapes to meshes via ``newton.Mesh.create_*()``
        ensures every world has the same shape type (MESH), which is required by
        the MuJoCo solver in ``separate_worlds=True`` mode.
        """
        shape = self.world_shapes[world_id]
        hs = float(self.world_half_sizes[world_id])

        if shape == ObjectShape.BOX:
            mesh = newton.Mesh.create_box(hs, hs, hs, compute_inertia=False)
        elif shape == ObjectShape.SPHERE:
            mesh = newton.Mesh.create_sphere(hs, compute_inertia=False)
        elif shape == ObjectShape.CYLINDER:
            mesh = newton.Mesh.create_cylinder(hs, hs, up_axis=newton.Axis.X, compute_inertia=False)
        elif shape == ObjectShape.CAPSULE:
            mesh = newton.Mesh.create_capsule(hs, hs, up_axis=newton.Axis.X, compute_inertia=False)
        elif shape == ObjectShape.ELLIPSOID:
            mesh = newton.Mesh.create_ellipsoid(hs * 2.0, hs, hs, compute_inertia=False)
        elif shape in _MESH_SHAPES:
            mesh = self.mesh_objects[shape]
        else:
            raise ValueError(f"Unknown object shape: {shape}")

        if shape in _MESH_SHAPES:
            verts = mesh.vertices
            extents = verts.max(axis=0) - verts.min(axis=0)
            max_extent = float(extents.max())
            target_extent = 2.0 * hs
            sc = target_extent / max_extent if max_extent > 0 else 1.0
            scale = wp.vec3(sc, sc, sc)
            self._world_y_half[world_id] = float(extents[1]) / 2.0 * sc
            self._world_z_half[world_id] = float(extents[2]) / 2.0 * sc
        else:
            scale = wp.vec3(1.0, 1.0, 1.0)
            self._world_y_half[world_id] = hs
            self._world_z_half[world_id] = hs

        tt = self.table_top
        z_gap = 0.0005
        z_half = self._world_z_half[world_id]
        obj_z = float(tt[2]) + z_half + z_gap

        if shape == ObjectShape.NUT:
            obj_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
        else:
            obj_rot = wp.quat_identity()

        obj_xform = wp.transform(wp.vec3(float(tt[0]), float(tt[1]), obj_z), obj_rot)
        obj_body = builder.add_body(xform=obj_xform, label="object")

        obj_cfg = replace(self.shape_cfg, density=self.object_density)
        builder.add_shape_mesh(
            body=obj_body,
            mesh=mesh,
            scale=scale,
            cfg=obj_cfg,
            label=f"object_shape_{shape.name.lower()}",
        )
        return obj_body

    def _build_scene(self, robot_builder):
        """Build the multi-world scene with per-world robot + table + object."""
        self.table_height = 0.1

        # Table as kinematic body (mass=0) so MuJoCo detects collisions per world.
        # Using body=-1 (static) causes MuJoCo to miss collisions for replicated worlds.
        table_mesh = newton.Mesh.create_box(self.table_half_xy, self.table_half_xy, 0.05, compute_inertia=False)
        table_cfg = replace(self.shape_cfg, density=0.0)

        scene = newton.ModelBuilder()
        for world_id in range(self.world_count):
            scene.begin_world()
            scene.add_builder(robot_builder)

            # Kinematic table body (does not respond to forces)
            table_body = scene.add_body(
                xform=wp.transform(self.table_pos, wp.quat_identity()),
                label="table",
                is_kinematic=True,
            )
            scene.add_shape_mesh(
                body=table_body,
                mesh=table_mesh,
                cfg=table_cfg,
                label="table_shape",
            )

            # Object on table surface at (0, -0.5, table_top + hs)
            obj_body = self._add_object(scene, world_id)

            # After first world: record the object's local body index
            if world_id == 0:
                self.object_body_offset = obj_body

            scene.end_world()

        scene.add_ground_plane()
        return scene

    def _setup_ik(self):
        """Create the IK solver with per-world problems.

        Uses the arm-only model (no gripper DOFs) for clean IK solving.
        The EE link is the ``attachment`` body at the Franka flange.
        The TCP offset (174mm) maps the IK target to the fingertip center.
        """
        # Find the attachment body in the arm-only model (last body)
        self.ik_ee_index = self.model_arm_only.body_count - 1

        # Evaluate FK to get initial EE pose
        state_single = self.model_arm_only.state()
        newton.eval_fk(self.model_arm_only, self.model_arm_only.joint_q, self.model_arm_only.joint_qd, state_single)
        body_q_np = state_single.body_q.numpy()
        ee_tf = wp.transform(*body_q_np[self.ik_ee_index])
        init_ee_pos = wp.transform_get_translation(ee_tf)
        init_ee_rot = wp.transform_get_rotation(ee_tf)

        wc = self.world_count

        # Robotiq 2F-85 TCP offset: 174mm from base to fingertip center (closed).
        # link_offset maps IK targets to the TCP so grasp_z = object center Z.
        # init target must be the TCP position (not EE body) so the initial config
        # is already a perfect solution and the IK doesn't drift.
        self.tcp_offset = 0.174
        tcp_offset_vec = wp.vec3(0.0, 0.0, self.tcp_offset)
        init_tcp_pos = init_ee_pos + wp.quat_rotate(init_ee_rot, tcp_offset_vec)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_ee_index,
            link_offset=tcp_offset_vec,
            target_positions=wp.array([init_tcp_pos] * wc, dtype=wp.vec3),
        )

        # Rotation target: use the arm-only model's attachment body rotation.
        # The Robotiq is rigidly mounted with an extra 90° Z rotation, but
        # the IK only controls the arm — it should keep the attachment body
        # at its initial orientation.
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ik_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([init_ee_rot[:4]] * wc, dtype=wp.vec4),
        )

        # Joint limit objective — replicate single-model limits for all problems
        ik_dofs = self.model_arm_only.joint_coord_count
        ll = self.model_arm_only.joint_limit_lower.numpy()
        lu = self.model_arm_only.joint_limit_upper.numpy()
        joint_limit_lower = wp.array(np.tile(ll, wc), dtype=wp.float32)
        joint_limit_upper = wp.array(np.tile(lu, wc), dtype=wp.float32)
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=joint_limit_lower.flatten(),
            joint_limit_upper=joint_limit_upper.flatten(),
        )

        # IK solution array: (world_count, joint_coord_count)
        # Initialize from the multi-world joint_q, taking each world's slice
        joint_q_np = self.model.joint_q.numpy().reshape(wc, -1)[:, :ik_dofs]
        self.joint_q_ik = wp.array(joint_q_np, dtype=wp.float32)

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_arm_only,
            n_problems=wc,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _setup_state_machine(self):
        """Initialize per-world state machine arrays and grasp parameters."""
        wc = self.world_count

        # Task state: index and timer per world
        self.task_idx = wp.zeros(wc, dtype=wp.int32)
        self.task_timer = wp.zeros(wc, dtype=wp.float32)

        # Task durations: [APPROACH, CLOSE_GRIPPER, SETTLE, LIFT, HOLD, DONE]
        durations = [1.5, 1.0, 0.5, 1.0, 1.5, 0.0]
        self.task_durations = wp.array(durations, dtype=wp.float32)

        # Per-world grasp parameters
        # IK link_offset handles the TCP (174mm) at the fully-closed pose.
        # When the gripper is partially open (grasping an object), the 4-bar
        # linkage tilts the pads slightly upward, giving extra vertical
        # clearance above what the pad geometry alone suggests. `grasp_clearance`
        # captures the effective object height we can grip without shifting the
        # end-effector upward. Objects taller than this get grasp_z raised so
        # the pads cover the top of the object rather than the gripper body.
        grasp_clearance = 0.05
        table_top_z = float(self.table_top[2])
        grasp_z_np = np.array(
            [
                table_top_z + 0.001 + max(0.0, 2.0 * self._world_z_half[i] - grasp_clearance)
                for i in range(wc)
            ],
            dtype=np.float32,
        )
        # lift_z: lift EE by 10cm above grasp position
        lift_z_np = grasp_z_np + 0.1
        # grasp_ctrl: per-world closure target computed from object Y-width + margin.
        # Margins are a percentage of object Y-width, derived from the fixed-size
        # margins in example_hydro_robotiq_gripper (3mm/30mm=10%, 4mm/30mm≈13%).
        # This scales correctly for our varying object sizes.
        # Per-shape grasp margins. Forces are non-monotonic due to mesh facet
        # alignment with pads and kh=2e11 stiffness. Margins tuned for reliable
        # grasping; pad forces may exceed the Robotiq's 235N spec on some shapes.
        grasp_margin_pct = {
            ObjectShape.BOX: 0.05,
            ObjectShape.SPHERE: 0.05,
            ObjectShape.CYLINDER: 0.05,
            ObjectShape.CAPSULE: 0.05,
            ObjectShape.ELLIPSOID: 0.05,
            ObjectShape.CUP: 0.22,
            ObjectShape.RUBBER_DUCK: 0.10,
            ObjectShape.LEGO_BRICK: 0.15,
            ObjectShape.RJ45_PLUG: 0.10,
            ObjectShape.BEAR: 0.15,
            ObjectShape.NUT: 0.08,
            ObjectShape.BOLT: 0.12,
        }
        stroke_mm = 85.0  # Robotiq 2F-85 full stroke
        grasp_ctrl_list = []
        for i in range(wc):
            shape = self.world_shapes[i]
            y_half = self._world_y_half[i]
            y_width_mm = 2.0 * y_half * 1000.0
            margin_mm = grasp_margin_pct[shape] * y_width_mm
            ctrl = min(255.0, max(0.0, 255.0 * (1.0 - (y_width_mm - 2.0 * margin_mm) / stroke_mm)))
            grasp_ctrl_list.append(ctrl)
        grasp_ctrl_np = np.array(grasp_ctrl_list, dtype=np.float32)

        # Object XY position (same for all worlds)
        self.object_xy = wp.vec2(float(self.table_top[0]), float(self.table_top[1]))

        # EE rotation target: use the arm-only model's attachment body rotation.
        # This matches the IK rotation objective so there's no mismatch.
        state_arm = self.model_arm_only.state()
        newton.eval_fk(self.model_arm_only, self.model_arm_only.joint_q, self.model_arm_only.joint_qd, state_arm)
        arm_ee_rot = wp.transform_get_rotation(wp.transform(*state_arm.body_q.numpy()[self.ik_ee_index]))
        self.ee_rot_down = wp.vec4(arm_ee_rot[0], arm_ee_rot[1], arm_ee_rot[2], arm_ee_rot[3])

        self.grasp_z = wp.array(grasp_z_np, dtype=wp.float32)
        self.lift_z = wp.array(lift_z_np, dtype=wp.float32)
        self.grasp_ctrl = wp.array(grasp_ctrl_np, dtype=wp.float32)

        # Target arrays (raw and interpolated)
        self.ee_pos_target = wp.zeros(wc, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(wc, dtype=wp.vec4)
        self.ee_pos_target_interp = wp.zeros(wc, dtype=wp.vec3)
        self.ee_rot_target_interp = wp.zeros(wc, dtype=wp.vec4)
        self.gripper_target = wp.zeros(wc, dtype=wp.float32)

        # Actual EE position (extracted from body_q each frame)
        self.ee_pos_actual = wp.zeros(wc, dtype=wp.vec3)

        # Global body indices for EE in each world
        # Use the attachment body (not Robotiq base) for EE tracking — this matches
        # the IK model's EE link so rotation targets are consistent.
        body_ws = self.model.body_world_start.numpy()
        ee_global_indices = np.array(
            [int(body_ws[i]) + self.attachment_body_idx for i in range(wc)],
            dtype=np.int32,
        )
        self.ee_body_global_indices = wp.array(ee_global_indices, dtype=wp.int32)

        # Snapshot of body_q at the start of each task (for interpolation)
        # Initialize with current state_0 body_q
        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.num_bodies_total = self.model.body_count

    def _setup_collision_sdf(self, builder):
        """Build SDFs on all collision shapes; mark fingertips as hydroelastic.

        Operates on the scene builder BEFORE finalize(), following the same
        pattern as ``example_hydro_robotiq_gripper``:
        Pass 1 — SDF on every collision shape (BOX→MESH + SDF build).
        Pass 2 — HYDROELASTIC flag on fingertip pads and object shapes.
        """
        if self.collision_mode not in (CollisionMode.NEWTON_SDF, CollisionMode.NEWTON_HYDROELASTIC):
            return

        sdf_narrow_band = (-0.0015, 0.0015)
        use_hydro = self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC

        # ---- Pass 1: build SDF on every collision shape ----
        # Object shapes get higher resolution (128) for better contact quality;
        # robot links, gripper, and table use 64.
        for shape_idx in range(builder.shape_count):
            if not (builder.shape_flags[shape_idx] & newton.ShapeFlags.COLLIDE_SHAPES):
                continue

            label = builder.shape_label[shape_idx] if shape_idx < len(builder.shape_label) else ""
            is_object = "object" in label
            if is_object and ("nut" in label or "bolt" in label):
                sdf_max_res = 96
            elif is_object:
                sdf_max_res = 96
            else:
                sdf_max_res = 64
            # Object meshes get a smaller margin to avoid inflating thin features
            sdf_margin = 0.0002 if is_object else self.shape_cfg.gap

            if builder.shape_type[shape_idx] == newton.GeoType.BOX:
                # Convert BOX to MESH + SDF (needed for pad shapes from MJCF)
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
                mesh.build_sdf(max_resolution=sdf_max_res, narrow_band_range=sdf_narrow_band, margin=sdf_margin)
                builder.shape_type[shape_idx] = newton.GeoType.MESH
                builder.shape_source[shape_idx] = mesh
                builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)

            elif builder.shape_type[shape_idx] == newton.GeoType.MESH:
                mesh = builder.shape_source[shape_idx]
                if mesh is None:
                    continue
                scale = np.asarray(builder.shape_scale[shape_idx], dtype=np.float32)
                if not np.allclose(scale, 1.0):
                    # Bake scale into vertices and rebuild SDF (required for hydroelastic)
                    mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
                    builder.shape_source[shape_idx] = mesh
                    builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
                    if mesh.sdf is not None:
                        mesh.clear_sdf()
                    mesh.build_sdf(max_resolution=sdf_max_res, narrow_band_range=sdf_narrow_band, margin=sdf_margin)
                elif mesh.sdf is None:
                    mesh.build_sdf(max_resolution=sdf_max_res, narrow_band_range=sdf_narrow_band, margin=sdf_margin)

        # ---- Pass 2: fingertip + object shapes get hydroelastic flag ----
        if use_hydro:
            pad_names = {"left_pad1", "left_pad2", "right_pad1", "right_pad2"}
            tongue_names = {"left_follower_geom_1", "right_follower_geom_0"}
            fingertip_names = pad_names | tongue_names

            hydro_count = 0
            for shape_idx, label in enumerate(builder.shape_label):
                short = label.split("/")[-1] if label else ""
                is_fingertip = short in fingertip_names
                is_object = "object" in short

                if is_fingertip or is_object:
                    cfg = self.shape_cfg
                    builder.shape_gap[shape_idx] = cfg.gap
                    builder.shape_material_mu[shape_idx] = cfg.mu
                    builder.shape_material_mu_torsional[shape_idx] = cfg.mu_torsional
                    builder.shape_material_mu_rolling[shape_idx] = cfg.mu_rolling
                    builder.shape_material_kh[shape_idx] = self.kh
                    builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC
                    if is_fingertip:
                        builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.VISIBLE
                    hydro_count += 1

            print(f"[SDF setup] Marked {hydro_count} shapes as HYDROELASTIC")

    def _create_collision_pipeline(self):
        """Create collision pipeline based on collision mode."""
        if self.collision_mode == CollisionMode.MUJOCO:
            self.collision_pipeline = None
            self.contacts = None
        elif self.collision_mode == CollisionMode.NEWTON_DEFAULT:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                broad_phase="nxn",
            )
            self.contacts = self.collision_pipeline.contacts()
        elif self.collision_mode == CollisionMode.NEWTON_SDF:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                broad_phase="explicit",
                reduce_contacts=True,
            )
            self.contacts = self.collision_pipeline.contacts()
        elif self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                broad_phase="explicit",
                reduce_contacts=True,
                sdf_hydroelastic_config=HydroelasticSDF.Config(
                    output_contact_surface=hasattr(self.viewer, "renderer"),
                    buffer_fraction=1.0,
                    buffer_mult_iso=2,
                    buffer_mult_contact=2,
                    anchor_contact=True,
                ),
            )
            self.contacts = self.collision_pipeline.contacts()
        else:
            raise ValueError(f"Unknown collision mode: {self.collision_mode}")

    def _create_solver(self):
        """Create the MuJoCo solver."""
        use_mujoco_contacts = self.collision_mode == CollisionMode.MUJOCO
        nconmax_per_world = self.rigid_contact_max // self.world_count
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=use_mujoco_contacts,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=50,
            ls_iterations=100,
            impratio=10.0,
            njmax=nconmax_per_world,
            nconmax=nconmax_per_world,
        )

    def _setup_contact_sensor(self):
        """Create a SensorContact on object bodies with pad counterpart shapes.

        Must be called BEFORE contacts objects are created so that the ``force``
        attribute is requested on the model.
        """
        self.contact_sensor_pad = SensorContact(
            self.model,
            sensing_obj_bodies="object",
            counterpart_shapes="*pad*",
        )
        self.contact_sensor_table = SensorContact(
            self.model,
            sensing_obj_bodies="object",
            counterpart_shapes="table*",
            request_contact_attributes=False,  # already requested by pad sensor
        )

    def _create_sensor_contacts(self):
        """Create or assign the Contacts object used by the contact sensor.

        For MuJoCo collision mode, a dedicated Contacts buffer is allocated from
        the solver's maximum contact count.  For Newton modes, the collision
        pipeline's Contacts already has the ``force`` attribute because
        ``_setup_contact_sensor`` was called first.
        """
        if self.collision_mode == CollisionMode.MUJOCO:
            self.contacts = Contacts(
                self.solver.get_max_contact_count(),
                0,
                requested_attributes=self.model.get_requested_contact_attributes(),
            )

    def _setup_metrics(self):
        """Initialize per-world metric tracking on GPU via GraspMetrics."""
        nt = int(TaskType.DONE)
        self.metrics = GraspMetrics(
            world_count=self.world_count,
            num_states=nt,
            hold_state=int(TaskType.HOLD),
            object_body_offset=self.object_body_offset,
        )
        self.metrics.capture_initial_object_z(self.state_0, self.model.body_world_start)

    def _setup_gui(self):
        """Register the side-panel GUI callback with the viewer."""
        self.selected_world = 0
        self.show_debug_frames = True
        self.show_isosurface = False
        # Debug-frame line buffers (9 lines per world: object + EE + TCP frames).
        wc = self.world_count
        self._debug_begins = wp.zeros(wc * 9, dtype=wp.vec3)
        self._debug_ends = wp.zeros(wc * 9, dtype=wp.vec3)
        self._debug_colors = wp.zeros(wc * 9, dtype=wp.vec3)
        # GUI readback staging cache (populated by stage_gui; read downstream).
        self._cached_gui = np.zeros(_GUI_STAGE_SIZE, dtype=np.float32)
        self._gui_read_interval = 10
        if hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = self.show_isosurface
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._gui_impl, position="side")

    def _gui_impl(self, imgui):
        wc = self.world_count

        # World selector
        changed, val = imgui.slider_int("World", self.selected_world, 0, wc - 1)
        if changed:
            self.selected_world = val
        w = self.selected_world

        imgui.separator()

        # Object description (static metadata -- no per-frame GPU read needed).
        shape_name = SHAPE_NAMES[self.world_shapes[w]] if hasattr(self, "world_shapes") else "?"
        hs_mm = float(self.world_half_sizes[w]) * 1000.0 if hasattr(self, "world_half_sizes") else 0.0
        # Mass is static; cache once via a tiny read on first access.
        if not hasattr(self, "_cached_body_mass"):
            self._cached_body_mass = self.model.body_mass.numpy().copy()
            self._cached_body_world_start = self.model.body_world_start.numpy().copy()
        obj_global = int(self._cached_body_world_start[w]) + self.object_body_offset
        mass = float(self._cached_body_mass[obj_global])
        imgui.text(f"Shape: {shape_name}")
        imgui.text(f"Mass:  {mass:.4f} kg")
        imgui.text(f"Size:  {hs_mm:.1f} mm (half-size)")

        imgui.separator()

        # Throttled GPU->CPU sync. Pack selected-world metrics into a 24-float
        # staging buffer via a single-thread Warp kernel; read back once.
        has_metrics = hasattr(self, "metrics")
        has_tasks = hasattr(self, "task_idx") and hasattr(self, "task_timer") and hasattr(self, "task_durations")
        has_ee = hasattr(self, "ee_pos_target") and hasattr(self, "ee_pos_actual")
        if has_metrics and has_tasks and has_ee and (self.episode_steps % self._gui_read_interval) == 0:
            self._cached_gui = self.metrics.stage_gui(
                self.selected_world,
                self.state_0,
                self.model.body_world_start,
                self.task_idx,
                self.task_timer,
                self.task_durations,
                self.ee_pos_target,
                self.ee_pos_actual,
            )
        s = self._cached_gui

        cur_pad_f = float(s[0])
        max_pad_f = float(s[1])
        cur_pad_fr = float(s[2])
        max_pad_fr = float(s[3])
        cur_tbl_f = float(s[4])
        max_tbl_f = float(s[5])
        cur_pen_mm = float(s[8]) * 1000.0
        max_pen_mm = float(s[9]) * 1000.0
        avg_vel_mms = float(s[10]) * 1000.0
        obj_z = float(s[11])
        init_z = float(s[12])
        max_z = float(s[13])
        task_val = int(s[14])
        nan_frame_val = int(s[15])
        cur_timer = float(s[16])
        task_dur = float(s[17])
        ee_target = (float(s[18]), float(s[19]), float(s[20]))
        ee_actual = (float(s[21]), float(s[22]), float(s[23]))

        # State machine (task name resolved from staged task index).
        if has_tasks:
            cur_task = task_val if has_metrics else 0
            task_name = TASK_NAMES[cur_task] if 0 <= cur_task < len(TASK_NAMES) else "?"
            imgui.text(f"Task:  {task_name}")
            imgui.text(f"Timer: {cur_timer:.2f}s / {task_dur:.2f}s")

        imgui.separator()

        # EE position error (read from staging buffer -- no per-frame .numpy()).
        if has_ee:
            err_x = ee_target[0] - ee_actual[0]
            err_y = ee_target[1] - ee_actual[1]
            err_z = ee_target[2] - ee_actual[2]
            imgui.text(f"EE err: x={err_x * 1000:+.2f} y={err_y * 1000:+.2f} z={err_z * 1000:+.2f} mm")

        imgui.separator()

        # Pad (finger) forces
        if has_metrics:
            imgui.text(f"Pad F:   {cur_pad_f:.1f} N  (max: {max_pad_f:.1f} N)")
            imgui.text(f"Pad Fr:  {cur_pad_fr:.1f} N  (max: {max_pad_fr:.1f} N)")

            # Table forces
            imgui.text(f"Table F: {cur_tbl_f:.1f} N  (max: {max_tbl_f:.1f} N)")

            # Penetration
            imgui.text(f"Penetration: {cur_pen_mm:.3f} mm  (max: {max_pen_mm:.3f} mm)")

            # Object running-average velocity (jitter indicator)
            imgui.text(f"Avg vel: {avg_vel_mms:.2f} mm/s")

        imgui.separator()

        # Object Z and lift
        if has_metrics:
            lift = obj_z - init_z
            max_lift = max_z - init_z
            imgui.text(f"Object Z: {obj_z:.4f} m  (init: {init_z:.4f} m)")
            imgui.text(f"Lift: {lift * 1000:.1f} mm  (max: {max_lift * 1000:.1f} mm)")

            # NaN indicator
            if nan_frame_val >= 0:
                imgui.text(f"NaN at frame {nan_frame_val}!")

        imgui.separator()
        _, self.show_debug_frames = imgui.checkbox("Show Frames", self.show_debug_frames)
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC and hasattr(self.viewer, "renderer"):
            changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
            if changed:
                self.viewer.show_hydro_contact_surface = self.show_isosurface
        imgui.text(f"Frame: {self.episode_steps}  t={self.sim_time:.2f}s")

    def _update_metrics(self):
        """Per-step metric update. All GPU-resident, no .numpy() calls."""
        self.solver.update_contacts(self.contacts, self.state_0)
        self.contact_sensor_pad.update(self.state_0, self.contacts)
        self.contact_sensor_table.update(self.state_0, self.contacts)

        self.metrics.update(
            state_0=self.state_0,
            pad_sensor=self.contact_sensor_pad,
            table_sensor=self.contact_sensor_table,
            mjw_data=self.solver.mjw_data,
            task_idx=self.task_idx,
            body_world_start_array=self.model.body_world_start,
            frame_count=self.episode_steps,
        )

    def set_joint_targets(self):
        """Compute IK targets from state machine and solve IK each frame."""
        wc = self.world_count

        # 1. Compute targets from state machine (with interpolation)
        wp.launch(
            set_target_pose_kernel,
            dim=wc,
            inputs=[
                self.task_idx,
                self.task_timer,
                self.task_durations,
                self.grasp_z,
                self.lift_z,
                self.grasp_ctrl,
                self.object_xy,
                self.ee_rot_down,
                self.task_init_body_q,
                self.ee_body_global_indices,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_rot_target,
                self.ee_pos_target_interp,
                self.ee_rot_target_interp,
                self.gripper_target,
            ],
        )

        # 2. Update IK objectives with INTERPOLATED targets (smooth motion)
        self.pos_obj.set_target_positions(self.ee_pos_target_interp)
        self.rot_obj.set_target_rotations(self.ee_rot_target_interp)

        # 3. Solve IK
        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        # 4. Write IK arm solution + gripper target to joint_targets
        wp.launch(
            write_gripper_targets_kernel,
            dim=wc,
            inputs=[
                self.joint_q_ik,
                self.gripper_target,
                self.joint_targets_2d,
                self.num_franka_arm_dofs,
                self.gripper_dof_start,
                self.num_gripper_dofs,
            ],
        )
        wp.copy(self.control.joint_target_pos, self.joint_targets_2d.flatten())

        # Write arm + gripper to mujoco ctrl (CTRL_DIRECT for all MJCF general actuators)
        if self.has_mujoco_ctrl:
            wp.launch(
                write_mujoco_ctrl_kernel,
                dim=wc,
                inputs=[
                    self.joint_q_ik,
                    self.gripper_target,
                    self.mujoco_ctrl_2d,
                    self.num_franka_arm_dofs,
                    self.gripper_actuator_idx,
                ],
            )

        # 5. Extract actual EE positions from simulation state
        wp.launch(
            extract_ee_pos_kernel,
            dim=wc,
            inputs=[
                self.state_0.body_q,
                self.ee_body_global_indices,
            ],
            outputs=[
                self.ee_pos_actual,
            ],
        )

        # 6. Advance state machine (snapshots body_q on transition)
        bodies_per_world = self.model.body_count // self.world_count
        wp.launch(
            advance_task_kernel,
            dim=wc,
            inputs=[
                self.task_idx,
                self.task_timer,
                self.task_durations,
                self.ee_pos_target,
                self.ee_pos_actual,
                self.state_0.body_q,
                self.task_init_body_q,
                bodies_per_world,
                self.frame_dt,
            ],
        )

    def capture(self):
        self.graph_sim = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph_sim = capture.graph
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if self.collision_pipeline and i % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.set_joint_targets()
        if self.graph_sim is not None:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()
        self._update_metrics()
        self.sim_time += self.frame_dt
        self.episode_steps += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
            if (
                self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC
                and self.collision_pipeline is not None
                and self.collision_pipeline.hydroelastic_sdf is not None
            ):
                self.viewer.log_hydro_contact_surface(
                    self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
                    penetrating_only=True,
                )
        self._render_debug_frames()
        self.viewer.end_frame()

    def _render_debug_frames(self):
        """Draw XYZ coordinate frames at object, EE base, and TCP. GPU-resident."""
        if not hasattr(self.viewer, "log_lines"):
            return
        if not self.show_debug_frames:
            self.viewer.log_lines("/debug_frames", None, None, None)
            return

        # world_offsets may be None if the viewer never configured them.
        if hasattr(self.viewer, "world_offsets") and self.viewer.world_offsets is not None:
            world_offsets = self.viewer.world_offsets
        else:
            if not hasattr(self, "_zero_world_offsets"):
                self._zero_world_offsets = wp.zeros(self.world_count, dtype=wp.vec3)
            world_offsets = self._zero_world_offsets

        wp.launch(
            compute_debug_frame_lines_kernel,
            dim=self.world_count,
            inputs=[
                self.state_0.body_q,
                self.model.body_world_start,
                world_offsets,
                self.object_body_offset,
                self.robotiq_base_body_idx,
                float(self.tcp_offset),
                0.08,
            ],
            outputs=[self._debug_begins, self._debug_ends, self._debug_colors],
        )
        self.viewer.log_lines(
            "/debug_frames",
            self._debug_begins,
            self._debug_ends,
            self._debug_colors,
        )

    def test_final(self):
        wc = self.world_count
        lift_threshold_m = 0.05  # 50 mm lift required for success

        # Batch readback of all GPU-resident metrics for final report.
        m = self.metrics.readback_all()
        max_pen = m["max_penetration"]
        max_pad_f = m["max_pad_force"]
        max_pad_fr = m["max_pad_friction"]
        max_tbl_f = m["max_table_force"]
        max_tbl_fr = m["max_table_friction"]
        max_obj_z = m["max_object_z"]
        obj_init_z = m["object_initial_z"]
        hold_start = m["object_z_at_hold_start"]
        hold_end = m["object_z_at_hold_end"]
        vel_sum = m["object_vel_sum"]
        vel_count = m["object_vel_count"]
        nan_frame_arr = m["world_nan_frame"]
        state_pad_sum = m["state_pad_force_sum"]
        state_pad_max = m["state_pad_force_max"]
        state_tbl_sum = m["state_table_force_sum"]
        state_tbl_max = m["state_table_force_max"]
        state_pen_sum = m["state_pen_sum"]
        state_pen_max = m["state_pen_max"]
        state_vel_sum = m["state_vel_sum"]
        state_count = m["state_count"]

        # Compute per-world results
        body_ws = self.model.body_world_start.numpy()
        body_mass_np = self.model.body_mass.numpy()
        results = []
        for w in range(wc):
            shape_name = SHAPE_NAMES[self.world_shapes[w]]
            obj_global = int(body_ws[w]) + self.object_body_offset
            mass = float(body_mass_np[obj_global])
            hs = float(self.world_half_sizes[w])
            has_nan = nan_frame_arr[w] >= 0
            lift = float(max_obj_z[w] - obj_init_z[w]) if not has_nan else 0.0
            success = lift > lift_threshold_m and not has_nan

            # Slippage: Z drop during HOLD phase
            if not np.isnan(hold_start[w]) and not np.isnan(hold_end[w]):
                slippage = float(hold_start[w] - hold_end[w])
            else:
                slippage = 0.0

            results.append(
                {
                    "world": w,
                    "shape": shape_name,
                    "mass": mass,
                    "half_size": hs,
                    "success": success,
                    "lift_m": lift,
                    "slippage_m": slippage,
                    "max_penetration_m": float(max_pen[w]),
                    "max_pad_force_N": float(max_pad_f[w]),
                    "max_pad_friction_N": float(max_pad_fr[w]),
                    "max_table_force_N": float(max_tbl_f[w]),
                    "max_table_friction_N": float(max_tbl_fr[w]),
                    "avg_vel_ms": float(vel_sum[w] / max(1, int(vel_count[w]))),
                    "has_nan": has_nan,
                    "nan_frame": int(nan_frame_arr[w]) if has_nan else None,
                }
            )

        # --- Per-world results table ---
        w = 140
        print("\n" + "=" * w)
        print(f"  HETEROGENEOUS GRASP TEST REPORT  (collision_mode={self.collision_mode.name})")
        print("=" * w)
        print(
            f"{'W':>3}  {'Shape':>12}  {'Mass(kg)':>8}  {'HS(mm)':>6}  {'OK':>3}  "
            f"{'Lift(mm)':>9}  {'Slip(mm)':>9}  {'Pen(mm)':>8}  "
            f"{'PadF(N)':>9}  {'PadFr(N)':>9}  {'TblF(N)':>9}  {'Vel(mm/s)':>10}  {'NaN':>5}"
        )
        print("-" * w)
        for r in results:
            ok_str = "YES" if r["success"] else "NO"
            nan_str = str(r["nan_frame"]) if r["has_nan"] else "-"
            print(
                f"{r['world']:3d}  {r['shape']:>12s}  {r['mass']:8.4f}  "
                f"{r['half_size'] * 1000:6.1f}  {ok_str:>3s}  "
                f"{r['lift_m'] * 1000:9.1f}  {r['slippage_m'] * 1000:9.1f}  "
                f"{r['max_penetration_m'] * 1000:8.2f}  "
                f"{r['max_pad_force_N']:9.1f}  {r['max_pad_friction_N']:9.1f}  "
                f"{r['max_table_force_N']:9.1f}  "
                f"{r['avg_vel_ms'] * 1000:10.2f}  {nan_str:>5s}"
            )

        # --- Per-shape aggregation ---
        print("\n" + "-" * 100)
        print("  PER-SHAPE AGGREGATION")
        print("-" * 100)
        print(
            f"{'Shape':>12}  {'Count':>5}  {'Success':>7}  "
            f"{'Lift(mm)':>9}  {'Slip(mm)':>9}  "
            f"{'PadF(N)':>9}  {'TblF(N)':>9}  {'Vel(mm/s)':>10}"
        )
        print("-" * 100)
        for shape in ObjectShape:
            shape_results = [r for r in results if r["shape"] == shape.name]
            if not shape_results:
                continue
            n = len(shape_results)
            n_success = sum(1 for r in shape_results if r["success"])
            avg_lift = np.mean([r["lift_m"] for r in shape_results])
            avg_slip = np.mean([r["slippage_m"] for r in shape_results])
            avg_pad_f = np.mean([r["max_pad_force_N"] for r in shape_results])
            avg_tbl_f = np.mean([r["max_table_force_N"] for r in shape_results])
            avg_vel = np.mean([r["avg_vel_ms"] for r in shape_results])
            print(
                f"{shape.name:>12s}  {n:5d}  {n_success:3d}/{n:<3d}  "
                f"{avg_lift * 1000:9.1f}  {avg_slip * 1000:9.1f}  "
                f"{avg_pad_f:9.1f}  {avg_tbl_f:9.1f}  {avg_vel * 1000:10.2f}"
            )

        # --- Aggregate statistics ---
        n_success = sum(1 for r in results if r["success"])
        n_nan = sum(1 for r in results if r["has_nan"])
        success_rate = n_success / wc if wc > 0 else 0.0

        print("\n" + "-" * 80)
        print("  AGGREGATE STATISTICS")
        print("-" * 80)
        print(f"  Success rate:       {n_success}/{wc} = {success_rate:.1%}")
        print(f"  NaN worlds:         {n_nan}/{wc}")
        print(f"  Penetration (mean): {np.mean(max_pen) * 1000:.3f} mm")
        print(f"  Penetration (max):  {np.max(max_pen) * 1000:.3f} mm")
        print(f"  Slippage (mean):    {np.mean([r['slippage_m'] for r in results]) * 1000:.3f} mm")
        print(f"  Slippage (max):     {np.max([r['slippage_m'] for r in results]) * 1000:.3f} mm")
        print(f"  Pad force (mean max):   {np.mean(max_pad_f):.1f} N")
        print(f"  Pad force (peak):       {np.max(max_pad_f):.1f} N")
        print(f"  Table force (mean max): {np.mean(max_tbl_f):.1f} N")
        print(f"  Table force (peak):     {np.max(max_tbl_f):.1f} N")
        print(f"  Avg object velocity:    {np.mean([r['avg_vel_ms'] for r in results]) * 1000:.2f} mm/s")
        print("=" * w)

        # --- Per-state breakdown ---
        for task in TaskType:
            if task == TaskType.DONE:
                continue
            t = int(task)
            # Only print if any world spent time in this state
            if np.sum(state_count[:, t]) == 0:
                continue
            print(f"\n  PER-STATE BREAKDOWN: {task.name}")
            print(
                f"  {'W':>3}  {'Shape':>12}  "
                f"{'AvgPadF(N)':>10}  {'MaxPadF(N)':>10}  "
                f"{'AvgTblF(N)':>10}  {'MaxTblF(N)':>10}  "
                f"{'AvgPen(mm)':>10}  {'MaxPen(mm)':>10}  "
                f"{'AvgVel(mm/s)':>12}"
            )
            print("  " + "-" * 110)
            for wi in range(wc):
                cnt = int(state_count[wi, t])
                if cnt == 0:
                    continue
                shape_name = SHAPE_NAMES[self.world_shapes[wi]]
                avg_pf = state_pad_sum[wi, t] / cnt
                max_pf = state_pad_max[wi, t]
                avg_tf = state_tbl_sum[wi, t] / cnt
                max_tf = state_tbl_max[wi, t]
                avg_pen = state_pen_sum[wi, t] / cnt
                state_max_pen = state_pen_max[wi, t]
                avg_vel = state_vel_sum[wi, t] / cnt if cnt > 0 else 0.0
                print(
                    f"  {wi:3d}  {shape_name:>12s}  "
                    f"{avg_pf:10.1f}  {max_pf:10.1f}  "
                    f"{avg_tf:10.1f}  {max_tf:10.1f}  "
                    f"{avg_pen:10.2f}  {state_max_pen:10.2f}  "
                    f"{avg_vel:12.2f}"
                )

        # --- CI assertions ---
        nan_ratio = n_nan / wc if wc > 0 else 0.0
        assert nan_ratio <= 0.25, (
            f"NaN detected in {n_nan}/{wc} world(s) ({nan_ratio:.0%}), exceeds 25% tolerance: "
            f"{[r['world'] for r in results if r['has_nan']]}"
        )
        assert success_rate >= 0.50, (
            f"Success rate {success_rate:.1%} is below 50% threshold "
            f"(failed worlds: {[r['world'] for r in results if not r['success']]})"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=700)
        parser.set_defaults(world_count=12)
        parser.add_argument(
            "--collision-mode",
            type=str,
            choices=["mujoco", "newton_default", "newton_sdf", "newton_hydroelastic"],
            default="newton_hydroelastic",
            help="Collision pipeline to use",
        )
        parser.add_argument("--seed", type=int, default=42, help="RNG seed for mass/size variation")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        # Tuning parameters — remove after tuning
        parser.add_argument("--kh", type=float, default=2e11, help="Hydroelastic stiffness [Pa]")
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per frame")
        parser.add_argument("--collide-substeps", type=int, default=4, help="Collide every N substeps")
        parser.add_argument(
            "--table-half-xy",
            type=float,
            default=0.15,
            help="Table half-extent in X and Y [m] (height fixed at 0.05m). Smaller values improve SDF resolution near objects.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
