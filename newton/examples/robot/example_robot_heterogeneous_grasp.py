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
from dataclasses import dataclass, field, replace
from enum import IntEnum

import numpy as np
import warp as wp
from pxr import Usd

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

# Table dimensions. The table is centered at table_pos and extends along all
# three axes by these half-extents. _TABLE_HALF_X is large enough to support
# the Franka base at X = -0.5 plus the spawn region at X ~= 0; _TABLE_HALF_Y
# leaves a 5 cm margin around the spawn region. If you change either, also
# update the table SDF resolution in _setup_collision_sdf to keep the voxel
# size near 5 mm (the SDF builder aligns to 8-voxel chunks, so round to a
# multiple of 8).
_TABLE_HALF_X = 0.425
_TABLE_HALF_Y = 0.15
_TABLE_HEIGHT = 0.10

# Robotiq 2F-85 TCP offset: 174 mm from gripper base to fingertip center (closed).
_ROBOTIQ_TCP_OFFSET_M = 0.174
# Franka flange-to-Robotiq mount offset: 107 mm.
_PANDA_FLANGE_OFFSET_M = 0.107
# Hydroelastic stiffness for the table (Pa). Picked to dominate softer object materials
# so contact penetration is bounded by object compliance, not the table.
_TABLE_KH_PA = 1e10
# Vertical clearance between the spawn point and the object's lowest extent.
_GRASP_FLOOR_OFFSET_M = 0.0005


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

_MESH_SHAPES = frozenset(
    {
        ObjectShape.CUP,
        ObjectShape.RUBBER_DUCK,
        ObjectShape.LEGO_BRICK,
        ObjectShape.RJ45_PLUG,
        ObjectShape.BEAR,
        ObjectShape.NUT,
        ObjectShape.BOLT,
    }
)

# Builders for the primitive (non-mesh-asset) object shapes. Each entry takes the
# per-world half-size and returns a ``newton.Mesh`` -- we convert primitives to
# meshes so every world ends up with the same shape type, which the MuJoCo
# solver requires in ``separate_worlds=True`` mode.
_PRIMITIVE_MESH_FACTORIES = {
    ObjectShape.BOX: lambda hs: newton.Mesh.create_box(hs, hs, hs, compute_inertia=False),
    ObjectShape.SPHERE: lambda hs: newton.Mesh.create_sphere(hs, compute_inertia=False),
    ObjectShape.CYLINDER: lambda hs: newton.Mesh.create_cylinder(hs, hs, up_axis=newton.Axis.X, compute_inertia=False),
    ObjectShape.CAPSULE: lambda hs: newton.Mesh.create_capsule(hs, hs, up_axis=newton.Axis.X, compute_inertia=False),
    ObjectShape.ELLIPSOID: lambda hs: newton.Mesh.create_ellipsoid(hs * 2.0, hs, hs, compute_inertia=False),
}


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
            [-ox, -oy, 0],
            [+ox, -oy, 0],
            [+ox, +oy, 0],
            [-ox, +oy, 0],
            [-ox, -oy, H],
            [+ox, -oy, H],
            [+ox, +oy, H],
            [-ox, +oy, H],
            [-inx, -iny, 0],
            [+inx, -iny, 0],
            [+inx, +iny, 0],
            [-inx, +iny, 0],
            [-inx, -iny, H - T],
            [+inx, -iny, H - T],
            [+inx, +iny, H - T],
            [-inx, +iny, H - T],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [1, 2, 6],
            [1, 6, 5],
            [0, 8, 9],
            [0, 9, 1],
            [1, 9, 10],
            [1, 10, 2],
            [2, 10, 11],
            [2, 11, 3],
            [3, 11, 8],
            [3, 8, 0],
            [9, 8, 12],
            [9, 12, 13],
            [11, 10, 14],
            [11, 14, 15],
            [8, 11, 15],
            [8, 15, 12],
            [10, 9, 13],
            [10, 13, 14],
            [12, 15, 14],
            [12, 14, 13],
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
                    _LEGO_STUD_RADIUS,
                    _LEGO_STUD_HEIGHT,
                    seg,
                    cx=sx,
                    cy=sy,
                    cz=_LEGO_BRICK_HEIGHT,
                    bottom_cap=False,
                )
            )
    tube_meshes = []
    if ny == 2:
        tube_height = _LEGO_BRICK_HEIGHT - _LEGO_TOP_THICKNESS
        for i in range(nx - 1):
            tx = (i - (nx - 2) / 2.0) * _LEGO_PITCH
            tube_meshes.append(_lego_cylinder_mesh(_LEGO_TUBE_OUTER_RADIUS, tube_height, seg, cx=tx, cy=0.0, cz=0.0))
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


def zero_init(shape, dtype: type = wp.float32):
    """Allocate a Warp array filled with zeros. Thin wrapper for readability at call sites."""
    return wp.zeros(shape, dtype=dtype)


def full_init(shape, value, dtype: type = wp.float32):
    """Allocate a Warp array filled with ``value``. Thin wrapper for readability at call sites."""
    return wp.full(shape, value, dtype=dtype)


def alloc_line_buffers(n: int):
    """Allocate a paired ``(begin, end)`` ``wp.vec3`` buffer for log_lines channels."""
    return wp.zeros(n, dtype=wp.vec3), wp.zeros(n, dtype=wp.vec3)


def _print_table(title, columns, rows, *, separator="="):
    """Render a right-aligned table with auto-sized columns.

    ``columns``: ``[(header, formatter), ...]`` where ``formatter(row) -> str``.
    ``rows``: any iterable; each element is passed to every formatter.
    """
    headers = [h for h, _ in columns]
    cells = [[fmt(row) for _, fmt in columns] for row in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line_len = sum(widths) + 2 * max(0, len(widths) - 1)
    print(f"\n{separator * line_len}\n  {title}\n{separator * line_len}")
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths, strict=True)))
    print("-" * line_len)
    for row_cells in cells:
        print("  ".join(c.rjust(w) for c, w in zip(row_cells, widths, strict=True)))


def margin_pct_to_ctrl(margin_pct: float, y_half_m: float, stroke_mm: float = 85.0) -> float:
    """Convert per-shape margin_pct (fraction of object y-width) to Robotiq [0, 255] control.

    ``ctrl = 255 * (1 - (y_width - 2*margin) / stroke)``, clamped to ``[0, 255]``.
    Default stroke is the Robotiq 2F-85 full opening (85 mm).
    """
    y_width_mm = 2.0 * y_half_m * 1000.0
    margin_mm = margin_pct * y_width_mm
    ctrl = 255.0 * (1.0 - (y_width_mm - 2.0 * margin_mm) / stroke_mm)
    return min(255.0, max(0.0, ctrl))


def _quat_to_euler_zyx_deg(q: wp.quat) -> wp.vec3:
    """Convert a wp.quat to ZYX intrinsic Euler angles in degrees."""
    return 180.0 / wp.pi * wp.quat_to_rpy(q)


def _euler_zyx_deg_to_quat(rpy_deg: wp.vec3) -> wp.quat:
    """Convert ZYX intrinsic Euler angles in degrees to a wp.quat."""
    return wp.quat_from_euler(wp.pi / 180.0 * rpy_deg, 0, 1, 2)


# Per-shape vertical seed for the GUI's offset_local.z slider. Computed at init
# (see derive_offset_local_z) so the gripper starts above the table for every
# shape; the user retunes via the Apply button.
_GRASP_CLEARANCE = 0.05
_GRASP_Z_EXTRA = {
    ObjectShape.BOLT: 0.02,
    ObjectShape.BEAR: 0.01,
}


@dataclass(frozen=True)
class GraspDesign:
    """Per-shape grasp pose, authored in the object's COM frame.

    Both ``offset_local`` and ``quat_local`` are expressed in the body's COM-aligned
    frame. ``offset_local`` is in **units of the per-world object half-size** so the
    design stays size-invariant under per-world half-size jitter; the kernel
    multiplies it by ``half_size`` to recover meters. ``quat_local`` rotates the EE
    target around the body, on top of the shared ``base_ee_rot``.

    Composed at runtime as:
        grasp_pos_world = com_world + body_q.q * (offset_local * half_size)
        grasp_rot_world = body_q.q * base_ee_rot * quat_local
        grasp_ctrl      = margin_pct_to_ctrl(margin_pct, y_half)
    """

    offset_local: wp.vec3 = field(default_factory=lambda: wp.vec3(0.0, 0.0, 0.0))
    quat_local: wp.quat = field(default_factory=wp.quat_identity)
    margin_pct: float = 0.05


def derive_offset_local_z(
    shape: ObjectShape,
    half_size: float,
    z_half: float,
    grasp_clearance: float = _GRASP_CLEARANCE,
) -> float:
    """Seed offset_local.z so the EE starts at spawn_center + grasp_clearance for the shape."""
    extra = _GRASP_Z_EXTRA.get(shape, 0.0)
    return (_GRASP_FLOOR_OFFSET_M + max(0.0, 2.0 * z_half - grasp_clearance) - z_half + extra) / half_size


GRASP_DESIGNS: dict[ObjectShape, GraspDesign] = {
    ObjectShape.BOX: GraspDesign(margin_pct=0.05),
    ObjectShape.SPHERE: GraspDesign(margin_pct=0.05),
    ObjectShape.CYLINDER: GraspDesign(margin_pct=0.05),
    ObjectShape.CAPSULE: GraspDesign(margin_pct=0.05),
    ObjectShape.ELLIPSOID: GraspDesign(margin_pct=0.05),
    ObjectShape.CUP: GraspDesign(margin_pct=0.22),
    ObjectShape.RUBBER_DUCK: GraspDesign(margin_pct=0.10),
    ObjectShape.LEGO_BRICK: GraspDesign(margin_pct=0.15),
    ObjectShape.RJ45_PLUG: GraspDesign(
        offset_local=wp.vec3(0.0, 0.30, 0.0),
        quat_local=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 90.0 * wp.pi / 180.0),
        margin_pct=0.10,
    ),
    ObjectShape.BEAR: GraspDesign(margin_pct=0.25),
    ObjectShape.NUT: GraspDesign(
        quat_local=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 30.0 * wp.pi / 180.0),
        margin_pct=0.15,
    ),
    ObjectShape.BOLT: GraspDesign(margin_pct=0.30),
}


# ---- Warp kernels for IK + state machine ----


@wp.kernel(enable_backward=False)
def compute_grasp_targets(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    world_hs: wp.array[wp.float32],
    design_offset_local: wp.array[wp.vec3],
    design_quat_local: wp.array[wp.quat],
    design_ctrl: wp.array[wp.float32],
    base_ee_rot: wp.quat,
    # outputs
    grasp_pos: wp.array[wp.vec3],
    grasp_rot: wp.array[wp.quat],
    grasp_ctrl: wp.array[wp.float32],
):
    """Compute world-frame grasp target from per-shape COM-frame design.

    Rotation composition: ``body_q.q * base_ee_rot * design_quat_local[w]``.
    """
    w = wp.tid()
    obj_global = body_world_start[w] + object_body_offset
    x_wb = body_q[obj_global]
    body_tr = wp.transform_get_translation(x_wb)
    body_q_rot = wp.transform_get_rotation(x_wb)

    com_local = body_com[obj_global]
    com_world = body_tr + wp.quat_rotate(body_q_rot, com_local)

    hs_w = world_hs[w]
    offset_local = design_offset_local[w] * hs_w
    offset_world = wp.quat_rotate(body_q_rot, offset_local)

    grasp_pos[w] = com_world + offset_world
    grasp_rot[w] = body_q_rot * base_ee_rot * design_quat_local[w]
    grasp_ctrl[w] = design_ctrl[w]


@wp.kernel(enable_backward=False)
def compute_grasp_targets_slot(
    world_id: wp.int32,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    world_hs: wp.array[wp.float32],
    design_offset_local: wp.array[wp.vec3],
    design_quat_local: wp.array[wp.quat],
    design_ctrl: wp.array[wp.float32],
    base_ee_rot: wp.quat,
    # outputs (only slot world_id is written)
    grasp_pos: wp.array[wp.vec3],
    grasp_rot: wp.array[wp.quat],
    grasp_ctrl: wp.array[wp.float32],
):
    """Single-slot variant of compute_grasp_targets. Launched with dim=1."""
    w = world_id
    obj_global = body_world_start[w] + object_body_offset
    x_wb = body_q[obj_global]
    body_tr = wp.transform_get_translation(x_wb)
    body_q_rot = wp.transform_get_rotation(x_wb)
    com_local = body_com[obj_global]
    com_world = body_tr + wp.quat_rotate(body_q_rot, com_local)
    hs_w = world_hs[w]
    offset_world = wp.quat_rotate(body_q_rot, design_offset_local[w] * hs_w)
    grasp_pos[w] = com_world + offset_world
    grasp_rot[w] = body_q_rot * base_ee_rot * design_quat_local[w]
    grasp_ctrl[w] = design_ctrl[w]


@wp.func
def _write_triad(
    base: wp.int32,
    origin: wp.vec3,
    rot: wp.quat,
    axis_length: wp.float32,
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    """Write three RGB axis lines from ``origin`` rotated by ``rot``, into slots [base..base+2]."""
    ex = wp.quat_rotate(rot, wp.vec3(axis_length, 0.0, 0.0))
    ey = wp.quat_rotate(rot, wp.vec3(0.0, axis_length, 0.0))
    ez = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, axis_length))
    line_begin[base + 0] = origin
    line_end[base + 0] = origin + ex
    line_begin[base + 1] = origin
    line_end[base + 1] = origin + ey
    line_begin[base + 2] = origin
    line_end[base + 2] = origin + ez


@wp.func
def _zero_triad(
    base: wp.int32,
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    """Clear the three line segments owned by world ``base // 3``."""
    zero = wp.vec3(0.0, 0.0, 0.0)
    line_begin[base + 0] = zero
    line_end[base + 0] = zero
    line_begin[base + 1] = zero
    line_end[base + 1] = zero
    line_begin[base + 2] = zero
    line_end[base + 2] = zero


@wp.kernel(enable_backward=False)
def compute_object_frame_lines(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    world_offsets: wp.array[wp.vec3],
    visible_worlds_mask: wp.array[wp.int32],
    axis_length: wp.float32,
    # outputs
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    """Draw an XYZ triad per object body at body_q.t with body_q.q-aligned axes."""
    w = wp.tid()
    base = w * 3
    if visible_worlds_mask and visible_worlds_mask[w] == 0:
        _zero_triad(base, line_begin, line_end)
        return
    x_wb = body_q[body_world_start[w] + object_body_offset]
    origin = wp.transform_get_translation(x_wb) + world_offsets[w]
    _write_triad(base, origin, wp.transform_get_rotation(x_wb), axis_length, line_begin, line_end)


@wp.kernel(enable_backward=False)
def compute_object_com_frame_lines(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    world_offsets: wp.array[wp.vec3],
    visible_worlds_mask: wp.array[wp.int32],
    axis_length: wp.float32,
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    """Draw an XYZ triad at body_q * body_com with body_q.q-aligned axes."""
    w = wp.tid()
    base = w * 3
    if visible_worlds_mask and visible_worlds_mask[w] == 0:
        _zero_triad(base, line_begin, line_end)
        return
    obj_global = body_world_start[w] + object_body_offset
    x_wb = body_q[obj_global]
    rot = wp.transform_get_rotation(x_wb)
    com_w = wp.transform_get_translation(x_wb) + wp.quat_rotate(rot, body_com[obj_global]) + world_offsets[w]
    _write_triad(base, com_w, rot, axis_length, line_begin, line_end)


@wp.kernel(enable_backward=False)
def compute_ee_base_frame_lines(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    ee_body_offset: wp.int32,
    world_offsets: wp.array[wp.vec3],
    visible_worlds_mask: wp.array[wp.int32],
    axis_length: wp.float32,
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    """Draw an XYZ triad at the EE body's body_q.t with body_q.q-aligned axes."""
    w = wp.tid()
    base = w * 3
    if visible_worlds_mask and visible_worlds_mask[w] == 0:
        _zero_triad(base, line_begin, line_end)
        return
    x_wb = body_q[body_world_start[w] + ee_body_offset]
    origin = wp.transform_get_translation(x_wb) + world_offsets[w]
    _write_triad(base, origin, wp.transform_get_rotation(x_wb), axis_length, line_begin, line_end)


@wp.kernel(enable_backward=False)
def compute_tcp_frame_lines(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    ee_body_offset: wp.int32,
    tcp_offset: wp.float32,
    world_offsets: wp.array[wp.vec3],
    visible_worlds_mask: wp.array[wp.int32],
    axis_length: wp.float32,
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    """Draw an XYZ triad at the TCP (EE body translated by tcp_offset along its local Z)."""
    w = wp.tid()
    base = w * 3
    if visible_worlds_mask and visible_worlds_mask[w] == 0:
        _zero_triad(base, line_begin, line_end)
        return
    x_wb = body_q[body_world_start[w] + ee_body_offset]
    rot = wp.transform_get_rotation(x_wb)
    tcp_local = wp.vec3(0.0, 0.0, tcp_offset)
    origin = wp.transform_get_translation(x_wb) + wp.quat_rotate(rot, tcp_local) + world_offsets[w]
    _write_triad(base, origin, rot, axis_length, line_begin, line_end)


@wp.kernel(enable_backward=False)
def compute_world_frame_lines(
    world_offsets: wp.array[wp.vec3],
    visible_worlds_mask: wp.array[wp.int32],
    axis_length: wp.float32,
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    """Draw a world-axis-aligned XYZ triad at each world's render origin."""
    w = wp.tid()
    base = w * 3
    if visible_worlds_mask and visible_worlds_mask[w] == 0:
        _zero_triad(base, line_begin, line_end)
        return
    _write_triad(base, world_offsets[w], wp.quat_identity(), axis_length, line_begin, line_end)


@wp.kernel(enable_backward=False)
def compute_spawn_region_lines(
    world_offsets: wp.array[wp.vec3],
    visible_worlds_mask: wp.array[wp.int32],
    region_center: wp.vec3,
    half_x: wp.float32,
    half_y: wp.float32,
    # outputs: 4 line segments per world (square outline)
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    """Draw a flat square outline at the per-world object spawn region (XY range, on the table top)."""
    w = wp.tid()
    base = w * 4
    if visible_worlds_mask and visible_worlds_mask[w] == 0:
        zero = wp.vec3(0.0, 0.0, 0.0)
        for i in range(4):
            line_begin[base + i] = zero
            line_end[base + i] = zero
        return
    c = region_center + world_offsets[w]
    p_mm = wp.vec3(c[0] - half_x, c[1] - half_y, c[2])
    p_pm = wp.vec3(c[0] + half_x, c[1] - half_y, c[2])
    p_pp = wp.vec3(c[0] + half_x, c[1] + half_y, c[2])
    p_mp = wp.vec3(c[0] - half_x, c[1] + half_y, c[2])
    line_begin[base + 0] = p_mm
    line_end[base + 0] = p_pm
    line_begin[base + 1] = p_pm
    line_end[base + 1] = p_pp
    line_begin[base + 2] = p_pp
    line_end[base + 2] = p_mp
    line_begin[base + 3] = p_mp
    line_end[base + 3] = p_mm


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_idx: wp.array[wp.int32],
    task_timer: wp.array[wp.float32],
    task_durations: wp.array[wp.float32],
    grasp_pos: wp.array[wp.vec3],
    grasp_rot: wp.array[wp.quat],
    grasp_ctrl: wp.array[wp.float32],
    lift_distance_m: wp.float32,
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

    gp = grasp_pos[tid]
    gr = grasp_rot[tid]
    gc = grasp_ctrl[tid]
    lift_vec = wp.vec3(0.0, 0.0, lift_distance_m)

    if state == wp.static(int(TaskType.APPROACH)):
        target_pos = gp
        ctrl = 0.0
    elif state == wp.static(int(TaskType.CLOSE_GRIPPER)):
        target_pos = gp
        dur_close = task_durations[wp.static(int(TaskType.CLOSE_GRIPPER))]
        alpha = wp.clamp(timer / dur_close, 0.0, 1.0)
        ctrl = alpha * gc
    elif state == wp.static(int(TaskType.SETTLE)):
        target_pos = gp
        ctrl = gc
    elif state == wp.static(int(TaskType.LIFT)):
        target_pos = gp + lift_vec
        ctrl = gc
    elif state == wp.static(int(TaskType.HOLD)):
        target_pos = gp + lift_vec
        ctrl = gc
    else:
        target_pos = gp + lift_vec
        ctrl = gc

    ee_pos_target[tid] = target_pos
    # rot target: wp.vec4 because downstream consumers still take vec4
    ee_rot_target[tid] = wp.vec4(gr[0], gr[1], gr[2], gr[3])
    gripper_target[tid] = ctrl

    dur = task_durations[state] if state < wp.static(int(TaskType.DONE)) else 1.0
    t = wp.clamp(timer / dur, 0.0, 1.0)

    ee_body_idx = ee_body_global_indices[tid]
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_idx])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_idx])
    tcp_offset_local = wp.vec3(0.0, 0.0, wp.static(_ROBOTIQ_TCP_OFFSET_M))
    tcp_pos_prev = ee_pos_prev + wp.quat_rotate(ee_quat_prev, tcp_offset_local)

    ee_pos_target_interp[tid] = tcp_pos_prev * (1.0 - t) + target_pos * t
    ee_quat_interp = wp.quat_slerp(ee_quat_prev, gr, t)
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
    body_count: int,
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
    body_start = tid * body_count
    for i in range(body_count):
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
    tcp_offset_local = wp.vec3(0.0, 0.0, wp.static(_ROBOTIQ_TCP_OFFSET_M))
    ee_pos_actual[tid] = ee_pos + wp.quat_rotate(ee_quat, tcp_offset_local)


@wp.kernel(enable_backward=False)
def update_penetration_kernel(
    contact_dist: wp.array[wp.float32],  # mjw_data.contact.dist, shape (naconmax,)
    contact_worldid: wp.array[wp.int32],  # mjw_data.contact.worldid, shape (naconmax,)
    nacon: wp.array[wp.int32],  # mjw_data.nacon, shape (1,)
    world_count: wp.int32,
    penetration_cur: wp.array[wp.float32],  # shape (world_count,)
    penetration_max: wp.array[wp.float32],  # shape (world_count,)
):
    tid = wp.tid()
    if tid >= nacon[0]:
        return
    w = contact_worldid[tid]
    if w < 0 or w >= world_count:
        return
    pen = wp.max(-contact_dist[tid], 0.0)
    wp.atomic_max(penetration_cur, w, pen)
    wp.atomic_max(penetration_max, w, pen)


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
    task_state_count: wp.int32,
    hold_state: wp.int32,
    frame_count: wp.int32,
    pad_force_cur: wp.array[wp.float32],
    pad_friction_cur: wp.array[wp.float32],
    table_force_cur: wp.array[wp.float32],
    table_friction_cur: wp.array[wp.float32],
    pad_force_max: wp.array[wp.float32],
    pad_friction_max: wp.array[wp.float32],
    table_force_max: wp.array[wp.float32],
    table_friction_max: wp.array[wp.float32],
    object_z_max: wp.array[wp.float32],
    object_z_hold_start: wp.array[wp.float32],
    object_z_hold_end: wp.array[wp.float32],
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
    penetration_cur: wp.array[wp.float32],
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
    table_counterpart_count = table_force_matrix.shape[1]
    tbl_f_sum = wp.vec3(0.0, 0.0, 0.0)
    tbl_fr_sum = wp.vec3(0.0, 0.0, 0.0)
    for c in range(table_counterpart_count):
        tbl_f_sum = tbl_f_sum + table_force_matrix[w, c]
        tbl_fr_sum = tbl_fr_sum + table_force_matrix_friction[w, c]
    tf = wp.length(tbl_f_sum)
    tfr = wp.length(tbl_fr_sum)

    pad_force_cur[w] = pf
    pad_friction_cur[w] = pfr
    table_force_cur[w] = tf
    table_friction_cur[w] = tfr
    pad_force_max[w] = wp.max(pad_force_max[w], pf)
    pad_friction_max[w] = wp.max(pad_friction_max[w], pfr)
    table_force_max[w] = wp.max(table_force_max[w], tf)
    table_friction_max[w] = wp.max(table_friction_max[w], tfr)

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
        object_z_max[w] = wp.max(object_z_max[w], obj_z)

    vel_is_nan = wp.isnan(vel_mag)
    if not vel_is_nan:
        object_vel_sum[w] = object_vel_sum[w] + vel_mag
        object_vel_count[w] = object_vel_count[w] + 1

    # State-bucketed accumulation
    cur_task = task_idx[w]
    prev_task = prev_task_idx[w]
    if cur_task < task_state_count:
        t = cur_task
        pen_mm = penetration_cur[w] * 1000.0
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
            object_z_hold_start[w] = obj_z
        object_z_hold_end[w] = obj_z


@wp.kernel(enable_backward=False)
def copy_prev_task_kernel(
    task_idx: wp.array[wp.int32],
    prev_task_idx: wp.array[wp.int32],
):
    tid = wp.tid()
    prev_task_idx[tid] = task_idx[tid]


# GUI staging buffer indices (keep in sync with stage_gui_metrics_kernel below).
_GUI_STAGE_FIELDS = [
    "pad_force_cur",
    "pad_force_max",
    "pad_friction_cur",
    "pad_friction_max",
    "table_force_cur",
    "table_force_max",
    "table_friction_cur",
    "table_friction_max",
    "penetration_cur",
    "penetration_max",
    "avg_vel",
    "object_z",
    "object_z_init",
    "object_z_max",
    "task_idx",  # stored as float for uniform array
    "world_nan_frame",  # -1 sentinel -> NaN semantics preserved via float cast
    "task_timer",  # task_timer[selected_world]
    "task_dur",  # task_durations[cur_task], or 0 if cur_task >= task_duration_count
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
    pad_force_cur: wp.array[wp.float32],
    pad_force_max: wp.array[wp.float32],
    pad_friction_cur: wp.array[wp.float32],
    pad_friction_max: wp.array[wp.float32],
    table_force_cur: wp.array[wp.float32],
    table_force_max: wp.array[wp.float32],
    table_friction_cur: wp.array[wp.float32],
    table_friction_max: wp.array[wp.float32],
    penetration_cur: wp.array[wp.float32],
    penetration_max: wp.array[wp.float32],
    object_vel_sum: wp.array[wp.float32],
    object_vel_count: wp.array[wp.int32],
    object_z_init: wp.array[wp.float32],
    object_z_max: wp.array[wp.float32],
    task_idx: wp.array[wp.int32],
    world_nan_frame: wp.array[wp.int32],
    task_timer: wp.array[wp.float32],
    task_durations: wp.array[wp.float32],
    task_duration_count: wp.int32,
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_actual: wp.array[wp.vec3],
    out: wp.array[wp.float32],
):
    # dim=1 -- single-thread pack.
    # Output indices must match _GUI_STAGE_FIELDS order.
    w = selected_world
    obj_global = body_world_start[w] + object_body_offset
    obj_pos = wp.transform_get_translation(body_q[obj_global])
    out[0] = pad_force_cur[w]
    out[1] = pad_force_max[w]
    out[2] = pad_friction_cur[w]
    out[3] = pad_friction_max[w]
    out[4] = table_force_cur[w]
    out[5] = table_force_max[w]
    out[6] = table_friction_cur[w]
    out[7] = table_friction_max[w]
    out[8] = penetration_cur[w]
    out[9] = penetration_max[w]
    vel_count = object_vel_count[w]
    if vel_count > 0:
        out[10] = object_vel_sum[w] / wp.float32(vel_count)
    else:
        out[10] = 0.0
    out[11] = obj_pos[2]
    out[12] = object_z_init[w]
    out[13] = object_z_max[w]
    out[14] = wp.float32(task_idx[w])
    # world_nan_frame is an int32 frame index; float32 exactly represents
    # integers up to 2**24 (~16.7M frames), well beyond any realistic episode.
    out[15] = wp.float32(world_nan_frame[w])
    out[16] = task_timer[w]
    cur_task = task_idx[w]
    # Guard: task_idx can be >= task_duration_count (DONE state); pack 0.0 then.
    if cur_task < task_duration_count:
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
def _seed_object_z_init_kernel(
    body_q: wp.array[wp.transform],
    body_world_start: wp.array[wp.int32],
    object_body_offset: wp.int32,
    object_z_init: wp.array[wp.float32],
):
    w = wp.tid()
    obj_global = body_world_start[w] + object_body_offset
    object_z_init[w] = wp.transform_get_translation(body_q[obj_global])[2]


class GraspMetrics:
    """GPU-resident per-world + per-state metric accumulators for the grasp example.

    All state lives in Warp arrays. ``update()`` launches Warp kernels only --
    no Python-level branches on GPU data, no ``.numpy()`` calls. CPU readbacks
    happen only in ``stage_gui()`` (one selected world, throttled by the GUI)
    and ``readback_all()`` (one batched copy at episode end for ``test_final``).
    """

    def __init__(self, world_count: int, task_state_count: int, hold_state: int, object_body_offset: int):
        self.world_count = world_count
        self.task_state_count = task_state_count
        self.hold_state = hold_state
        self.object_body_offset = object_body_offset

        # Per-world running scalars
        n = world_count
        self.pad_force_cur = zero_init(n)
        self.pad_friction_cur = zero_init(n)
        self.table_force_cur = zero_init(n)
        self.table_friction_cur = zero_init(n)
        self.penetration_cur = zero_init(n)

        self.pad_force_max = zero_init(n)
        self.pad_friction_max = zero_init(n)
        self.table_force_max = zero_init(n)
        self.table_friction_max = zero_init(n)
        self.penetration_max = zero_init(n)

        self.object_z_max = full_init(n, -wp.inf)
        self.object_z_init = zero_init(n)
        self.object_z_hold_start = full_init(n, wp.nan)
        self.object_z_hold_end = full_init(n, wp.nan)
        self.object_vel_sum = zero_init(n)
        self.object_vel_count = zero_init(n, wp.int32)

        self.world_nan_frame = full_init(n, -1, wp.int32)
        self.prev_task_idx = zero_init(n, wp.int32)

        # GUI staging (pre-allocated; populated on demand).
        self.gui_staging = zero_init(_GUI_STAGE_SIZE)

        per_state = (n, task_state_count)
        self.state_pad_force_sum = zero_init(per_state)
        self.state_pad_force_max = zero_init(per_state)
        self.state_table_force_sum = zero_init(per_state)
        self.state_table_force_max = zero_init(per_state)
        self.state_pen_sum = zero_init(per_state)
        self.state_pen_max = zero_init(per_state)
        self.state_vel_sum = zero_init(per_state)
        self.state_count = zero_init(per_state, wp.int32)

    def capture_initial_object_z(self, state_0, body_world_start_array):
        wp.launch(
            _seed_object_z_init_kernel,
            dim=self.world_count,
            inputs=[state_0.body_q, body_world_start_array, self.object_body_offset],
            outputs=[self.object_z_init],
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
        self.penetration_cur.zero_()
        naconmax = mjw_data.naconmax
        if naconmax > 0:
            wp.launch(
                update_penetration_kernel,
                dim=naconmax,
                inputs=[
                    mjw_data.contact.dist,
                    mjw_data.contact.worldid,
                    mjw_data.nacon,
                    self.world_count,
                ],
                outputs=[self.penetration_cur, self.penetration_max],
            )

        wp.launch(
            update_metrics_kernel,
            dim=self.world_count,
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
                self.task_state_count,
                self.hold_state,
                frame_count,
            ],
            outputs=[
                self.pad_force_cur,
                self.pad_friction_cur,
                self.table_force_cur,
                self.table_friction_cur,
                self.pad_force_max,
                self.pad_friction_max,
                self.table_force_max,
                self.table_friction_max,
                self.object_z_max,
                self.object_z_hold_start,
                self.object_z_hold_end,
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
                self.penetration_cur,
            ],
        )

        wp.launch(copy_prev_task_kernel, dim=self.world_count, inputs=[task_idx], outputs=[self.prev_task_idx])

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
                self.pad_force_cur,
                self.pad_force_max,
                self.pad_friction_cur,
                self.pad_friction_max,
                self.table_force_cur,
                self.table_force_max,
                self.table_friction_cur,
                self.table_friction_max,
                self.penetration_cur,
                self.penetration_max,
                self.object_vel_sum,
                self.object_vel_count,
                self.object_z_init,
                self.object_z_max,
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
            "pad_force_max": self.pad_force_max.numpy(),
            "pad_friction_max": self.pad_friction_max.numpy(),
            "table_force_max": self.table_force_max.numpy(),
            "table_friction_max": self.table_friction_max.numpy(),
            "penetration_max": self.penetration_max.numpy(),
            "object_z_max": self.object_z_max.numpy(),
            "object_z_init": self.object_z_init.numpy(),
            "object_z_hold_start": self.object_z_hold_start.numpy(),
            "object_z_hold_end": self.object_z_hold_end.numpy(),
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
    arm_dof_count: int,
    gripper_dof_start: int,
    gripper_dof_count: int,
):
    """Copy IK arm solution + gripper target to joint_targets 2D array."""
    tid = wp.tid()

    # Copy arm DOFs from IK solution
    for j in range(arm_dof_count):
        joint_targets[tid, j] = ik_solution[tid, j]

    # Write gripper DOFs: all gripper joints get the same control value
    ctrl = gripper_target[tid]
    for j in range(gripper_dof_count):
        joint_targets[tid, gripper_dof_start + j] = ctrl


@wp.kernel(enable_backward=False)
def write_mujoco_ctrl_kernel(
    ik_solution: wp.array2d[wp.float32],
    gripper_target: wp.array[wp.float32],
    mujoco_ctrl: wp.array2d[wp.float32],
    arm_actuator_count: int,
    gripper_actuator_idx: int,
):
    """Write IK arm solution + gripper target to MuJoCo ctrl."""
    tid = wp.tid()
    for j in range(arm_actuator_count):
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
        self.spawn_xy_range = args.spawn_xy_range
        self.spawn_yaw_range_rad = args.spawn_yaw_range * wp.pi / 180.0
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
            actuator_count = ctrl.shape[0] // self.world_count
            self.mujoco_ctrl_2d = ctrl.reshape((self.world_count, actuator_count))
            # Gripper actuator is the last one (after 7 arm actuators)
            self.gripper_actuator_idx = self.arm_dof_count
            print(f"MuJoCo ctrl: {actuator_count} actuators/world, gripper at idx {self.gripper_actuator_idx}")

            # Initialize ctrl to current arm joint positions so arm holds its pose
            init_q = self.model.joint_q.numpy()
            ctrl_np = ctrl.numpy().reshape(self.world_count, actuator_count)
            dofs_per_world = self.model.joint_coord_count // self.world_count
            for w in range(self.world_count):
                q_start = w * dofs_per_world
                for j in range(self.arm_dof_count):
                    ctrl_np[w, j] = init_q[q_start + j]
            wp.copy(ctrl, wp.array(ctrl_np.flatten(), dtype=wp.float32))

        self._setup_contact_sensor()
        self._create_collision_pipeline()
        self._create_solver()
        self._create_sensor_contacts()
        self._setup_metrics()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.5, 0.5, 0.5), -15, -140)
        self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True
        self._setup_gui()
        self.capture()

    # ------------------------------------------------------------------
    # Public API: capture / simulate / step / render / test_final / create_parser
    # ------------------------------------------------------------------

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
        self._set_joint_targets()
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

    def test_final(self):
        lift_threshold_m = 0.05  # 50 mm lift required for success
        m = self.metrics.readback_all()
        N = self.world_count

        # Vectorized per-world derived metrics. Builds the columns the report consumes
        # without any Python-level loop.
        body_ws = self.model.body_world_start.numpy()[:N].astype(np.int64)
        mass = self.model.body_mass.numpy()[body_ws + self.object_body_offset]
        shape_names = np.array([SHAPE_NAMES[s] for s in self.world_shapes])
        half_size_mm = np.asarray(self.world_half_sizes, dtype=np.float32) * 1000.0
        has_nan = m["world_nan_frame"] >= 0
        nan_frame = m["world_nan_frame"]
        lift_mm = np.where(has_nan, 0.0, m["object_z_max"] - m["object_z_init"]) * 1000.0
        hs, he = m["object_z_hold_start"], m["object_z_hold_end"]
        slip_mm = np.where(np.isnan(hs) | np.isnan(he), 0.0, hs - he) * 1000.0
        pen_mm = m["penetration_max"] * 1000.0
        pad_f, pad_fr = m["pad_force_max"], m["pad_friction_max"]
        tbl_f = m["table_force_max"]
        avg_vel_mmps = m["object_vel_sum"] / np.maximum(1, m["object_vel_count"]) * 1000.0
        success = (lift_mm > lift_threshold_m * 1000.0) & ~has_nan

        # --- Per-world results table ---
        per_world_cols = [
            ("W", lambda i: f"{i:d}"),
            ("Shape", lambda i: shape_names[i]),
            ("Mass(kg)", lambda i: f"{mass[i]:.4f}"),
            ("HS(mm)", lambda i: f"{half_size_mm[i]:.1f}"),
            ("OK", lambda i: "YES" if success[i] else "NO"),
            ("Lift(mm)", lambda i: f"{lift_mm[i]:.1f}"),
            ("Slip(mm)", lambda i: f"{slip_mm[i]:.1f}"),
            ("Pen(mm)", lambda i: f"{pen_mm[i]:.2f}"),
            ("PadF(N)", lambda i: f"{pad_f[i]:.1f}"),
            ("PadFr(N)", lambda i: f"{pad_fr[i]:.1f}"),
            ("TblF(N)", lambda i: f"{tbl_f[i]:.1f}"),
            ("Vel(mm/s)", lambda i: f"{avg_vel_mmps[i]:.2f}"),
            ("NaN", lambda i: str(int(nan_frame[i])) if has_nan[i] else "-"),
        ]
        _print_table(
            f"HETEROGENEOUS GRASP TEST REPORT  (collision_mode={self.collision_mode.name})",
            per_world_cols,
            range(N),
        )

        # --- Per-shape aggregation ---
        shapes_present = [s for s in ObjectShape if (shape_names == s.name).any()]
        per_shape_cols = [
            ("Shape", lambda s: s.name),
            ("Count", lambda s: f"{int((shape_names == s.name).sum())}"),
            ("Success", lambda s: f"{int(success[shape_names == s.name].sum())}/{int((shape_names == s.name).sum())}"),
            ("Lift(mm)", lambda s: f"{float(np.mean(lift_mm[shape_names == s.name])):.1f}"),
            ("Slip(mm)", lambda s: f"{float(np.mean(slip_mm[shape_names == s.name])):.1f}"),
            ("PadF(N)", lambda s: f"{float(np.mean(pad_f[shape_names == s.name])):.1f}"),
            ("TblF(N)", lambda s: f"{float(np.mean(tbl_f[shape_names == s.name])):.1f}"),
            ("Vel(mm/s)", lambda s: f"{float(np.mean(avg_vel_mmps[shape_names == s.name])):.2f}"),
        ]
        _print_table("PER-SHAPE AGGREGATION", per_shape_cols, shapes_present, separator="-")

        # --- Aggregate statistics ---
        n_success = int(success.sum())
        nan_world_count = int(has_nan.sum())
        success_rate = n_success / N if N > 0 else 0.0
        print("\n" + "-" * 80)
        print("  AGGREGATE STATISTICS")
        print("-" * 80)
        print(f"  Success rate:       {n_success}/{N} = {success_rate:.1%}")
        print(f"  NaN worlds:         {nan_world_count}/{N}")
        print(f"  Penetration:        mean {pen_mm.mean():.3f} mm  /  max {pen_mm.max():.3f} mm")
        print(f"  Slippage:           mean {slip_mm.mean():.3f} mm  /  max {slip_mm.max():.3f} mm")
        print(f"  Pad force (max):    mean {pad_f.mean():.1f} N  /  peak {pad_f.max():.1f} N")
        print(f"  Table force (max):  mean {tbl_f.mean():.1f} N  /  peak {tbl_f.max():.1f} N")
        print(f"  Avg object velocity: {avg_vel_mmps.mean():.2f} mm/s")

        # --- Per-state breakdown (one table per task that any world entered) ---
        state_count_arr = m["state_count"]
        state_counts_safe = np.maximum(1, state_count_arr)
        state_pad_avg = m["state_pad_force_sum"] / state_counts_safe
        state_tbl_avg = m["state_table_force_sum"] / state_counts_safe
        state_pen_avg = m["state_pen_sum"] / state_counts_safe
        state_vel_avg = m["state_vel_sum"] / state_counts_safe
        state_pad_max = m["state_pad_force_max"]
        state_tbl_max = m["state_table_force_max"]
        state_pen_max = m["state_pen_max"]
        for task in TaskType:
            t = int(task)
            if task == TaskType.DONE or state_count_arr[:, t].sum() == 0:
                continue
            active = [i for i in range(N) if state_count_arr[i, t] > 0]
            cols = [
                ("W", lambda i: f"{i:d}"),
                ("Shape", lambda i: shape_names[i]),
                ("AvgPadF(N)", lambda i, t=t: f"{state_pad_avg[i, t]:.1f}"),
                ("MaxPadF(N)", lambda i, t=t: f"{state_pad_max[i, t]:.1f}"),
                ("AvgTblF(N)", lambda i, t=t: f"{state_tbl_avg[i, t]:.1f}"),
                ("MaxTblF(N)", lambda i, t=t: f"{state_tbl_max[i, t]:.1f}"),
                ("AvgPen(mm)", lambda i, t=t: f"{state_pen_avg[i, t]:.2f}"),
                ("MaxPen(mm)", lambda i, t=t: f"{state_pen_max[i, t]:.2f}"),
                ("AvgVel(mm/s)", lambda i, t=t: f"{state_vel_avg[i, t]:.2f}"),
            ]
            _print_table(f"PER-STATE BREAKDOWN: {task.name}", cols, active, separator="-")

        # --- CI assertions ---
        nan_ratio = nan_world_count / N if N > 0 else 0.0
        nan_worlds = [int(i) for i in np.where(has_nan)[0]]
        failed_worlds = [int(i) for i in np.where(~success)[0]]
        assert nan_ratio <= 0.25, (
            f"NaN detected in {nan_world_count}/{N} world(s) ({nan_ratio:.0%}), exceeds 25% tolerance: {nan_worlds}"
        )
        assert success_rate >= 0.50, (
            f"Success rate {success_rate:.1%} is below 50% threshold (failed worlds: {failed_worlds})"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=700)
        parser.set_defaults(world_count=24)
        parser.add_argument(
            "--collision-mode",
            type=str,
            choices=["mujoco", "newton_default", "newton_sdf", "newton_hydroelastic"],
            default="newton_hydroelastic",
            help="Collision pipeline to use",
        )
        parser.add_argument("--seed", type=int, default=42, help="RNG seed for mass/size variation")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument(
            "--spawn-xy-range",
            type=float,
            default=0.10,
            help="Half-range (m) of the uniform XY spawn-position randomization on the table. 0 disables.",
        )
        parser.add_argument(
            "--spawn-yaw-range",
            type=float,
            default=30.0,
            help="Half-range (deg) of the uniform Z-rotation spawn randomization. 0 disables.",
        )
        parser.add_argument("--kh", type=float, default=2e11, help="Hydroelastic stiffness [Pa]")
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per frame")
        parser.add_argument("--collide-substeps", type=int, default=4, help="Collide every N substeps")
        return parser

    # ------------------------------------------------------------------
    # Setup / build
    # ------------------------------------------------------------------

    def _download_assets(self):
        """Download menagerie assets once and cache paths."""
        if hasattr(self, "_franka_dir"):
            return
        self._franka_dir = newton.examples.download_external_git_folder(
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            "franka_emika_panda",
        )
        self._robotiq_dir = newton.examples.download_external_git_folder(
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
        # Scene centred on the world-Y axis: table at (0, 0), robot 0.5m behind it in -X.
        self.table_pos = wp.vec3(-0.275, 0.0, _TABLE_HEIGHT / 2.0)  # table body center, shifted under the robot
        self.spawn_center = wp.vec3(0.0, 0.0, _TABLE_HEIGHT)  # object spawn anchor on the table top
        self.robot_base_pos = wp.vec3(-0.5, 0.0, _TABLE_HEIGHT)  # on table, 0.5m behind center

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
        # Attach Robotiq to link7 (last link with mass), applying the attachment body's
        # transform plus an extra 90° Z rotation for better jaw alignment. The attachment
        # body in panda_nohand.xml is pos="0 0 0.107" quat="0.3826834 0 0 0.9238795";
        # MJCF stores quats (w, x, y, z) so we re-order to Warp's (x, y, z, w) below.
        q_attach = wp.quat(0.0, 0.0, 0.9238795, 0.3826834)
        q_90z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        builder.add_mjcf(
            str(self._robotiq_dir / "2f85.xml"),
            parent_body=find_body("link7"),
            xform=wp.transform(wp.vec3(0.0, 0.0, _PANDA_FLANGE_OFFSET_M), q_attach * q_90z),
            enable_self_collisions=False,
        )

        # Track key body indices.
        # The attachment body from panda_nohand.xml has zero mass/inertia;
        # give it a tiny mass so the inertia validator doesn't warn.
        self.ee_attachment_body_idx = find_body("attachment")
        builder.body_mass[self.ee_attachment_body_idx] = 1e-6
        builder.body_inertia[self.ee_attachment_body_idx] = wp.mat33(np.eye(3, dtype=np.float32) * 1e-9)
        franka_body_count = self.ee_attachment_body_idx + 1
        self.ee_base_body_idx = next(
            i for i in range(franka_body_count, builder.body_count) if builder.body_label[i].endswith("/base")
        )

        # --- Joint configuration ---
        # panda_nohand: 7 arm DOFs, no finger joints
        self.arm_dof_count = 7
        self.gripper_dof_start = self.arm_dof_count  # Robotiq DOFs start right after arm
        self.gripper_dof_count = builder.joint_coord_count - self.arm_dof_count

        print(
            f"Joint layout: {self.arm_dof_count} arm + "
            f"{self.gripper_dof_count} gripper = {builder.joint_coord_count} total DOFs"
        )
        print(f"Gripper DOF range: [{self.gripper_dof_start}, {self.gripper_dof_start + self.gripper_dof_count})")

        # Gripper armature scaling for hydroelastic stability (from example_hydro_robotiq_gripper)
        # The MJCF default armature is too small for hydroelastic contact stiffness.
        if self.gripper_dof_count == 6:
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
        for dof_idx in range(self.arm_dof_count):
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
        n = self.world_count

        # Round-robin shape assignment across all ObjectShape values
        self.world_shapes = [ObjectShape(i % NUM_SHAPES) for i in range(n)]

        # Fixed density (1000 kg/m^3 ~= water): mass scales with shape volume and size,
        # so we don't have to combine an unrealistic density with the per-world size jitter.
        self.object_density = 1000.0

        # Per-world object half-size with +/- 25% uniform jitter around the base.
        base_half_size = 0.025
        self.world_half_sizes = base_half_size * rng.uniform(0.75, 1.25, size=n)

        # Per-world spawn pose randomization: XY offset on the table and Z-yaw rotation.
        # Both draws are uniform in [-range, +range]; ranges of 0 collapse to deterministic
        # spawns at the table center with the per-shape default orientation.
        self._world_spawn_xy = rng.uniform(-self.spawn_xy_range, self.spawn_xy_range, size=(n, 2)).astype(np.float32)
        self._world_spawn_yaw = rng.uniform(-self.spawn_yaw_range_rad, self.spawn_yaw_range_rad, size=n).astype(
            np.float32
        )

        if self.verbose:
            for i in range(n):
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
            asset_path = newton.examples.download_external_git_folder(_NUT_BOLT_REPO_URL, _NUT_BOLT_FOLDER)
            if ObjectShape.NUT in needed:
                nut_file = str(asset_path / f"factory_nut_{_NUT_BOLT_ASSEMBLY}_subdiv_3x.obj")
                self.mesh_objects[ObjectShape.NUT] = _load_obj_mesh_trimesh(nut_file)
            if ObjectShape.BOLT in needed:
                bolt_file = str(asset_path / f"factory_bolt_{_NUT_BOLT_ASSEMBLY}.obj")
                self.mesh_objects[ObjectShape.BOLT] = _load_obj_mesh_trimesh(bolt_file)

    def _add_object(self, builder: newton.ModelBuilder, world_id: int):
        """Add a grasp object to the builder for one world."""
        shape = self.world_shapes[world_id]
        half_size = self.world_half_sizes[world_id]
        mesh = self.mesh_objects[shape] if shape in _MESH_SHAPES else _PRIMITIVE_MESH_FACTORIES[shape](half_size)

        if shape in _MESH_SHAPES:
            extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
            uniform_scale = 2.0 * half_size / extents.max() if extents.max() > 0 else 1.0
            # RJ45_PLUG is spawned with a 90° Z-yaw so its body-frame Y is the mesh's X-extent.
            y_extent = extents[0] if shape == ObjectShape.RJ45_PLUG else extents[1]
            self._world_y_half[world_id] = y_extent / 2.0 * uniform_scale
            self._world_z_half[world_id] = extents[2] / 2.0 * uniform_scale
        else:
            uniform_scale = 1.0
            self._world_y_half[world_id] = half_size
            self._world_z_half[world_id] = half_size
        scale = wp.vec3(uniform_scale, uniform_scale, uniform_scale)

        z_axis = wp.vec3(0.0, 0.0, 1.0)
        obj_rot = wp.quat_from_axis_angle(z_axis, wp.pi / 2.0) if shape == ObjectShape.RJ45_PLUG else wp.quat_identity()
        # Per-world spawn randomization: XY offset + Z-yaw composed onto obj_rot.
        sx, sy, sz = self.spawn_center
        dx, dy = self._world_spawn_xy[world_id]
        yaw_rot = wp.quat_from_axis_angle(z_axis, self._world_spawn_yaw[world_id])
        obj_z = sz + self._world_z_half[world_id] + _GRASP_FLOOR_OFFSET_M
        obj_body = builder.add_body(
            xform=wp.transform(wp.vec3(sx + dx, sy + dy, obj_z), yaw_rot * obj_rot),
            label="object",
        )

        obj_cfg = replace(self.shape_cfg, density=self.object_density)
        builder.add_shape_mesh(
            body=obj_body,
            mesh=mesh,
            scale=scale,
            cfg=obj_cfg,
            label=f"object_shape_{shape.name.lower()}",
        )

        # The LEGO brick gets explicit floor + four wall colliders inside its hollow shell;
        # other shapes get five tiny placeholder boxes so every world has identical body
        # topology (required by MuJoCo's separate_worlds=True mode).
        collider_cfg = replace(self.shape_cfg, density=0.0, is_visible=False)
        if shape == ObjectShape.LEGO_BRICK:
            sf = uniform_scale
            inset = 0.0001 * sf
            ox = _LEGO_PITCH * 2 * sf
            oy = _LEGO_PITCH * sf
            center_z = (_LEGO_BRICK_HEIGHT + _LEGO_STUD_HEIGHT) / 2.0 * sf
            box_hz = 0.5 * _LEGO_BRICK_HEIGHT * sf - inset
            box_cz = 0.5 * _LEGO_BRICK_HEIGHT * sf - center_z
            wt = _LEGO_WALL_THICKNESS * sf
            wall_hx = 0.5 * wt - inset
            wall_hy = 0.5 * wt - inset
            specs = [
                ((0.0, 0.0, box_cz), ox - inset, oy - inset, box_hz),
                ((ox - 0.5 * wt, 0.0, box_cz), wall_hx, oy - inset, box_hz),
                ((-(ox - 0.5 * wt), 0.0, box_cz), wall_hx, oy - inset, box_hz),
                ((0.0, oy - 0.5 * wt, box_cz), ox - inset, wall_hy, box_hz),
                ((0.0, -(oy - 0.5 * wt), box_cz), ox - inset, wall_hy, box_hz),
            ]
        else:
            t = 1.0e-4
            specs = [((0.0, 0.0, 0.0), t, t, t)] * 5
        for pos, hx, hy, hz in specs:
            builder.add_shape_box(
                body=obj_body,
                hx=hx,
                hy=hy,
                hz=hz,
                xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
                cfg=collider_cfg,
                label="object_collider",
            )

        return obj_body

    def _build_scene(self, robot_builder):
        """Build the multi-world scene with per-world robot + table + object."""
        # Table as kinematic body (mass=0) so MuJoCo detects collisions per world.
        # Using body=-1 (static) causes MuJoCo to miss collisions for replicated worlds.
        table_mesh = newton.Mesh.create_box(_TABLE_HALF_X, _TABLE_HALF_Y, _TABLE_HEIGHT / 2.0, compute_inertia=False)
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
                color=(0.20, 0.20, 0.22),
                label="table_shape",
            )

            # Object on table surface at (0, 0, spawn_center.z + hs) plus the per-world spawn jitter
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

        # link_offset maps IK targets to the TCP, so grasp_pos values land at the
        # fingertip center rather than the gripper base. init target must be the
        # TCP position (not EE body) so the initial config is already a perfect
        # solution and the IK doesn't drift.
        tcp_offset_vec = wp.vec3(0.0, 0.0, _ROBOTIQ_TCP_OFFSET_M)
        init_tcp_pos = init_ee_pos + wp.quat_rotate(init_ee_rot, tcp_offset_vec)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_ee_index,
            link_offset=tcp_offset_vec,
            target_positions=wp.array([init_tcp_pos] * self.world_count, dtype=wp.vec3),
        )

        # Rotation target: use the arm-only model's attachment body rotation.
        # The Robotiq is rigidly mounted with an extra 90° Z rotation, but
        # the IK only controls the arm — it should keep the attachment body
        # at its initial orientation.
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ik_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([list(init_ee_rot)] * self.world_count, dtype=wp.vec4),
        )

        # Joint limit objective — replicate single-model limits for all problems
        ik_dofs = self.model_arm_only.joint_coord_count
        ll = self.model_arm_only.joint_limit_lower.numpy()
        lu = self.model_arm_only.joint_limit_upper.numpy()
        joint_limit_lower = wp.array(np.tile(ll, self.world_count), dtype=wp.float32)
        joint_limit_upper = wp.array(np.tile(lu, self.world_count), dtype=wp.float32)
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=joint_limit_lower.flatten(),
            joint_limit_upper=joint_limit_upper.flatten(),
        )

        # IK solution array: (world_count, joint_coord_count)
        # Initialize from the multi-world joint_q, taking each world's slice
        joint_q_np = self.model.joint_q.numpy().reshape(self.world_count, -1)[:, :ik_dofs]
        self.joint_q_ik = wp.array(joint_q_np, dtype=wp.float32)

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_arm_only,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _setup_state_machine(self):
        """Initialize per-world state machine arrays and grasp parameters."""
        # Task state: index and timer per world
        self.task_idx = wp.zeros(self.world_count, dtype=wp.int32)
        self.task_timer = wp.zeros(self.world_count, dtype=wp.float32)

        # Task durations: [APPROACH, CLOSE_GRIPPER, SETTLE, LIFT, HOLD, DONE]
        durations = [1.5, 1.0, 0.5, 1.0, 1.5, 0.0]
        self.task_durations = wp.array(durations, dtype=wp.float32)

        # EE rotation target: use the arm-only model's attachment body rotation.
        # This matches the IK rotation objective so there's no mismatch.
        state_arm = self.model_arm_only.state()
        newton.eval_fk(self.model_arm_only, self.model_arm_only.joint_q, self.model_arm_only.joint_qd, state_arm)
        arm_ee_rot = wp.transform_get_rotation(wp.transform(*state_arm.body_q.numpy()[self.ik_ee_index]))

        # --- Object-centric grasp: per-world design inputs + SoA runtime buffers ---
        # Pre-allocated once; the Apply path reuses them in place.
        self._shape_mask: dict[ObjectShape, np.ndarray] = {
            shape: np.where(np.asarray(self.world_shapes) == shape)[0].astype(np.int32)
            for shape in set(self.world_shapes)
        }

        # CPU mirrors
        self._design_offset_local_np = np.zeros((self.world_count, 3), dtype=np.float32)
        self._design_quat_local_np = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (self.world_count, 1))
        self._design_ctrl_np = np.zeros(self.world_count, dtype=np.float32)

        # Fill per-world from GRASP_DESIGNS, deriving offset_local.z per shape
        for i, shape in enumerate(self.world_shapes):
            design = GRASP_DESIGNS[shape]
            z_local = derive_offset_local_z(shape, half_size=self.world_half_sizes[i], z_half=self._world_z_half[i])
            self._design_offset_local_np[i] = (*design.offset_local[:2], z_local)
            self._design_quat_local_np[i] = design.quat_local
            self._design_ctrl_np[i] = margin_pct_to_ctrl(design.margin_pct, y_half_m=self._world_y_half[i])

        # GPU design inputs
        self._design_offset_local = wp.array(self._design_offset_local_np, dtype=wp.vec3)
        self._design_quat_local = wp.array(self._design_quat_local_np, dtype=wp.quat)
        self._design_ctrl = wp.array(self._design_ctrl_np, dtype=wp.float32)

        # Single-slot upload sources reused on every Apply call (no per-Apply allocation).
        self._single_offset_local_src = wp.zeros(1, dtype=wp.vec3)
        self._single_quat_local_src = wp.zeros(1, dtype=wp.quat)
        self._single_ctrl_src = wp.zeros(1, dtype=wp.float32)

        # GPU runtime SoA outputs
        self.grasp_pos = wp.zeros(self.world_count, dtype=wp.vec3)
        self.grasp_rot = wp.zeros(self.world_count, dtype=wp.quat)
        self.grasp_ctrl = wp.zeros(self.world_count, dtype=wp.float32)

        # Per-world half-size array for the init kernel
        self._world_half_size_array = wp.array(np.asarray(self.world_half_sizes, dtype=np.float32), dtype=wp.float32)

        # base_ee_rot as a wp.quat scalar (used by the init kernel)
        self.base_ee_rot = wp.quat(*arm_ee_rot)

        if self.verbose:
            body_com_np = self.model.body_com.numpy()
            body_ws_np = self.model.body_world_start.numpy()
            first_world = {shape: idx for idx, shape in reversed(list(enumerate(self.world_shapes)))}
            print("[grasp] per-shape body_com magnitudes (body-local frame):")
            for shape, idx in sorted(first_world.items(), key=lambda kv: kv[0].name):
                com_norm = np.linalg.norm(body_com_np[int(body_ws_np[idx]) + self.object_body_offset])
                flag = "  (non-zero, review offset_local)" if com_norm > 1e-4 else ""
                print(f"  {shape.name:<12} |body_com| = {com_norm * 1000.0:8.3f} mm{flag}")

        self.lift_distance_m: float = 0.1  # tunable via the GUI lift slider

        # state_0 is fully initialised by the time _setup_state_machine runs.
        wp.launch(
            compute_grasp_targets,
            dim=self.world_count,
            inputs=[
                self.state_0.body_q,
                self.model.body_com,
                self.model.body_world_start,
                self.object_body_offset,
                self._world_half_size_array,
                self._design_offset_local,
                self._design_quat_local,
                self._design_ctrl,
                self.base_ee_rot,
            ],
            outputs=[self.grasp_pos, self.grasp_rot, self.grasp_ctrl],
        )

        # Target arrays (raw and interpolated)
        self.ee_pos_target = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(self.world_count, dtype=wp.vec4)
        self.ee_pos_target_interp = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_rot_target_interp = wp.zeros(self.world_count, dtype=wp.vec4)
        self.gripper_target = wp.zeros(self.world_count, dtype=wp.float32)

        # Actual EE position (extracted from body_q each frame)
        self.ee_pos_actual = wp.zeros(self.world_count, dtype=wp.vec3)

        # Global body indices for EE in each world. Using the attachment body (not the
        # Robotiq base) keeps EE tracking aligned with the IK model's EE link.
        body_ws = self.model.body_world_start.numpy()[: self.world_count].astype(np.int32)
        self.ee_body_global_indices = wp.array(body_ws + self.ee_attachment_body_idx, dtype=wp.int32)

        # Snapshot of body_q at the start of each task (for interpolation)
        # Initialize with current state_0 body_q
        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.body_count_total = self.model.body_count

    def _setup_collision_sdf(self, builder):
        """Build SDFs on all collision shapes; mark fingertips, objects, and table hydroelastic.

        Operates on the scene builder BEFORE finalize(), following the same
        pattern as ``example_hydro_robotiq_gripper``:
        Pass 1 - SDF on every collision shape (BOX -> MESH + SDF build).
        Pass 2 - HYDROELASTIC flag on fingertip pads, object shapes, and the table
        (so the contact-surface visualization always has at least one pair to draw).
        """
        if self.collision_mode not in (CollisionMode.NEWTON_SDF, CollisionMode.NEWTON_HYDROELASTIC):
            return

        sdf_narrow_band = (-0.0015, 0.0015)
        hydroelastic_enabled = self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC

        # ---- Pass 1: build SDF on every collision shape ----
        # Object shapes get higher resolution (96) for better contact quality;
        # robot links, gripper, and table use 64.
        for shape_idx in range(builder.shape_count):
            if not (builder.shape_flags[shape_idx] & newton.ShapeFlags.COLLIDE_SHAPES):
                continue

            label = builder.shape_label[shape_idx] if shape_idx < len(builder.shape_label) else ""
            is_object = "object" in label
            is_table = "table" in label
            if is_object:
                sdf_max_res = 96
            elif is_table:
                # 0.85 m table / ~5 mm voxel target, rounded up to a multiple of 8 (tile-aligned).
                sdf_max_res = 176
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

        # ---- Pass 2: fingertip, object, and table shapes get hydroelastic flag ----
        if hydroelastic_enabled:
            fingertip_names = {
                "left_pad1",
                "left_pad2",
                "right_pad1",
                "right_pad2",
                "left_follower_geom_1",
                "right_follower_geom_0",
            }

            hydro_count = 0
            for shape_idx, label in enumerate(builder.shape_label):
                short = label.split("/")[-1] if label else ""
                is_fingertip = short in fingertip_names
                is_object = "object" in short
                is_table = "table" in short

                if is_fingertip or is_object or is_table:
                    cfg = self.shape_cfg
                    builder.shape_gap[shape_idx] = cfg.gap
                    builder.shape_material_mu[shape_idx] = cfg.mu
                    builder.shape_material_mu_torsional[shape_idx] = cfg.mu_torsional
                    builder.shape_material_mu_rolling[shape_idx] = cfg.mu_rolling
                    if is_table:
                        builder.shape_material_kh[shape_idx] = _TABLE_KH_PA
                    else:
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
            iterations=100,
            ls_iterations=200,
            impratio=50.0,
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
            task_state_count=nt,
            hold_state=int(TaskType.HOLD),
            object_body_offset=self.object_body_offset,
        )
        self.metrics.capture_initial_object_z(self.state_0, self.model.body_world_start)

    def _setup_gui(self):
        """Register the side-panel GUI callback with the viewer."""
        self.selected_world = 0
        self.show_isosurface = False
        # Frame-visualization toggles. Each draws 3 lines per world (X/Y/Z axes)
        # with body_q-aligned axes; disambiguate by position.
        self.show_object_frames = False
        self.show_object_com_frames = False
        self.show_ee_base_frames = False
        self.show_tcp_frames = False
        self.show_world_frames = False
        self.show_spawn_region = False
        # Pre-allocated line buffers, one (begin, end) pair per channel.
        # 3 segments per world for triads, 4 for the spawn-region square.
        n3 = 3 * self.world_count
        n4 = 4 * self.world_count
        self._object_frame_begin, self._object_frame_end = alloc_line_buffers(n3)
        self._object_com_frame_begin, self._object_com_frame_end = alloc_line_buffers(n3)
        self._ee_base_frame_begin, self._ee_base_frame_end = alloc_line_buffers(n3)
        self._tcp_frame_begin, self._tcp_frame_end = alloc_line_buffers(n3)
        self._world_frame_begin, self._world_frame_end = alloc_line_buffers(n3)
        self._spawn_region_begin, self._spawn_region_end = alloc_line_buffers(n4)
        self._spawn_region_colors = wp.array(
            np.tile(np.array([1.0, 0.85, 0.0], dtype=np.float32), (n4, 1)),
            dtype=wp.vec3,
        )
        # Per-axis colors: red (X), green (Y), blue (Z), tiled per-world.
        # Shared across all five frame channels.
        colors_np = np.tile(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            (self.world_count, 1),
        )
        self._frame_colors = wp.array(colors_np, dtype=wp.vec3)
        # Zero-offset fallback used when the viewer doesn't expose world_offsets.
        self._zero_world_offsets = wp.zeros(self.world_count, dtype=wp.vec3)
        # Static mass/world-start mirrors used by the per-frame GUI text.
        self._cached_body_mass = self.model.body_mass.numpy().copy()
        self._cached_body_world_start = self.model.body_world_start.numpy().copy()
        # GUI readback staging cache (populated by stage_gui; read downstream).
        self._cached_gui = np.zeros(_GUI_STAGE_SIZE, dtype=np.float32)
        self._gui_read_interval = 10
        # Per-shape GUI edit buffer. Seeded from the actual per-world design buffer
        # so the slider reflects what the simulation is currently using -- particularly
        # offset_local.z, which _setup_state_machine baked from derive_offset_local_z and
        # is non-zero even though the GRASP_DESIGNS placeholder is (0, 0, 0).
        self._gui_grasp_edits: dict[ObjectShape, dict] = {}
        for shape, design in GRASP_DESIGNS.items():
            mask = self._shape_mask.get(shape)
            if mask is None or len(mask) == 0:
                seed_offset = list(design.offset_local)
            else:
                seed_offset = self._design_offset_local_np[int(mask[0])].tolist()
            self._gui_grasp_edits[shape] = {
                "offset_local": seed_offset,
                "euler_deg": list(_quat_to_euler_zyx_deg(design.quat_local)),
                "margin_pct": design.margin_pct,
            }
        self._gui_lift_mm = int(round(self.lift_distance_m * 1000.0))
        self._gui_broadcast_apply = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = self.show_isosurface
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._gui_impl, position="side")

    # ------------------------------------------------------------------
    # Per-step / per-frame internals
    # ------------------------------------------------------------------

    def _gui_impl(self, imgui):
        # ---- World selector: shape filter + slider ----
        if not hasattr(self, "_gui_shape_filter"):
            self._gui_shape_filter = -1  # -1 = All

        shape_keys: list[ObjectShape] = sorted(self._shape_mask.keys(), key=lambda s: s.name)
        shape_names = ["All"] + [s.name for s in shape_keys]

        # combo_idx 0 -> "All" (filter == -1); 1..N -> shape_keys[0..N-1] (filter == 0..N-1)
        changed, new_idx = imgui.combo("Filter by shape", self._gui_shape_filter + 1, shape_names)
        if changed:
            self._gui_shape_filter = new_idx - 1

        if self._gui_shape_filter >= 0:
            filter_shape = shape_keys[self._gui_shape_filter]
            candidates = self._shape_mask[filter_shape]
            if len(candidates) > 0 and self.selected_world not in candidates:
                self.selected_world = int(candidates[0])

        changed, val = imgui.slider_int("World", self.selected_world, 0, self.world_count - 1)
        if changed:
            self.selected_world = max(0, min(self.world_count - 1, val))
        w = self.selected_world

        imgui.separator()

        # Object description (static metadata -- no per-frame GPU read needed).
        shape_name = SHAPE_NAMES[self.world_shapes[w]]
        hs_mm = self.world_half_sizes[w] * 1000.0
        obj_global = int(self._cached_body_world_start[w]) + self.object_body_offset
        mass = self._cached_body_mass[obj_global]
        imgui.text(f"Shape: {shape_name}")
        imgui.text(f"Mass:  {mass:.4f} kg")
        imgui.text(f"Size:  {hs_mm:.1f} mm (half-size)")

        imgui.separator()

        # Throttled GPU->CPU sync. Pack selected-world metrics into a 24-float
        # staging buffer via a single-thread Warp kernel; read back once.
        if (self.episode_steps % self._gui_read_interval) == 0:
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
        gui = self._cached_gui

        cur_pad_f = gui[0]
        max_pad_f = gui[1]
        cur_pad_fr = gui[2]
        max_pad_fr = gui[3]
        cur_tbl_f = gui[4]
        max_tbl_f = gui[5]
        cur_pen_mm = gui[8] * 1000.0
        max_pen_mm = gui[9] * 1000.0
        avg_vel_mms = gui[10] * 1000.0
        obj_z = gui[11]
        init_z = gui[12]
        max_z = gui[13]
        task_val = int(gui[14])
        nan_frame_val = int(gui[15])
        cur_timer = gui[16]
        task_dur = gui[17]
        ee_target = (gui[18], gui[19], gui[20])
        ee_actual = (gui[21], gui[22], gui[23])

        # State machine (task name resolved from staged task index).
        task_name = TASK_NAMES[task_val] if 0 <= task_val < len(TASK_NAMES) else "?"
        imgui.text(f"Task:  {task_name}")
        imgui.text(f"Timer: {cur_timer:.2f}s / {task_dur:.2f}s")

        imgui.separator()

        # EE position error (read from staging buffer -- no per-frame .numpy()).
        err_x = ee_target[0] - ee_actual[0]
        err_y = ee_target[1] - ee_actual[1]
        err_z = ee_target[2] - ee_actual[2]
        imgui.text(f"EE err: x={err_x * 1000:+.2f} y={err_y * 1000:+.2f} z={err_z * 1000:+.2f} mm")

        imgui.separator()

        # Pad (finger) forces
        imgui.text(f"Pad F:   {cur_pad_f:.1f} N  (max: {max_pad_f:.1f} N)")
        imgui.text(f"Pad Fr:  {cur_pad_fr:.1f} N  (max: {max_pad_fr:.1f} N)")
        imgui.text(f"Table F: {cur_tbl_f:.1f} N  (max: {max_tbl_f:.1f} N)")
        imgui.text(f"Penetration: {cur_pen_mm:.3f} mm  (max: {max_pen_mm:.3f} mm)")
        imgui.text(f"Avg vel: {avg_vel_mms:.2f} mm/s")

        imgui.separator()

        # Object Z and lift
        lift = obj_z - init_z
        max_lift = max_z - init_z
        imgui.text(f"Object Z: {obj_z:.4f} m  (init: {init_z:.4f} m)")
        imgui.text(f"Lift: {lift * 1000:.1f} mm  (max: {max_lift * 1000:.1f} mm)")
        if nan_frame_val >= 0:
            imgui.text(f"NaN at frame {nan_frame_val}!")

        imgui.separator()
        # Sliders below double as text inputs: Ctrl+click (or Shift+click) on the
        # bar swaps the slider for an editable field, so a separate input_float
        # companion isn't needed.
        imgui.text("Grasp Pose (current world's shape)")
        shape = self.world_shapes[w]
        edits = self._gui_grasp_edits[shape]

        for i, label in enumerate(("offset.x", "offset.y", "offset.z")):
            changed, val = imgui.slider_float(label, edits["offset_local"][i], -2.0, 2.0)
            if changed:
                edits["offset_local"][i] = val

        for i, label in enumerate(("euler.x (deg)", "euler.y (deg)", "euler.z (deg)")):
            changed, val = imgui.slider_float(label, edits["euler_deg"][i], -180.0, 180.0)
            if changed:
                edits["euler_deg"][i] = val

        changed, val = imgui.slider_float("margin_pct", edits["margin_pct"], 0.0, 0.4)
        if changed:
            edits["margin_pct"] = val
        derived_ctrl = margin_pct_to_ctrl(edits["margin_pct"], self._world_y_half[w])
        imgui.text(f"  -> derived ctrl: {derived_ctrl:.2f}")

        imgui.separator()
        imgui.text("Global (all worlds)")
        changed, val = imgui.slider_int("lift_distance (mm)", self._gui_lift_mm, 0, 300)
        if changed:
            self._gui_lift_mm = val

        imgui.separator()
        _, self._gui_broadcast_apply = imgui.checkbox("Apply to all worlds of this shape", self._gui_broadcast_apply)
        # Use the GUI-staged task index (refreshed every _gui_read_interval frames)
        # so we don't pay a per-frame GPU->CPU sync. task_val was extracted above
        # from self._cached_gui at index 14.
        can_apply = task_val <= int(TaskType.APPROACH)
        if not can_apply:
            imgui.text_disabled("Apply disabled: world has left APPROACH")
        if can_apply and imgui.button("Apply"):
            self._apply_grasp_edits(w, broadcast=self._gui_broadcast_apply)
        if imgui.button("Print current poses"):
            self._print_current_poses()

        imgui.separator()
        _, self.show_object_frames = imgui.checkbox("Show object frames", self.show_object_frames)
        _, self.show_object_com_frames = imgui.checkbox("Show object COM frames", self.show_object_com_frames)
        _, self.show_ee_base_frames = imgui.checkbox("Show EE base frames", self.show_ee_base_frames)
        _, self.show_tcp_frames = imgui.checkbox("Show TCP frames", self.show_tcp_frames)
        _, self.show_world_frames = imgui.checkbox("Show world frames", self.show_world_frames)
        _, self.show_spawn_region = imgui.checkbox("Show spawn region", self.show_spawn_region)
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC and hasattr(self.viewer, "renderer"):
            changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
            if changed:
                self.viewer.show_hydro_contact_surface = self.show_isosurface
        imgui.text(f"Frame: {self.episode_steps}  t={self.sim_time:.2f}s")

    def _apply_grasp_edits(self, w: int, broadcast: bool) -> None:
        """Commit the GUI edit buffer for world w's shape to GRASP_DESIGNS and the GPU."""
        shape = self.world_shapes[w]
        edits = self._gui_grasp_edits[shape]

        # 1. Mutate GRASP_DESIGNS (single entry per shape).
        new_design = GraspDesign(
            offset_local=wp.vec3(*edits["offset_local"]),
            quat_local=_euler_zyx_deg_to_quat(wp.vec3(*edits["euler_deg"])),
            margin_pct=edits["margin_pct"],
        )
        GRASP_DESIGNS[shape] = new_design

        # 2. Update global lift.
        self.lift_distance_m = self._gui_lift_mm / 1000.0

        # 3. Rewrite the affected CPU mirrors. The user's full 3-component
        # offset_local is written through verbatim; derive_offset_local_z is only
        # used as a one-shot init seed in _setup_state_machine.
        affected = self._shape_mask[shape] if broadcast else np.array([w], dtype=np.int32)
        for idx in affected:
            idx_int = int(idx)
            self._design_offset_local_np[idx_int] = new_design.offset_local
            self._design_quat_local_np[idx_int] = new_design.quat_local
            self._design_ctrl_np[idx_int] = margin_pct_to_ctrl(
                new_design.margin_pct, y_half_m=self._world_y_half[idx_int]
            )

        # 4. Upload mutated CPU mirrors back to GPU and relaunch the grasp-target kernel.
        self._upload_and_relaunch_grasp_targets(slot=None if broadcast else w)

    def _upload_and_relaunch_grasp_targets(self, slot: int | None) -> None:
        """Push design buffers (CPU mirror -> GPU) and recompute grasp targets.

        ``slot=None`` rebuilds every world via a full-array assign + dim=world_count launch.
        ``slot=i`` patches just world ``i`` with a 1-element copy + dim=1 launch.
        """
        common_inputs = [
            self.state_0.body_q,
            self.model.body_com,
            self.model.body_world_start,
            self.object_body_offset,
            self._world_half_size_array,
            self._design_offset_local,
            self._design_quat_local,
            self._design_ctrl,
            self.base_ee_rot,
        ]
        outputs = [self.grasp_pos, self.grasp_rot, self.grasp_ctrl]
        if slot is None:
            self._design_offset_local.assign(self._design_offset_local_np)
            self._design_quat_local.assign(self._design_quat_local_np)
            self._design_ctrl.assign(self._design_ctrl_np)
            wp.launch(compute_grasp_targets, dim=self.world_count, inputs=common_inputs, outputs=outputs)
        else:
            self._single_offset_local_src.assign(self._design_offset_local_np[slot : slot + 1])
            self._single_quat_local_src.assign(self._design_quat_local_np[slot : slot + 1])
            self._single_ctrl_src.assign(self._design_ctrl_np[slot : slot + 1])
            wp.copy(self._design_offset_local, self._single_offset_local_src, dest_offset=slot, count=1)
            wp.copy(self._design_quat_local, self._single_quat_local_src, dest_offset=slot, count=1)
            wp.copy(self._design_ctrl, self._single_ctrl_src, dest_offset=slot, count=1)
            wp.launch(compute_grasp_targets_slot, dim=1, inputs=[slot, *common_inputs], outputs=outputs)

    def _print_current_poses(self) -> None:
        """Dump GRASP_DESIGNS and self.lift_distance_m as a pasteable Python literal."""
        print("# --- Current grasp designs (paste over GRASP_DESIGNS and self.lift_distance_m) ---")
        print("GRASP_DESIGNS = {")
        for shape in ObjectShape:
            if shape not in GRASP_DESIGNS:
                continue
            d = GRASP_DESIGNS[shape]
            # Iterate the wp.vec3 / wp.quat to get plain Python floats so the
            # repr is pasteable (`0.5` rather than `np.float32(0.5)`).
            o_str = ", ".join(repr(x) for x in d.offset_local)
            q_str = ", ".join(repr(x) for x in d.quat_local)
            print(
                f"    ObjectShape.{shape.name:<12}: GraspDesign("
                f"offset_local=wp.vec3({o_str}), "
                f"quat_local=wp.quat({q_str}), "
                f"margin_pct={d.margin_pct!r}),"
            )
        print("}")
        print(f"self.lift_distance_m = {self.lift_distance_m!r}")

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

    def _set_joint_targets(self):
        """Compute IK targets from state machine and solve IK each frame."""
        # 1. Compute targets from state machine (with interpolation)
        wp.launch(
            set_target_pose_kernel,
            dim=self.world_count,
            inputs=[
                self.task_idx,
                self.task_timer,
                self.task_durations,
                self.grasp_pos,
                self.grasp_rot,
                self.grasp_ctrl,
                self.lift_distance_m,
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
            dim=self.world_count,
            inputs=[
                self.joint_q_ik,
                self.gripper_target,
                self.joint_targets_2d,
                self.arm_dof_count,
                self.gripper_dof_start,
                self.gripper_dof_count,
            ],
        )
        wp.copy(self.control.joint_target_pos, self.joint_targets_2d.flatten())

        # Write arm + gripper to mujoco ctrl (CTRL_DIRECT for all MJCF general actuators)
        if self.has_mujoco_ctrl:
            wp.launch(
                write_mujoco_ctrl_kernel,
                dim=self.world_count,
                inputs=[
                    self.joint_q_ik,
                    self.gripper_target,
                    self.mujoco_ctrl_2d,
                    self.arm_dof_count,
                    self.gripper_actuator_idx,
                ],
            )

        # 5. Extract actual EE positions from simulation state
        wp.launch(
            extract_ee_pos_kernel,
            dim=self.world_count,
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
            dim=self.world_count,
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

    def _render_frame_channel(self, *, enabled, log_path, kernel, kernel_inputs, begin, end, colors):
        """Render one debug-frame channel: launch the kernel into begin/end, then log_lines.

        When the channel is disabled we still emit a None tuple so the viewer can clear the
        previous frame's geometry instead of leaving it stale.
        """
        if enabled:
            wp.launch(kernel, dim=self.world_count, inputs=kernel_inputs, outputs=[begin, end])
            self.viewer.log_lines(log_path, begin, end, colors)
        else:
            self.viewer.log_lines(log_path, None, None, None)

    def _render_debug_frames(self):
        """Draw XYZ coordinate frames and the spawn region for whichever toggles are on.

        ``Show object frames``     -> RGB triad at ``body_q[obj].t`` (rotated by body_q.q).
        ``Show object COM frames`` -> RGB triad at ``body_q[obj] * body_com`` (rotated).
        ``Show EE base frames``    -> RGB triad at ``body_q[ee].t`` (rotated by body_q.q).
        ``Show TCP frames``        -> RGB triad at the EE TCP, rotated by body_q[ee].q.
        ``Show world frames``      -> RGB triad at each world's render origin, world-axis-aligned.
        ``Show spawn region``      -> Yellow square outline showing where each world's object
                                      can spawn (XY half-range = self.spawn_xy_range).
        All kernels are GPU-resident; toggling a channel off skips its kernel launch.
        """
        if not hasattr(self.viewer, "log_lines"):
            return

        # Resolve world_offsets once; used by all frame channels below.
        viewer_offsets = getattr(self.viewer, "world_offsets", None)
        world_offsets = viewer_offsets if viewer_offsets is not None else self._zero_world_offsets
        visible = getattr(self.viewer, "_visible_worlds_mask", None)
        axis_len = 0.05
        body_q = self.state_0.body_q
        body_world_start = self.model.body_world_start

        self._render_frame_channel(
            enabled=self.show_object_frames,
            log_path="/frames/object",
            kernel=compute_object_frame_lines,
            kernel_inputs=[
                body_q,
                body_world_start,
                self.object_body_offset,
                world_offsets,
                visible,
                axis_len,
            ],
            begin=self._object_frame_begin,
            end=self._object_frame_end,
            colors=self._frame_colors,
        )
        self._render_frame_channel(
            enabled=self.show_object_com_frames,
            log_path="/frames/object_com",
            kernel=compute_object_com_frame_lines,
            kernel_inputs=[
                body_q,
                self.model.body_com,
                body_world_start,
                self.object_body_offset,
                world_offsets,
                visible,
                axis_len,
            ],
            begin=self._object_com_frame_begin,
            end=self._object_com_frame_end,
            colors=self._frame_colors,
        )
        self._render_frame_channel(
            enabled=self.show_ee_base_frames,
            log_path="/frames/ee_base",
            kernel=compute_ee_base_frame_lines,
            kernel_inputs=[body_q, body_world_start, self.ee_base_body_idx, world_offsets, visible, axis_len],
            begin=self._ee_base_frame_begin,
            end=self._ee_base_frame_end,
            colors=self._frame_colors,
        )
        self._render_frame_channel(
            enabled=self.show_tcp_frames,
            log_path="/frames/tcp",
            kernel=compute_tcp_frame_lines,
            kernel_inputs=[
                body_q,
                body_world_start,
                self.ee_base_body_idx,
                _ROBOTIQ_TCP_OFFSET_M,
                world_offsets,
                visible,
                axis_len,
            ],
            begin=self._tcp_frame_begin,
            end=self._tcp_frame_end,
            colors=self._frame_colors,
        )
        # World frame is drawn longer than the body frames so the world axes stand out.
        self._render_frame_channel(
            enabled=self.show_world_frames,
            log_path="/frames/world",
            kernel=compute_world_frame_lines,
            kernel_inputs=[world_offsets, visible, 0.20],
            begin=self._world_frame_begin,
            end=self._world_frame_end,
            colors=self._frame_colors,
        )
        self._render_frame_channel(
            enabled=self.show_spawn_region,
            log_path="/frames/spawn_region",
            kernel=compute_spawn_region_lines,
            kernel_inputs=[world_offsets, visible, self.spawn_center, self.spawn_xy_range, self.spawn_xy_range],
            begin=self._spawn_region_begin,
            end=self._spawn_region_end,
            colors=self._spawn_region_colors,
        )


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
