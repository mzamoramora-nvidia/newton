# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

from newton.examples.robot.example_robot_heterogeneous_grasp import margin_pct_to_ctrl


class TestMarginPctToCtrl(unittest.TestCase):
    def test_zero_margin_tight_grasp_is_high_ctrl(self):
        # 30 mm object, 0% margin -> ctrl = 255 * (1 - 30/85) ~= 164.85
        ctrl = margin_pct_to_ctrl(margin_pct=0.0, y_half_m=0.015)
        self.assertAlmostEqual(ctrl, 255.0 * (1.0 - 30.0 / 85.0), places=3)

    def test_formula_self_consistent(self):
        margin_pct = 0.15
        y_half_m = 0.020
        y_width_mm = 2.0 * y_half_m * 1000.0
        margin_mm = margin_pct * y_width_mm
        expected = min(255.0, max(0.0, 255.0 * (1.0 - (y_width_mm - 2.0 * margin_mm) / 85.0)))
        self.assertAlmostEqual(margin_pct_to_ctrl(margin_pct, y_half_m), expected, places=5)

    def test_clamped_low(self):
        # Object wider than full stroke (y_width_mm=100 > stroke_mm=85) with 0 margin
        # -> raw ctrl < 0, clamped to 0.0 (pads wide open).
        self.assertEqual(margin_pct_to_ctrl(margin_pct=0.0, y_half_m=0.050), 0.0)

    def test_clamped_high(self):
        # margin_pct=0.5 -> margin_mm == y_half_mm -> net width = 0 -> ctrl = 255, clamped high.
        self.assertEqual(margin_pct_to_ctrl(margin_pct=0.5, y_half_m=0.020), 255.0)


import sys  # noqa: E402

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton.examples as nex  # noqa: E402
from newton.examples.robot.example_robot_heterogeneous_grasp import (  # noqa: E402
    GRASP_SPECS,
    OBJECT_CATALOG_PRIMITIVES,
    ObjectShape,
    compute_grasp_targets,
    derive_offset_local_z,
)
from newton.examples.robot.example_robot_heterogeneous_grasp import (  # noqa: E402
    Example as GraspExample,
)


class TestGraspSpecs(unittest.TestCase):
    def test_every_shape_has_an_entry(self):
        for shape in ObjectShape:
            self.assertIn(shape, GRASP_SPECS, f"Missing GRASP_SPECS entry for {shape}")

    def test_margin_pct_in_valid_range(self):
        # Sanity: every shipped margin_pct value is positive and < 0.5 (above 0.5 the
        # pads would close past each other).
        for shape, spec in GRASP_SPECS.items():
            self.assertGreater(spec.margin_pct, 0.0, msg=f"{shape}")
            self.assertLess(spec.margin_pct, 0.5, msg=f"{shape}")

    def test_derive_offset_local_z_formula(self):
        # offset_local.z = (0.0005 + max(0, 2*z_half - grasp_clearance) - z_half + extra) / half_size
        grasp_clearance = 0.05
        cases = [
            # (shape, half_size, z_half, extra_offset)
            (ObjectShape.BOX, 0.010, 0.010, 0.0),
            (ObjectShape.CUP, 0.012, 0.012, 0.0),
            (ObjectShape.BOLT, 0.015, 0.015, 0.02),
            (ObjectShape.BEAR, 0.015, 0.015, 0.01),
        ]
        for shape, half_size, z_half, extra in cases:
            expected = (0.0005 + max(0.0, 2.0 * z_half - grasp_clearance) - z_half + extra) / half_size
            self.assertAlmostEqual(
                derive_offset_local_z(shape, half_size=half_size, z_half=z_half, grasp_clearance=grasp_clearance),
                expected,
                places=6,
                msg=f"offset_local.z mismatch for {shape}",
            )


class TestComputeGraspTargetsKernel(unittest.TestCase):
    def test_single_world_identity_pose(self):
        wp.init()
        # One world, object at (1, 2, 3) with identity body_q, zero body_com.
        body_q = wp.array([wp.transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))], dtype=wp.transform)
        body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        body_world_start = wp.array([0], dtype=wp.int32)
        world_hs = wp.array([0.01], dtype=wp.float32)

        spec_offset_local = wp.array([wp.vec3(0.0, 0.0, 0.5)], dtype=wp.vec3)
        spec_quat_local = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
        spec_ctrl = wp.array([123.0], dtype=wp.float32)
        base_ee_rot = wp.quat(0.0, 0.0, 0.0, 1.0)

        grasp_pos = wp.zeros(1, dtype=wp.vec3)
        grasp_rot = wp.zeros(1, dtype=wp.quat)
        grasp_ctrl = wp.zeros(1, dtype=wp.float32)

        wp.launch(
            compute_grasp_targets,
            dim=1,
            inputs=[
                body_q,
                body_com,
                body_world_start,
                0,
                world_hs,
                spec_offset_local,
                spec_quat_local,
                spec_ctrl,
                base_ee_rot,
            ],
            outputs=[grasp_pos, grasp_rot, grasp_ctrl],
        )

        # offset_local.z = 0.5, hs = 0.01 -> 0.005 m above COM = body_q.t + (0, 0, 0.005)
        np.testing.assert_allclose(grasp_pos.numpy()[0], [1.0, 2.0, 3.005], atol=1e-6)
        np.testing.assert_allclose(grasp_rot.numpy()[0], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(float(grasp_ctrl.numpy()[0]), 123.0, places=4)

    def test_com_offset_is_applied(self):
        wp.init()
        body_q = wp.array([wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))], dtype=wp.transform)
        body_com = wp.array([wp.vec3(0.1, 0.2, 0.3)], dtype=wp.vec3)
        body_world_start = wp.array([0], dtype=wp.int32)
        world_hs = wp.array([0.01], dtype=wp.float32)

        spec_offset_local = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        spec_quat_local = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
        spec_ctrl = wp.array([0.0], dtype=wp.float32)
        base_ee_rot = wp.quat(0.0, 0.0, 0.0, 1.0)

        grasp_pos = wp.zeros(1, dtype=wp.vec3)
        grasp_rot = wp.zeros(1, dtype=wp.quat)
        grasp_ctrl = wp.zeros(1, dtype=wp.float32)

        wp.launch(
            compute_grasp_targets,
            dim=1,
            inputs=[
                body_q,
                body_com,
                body_world_start,
                0,
                world_hs,
                spec_offset_local,
                spec_quat_local,
                spec_ctrl,
                base_ee_rot,
            ],
            outputs=[grasp_pos, grasp_rot, grasp_ctrl],
        )

        # With zero offset_local and identity quat, grasp_pos = COM world = body_q.t + body_com.
        np.testing.assert_allclose(grasp_pos.numpy()[0], [0.1, 0.2, 0.3], atol=1e-6)

    def test_rotation_composition(self):
        # Verifies grasp_rot = body_q.q * base_ee_rot * quat_local with non-identity base_ee_rot.
        wp.init()
        # body_q: identity rotation. base_ee_rot: 90 deg about Y. quat_local: identity.
        # Expected: 90 deg about Y in world frame -> quat (0, sin45, 0, cos45).
        body_q = wp.array([wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))], dtype=wp.transform)
        body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        body_world_start = wp.array([0], dtype=wp.int32)
        world_hs = wp.array([0.01], dtype=wp.float32)

        spec_offset_local = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        spec_quat_local = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
        spec_ctrl = wp.array([0.0], dtype=wp.float32)
        sin45 = np.sin(np.pi / 4.0)
        cos45 = np.cos(np.pi / 4.0)
        base_ee_rot = wp.quat(0.0, sin45, 0.0, cos45)

        grasp_pos = wp.zeros(1, dtype=wp.vec3)
        grasp_rot = wp.zeros(1, dtype=wp.quat)
        grasp_ctrl = wp.zeros(1, dtype=wp.float32)

        wp.launch(
            compute_grasp_targets,
            dim=1,
            inputs=[
                body_q,
                body_com,
                body_world_start,
                0,
                world_hs,
                spec_offset_local,
                spec_quat_local,
                spec_ctrl,
                base_ee_rot,
            ],
            outputs=[grasp_pos, grasp_rot, grasp_ctrl],
        )

        np.testing.assert_allclose(grasp_rot.numpy()[0], [0.0, sin45, 0.0, cos45], atol=1e-6)


def _quat_rotate_np(q_xyzw, v):
    """Rotate vector v by quaternion q (xyzw convention, double precision)."""
    x, y, z, w = q_xyzw
    t = 2.0 * np.cross([x, y, z], v)
    return np.asarray(v, dtype=np.float64) + w * t + np.cross([x, y, z], t)


def _quat_mul_np(a_xyzw, b_xyzw):
    """Hamilton product of two quaternions (xyzw convention)."""
    ax, ay, az, aw = a_xyzw
    bx, by, bz, bw = b_xyzw
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


class TestGraspTargetsMatchReference(unittest.TestCase):
    """Regression: compute_grasp_targets kernel output matches a NumPy reference of the same formula.

    Catches kernel-side bugs (bad quat composition, wrong COM sign, etc.) by mirroring the
    same composition (``body_q * body_com`` for the COM, ``body_q.q * base_ee_rot * quat_local``
    for the rotation) on the CPU and asserting equality across every shape.
    """

    def test_kernel_matches_cpu_reference(self):
        wp.init()
        parser = GraspExample.create_parser()
        sys.argv = ["test", "--viewer", "null", "--world-count", "12", "--num-frames", "1"]
        viewer, args = nex.init(parser)

        example = GraspExample(viewer, args)

        grasp_pos = example.grasp_pos.numpy()
        grasp_rot = example.grasp_rot.numpy()
        grasp_ctrl = example.grasp_ctrl.numpy()

        wc = example.world_count
        body_q_np = example.state_0.body_q.numpy()
        body_com_np = example.model.body_com.numpy()
        body_ws_np = example.model.body_world_start.numpy()
        base_ee_rot = np.array([float(example.base_ee_rot[i]) for i in range(4)], dtype=np.float64)

        expected_pos = np.zeros((wc, 3), dtype=np.float32)
        expected_rot = np.zeros((wc, 4), dtype=np.float32)
        for i in range(wc):
            obj_global = int(body_ws_np[i]) + example.object_body_offset
            tr = body_q_np[obj_global]
            body_tr = np.array([tr[0], tr[1], tr[2]], dtype=np.float64)
            body_rot = np.array([tr[3], tr[4], tr[5], tr[6]], dtype=np.float64)
            com_local = body_com_np[obj_global].astype(np.float64)
            hs_i = float(example.world_half_sizes[i])
            offset_local = example.spec.offset_local_np[i].astype(np.float64)
            quat_local = example.spec.quat_local_np[i].astype(np.float64)

            com_world = body_tr + _quat_rotate_np(body_rot, com_local)
            offset_world = _quat_rotate_np(body_rot, offset_local * hs_i)
            expected_pos[i] = (com_world + offset_world).astype(np.float32)
            expected_rot[i] = _quat_mul_np(_quat_mul_np(body_rot, base_ee_rot), quat_local).astype(np.float32)

        expected_ctrl = example.spec.ctrl_np.astype(np.float32)

        np.testing.assert_allclose(
            grasp_pos, expected_pos, atol=1e-5, err_msg="grasp_pos kernel output disagrees with CPU reference"
        )
        np.testing.assert_allclose(
            grasp_rot, expected_rot, atol=1e-5, err_msg="grasp_rot kernel output disagrees with CPU reference"
        )
        np.testing.assert_allclose(
            grasp_ctrl, expected_ctrl, atol=1e-3, err_msg="grasp_ctrl kernel output disagrees with spec inputs"
        )


class TestSpawnRandomization(unittest.TestCase):
    def test_spawn_xy_within_range(self):
        wp.init()
        parser = GraspExample.create_parser()
        sys.argv = [
            "test",
            "--viewer",
            "null",
            "--world-count",
            "12",
            "--num-frames",
            "1",
            "--seed",
            "7",
            "--spawn-xy-range",
            "0.10",
            "--spawn-yaw-range",
            "0",
        ]
        viewer, args = nex.init(parser)
        example = GraspExample(viewer, args)

        body_q = example.state_0.body_q.numpy()
        body_ws = example.model.body_world_start.numpy()
        tt = np.array([float(example.spawn_center[0]), float(example.spawn_center[1])])
        for w in range(example.world_count):
            obj = body_q[int(body_ws[w]) + example.object_body_offset]
            dxy = np.array([obj[0], obj[1]]) - tt
            np.testing.assert_array_less(np.abs(dxy), 0.1001)


# ----------------------------------------------------------------------------
# GraspProbe — diagnostic surface for the heterogeneous-grasp example.
#
# Attach via Example(viewer, args, probe=probe). Headless regression tests get
# measurement and assertion; GL runs additionally render the tuning panel and
# debug-frame overlays via on_gui_render / on_render hooks (added in later
# tasks). The probe lives in the test file so the example stays focused on
# simulation.
#
# Graph-safety: on_step launches kernels outside the captured CUDA graph (the
# example only captures simulate()), so allocations in __init__ / on_init are
# safe and on_step has no graph-capture constraints.
# ----------------------------------------------------------------------------

from newton import Contacts  # noqa: E402
from newton.examples.robot.example_robot_heterogeneous_grasp import (  # noqa: E402
    _GUI_STAGE_SIZE,
    _ROBOTIQ_TCP_OFFSET_M,
    SHAPE_NAMES,
    TASK_NAMES,
    CollisionMode,
    TaskType,
    _euler_zyx_deg_to_quat,
    _print_table,
    _quat_to_euler_zyx_deg,
    alloc_line_buffers,
    full_init,
    zero_init,
)
from newton.sensors import SensorContact  # noqa: E402


@wp.kernel(enable_backward=False)
def update_penetration_kernel(
    contact_dist: wp.array[wp.float32],
    contact_worldid: wp.array[wp.int32],
    nacon: wp.array[wp.int32],
    world_count: wp.int32,
    penetration_cur: wp.array[wp.float32],
    penetration_max: wp.array[wp.float32],
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

    pad_f_sum = wp.vec3(0.0, 0.0, 0.0)
    pad_fr_sum = wp.vec3(0.0, 0.0, 0.0)
    for c in range(n_counterparts):
        pad_f_sum = pad_f_sum + pad_force_matrix[w, c]
        pad_fr_sum = pad_fr_sum + pad_force_matrix_friction[w, c]
    pf = wp.length(pad_f_sum)
    pfr = wp.length(pad_fr_sum)

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

    obj_global = body_world_start[w] + object_body_offset
    obj_q = body_q[obj_global]
    obj_pos = wp.transform_get_translation(obj_q)
    obj_z = obj_pos[2]
    obj_vel = wp.spatial_bottom(body_qd[obj_global])
    vel_mag = wp.length(obj_vel)

    obj_z_is_nan = wp.isnan(obj_z)
    if obj_z_is_nan and world_nan_frame[w] < 0:
        world_nan_frame[w] = frame_count

    if not obj_z_is_nan:
        object_z_max[w] = wp.max(object_z_max[w], obj_z)

    vel_is_nan = wp.isnan(vel_mag)
    if not vel_is_nan:
        object_vel_sum[w] = object_vel_sum[w] + vel_mag
        object_vel_count[w] = object_vel_count[w] + 1

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
    out[15] = wp.float32(world_nan_frame[w])
    out[16] = task_timer[w]
    cur_task = task_idx[w]
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


# ----------------------------------------------------------------------------
# Warp kernels: frame visualization
#
# Six per-channel triad/spawn-region kernels plus two wp.func helpers. The
# probe owns the line buffers and toggles; on_render dispatches them per
# frame when the matching toggle is set.
# ----------------------------------------------------------------------------


@wp.func
def _write_triad(
    base: wp.int32,
    origin: wp.vec3,
    rot: wp.quat,
    axis_length: wp.float32,
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
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
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
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
    line_begin: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
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


class GraspProbe:
    """GPU-resident per-world + per-state metrics for the heterogeneous-grasp example.

    Attach via ``Example(viewer, args, probe=probe)``. The example calls
    ``on_init`` once after build, ``on_step`` every frame after the sim graph
    replays, and ``on_finish`` from ``test_final`` / from the unittest. All
    arrays are pre-allocated in ``__init__`` so kernel launches inside
    ``on_step`` allocate nothing.
    """

    def __init__(self, *, world_count: int, task_state_count: int, hold_state: int):
        self.world_count = world_count
        self.task_state_count = task_state_count
        self.hold_state = hold_state
        # Set from example.object_body_offset during on_init -- it is only known
        # after the example finishes building its scene.
        self.object_body_offset = -1

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

    def on_init(self, example):
        self.object_body_offset = example.object_body_offset

        if example.verbose:
            self._print_verbose_diagnostics(example)

        # Contact sensors. Example.__init__ already requested the ``force``
        # contact attribute on the model, so the collision pipeline's Contacts
        # buffer (Newton modes) already carries it; the second sensor sets
        # request_contact_attributes=False to avoid a redundant request.
        self.contact_sensor_pad = SensorContact(
            example.model,
            sensing_obj_bodies="object",
            counterpart_shapes="*pad*",
        )
        self.contact_sensor_table = SensorContact(
            example.model,
            sensing_obj_bodies="object",
            counterpart_shapes="table*",
            request_contact_attributes=False,
        )

        # MuJoCo mode has no collision pipeline, so the probe allocates a
        # dedicated Contacts buffer and hands it to the example so simulate(),
        # render(), and on_step all share the same instance.
        if example.collision_mode == CollisionMode.MUJOCO:
            example.contacts = Contacts(
                example.solver.get_max_contact_count(),
                0,
                requested_attributes=example.model.get_requested_contact_attributes(),
            )

        wp.launch(
            _seed_object_z_init_kernel,
            dim=self.world_count,
            inputs=[example.state_0.body_q, example.model.body_world_start, self.object_body_offset],
            outputs=[self.object_z_init],
        )

        # GUI panel state -- cached numpy mirrors so the per-frame side panel
        # never round-trips to the GPU outside of the throttled stage_gui sync.
        self._cached_body_mass = example.model.body_mass.numpy().copy()
        self._cached_body_world_start = example.model.body_world_start.numpy().copy()
        self._cached_gui = np.zeros(_GUI_STAGE_SIZE, dtype=np.float32)
        self._gui_read_interval = 10
        self._gui_shape_filter = -1  # -1 = "All"; >=0 indexes into sorted shape_keys
        self._gui_broadcast_apply = False
        self._gui_lift_mm = int(round(example.lift_distance_m * 1000.0))
        # Per-shape edit buffer seeded from the spec arrays the example is
        # currently using -- particularly offset_local.z, which the example
        # bakes via derive_offset_local_z from a (0, 0, 0) GRASP_SPECS seed.
        self._gui_grasp_edits: dict[ObjectShape, dict] = {}
        for shape, spec in GRASP_SPECS.items():
            mask = example._shape_mask.get(shape)
            if mask is None or len(mask) == 0:
                seed_offset = list(spec.offset_local)
            else:
                seed_offset = example.spec.offset_local_np[int(mask[0])].tolist()
            self._gui_grasp_edits[shape] = {
                "offset_local": seed_offset,
                "euler_deg": list(_quat_to_euler_zyx_deg(spec.quat_local)),
                "margin_pct": spec.margin_pct,
            }

        # Debug-frame overlay state. The probe owns the line buffers, the
        # render-channel toggles, and the selected_world / show_isosurface
        # toggles that the panel mutates.
        self.selected_world = 0
        self.show_isosurface = False
        self.show_object_frames = False
        self.show_object_com_frames = False
        self.show_ee_base_frames = False
        self.show_tcp_frames = False
        self.show_world_frames = False
        self.show_spawn_region = False
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
        self._zero_world_offsets = wp.zeros(self.world_count, dtype=wp.vec3)
        if hasattr(example.viewer, "renderer"):
            example.viewer.show_hydro_contact_surface = self.show_isosurface

    def on_render(self, example) -> None:
        """Draw the debug-frame triads / spawn-region square via log_lines.

        Only runs when the viewer exposes log_lines (GL viewer). Each channel
        either launches its compute_*_frame_lines kernel (when the matching
        toggle is on) or pushes a None tuple so the viewer drops the previous
        frame's stale geometry.
        """
        if not hasattr(example.viewer, "log_lines"):
            return

        viewer_offsets = getattr(example.viewer, "world_offsets", None)
        world_offsets = viewer_offsets if viewer_offsets is not None else self._zero_world_offsets
        visible = getattr(example.viewer, "_visible_worlds_mask", None)
        axis_len = 0.05
        body_q = example.state_0.body_q
        body_world_start = example.model.body_world_start

        self._draw_frame_channel(
            example,
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
        self._draw_frame_channel(
            example,
            enabled=self.show_object_com_frames,
            log_path="/frames/object_com",
            kernel=compute_object_com_frame_lines,
            kernel_inputs=[
                body_q,
                example.model.body_com,
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
        self._draw_frame_channel(
            example,
            enabled=self.show_ee_base_frames,
            log_path="/frames/ee_base",
            kernel=compute_ee_base_frame_lines,
            kernel_inputs=[body_q, body_world_start, example.ee_base_body_idx, world_offsets, visible, axis_len],
            begin=self._ee_base_frame_begin,
            end=self._ee_base_frame_end,
            colors=self._frame_colors,
        )
        self._draw_frame_channel(
            example,
            enabled=self.show_tcp_frames,
            log_path="/frames/tcp",
            kernel=compute_tcp_frame_lines,
            kernel_inputs=[
                body_q,
                body_world_start,
                example.ee_base_body_idx,
                _ROBOTIQ_TCP_OFFSET_M,
                world_offsets,
                visible,
                axis_len,
            ],
            begin=self._tcp_frame_begin,
            end=self._tcp_frame_end,
            colors=self._frame_colors,
        )
        self._draw_frame_channel(
            example,
            enabled=self.show_world_frames,
            log_path="/frames/world",
            kernel=compute_world_frame_lines,
            kernel_inputs=[world_offsets, visible, 0.20],
            begin=self._world_frame_begin,
            end=self._world_frame_end,
            colors=self._frame_colors,
        )
        self._draw_frame_channel(
            example,
            enabled=self.show_spawn_region,
            log_path="/frames/spawn_region",
            kernel=compute_spawn_region_lines,
            kernel_inputs=[
                world_offsets,
                visible,
                example.spawn_center,
                example.spawn_xy_range,
                example.spawn_xy_range,
            ],
            begin=self._spawn_region_begin,
            end=self._spawn_region_end,
            colors=self._spawn_region_colors,
        )

    def _draw_frame_channel(self, example, *, enabled, log_path, kernel, kernel_inputs, begin, end, colors) -> None:
        if enabled:
            wp.launch(kernel, dim=self.world_count, inputs=kernel_inputs, outputs=[begin, end])
            example.viewer.log_lines(log_path, begin, end, colors)
        else:
            example.viewer.log_lines(log_path, None, None, None)

    def on_gui_render(self, example, imgui) -> None:
        """Full tuning panel -- world selector, per-world live metrics,
        per-shape grasp-spec sliders, and frame-overlay toggles."""
        shape_keys: list[ObjectShape] = sorted(example._shape_mask.keys(), key=lambda s: s.name)
        shape_names = ["All"] + [s.name for s in shape_keys]
        changed, new_idx = imgui.combo("Filter by shape", self._gui_shape_filter + 1, shape_names)
        if changed:
            self._gui_shape_filter = new_idx - 1

        if self._gui_shape_filter >= 0:
            filter_shape = shape_keys[self._gui_shape_filter]
            candidates = example._shape_mask[filter_shape]
            if len(candidates) > 0 and self.selected_world not in candidates:
                self.selected_world = int(candidates[0])

        changed, val = imgui.slider_int("World", self.selected_world, 0, example.world_count - 1)
        if changed:
            self.selected_world = max(0, min(example.world_count - 1, val))
        w = self.selected_world

        imgui.separator()

        shape_name = SHAPE_NAMES[example.world_shapes[w]]
        hs_mm = example.world_half_sizes[w] * 1000.0
        obj_global = int(self._cached_body_world_start[w]) + self.object_body_offset
        mass = self._cached_body_mass[obj_global]
        imgui.text(f"Shape: {shape_name}")
        imgui.text(f"Mass:  {mass:.4f} kg")
        imgui.text(f"Size:  {hs_mm:.1f} mm (half-size)")

        imgui.separator()

        if (example.episode_steps % self._gui_read_interval) == 0:
            self._cached_gui = self.stage_gui(
                self.selected_world,
                example.state_0,
                example.model.body_world_start,
                example.task_idx,
                example.task_timer,
                example.task_durations,
                example.ee_pos_target,
                example.ee_pos_actual,
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

        task_name = TASK_NAMES[task_val] if 0 <= task_val < len(TASK_NAMES) else "?"
        imgui.text(f"Task:  {task_name}")
        imgui.text(f"Timer: {cur_timer:.2f}s / {task_dur:.2f}s")

        imgui.separator()

        err_x = ee_target[0] - ee_actual[0]
        err_y = ee_target[1] - ee_actual[1]
        err_z = ee_target[2] - ee_actual[2]
        imgui.text(f"EE err: x={err_x * 1000:+.2f} y={err_y * 1000:+.2f} z={err_z * 1000:+.2f} mm")

        imgui.separator()

        imgui.text(f"Pad F:   {cur_pad_f:.1f} N  (max: {max_pad_f:.1f} N)")
        imgui.text(f"Pad Fr:  {cur_pad_fr:.1f} N  (max: {max_pad_fr:.1f} N)")
        imgui.text(f"Table F: {cur_tbl_f:.1f} N  (max: {max_tbl_f:.1f} N)")
        imgui.text(f"Penetration: {cur_pen_mm:.3f} mm  (max: {max_pen_mm:.3f} mm)")
        imgui.text(f"Avg vel: {avg_vel_mms:.2f} mm/s")

        imgui.separator()

        lift = obj_z - init_z
        max_lift = max_z - init_z
        imgui.text(f"Object Z: {obj_z:.4f} m  (init: {init_z:.4f} m)")
        imgui.text(f"Lift: {lift * 1000:.1f} mm  (max: {max_lift * 1000:.1f} mm)")
        if nan_frame_val >= 0:
            imgui.text(f"NaN at frame {nan_frame_val}!")

        imgui.separator()
        # Sliders below double as text inputs: Ctrl+click (or Shift+click) on
        # the bar swaps the slider for an editable field, so a separate
        # input_float companion isn't needed.
        imgui.text("Grasp Pose (current world's shape)")
        shape = example.world_shapes[w]
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
        derived_ctrl = margin_pct_to_ctrl(edits["margin_pct"], example._world_y_half[w])
        imgui.text(f"  -> derived ctrl: {derived_ctrl:.2f}")

        imgui.separator()
        imgui.text("Global (all worlds)")
        changed, val = imgui.slider_int("lift_distance (mm)", self._gui_lift_mm, 0, 300)
        if changed:
            self._gui_lift_mm = val

        imgui.separator()
        _, self._gui_broadcast_apply = imgui.checkbox("Apply to all worlds of this shape", self._gui_broadcast_apply)
        # Use the GUI-staged task index (refreshed every _gui_read_interval frames)
        # so we don't pay a per-frame GPU->CPU sync.
        can_apply = task_val <= int(TaskType.APPROACH)
        if not can_apply:
            imgui.text_disabled("Apply disabled: world has left APPROACH")
        if can_apply and imgui.button("Apply"):
            shape = example.world_shapes[w]
            edits = self._gui_grasp_edits[shape]
            example.apply_grasp_edits(
                w,
                offset_local=wp.vec3(*edits["offset_local"]),
                quat_local=_euler_zyx_deg_to_quat(wp.vec3(*edits["euler_deg"])),
                margin_pct=edits["margin_pct"],
                lift_distance_m=self._gui_lift_mm / 1000.0,
                broadcast=self._gui_broadcast_apply,
            )
        if imgui.button("Print current poses"):
            self._print_current_poses(example)

        imgui.separator()
        _, self.show_object_frames = imgui.checkbox("Show object frames", self.show_object_frames)
        _, self.show_object_com_frames = imgui.checkbox("Show object COM frames", self.show_object_com_frames)
        _, self.show_ee_base_frames = imgui.checkbox("Show EE base frames", self.show_ee_base_frames)
        _, self.show_tcp_frames = imgui.checkbox("Show TCP frames", self.show_tcp_frames)
        _, self.show_world_frames = imgui.checkbox("Show world frames", self.show_world_frames)
        _, self.show_spawn_region = imgui.checkbox("Show spawn region", self.show_spawn_region)
        if example.collision_mode == CollisionMode.NEWTON_HYDROELASTIC and hasattr(example.viewer, "renderer"):
            changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
            if changed:
                example.viewer.show_hydro_contact_surface = self.show_isosurface
        imgui.text(f"Frame: {example.episode_steps}  t={example.sim_time:.2f}s")

    def _print_current_poses(self, example) -> None:
        """Dump GRASP_SPECS and example.lift_distance_m as a pasteable literal."""
        print("# --- Current grasp specs (paste over GRASP_SPECS and self.lift_distance_m) ---")
        print("GRASP_SPECS = {")
        for shape in ObjectShape:
            if shape not in GRASP_SPECS:
                continue
            d = GRASP_SPECS[shape]
            o_str = ", ".join(repr(x) for x in d.offset_local)
            q_str = ", ".join(repr(x) for x in d.quat_local)
            print(
                f"    ObjectShape.{shape.name:<12}: GraspSpec("
                f"offset_local=wp.vec3({o_str}), "
                f"quat_local=wp.quat({q_str}), "
                f"margin_pct={d.margin_pct!r}),"
            )
        print("}")
        print(f"self.lift_distance_m = {example.lift_distance_m!r}")

    @staticmethod
    def _print_verbose_diagnostics(example) -> None:
        for i in range(example.world_count):
            print(
                f"  World {i:3d}: shape={SHAPE_NAMES[example.world_shapes[i]]:>12s}  "
                f"hs={example.world_half_sizes[i] * 1000:.1f} mm"
            )
        body_com_np = example.model.body_com.numpy()
        body_ws_np = example.model.body_world_start.numpy()
        first_world = {shape: idx for idx, shape in reversed(list(enumerate(example.world_shapes)))}
        print("[grasp] per-shape body_com magnitudes (body-local frame):")
        for shape, idx in sorted(first_world.items(), key=lambda kv: kv[0].name):
            com_norm = np.linalg.norm(body_com_np[int(body_ws_np[idx]) + example.object_body_offset])
            flag = "  (non-zero, review offset_local)" if com_norm > 1e-4 else ""
            print(f"  {shape.name:<12} |body_com| = {com_norm * 1000.0:8.3f} mm{flag}")

    def on_step(self, example):
        # Pre-kernel: refresh contact attribute and probe-owned sensor force
        # matrices. The probe always reuses example.contacts so simulate(),
        # render(), and the sensors see one shared buffer.
        example.solver.update_contacts(example.contacts, example.state_0)
        self.contact_sensor_pad.update(example.state_0, example.contacts)
        self.contact_sensor_table.update(example.state_0, example.contacts)

        self.penetration_cur.zero_()
        mjw_data = example.solver.mjw_data
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
                self.contact_sensor_pad.force_matrix,
                self.contact_sensor_pad.force_matrix_friction,
                self.contact_sensor_table.force_matrix,
                self.contact_sensor_table.force_matrix_friction,
                example.state_0.body_q,
                example.state_0.body_qd,
                example.model.body_world_start,
                self.object_body_offset,
                example.task_idx,
                self.prev_task_idx,
                self.task_state_count,
                self.hold_state,
                example.episode_steps,
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

        wp.launch(
            copy_prev_task_kernel,
            dim=self.world_count,
            inputs=[example.task_idx],
            outputs=[self.prev_task_idx],
        )

    def on_finish(self, example) -> dict:
        """Single batched readback + derived per-world / aggregate stats."""
        result = self._readback_all()
        N = example.world_count
        body_ws = example.model.body_world_start.numpy()[:N].astype(np.int64)
        result["mass"] = example.model.body_mass.numpy()[body_ws + example.object_body_offset]
        result["shape_names"] = np.array([SHAPE_NAMES[s] for s in example.world_shapes])
        result["half_size_mm"] = np.asarray(example.world_half_sizes, dtype=np.float32) * 1000.0
        has_nan = result["world_nan_frame"] >= 0
        lift_mm = np.where(has_nan, 0.0, result["object_z_max"] - result["object_z_init"]) * 1000.0
        hs, he = result["object_z_hold_start"], result["object_z_hold_end"]
        slip_mm = np.where(np.isnan(hs) | np.isnan(he), 0.0, hs - he) * 1000.0
        pen_mm = result["penetration_max"] * 1000.0
        avg_vel_mmps = result["object_vel_sum"] / np.maximum(1, result["object_vel_count"]) * 1000.0
        success = (lift_mm > 50.0) & ~has_nan
        result["lift_mm"] = lift_mm
        result["slip_mm"] = slip_mm
        result["pen_mm"] = pen_mm
        result["avg_vel_mmps"] = avg_vel_mmps
        result["has_nan"] = has_nan
        result["success_per_world"] = success
        result["collision_mode"] = example.collision_mode
        result["success_rate"] = float(success.sum()) / N if N > 0 else 0.0
        result["nan_rate"] = float(has_nan.sum()) / N if N > 0 else 0.0
        return result

    def print_summary(self, result: dict) -> None:
        """Print per-world / per-shape / aggregate / per-state tables. Called
        from the regression test when ``do_rendering=True`` and from local
        developer runs that wire the probe directly."""
        N = len(result["shape_names"])
        shape_names = result["shape_names"]
        mass = result["mass"]
        half_size_mm = result["half_size_mm"]
        lift_mm = result["lift_mm"]
        slip_mm = result["slip_mm"]
        pen_mm = result["pen_mm"]
        avg_vel_mmps = result["avg_vel_mmps"]
        has_nan = result["has_nan"]
        nan_frame = result["world_nan_frame"]
        success = result["success_per_world"]
        pad_f = result["pad_force_max"]
        pad_fr = result["pad_friction_max"]
        tbl_f = result["table_force_max"]

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
            f"HETEROGENEOUS GRASP TEST REPORT  (collision_mode={result['collision_mode'].name})",
            per_world_cols,
            range(N),
        )

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

        n_success = int(success.sum())
        nan_world_count = int(has_nan.sum())
        print("\n" + "-" * 80)
        print("  AGGREGATE STATISTICS")
        print("-" * 80)
        print(f"  Success rate:       {n_success}/{N} = {result['success_rate']:.1%}")
        print(f"  NaN worlds:         {nan_world_count}/{N}")
        print(f"  Penetration:        mean {pen_mm.mean():.3f} mm  /  max {pen_mm.max():.3f} mm")
        print(f"  Slippage:           mean {slip_mm.mean():.3f} mm  /  max {slip_mm.max():.3f} mm")
        print(f"  Pad force (max):    mean {pad_f.mean():.1f} N  /  peak {pad_f.max():.1f} N")
        print(f"  Table force (max):  mean {tbl_f.mean():.1f} N  /  peak {tbl_f.max():.1f} N")
        print(f"  Avg object velocity: {avg_vel_mmps.mean():.2f} mm/s")

        state_count_arr = result["state_count"]
        state_counts_safe = np.maximum(1, state_count_arr)
        state_pad_avg = result["state_pad_force_sum"] / state_counts_safe
        state_tbl_avg = result["state_table_force_sum"] / state_counts_safe
        state_pen_avg = result["state_pen_sum"] / state_counts_safe
        state_vel_avg = result["state_vel_sum"] / state_counts_safe
        state_pad_max = result["state_pad_force_max"]
        state_tbl_max = result["state_table_force_max"]
        state_pen_max = result["state_pen_max"]
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
        """Pack selected-world metrics into the staging buffer. Returns numpy view.

        Used by the GUI panel (which moves into ``on_gui_render`` in Task 4).
        """
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
        return self.gui_staging.numpy()

    def _readback_all(self) -> dict:
        """Single batched GPU -> CPU transfer at episode end."""
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


class TestHeterogeneousGraspRegression(unittest.TestCase):
    """End-to-end regression for the heterogeneous-grasp example.

    Builds the example with a GraspProbe attached, steps a fixed number of
    frames, then asserts on aggregate success / NaN rates. Mirrors the
    ``do_rendering=False`` class-attribute pattern from ``test_robot_composer``,
    ``test_cloth``, etc.; flipping it (e.g., in a developer subclass) routes
    through the GL viewer and prints the full summary tables.
    """

    # Set GRASP_TEST_RENDER=1 to route through ViewerGL: fires probe.on_render
    # (debug-frame triads, spawn-region) and probe.on_gui_render (tuning panel)
    # every frame and prints the summary tables at the end. CI leaves the env
    # var unset, so do_rendering stays False.
    do_rendering = os.environ.get("GRASP_TEST_RENDER", "0") == "1"
    # 12 object shapes round-robin into worlds; 24 = each shape appears twice so
    # the success_rate is a per-shape average instead of a single-roll lottery.
    world_count = 24
    # State-machine durations sum to 5.5 s -> 550 frames at 100 Hz; 700 frames
    # gives enough headroom past the HOLD phase for the lift metric to settle.
    num_frames = 700
    max_nan_rate = 0.25
    # None = full 12-shape catalog. Subclasses can narrow this (e.g. the
    # primitives-only smoke test) by overriding to an ObjectShape list.
    objects: "list | None" = None

    def _run(self, collision_mode: str) -> dict:
        wp.init()
        parser = GraspExample.create_parser()
        sys.argv = [
            "test",
            "--viewer",
            "gl" if self.do_rendering else "null",
            "--world-count",
            str(self.world_count),
            "--num-frames",
            str(self.num_frames),
            "--collision-mode",
            collision_mode,
        ]
        viewer, args = nex.init(parser)
        probe = GraspProbe(
            world_count=self.world_count,
            task_state_count=int(TaskType.DONE),
            hold_state=int(TaskType.HOLD),
        )
        example = GraspExample(viewer, args, probe=probe, objects=self.objects)
        for _ in range(self.num_frames):
            example.step()
            if self.do_rendering:
                example.render()
        result = probe.on_finish(example)
        if self.do_rendering:
            probe.print_summary(result)
        return result

    def _assert_baseline(self, mode: str, min_success_rate: float) -> None:
        r = self._run(mode)
        self.assertGreaterEqual(r["success_rate"], min_success_rate, msg=f"{mode} regression: {r['success_rate']:.1%}")
        self.assertLessEqual(r["nan_rate"], self.max_nan_rate, msg=f"{mode} NaN rate: {r['nan_rate']:.1%}")

    def test_mujoco_baseline(self):
        # MuJoCo native contacts struggle with BEAR and BOLT (0/2 each at
        # baseline). Demonstrating that gap is part of why the example exists.
        # 0.50 leaves an ~8 pp margin below the measured 14/24 = 58%.
        self._assert_baseline("mujoco", min_success_rate=0.50)

    def test_newton_default_baseline(self):
        # Baseline 22/24 = 92% (only RJ45_PLUG and RUBBER_DUCK miss).
        self._assert_baseline("newton_default", min_success_rate=0.80)

    def test_newton_sdf_baseline(self):
        # Baseline 24/24 = 100%. 0.90 protects against one or two stochastic flips.
        self._assert_baseline("newton_sdf", min_success_rate=0.90)

    def test_newton_hydroelastic_baseline(self):
        # Baseline 22/24 = 92%. Hydroelastic is the headline mode the
        # example exists to demo; this test catches regressions on that path.
        self._assert_baseline("newton_hydroelastic", min_success_rate=0.80)


class TestHeterogeneousGraspRegressionPrimitives(TestHeterogeneousGraspRegression):
    """Fast smoke variant: only the 5 primitive shapes (no mesh-asset downloads).

    Same assertion structure as the base class -- catches regressions on the
    fast / asset-light path. Useful when the full 12-shape catalog can't run
    (offline / asset cache cold).
    """

    # 10 worlds = each of the 5 primitives appears twice, matching the
    # per-shape-average pattern of the full-catalog regression.
    world_count = 10
    objects: "list | None" = OBJECT_CATALOG_PRIMITIVES


if __name__ == "__main__":
    # ``--render`` (or ``-r``) flips do_rendering on the regression class so
    # the test routes through ViewerGL. Strip it from sys.argv before
    # delegating to unittest.main so unittest doesn't see an unknown flag.
    if "--render" in sys.argv or "-r" in sys.argv:
        TestHeterogeneousGraspRegression.do_rendering = True
        sys.argv = [a for a in sys.argv if a not in ("--render", "-r")]
    unittest.main()
