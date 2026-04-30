# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    GRASP_DESIGNS,
    ObjectShape,
    compute_grasp_targets,
    derive_offset_local_z,
)
from newton.examples.robot.example_robot_heterogeneous_grasp import (  # noqa: E402
    Example as GraspExample,
)


class TestGraspDesigns(unittest.TestCase):
    def test_every_shape_has_an_entry(self):
        for shape in ObjectShape:
            self.assertIn(shape, GRASP_DESIGNS, f"Missing GRASP_DESIGNS entry for {shape}")

    def test_margin_pct_in_valid_range(self):
        # Sanity: every shipped margin_pct value is positive and < 0.5 (above 0.5 the
        # pads would close past each other).
        for shape, design in GRASP_DESIGNS.items():
            self.assertGreater(design.margin_pct, 0.0, msg=f"{shape}")
            self.assertLess(design.margin_pct, 0.5, msg=f"{shape}")

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

        design_offset_local = wp.array([wp.vec3(0.0, 0.0, 0.5)], dtype=wp.vec3)
        design_quat_local = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
        design_ctrl = wp.array([123.0], dtype=wp.float32)
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
                design_offset_local,
                design_quat_local,
                design_ctrl,
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

        design_offset_local = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        design_quat_local = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
        design_ctrl = wp.array([0.0], dtype=wp.float32)
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
                design_offset_local,
                design_quat_local,
                design_ctrl,
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

        design_offset_local = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        design_quat_local = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
        design_ctrl = wp.array([0.0], dtype=wp.float32)
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
                design_offset_local,
                design_quat_local,
                design_ctrl,
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
            offset_local = example._design_offset_local_np[i].astype(np.float64)
            quat_local = example._design_quat_local_np[i].astype(np.float64)

            com_world = body_tr + _quat_rotate_np(body_rot, com_local)
            offset_world = _quat_rotate_np(body_rot, offset_local * hs_i)
            expected_pos[i] = (com_world + offset_world).astype(np.float32)
            expected_rot[i] = _quat_mul_np(_quat_mul_np(body_rot, base_ee_rot), quat_local).astype(np.float32)

        expected_ctrl = example._design_ctrl_np.astype(np.float32)

        np.testing.assert_allclose(
            grasp_pos, expected_pos, atol=1e-5, err_msg="grasp_pos kernel output disagrees with CPU reference"
        )
        np.testing.assert_allclose(
            grasp_rot, expected_rot, atol=1e-5, err_msg="grasp_rot kernel output disagrees with CPU reference"
        )
        np.testing.assert_allclose(
            grasp_ctrl, expected_ctrl, atol=1e-3, err_msg="grasp_ctrl kernel output disagrees with design inputs"
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


if __name__ == "__main__":
    unittest.main()
