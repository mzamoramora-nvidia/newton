# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton.examples.robot.example_robot_heterogeneous_grasp import overclose_to_ctrl


class TestGripOvercloseToCtrl(unittest.TestCase):
    def test_zero_overclose_just_touches_surface(self):
        # 30 mm object, 0 overclose -> ctrl = 255 * (1 - 30/85) ~= 164.85
        ctrl = overclose_to_ctrl(overclose_fraction=0.0, y_half_m=0.015)
        self.assertAlmostEqual(ctrl, 255.0 * (1.0 - 30.0 / 85.0), places=3)

    def test_formula_self_consistent(self):
        overclose = 0.15
        y_half_m = 0.020
        y_width_mm = 2.0 * y_half_m * 1000.0
        overclose_mm = overclose * y_width_mm
        expected = min(255.0, max(0.0, 255.0 * (1.0 - (y_width_mm - 2.0 * overclose_mm) / 85.0)))
        self.assertAlmostEqual(overclose_to_ctrl(overclose, y_half_m), expected, places=5)

    def test_clamped_low(self):
        # Object wider than full stroke (y_width_mm=100 > stroke_mm=85) with 0 overclose
        # -> raw ctrl < 0, clamped to 0.0 (pads wide open).
        self.assertEqual(overclose_to_ctrl(overclose_fraction=0.0, y_half_m=0.050), 0.0)

    def test_clamped_high(self):
        # overclose=0.5 -> overclose_mm == y_half_mm -> net width = 0 -> ctrl = 255, clamped high.
        self.assertEqual(overclose_to_ctrl(overclose_fraction=0.5, y_half_m=0.020), 255.0)


import sys  # noqa: E402

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton.examples as nex  # noqa: E402
from newton.examples.robot.example_robot_heterogeneous_grasp import (  # noqa: E402
    GRASP_SPECS,
    ObjectShape,
    compute_grasp_targets,
    derive_pos_offset_z,
)
from newton.examples.robot.example_robot_heterogeneous_grasp import (  # noqa: E402
    Example as GraspExample,
)


class TestGraspSpecs(unittest.TestCase):
    def test_every_shape_has_an_entry(self):
        for shape in ObjectShape:
            self.assertIn(shape, GRASP_SPECS, f"Missing GRASP_SPECS entry for {shape}")

    def test_overclose_fraction_in_valid_range(self):
        # Sanity: every shipped overclose_fraction is positive and < 0.5
        # (above 0.5 the pads would close past each other).
        for shape, spec in GRASP_SPECS.items():
            self.assertGreater(spec.overclose_fraction, 0.0, msg=f"{shape}")
            self.assertLess(spec.overclose_fraction, 0.5, msg=f"{shape}")

    def test_derive_pos_offset_z_formula(self):
        # Returns the absolute Z offset (m) from the object COM to the EE seed.
        grasp_depth = 0.05
        cases = [0.010, 0.012, 0.015, 0.020, 0.030]  # z_half values to check
        for z_half in cases:
            expected = 0.0005 + max(0.0, 2.0 * z_half - grasp_depth) - z_half
            self.assertAlmostEqual(
                derive_pos_offset_z(z_half=z_half, grasp_depth=grasp_depth),
                expected,
                places=6,
                msg=f"pos_offset.z mismatch for z_half={z_half}",
            )


class TestComputeGraspTargetsKernel(unittest.TestCase):
    def test_single_world_identity_pose(self):
        wp.init()
        # One world, object at (1, 2, 3) with identity body_q, zero body_com.
        body_q = wp.array([wp.transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))], dtype=wp.transform)
        body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        body_world_start = wp.array([0], dtype=wp.int32)
        world_hs = wp.array([0.01], dtype=wp.float32)

        spec_pos_offset_fractional = wp.array([wp.vec3(0.0, 0.0, 0.5)], dtype=wp.vec3)
        spec_pos_offset_absolute = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        spec_quat_offset = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
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
                spec_pos_offset_fractional,
                spec_pos_offset_absolute,
                spec_quat_offset,
                spec_ctrl,
                base_ee_rot,
            ],
            outputs=[grasp_pos, grasp_rot, grasp_ctrl],
        )

        # fractional.z = 0.5, hs = 0.01 -> 0.005 m above COM = body_q.t + (0, 0, 0.005)
        np.testing.assert_allclose(grasp_pos.numpy()[0], [1.0, 2.0, 3.005], atol=1e-6)
        np.testing.assert_allclose(grasp_rot.numpy()[0], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(float(grasp_ctrl.numpy()[0]), 123.0, places=4)

    def test_com_offset_is_applied(self):
        wp.init()
        body_q = wp.array([wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))], dtype=wp.transform)
        body_com = wp.array([wp.vec3(0.1, 0.2, 0.3)], dtype=wp.vec3)
        body_world_start = wp.array([0], dtype=wp.int32)
        world_hs = wp.array([0.01], dtype=wp.float32)

        spec_pos_offset_fractional = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        spec_pos_offset_absolute = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        spec_quat_offset = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
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
                spec_pos_offset_fractional,
                spec_pos_offset_absolute,
                spec_quat_offset,
                spec_ctrl,
                base_ee_rot,
            ],
            outputs=[grasp_pos, grasp_rot, grasp_ctrl],
        )

        # With zero pos offsets and identity quat, grasp_pos = COM world = body_q.t + body_com.
        np.testing.assert_allclose(grasp_pos.numpy()[0], [0.1, 0.2, 0.3], atol=1e-6)

    def test_rotation_composition(self):
        # Verifies grasp_rot = body_q.q * base_ee_rot * quat_offset with non-identity base_ee_rot.
        wp.init()
        # body_q: identity rotation. base_ee_rot: 90 deg about Y. quat_offset: identity.
        # Expected: 90 deg about Y in world frame -> quat (0, sin45, 0, cos45).
        body_q = wp.array([wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))], dtype=wp.transform)
        body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        body_world_start = wp.array([0], dtype=wp.int32)
        world_hs = wp.array([0.01], dtype=wp.float32)

        spec_pos_offset_fractional = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        spec_pos_offset_absolute = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
        spec_quat_offset = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
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
                spec_pos_offset_fractional,
                spec_pos_offset_absolute,
                spec_quat_offset,
                spec_ctrl,
                base_ee_rot,
            ],
            outputs=[grasp_pos, grasp_rot, grasp_ctrl],
        )

        np.testing.assert_allclose(grasp_rot.numpy()[0], [0.0, sin45, 0.0, cos45], atol=1e-6)


class TestGraspTargetsMatchReference(unittest.TestCase):
    """Regression: compute_grasp_targets kernel output matches a host-side
    reference built from the same composition.

    The reference uses ``wp.quat_rotate`` and ``wp.quat`` multiplication on
    the host -- not a hand-rolled numpy implementation -- so the test
    validates *our* composition order (COM offset, base_ee_rot baking,
    quat_offset stacking) without re-testing Warp's quaternion math.
    """

    def test_kernel_matches_host_reference(self):
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
        base_ee_rot = example.base_ee_rot

        # One-shot CPU read of the per-world spec arrays.
        spec_pos_offset_fractional = example.spec.pos_offset_fractional.numpy()
        spec_pos_offset_absolute = example.spec.pos_offset_absolute.numpy()
        spec_quat_offset = example.spec.quat_offset.numpy()
        expected_ctrl = example.spec.ctrl.numpy().astype(np.float32)

        expected_pos = np.zeros((wc, 3), dtype=np.float32)
        expected_rot = np.zeros((wc, 4), dtype=np.float32)
        for i in range(wc):
            obj_global = int(body_ws_np[i]) + example.object_body_offset
            body_q = wp.transform(*body_q_np[obj_global])
            com_local = wp.vec3(*body_com_np[obj_global])
            hs_i = float(example.world_half_sizes[i, 0])
            pos_frac = wp.vec3(*spec_pos_offset_fractional[i])
            pos_abs = wp.vec3(*spec_pos_offset_absolute[i])
            quat_offset = wp.quat(*spec_quat_offset[i])

            pos = wp.transform_point(body_q, com_local + pos_frac * hs_i + pos_abs)
            rot = wp.transform_get_rotation(body_q) * base_ee_rot * quat_offset
            expected_pos[i] = [pos[0], pos[1], pos[2]]
            expected_rot[i] = [rot[0], rot[1], rot[2], rot[3]]

        np.testing.assert_allclose(
            grasp_pos, expected_pos, atol=1e-5, err_msg="grasp_pos kernel output disagrees with host reference"
        )
        np.testing.assert_allclose(
            grasp_rot, expected_rot, atol=1e-5, err_msg="grasp_rot kernel output disagrees with host reference"
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


if __name__ == "__main__":
    unittest.main()
