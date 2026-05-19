# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the heterogeneous-grasp example.

Spins up the full Example, runs the grasp sequence to completion, and asserts
that at least ``MIN_SUCCESS_RATE`` worlds raised their object by at least
``LIFT_DELTA_M`` above its initial Z. Future PRs extend this with per-mode
coverage and richer success metrics.

The example builds hydroelastic SDFs which require a CUDA device, so the test
is registered per CUDA device only via :func:`get_test_devices`.
"""

import sys
import unittest
import unittest.mock

import numpy as np
import warp as wp

import newton.examples as nex
from newton.examples.robot.example_robot_heterogeneous_grasp import (
    Example as GraspExample,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestHeterogeneousGraspLift(unittest.TestCase):
    """At least ``MIN_SUCCESS_RATE`` worlds raise their object by ``LIFT_DELTA_M``."""

    WORLD_COUNT = 24  # matches the example's default world_count
    NUM_FRAMES = 700
    SEED = 42
    COLLISION_MODE = "newton_hydroelastic"
    LIFT_DELTA_M = 0.07  # nominal lift is 0.10 m; allow ~3 cm of settle
    MIN_SUCCESS_RATE = 0.80


def _read_object_z(example):
    """Per-world object body-origin Z, as a float32 numpy array."""
    body_q_np = example.state_0.body_q.numpy()
    body_ws_np = example.model.body_world_start.numpy()
    return np.array(
        [body_q_np[int(body_ws_np[w]) + example.object_body_offset, 2] for w in range(example.world_count)]
    ), body_q_np


def test_most_worlds_lift_object(test, device):
    parser = GraspExample.create_parser()
    argv = [
        "test",
        "--viewer",
        "null",
        "--world-count",
        str(test.WORLD_COUNT),
        "--num-frames",
        str(test.NUM_FRAMES),
        "--seed",
        str(test.SEED),
        "--collision-mode",
        test.COLLISION_MODE,
    ]
    with unittest.mock.patch.object(sys, "argv", argv), wp.ScopedDevice(device):
        viewer, args = nex.init(parser)
        example = GraspExample(viewer, args)
        initial_z, _ = _read_object_z(example)
        for _ in range(test.NUM_FRAMES):
            example.step()
        final_z, body_q_np = _read_object_z(example)

    # NaN guard first -- a single NaN body invalidates the lift numbers.
    nan_count = int(np.isnan(body_q_np).any(axis=-1).sum())
    test.assertEqual(nan_count, 0, msg=f"NaN detected in {nan_count} body transform(s)")

    delta = final_z - initial_z
    lifted = delta > test.LIFT_DELTA_M
    success_rate = int(lifted.sum()) / test.WORLD_COUNT
    if success_rate < test.MIN_SUCCESS_RATE:
        failing = [
            f"  W{w:2d} {example.world_shapes[w].name:<12} init={initial_z[w]:.4f} final={final_z[w]:.4f} delta={delta[w]:+.4f}"
            for w in range(test.WORLD_COUNT)
            if not lifted[w]
        ]
        test.fail(
            "Lift success {:.2%} below {:.2%} (seed={}, mode={}, device={}). Failing worlds:\n{}".format(
                success_rate,
                test.MIN_SUCCESS_RATE,
                test.SEED,
                test.COLLISION_MODE,
                device,
                "\n".join(failing),
            )
        )


# CUDA-only: the hydroelastic SDF path requires a CUDA device.
_cuda_devices = [d for d in get_test_devices() if d.is_cuda]
add_function_test(
    TestHeterogeneousGraspLift,
    "test_most_worlds_lift_object",
    test_most_worlds_lift_object,
    devices=_cuda_devices,
    check_output=False,  # the example prints joint layout / SDF setup at init time
)


if __name__ == "__main__":
    unittest.main()
