# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the heterogeneous-grasp example.

Spins up the full Example, runs the grasp sequence to completion, and
asserts that at least ``MIN_SUCCESS_RATE`` worlds raised their object by
``LIFT_THRESHOLD_M`` above the table top. Future PRs extend this with
per-collision-mode coverage and richer success metrics.
"""

import sys
import unittest

import numpy as np
import warp as wp

import newton.examples as nex
from newton.examples.robot.example_robot_heterogeneous_grasp import (
    _TABLE_HEIGHT,
)
from newton.examples.robot.example_robot_heterogeneous_grasp import (
    Example as GraspExample,
)


class TestHeterogeneousGraspLift(unittest.TestCase):
    """At least half the worlds lift their object above ``LIFT_THRESHOLD_M``."""

    WORLD_COUNT = 12
    NUM_FRAMES = 700
    SEED = 42
    COLLISION_MODE = "newton_hydroelastic"
    LIFT_THRESHOLD_M = 0.03  # measured above the table top
    MIN_SUCCESS_RATE = 0.50

    def test_most_worlds_lift_object(self):
        wp.init()
        parser = GraspExample.create_parser()
        sys.argv = [
            "test",
            "--viewer",
            "null",
            "--world-count",
            str(self.WORLD_COUNT),
            "--num-frames",
            str(self.NUM_FRAMES),
            "--seed",
            str(self.SEED),
            "--collision-mode",
            self.COLLISION_MODE,
        ]
        viewer, args = nex.init(parser)
        example = GraspExample(viewer, args)
        for _ in range(self.NUM_FRAMES):
            example.step()

        body_q_np = example.state_0.body_q.numpy()
        body_ws_np = example.model.body_world_start.numpy()
        obj_z = np.array(
            [body_q_np[int(body_ws_np[w]) + example.object_body_offset, 2] for w in range(self.WORLD_COUNT)]
        )
        lifted = int((obj_z > _TABLE_HEIGHT + self.LIFT_THRESHOLD_M).sum())
        success_rate = lifted / self.WORLD_COUNT
        self.assertGreaterEqual(
            success_rate,
            self.MIN_SUCCESS_RATE,
            msg=f"Lift success {success_rate:.2f} below {self.MIN_SUCCESS_RATE} ({lifted}/{self.WORLD_COUNT} worlds)",
        )


if __name__ == "__main__":
    unittest.main()
