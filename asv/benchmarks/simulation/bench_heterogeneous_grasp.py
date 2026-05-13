# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ASV regression benchmark for the heterogeneous-grasp example.

Drives the example through GraspProbe (the same probe the unittest uses) to
expose both per-step wallclock and the end-of-episode success rate. The
class-level ``params`` grid covers world counts and collision modes; ASV
runs ``setup`` once per cell.
"""

import sys

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

import newton.examples as nex

try:
    from newton.examples.robot.example_robot_heterogeneous_grasp import Example as _GraspExample
    from newton.examples.robot.example_robot_heterogeneous_grasp import TaskType as _TaskType
    from newton.tests.test_object_centric_grasp import GraspProbe as _GraspProbe
except ModuleNotFoundError:  # pragma: no cover
    _GraspExample = _GraspProbe = _TaskType = None  # type: ignore[assignment]


class HeterogeneousGraspBenchmark:
    """Per-step time + episode success rate for the heterogeneous-grasp example."""

    params = ([12, 64], ["mujoco", "newton_default"])
    param_names = ["world_count", "collision_mode"]
    timeout = 600
    repeat = 2
    number = 1

    def setup(self, world_count, collision_mode):
        if _GraspExample is None:
            raise SkipNotImplemented
        parser = _GraspExample.create_parser()
        sys.argv = [
            "asv",
            "--viewer",
            "null",
            "--world-count",
            str(world_count),
            "--num-frames",
            "700",
            "--collision-mode",
            collision_mode,
        ]
        viewer, args = nex.init(parser)
        self.probe = _GraspProbe(
            world_count=world_count,
            task_state_count=int(_TaskType.DONE),
            hold_state=int(_TaskType.HOLD),
        )
        self.example = _GraspExample(viewer, args, probe=self.probe)
        # Warmup past first-step CUDA-graph capture / kernel-load cost so the
        # timed loop measures steady-state throughput.
        for _ in range(20):
            self.example.step()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, world_count, collision_mode):
        for _ in range(100):
            self.example.step()
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_success_rate(self, world_count, collision_mode):
        # State machine totals 5.5 s = 550 frames at 100 Hz. 700 (matching the
        # regression test) clears HOLD with headroom. setup() already ran the
        # 20-frame warmup, so we step the remaining 680 here.
        for _ in range(680):
            self.example.step()
        return float(self.probe.on_finish(self.example)["success_rate"])

    track_success_rate.unit = "fraction"


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.parse_known_args()
    run_benchmark(HeterogeneousGraspBenchmark)
