# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the live-progress + early-abort helpers used by the autonomous
OSC tuning loop.

The pure helpers ``check_abort`` and ``compute_running_trial_score`` live in
``newton.examples.robot.example_robot_panda_osc_step_response`` so the
runtime loop and these tests share one definition of the abort logic. Tests
exercise just the helpers -- no Warp / sim / IsaacLab dependency -- so they
run in milliseconds and can be wired into the tuning loop's pre-flight.
"""

from __future__ import annotations

import unittest

from newton.examples.robot.example_robot_panda_osc_step_response import (
    check_abort,
    compute_running_trial_score,
)


class TestComputeRunningTrialScore(unittest.TestCase):
    def test_zero_score_when_measured_matches_baseline(self):
        home = [0.0, 0.0, 0.2]
        baseline = [[0.01, 0.0, 0.2], [0.02, 0.0, 0.2], [0.03, 0.0, 0.2]]
        measured = list(baseline)  # exact match
        score, err_mm, ref_mm = compute_running_trial_score(measured, baseline, home)
        self.assertAlmostEqual(score, 0.0, places=12)
        self.assertAlmostEqual(err_mm, 0.0, places=12)
        self.assertGreater(ref_mm, 0.0)

    def test_score_is_normalized_to_baseline_excursion(self):
        # Baseline walks +5 cm in x, measured walks +10 cm -> 5 cm overshoot.
        # err_sq = (5cm)^2 * N, ref_sq = (5cm)^2 * N, so score should be 1.0.
        home = [0.0, 0.0, 0.0]
        baseline = [[0.05 * (i / 9.0), 0.0, 0.0] for i in range(10)]
        measured = [[0.10 * (i / 9.0), 0.0, 0.0] for i in range(10)]
        score, _, _ = compute_running_trial_score(measured, baseline, home)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_zero_score_when_baseline_is_static(self):
        home = [0.0, 0.0, 0.2]
        # Baseline holds at home -> ref_sq = 0 -> score returns 0.0.
        baseline = [list(home) for _ in range(5)]
        measured = [[0.05, 0.0, 0.2] for _ in range(5)]  # measured walks off
        score, _, _ = compute_running_trial_score(measured, baseline, home)
        self.assertEqual(score, 0.0)

    def test_truncates_to_shorter_list(self):
        home = [0.0, 0.0, 0.0]
        baseline = [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]]
        measured = [[0.01, 0.0, 0.0]]  # only one sample so far
        score, _, _ = compute_running_trial_score(measured, baseline, home)
        self.assertAlmostEqual(score, 0.0, places=12)

    def test_score_invariant_to_world_offset_via_baseline_home(self):
        # Newton at (-0.5, -0.5, 0.1), IsaacLab at (0,0,0). Both walk +5cm
        # in x. With ``baseline_home_pos`` provided, the score is zero
        # because the *excursions* match -- absolute world positions
        # differ by 0.7+ m but that's frame offset, not tracking error.
        newton_home = [0.0942, -0.4999, 0.2035]
        isaaclab_home = [0.5942, 0.0001, 0.1035]
        # Both walk +5 cm in x, sampled identically.
        n = 60
        measured = [
            [newton_home[0] + 0.05 * (k / (n - 1)), newton_home[1], newton_home[2]] for k in range(n)
        ]
        baseline = [
            [isaaclab_home[0] + 0.05 * (k / (n - 1)), isaaclab_home[1], isaaclab_home[2]]
            for k in range(n)
        ]
        # Without baseline_home_pos: score is large (different world frames).
        # With matching excursions, the absolute-position mismatch reflects
        # only the world-frame offset (~0.5 m in xy, 0.1 m in z), which when
        # squared and divided by the baseline-relative-to-newton-home
        # excursion comes out near 1.0 -- definitely not "tracking the
        # excursion well".
        bad_score, _, _ = compute_running_trial_score(measured, baseline, newton_home)
        self.assertGreater(bad_score, 0.5)
        # With baseline_home_pos: score is ~0.
        good_score, err_mm, _ = compute_running_trial_score(
            measured, baseline, newton_home, baseline_home_pos=isaaclab_home
        )
        self.assertAlmostEqual(good_score, 0.0, places=10)
        self.assertAlmostEqual(err_mm, 0.0, places=6)


class TestCheckAbort(unittest.TestCase):
    def setUp(self):
        self.home = [0.0, 0.0, 0.2]
        self.kwargs = dict(
            home_pos=self.home,
            running_score=0.05,
            consecutive_over_threshold=0,
            abort_tcp_excursion_m=0.30,
            abort_trial_score=5.0,
            abort_on_nan=True,
        )

    def test_no_abort_for_baseline_match(self):
        # TCP at home, low score, no over-threshold streak.
        self.assertIsNone(check_abort(tcp_pos=list(self.home), **self.kwargs))

    def test_abort_on_excessive_tcp_excursion(self):
        # TCP wanders 1 m off home in x -- well past the 30 cm cap.
        far_pos = [self.home[0] + 1.0, self.home[1], self.home[2]]
        reason = check_abort(tcp_pos=far_pos, **self.kwargs)
        self.assertEqual(reason, "tcp_excursion")

    def test_abort_on_nan(self):
        nan_pos = [float("nan"), 0.0, 0.2]
        reason = check_abort(tcp_pos=nan_pos, **self.kwargs)
        self.assertEqual(reason, "nan")

    def test_no_abort_on_nan_when_disabled(self):
        nan_pos = [float("nan"), 0.0, 0.2]
        kwargs = dict(self.kwargs)
        kwargs["abort_on_nan"] = False
        # NaN squashes the excursion check (max(|nan|, ...) is NaN, comparisons
        # return False); explicit nan-disable should let it through. The trial-
        # score path also needs the consecutive counter, so this is still ok.
        self.assertIsNone(check_abort(tcp_pos=nan_pos, **kwargs))

    def test_abort_on_sustained_high_score(self):
        kwargs = dict(self.kwargs)
        kwargs["running_score"] = 8.0
        kwargs["consecutive_over_threshold"] = 5
        reason = check_abort(tcp_pos=list(self.home), **kwargs)
        self.assertEqual(reason, "trial_score")

    def test_no_abort_for_brief_score_spike(self):
        kwargs = dict(self.kwargs)
        kwargs["running_score"] = 8.0
        kwargs["consecutive_over_threshold"] = 3  # below the 5-sample threshold
        self.assertIsNone(check_abort(tcp_pos=list(self.home), **kwargs))

    def test_excursion_takes_priority_over_score(self):
        kwargs = dict(self.kwargs)
        kwargs["running_score"] = 8.0
        kwargs["consecutive_over_threshold"] = 100
        far_pos = [self.home[0] + 1.0, self.home[1], self.home[2]]
        reason = check_abort(tcp_pos=far_pos, **kwargs)
        self.assertEqual(reason, "tcp_excursion")


class TestEndToEndBaselineMatchTrajectory(unittest.TestCase):
    """Synthetic 12-trial baseline = measured -> no abort across all trials."""

    def test_no_abort_on_baseline_match_all_trials(self):
        home = [0.0, 0.0, 0.2]
        for trial in range(12):
            # 60 samples per trial. Each trial walks +5 cm in a different
            # axis to mimic the real schedule.
            axis = trial % 3
            sign = 1.0 if (trial // 3) % 2 == 0 else -1.0
            baseline = []
            for k in range(60):
                p = list(home)
                p[axis] += sign * 0.05 * (k / 59.0)
                baseline.append(p)
            measured = list(baseline)
            consecutive = 0
            for k in range(60):
                score, _, _ = compute_running_trial_score(measured[: k + 1], baseline[: k + 1], home)
                self.assertLess(score, 1e-9)
                if score > 5.0:
                    consecutive += 1
                else:
                    consecutive = 0
                reason = check_abort(
                    tcp_pos=measured[k],
                    home_pos=home,
                    running_score=score,
                    consecutive_over_threshold=consecutive,
                    abort_tcp_excursion_m=0.30,
                    abort_trial_score=5.0,
                    abort_on_nan=True,
                )
                self.assertIsNone(reason, f"unexpected abort {reason} at trial {trial} k={k}")


if __name__ == "__main__":
    unittest.main()
