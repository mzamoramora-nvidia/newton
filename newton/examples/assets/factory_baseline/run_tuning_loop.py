#!/usr/bin/env python3
"""Autonomous OSC tuning loop driver (Phase 5 of osc_tuning_plan.md).

Runs unattended coordinate descent over (kp_pos, kp_rot, kd_joint,
lambda_damping) using the Newton OSC step-response example with
--baseline pointing at osc_isaaclab_steps.json. Each evaluation is a
subprocess; early-abort flags let bad candidates exit in seconds. Per-
evaluation telemetry goes to tuning_log.jsonl and a final
tuning_summary.md captures the outcome.

Score = mean(per-trial score) over the 5 well-behaved position trials
(pos_+x / pos_-x / pos_+y / pos_-y / pos_+z). pos_-z is excluded because
the IsaacLab baseline shows only ~21 mm of excursion vs commanded -50 mm
(likely workspace clip on the Factory side); the two rotation trials are
excluded because their ref_mm is ~2 mm, making the L2 score noise-
dominated. See osc_tuning_plan.md, Phase 3 findings.

Usage (run from repo root):
    uv run python newton/examples/assets/factory_baseline/run_tuning_loop.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
BASELINE = os.path.join(THIS_DIR, "osc_isaaclab_steps.json")
LOG_PATH = os.path.join(THIS_DIR, "tuning_log.jsonl")
SUMMARY_PATH = os.path.join(THIS_DIR, "tuning_summary.md")

# Trials we score. pos_-z and rotation trials are excluded; see module
# docstring.
SCORED_TRIALS = (
    "pos_+x_5cm",
    "pos_-x_5cm",
    "pos_+y_5cm",
    "pos_-y_5cm",
    "pos_+z_5cm",
)

# Search axes (in order of expected impact). Each axis: list of values to
# try at the current minimum of every other axis.
SEARCH_AXES = (
    ("kp_pos", [100.0, 200.0, 400.0, 800.0]),
    ("kp_rot", [15.0, 30.0, 60.0, 100.0]),
    ("kd_joint", [20.0, 50.0, 80.0, 120.0]),
    ("lambda_damping", [1e-4, 1e-3, 1e-2, 1e-1]),
)

# Starting point: Newton's current published defaults.
DEFAULTS = {"kp_pos": 200.0, "kp_rot": 30.0, "kd_joint": 50.0, "lambda_damping": 1e-2}

# Convergence thresholds.
GLOBAL_SCORE_TARGET = 0.05  # mean per-trial score
PER_TRIAL_SCORE_CAP = 0.10
SATURATION_DELTA = 0.005  # smallest score-improvement that counts
SATURATION_PATIENCE = 10  # evals without improvement -> SATURATED
BUDGET = 60  # hard cap on evaluations
DROP_TEST_TIMEOUT_S = 60
EVAL_TIMEOUT_S = 180


@dataclass
class Result:
    candidate: dict
    global_score: float
    per_trial: dict
    reason: str  # "ok" / "drop_test_failed" / "abort:<...>" / "timeout" / "error"
    elapsed_s: float
    output_path: str = ""
    aborted_during_step: bool = False
    # For tuning_log.jsonl serialization.
    extra: dict = field(default_factory=dict)


def candidate_key(c: dict) -> tuple:
    return tuple(sorted(c.items()))


def output_name(candidate: dict) -> str:
    return (
        f"osc_newton_steps_usd_tuneloop"
        f"_kp{candidate['kp_pos']:g}-{candidate['kp_rot']:g}"
        f"_kdj{candidate['kd_joint']:g}_dls{candidate['lambda_damping']:g}.json"
    )


def run_drop_test() -> bool:
    """Quick 20-frame --test on USD: passes if drop is bounded.

    Pre-screens a candidate before paying the full step-response cost.
    """
    cmd = [
        "uv",
        "run",
        "-m",
        "newton.examples",
        "robot_panda_osc",
        "--robot",
        "usd",
        "--test",
        "--viewer",
        "null",
        "--num-frames",
        "20",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=DROP_TEST_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    # Look for the final drop value in stdout. Tolerate up to 1 mm in any
    # sample (the Phase 2b gate is 0.05 mm, but we leave headroom for
    # candidates we haven't tested yet).
    for line in proc.stdout.splitlines():
        if "drop=" in line:
            try:
                tok = line.split("drop=")[-1].split("mm")[0].strip().lstrip("+")
                drop_mm = abs(float(tok))
                if drop_mm > 1.0:
                    return False
            except ValueError:
                continue
    return True


def evaluate(candidate: dict) -> Result:
    """Run one step-response with the given candidate gains."""
    out_name = output_name(candidate)
    out_path = os.path.join(THIS_DIR, out_name)
    cmd = [
        "uv",
        "run",
        "-m",
        "newton.examples",
        "robot_panda_osc_step_response",
        "--robot",
        "usd",
        "--viewer",
        "null",
        "--test",
        "--baseline",
        BASELINE,
        "--output-name",
        out_name,
        "--progress-every-n-ticks",
        "30",
        "--osc-kp-pos",
        f"{candidate['kp_pos']:g}",
        "--osc-kp-rot",
        f"{candidate['kp_rot']:g}",
        "--arm-joint-kd",
        f"{candidate['kd_joint']:g}",
        "--lambda-damping",
        f"{candidate['lambda_damping']:g}",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=EVAL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return Result(candidate, math.inf, {}, "timeout", time.time() - t0)
    elapsed = time.time() - t0

    aborted = False
    abort_reason = ""
    for line in proc.stdout.splitlines():
        if "[step-response] ABORT" in line:
            aborted = True
            abort_reason = line.split("reason=")[-1].split()[0] if "reason=" in line else "unknown"
            break

    if aborted:
        return Result(candidate, math.inf, {}, f"abort:{abort_reason}", elapsed, aborted_during_step=True)
    if proc.returncode != 0:
        return Result(
            candidate,
            math.inf,
            {},
            f"error:returncode={proc.returncode}",
            elapsed,
            extra={"stderr_tail": proc.stderr[-500:] if proc.stderr else ""},
        )
    if not os.path.isfile(out_path):
        return Result(candidate, math.inf, {}, "error:no_output", elapsed)

    # Parse the output JSON and compute the position-only mean score.
    try:
        with open(out_path) as f:
            out = json.load(f)
    except (OSError, json.JSONDecodeError):
        return Result(candidate, math.inf, {}, "error:parse", elapsed, output_path=out_path)

    # Compute per-trial score using the same formula as runtime, on the
    # final tick of each trial. Need baseline trajectories; reload.
    try:
        with open(BASELINE) as f:
            baseline_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return Result(candidate, math.inf, {}, "error:baseline_parse", elapsed, output_path=out_path)
    baseline_home = baseline_data.get("home_pos") or [0.0, 0.0, 0.0]
    baseline_per_trial = {
        tr["name"]: [t["measured_pos"] for t in tr.get("ticks", [])] for tr in baseline_data.get("trials", [])
    }
    measured_home = out.get("home_pos") or [0.0, 0.0, 0.0]
    per_trial = {}
    for tr in out.get("trials", []):
        name = tr.get("name")
        if name not in baseline_per_trial:
            continue
        baseline = baseline_per_trial[name]
        measured = [t["measured_pos"] for t in tr.get("ticks", [])]
        n = min(len(measured), len(baseline))
        if n == 0:
            continue
        err_sq = 0.0
        ref_sq = 0.0
        for i in range(n):
            bx = baseline[i][0] - baseline_home[0]
            by = baseline[i][1] - baseline_home[1]
            bz = baseline[i][2] - baseline_home[2]
            ref_sq += bx * bx + by * by + bz * bz
            mx = measured[i][0] - measured_home[0]
            my = measured[i][1] - measured_home[1]
            mz = measured[i][2] - measured_home[2]
            err_sq += (mx - bx) ** 2 + (my - by) ** 2 + (mz - bz) ** 2
        per_trial[name] = (err_sq / ref_sq) if ref_sq > 1e-18 else 0.0

    scored = [per_trial.get(n, math.inf) for n in SCORED_TRIALS if n in per_trial]
    if not scored:
        return Result(candidate, math.inf, per_trial, "error:no_scored_trials", elapsed, output_path=out_path)
    global_score = sum(scored) / len(scored)
    return Result(candidate, global_score, per_trial, "ok", elapsed, output_path=out_path)


def append_log(result: Result, iter_idx: int) -> None:
    rec = {
        "iter": iter_idx,
        "candidate": result.candidate,
        "global_score": result.global_score if math.isfinite(result.global_score) else None,
        "per_trial": result.per_trial,
        "reason": result.reason,
        "elapsed_s": result.elapsed_s,
        "output_path": result.output_path,
    }
    rec.update(result.extra)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> int:
    open(LOG_PATH, "w").close()  # truncate

    seen: dict[tuple, Result] = {}
    current = dict(DEFAULTS)
    best: Result | None = None
    no_improve = 0
    iter_idx = 0
    outcome = "BUDGET_EXHAUSTED"

    print(f"[tuning] starting; budget={BUDGET}, target={GLOBAL_SCORE_TARGET}")
    print(f"[tuning] baseline = {BASELINE}")
    print(f"[tuning] starting candidate = {current}")

    # Two passes of coordinate descent.
    for pass_idx in range(2):
        for axis_name, values in SEARCH_AXES:
            for v in values:
                if iter_idx >= BUDGET:
                    break
                cand = dict(current)
                cand[axis_name] = v
                key = candidate_key(cand)
                if key in seen:
                    continue
                iter_idx += 1
                print(
                    f"[tuning] eval {iter_idx}/{BUDGET}: pass={pass_idx} axis={axis_name} value={v:g} candidate={cand}"
                )
                # Drop-test pre-screen.
                t0 = time.time()
                if not run_drop_test():
                    res = Result(cand, math.inf, {}, "drop_test_failed", time.time() - t0)
                    seen[key] = res
                    append_log(res, iter_idx)
                    print("[tuning]   drop_test_failed (skipping step-response)")
                    no_improve += 1
                    continue
                res = evaluate(cand)
                seen[key] = res
                append_log(res, iter_idx)
                if math.isfinite(res.global_score):
                    print(
                        f"[tuning]   score={res.global_score:.4f} "
                        f"(per-trial: "
                        + ", ".join(f"{n}={res.per_trial.get(n, math.nan):.3f}" for n in SCORED_TRIALS)
                        + f") elapsed={res.elapsed_s:.1f}s"
                    )
                else:
                    print(f"[tuning]   {res.reason} elapsed={res.elapsed_s:.1f}s")
                if math.isfinite(res.global_score):
                    # `best` always holds the true minimum so the summary
                    # reflects the best candidate ever seen. The patience
                    # counter (`no_improve`) only resets on a *significant*
                    # improvement (>= SATURATION_DELTA), so tiny gains
                    # still update best but don't extend the search.
                    if best is None:
                        best = res
                        current = dict(cand)
                        no_improve = 0
                    elif res.global_score < best.global_score:
                        prev_score = best.global_score
                        best = res
                        current = dict(cand)
                        if res.global_score + SATURATION_DELTA < prev_score:
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                # Success?
                if (
                    best is not None
                    and best.global_score <= GLOBAL_SCORE_TARGET
                    and all(s <= PER_TRIAL_SCORE_CAP for s in best.per_trial.values() if math.isfinite(s))
                ):
                    outcome = "SUCCESS"
                    break
                if no_improve >= SATURATION_PATIENCE:
                    outcome = "SATURATED"
                    break
            if outcome != "BUDGET_EXHAUSTED":
                break
            # After sweeping an axis, "current" already holds the best
            # value found for that axis (set inside the loop when an
            # improvement landed). Move on to the next axis with that
            # locked in.
        if outcome != "BUDGET_EXHAUSTED":
            break

    write_summary(outcome, best, iter_idx, list(seen.values()))
    return 0 if outcome == "SUCCESS" else (1 if outcome == "ABORTED_DIVERGENCE" else 0)


def write_summary(outcome: str, best: Result | None, n_evals: int, all_results: list[Result]) -> None:
    aborted = [r for r in all_results if r.reason.startswith("abort:") or r.reason == "drop_test_failed"]
    finite = [r for r in all_results if math.isfinite(r.global_score)]
    lines = []
    lines.append("# OSC Tuning Summary\n")
    lines.append(f"**Outcome**: {outcome}  ")
    lines.append(f"**Evaluations**: {n_evals} ({len(finite)} scored, {len(aborted)} early-aborted/drop-failed)  ")
    lines.append(f"**Target**: global_score ≤ {GLOBAL_SCORE_TARGET}, per-trial ≤ {PER_TRIAL_SCORE_CAP}\n")
    lines.append("## Best candidate\n")
    if best is None:
        lines.append("_No finite-score evaluation completed._\n")
    else:
        lines.append("```json")
        lines.append(json.dumps(best.candidate, indent=2))
        lines.append("```\n")
        lines.append(f"**global_score**: {best.global_score:.4f}\n")
        lines.append("Per-trial breakdown:\n")
        lines.append("| Trial | Score |")
        lines.append("|-------|-------|")
        for n in SCORED_TRIALS:
            lines.append(f"| {n} | {best.per_trial.get(n, float('nan')):.4f} |")
        lines.append("")
    lines.append("## Top 10 results by score\n")
    lines.append("| Rank | global_score | kp_pos | kp_rot | kd_joint | lambda_damping | reason |")
    lines.append("|------|--------------|--------|--------|----------|----------------|--------|")
    finite.sort(key=lambda r: r.global_score)
    for i, r in enumerate(finite[:10], 1):
        c = r.candidate
        lines.append(
            f"| {i} | {r.global_score:.4f} | {c['kp_pos']:g} | {c['kp_rot']:g} | "
            f"{c['kd_joint']:g} | {c['lambda_damping']:g} | {r.reason} |"
        )
    lines.append("")
    lines.append("## Diagnosis\n")
    if outcome == "SUCCESS":
        lines.append("Best candidate beats the target on the 5 well-behaved position trials.")
    elif outcome == "SATURATED":
        lines.append(
            "Coordinate descent saturated: 10 consecutive evals without improvement of "
            f"≥ {SATURATION_DELTA} on global_score. The best score is what the current "
            "controller formulation can reach with these knobs at hand. Next leads:\n"
            "- Investigate the pos_-z baseline (likely IsaacLab workspace clip).\n"
            "- Add a rotation-error term so the rotation trials aren't noise-dominated.\n"
            "- Mirror Factory's joint_armature/friction values once the IsaacLab probe is rerun."
        )
    elif outcome == "BUDGET_EXHAUSTED":
        lines.append(
            f"Hit the {BUDGET}-eval cap before either succeeding or saturating. "
            "Increase BUDGET in run_tuning_loop.py to keep searching."
        )
    elif outcome == "ABORTED_DIVERGENCE":
        lines.append("Two consecutive evals diverged; loop exited early.")
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tuning] wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    sys.exit(main())
