# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab Factory OSC step-response benchmark.

Drives Factory's OSC with a fixed schedule of step-pose targets (5 cm in
x/y/z, +/- 30 deg around each axis) and records per-control-tick TCP
state, target pose, joint torques, and gains. The trained policy is
NOT loaded - the script is a pure controller probe.

The probe bypasses Factory's action machinery entirely: after a normal
reset (which uses IK to drive the arm to a Factory home pose), we call
``generate_ctrl_signals`` directly with our test target each substep
and advance physics via Factory's ``step_sim_no_action`` helper.

Output: scripts/osc_isaaclab_steps.json with one entry per trial.

Usage:
    ./isaaclab.sh -p scripts/osc_step_response.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


TASK = "Isaac-Factory-NutThread-Direct-v0"
PROBE_OUTPUT = os.path.join(os.path.dirname(__file__), "osc_isaaclab_steps.json")

# How many control ticks to record per trial. With decimation=8 over 120 Hz
# physics, env-step rate = 15 Hz -> 60 ticks = 4 s.
TRIAL_TICKS = 60
# Optional return-to-home settling between trials.
RETURN_TICKS = 30


# ---------------------------------------------------------------------------
# Plain quaternion helpers (xyzw). We avoid IsaacLab's torch_utils so this
# script remains compact and easy to port. IsaacLab buffers use wxyz; convert
# at the boundary with _wxyz_to_xyzw / _xyzw_to_wxyz.
# ---------------------------------------------------------------------------


def _quat_from_axis_angle(axis: list[float], angle_rad: float) -> list[float]:
    half = angle_rad * 0.5
    s = math.sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_angle_deg(qa: list[float], qb: list[float]) -> float:
    dot = abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3])
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _vec3_dist_mm(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))) * 1000.0


def _wxyz_to_xyzw(q: list[float]) -> list[float]:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _xyzw_to_wxyz(q: list[float]) -> list[float]:
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


# ---------------------------------------------------------------------------
# Trial schedule
# ---------------------------------------------------------------------------


def _build_trials(pos_step_m: float = 0.05, rot_step_deg: float = 30.0) -> list[dict]:
    rot_step_rad = math.radians(rot_step_deg)
    trials = []
    for axis_name, axis_vec in (("x", [1.0, 0.0, 0.0]), ("y", [0.0, 1.0, 0.0]), ("z", [0.0, 0.0, 1.0])):
        for sign, sign_str in ((+1.0, "+"), (-1.0, "-")):
            trials.append(
                {
                    "name": f"pos_{sign_str}{axis_name}_{int(pos_step_m * 100)}cm",
                    "kind": "pos",
                    "axis": axis_name,
                    "sign": sign,
                    "delta_vec": [sign * pos_step_m * v for v in axis_vec],
                }
            )
    for axis_name, axis_vec in (("x", [1.0, 0.0, 0.0]), ("y", [0.0, 1.0, 0.0]), ("z", [0.0, 0.0, 1.0])):
        for sign, sign_str in ((+1.0, "+"), (-1.0, "-")):
            trials.append(
                {
                    "name": f"rot_{sign_str}{axis_name}_{int(rot_step_deg)}deg",
                    "kind": "rot",
                    "axis": axis_name,
                    "sign": sign,
                    "axis_vec": axis_vec,
                    "angle_rad": sign * rot_step_rad,
                }
            )
    return trials


def _compute_target_pose(
    trial: dict, home_pos: list[float], home_quat_xyzw: list[float]
) -> tuple[list[float], list[float]]:
    if trial["kind"] == "pos":
        target_pos = [home_pos[i] + trial["delta_vec"][i] for i in range(3)]
        return target_pos, list(home_quat_xyzw)
    delta_q = _quat_from_axis_angle(trial["axis_vec"], trial["angle_rad"])
    return list(home_pos), _quat_mul(delta_q, home_quat_xyzw)


# ---------------------------------------------------------------------------
# Step-response metrics. Project the trajectory onto the target direction
# so we report a scalar (distance traveled toward target).
# ---------------------------------------------------------------------------


def _step_response_metrics(target: list[float], measured: list[list[float]], control_dt: float) -> dict:
    if not measured or len(measured) < 2:
        return {}
    p0 = measured[0]
    direction = [target[i] - p0[i] for i in range(3)]
    target_dist = math.sqrt(sum(d * d for d in direction))
    if target_dist < 1e-6:
        return {"target_dist_mm": 0.0}
    direction = [d / target_dist for d in direction]
    progress = []
    for p in measured:
        dp = [p[i] - p0[i] for i in range(3)]
        progress.append(sum(dp[i] * direction[i] for i in range(3)))
    final = progress[-1]
    peak = max(progress)
    overshoot_mm = max(0.0, peak - target_dist) * 1000.0
    rise_t = None
    for i, p in enumerate(progress):
        if p >= 0.9 * target_dist:
            rise_t = i * control_dt
            break
    settle_t = None
    settle_band = 0.05 * target_dist
    for i in range(len(progress) - 1, -1, -1):
        if abs(progress[i] - target_dist) > settle_band:
            settle_t = (i + 1) * control_dt if i + 1 < len(progress) else None
            break
    if settle_t is None:
        settle_t = 0.0
    return {
        "target_dist_mm": target_dist * 1000.0,
        "rise_time_s": rise_t,
        "settling_time_s": settle_t,
        "overshoot_mm": overshoot_mm,
        "final_proj_error_mm": (target_dist - final) * 1000.0,
    }


# ---------------------------------------------------------------------------
# OSC driver: bypass the action machinery, call generate_ctrl_signals each
# substep with our fixed test target, advance via step_sim_no_action.
# ---------------------------------------------------------------------------


def run_one_tick(env_unwrapped, target_pos_t: torch.Tensor, target_quat_wxyz_t: torch.Tensor) -> None:
    """One env step (= ``decimation`` substeps) at the given fixed target."""
    for _ in range(env_unwrapped.cfg.decimation):
        env_unwrapped.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=target_pos_t,
            ctrl_target_fingertip_midpoint_quat=target_quat_wxyz_t,
            ctrl_target_gripper_dof_pos=0.0,
        )
        env_unwrapped.step_sim_no_action()


def main():
    parser = argparse.ArgumentParser(description="OSC step-response benchmark on IsaacLab Factory.")
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
    parser.add_argument("--trial-ticks", type=int, default=TRIAL_TICKS)
    parser.add_argument("--return-ticks", type=int, default=RETURN_TICKS)
    parser.add_argument("--pos-step", type=float, default=0.05, help="Position step magnitude [m].")
    parser.add_argument("--rot-step-deg", type=float, default=30.0, help="Rotation step magnitude [deg].")
    add_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent)
    env_cfg.scene.num_envs = 1
    # Disable randomization so the post-reset task home is deterministic
    # and step-response trajectories are repeatable for benchmarking.
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    if hasattr(env_cfg.task, "fixed_asset_init_pos_noise"):
        env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    if hasattr(env_cfg.task, "fixed_asset_init_orn_range_deg"):
        env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()

        u = env.unwrapped
        device = u.device

        physics_dt = u.physics_dt
        decimation = u.cfg.decimation
        control_dt = float(physics_dt * decimation)
        print(f"[step] control_dt = {control_dt:.4f} s ({1.0 / control_dt:.1f} Hz), decimation = {decimation}")

        # Settle so the IK driven inside reset has converged before we
        # record the home pose. With noise disabled this also gives a
        # deterministic home that downstream Newton runs can match.
        for _ in range(8):
            env.step(torch.zeros(env.action_space.shape, device=device))

        # Refresh intermediate values so fingertip_midpoint_pos is current.
        u._compute_intermediate_values(dt=physics_dt)
        home_pos = u.fingertip_midpoint_pos[0].cpu().tolist()
        home_quat_wxyz = u.fingertip_midpoint_quat[0].cpu().tolist()
        home_quat_xyzw = _wxyz_to_xyzw(home_quat_wxyz)
        # Also capture the actual settled joint config so Newton can
        # initialize at the identical pose.
        home_joint_pos = u._robot.data.joint_pos.torch[0].cpu().tolist()
        home_joint_names = list(u._robot.joint_names)
        print(f"[step] home_pos (m) = {home_pos}")
        print(f"[step] home_quat (xyzw) = {home_quat_xyzw}")
        print(f"[step] home_joint_pos = {home_joint_pos}")

        # Cache home tensors for the return-to-home phase.
        home_pos_t = torch.tensor(home_pos, device=device).unsqueeze(0)
        home_quat_t = torch.tensor(home_quat_wxyz, device=device).unsqueeze(0)

        # Read OSC gains and other config for the record.
        task_prop_gains = u.task_prop_gains[0].cpu().tolist() if hasattr(u, "task_prop_gains") else None
        task_deriv_gains = u.task_deriv_gains[0].cpu().tolist() if hasattr(u, "task_deriv_gains") else None
        print(f"[step] task_prop_gains  = {task_prop_gains}")
        print(f"[step] task_deriv_gains = {task_deriv_gains}")

        trials = _build_trials(args_cli.pos_step, args_cli.rot_step_deg)
        results = {
            "task": args_cli.task,
            "control_dt": control_dt,
            "control_hz": 1.0 / control_dt,
            "trial_ticks": args_cli.trial_ticks,
            "return_ticks": args_cli.return_ticks,
            "pos_step_m": args_cli.pos_step,
            "rot_step_deg": args_cli.rot_step_deg,
            "home_pos": home_pos,
            "home_quat_xyzw": home_quat_xyzw,
            "home_joint_pos": home_joint_pos,
            "home_joint_names": home_joint_names,
            "task_prop_gains": task_prop_gains,
            "task_deriv_gains": task_deriv_gains,
            "trials": [],
        }

        for trial in trials:
            target_pos, target_quat_xyzw = _compute_target_pose(trial, home_pos, home_quat_xyzw)
            target_quat_wxyz = _xyzw_to_wxyz(target_quat_xyzw)
            target_pos_t = torch.tensor(target_pos, device=device).unsqueeze(0)
            target_quat_t = torch.tensor(target_quat_wxyz, device=device).unsqueeze(0)

            print(
                f"\n[step] trial {trial['name']:<22s} "
                f"target_pos={[f'{v:+.3f}' for v in target_pos]}  "
                f"target_quat_xyzw={[f'{v:+.3f}' for v in target_quat_xyzw]}"
            )

            # Return to home first.
            for _ in range(args_cli.return_ticks):
                run_one_tick(u, home_pos_t, home_quat_t)

            tick_records = []
            for tick in range(args_cli.trial_ticks):
                run_one_tick(u, target_pos_t, target_quat_t)
                t_now = (tick + 1) * control_dt
                meas_pos = u.fingertip_midpoint_pos[0].cpu().tolist()
                meas_quat_wxyz = u.fingertip_midpoint_quat[0].cpu().tolist()
                meas_quat_xyzw = _wxyz_to_xyzw(meas_quat_wxyz)
                joint_torque = u.joint_torque[0].cpu().tolist() if hasattr(u, "joint_torque") else None
                tick_records.append(
                    {
                        "t": t_now,
                        "measured_pos": meas_pos,
                        "measured_quat_xyzw": meas_quat_xyzw,
                        "joint_torque": joint_torque,
                    }
                )

            measured_traj = [r["measured_pos"] for r in tick_records]
            metrics = _step_response_metrics(target_pos, measured_traj, control_dt)
            metrics["final_pos_err_mm"] = _vec3_dist_mm(target_pos, measured_traj[-1])
            metrics["final_rot_err_deg"] = _quat_angle_deg(target_quat_xyzw, tick_records[-1]["measured_quat_xyzw"])
            print(
                f"[step] {trial['name']:<22s} "
                f"target_dist={metrics.get('target_dist_mm', 0):.1f} mm  "
                f"final_pos_err={metrics['final_pos_err_mm']:.2f} mm  "
                f"final_rot_err={metrics['final_rot_err_deg']:.2f} deg  "
                f"rise_t={metrics.get('rise_time_s')}  "
                f"settle_t={metrics.get('settling_time_s'):.2f}s  "
                f"overshoot={metrics.get('overshoot_mm', 0.0):.2f} mm"
            )

            results["trials"].append(
                {
                    "name": trial["name"],
                    "kind": trial["kind"],
                    "axis": trial["axis"],
                    "sign": trial["sign"],
                    "target_pos": target_pos,
                    "target_quat_xyzw": target_quat_xyzw,
                    "metrics": metrics,
                    "ticks": tick_records,
                }
            )

        with open(PROBE_OUTPUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[step] Wrote {PROBE_OUTPUT}")
        env.close()


if __name__ == "__main__":
    main()
