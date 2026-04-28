# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Print the offset between Factory's fingertip_midpoint and the Panda hand frame.

Boots ``Isaac-Factory-NutThread-Direct-v0`` (no policy, no controller),
sets the robot to a few test joint configurations, and prints:

  * world-frame pose of the panda hand body
  * world-frame pose of ``panda_fingertip_centered`` (Factory's
    ``fingertip_midpoint``)
  * the relative transform ``T_hand_local_to_fingertip`` -- this is the
    constant offset that defines Factory's TCP frame relative to the
    hand link, and is what the Newton-side compare viewer should match.

Output is also written to ``factory_tcp_probe.json`` next to the script.

Usage:
    ./isaaclab.sh -p scripts/probe_fingertip.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


TASK = "Isaac-Factory-NutThread-Direct-v0"
HAND_BODY_NAME = "panda_hand"
FINGERTIP_BODY_NAME = "panda_fingertip_centered"
LEFTFINGER_BODY_NAME = "panda_leftfinger"
RIGHTFINGER_BODY_NAME = "panda_rightfinger"
PROBE_OUTPUT = os.path.join(os.path.dirname(__file__), "factory_tcp_probe.json")


def _quat_inv(q):
    """Inverse of a unit quaternion (xyzw)."""
    return [-q[0], -q[1], -q[2], q[3]]


def _quat_mul(a, b):
    """Hamilton product (xyzw)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_rotate(q, v):
    """Rotate v (3-list) by q (xyzw)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # t = 2 * (q.xyz x v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v + qw * t + q.xyz x t
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _xform_inverse(p, q):
    """Inverse of a (pos, quat) transform (xyzw quaternion convention)."""
    qi = _quat_inv(q)
    p_neg = [-p[0], -p[1], -p[2]]
    p_inv = _quat_rotate(qi, p_neg)
    return p_inv, qi


def _xform_compose(pa, qa, pb, qb):
    """Compose two (pos, quat) transforms: result = T_a * T_b."""
    p = [
        pa[0] + _quat_rotate(qa, pb)[0],
        pa[1] + _quat_rotate(qa, pb)[1],
        pa[2] + _quat_rotate(qa, pb)[2],
    ]
    q = _quat_mul(qa, qb)
    return p, q


def _hand_to_fingertip(hand_pose, finger_pose):
    """Compute the relative transform from the hand frame to the fingertip frame.

    Both inputs are (pos, quat) tuples in the same world frame.
    Returns (pos, quat) of fingertip expressed in hand-local coordinates.
    """
    p_hi, q_hi = _xform_inverse(hand_pose[0], hand_pose[1])
    return _xform_compose(p_hi, q_hi, finger_pose[0], finger_pose[1])


# IsaacLab convention for body_quat_w is [w, x, y, z]; convert to xyzw to match
# every other system in the loop.
def _wxyz_to_xyzw(q):
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _build_test_joint_configs(num_arm_dofs: int, init_arm_q: list[float]):
    return {
        "zeros": [0.0] * num_arm_dofs,
        "init_arm_q": list(init_arm_q),
        "all_quarter_pi": [math.pi * 0.25] * num_arm_dofs,
    }


# Initial joint configuration that Factory itself uses (lifted from
# factory_env_cfg's home_q).
INIT_ARM_Q = [0.0, -math.pi / 4, 0.0, -3.0 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]


def main():
    parser = argparse.ArgumentParser(description="Probe Factory's fingertip_midpoint offset.")
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
    add_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    # Resolve the env config (no agent params needed for this probe).
    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent)

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1

        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()

        unwrapped = env.unwrapped
        robot = unwrapped._robot
        sim = unwrapped.sim

        body_names = list(robot.body_names)
        print(f"[probe] Robot body_names ({len(body_names)}):")
        for i, name in enumerate(body_names):
            print(f"  [{i:2d}] {name}")

        for required in (HAND_BODY_NAME, FINGERTIP_BODY_NAME, LEFTFINGER_BODY_NAME, RIGHTFINGER_BODY_NAME):
            if required not in body_names:
                print(f"[probe] ERROR: '{required}' not in body_names; update probe constants.")
                sys.exit(1)

        hand_idx = body_names.index(HAND_BODY_NAME)
        finger_idx = body_names.index(FINGERTIP_BODY_NAME)
        left_idx = body_names.index(LEFTFINGER_BODY_NAME)
        right_idx = body_names.index(RIGHTFINGER_BODY_NAME)
        print(
            f"[probe] hand={hand_idx}, fingertip={finger_idx}, "
            f"left={left_idx}, right={right_idx}"
        )

        joint_names = list(robot.joint_names)
        print(f"[probe] joint_names ({len(joint_names)}): {joint_names}")

        num_arm_dofs = 7
        configs = _build_test_joint_configs(num_arm_dofs, INIT_ARM_Q)

        results = {}
        for case_name, q_arm in configs.items():
            # Write joint_state_to_sim. We also zero the gripper so the hand
            # body is well-defined; the actual gripper mode doesn't affect the
            # hand frame's pose. ProxyArrays expose the underlying torch
            # tensor via the .torch attribute.
            joint_pos = robot.data.default_joint_pos.torch.clone()
            joint_vel = robot.data.default_joint_vel.torch.clone()
            joint_pos[:, :num_arm_dofs] = torch.tensor(
                q_arm, device=joint_pos.device, dtype=joint_pos.dtype
            )
            # Re-write the joint state on every settling step so gravity drift
            # cannot accumulate between writes. After 5 writes/steps the body
            # buffers should reflect the commanded q with sub-millimeter drift.
            for _ in range(5):
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(sim.get_physics_dt())

            # Pull world-frame pose for the four bodies of interest, plus the
            # actual joint positions PhysX is reporting (to detect drift).
            body_state_w = robot.data.body_state_w.torch  # (N_envs, N_bodies, 13)

            def _pose_at(idx):
                pos = body_state_w[0, idx, 0:3].cpu().tolist()
                q_wxyz = body_state_w[0, idx, 3:7].cpu().tolist()
                return pos, _wxyz_to_xyzw(q_wxyz)

            hand_pos, hand_quat = _pose_at(hand_idx)
            finger_pos, finger_quat = _pose_at(finger_idx)
            left_pos, left_quat = _pose_at(left_idx)
            right_pos, right_quat = _pose_at(right_idx)
            measured_q = robot.data.joint_pos.torch[0].cpu().tolist()

            rel_pos, rel_quat = _hand_to_fingertip((hand_pos, hand_quat), (finger_pos, finger_quat))
            left_rel_pos, _ = _hand_to_fingertip((hand_pos, hand_quat), (left_pos, left_quat))
            right_rel_pos, _ = _hand_to_fingertip((hand_pos, hand_quat), (right_pos, right_quat))

            results[case_name] = {
                "q_arm_commanded": q_arm,
                "q_full_measured": measured_q,
                "hand_world_pos": hand_pos,
                "hand_world_quat_xyzw": hand_quat,
                "fingertip_world_pos": finger_pos,
                "fingertip_world_quat_xyzw": finger_quat,
                "leftfinger_world_pos": left_pos,
                "rightfinger_world_pos": right_pos,
                "fingertip_in_hand_local_pos": rel_pos,
                "fingertip_in_hand_local_quat_xyzw": rel_quat,
                "leftfinger_in_hand_local_pos": left_rel_pos,
                "rightfinger_in_hand_local_pos": right_rel_pos,
            }
            print(f"\n[probe] Config '{case_name}':")
            print(f"  q_arm commanded           = {[f'{v:+.4f}' for v in q_arm]}")
            print(f"  q_full  measured          = {[f'{v:+.4f}' for v in measured_q]}")
            print(f"  hand world pos            = {[f'{v:+.4f}' for v in hand_pos]}")
            print(f"  fingertip world pos       = {[f'{v:+.4f}' for v in finger_pos]}")
            print(f"  fingertip in hand local   = {[f'{v:+.6f}' for v in rel_pos]}")
            print(f"  leftfinger in hand local  = {[f'{v:+.6f}' for v in left_rel_pos]}")
            print(f"  rightfinger in hand local = {[f'{v:+.6f}' for v in right_rel_pos]}")

        # Sanity: the offset should be constant across configs (rigid-body
        # attachment). Compute max deviation across the three samples.
        offsets = [(case, r["fingertip_in_hand_local_pos"]) for case, r in results.items()]
        baseline = offsets[0][1]
        max_dev_mm = 0.0
        for case, p in offsets[1:]:
            d = ((p[0] - baseline[0]) ** 2 + (p[1] - baseline[1]) ** 2 + (p[2] - baseline[2]) ** 2) ** 0.5
            max_dev_mm = max(max_dev_mm, d * 1000.0)
            print(f"[probe] dev '{case}' vs '{offsets[0][0]}': {d * 1000.0:.4f} mm")

        print(
            f"\n[probe] Max relative-offset deviation across configs: {max_dev_mm:.4f} mm"
        )
        if max_dev_mm > 1.0:
            print(
                "[probe] WARNING: offset is not constant across configs - "
                "fingertip is not rigidly attached to the hand frame, or the "
                "two bodies live in different articulation roots."
            )
        else:
            print("[probe] OK: offset is constant - fingertip is rigidly attached to the hand frame.")

        with open(PROBE_OUTPUT, "w") as f:
            json.dump({"task": args_cli.task, "results": results}, f, indent=2)
        print(f"\n[probe] Wrote {PROBE_OUTPUT}")

        env.close()


if __name__ == "__main__":
    main()
