# Companion to factory_baseline/probe_fingertip.py. Resets the Factory env
# (which IK-drives the arm to the task home) and dumps the resulting joint
# angles + fingertip pose, so we can compare Newton's IK solution to
# IsaacLab's IK-solved joint configuration.
#
# Run from inside IsaacLab:
#   ./isaaclab.sh -p scripts/probe_joint_pos.py
import argparse
import json
import os

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg

cfg = FactoryTaskPegInsertCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device

# Disable IK randomization so the post-reset pose is deterministic.
cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
cfg.task.fixed_asset_init_orn_range_deg = 0.0

env = FactoryEnv(cfg)
env.reset()

# Settle a few steps so IK converges.
for _ in range(8):
    env.step(torch.zeros(env.action_space.shape, device=env.device))

robot = env.scene["robot"]
joint_pos = robot.data.joint_pos.torch.cpu().numpy()[0]
joint_names = robot.joint_names

# Per-DOF actuator/dynamics config that the Newton USD path mirrors verbatim.
# Pulled directly from the live ArticulationView so we capture exactly what
# Factory's reset+settle is using (rather than re-reading the USD/cfg by hand
# and risking a drift). Order matches `joint_names` above.
joint_armature = robot.data.joint_armature.torch.cpu().numpy()[0]
joint_friction = robot.data.joint_friction.torch.cpu().numpy()[0]
joint_damping = robot.data.joint_damping.torch.cpu().numpy()[0]
joint_stiffness = robot.data.joint_stiffness.torch.cpu().numpy()[0]

# Get fingertip pose for verification.
fingertip_idx = robot.body_names.index("panda_fingertip_centered")
ft_state = robot.data.body_state_w.torch.cpu().numpy()[0, fingertip_idx]
ft_pos = ft_state[:3].tolist()
ft_quat_wxyz = ft_state[3:7].tolist()
ft_quat_xyzw = [ft_quat_wxyz[1], ft_quat_wxyz[2], ft_quat_wxyz[3], ft_quat_wxyz[0]]

# Robot base pose (so fingertip can be expressed robot-base-relative).
base_idx = robot.body_names.index("panda_link0")
base_state = robot.data.body_state_w.torch.cpu().numpy()[0, base_idx]

result = {
    "task": "FactoryTaskPegInsertCfg",
    "joint_names": list(joint_names),
    "joint_pos_after_ik": [float(v) for v in joint_pos],
    "joint_armature": [float(v) for v in joint_armature],
    "joint_friction": [float(v) for v in joint_friction],
    "joint_damping": [float(v) for v in joint_damping],
    "joint_stiffness": [float(v) for v in joint_stiffness],
    "fingertip_world_pos": [float(v) for v in ft_pos],
    "fingertip_world_quat_xyzw": [float(v) for v in ft_quat_xyzw],
    "robot_base_world_pos": [float(v) for v in base_state[:3]],
    "robot_base_world_quat_xyzw": [
        float(base_state[4]),
        float(base_state[5]),
        float(base_state[6]),
        float(base_state[3]),
    ],
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factory_joint_pos.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nWrote {out}")
print("\nJoint angles after IK reset:")
for n, v in zip(joint_names, joint_pos, strict=True):
    print(f"  {n:<22s}  {v:+.6f}")
print("\nPer-DOF actuator/dynamics config (armature / friction / damping / stiffness):")
for n, a, f, d, k in zip(joint_names, joint_armature, joint_friction, joint_damping, joint_stiffness, strict=True):
    print(f"  {n:<22s}  arm={a:+.5f}  fri={f:+.5f}  damp={d:+.3f}  stif={k:+.3f}")
print(f"\nFingertip world pos:  {ft_pos}")
print(f"Fingertip world quat (xyzw): {ft_quat_xyzw}")
print(f"Robot base world pos: {base_state[:3].tolist()}")

simulation_app.close()
