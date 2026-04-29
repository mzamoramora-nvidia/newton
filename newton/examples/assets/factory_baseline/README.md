# IsaacLab Factory baseline data and probe scripts

This directory holds the reference data we use to compare Newton's
Panda OSC against IsaacLab's Factory implementation, along with the
two scripts that produced it. Both scripts are meant to be run from
**inside an IsaacLab installation** - they do not run from the Newton
repo; they are mirrored here so the lineage of the JSON files is
visible and so anyone reproducing the comparison can see exactly what
was measured.

## Files

### `factory_joint_pos.json`

Output of `probe_joint_pos.py`. Resets Factory's peg-insert env with **all
randomization noise disabled** (so the post-reset pose is deterministic),
runs a few settle steps to let Factory's IK converge, and dumps the
resulting joint angles plus fingertip world pose. The post-IK joint
values are baked into Newton's example as
`INIT_ARM_Q_FACTORY_TASK_HOME` (in `example_robot_panda_osc.py` /
`example_robot_panda_compare.py`) so Newton's home matches the pose at
which `osc_isaaclab_steps.json` was recorded - any apples-to-apples
step-response comparison should set Newton's `INIT_ARM_Q` to this
constant.

### `factory_tcp_probe.json`

Output of `probe_fingertip.py`. Sets the Factory env to a few test
joint configurations, reads the world-frame poses of `panda_hand`
and `panda_fingertip_centered`, and computes the relative offset
(Factory's TCP offset) for each pose.

The init-pose row is the trustworthy one (the arm sits at its natural
gravity equilibrium with our patched probe; other configs drift under
gravity over the settling steps). It reports the offset as
`(0, 0, -0.1121)` m in hand-local coordinates, which our compare
viewer (`example_robot_panda_compare.py`) cross-checks visually.

### `osc_isaaclab_steps.json`

Output of `osc_step_response.py`. Runs Factory's OSC against twelve
step-pose targets (5 cm in +/- x/y/z, +/- 30 deg about each axis) at
Factory's published gains and records per-control-tick TCP pose,
target pose, and joint torques. This is the reference step-response
behavior the Newton OSC will be tuned against.

Header fields of interest:

```
control_hz       = 15.0          # Factory env step rate
task_prop_gains  = [100, 100, 100, 30, 30, 30]
task_deriv_gains = [20, 20, 20, ~10.95, ~10.95, ~10.95]
home_pos         = [+0.59, +0.05, +0.14] m   (Factory's task home)
trial_ticks      = 60            # 4 s of data per trial
```

Each trial entry contains the step target, summary metrics
(`rise_time_s`, `settling_time_s`, `overshoot_mm`, `final_pos_err_mm`,
`final_rot_err_deg`), and the full per-tick trajectory.

### `probe_joint_pos.py`

Companion to `factory_joint_pos.json`. Drops into IsaacLab's `scripts/`
directory and runs as:

```bash
cd <IsaacLab>
./isaaclab.sh -p scripts/probe_joint_pos.py
```

Disables Factory's IK randomization (`hand_init_pos_noise`,
`hand_init_orn_noise`, `fixed_asset_init_pos_noise`,
`fixed_asset_init_orn_range_deg`) so the post-reset pose is
deterministic, runs eight zero-action settle steps, and writes the
resulting joint config and fingertip world pose to
`<IsaacLab>/scripts/factory_joint_pos.json`.

### `probe_fingertip.py`

Companion to `factory_tcp_probe.json`. Drops into IsaacLab's
`scripts/` directory and runs as:

```bash
cd <IsaacLab>
./isaaclab.sh -p scripts/probe_fingertip.py
```

The output JSON lands at `<IsaacLab>/scripts/factory_tcp_probe.json`;
the version checked in here reflects the most recent run.

### `osc_step_response.py`

Companion to `osc_isaaclab_steps.json`. Same setup as the probe:

```bash
cd <IsaacLab>
./isaaclab.sh -p scripts/osc_step_response.py
```

The script bypasses Factory's action machinery entirely: after the
env's normal reset (which IK-drives the arm to a Factory home pose)
it calls `unwrapped.generate_ctrl_signals(target_pos, target_quat)`
each substep with a fixed step target, then advances physics via
Factory's `step_sim_no_action()` helper. This makes the recorded
trajectory a pure step-response of the Khatib OSC at Factory's
published gains, with no policy or action smoothing in the loop.

Output JSON lands at `<IsaacLab>/scripts/osc_isaaclab_steps.json`.

## Reproducing the data

1. Copy the two `.py` files from this directory into your IsaacLab
   `scripts/` directory.
2. Run them with `./isaaclab.sh -p scripts/<name>.py`.
3. Replace the JSON files here with the regenerated outputs if you
   want to update the baseline.

## License note

The Factory env that produced these files is part of IsaacLab
(BSD-3-Clause) and Isaac Sim (NVIDIA Omniverse license for the asset
bundle). Only the JSON output and our probe scripts are checked in
here, both under this repo's open-source license. The Factory USD
asset itself remains uncommitted - see
`newton/examples/assets/franka_mimic/README.md` for that workflow.
