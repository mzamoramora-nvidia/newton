.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Tuning Solver Porting:

Porting Tasks Between Solvers
=============================

Moving a working task to another physics backend (for example PhysX to
:class:`~newton.solvers.SolverMuJoCo`) has its own failure modes. The workflow in
:ref:`Simulation Tuning` still applies; this page adds what it misses when the
controller optimizes against the simulation, most commonly a policy trained in a
downstream framework such as Isaac Lab.

Feature-Parity Audit Before Any Tuning
--------------------------------------

Two solvers given identical configuration are not the same physical model. Before
tuning anything, check every limit and drive feature the source solver enforces
against the target solver (the "Joint feature support" table and the solver source
are the ground truth):

- **Joint velocity limits.** Source engines such as PhysX hard-clamp joint
  velocities every step; as of this writing no listed Newton solver enforces
  ``joint_velocity_limit`` (bookkeeping only; assume the gap and confirm in the
  feature table). An unclamped position PD drive
  (``joint_target_ke``/``joint_target_kd``, :math:`k_p`/:math:`k_d` below) moves
  freely at

  .. math::

     v_{\mathrm{free}} \approx \frac{k_p \, e}{k_d}

  for commanded error :math:`e`: a drive tuned under a clamp overshoots, whips the
  arm, and snaps grippers shut once the clamp disappears.
- **Effort limits, armature, joint friction.** Check each is supported *and*
  actually applied on the target path.
- **Drive integration.** Implicit PD differs numerically between solvers; the same
  gains give different effective damping. Verify by overlay (below), not by
  reading the config.

**Trajectory-overlay test:** drive both backends through an identical pre-recorded
action sequence and overlay joint positions, joint velocities, and end-effector
paths. Minutes of compute; run it before any long closed-loop run (for example
policy training). Peak joint velocities above the source solver's limits are a
model gap to close, not noise.

Closing a Missing-Velocity-Limit Gap
------------------------------------

In order of fidelity:

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Remedy
     - Action
     - Limitation
   * - Clamp in control code
     - Rate-limit the drive targets, or clamp joint velocities after each
       substep, to the source solver's limits.
     - The only remedy an optimizing controller cannot defeat; prefer it
       whenever you can touch the control or integration layer.
   * - Raise drive damping
     - Raise :math:`k_d` until :math:`k_p e_{\max} / k_d` is at or below the
       source limit; verify with the trajectory overlay.
     - The cap scales with commanded error: a controller that commands larger
       errors moves fast anyway.
   * - Add armature
     - Extra rotor inertia slows the whole joint, including high-frequency
       components damping misses.
     - Changes the physical model; keep it small and justified.

Optimizing Controllers Amplify Solver Gaps
------------------------------------------

An optimizing controller (a trained policy, a trajectory optimizer, a well-tuned
MPC) is an adversarial probe of the simulation: anything physically wrong that
correlates with its objective will be found and amplified, long after
random-motion tests pass.

- **Random-motion tests understate the problem.** A scene can survive thousands
  of random-action episodes yet fail under a controller that steers into the
  corner case (e.g. flicking an object upward instead of grasping it, if
  unclamped velocities make throwing possible). Judge fixes against **worst-case
  inputs** (bang-bang sequences and action sequences recorded from failed runs),
  not only random ones.
- **Replay recorded exploits.** Re-running a failure's action sequence under a
  candidate fix is a minutes-cheap test of whether the fix removes the exploit or
  merely inconveniences it. **Prefer removing the exploit physically** (clamp in
  control code, close the model gap): configurations where it stays reachable
  behave inconsistently from run to run.

.. note::

   Training-side diagnosis (reward curves, termination-cause rates, policy
   entropy) is the domain of the downstream RL framework; consult its
   documentation. Newton-side evidence is the replayed action sequence and the
   peak joint velocities, penetrations, and contact forces it produces.

Interacting Knobs: Test Pairs, Not Only Singles
-----------------------------------------------

Changing one knob at a time is right for *attribution* but wrong for *acceptance*
of compound failures. A grasp-and-move failure typically couples an overshooting
drive with contact too soft for the resulting impacts: fixing either alone shows
little improvement, and a strict one-knob protocol wrongly rejects both.

1. **Build a scripted repro first** (e.g. approach-grasp-lift-shake). Iterate
   physics there in seconds per test, not in hour-long closed-loop runs such as
   policy training.
2. **Sweep single knobs** in the repro for attribution.
3. **Accept or reject combinations** in the repro; only repro-passing
   combinations go to full task runs.
4. **Keep the change set minimal.** Every extra "safety" change has its own
   failure mode: heavy gripper damping slows the close enough to miss grasps,
   joint friction stalls small exploratory motions, over-stiff contact needs
   smaller timesteps.
