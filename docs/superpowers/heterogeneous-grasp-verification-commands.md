# Heterogeneous-Grasp Refactor: Verification Commands

Companion to `plans/2026-05-05-heterogeneous-grasp-probe-refactor.md`. One-line
copy-paste commands for verifying the Tasks 1–10 refactor in both headless and
GL-viewer modes.

## Headless example (bare run, NaN-guard fallback path)

Default config (hydroelastic, 24 worlds, 700 frames):

```bash
uv run -m newton.examples.robot.example_robot_heterogeneous_grasp --test --viewer null --quiet
```

All four collision modes in sequence:

```bash
for mode in mujoco newton_default newton_sdf newton_hydroelastic; do uv run -m newton.examples.robot.example_robot_heterogeneous_grasp --test --viewer null --quiet --collision-mode $mode; done
```

Fast smoke (4 worlds, 50 frames):

```bash
uv run -m newton.examples.robot.example_robot_heterogeneous_grasp --test --viewer null --world-count 4 --num-frames 50 --quiet
```

## Interactive example (ViewerGL, minimal info panel — no probe attached)

Default hydroelastic:

```bash
uv run -m newton.examples.robot.example_robot_heterogeneous_grasp
```

Other modes:

```bash
uv run -m newton.examples.robot.example_robot_heterogeneous_grasp --collision-mode mujoco
```

```bash
uv run -m newton.examples.robot.example_robot_heterogeneous_grasp --collision-mode newton_default
```

```bash
uv run -m newton.examples.robot.example_robot_heterogeneous_grasp --collision-mode newton_sdf
```

## Headless regression tests (probe attached, asserts on success_rate)

All regression tests (4 base + 4 primitives subclass, ~8 min):

```bash
uv run --extra dev -m newton.tests -k TestHeterogeneousGraspRegression
```

Full-catalog base class only (4 tests, ~3 min):

```bash
uv run --extra dev -m newton.tests -k "TestHeterogeneousGraspRegression and not Primitives"
```

Primitives-only subclass (5 shapes × 2 worlds, fast):

```bash
uv run --extra dev -m newton.tests -k TestHeterogeneousGraspRegressionPrimitives
```

Single collision mode (any of mujoco / newton_default / newton_sdf / newton_hydroelastic):

```bash
uv run --extra dev -m newton.tests -k test_newton_hydroelastic_baseline
```

## GL regression test (rich tuning panel + debug frames + summary tables)

Two ways to flip `do_rendering=True` and route through ViewerGL.

**Direct-file form with `--render` flag** — cleanest for a single test:

```bash
uv run python newton/tests/test_object_centric_grasp.py --render TestHeterogeneousGraspRegression.test_newton_hydroelastic_baseline
```

Swap the trailing argument for any of `test_mujoco_baseline`, `test_newton_default_baseline`, or `test_newton_sdf_baseline`. Drop the trailing arg entirely to run all four base tests.

**Env-var form with `-m newton.tests` discovery** — needed when going through the parallel runner (which rejects unknown CLI flags):

```bash
GRASP_TEST_RENDER=1 uv run --extra dev -m newton.tests -k test_newton_hydroelastic_baseline
```

The env var also works with the primitives subclass and with broader filters like `-k TestHeterogeneousGraspRegression`.

What this exercises:

- Routes the example through ViewerGL.
- Fires `probe.on_render` (debug-frame triads / spawn-region square).
- Fires `probe.on_gui_render` (full tuning panel: per-world metrics, grasp-spec sliders, frame toggles).
- Calls `probe.print_summary(result)` at the end so the per-world / per-shape / aggregate / per-state tables print to stdout while the GL window stays open.

## Unit tests (kernel + spec sanity, no full example build)

12 fast tests, ~80s:

```bash
uv run --extra dev -m newton.tests -k "TestGraspSpecs or TestComputeGraspTargetsKernel or TestSpawnRandomization or TestGraspTargetsMatchReference or TestMarginPctToCtrl"
```

## ASV benchmark

Quick local run on the latest commit:

```bash
uvx --with virtualenv asv run --launch-method spawn --quick HEAD^!
```

Single-cell sanity via the bench's `__main__` block:

```bash
uv run python asv/benchmarks/simulation/bench_heterogeneous_grasp.py
```

## Lint

```bash
uvx pre-commit run --files newton/examples/robot/example_robot_heterogeneous_grasp.py newton/tests/test_object_centric_grasp.py asv/benchmarks/simulation/bench_heterogeneous_grasp.py
```
