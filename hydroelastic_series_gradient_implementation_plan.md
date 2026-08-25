# Hydroelastic pressure-law and series-gradient implementation plan

Status: implemented locally; focused validation complete
Issue: [newton-physics/newton#3503](https://github.com/newton-physics/newton/issues/3503)
Rebased on Newton commit: `e5cc054bb95a6ba8889da983b0fdab3d977d32c9`
Feature branch: `mzamoramora/hydro-series-gradient`
Worktree: `/home/mzamoramora/build_playground/newton/hydro-private/newton-hydro-series-gradient`

## Executive decision

Do not add a closed `PAIR_SEPARATION` / `SERIES_GRADIENT` enum. Instead, add a
richer optional pressure-law callback alongside the existing scalar-pressure
callback:

```python
@dataclass
class HydroelasticSDF.Config:
    pressure_func: Any = None
    pressure_law_func: Any = None
    pressure_data: Any = None
```

The two callbacks have deliberately different contracts:

```python
pressure_func(signed_depth, shape_idx, pressure_data) -> pressure

pressure_law_func(signed_depth, shape_idx, pressure_data) \
    -> wp.vec2f(pressure, compression_slope)
```

Here `compression_slope = -dp/dd` is positive for a monotonically increasing
pressure under compression and has units Pa/m.

The scalar-pressure path preserves current behavior and remains the default.
Selecting `pressure_law_func` opts penetrating contacts into the
Masterjohn/Drake velocity-level tangent computed from both projected pressure
gradients. Provide a public built-in rich linear law so users can request this
behavior without writing a callback:

```python
from newton.geometry import (
    HydroelasticSDF,
    hydroelastic_pressure_law_linear,
)

hydro_config = HydroelasticSDF.Config(
    pressure_law_func=hydroelastic_pressure_law_linear,
)
```

This is an open constitutive-law interface rather than a list of solver modes.
Users can define linear, power, cubic, or other differentiable pressure laws;
Newton remains responsible for mapping their pressure and slope to contacts.

The implementation must keep two different quantities separate:

- `pair_separation`: margin-relative geometry used to classify penetrating,
  speculative, and rejected faces;
- `solver_distance`: the signed distance encoded in the solver contact points.

They are identical on the scalar-pressure path. They differ for penetrating
faces on the rich-law/projected-gradient path.

This plan supersedes earlier plans that describe Newton's current penetrating
distance as `2*d_B`. Margin/gap work in PR #3719 has already replaced that
single-body construction with a two-body pair separation.

## Current behavior after margin/gap and determinism changes

At an extracted contact face Newton now forms the margin-adjusted SDF values:

```text
d_A = sdf_A - margin_A
d_B = sdf_B - margin_B
```

and the pair separation:

```text
d_pair = d_A + d_B
```

The contact bands are:

```text
d_pair < 0                     penetrating
0 <= d_pair <= gap_A + gap_B  speculative
d_pair > gap_A + gap_B        rejected
```

For a penetrating unreduced face, current Newton exports:

```text
k_pair = A p_0 / (-d_pair)
phi_0  = d_pair
```

which preserves the instantaneous face force:

```text
k_pair (-phi_0) = A p_0
```

The relevant implementation is in:

- [`newton/_src/geometry/sdf_hydroelastic.py`](newton/_src/geometry/sdf_hydroelastic.py),
  especially face generation and unreduced decode;
- [`newton/_src/geometry/contact_reduction_hydroelastic.py`](newton/_src/geometry/contact_reduction_hydroelastic.py),
  which preserves aggregate pressure force during reduction; and
- [`newton/_src/geometry/contact_reduction_global.py`](newton/_src/geometry/contact_reduction_global.py),
  which owns contact-buffer and per-bin aggregate storage.

### What current Newton already fixes

For the built-in linear pressure law, the equal-pressure surface satisfies:

```text
p_0 = -k_A d_A = -k_B d_B
```

Therefore:

```text
-d_pair = p_0/k_A + p_0/k_B

             p_0
k_pair/A = ---------
            -d_pair

             k_A k_B
         = -----------
            k_A + k_B
```

Current Newton consequently already produces the two-material series slope for
ideal aligned SDFs, including unequal `k_A` and `k_B`. An unequal-material
head-on test is now a characterization test, not a failing #3503 regression.

### What remains missing

Define the projections onto the unit contact normal from A to B:

```text
alpha_A =  grad(phi_A) dot n_hat
alpha_B = -grad(phi_B) dot n_hat
```

The requested directional pressure gradients are:

```text
g_A = k_A alpha_A
g_B = k_B alpha_B
```

and their series combination is:

```text
                  g_A g_B
g_projected = -------------
                  g_A + g_B
```

The requested solver pair is:

```text
k_tangent = A g_projected
phi_0     = -p_0 / g_projected
```

Current Newton instead behaves as though both projections were one. The two
formulations diverge for oblique or conforming contact and wherever the sampled
SDF gradient magnitude differs from one.

Margins do not add a derivative term because subtracting a constant margin does
not change the SDF gradient.

## Public interface

Keep the existing callback unchanged:

```python
@wp.func
def my_pressure_func(
    signed_depth: float,
    shape_idx: int,
    pressure_data: MyPressureData,
) -> float:
    ...
```

Add a mutually exclusive richer callback:

```python
@wp.func
def my_pressure_law(
    signed_depth: float,
    shape_idx: int,
    pressure_data: MyPressureData,
) -> wp.vec2f:
    pressure = ...
    compression_slope = ...  # -dp/dd, in Pa/m
    return wp.vec2f(pressure, compression_slope)
```

For example, a globally defined monotone cubic law can expose its analytical
slope without any Newton-side finite differencing:

```python
@wp.func
def cubic_pressure_law(
    signed_depth: wp.float32,
    shape_idx: wp.int32,
    data: CubicPressureData,
) -> wp.vec2f:
    d = signed_depth
    kh = data.shape_kh[shape_idx]
    c = data.shape_cubic[shape_idx]
    pressure = -kh * d - c * d * d * d
    compression_slope = kh + 3.0 * c * d * d
    return wp.vec2f(pressure, compression_slope)
```

For nonnegative `kh` and `c`, this remains finite and monotone over positive and
negative signed depths, which is important because extraction evaluates the law
on both sides of the nominal surface.

The configuration field is:

```python
pressure_law_func: Any = None
```

Return `wp.vec2f` rather than a tuple or struct so the function is straightforward
to call from generated Warp kernels. Name and document the two components at the
Python API boundary; do not make callers infer their order from examples alone.

Export a supported public built-in:

```python
@wp.func
def hydroelastic_pressure_law_linear(
    signed_depth: float,
    shape_idx: int,
    pressure_data: LinearPressureData,
) -> wp.vec2f:
    ...
```

Expose it canonically from `newton.geometry`. When this exact built-in function
is selected and `pressure_data` is omitted, construct the existing internal
`LinearPressureData` from `shape_material_kh`, just as the current built-in
scalar law does. A custom rich law requires explicit `pressure_data`.

### Why the callback returns pressure and slope, not `(k, phi)`

The pressure law owns constitutive behavior only. Newton owns geometry and
solver mapping:

```text
alpha_A =  grad(phi_A) dot n_hat
alpha_B = -grad(phi_B) dot n_hat

g_A = slope_A alpha_A
g_B = slope_B alpha_B

                  g_A g_B
g_projected = -------------
                  g_A + g_B

k   = A g_projected
phi = -p_0/g_projected
```

Allowing a callback to return `(k, phi)` would expose contact area, normal
orientation, margin/gap policy, sign conventions, reduction budgets, and
determinism details. It would also allow callbacks to violate force consistency.
Keeping those concerns inside Newton makes the module deeper and lets the
reducer preserve a single well-defined tangent budget.

The richer callback also avoids finite-differencing arbitrary `pressure_func`
implementations in contact kernels. Nonlinear custom laws provide their
analytical local slope directly.

Resolve the callback choice during `HydroelasticSDF` construction and specialize
generated kernels statically. The scalar-pressure specialization must not sample
gradients, allocate projected-tangent hot-path storage, or execute runtime
callback-choice branches.

### Initial configuration rules

```text
neither callback                         current built-in scalar path
pressure_func only                       current custom scalar path
pressure_law_func=public linear law      built-in projected-gradient path
pressure_law_func=custom rich law        custom projected-gradient path
pressure_func + pressure_law_func        ValueError
custom pressure_law_func without data    ValueError
```

The existing `pressure_func` remains source-compatible and keeps pair separation
as its solver distance. Do not infer its derivative with finite differences.

The rich law must return finite pressure and a finite, nonnegative compression
slope for every sampled signed depth. The pressure must remain monotonically
non-increasing in signed depth over the law's supported domain. Debug validation
can check sampled values, but the API contract must state these requirements
because a kernel cannot prove global monotonicity.

At marching-cubes corners and other pressure-only call sites, use a private
adapter that returns `pressure_law_func(...)[0]`. At accepted penetrating face
centroids, evaluate the rich law for both shapes to obtain their separate slopes.

At a numerically imperfect equal-pressure face, define the face pressure
symmetrically:

```text
p_0 = 0.5 (p_A + p_B)
```

For the built-in linear law, `p_A` and `p_B` should agree up to extraction
tolerance. The symmetric definition avoids choosing a privileged shape for
custom or numerically imperfect laws. The scalar-pressure path continues using
its current pressure evaluation exactly.

## Behavioral invariants

For every emitted penetrating source face on either path:

```text
k_i > 0
k_i and phi_i are finite
k_i (-phi_i) = A_i p_i
```

Additional rich-law/projected-gradient invariants are:

```text
k_i = A_i g_i
phi_i = -p_i/g_i
```

Swapping shapes A and B must not change physical stiffness, solver distance, or
force after accounting for normal orientation.

A projected-gradient face is invalid when any required value is non-finite or:

```text
p_0 <= epsilon
g_A <= epsilon
g_B <= epsilon
g_A + g_B <= epsilon
g_projected <= epsilon
```

Skip an invalid penetrating face before it contributes to either force or
tangent aggregates. Use a private named threshold initially. Do not silently
fall back to pair-separation linearization or clamp the projected gradient.
An aggregate invalid-face diagnostic is deferred because the current collision
pipeline does not expose per-law diagnostic storage; adding it should be a
separate observability change rather than part of the constitutive API.

Use raw sampled gradients. Do not normalize them. Gradient magnitude is part of
the derivative of Newton's sampled pressure field, and Drake's stiffness path
does not normalize this magnitude.

## Data model

Keep the existing `position_depth.w` meaning:

```text
position_depth.w = pair_separation
```

It remains the source of truth for:

- contact-band classification;
- speculative distance;
- pre-pruning and reduction selection;
- contact-surface visualization;
- geometric depth-volume reliability checks; and
- compatibility behavior on the scalar-pressure path.

Add a rich-law-only per-face value:

```text
contact_tangent_stiffness = A_i g_i  [N/m]
```

Add a per-normal-bin source tangent budget:

```text
agg_tangent_stiffness = sum_i A_i g_i  [N/m]
```

Add a per-normal-bin integer count of reduced penetrating winners:

```text
total_contact_count_reduced
```

The stored count includes normal-bin winners and voxel-bin copies that map to
the normal bin. During export, the optional anchor is added exactly once to form
the number of penetrating outputs that share the bin budget. Speculative
representatives are excluded.

Prefer empty arrays plus static kernel specialization for projected-tangent
storage on the scalar-pressure path.

## Implementation stages

### Stage 0: rebaseline tests against current main

Before changing implementation code:

1. Add a characterization test showing that unequal-`kh` planar contact already
   gives the material series slope.
2. Assert current stiffness and signed distance independently, not only their
   product.
3. Add an algebra-level projected-series test with known `A`, `p_0`, `g_A`, and
   `g_B`.
4. Add an oblique or conforming pipeline fixture in which at least one projected
   gradient differs materially from one.
5. Add callback-contract tests for a custom rich linear law and a nonlinear law
   with a known analytical compression slope.
6. Confirm that the discriminating projected-gradient test fails on unmodified
   main.
7. Preserve existing margin/gap, custom-scalar-pressure, deterministic,
   reduced-force, and moment-matching tests.

The unequal-material head-on case remains useful, but it must not be presented
as the primary regression because current main already handles it.

### Stage 1: add the richer pressure-law seam

In `newton/_src/geometry/sdf_hydroelastic.py`:

1. Add and document `pressure_law_func` on `HydroelasticSDF.Config`.
2. Add the public `hydroelastic_pressure_law_linear` Warp function and export it
   from `newton.geometry`.
3. Validate mutual exclusion with `pressure_func` and the `pressure_data` rules
   during construction. Warp validates the callback signature when it
   specializes the pressure-only adapter; preflight signature introspection is
   deferred until Warp exposes a stable public callable-inspection API.
4. Recognize the public built-in rich linear law and synthesize its internal
   `LinearPressureData` from `shape_material_kh` when data is omitted.
5. Resolve a private pressure-only adapter for marching-cubes and pruning call
   sites and a rich-law adapter for accepted penetrating faces.
6. Statically specialize generation, decode, and reduction kernel factories on
   whether a rich law is present.
7. During development, reject `pressure_law_func` with
   `reduce_contacts=True` until the tangent-aware reducer is complete.

Neither-callback default behavior and every existing `pressure_func` use must
produce identical output to current main.

### Stage 2: implement unreduced projected-series contacts

Only after a face is classified as penetrating on the rich-law path:

1. Recover or retain both margin-adjusted centroid depths. If only `d_B` and
   `d_pair` are currently retained, compute `d_A = d_pair - d_B`.
2. Evaluate `pressure_law_func(d_A, shape_A, data)` and
   `pressure_law_func(d_B, shape_B, data)`.
3. Form `p_0 = 0.5*(p_A + p_B)` and retain each returned compression slope.
4. Sample both SDF gradients at the accepted face centroid.
5. Transform them into the B-local frame used by the MC face normal.
6. Compute `g_A`, `g_B`, and `g_projected` from the returned slopes and raw
   projected SDF gradients.
7. Reject the face if the callback or projected-gradient validity contract
   fails.
8. Store `A*g_projected` with the face.
9. In unreduced decode, export:

   ```text
   k_i   = contact_tangent_stiffness_i
   phi_i = -A_i p_i / k_i
   ```

10. Continue to store and visualize the original `pair_separation`.

A small `|p_A - p_B|` mismatch is expected because a marching-cubes triangle
centroid lies on the extracted piecewise-linear surface, not necessarily on the
exact nonlinear iso-pressure surface. Do not reject solely on that mismatch in
the first implementation; the symmetric average keeps shape ordering from
choosing the force. A debug mismatch statistic and mesh-refinement convergence
test remain useful follow-up instrumentation, but are not required to define
the public callback contract.

Use
[`texture_sample_sdf_grad()`](newton/_src/geometry/sdf_texture.py)
for the initial correctness implementation. It analytically differentiates the
software trilinear interpolant and is documented as the accuracy-oriented
hydroelastic stress-integration path. Benchmark hardware finite-difference
sampling separately; do not make it the initial reference.

Do not use the rich law's corner pressure as the accepted face pressure. Evaluate
both shapes at the accepted centroid as above; corner evaluation remains an
extraction/pruning concern. The scalar-pressure path retains its current
face-pressure behavior exactly.

### Stage 3: accumulate the source tangent through reduction

For each valid rich-law penetrating source face, before pre-pruning:

```text
F_bin = sum_i A_i p_i n_i
K_bin = sum_i A_i g_i
```

In nondeterministic mode, accumulate `K_bin` alongside the existing force
aggregate using the same normal-bin key.

In deterministic mode:

1. Keep pre-pruning disabled, as current Newton already does.
2. Store `contact_tangent_stiffness` per buffered face.
3. Add a fixed-point accumulator slot for tangent stiffness.
4. Add a dedicated fixed-point scale family for its units.
5. Record scale and accumulate it in the deterministic unreduced-aggregate
   kernel.
6. Finalize it with the other unreduced aggregates.

Do not use a floating-point atomic for projected tangent in deterministic mode;
that would violate the current bit-exact accumulation guarantee.

Thread the new arrays through:

- `GlobalContactReducerData`;
- allocation and empty-array paths;
- data-struct construction;
- active-entry clearing;
- ordinary contact-buffer export;
- both penetrating pre-prune winner records;
- every manual winner write;
- deterministic accumulation and finalization; and
- reduced export.

### Stage 4: preserve force and scalar tangent during reduced export

The current reducer gives every selected penetrating representative in a normal
bin one shared secant stiffness. Preserve its selected contacts, centers,
normals, force weights, normal matching, anchor, and moment matching.

Let:

```text
F_bin = aggregate source-face pressure force
K_bin = aggregate source-face projected tangent
N_bin = number of penetrating outputs sharing the bin budget

q_j = -d_pair,j
V   = sum_j q_j n_j
```

Include the anchor depth and normal in `N_bin` and `V` when an anchor is emitted.

Assign:

```text
k_shared = K_bin / N_bin
```

With normal matching enabled, compute:

```text
                   |F_bin|
distance_scale = ----------------
                  k_shared |V|
```

and export each representative as:

```text
phi_j = distance_scale d_pair,j
```

The anchor uses:

```text
phi_anchor = -distance_scale q_anchor
```

This preserves:

```text
sum_j k_shared = K_bin
```

and, after the existing normal-matching rotation:

```text
sum_j k_shared (-phi_j) n_j = F_bin
```

It also preserves every representative's current normal-force weight because:

```text
k_shared (-phi_j)
```

equals the force produced by today's shared secant stiffness and unscaled pair
separation. Existing center-of-pressure, friction-capacity, and moment-matching
targets therefore remain unchanged.

With normal matching disabled, use the existing scalar total-depth denominator.
This preserves scalar force magnitude and scalar tangent budget but retains the
current limitation on exact vector-force direction.

Voxel-bin copies must consume their owning normal bin's budget and must be
included in `N_bin`; they must not independently reproduce all of `K_bin`.

If a rich-law penetrating fallback has no valid normal-bin aggregate, use its
stored per-face tangent and per-face solver distance. Never invent a
pair-separation secant after a rich law has explicitly supplied a derivative.

After reduced tests pass, remove the temporary rich-law/reduction configuration
error.

### Stage 5: leave speculative margin/gap contacts unchanged

Do not sample gradients for speculative faces in the initial implementation.
Continue to export:

```text
contact_distance = d_pair >= 0

activation_stiffness =
    margin_contact_area * k_A k_B / (k_A + k_B)
```

Speculative faces must not contribute to penetrating force, tangent, winner
count, normal matching, or anchor budgets.

This deliberate limitation avoids two gradient samples across a potentially
large speculative band and does not expand issue #3503 beyond penetrating
contacts. Document that a cached speculative contact which crosses into
penetration without collision regeneration retains its material-only activation
regularization. Revisit that behavior when the deprecated
`margin_contact_area` model is replaced.

### Stage 6: hardening and performance

Verify:

- the default and custom scalar-pressure paths perform no new gradient samples;
- CPU and CUDA agreement where supported;
- CUDA graph capture performs no launch-time allocation;
- deterministic output remains bit-exact across repeated runs on the same GPU;
- buffer overflow and hashtable failure paths remain bounds-safe;
- multiple SDF resolutions show convergence rather than normalization artifacts;
- the analytical gradient path's cost is measured on representative scenes;
- large nonzero gaps do not incur projected-gradient sampling for speculative
  faces;
- contact ordering and sort subkeys remain deterministic; and
- semi-implicit, Featherstone, XPBD, VBD, and MuJoCo-facing contact consumers
  tolerate the rescaled solver point separation.

The reduced implementation preserves a scalar normal-bin tangent, not the full
patch tensor:

```text
K_tensor = sum_i A_i g_i (n_i tensor n_i)
```

Tensor fitting, rotational Jacobians, and exact curved-patch tangent matching are
separate future work.

## Test matrix

### Interface and compatibility

- `Config()` preserves current built-in scalar behavior exactly.
- Existing custom scalar linear, power, cubic, and decoupled `pressure_func`
  laws are unchanged.
- The public `hydroelastic_pressure_law_linear` works without explicitly
  supplying `pressure_data`.
- A custom rich linear law matches the public built-in rich law.
- A custom nonlinear rich law uses its supplied analytical slope.
- Supplying both callbacks raises clearly at construction.
- A custom `pressure_law_func` without `pressure_data` raises clearly.
- Non-finite pressure, non-finite slope, and negative compression slope are
  rejected without emitting invalid contacts.
- Margin and gap classification is unchanged across scalar and rich-law paths.

### Unreduced physics

- Unequal-material planar contact characterizes current material-series behavior.
- Equal-material planar contact agrees between scalar and rich-law paths within
  SDF tolerance.
- Oblique/conforming contact asserts projected tangent and solver distance.
- Shape A/B swap preserves physical results.
- Static face force is identical between scalar and rich-law linear paths.
- Invalid gradients produce no NaN, infinity, or negative stiffness.
- At least two SDF resolutions exercise gradient convergence.

### Reduced physics

- Reduced and unreduced aggregate pressure force agree.
- Reduced and unreduced scalar tangent budgets agree.
- Each selected representative retains its prior normal-force weight.
- Pre-pruning enabled and disabled.
- Normal matching enabled and disabled.
- Anchor enabled and disabled.
- Moment matching enabled.
- Voxel-bin duplicate paths spend the tangent budget once.
- Missing-bin fallback uses the stored per-face rich-law pair.

### Determinism

- Repeated reduced rich-law runs are bit-exact.
- Repeated unreduced rich-law runs are bit-exact after contact sorting.
- Fixed-point `K_bin` agrees with the unreduced reference within its quantization
  tolerance.
- Anchor and moment-matching deterministic cases remain covered.
- Rich-law-path fingerprints and sort subkeys remain stable.

### Solver integration

- Assert public `rigid_contact_stiffness` directly.
- Reconstruct signed distance from `rigid_contact_point0/1` and assert it directly.
- Verify `k*(-phi)` is unchanged.
- Exercise Newton contacts in SemiImplicit, Featherstone, XPBD/VBD where
  supported, and MuJoCo Warp.
- Check that contact-point rescaling along the normal does not alter static
  force or normal-force moment.

## Expected file changes

- `newton/_src/geometry/sdf_hydroelastic.py`
  - richer callback field, resolution, and validation;
  - public built-in rich linear law implementation;
  - private pressure-only adapter and static kernel selection;
  - penetrating gradient sampling;
  - per-face tangent storage;
  - unreduced solver mapping.
- `newton/geometry.py`
  - canonical public export of `hydroelastic_pressure_law_linear`.
- `newton/_src/geometry/contact_reduction_global.py`
  - tangent and winner-count arrays;
  - data-struct, allocation, clear, and empty-array plumbing.
- `newton/_src/geometry/contact_reduction_hydroelastic.py`
  - nondeterministic and deterministic tangent aggregation;
  - fixed-point slots and scale family;
  - reduced tangent allocation and distance rescaling.
- `newton/tests/test_hydroelastic.py`
  - characterization, regression, reduction, margin/gap, and determinism tests.
- `docs/concepts/collisions.rst`
  - scalar and rich pressure-law interfaces, raw-gradient policy, and reduction
    limitation.
- `changelog/+hydro-series-gradient-<id>.added.md`
  - Towncrier fragment for the opt-in richer pressure-law callback and built-in
    linear law.

The new public function requires an explicit `newton.geometry` export. Follow
the public-interface documentation procedure, update `__all__`, and run API
generation to verify the generated reference output.

## Development sequence

Suggested commits:

1. `Add hydro pressure law callback`
2. `Implement unreduced series gradients`
3. `Preserve series tangent in reduction`
4. `Document hydro series gradients`

These are development checkpoints, not separately advertised releases. Do not
advertise the richer pressure-law callback or public built-in law until both
reduced and unreduced paths are complete.

For each behavior change:

1. Add the regression first.
2. Confirm it fails without the implementation.
3. Implement the smallest stage.
4. Run focused tests.
5. Run the broader suite before committing the completed feature.

Primary commands:

```bash
uv run --extra dev -m newton.tests -k test_hydroelastic
uv run --extra dev -m newton.tests
uvx pre-commit run -a
```

Preview the Towncrier fragment according to `changelog/README.md`, using the
pinned Towncrier version specified by the repository workflow instructions.

## Definition of done

- [x] The default and custom scalar-pressure paths are behaviorally and
      performance compatible with current main.
- [x] The issue's oblique/conforming regression fails before and passes after
      the change.
- [x] The public built-in rich linear law exports unreduced `k=A*g` and
      `phi=-p/g`.
- [x] A custom nonlinear rich law uses its analytical compression slope and
      exports the expected projected tangent.
- [x] Reduced rich-law contacts preserve aggregate force and scalar tangent.
- [x] Deterministic rich-law reduction uses fixed-point tangent accumulation.
- [x] Pair separation remains the source of truth for margin/gap classification.
- [x] Speculative activation behavior remains unchanged and documented.
- [x] Existing custom `pressure_func` behavior remains unchanged.
- [x] Callback mutual exclusion and custom-data validation are covered.
- [x] All emitted stiffnesses and distances are finite with positive
      penetrating stiffness.
- [x] Documentation and a Towncrier fragment describe the opt-in behavior.
- [x] Focused hydroelastic tests and pre-commit pass.
- [ ] The full repository suite passes without infrastructure errors. The
      6,342-test run completed with 6,177 passes, 164 skips, and one CUDA
      kernel-build/cache error in
      `test_mujoco_hydroelastic_penetration_depth_cuda_0`; that test passed in
      both the dedicated hydroelastic run and its isolated retry.

## Follow-up verification before upstream PR

- Add shape-swap and multi-resolution convergence coverage.
- Add repeated deterministic unreduced and fixed-point quantization checks.
- Exercise the rich callback through each supported solver integration.
- Benchmark gradient sampling on representative hydroelastic scenes.
- Revisit callback preflight validation and invalid-face counters when Warp and
  the collision diagnostics API provide stable extension points.
