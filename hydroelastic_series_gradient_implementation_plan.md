# Hydroelastic pressure-law and series-gradient implementation plan

Status: implemented and validated locally
Issue: [newton-physics/newton#3503](https://github.com/newton-physics/newton/issues/3503)
Rebased on Newton commit: `e5cc054bb95a6ba8889da983b0fdab3d977d32c9`
Feature branch: `mzamoramora/hydro-series-gradient`
Worktree: `/home/mzamoramora/build_playground/newton/hydro-private/newton-hydro-series-gradient`

## Executive decision

Do not add a pressure-law descriptor class and do not add a closed
`PAIR_SEPARATION` / `SERIES_GRADIENT` enum. Newton generally favors flat Config
data and Warp callbacks, so expose one canonical callback field plus explicit
host-static callback-contract metadata:

```python
@dataclass
class HydroelasticSDF.Config:
    pressure_law_func: Any = None
    pressure_data: Any = None
    pressure_law_returns_tangent: bool | None = None
    pressure_func: Any = None  # Deprecated transition alias.
```

`pressure_law_func` accepts one of two documented Warp callback contracts:

```python
pressure_law_func(signed_depth, shape_idx, pressure_data) -> pressure

pressure_law_func(signed_depth, shape_idx, pressure_data) \
    -> wp.vec2f(pressure, compression_slope)
```

`pressure_law_returns_tangent` describes which callback contract is supplied;
it is not a solver mode and cannot vary per contact. For Newton's public
built-ins, `None` resolves from the exact built-in function identity. For a
custom callback, `None` means pressure-only and a tangent-returning callback
must set the field to `True`. This avoids relying on unstable Warp return-type
introspection or a numerical sentinel such as zero or NaN.

Here `compression_slope = -dp/dd` is nonnegative for a monotonically increasing
pressure under compression and has units Pa/m.

Provide two flat public built-ins with distinct, prefix-clustered names:

```python
from newton.geometry import (
    HydroelasticSDF,
    hydroelastic_pressure_law_linear,
    hydroelastic_pressure_law_linear_tangent,
)

# Default pair-separation/secant formulation, written explicitly.
default_config = HydroelasticSDF.Config(
    pressure_law_func=hydroelastic_pressure_law_linear,
)

# Projected-series tangent formulation requested by issue #3503.
tangent_config = HydroelasticSDF.Config(
    pressure_law_func=hydroelastic_pressure_law_linear_tangent,
)
```

`Config()` remains exactly equivalent to the first form. The pair-separation
secant formulation is the supported default path. The tangent formulation is an
opt-in alternative.

Both built-ins represent the same constitutive law, `p = -kh*d`. Their names
distinguish the information returned to Newton and therefore the solver
linearization Newton can construct. Custom linear, power, cubic, or other laws
use the same flat fields; Newton remains responsible for mapping pressure and
optional slope to contacts.

The already-released `pressure_func` field cannot disappear immediately.
Deprecate it for at least one full minor release and resolve it internally as a
pressure-only `pressure_law_func`. Supplying both names raises `ValueError`.
The branch-local tangent-returning `pressure_law_func` interface has not
shipped, so it may be reshaped directly before upstreaming.

The implementation must keep two different quantities separate:

- `pair_separation`: margin-relative geometry used to classify penetrating,
  speculative, and rejected faces;
- `solver_distance`: the signed distance encoded in the solver contact points.

They are identical on the default pressure-only path. They differ for
penetrating faces when the selected law returns a tangent.

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

Replace the canonical `pressure_func` name with the more accurate
`pressure_law_func` name. Keep the shipped name temporarily as a deprecated
Config alias, but resolve both names to the same private flat state.

The pressure-only callback contract is:

```python
@wp.func
def my_pressure_law(
    signed_depth: float,
    shape_idx: int,
    pressure_data: MyPressureData,
) -> float:
    ...
```

The pressure-and-tangent contract uses the same field and arguments:

```python
@wp.func
def my_pressure_law_with_tangent(
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

The flat Config fields are:

```python
pressure_law_func: Any = None
pressure_data: Any = None
pressure_law_returns_tangent: bool | None = None
```

`pressure_law_returns_tangent` is callback ABI metadata resolved once during
pipeline construction. It is not a per-contact switch or a closed physics mode.
The resolution rules are:

```text
Config()                                      built-in linear, pressure-only
public linear law, metadata=None              pressure-only
public linear-tangent law, metadata=None      pressure-and-tangent
custom law, metadata=None/False               pressure-only
custom law, metadata=True                     pressure-and-tangent
deprecated pressure_func                      pressure-only adapter + warning
deprecated and canonical callbacks together  ValueError
custom callback without pressure_data         ValueError
```

Reject an explicit `pressure_law_returns_tangent` value that contradicts either
public built-in. This catches configuration mistakes without introspecting an
arbitrary Warp function's return type.

The pressure-and-tangent contract returns `wp.vec2f` rather than a tuple or
struct so it is straightforward to call from generated Warp kernels. Name and
document the two components at the Python interface; do not make callers infer
their order from examples alone.

Export two supported public built-ins:

```python
@wp.func
def hydroelastic_pressure_law_linear(
    signed_depth: float,
    shape_idx: int,
    pressure_data: LinearPressureData,
) -> float:
    ...


@wp.func
def hydroelastic_pressure_law_linear_tangent(
    signed_depth: float,
    shape_idx: int,
    pressure_data: LinearPressureData,
) -> wp.vec2f:
    ...
```

Expose both canonically from `newton.geometry`. When either exact built-in is
selected and `pressure_data` is omitted, construct the existing internal
`LinearPressureData` from `shape_material_kh`. A custom law requires explicit
`pressure_data`.

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

The tangent callback avoids finite-differencing arbitrary pressure-only laws in
contact kernels. Nonlinear custom laws provide their analytical local slope
directly.

Resolve the callback choice during `HydroelasticSDF` construction and specialize
generated kernels statically from `pressure_law_returns_tangent`. The
pressure-only specialization must not sample gradients, allocate
projected-tangent hot-path storage, or execute runtime callback-choice branches.

The default pressure-only path keeps pair separation as its solver distance and
uses the pressure-force secant stiffness. Do not infer its derivative with
finite differences. Do not call it legacy or compatibility behavior in code,
documentation, tests, or the changelog.

A tangent-returning law must return finite pressure and a finite, nonnegative
compression slope for every sampled signed depth. The pressure must remain
monotonically non-increasing in signed depth over the law's supported domain.
Debug validation can check sampled values, but the API contract must state
these requirements because a kernel cannot prove global monotonicity.

For a tangent-returning callback, use a private adapter returning
`pressure_law_func(...)[0]` at marching-cubes corners and other pressure-only
call sites. At accepted penetrating face centroids, evaluate the callback for
both shapes to obtain their separate slopes.

At a numerically imperfect equal-pressure face, define the face pressure
symmetrically:

```text
p_0 = 0.5 (p_A + p_B)
```

For the built-in linear law, `p_A` and `p_B` should agree up to extraction
tolerance. The symmetric definition avoids choosing a privileged shape for
custom or numerically imperfect laws. The default pressure-only path continues
using its current pressure evaluation exactly.

## Behavioral invariants

For every emitted penetrating source face on either path:

```text
k_i > 0
k_i and phi_i are finite
k_i (-phi_i) = A_i p_i
```

Additional tangent-returning/projected-gradient invariants are:

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
- solver mapping on the default pressure-only path.

Add a tangent-path-only per-face value:

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
storage on the default pressure-only path.

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
5. Add callback-contract tests for a custom tangent-returning linear law and a
   nonlinear law with a known analytical compression slope.
6. Confirm that the discriminating projected-gradient test fails on unmodified
   main.
7. Preserve existing margin/gap, custom pressure-only, deterministic,
   reduced-force, and moment-matching tests.

The unequal-material head-on case remains useful, but it must not be presented
as the primary regression because current main already handles it.

### Stage 1: flatten the pressure-law seam

In `newton/_src/geometry/sdf_hydroelastic.py`:

1. Make `pressure_law_func` the canonical Config callback field.
2. Add `pressure_law_returns_tangent` as flat, host-static callback-contract
   metadata. Do not expose a descriptor class or solver-mode enum.
3. Keep `pressure_data` flat and shared by both callback contracts.
4. Rename the current scalar built-in to the public
   `hydroelastic_pressure_law_linear` and rename the branch's
   tangent-returning built-in to `hydroelastic_pressure_law_linear_tangent`;
   export both from `newton.geometry`.
5. Infer callback-contract metadata for those exact built-ins. Require custom
   tangent-returning callbacks to set `pressure_law_returns_tangent=True`.
6. Retain `pressure_func` as a deprecated Config alias for at least one full
   minor release. Emit a migration warning, reject simultaneous old/new names,
   and resolve the alias into the canonical pressure-only state.
7. Synthesize internal `LinearPressureData` from `shape_material_kh` when either
   built-in is selected and data is omitted.
8. Resolve a private pressure-only adapter for marching-cubes and pruning call
   sites when the selected callback returns a tangent.
9. Statically specialize generation, decode, allocation, and reduction on the
   resolved callback-contract Boolean.

`Config()` and every deprecated `pressure_func` use must produce identical
output to current main. The existing physics implementation should be retained;
this stage changes ownership and naming at the public seam.

### Stage 2: implement unreduced projected-series contacts

Only after a face is classified as penetrating and the selected law returns a
tangent:

1. Recover or retain both margin-adjusted centroid depths. If only `d_B` and
   `d_pair` are currently retained, compute `d_A = d_pair - d_B`.
2. Evaluate the tangent-returning `pressure_law_func(d_A, shape_A, data)` and
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

Do not use the tangent-returning law's corner pressure as the accepted face
pressure. Evaluate both shapes at the accepted centroid as above; corner
evaluation remains an extraction/pruning concern. The default pressure-only
path retains its current face-pressure behavior exactly.

### Stage 3: accumulate the source tangent through reduction

For each valid tangent-returning-law penetrating source face, before
pre-pruning:

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

If a tangent-returning penetrating fallback has no valid normal-bin aggregate,
use its stored per-face tangent and per-face solver distance. Never invent a
pair-separation secant after a law has explicitly supplied a derivative.

After reduced tests pass, remove the temporary tangent-path/reduction configuration
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

- the default and custom pressure-only paths perform no new gradient samples;
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

- `Config()` preserves the current default pair-separation/secant behavior
  exactly.
- Existing custom pressure-only linear, power, cubic, and decoupled laws are
  unchanged through the deprecated `pressure_func` alias and match their
  canonical `pressure_law_func` configurations.
- Both public linear built-ins work without explicitly supplying
  `pressure_data`.
- A custom tangent-returning linear law matches the public linear-tangent law.
- A custom nonlinear tangent-returning law uses its supplied analytical slope.
- Supplying deprecated `pressure_func` with canonical `pressure_law_func`
  raises clearly at construction.
- A custom `pressure_law_func` without `pressure_data` raises clearly.
- A custom tangent-returning callback documents and supplies
  `pressure_law_returns_tangent=True`. Without that metadata Newton selects the
  pressure-only specialization, so an incompatible return type fails during
  Warp callback compilation; contextual return-contract preflight remains
  follow-up work.
- Non-finite pressure, non-finite slope, and negative compression slope are
  rejected without emitting invalid contacts.
- Margin and gap classification is unchanged across pressure-only and tangent
  paths.

### Unreduced physics

- Unequal-material planar contact characterizes current material-series behavior.
- Equal-material planar contact agrees between pressure-only and tangent paths
  within SDF tolerance.
- Oblique/conforming contact asserts projected tangent and solver distance.
- Shape A/B swap preserves physical results.
- Static face force is identical between the two built-in linear paths.
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
- Missing-bin fallback uses the stored per-face tangent pair.
- An unreliable normal-bin aggregate uses the stored per-face tangent pair
  instead of combining shared tangent stiffness with raw pair separation.

### Determinism

- Repeated reduced tangent-path runs are bit-exact.
- Repeated unreduced tangent-path runs are bit-exact after contact sorting.
- Fixed-point `K_bin` agrees with the unreduced reference within its quantization
  tolerance.
- Anchor and moment-matching deterministic cases remain covered.
- Tangent-path fingerprints and sort subkeys remain stable.

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
  - flat callback fields, contract resolution, deprecation, and validation;
  - public pressure-only and tangent-returning linear built-ins;
  - private pressure-only adapter and static kernel selection;
  - penetrating gradient sampling;
  - per-face tangent storage;
  - unreduced solver mapping.
- `newton/geometry.py`
  - canonical public exports of `hydroelastic_pressure_law_linear` and
    `hydroelastic_pressure_law_linear_tangent`.
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
  - default pressure-only and tangent-returning interfaces, deprecation
    migration, raw-gradient policy, and reduction limitation;
  - consistently call pair-separation/secant the default path, never legacy.
- `changelog/+hydro-series-gradient-<id>.added.md`
  - Towncrier fragment for the flat pressure-law callback, both linear
    built-ins, and the opt-in tangent formulation.

The two public functions require explicit `newton.geometry` exports. Follow the
public-interface documentation procedure, update `__all__`, and run API
generation to verify the generated reference output.

## Development sequence

The physics implementation is already present in the local feature commit. The
next local changes should be reviewable as an interface-focused commit:

1. `Flatten hydro pressure law interface`
   - add the canonical flat fields and both built-ins;
   - resolve one private pressure-only/tangent capability Boolean;
   - adapt the shipped `pressure_func` name with a deprecation warning;
   - update existing tests without changing contact physics.
2. `Document hydro pressure law choices`
   - describe the default and tangent formulations as peers;
   - add migration guidance and update the Towncrier fragment;
   - regenerate the public interface reference.
3. `Expand hydro tangent verification`
   - add the remaining shape-swap, resolution, deterministic, and solver cases
     before proposing the branch upstream.

These are development checkpoints, not separately advertised releases. Do not
advertise the flat interface until both reduced and unreduced tangent paths use
the same resolved callback contract.

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

- [x] `pressure_law_func`, `pressure_data`, and
      `pressure_law_returns_tangent` are the canonical flat Config fields.
- [x] No public pressure-law descriptor class or closed solver-mode enum is
      introduced.
- [x] `hydroelastic_pressure_law_linear` selects the explicit default
      pressure-only formulation.
- [x] `hydroelastic_pressure_law_linear_tangent` selects the projected-series
      tangent formulation without an extra flag at the common call site.
- [x] Custom tangent callbacks declare
      `pressure_law_returns_tangent=True`; no numerical sentinel or Warp
      return-type introspection selects the path.
- [x] The shipped `pressure_func` field remains functional with a documented
      deprecation warning and exact canonical replacement.
- [x] The default and custom pressure-only paths are behaviorally and
      performance compatible with current main.
- [x] The issue's oblique/conforming regression fails before and passes after
      the change.
- [x] The tangent-returning linear law exports unreduced `k=A*g` and
      `phi=-p/g`.
- [x] A custom nonlinear tangent-returning law uses its analytical compression
      slope and exports the expected projected tangent.
- [x] Reduced tangent-path contacts preserve aggregate force and scalar tangent.
- [x] Deterministic tangent reduction uses fixed-point tangent accumulation.
- [x] Pair separation remains the source of truth for margin/gap classification.
- [x] Speculative activation behavior remains unchanged and documented.
- [x] Deprecated/canonical callback conflict, custom data, and callback-contract
      metadata validation are covered.
- [x] All emitted stiffnesses and distances are finite with positive
      penetrating stiffness.
- [x] Documentation and the Towncrier fragment use “default” rather than
      “legacy” or “compatibility” for the pair-separation/secant formulation.
- [x] Focused hydroelastic tests and pre-commit pass after the flat-interface
      refactor.
- [x] The full repository suite passes: 6,344 tests in 1,573.860 seconds, with
      164 expected skips and no failures or errors. The previously intermittent
      `test_mujoco_hydroelastic_penetration_depth_cuda_0` test also passed in
      this clean run.

## Follow-up verification before upstream PR

- Add shape-swap and multi-resolution convergence coverage.
- Add repeated deterministic unreduced and fixed-point quantization checks.
- Exercise the tangent-returning callback through each supported solver
  integration.
- Benchmark gradient sampling on representative hydroelastic scenes.
- Revisit callback preflight validation and invalid-face counters when Warp and
  the collision diagnostics API provide stable extension points.
