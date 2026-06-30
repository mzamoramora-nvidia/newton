# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity tests for the example ``--load-usd`` flag.

For each deformable example that supports ``--load-usd``, build it both ways
(procedural default and from the bundled USD asset) and assert the constructed
model and initial simulation state are equivalent: identical element counts and
matching point clouds / mass multisets. This guarantees the USD round-trip
reproduces the procedural setup; trajectory parity beyond t=0 is not asserted, as
the asset's float round-trip leaves sub-1e-4 differences a chaotic solve would
amplify.

Each build runs in its own subprocess (like the example tests) so global
state from one example never leaks into another's construction.
"""

import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test

# Examples carrying a --load-usd flag (procedural default + USD opt-in).
_LOAD_USD_EXAMPLES = [
    "newton.examples.cable.example_cable_twist",
    "newton.examples.cable.example_cable_pile",
    "newton.examples.cable.example_cable_bundle_hysteresis",
    "newton.examples.cable.example_cable_cross_slide_table",
    "newton.examples.cloth.example_cloth_bending",
    "newton.examples.cloth.example_cloth_hanging",
    "newton.examples.softbody.example_softbody_hanging",
    "newton.examples.multiphysics.example_rigid_soft_contact",
]


def _dump_initial_state(module_path: str, load_usd: bool, out_path: str):
    """Build the example on CPU and save its counts + initial state to ``out_path``."""
    mod = importlib.import_module(module_path)
    parser = mod.Example.create_parser()
    args = newton.examples.default_args(parser)
    args.load_usd = load_usd

    viewer = newton.viewer.ViewerNull(num_frames=1)
    with wp.ScopedDevice("cpu"):
        example = mod.Example(viewer, args)

    model = example.model
    state = getattr(example, "state_0", None) or model.state()
    arrays = {
        "particle_count": np.int64(model.particle_count),
        "body_count": np.int64(model.body_count),
        "joint_count": np.int64(model.joint_count),
        "tri_count": np.int64(getattr(model, "tri_count", 0)),
        "edge_count": np.int64(getattr(model, "edge_count", 0)),
        "tet_count": np.int64(getattr(model, "tet_count", 0)),
    }
    particle_pos = None
    if model.particle_count:
        particle_pos = state.particle_q.numpy()
        arrays["particle_pos"] = particle_pos
        arrays["particle_mass"] = model.particle_mass.numpy()
    if model.body_count:
        arrays["body_pos"] = state.body_q.numpy()[:, :3]
        arrays["body_mass"] = model.body_mass.numpy()

    # Connectivity (canonicalized by vertex position so it is independent of vertex
    # / element ordering) and per-element stiffness, so a topology or material-mapping
    # change between the two build paths is caught, not just geometry + mass.
    def _store_elements(prefix, indices_attr, material_attr):
        idx = getattr(model, indices_attr, None)
        if idx is None or particle_pos is None:
            return
        idx = idx.numpy()
        if idx.size:
            arrays[f"{prefix}_canon"] = _canon_elements(particle_pos, idx)
        mat = getattr(model, material_attr, None)
        if mat is not None and mat.numpy().size:
            arrays[f"{prefix}_material"] = mat.numpy()

    _store_elements("tri", "tri_indices", "tri_materials")
    _store_elements("tet", "tet_indices", "tet_materials")
    edge_bending = getattr(model, "edge_bending_properties", None)
    if edge_bending is not None and edge_bending.numpy().size:
        arrays["edge_bending"] = edge_bending.numpy()
    if model.joint_count:
        arrays["joint_target_ke"] = model.joint_target_ke.numpy()
    np.savez(out_path, **arrays)


def _canon_elements(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Order-independent connectivity signature: each element's vertex positions
    sorted within the element, then all elements sorted. Robust to vertex/element
    reordering between the two build paths."""
    rows = []
    for elem in indices:
        verts = positions[elem]
        rows.append(verts[np.lexsort(verts.T[::-1])].flatten())
    out = np.array(rows, dtype=np.float64)
    return out[np.lexsort(out.T[::-1])]


def _sorted_rows(arr: np.ndarray) -> np.ndarray:
    return arr if arr.size == 0 else arr[np.lexsort(arr.T[::-1])]


def _build_in_subprocess(module_path: str, load_usd: bool, out_path: str):
    """Run _dump_initial_state for one (example, flag) in an isolated subprocess."""
    cmd = [sys.executable, __file__, module_path, "1" if load_usd else "0", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"building {module_path} (load_usd={load_usd}) failed:\n{result.stderr}")


def _check_load_usd_parity(test: unittest.TestCase, device, module_path: str):
    del device  # builds run on CPU inside the subprocess
    with tempfile.TemporaryDirectory() as tmp:
        proc_path = str(Path(tmp) / "proc.npz")
        usd_path = str(Path(tmp) / "usd.npz")
        _build_in_subprocess(module_path, False, proc_path)
        _build_in_subprocess(module_path, True, usd_path)
        proc = np.load(proc_path)
        usd = np.load(usd_path)

        for key in ("particle_count", "body_count", "joint_count", "tri_count", "edge_count", "tet_count"):
            test.assertEqual(
                int(proc[key]), int(usd[key]), f"{module_path}: {key} differs ({int(proc[key])} vs {int(usd[key])})"
            )
        for key, atol in (("particle_pos", 1e-3), ("body_pos", 1e-3)):
            if key in proc.files:
                np.testing.assert_allclose(
                    _sorted_rows(usd[key]), _sorted_rows(proc[key]), atol=atol, err_msg=f"{module_path}: {key} differs"
                )
        # body_mass is intentionally NOT compared for rigid cables: the importer applies the AOUSD
        # cylinder-volume cable mass while the procedural add_rod path uses the capsule
        # collision-shape mass (cylinder + hemispherical caps), so the two diverge by the cap term.
        for key in ("particle_mass",):
            if key in proc.files:
                np.testing.assert_allclose(
                    np.sort(usd[key]), np.sort(proc[key]), rtol=1e-4, atol=1e-6, err_msg=f"{module_path}: {key} differs"
                )

        # Connectivity: canonicalized element signatures must match exactly.
        for key in ("tri_canon", "tet_canon"):
            if key in proc.files:
                np.testing.assert_allclose(
                    usd[key], proc[key], atol=1e-3, err_msg=f"{module_path}: {key} (connectivity) differs"
                )

        # Per-element stiffness multisets (order-independent).
        for key in ("tri_material", "tet_material", "edge_bending"):
            if key in proc.files:
                np.testing.assert_allclose(
                    _sorted_rows(usd[key]),
                    _sorted_rows(proc[key]),
                    rtol=1e-3,
                    atol=1e-6,
                    err_msg=f"{module_path}: {key} (stiffness) differs",
                )
        if "joint_target_ke" in proc.files:
            # Cable per-joint stiffness round-trips through the base-schema modulus
            # (stiffness = modulus * I / segment_length). For a non-straight cable the
            # mean segment length differs a few percent from the value the example used
            # to set its uniform stiffness directly, so allow a loose tolerance here -
            # it still catches gross (e.g. 2x) conversion errors.
            np.testing.assert_allclose(
                np.sort(usd["joint_target_ke"]),
                np.sort(proc["joint_target_ke"]),
                rtol=5e-2,
                atol=1e-6,
                err_msg=f"{module_path}: joint_target_ke (stiffness) differs",
            )


class TestExampleLoadUsdParity(unittest.TestCase):
    """``--load-usd`` reproduces the procedural example setup."""


for _module_path in _LOAD_USD_EXAMPLES:
    _short = _module_path.rsplit(".", 1)[1].removeprefix("example_")
    add_function_test(
        TestExampleLoadUsdParity,
        f"test_load_usd_parity_{_short}",
        _check_load_usd_parity,
        devices=None if USD_AVAILABLE else [],
        check_output=False,
        module_path=_module_path,
    )


if __name__ == "__main__":
    # Subprocess entry point: build one example variant and dump its initial state.
    _dump_initial_state(sys.argv[1], sys.argv[2] == "1", sys.argv[3])
