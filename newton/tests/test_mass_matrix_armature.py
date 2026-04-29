# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test for whether eval_mass_matrix folds joint_armature into H.

Newton's :func:`newton.eval_mass_matrix` (and :meth:`ArticulationView.eval_mass_matrix`)
returns ``H = J^T M J`` -- the rigid-body inertia matrix only. Many OSC
formulations expect ``H_eff = H + diag(joint_armature)`` so that effort
applied through ``joint_f`` accelerates the joint via the same effective
inertia the integrator sees.

This test pins the current behavior (no armature) and documents the
expectation. If the upstream behavior ever changes to fold armature in
automatically, this test will fail and the OSC's local patch should be
removed.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_2dof_chain_with_armature(device, armature_values):
    """Build a 2-DOF revolute chain with explicit per-DOF armature."""
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)

    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    b2 = builder.add_link(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        mass=2.0,
    )
    builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)

    j1 = builder.add_joint_revolute(
        parent=-1,
        child=b1,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        armature=armature_values[0],
    )
    j2 = builder.add_joint_revolute(
        parent=b1,
        child=b2,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        armature=armature_values[1],
    )
    builder.add_articulation([j1, j2], label="chain")

    return builder.finalize(device=device)


def test_mass_matrix_does_not_include_armature(test, device):
    """Pin the current behavior: ``H`` from ``eval_mass_matrix`` is ``J^T M J`` only.

    Build a 2-DOF chain with non-zero ``joint_armature``, evaluate the mass
    matrix once with armature and once without, and confirm both match. This
    proves armature is *not* added to ``H``'s diagonal.
    """
    armature_values = [0.123, 0.456]

    # Model with armature.
    model_with = _build_2dof_chain_with_armature(device, armature_values)
    state_with = model_with.state()
    newton.eval_fk(model_with, state_with.joint_q, state_with.joint_qd, state_with)
    H_with = newton.eval_mass_matrix(model_with, state_with).numpy()[0]

    # Model without armature.
    model_without = _build_2dof_chain_with_armature(device, [0.0, 0.0])
    state_without = model_without.state()
    newton.eval_fk(model_without, state_without.joint_q, state_without.joint_qd, state_without)
    H_without = newton.eval_mass_matrix(model_without, state_without).numpy()[0]

    # Same H regardless of armature -> armature is not added to H.
    np.testing.assert_allclose(H_with, H_without, atol=1.0e-6, rtol=1.0e-6)

    # Sanity: the model actually stores the armature values we set.
    armature_np = model_with.joint_armature.numpy()
    np.testing.assert_allclose(armature_np, np.asarray(armature_values, dtype=np.float32))


def test_mass_matrix_plus_armature_is_consumer_responsibility(test, device):
    """Document the expected consumer-side patch: ``H_eff = H + diag(armature)``.

    OSC controllers that need the effective inertia must add ``joint_armature``
    to the diagonal at the consumer site (see
    ``newton/examples/robot/osc.py`` for the local patch).
    """
    armature_values = [0.5, 0.25]
    model = _build_2dof_chain_with_armature(device, armature_values)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    H = newton.eval_mass_matrix(model, state).numpy()[0]

    armature_np = model.joint_armature.numpy()
    H_eff = H + np.diag(armature_np)

    # Diagonal entries differ by exactly the armature values.
    diag_diff = np.diag(H_eff) - np.diag(H)
    np.testing.assert_allclose(diag_diff, armature_np, atol=1.0e-6, rtol=1.0e-6)
    # Off-diagonal entries are unchanged.
    np.testing.assert_allclose(H_eff - np.diag(np.diag(H_eff)), H - np.diag(np.diag(H)), atol=1.0e-6)


def test_articulation_view_mass_matrix_no_armature(test, device):
    """Same pin via the ArticulationView API used by OSCController."""
    armature_values = [0.2, 0.4]
    model = _build_2dof_chain_with_armature(device, armature_values)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    view = newton.selection.ArticulationView(model, pattern="chain")
    H = view.eval_mass_matrix(state).numpy()[0]

    # Re-evaluate with zero armature: should produce the same matrix because
    # eval_mass_matrix doesn't read joint_armature.
    model_zero = _build_2dof_chain_with_armature(device, [0.0, 0.0])
    state_zero = model_zero.state()
    newton.eval_fk(model_zero, state_zero.joint_q, state_zero.joint_qd, state_zero)
    view_zero = newton.selection.ArticulationView(model_zero, pattern="chain")
    H_zero = view_zero.eval_mass_matrix(state_zero).numpy()[0]

    np.testing.assert_allclose(H, H_zero, atol=1.0e-6, rtol=1.0e-6)


class TestMassMatrixArmature(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestMassMatrixArmature,
    "test_mass_matrix_does_not_include_armature",
    test_mass_matrix_does_not_include_armature,
    devices=devices,
)
add_function_test(
    TestMassMatrixArmature,
    "test_mass_matrix_plus_armature_is_consumer_responsibility",
    test_mass_matrix_plus_armature_is_consumer_responsibility,
    devices=devices,
)
add_function_test(
    TestMassMatrixArmature,
    "test_articulation_view_mass_matrix_no_armature",
    test_articulation_view_mass_matrix_no_armature,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
