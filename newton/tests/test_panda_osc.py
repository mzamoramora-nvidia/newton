# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the Panda OSC example.

Focus is on the parts that the OSC implementation owns:

* TCP frame extraction from ``body_q`` / ``body_qd`` (Newton reports velocity
  at the COM, OSC needs it at the TCP).
* TCP-frame Jacobian shift (the linear rows are translated from COM to TCP).
* Arm-DOF slicing inside the multi-world flat ``joint_f`` array.
* Steady-state and reach behavior of the full OSC pipeline.

Newton's underlying ``eval_jacobian`` / ``eval_mass_matrix`` are tested in
``test_jacobian_mass_matrix.py``; we don't re-prove those here.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.utils
from newton.examples.robot.osc import OSCController, scatter_arm_torque_to_joint_f_kernel
from newton.tests.unittest_utils import add_function_test, get_test_devices

# ---------------------------------------------------------------------------
# Test fixture: build a single-world (or replicated) Panda the same way the
# example does, but without table or solver - we only need FK / Jacobian /
# kernel testing here.
# ---------------------------------------------------------------------------

EE_BODY_INDEX = 11
N_ARM_DOFS = 7
N_DOFS_PER_WORLD = 9

INIT_ARM_Q = (
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236e00,
    -1.2918962e-04,
    2.3922248e00,
    7.8549200e-01,
)
INIT_FINGER_Q = (0.05, 0.05)


def _build_panda_model(device, world_count: int = 1):
    """Build a Panda model identical to the OSC example (sans table/solver)."""
    robot_builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)
    robot_builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=True,
    )
    robot_builder.joint_q[:N_DOFS_PER_WORLD] = [*INIT_ARM_Q, *INIT_FINGER_Q]

    scene = newton.ModelBuilder()
    scene.replicate(robot_builder, world_count)
    model = scene.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state


def _make_osc(model, world_count: int = 1) -> OSCController:
    art_view = newton.selection.ArticulationView(model, "fr3")
    return OSCController(
        model=model,
        articulation_view=art_view,
        world_count=world_count,
        ee_body_index=EE_BODY_INDEX,
        tcp_offset_local=wp.transform_identity(),
        n_arm_dofs=N_ARM_DOFS,
        num_bodies_per_world=model.body_count // world_count,
        n_dofs_per_world=N_DOFS_PER_WORLD,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tcp_pose_identity_offset(test, device):
    """With identity TCP offset, kernel-extracted TCP pose must equal body_q[ee]."""
    model, state = _build_panda_model(device, world_count=1)
    osc = _make_osc(model, world_count=1)
    osc.update_tcp_state(state)

    body_q = state.body_q.numpy()
    ee_xform = body_q[EE_BODY_INDEX]
    tcp_pos = osc.tcp_pos.numpy()[0]
    tcp_quat = osc.tcp_quat.numpy()[0]

    np.testing.assert_allclose(tcp_pos, ee_xform[:3], atol=1e-6)
    # Quaternion sign-ambiguous: compare up to sign.
    cos_half = abs(float(np.dot(tcp_quat, ee_xform[3:])))
    test.assertAlmostEqual(cos_half, 1.0, places=5)


def _quat_log_xyz_w(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Return axis * angle for a quaternion (xyzw), short-path corrected.

    Used only inside the finite-difference test to validate orientation
    derivatives. The OSC implementation itself uses Warp's helpers.
    """
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    sin_half = float(np.sqrt(max(0.0, 1.0 - qw * qw)))
    if sin_half < 1e-9:
        return np.array([2.0 * qx, 2.0 * qy, 2.0 * qz])
    angle = 2.0 * float(np.arccos(np.clip(qw, -1.0, 1.0)))
    axis = np.array([qx, qy, qz]) / sin_half
    return axis * angle


def _quat_mul(a, b):
    """Hamilton product of two xyzw quats."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def test_tcp_jacobian_matches_finite_difference(test, device):
    """J_tcp must match a central finite-difference Jacobian of TCP pose w.r.t. q."""
    model, state = _build_panda_model(device, world_count=1)
    osc = _make_osc(model, world_count=1)

    q0 = state.joint_q.numpy().copy()
    qd0 = np.zeros_like(state.joint_qd.numpy())
    state.joint_q.assign(q0)
    state.joint_qd.assign(qd0)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    osc.update_tcp_state(state)
    osc.update_tcp_jacobian(state)

    j_tcp = osc.j_tcp.numpy()[0]  # (6, n_arm_dofs)

    # Central difference: O(h^2). h=1e-3 keeps FD noise below ~5e-4 with
    # float32 storage of body_q.
    h = 1e-3
    j_fd = np.zeros((6, N_ARM_DOFS), dtype=np.float64)
    for d in range(N_ARM_DOFS):
        # +h
        q_plus = q0.copy()
        q_plus[d] += h
        state.joint_q.assign(q_plus)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        osc.update_tcp_state(state)
        p_plus = osc.tcp_pos.numpy()[0].copy()
        q_plus_quat = osc.tcp_quat.numpy()[0].copy()

        # -h
        q_minus = q0.copy()
        q_minus[d] -= h
        state.joint_q.assign(q_minus)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        osc.update_tcp_state(state)
        p_minus = osc.tcp_pos.numpy()[0].copy()
        q_minus_quat = osc.tcp_quat.numpy()[0].copy()

        # Linear: central difference in position.
        j_fd[0:3, d] = (p_plus - p_minus) / (2.0 * h)

        # Angular: log( q_plus * q_minus^-1 ) / (2h). For small h, this is
        # twice the rotation between the two perturbations.
        q_rel = _quat_mul(q_plus_quat, _quat_inv(q_minus_quat))
        aa = _quat_log_xyz_w(*q_rel)
        j_fd[3:6, d] = aa / (2.0 * h)

    state.joint_q.assign(q0)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    rel_err = np.linalg.norm(j_tcp - j_fd) / max(np.linalg.norm(j_tcp), 1e-9)
    test.assertLess(
        rel_err,
        2e-3,
        msg=f"TCP Jacobian mismatch vs central FD, rel_err={rel_err:.2e}\nanalytic=\n{j_tcp}\nFD=\n{j_fd}",
    )


def test_tcp_jacobian_matches_body_qd_after_shift(test, device):
    """Exact (no-FD) check: J_tcp @ qd_arm must equal the TCP twist derived
    from ``state.body_qd`` after the COM->TCP shift.

    Newton's J_full satisfies ``J_full @ qd == body_qd`` (covered by
    test_jacobian_mass_matrix.py). This test then verifies our TCP shift on
    both the Jacobian and the velocity is internally consistent.
    """
    model, state = _build_panda_model(device, world_count=1)
    osc = _make_osc(model, world_count=1)

    qd_arm = np.array([0.1, -0.2, 0.15, -0.3, 0.05, 0.25, -0.1], dtype=np.float32)
    qd_full = np.zeros(model.joint_dof_count, dtype=np.float32)
    qd_full[:N_ARM_DOFS] = qd_arm
    state.joint_qd.assign(qd_full)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    osc.update_tcp_state(state)
    osc.update_tcp_jacobian(state)

    pred_twist = osc.j_tcp.numpy()[0] @ qd_arm.astype(np.float64)
    measured_twist = np.concatenate([osc.tcp_linvel.numpy()[0], osc.tcp_angvel.numpy()[0]])

    np.testing.assert_allclose(pred_twist, measured_twist, atol=1e-4, rtol=0)


def test_mass_matrix_arm_block_is_symmetric_and_pd(test, device):
    """The 7x7 arm sub-block of H must be symmetric and positive-definite."""
    model, state = _build_panda_model(device, world_count=1)
    art_view = newton.selection.ArticulationView(model, "fr3")
    H = art_view.eval_mass_matrix(state).numpy()  # (1, n_dofs, n_dofs)
    H_arm = H[0, :N_ARM_DOFS, :N_ARM_DOFS]

    np.testing.assert_allclose(H_arm, H_arm.T, atol=1e-5, rtol=0)
    # Cholesky proves PD.
    np.linalg.cholesky(H_arm)
    # Diagonal must be positive and order-of-magnitude reasonable for a Franka.
    diag = np.diag(H_arm)
    test.assertTrue((diag > 0).all())
    test.assertLess(diag.max(), 100.0)


def test_mass_matrix_arm_slice_excludes_finger_dofs(test, device):
    """Slicing H[:7,:7] must give a different matrix from any 7-DOF slice that
    includes a finger column. Catches the indexing bug class equivalent to
    IsaacLab's ``test_floating_base_osc_action_term_indexing``.
    """
    model, state = _build_panda_model(device, world_count=1)
    art_view = newton.selection.ArticulationView(model, "fr3")
    H = art_view.eval_mass_matrix(state).numpy()[0]
    H_arm = H[:N_ARM_DOFS, :N_ARM_DOFS]
    # Wrong slice: drop the last arm DOF, include the first finger DOF.
    bad_idx = [*range(N_ARM_DOFS - 1), N_ARM_DOFS]
    H_bad = H[np.ix_(bad_idx, bad_idx)]
    test.assertFalse(np.allclose(H_arm, H_bad, atol=1e-6))


def test_zero_target_error_produces_zero_torque(test, device):
    """With target == current TCP and zero velocity, OSC torque should be ~0.

    Validates the whole pipeline (pose error -> wrench -> J^T -> joint_f scatter)
    has no constant-offset bug.
    """
    if device.is_cpu:
        # ArticulationView eval_jacobian / eval_mass_matrix only support CUDA.
        test.skipTest("OSC kernels currently target CUDA only")
    model, state = _build_panda_model(device, world_count=1)
    osc = _make_osc(model, world_count=1)

    osc.update_tcp_state(state)
    osc.update_tcp_jacobian(state)
    osc.set_target(osc.tcp_pos, osc.tcp_quat)

    kp = wp.array([[200.0] * 3 + [50.0] * 3], dtype=float, device=device)
    kd = wp.array([[28.28] * 3 + [14.14] * 3], dtype=float, device=device)
    osc.set_gains(kp, kd)

    control = model.control()
    osc.compute_torques(control)

    arm_torque = osc.arm_torque.numpy()[0]
    np.testing.assert_allclose(arm_torque, np.zeros(N_ARM_DOFS), atol=1e-5)


def test_multi_world_targets_are_independent(test, device):
    """With 4 worlds and 4 different targets, each world's torque must differ.

    Validates that target_pos / target_quat / gains are read per-world and that
    pose error / J_tcp / arm_torque are sliced per-world. A single-world bug
    (e.g. only world 0 ever read) would produce identical torques in all worlds.
    """
    if device.is_cpu:
        test.skipTest("OSC kernels currently target CUDA only")
    world_count = 4
    model, state = _build_panda_model(device, world_count=world_count)
    osc = _make_osc(model, world_count=world_count)

    osc.update_tcp_state(state)
    osc.update_tcp_jacobian(state)

    # Targets: world w gets a position offset of (0.05*w, 0, 0) from current TCP.
    tcp_pos = osc.tcp_pos.numpy()
    tcp_quat = osc.tcp_quat.numpy()
    target_pos_host = tcp_pos.copy()
    for w in range(world_count):
        target_pos_host[w][0] += 0.05 * w  # world 0 has zero error
    target_pos = wp.array(target_pos_host, dtype=wp.vec3, device=device)
    target_quat = wp.array(tcp_quat, dtype=wp.vec4, device=device)
    osc.set_target(target_pos, target_quat)

    kp_host = np.tile([200.0, 200.0, 200.0, 50.0, 50.0, 50.0], (world_count, 1))
    kd_host = 2.0 * np.sqrt(kp_host)
    osc.set_gains(
        wp.array(kp_host, dtype=float, device=device),
        wp.array(kd_host, dtype=float, device=device),
    )

    control = model.control()
    osc.compute_torques(control)

    arm_torque = osc.arm_torque.numpy()
    # World 0 has zero error -> zero torque.
    np.testing.assert_allclose(arm_torque[0], np.zeros(N_ARM_DOFS), atol=1e-5)
    # Other worlds have non-zero, unique torques.
    for w in range(1, world_count):
        test.assertGreater(np.linalg.norm(arm_torque[w]), 1e-3)
    # Worlds must not collapse into the same torque vector.
    for w in range(1, world_count - 1):
        diff = np.linalg.norm(arm_torque[w] - arm_torque[w + 1])
        test.assertGreater(diff, 1e-4)


def test_arm_torque_scatter_leaves_finger_dofs_untouched(test, device):
    """``scatter_arm_torque_to_joint_f_kernel`` must only write the arm slice."""
    if device.is_cpu:
        test.skipTest("OSC kernels currently target CUDA only")
    world_count = 2
    model, state = _build_panda_model(device, world_count=world_count)
    osc = _make_osc(model, world_count=world_count)

    osc.update_tcp_state(state)
    osc.update_tcp_jacobian(state)
    # Set a non-trivial arm torque manually (bypass compute_torques).
    arm_torque_host = np.zeros((world_count, N_ARM_DOFS), dtype=np.float32)
    arm_torque_host[:, :] = 1.5
    osc.arm_torque = wp.array(arm_torque_host, dtype=float, device=device)

    # Pre-fill joint_f with a sentinel value to detect over-writes on finger DOFs.
    control = model.control()
    sentinel = -7.0
    n_total_dofs = world_count * N_DOFS_PER_WORLD
    control.joint_f = wp.array(np.full(n_total_dofs, sentinel, dtype=np.float32), dtype=float, device=device)

    wp.launch(
        scatter_arm_torque_to_joint_f_kernel,
        dim=(world_count, N_ARM_DOFS),
        inputs=[osc.arm_torque, N_ARM_DOFS, N_DOFS_PER_WORLD],
        outputs=[control.joint_f],
        device=device,
    )

    jf = control.joint_f.numpy()
    for w in range(world_count):
        # Arm DOFs got the new torque.
        np.testing.assert_allclose(jf[w * N_DOFS_PER_WORLD : w * N_DOFS_PER_WORLD + N_ARM_DOFS], 1.5, atol=1e-6)
        # Finger DOFs (7, 8) remain at sentinel.
        np.testing.assert_allclose(
            jf[w * N_DOFS_PER_WORLD + N_ARM_DOFS : (w + 1) * N_DOFS_PER_WORLD],
            sentinel,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Test registration
# ---------------------------------------------------------------------------


class TestPandaOSC(unittest.TestCase):
    pass


# Most tests need CUDA because Newton's eval_jacobian/eval_mass_matrix kernels
# target CUDA via ArticulationView. Use only CUDA devices when available.
devices = [d for d in get_test_devices() if d.is_cuda]
if not devices:
    # Fall back to whatever's available so the file at least loads.
    devices = get_test_devices()

add_function_test(TestPandaOSC, "test_tcp_pose_identity_offset", test_tcp_pose_identity_offset, devices=devices)
add_function_test(
    TestPandaOSC,
    "test_tcp_jacobian_matches_finite_difference",
    test_tcp_jacobian_matches_finite_difference,
    devices=devices,
)
add_function_test(
    TestPandaOSC,
    "test_tcp_jacobian_matches_body_qd_after_shift",
    test_tcp_jacobian_matches_body_qd_after_shift,
    devices=devices,
)
add_function_test(
    TestPandaOSC,
    "test_mass_matrix_arm_block_is_symmetric_and_pd",
    test_mass_matrix_arm_block_is_symmetric_and_pd,
    devices=devices,
)
add_function_test(
    TestPandaOSC,
    "test_mass_matrix_arm_slice_excludes_finger_dofs",
    test_mass_matrix_arm_slice_excludes_finger_dofs,
    devices=devices,
)
add_function_test(
    TestPandaOSC,
    "test_zero_target_error_produces_zero_torque",
    test_zero_target_error_produces_zero_torque,
    devices=devices,
)
add_function_test(
    TestPandaOSC,
    "test_arm_torque_scatter_leaves_finger_dofs_untouched",
    test_arm_torque_scatter_leaves_finger_dofs_untouched,
    devices=devices,
)
add_function_test(
    TestPandaOSC,
    "test_multi_world_targets_are_independent",
    test_multi_world_targets_are_independent,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
