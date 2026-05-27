# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.differentiable_contacts import launch_differentiable_hydroelastic_stiffness
from newton.geometry import HydroelasticSDF
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


@wp.kernel
def body_height_loss_kernel(body_q: wp.array[wp.transform], body_index: int, target_z: float, loss: wp.array[float]):
    p = wp.transform_get_translation(body_q[body_index])
    err = p[2] - target_z
    loss[0] = err * err


def test_no_overhead_when_disabled(test, device):
    """Differentiable arrays are None when requires_grad=False."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=False)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        test.assertIsNone(contacts.rigid_contact_diff_distance)
        test.assertIsNone(contacts.rigid_contact_diff_normal)
        test.assertIsNone(contacts.rigid_contact_diff_point0_world)
        test.assertIsNone(contacts.rigid_contact_diff_point1_world)


def test_arrays_allocated_when_enabled(test, device):
    """Differentiable arrays are allocated when requires_grad=True."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        test.assertIsNotNone(contacts.rigid_contact_diff_distance)
        test.assertIsNotNone(contacts.rigid_contact_diff_normal)
        test.assertIsNotNone(contacts.rigid_contact_diff_point0_world)
        test.assertIsNotNone(contacts.rigid_contact_diff_point1_world)
        test.assertTrue(contacts.rigid_contact_diff_distance.requires_grad)


def test_hydroelastic_stiffness_gradient_through_kh(test, device):
    """Hydroelastic stiffness replay differentiates through per-shape kh."""
    with wp.ScopedDevice(device):
        contacts = newton.Contacts(
            rigid_contact_max=4,
            soft_contact_max=0,
            requires_grad=True,
            device=device,
            per_contact_shape_properties=True,
        )
        contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(np.array([0, -1, -1, -1], dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.array([1, -1, -1, -1], dtype=np.int32))

        kh = wp.array([10.0, 30.0], dtype=wp.float32, requires_grad=True, device=device)
        # k_eff = 10 * 30 / (10 + 30) = 7.5. Frozen stiffness 3.75 gives
        # geometry factor 0.5, so d stiffness / d kh = 0.5 * d k_eff / d kh.
        contacts.rigid_contact_stiffness.assign(np.array([3.75, 0.0, 0.0, 0.0], dtype=np.float32))

        with wp.Tape() as tape:
            launch_differentiable_hydroelastic_stiffness(contacts, kh, device=device)

        seed = wp.array([1.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=device)
        tape.backward(grads={contacts.rigid_contact_stiffness: seed})

        test.assertAlmostEqual(contacts.rigid_contact_stiffness.numpy()[0], 3.75, places=5)
        grad_kh = tape.gradients.get(kh)
        test.assertIsNotNone(grad_kh, "kh gradient should be recorded on tape")
        np.testing.assert_allclose(grad_kh.numpy(), np.array([0.28125, 0.03125], dtype=np.float32), rtol=1e-5)


def test_hydroelastic_kh_gradient_through_solver_step(test, device):
    """Gradients flow from a body trajectory loss to hydroelastic shape kh."""
    with wp.ScopedDevice(device):
        box_half = 0.1
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            sdf_max_resolution=32,
            sdf_narrow_band_range=(-0.04, 0.04),
            is_hydroelastic=True,
            gap=0.01,
            kh=2.0e5,
            mu=0.0,
        )

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, box_half), wp.quat_identity()),
            hx=box_half,
            hy=box_half,
            hz=box_half,
            cfg=shape_cfg,
        )
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * box_half - 0.015), wp.quat_identity()))
        builder.add_shape_box(body=body, hx=box_half, hy=box_half, hz=box_half, cfg=shape_cfg)
        model = builder.finalize(device=device, requires_grad=True)

        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                buffer_fraction=1.0,
                buffer_mult_iso=3,
                buffer_mult_contact=2,
            ),
            requires_grad=True,
        )
        contacts = pipeline.contacts()
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)

        with wp.Tape() as tape:
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 30.0)
            wp.launch(body_height_loss_kernel, dim=1, inputs=[state_1.body_q, body, 0.21], outputs=[loss])

        tape.backward(loss)
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0, "Expected hydroelastic contacts")
        grad_kh = tape.gradients.get(model.shape_material_kh)
        test.assertIsNotNone(grad_kh, "shape_material_kh gradient should be recorded on tape")
        grad_np = grad_kh.numpy()
        test.assertGreater(float(np.max(np.abs(grad_np))), 1.0e-12, f"Expected non-zero kh gradient, got {grad_np}")


def test_sphere_on_plane_distance(test, device):
    """Sphere penetrating ground plane produces correct differentiable distance."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        sphere_height = 0.3
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_height)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()

        pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0, "Expected at least one contact")

        diff_dist = contacts.rigid_contact_diff_distance.numpy()[:count]
        test.assertTrue(
            np.any(diff_dist < 0.0),
            f"Expected negative (penetrating) distance, got {diff_dist}",
        )


def test_gradient_flow_through_body_q(test, device):
    """Verify gradients flow from diff distance through body_q via wp.Tape."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state(requires_grad=True)

        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        tape.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )

        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q, "body_q gradient should be recorded on tape")

        grad_np = grad_q.numpy()
        test.assertFalse(
            np.allclose(grad_np, 0.0),
            "body_q gradient should be non-zero for penetrating sphere",
        )


def test_gradient_direction(test, device):
    """Moving the sphere upward should increase (make less negative) the contact distance."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state(requires_grad=True)

        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        tape.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )

        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q)

        grad_np = grad_q.numpy()
        # wp.transform stores (px, py, pz, qw, qx, qy, qz)
        dz = grad_np[0, 2]  # body 0, z-translation component
        test.assertGreater(
            dz,
            0.0,
            f"Expected positive z-gradient (moving up increases distance), got dz={dz}",
        )


def test_collide_outside_tape(test, device):
    """collide() works correctly outside a tape (no gradients, no crash)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()

        pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0)
        diff_dist = contacts.rigid_contact_diff_distance.numpy()[:count]
        test.assertTrue(np.any(diff_dist < 0.0))


def test_two_body_contact(test, device):
    """Two dynamic bodies in contact both receive non-zero gradients."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        builder.add_shape_box(body=body_a, hx=0.5, hy=0.5, hz=0.5)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)))
        builder.add_shape_box(body=body_b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state(requires_grad=True)

        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0, "Expected contacts between two overlapping boxes")

        tape.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )

        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q)
        grad_np = grad_q.numpy()

        grad_a = grad_np[0]
        grad_b = grad_np[1]
        test.assertFalse(np.allclose(grad_a, 0.0), f"Body A gradient should be non-zero, got {grad_a}")
        test.assertFalse(np.allclose(grad_b, 0.0), f"Body B gradient should be non-zero, got {grad_b}")


def test_world_points_correctness(test, device):
    """Differentiable world-space points and distance are geometrically consistent."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        sphere_height = 0.3
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_height)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0)

        p0 = contacts.rigid_contact_diff_point0_world.numpy()[:count]
        p1 = contacts.rigid_contact_diff_point1_world.numpy()[:count]
        normals = contacts.rigid_contact_diff_normal.numpy()[:count]
        distances = contacts.rigid_contact_diff_distance.numpy()[:count]
        margins0 = contacts.rigid_contact_margin0.numpy()[:count]
        margins1 = contacts.rigid_contact_margin1.numpy()[:count]

        for i in range(count):
            # Verify distance identity: d = dot(n, p1 - p0) - thickness
            gap = np.dot(normals[i], p1[i] - p0[i])
            thickness = margins0[i] + margins1[i]
            expected_d = gap - thickness
            test.assertAlmostEqual(
                float(distances[i]),
                float(expected_d),
                places=4,
                msg=f"Contact {i}: distance {distances[i]} != dot(n, p1-p0) - thickness = {expected_d}",
            )

            # Normal should be approximately unit length
            n_len = np.linalg.norm(normals[i])
            test.assertAlmostEqual(
                n_len,
                1.0,
                places=3,
                msg=f"Contact {i}: normal length {n_len} != 1.0",
            )


def test_finite_difference_distance_gradient(test, device):
    """Tape gradient of distance w.r.t. z-translation matches finite differences."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        h0 = 0.3
        r = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, h0)))
        builder.add_shape_sphere(body=body, radius=r)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        # Analytical gradient via tape
        state = model.state(requires_grad=True)
        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0)
        grad_seed = wp.zeros(contacts.rigid_contact_max, dtype=float, device=device)
        grad_seed_np = grad_seed.numpy()
        grad_seed_np[:count] = 1.0
        grad_seed = wp.array(grad_seed_np, dtype=float, device=device)

        tape.backward(grads={contacts.rigid_contact_diff_distance: grad_seed})
        grad_q = tape.gradients.get(state.body_q)
        analytic_dz = grad_q.numpy()[0, 2]

        # Finite difference: perturb z by eps
        eps = 1e-4
        dist_vals = []
        for sign in [-1.0, 1.0]:
            state_fd = model.state()
            q_np = state_fd.body_q.numpy()
            q_np[0, 2] += sign * eps
            state_fd.body_q = wp.array(q_np, dtype=wp.transform, device=device)
            pipeline.collide(state_fd, contacts)
            c = contacts.rigid_contact_count.numpy()[0]
            d = contacts.rigid_contact_diff_distance.numpy()[:c].sum() if c > 0 else 0.0
            dist_vals.append(d)

        fd_dz = (dist_vals[1] - dist_vals[0]) / (2.0 * eps)

        test.assertAlmostEqual(
            analytic_dz,
            fd_dz,
            places=2,
            msg=f"Analytic dz={analytic_dz:.6f} vs FD dz={fd_dz:.6f}",
        )


def test_repeated_collide_independent_gradients(test, device):
    """Calling collide() twice in separate tapes gives independent gradients."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3)))
        builder.add_shape_sphere(body=body, radius=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        # First tape
        state1 = model.state(requires_grad=True)
        with wp.Tape() as tape1:
            pipeline.collide(state1, contacts)
        tape1.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )
        grad1 = tape1.gradients.get(state1.body_q).numpy().copy()

        # Second tape with same state values
        state2 = model.state(requires_grad=True)
        with wp.Tape() as tape2:
            pipeline.collide(state2, contacts)
        tape2.backward(
            grads={
                contacts.rigid_contact_diff_distance: wp.ones(contacts.rigid_contact_max, dtype=float, device=device)
            }
        )
        grad2 = tape2.gradients.get(state2.body_q).numpy().copy()

        np.testing.assert_allclose(
            grad1,
            grad2,
            atol=1e-6,
            err_msg="Repeated collide() should produce identical gradients",
        )


def test_finite_difference_two_body_gradient(test, device):
    """Tape gradients match central finite differences for two overlapping boxes across all translation DOFs."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        builder.add_shape_box(body=body_a, hx=0.5, hy=0.5, hz=0.5)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)))
        builder.add_shape_box(body=body_b, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device=device, requires_grad=True)

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()

        # Analytical gradient via tape
        state = model.state(requires_grad=True)
        with wp.Tape() as tape:
            pipeline.collide(state, contacts)

        count = contacts.rigid_contact_count.numpy()[0]
        test.assertGreater(count, 0, "Expected contacts between two overlapping boxes")

        grad_seed = wp.zeros(contacts.rigid_contact_max, dtype=float, device=device)
        grad_seed_np = grad_seed.numpy()
        grad_seed_np[:count] = 1.0
        grad_seed = wp.array(grad_seed_np, dtype=float, device=device)

        tape.backward(grads={contacts.rigid_contact_diff_distance: grad_seed})
        grad_q = tape.gradients.get(state.body_q)
        test.assertIsNotNone(grad_q)
        analytic_grad = grad_q.numpy()

        eps = 1e-4
        for body_idx in range(2):
            for axis in range(3):
                dist_vals = []
                for sign in [-1.0, 1.0]:
                    state_fd = model.state()
                    q_np = state_fd.body_q.numpy()
                    q_np[body_idx, axis] += sign * eps
                    state_fd.body_q = wp.array(q_np, dtype=wp.transform, device=device)
                    pipeline.collide(state_fd, contacts)
                    c = contacts.rigid_contact_count.numpy()[0]
                    d = contacts.rigid_contact_diff_distance.numpy()[:c].sum() if c > 0 else 0.0
                    dist_vals.append(d)

                fd_grad = (dist_vals[1] - dist_vals[0]) / (2.0 * eps)
                analytic_val = float(analytic_grad[body_idx, axis])

                test.assertAlmostEqual(
                    analytic_val,
                    fd_grad,
                    places=2,
                    msg=f"Body {body_idx} axis {axis}: analytic={analytic_val:.6f} vs FD={fd_grad:.6f}",
                )


@wp.kernel
def _body_position_loss_kernel(
    body_q: wp.array[wp.transform],
    target: wp.vec3,
    loss: wp.array[float],
):
    pos = wp.transform_get_translation(body_q[0])
    delta = pos - target
    loss[0] = wp.dot(delta, delta)


def test_multistep_gradient_flow(test, device):
    """Multi-step tape gradient of position loss w.r.t. initial z matches finite differences."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=-9.81)
        sphere_height = 2.0
        sphere_radius = 0.5
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_height)))
        builder.add_shape_sphere(body=body, radius=sphere_radius)
        builder.add_ground_plane()
        model = builder.finalize(device=device, requires_grad=True)
        model.soft_contact_ke = 1000.0
        model.soft_contact_kd = 10.0
        model.soft_contact_kf = 100.0
        model.soft_contact_mu = 0.5

        solver = newton.solvers.SolverSemiImplicit(model)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            soft_contact_margin=10.0,
            requires_grad=True,
        )

        sim_substeps = 4
        sim_dt = 1.0 / 60.0 / float(sim_substeps)
        target = wp.vec3(0.0, 0.0, 5.0)

        # --- Analytical gradient via tape ---
        control = model.control()
        states = [model.state(requires_grad=True) for _ in range(sim_substeps + 1)]
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        with wp.Tape() as tape:
            for t in range(sim_substeps):
                states[t].clear_forces()
                contacts = pipeline.contacts()
                pipeline.collide(states[t], contacts)
                solver.step(states[t], states[t + 1], control, contacts, sim_dt)

            wp.launch(
                _body_position_loss_kernel,
                dim=1,
                inputs=[states[-1].body_q, target],
                outputs=[loss],
                device=device,
            )

        tape.backward(loss)

        grad_q0 = tape.gradients.get(states[0].body_q)
        test.assertIsNotNone(grad_q0, "Initial body_q gradient should exist after multi-step backward")
        analytic_dz = float(grad_q0.numpy()[0, 2])

        # --- Finite-difference reference ---
        eps = 1e-4
        loss_vals = []
        for sign in [-1.0, 1.0]:
            q_np = states[0].body_q.numpy()
            q_np[0, 2] = sphere_height + sign * eps
            model_fd = builder.finalize(device=device, requires_grad=False)
            model_fd.soft_contact_ke = 1000.0
            model_fd.soft_contact_kd = 10.0
            model_fd.soft_contact_kf = 100.0
            model_fd.soft_contact_mu = 0.5
            solver_fd = newton.solvers.SolverSemiImplicit(model_fd)
            pipeline_fd = newton.CollisionPipeline(
                model_fd,
                broad_phase="explicit",
                soft_contact_margin=10.0,
                requires_grad=False,
            )
            # Override initial body_q with perturbed value
            fd_states_0 = model_fd.state()
            fd_q = fd_states_0.body_q.numpy()
            fd_q[0, 2] = sphere_height + sign * eps
            fd_states_0.body_q = wp.array(fd_q, dtype=wp.transform, device=device)

            fd_control = model_fd.control()
            fd_states = [fd_states_0] + [model_fd.state() for _ in range(sim_substeps)]
            fd_loss = wp.zeros(1, dtype=float, device=device)
            for t in range(sim_substeps):
                fd_states[t].clear_forces()
                fd_contacts = pipeline_fd.contacts()
                pipeline_fd.collide(fd_states[t], fd_contacts)
                solver_fd.step(fd_states[t], fd_states[t + 1], fd_control, fd_contacts, sim_dt)
            wp.launch(
                _body_position_loss_kernel,
                dim=1,
                inputs=[fd_states[-1].body_q, target],
                outputs=[fd_loss],
                device=device,
            )
            loss_vals.append(fd_loss.numpy()[0])

        fd_dz = (loss_vals[1] - loss_vals[0]) / (2.0 * eps)

        # Verify sign: target above sphere, so moving up reduces loss
        test.assertLess(
            analytic_dz,
            0.0,
            f"Expected negative z-gradient (moving up reduces loss toward target above), got dz={analytic_dz}",
        )

        # Verify magnitude matches finite differences
        test.assertAlmostEqual(
            analytic_dz,
            fd_dz,
            places=1,
            msg=f"Multi-step analytic dz={analytic_dz:.6f} vs FD dz={fd_dz:.6f}",
        )


class TestDifferentiableContacts(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestDifferentiableContacts, "test_no_overhead_when_disabled", test_no_overhead_when_disabled, devices=devices
)
add_function_test(
    TestDifferentiableContacts,
    "test_arrays_allocated_when_enabled",
    test_arrays_allocated_when_enabled,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts,
    "test_hydroelastic_stiffness_gradient_through_kh",
    test_hydroelastic_stiffness_gradient_through_kh,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts,
    "test_hydroelastic_kh_gradient_through_solver_step",
    test_hydroelastic_kh_gradient_through_solver_step,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts, "test_sphere_on_plane_distance", test_sphere_on_plane_distance, devices=devices
)
add_function_test(
    TestDifferentiableContacts, "test_gradient_flow_through_body_q", test_gradient_flow_through_body_q, devices=devices
)
add_function_test(TestDifferentiableContacts, "test_gradient_direction", test_gradient_direction, devices=devices)
add_function_test(TestDifferentiableContacts, "test_collide_outside_tape", test_collide_outside_tape, devices=devices)
add_function_test(TestDifferentiableContacts, "test_two_body_contact", test_two_body_contact, devices=devices)
add_function_test(
    TestDifferentiableContacts, "test_world_points_correctness", test_world_points_correctness, devices=devices
)
add_function_test(
    TestDifferentiableContacts,
    "test_finite_difference_distance_gradient",
    test_finite_difference_distance_gradient,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts,
    "test_repeated_collide_independent_gradients",
    test_repeated_collide_independent_gradients,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts,
    "test_finite_difference_two_body_gradient",
    test_finite_difference_two_body_gradient,
    devices=devices,
)
add_function_test(
    TestDifferentiableContacts,
    "test_multistep_gradient_flow",
    test_multistep_gradient_flow,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
