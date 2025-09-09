import warp as wp

GRADIENT_EPSILON = 1.0e-14


@wp.func
def compute_velocity_at_point(body_q: wp.transform, body_qd: wp.spatial_vector, R: wp.vec3f, twist_convention: int):
    # Special treatment due to convention of Featherstone's implementation.
    # See: https://newton-physics.github.io/newton/conventions.html#conventions
    # When using newton convention:
    # - body_w is the angular velocity in the body frame.
    # - body_v is the linear velocity of the COM in the world frame.
    # The velocity of a point R is computed as:
    # R_dot = body_v + wp.cross(body_w, R - com)
    # When using featherstone convention:
    # - body_w is the angular velocity in the world frame.
    # - body_v is the linear velocity of a hypothetical point on the moving body
    #   that is instantaneously at the world origin, expressed in the world frame.
    #   body_v = v_com - cross(body_w, com), where com is expressed in the world frame.
    # The velocity of the COM is computed as:
    # v_com = body_v + wp.cross(body_w, com)
    # The velocity of a point R is computed as:
    # R_dot = v_com + wp.cross(body_w, R-com)
    #       = body_v + wp.cross(body_w, com)  + wp.cross(body_w, R - com)
    #       = body_v + wp.cross(body_w, R)

    com = wp.transform_get_translation(body_q)
    body_w = wp.spatial_top(body_qd)  # angular velocity in world frame
    body_v = wp.spatial_bottom(body_qd)

    R_dot = body_v
    if twist_convention == 0:  # newton convention
        R_dot += wp.cross(body_w, R - com)
    elif twist_convention == 1:  # featherstone convention
        R_dot += wp.cross(body_w, R)
    elif twist_convention == 2:  # mujoco convention
        body_w_W = wp.transform_vector(body_q, body_w)
        R_dot += wp.cross(body_w_W, R - com)

    return R_dot


@wp.func
def compute_combined_hydroelastic_modulus(h_a: wp.float32, h_b: wp.float32):
    return (h_a * h_b) / (h_a + h_b)


@wp.func
def compute_combined_dissipation(h_a: wp.float32, h_b: wp.float32, d_a: wp.float32, d_b: wp.float32):
    return h_b / (h_a + h_b) * d_a + h_a / (h_a + h_b) * d_b


@wp.func
def compute_combined_friction_coefficient(mu_a: wp.float32, mu_b: wp.float32):
    denominator = mu_a + mu_b
    if denominator == 0.0:
        return 0.0
    return (2.0 * mu_a * mu_b) / denominator


@wp.func
def step5(x: wp.float32):
    x3 = x * x * x
    return x3 * (10.0 + x * (6.0 * x - 15.0))  # 10x^3 - 15x^4 + 6x^5


@wp.func
def compute_stribeck_friction_coefficient(
    Rab_dot_tangential: wp.vec3f, v_stiction_tolerance: wp.float32, mu_static: wp.float32, mu_dynamic: wp.float32
):
    v = wp.length(Rab_dot_tangential) / v_stiction_tolerance
    if v >= 3.0:
        return mu_dynamic
    elif v >= 1.0:
        return mu_static - (mu_static - mu_dynamic) * step5((v - 1.0) / 2.0)
    else:
        return mu_static * step5(v)


@wp.kernel
def compute_wrench(
    geom_pairs_found: wp.array(dtype=wp.vec2i),
    num_polygon_vertices: wp.array(dtype=wp.int32),
    polygon_vertices: wp.array(dtype=wp.vec3f),
    polygon_centroids: wp.array(dtype=wp.vec3f),
    polygon_normals: wp.array(dtype=wp.vec3f),
    cartesian_to_penetration: wp.array(dtype=wp.vec4f),
    body_a: wp.int32,
    body_b: wp.int32,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    gradient_a: wp.array(dtype=wp.vec3f),
    gradient_b: wp.array(dtype=wp.vec3f),
    hydroelastic_modulus_a: wp.float32,
    hydroelastic_modulus_b: wp.float32,
    d_a: wp.float32,  # hunt_crossley_dissipation
    d_b: wp.float32,  # hunt_crossley_dissipation
    mu_static_a: wp.float32,
    mu_static_b: wp.float32,
    mu_dynamic_a: wp.float32,
    mu_dynamic_b: wp.float32,
    quadrature_weights: wp.array(dtype=wp.float32),
    quadrature_coords: wp.array(dtype=wp.vec3f),
    soft_vs_soft: wp.array(dtype=wp.int32),
    twist_convention: int,
    # outputs
    force: wp.array(dtype=wp.vec3f),
    torque_a: wp.array(dtype=wp.vec3f),
    torque_b: wp.array(dtype=wp.vec3f),
    torque_a_body: wp.array(dtype=wp.vec3f),
    torque_b_body: wp.array(dtype=wp.vec3f),
    force_n: wp.array(dtype=wp.vec3f),
    force_t: wp.array(dtype=wp.vec3f),
):
    """Compute wrench forces on the isosurface."""
    tid = wp.tid()

    if num_polygon_vertices[tid] == 0:
        return

    if geom_pairs_found[tid][0] == -1 or geom_pairs_found[tid][1] == -1:
        return

    # Get the tet pair.
    tet_id_a = geom_pairs_found[tid][0]
    tet_id_b = geom_pairs_found[tid][1]

    # Initialize output variables.
    force_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_a_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_b_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_a_i_body = wp.vec3f(0.0, 0.0, 0.0)
    torque_b_i_body = wp.vec3f(0.0, 0.0, 0.0)
    force_n_i = wp.vec3f(0.0, 0.0, 0.0)
    force_t_i = wp.vec3f(0.0, 0.0, 0.0)

    mu_static = compute_combined_friction_coefficient(mu_static_a, mu_static_b)
    mu_dynamic = compute_combined_friction_coefficient(mu_dynamic_a, mu_dynamic_b)
    h = compute_combined_hydroelastic_modulus(hydroelastic_modulus_a, hydroelastic_modulus_b)
    d = compute_combined_dissipation(hydroelastic_modulus_a, hydroelastic_modulus_b, d_a, d_b)
    num_quadrature_points = quadrature_weights.shape[0]

    com_a = wp.vec3f(body_q[body_a][0], body_q[body_a][1], body_q[body_a][2])
    com_b = wp.vec3f(body_q[body_b][0], body_q[body_b][1], body_q[body_b][2])
    normal = polygon_normals[tid]

    # Integrate polygon.
    # For each vertex, we have a triangle.
    for cp_idx in range(num_polygon_vertices[tid]):
        # Get triangle vertices.
        index_b = cp_idx
        index_c = (cp_idx + 1) % num_polygon_vertices[tid]
        vertex_a = polygon_centroids[tid]
        vertex_b = polygon_vertices[8 * tid + index_b]
        vertex_c = polygon_vertices[8 * tid + index_c]

        # Compute triangle normal.
        triangle_normal = wp.cross(vertex_b - vertex_a, vertex_c - vertex_a)
        area = wp.length(triangle_normal) * 0.5
        triangle_normal = wp.normalize(triangle_normal)

        # Skip zero area triangles.
        # Drake uses 1.0e-14, but we use 1.0e-10 to avoid numerical issues.
        if area < 1.0e-10:
            continue

        # Make sure the triangle normal is aligned with the normal of the contact polygon.
        # Polygon normals should be normalized already.
        cos_normals = wp.dot(triangle_normal, normal)
        cos_threshold = wp.cos(30.0 * wp.pi / 180.0)
        if cos_normals < cos_threshold:
            vertex_b = polygon_vertices[8 * tid + index_c]
            vertex_c = polygon_vertices[8 * tid + index_b]
            triangle_normal = -triangle_normal

            cos_normals = wp.dot(triangle_normal, normal)
            if cos_normals < cos_threshold:
                # If normals are not aligned after the previous step, it is likely due the triangle being too small.
                # wp.printf("may day: normals are not aligned: %f, area: %e, tid: %d\n", cos_normals, area, tid)
                continue

        # Integrate triangle.
        for i in range(num_quadrature_points):
            # R = (vertex_a + vertex_b + vertex_c) * (1.0 / 3.0)
            R = (
                vertex_a * quadrature_coords[i][0]
                + vertex_b * quadrature_coords[i][1]
                + vertex_c * quadrature_coords[i][2]
            )

            # Compute iso_pressure.
            homogeneous_position = wp.vec4(R.x, R.y, R.z, 1.0)
            penetration_extent_a = wp.dot(cartesian_to_penetration[tid], homogeneous_position)
            pressure_a = hydroelastic_modulus_a * penetration_extent_a

            Ra_dot = compute_velocity_at_point(body_q[body_a], body_qd[body_a], R, twist_convention)
            Rb_dot = compute_velocity_at_point(body_q[body_b], body_qd[body_b], R, twist_convention)

            Rab_dot = Ra_dot - Rb_dot
            Rab_dot_normal = wp.dot(Rab_dot, normal) * normal
            Rab_dot_tangential = Rab_dot - Rab_dot_normal

            # Compute pressure gradient.
            # The gradient should be positve pointing into the body.
            # Some filtering is already done when creating the isosurface.
            # TODO: Should the gradient be computed at the quadrature point R? Is this a reasonable approximation?
            field_gradient_a_W = wp.transform_vector(body_q[body_a], gradient_a[tet_id_a])
            cos_theta_a = wp.dot(wp.normalize(field_gradient_a_W), normal)
            if cos_theta_a < 0.0:
                wp.printf(
                    "gradient a is negative: %f, tet_id_a: %d\n",
                    cos_theta_a,
                    tet_id_a,
                )

            g_a = wp.abs(wp.dot(field_gradient_a_W, normal))
            if g_a < GRADIENT_EPSILON:
                # wp.printf("gradient is too small: %f, %f\n", g_a, g_b)
                continue

            g = g_a

            if soft_vs_soft[0] >= 1:
                field_gradient_b_W = wp.transform_vector(body_q[body_b], gradient_b[tet_id_b])
                cos_theta_b = wp.dot(-wp.normalize(field_gradient_b_W), normal)
                if cos_theta_b < 0.0:
                    wp.printf(
                        "gradient b is negative: %f, tet_id_b: %d\n",
                        cos_theta_b,
                        tet_id_b,
                    )

                g_b = wp.abs(wp.dot(-field_gradient_b_W, normal))
                if g_b < GRADIENT_EPSILON:
                    # wp.printf("gradient is too small: %f, %f\n", g_a, g_b)
                    continue

                # From drake implementation:
                # // The expression below is mathematically equivalent to g =
                # // gN*gM/(gN+gM) but it has the advantage of also being valid if
                # // one of the gradients is infinity.
                g = 1.0 / (1.0 / g_a + 1.0 / g_b)

            # Compute dissipation.
            dissipation = d * (g / h) * wp.dot(Rab_dot, normal)

            # Compute total pressure. (This should be positive)
            hc = 1.0 - dissipation
            # TODO: Add a flag to enable printing of warnings.
            # if hc < 0.0:
            #     wp.printf("Hunt & Crossley  term is negative: %f, tid: %d\n", hc, tid)
            # if pressure_a < 0.0:
            #     wp.printf("pressure is negative: %f, tid: %d\n", pressure_a, tid)

            # pR = pressure_a * hc
            pR = wp.max(0.0, pressure_a * wp.max(0.0, hc))

            # Compute normal traction
            T_N = pR * normal

            # Compute friction traction
            T_F = wp.vec3f(0.0, 0.0, 0.0)  # - mu_static * pR * Rab_dot_tangential
            # if wp.length(Rab_dot_tangential) > 1.0e-9:
            #     T_F = -mu_dynamic * pR * wp.normalize(Rab_dot_tangential)

            # # Regularized friction
            # slip_regularizer = 1.0e-12
            # T_F = -mu_dynamic * pR * (1.0 / wp.sqrt(wp.length(Rab_dot_tangential) + slip_regularizer)) * Rab_dot_tangential

            # # # Stribeck friction model.
            # TODO: Make this a parameter.
            v_stiction_tolerance = 1.0e-4
            mu_stribeck = compute_stribeck_friction_coefficient(
                Rab_dot_tangential, v_stiction_tolerance, mu_static, mu_dynamic
            )
            # # Uncomment to monitor the regime in which the friction is computed. (Stick or slip)
            # if thread_id == 0 and i == 0:
            #     wp.printf("mu_stribeck, mu_static, mu_dynamic: %f, %f, %f\n", mu_stribeck, mu_static, mu_dynamic)
            T_F = -mu_stribeck * pR * wp.normalize(Rab_dot_tangential)

            # Compute total traction
            T_R = T_N + T_F

            # Compute force
            force_i += T_R * area * quadrature_weights[i]

            force_n_i += T_N * area * quadrature_weights[i]
            force_t_i += T_F * area * quadrature_weights[i]

            # Compute torques
            # Not sure if shoud be {R cross T_R} or {(R-com) cross T_R}
            # It seems that for featherstone, the torque is computed as {R cross T_R} as it uses
            # the torques in world frame.
            torque_a_i += wp.cross(R, T_R) * area * quadrature_weights[i]
            torque_a_i_body += wp.cross(R - com_a, T_R) * area * quadrature_weights[i]

            torque_b_i += wp.cross(R, T_R) * area * quadrature_weights[i]
            torque_b_i_body += wp.cross(R - com_b, T_R) * area * quadrature_weights[i]

    wp.atomic_add(force, 0, force_i)
    wp.atomic_add(torque_a, 0, torque_a_i)
    wp.atomic_add(torque_b, 0, torque_b_i)
    wp.atomic_add(torque_a_body, 0, torque_a_i_body)
    wp.atomic_add(torque_b_body, 0, torque_b_i_body)
    wp.atomic_add(force_n, 0, force_n_i)
    wp.atomic_add(force_t, 0, force_t_i)


@wp.kernel
def add_wrench_to_body_f(
    body_a: wp.int32,
    body_b: wp.int32,
    force: wp.array(dtype=wp.vec3f),
    torque_a: wp.array(dtype=wp.vec3f),
    torque_b: wp.array(dtype=wp.vec3f),
    twist_convention: int,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    if twist_convention == 0:
        body_f[body_a] += wp.spatial_vector(torque_a[0], force[0])
        body_f[body_b] -= wp.spatial_vector(torque_b[0], force[0])
    elif twist_convention == 1:
        body_f[body_a] -= wp.spatial_vector(torque_a[0], force[0])
        body_f[body_b] += wp.spatial_vector(torque_b[0], force[0])
    elif twist_convention == 2:
        # For mujoco, forces should be applied in the world frame.
        # See https://github.com/google-deepmind/mujoco/issues/691
        # https://github.com/google-deepmind/mujoco/discussions/2350#discussioncomment-11819398
        # and https://github.com/newton-physics/newton/pull/213
        body_f[body_a] += wp.spatial_vector(torque_a[0], force[0])
        body_f[body_b] -= wp.spatial_vector(torque_b[0], force[0])


def launch_compute_wrench(
    body_q,
    body_qd,
    mesh_a,
    mesh_b,
    isosurface,
    twist_convention,
):
    # This kernel loops over contact polygons and internally loops over triangles and quadrature points.
    wp.launch(
        compute_wrench,
        dim=isosurface.geom_pairs_found.shape[0],
        inputs=[
            isosurface.geom_pairs_found,
            isosurface.contact_polygon.vertex_counts,
            isosurface.contact_polygon.vertices,
            isosurface.contact_polygon.centroids,
            isosurface.contact_polygon.normals,
            isosurface.contact_polygon.cartesian_to_penetration,
            isosurface.body_a,
            isosurface.body_b,
            body_q,
            body_qd,
            mesh_a.volume_mesh.field_gradient,
            mesh_b.volume_mesh.field_gradient,
            mesh_a.hydroelastic_modulus,
            mesh_b.hydroelastic_modulus,
            mesh_a.hunt_crossley_dissipation,
            mesh_b.hunt_crossley_dissipation,
            mesh_a.mu_static,
            mesh_b.mu_static,
            mesh_a.mu_dynamic,
            mesh_b.mu_dynamic,
            isosurface.quadrature_weights,
            isosurface.quadrature_coords,
            isosurface.sotf_vs_soft_wp,
            twist_convention,
        ],
        outputs=[
            isosurface.force,
            isosurface.torque_a,
            isosurface.torque_b,
            isosurface.torque_a_body,
            isosurface.torque_b_body,
            isosurface.force_n,
            isosurface.force_t,
        ],
    )


def compute_isosurface_wrenches(
    isosurface,
    body_q,
    body_qd,
    hydroelastic_mesh_a,
    hydroelastic_mesh_b,
    twist_convention,
):
    """Compute wrench forces on the isosurface."""
    # Should we reset the arrays here?
    isosurface.force.zero_()
    isosurface.torque_a.zero_()
    isosurface.torque_b.zero_()
    isosurface.torque_a_body.zero_()
    isosurface.torque_b_body.zero_()
    isosurface.force_n.zero_()
    isosurface.force_t.zero_()

    # Compute resulting wrench.
    launch_compute_wrench(body_q, body_qd, hydroelastic_mesh_a, hydroelastic_mesh_b, isosurface, twist_convention)


def launch_add_wrench_to_body_f(
    body_a,
    body_b,
    force,
    torque_a,
    torque_b,
    twist_convention,
    body_f,
):
    wp.launch(
        add_wrench_to_body_f,
        dim=1,
        inputs=[body_a, body_b, force, torque_a, torque_b, twist_convention],
        outputs=[body_f],
    )


def compute_contact_forces(solver, state, contacts, twist_convention=0):
    if contacts.use_hydroelastic_inside_solver:
        return
    with wp.ScopedTimer("Computation of contact forces", print=False):
        # Integrate over isosurface to compute forces and torques.
        twist_convention_wp = wp.int32(twist_convention)
        if twist_convention == 0:  # newton convention
            for i in range(contacts.num_isosurfaces):
                compute_isosurface_wrenches(
                    contacts.isosurface[i],
                    state.body_q,
                    state.body_qd,
                    solver.model.hydro_mesh[contacts.isosurface[i].body_a],
                    solver.model.hydro_mesh[contacts.isosurface[i].body_b],
                    twist_convention_wp,
                )

                launch_add_wrench_to_body_f(
                    body_a=contacts.isosurface[i].body_a,
                    body_b=contacts.isosurface[i].body_b,
                    force=contacts.isosurface[i].force,
                    torque_a=contacts.isosurface[i].torque_a_body,
                    torque_b=contacts.isosurface[i].torque_b_body,
                    twist_convention=twist_convention_wp,
                    body_f=state.body_f,
                )
        elif twist_convention == 1:  # featherstone convention
            main_stream = wp.get_stream()
            init_event = main_stream.record_event()

            for isosurface in contacts.isosurface:
                isosurface.stream.wait_event(init_event)
                with wp.ScopedStream(isosurface.stream):
                    compute_isosurface_wrenches(
                        isosurface,
                        state.body_q,
                        solver.body_v_s,
                        solver.model.hydro_mesh[isosurface.body_a],
                        solver.model.hydro_mesh[isosurface.body_b],
                        twist_convention_wp,
                    )
                isosurface.stream.record_event(isosurface.sync_event)

            for isosurface in contacts.isosurface:
                main_stream.wait_event(isosurface.sync_event)

            # TODO: How can we make sure that things are synchronized correctly?
            for isosurface in contacts.isosurface:
                launch_add_wrench_to_body_f(
                    body_a=isosurface.body_a,
                    body_b=isosurface.body_b,
                    force=isosurface.force,
                    torque_a=isosurface.torque_a,
                    torque_b=isosurface.torque_b,
                    twist_convention=twist_convention_wp,
                    body_f=state.body_f,
                )
        elif twist_convention == 2:  # mujoco convention
            for i in range(contacts.num_isosurfaces):
                compute_isosurface_wrenches(
                    contacts.isosurface[i],
                    state.body_q,
                    state.body_qd,
                    solver.model.hydro_mesh[contacts.isosurface[i].body_a],
                    solver.model.hydro_mesh[contacts.isosurface[i].body_b],
                    twist_convention_wp,
                )

                launch_add_wrench_to_body_f(
                    body_a=contacts.isosurface[i].body_a,
                    body_b=contacts.isosurface[i].body_b,
                    force=contacts.isosurface[i].force,
                    torque_a=contacts.isosurface[i].torque_a_body,
                    torque_b=contacts.isosurface[i].torque_b_body,
                    twist_convention=twist_convention_wp,
                    body_f=state.body_f,
                )
