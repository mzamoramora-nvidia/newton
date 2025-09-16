import numpy as np
import warp as wp

import newton.viewer
from newton._src.hydroelastic.types import mat43h


@wp.func
def bourke_color_map_wp_func(low: wp.float32, high: wp.float32, v: wp.float32):
    c = wp.vec3(1.0, 1.0, 1.0)

    if v < low:
        v = low
    if v > high:
        v = high
    dv = high - low

    if v < (low + 0.25 * dv):
        c[0] = 0.0
        c[1] = 4.0 * (v - low) / dv
    elif v < (low + 0.5 * dv):
        c[0] = 0.0
        c[2] = 1.0 + 4.0 * (low + 0.25 * dv - v) / dv
    elif v < (low + 0.75 * dv):
        c[0] = 4.0 * (v - low - 0.5 * dv) / dv
        c[2] = 0.0
    else:
        c[1] = 1.0 + 4.0 * (low + 0.75 * dv - v) / dv
        c[2] = 0.0

    return c


@wp.kernel
def get_isosurface_normals_data(
    vertex_counts: wp.array2d(dtype=wp.int32),
    centroids: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    pressure_values: wp.array2d(dtype=wp.float32),
    vertex_offset: wp.vec3,
    # Outputs.
    starts: wp.array(dtype=wp.vec3),
    tips: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    surf_id = wp.tid()
    num_isosurfaces = centroids.shape[0]
    max_polygons_for_rendering = wp.int32(starts.shape[0] / num_isosurfaces)

    num_polygons = centroids.shape[1]
    valid_counter = wp.int32(0)
    min_pressure = wp.float32(0.0)
    max_pressure = wp.float32(0.0)
    # Loop over all polygons to set starts and tips, and to compute min and max pressure.
    for i in range(num_polygons):
        if vertex_counts[surf_id, i] > 0:
            if valid_counter < max_polygons_for_rendering:
                offset = surf_id * max_polygons_for_rendering
                starts[offset + valid_counter] = centroids[surf_id, i] + vertex_offset
                tips[offset + valid_counter] = starts[offset + valid_counter] + 0.01 * normals[surf_id, i]
                valid_counter += 1

            min_pressure = wp.min(min_pressure, pressure_values[surf_id, i])
            max_pressure = wp.max(max_pressure, pressure_values[surf_id, i])

    # Loop over all polygons to set colors, with early exit.
    valid_counter = wp.int32(0)
    for i in range(num_polygons):
        if vertex_counts[surf_id, i] > 0:
            offset = surf_id * max_polygons_for_rendering
            colors[offset + valid_counter] = bourke_color_map_wp_func(
                min_pressure, max_pressure, pressure_values[surf_id, i]
            )
            valid_counter += 1

        if valid_counter > max_polygons_for_rendering:
            break


@wp.kernel
def get_isosurface_edges_data(
    vertex_counts: wp.array2d(dtype=wp.int32),
    vertices: wp.array2d(dtype=wp.vec3),
    centroids: wp.array2d(dtype=wp.vec3),
    pressure_values: wp.array2d(dtype=wp.float32),
    vertex_offset: wp.vec3,
    # Outputs.
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    surf_id = wp.tid()
    num_isosurfaces = centroids.shape[0]
    max_vertices_per_isosurface = wp.int32(starts.shape[0] / num_isosurfaces)

    num_polygons = centroids.shape[1]
    valid_counter = wp.int32(0)
    min_pressure = wp.float32(0.0)
    max_pressure = wp.float32(0.0)
    for i in range(num_polygons):
        offset = surf_id * max_vertices_per_isosurface
        v_count = vertex_counts[surf_id, i]
        if v_count > 0:
            if 2 * v_count + valid_counter < max_vertices_per_isosurface:
                for j in range(v_count):
                    v_idx = 8 * i + j

                    # Edge between centroid and vertex.
                    starts[offset + valid_counter] = centroids[surf_id, i] + vertex_offset
                    ends[offset + valid_counter] = vertices[surf_id, v_idx] + vertex_offset
                    valid_counter += 1

                    # Edge between vertex and vertex.
                    v_idx_next = 8 * i + (j + 1) % v_count
                    starts[offset + valid_counter] = vertices[surf_id, v_idx] + vertex_offset
                    ends[offset + valid_counter] = vertices[surf_id, v_idx_next] + vertex_offset
                    valid_counter += 1

            min_pressure = wp.min(min_pressure, pressure_values[surf_id, i])
            max_pressure = wp.max(max_pressure, pressure_values[surf_id, i])

    # Loop over all polygons to set colors, with early exit.
    valid_counter = wp.int32(0)
    for i in range(num_polygons):
        offset = surf_id * max_vertices_per_isosurface
        v_count = vertex_counts[surf_id, i]
        if v_count > 0:
            if 2 * v_count + valid_counter < max_vertices_per_isosurface:
                # Set same color for all edges of the polygon.
                color = bourke_color_map_wp_func(min_pressure, max_pressure, pressure_values[surf_id, i])
                for _ in range(v_count):
                    colors[offset + valid_counter] = color
                    valid_counter += 1

                    colors[offset + valid_counter] = color
                    valid_counter += 1

            if valid_counter > max_vertices_per_isosurface:
                break


@wp.kernel
def compute_tet_mesh_edges(
    body_q: wp.array(dtype=wp.transform),
    body_id: wp.int32,
    indices: wp.array(dtype=wp.int32),
    default_points: wp.array(dtype=wp.vec3),
    # Outputs.
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    idx = 4 * tid
    element = wp.vec4i(indices[idx], indices[idx + 1], indices[idx + 2], indices[idx + 3])

    v = mat43h()
    for i in range(len(element)):
        v[i] = wp.transform_point(body_q[body_id], default_points[element[i]])

    for i in range(len(element)):
        j = (i + 1) % len(element)
        line_starts[tid * 6 + i] = v[i]
        line_ends[tid * 6 + i] = v[j]


def draw_force_arrows(viewer, id, pos, tips, arrow_width, color_arrow, tips_radii, tips_colors):
    viewer.log_lines(
        name=id,
        starts=pos,
        ends=tips,
        colors=color_arrow,
        width=arrow_width,
    )

    viewer.log_points(
        name=f"{id}_tip",
        points=tips,
        radii=tips_radii,
        colors=tips_colors,
    )


def render_forces(viewer, state_0, contacts, editable_vars):
    force_normal_id = "/isosurface_force_n"
    force_tangential_id = "/isosurface_force_t"
    torque_id = "/isosurface_torque"

    force_normal_exists = False
    if isinstance(viewer, newton.viewer.ViewerGL):
        force_normal_exists = force_normal_id in viewer.lines

    if not force_normal_exists and not editable_vars.render_forces_flag:
        return

    body_q = state_0.body_q.numpy()

    twist_pos = np.zeros((contacts.num_isosurfaces, 3))
    twist_torque = np.zeros((contacts.num_isosurfaces, 3))
    twist_force_n = np.zeros((contacts.num_isosurfaces, 3))
    twist_force_t = np.zeros((contacts.num_isosurfaces, 3))

    if editable_vars.render_forces_flag:
        for i in range(contacts.num_isosurfaces):
            body_a = contacts.isosurface[i].body_a
            twist_pos[i, :] = body_q[body_a, 0:3] + editable_vars.np_vertex_offset
            # force = self.contacts.isosurface[i].force.numpy()[0, :]
            twist_torque[i, :] = contacts.isosurface[i].torque_a_body.numpy()[0, :]
            twist_force_n[i, :] = contacts.isosurface[i].force_n.numpy()[0, :]
            twist_force_t[i, :] = contacts.isosurface[i].force_t.numpy()[0, :]

    twist_pos_wp = wp.array(twist_pos, dtype=wp.vec3)
    force_n_tip_wp = wp.array(twist_pos + twist_force_n * editable_vars.force_scale, dtype=wp.vec3)
    force_t_tip_wp = wp.array(twist_pos + twist_force_t * editable_vars.force_scale, dtype=wp.vec3)
    torque_tip_wp = wp.array(twist_pos + twist_torque * editable_vars.force_scale, dtype=wp.vec3)

    arrow_width = 0.001 * 2
    tips_color = wp.full(shape=twist_pos_wp.shape, value=wp.vec3(0.10, 0.10, 0.90))
    tips_radii = wp.full(shape=twist_pos_wp.shape, value=2 * arrow_width)

    draw_force_arrows(
        viewer,
        force_normal_id,
        twist_pos_wp,
        force_n_tip_wp,
        arrow_width,
        (0.90, 0.10, 0.10),
        tips_radii,
        tips_color,
    )
    draw_force_arrows(
        viewer,
        force_tangential_id,
        twist_pos_wp,
        force_t_tip_wp,
        arrow_width,
        (0.10, 0.90, 0.00),
        tips_radii,
        tips_color,
    )
    draw_force_arrows(
        viewer,
        torque_id,
        twist_pos_wp,
        torque_tip_wp,
        arrow_width,
        (0.10, 0.10, 0.90),
        tips_radii,
        tips_color,
    )


def render_visuals(viewer, state_0, visuals):
    pass


def init_isosurface_data_for_rendering(viewer, contacts, max_polygons_for_rendering=512):
    if hasattr(viewer, "isosurface_data"):
        return

    num_isosurfaces = contacts.isosurface_batch.v_counts.shape[0]
    normals_size = num_isosurfaces * max_polygons_for_rendering
    edges_size = num_isosurfaces * max_polygons_for_rendering * 16

    viewer.isosurface_data = {}
    # Normals
    viewer.isosurface_data["normals"] = {}
    viewer.isosurface_data["normals"]["lines_name"] = "/isosurface_batch_normals"
    viewer.isosurface_data["normals"]["points_name"] = "/isosurface_batch_normals_tips"
    viewer.isosurface_data["normals"]["starts"] = wp.zeros(normals_size, dtype=wp.vec3)
    viewer.isosurface_data["normals"]["tips"] = wp.zeros(normals_size, dtype=wp.vec3)
    viewer.isosurface_data["normals"]["colors"] = wp.zeros(normals_size, dtype=wp.vec3)
    viewer.isosurface_data["normals"]["radii"] = wp.full(normals_size, value=0.00125, dtype=wp.float32)

    # Edges
    viewer.isosurface_data["edges"] = {}
    viewer.isosurface_data["edges"]["lines_name"] = "/isosurface_batch_edges"
    viewer.isosurface_data["edges"]["starts"] = wp.zeros(edges_size, dtype=wp.vec3)
    viewer.isosurface_data["edges"]["ends"] = wp.zeros(edges_size, dtype=wp.vec3)
    viewer.isosurface_data["edges"]["colors"] = wp.zeros(edges_size, dtype=wp.vec3)


def render_isosurfaces_batch(viewer, state_0, contacts, editable_vars):
    # ================================
    # Update drawing data
    update_drawing_data_for_polygon_normals_batch(
        viewer,
        contacts.isosurface_batch,
        editable_vars.np_vertex_offset,
        editable_vars.render_isosurfaces_normals,
    )

    update_drawing_data_for_polygon_edges_batch(
        viewer,
        contacts.isosurface_batch,
        editable_vars.np_vertex_offset,
        editable_vars.render_isosurfaces_edges,
    )

    if editable_vars.render_isosurfaces_normals or editable_vars.render_isosurfaces_edges:
        wp.synchronize()

    # ================================
    with wp.ScopedTimer("draw_polygon_normals", print=False):
        draw_polygon_normals_batch(
            viewer,
            editable_vars.render_isosurfaces_normals,
        )

    with wp.ScopedTimer("draw_polygon_edges", print=False):
        draw_polygon_edges_batch(
            viewer,
            editable_vars.render_isosurfaces_edges,
        )


def check_if_lines_should_be_drawn(viewer, render_flag, lines_name):
    lines_exists = False
    if isinstance(viewer, newton.viewer.ViewerGL):
        lines_exists = lines_name in viewer.lines

    if not lines_exists and not render_flag:
        return False

    return True


def update_drawing_data_for_polygon_normals_batch(
    viewer,
    isosurface_batch,
    np_vertex_offset,
    render_normals,
):
    lines_name = viewer.isosurface_data["normals"]["lines_name"]
    if not check_if_lines_should_be_drawn(viewer, render_normals, lines_name):
        return

    viewer.isosurface_data["normals"]["starts"].zero_()
    viewer.isosurface_data["normals"]["tips"].zero_()
    viewer.isosurface_data["normals"]["colors"].zero_()
    if render_normals:
        vertex_offset = wp.vec3(np_vertex_offset)
        wp.launch(
            get_isosurface_normals_data,
            dim=isosurface_batch.centroids.shape[0],
            inputs=[
                isosurface_batch.v_counts,
                isosurface_batch.centroids,
                isosurface_batch.normals,
                isosurface_batch.centroid_pressure,
                vertex_offset,
            ],
            outputs=[
                viewer.isosurface_data["normals"]["starts"],
                viewer.isosurface_data["normals"]["tips"],
                viewer.isosurface_data["normals"]["colors"],
            ],
        )


def draw_polygon_normals_batch(
    viewer,
    render_normals,
):
    lines_name = viewer.isosurface_data["normals"]["lines_name"]
    points_name = viewer.isosurface_data["normals"]["points_name"]
    if not check_if_lines_should_be_drawn(viewer, render_normals, lines_name):
        return

    viewer.log_lines(
        name=lines_name,
        starts=viewer.isosurface_data["normals"]["starts"],
        ends=viewer.isosurface_data["normals"]["tips"],
        colors=viewer.isosurface_data["normals"]["colors"],
        width=0.0005,
        hidden=not render_normals,
    )

    viewer.log_points(
        name=points_name,
        points=viewer.isosurface_data["normals"]["tips"],
        radii=viewer.isosurface_data["normals"]["radii"],
        colors=viewer.isosurface_data["normals"]["colors"],
        hidden=not render_normals,
    )


def update_drawing_data_for_polygon_edges_batch(
    viewer,
    isosurface_batch,
    np_vertex_offset,
    render_edges,
):
    edges_name = viewer.isosurface_data["edges"]["lines_name"]
    if not check_if_lines_should_be_drawn(viewer, render_edges, edges_name):
        return

    viewer.isosurface_data["edges"]["starts"].zero_()
    viewer.isosurface_data["edges"]["ends"].zero_()
    viewer.isosurface_data["edges"]["colors"].zero_()
    if render_edges:
        vertex_offset = wp.vec3(np_vertex_offset)
        wp.launch(
            get_isosurface_edges_data,
            dim=isosurface_batch.centroids.shape[0],
            inputs=[
                isosurface_batch.v_counts,
                isosurface_batch.vertices,
                isosurface_batch.centroids,
                isosurface_batch.centroid_pressure,
                vertex_offset,
            ],
            outputs=[
                viewer.isosurface_data["edges"]["starts"],
                viewer.isosurface_data["edges"]["ends"],
                viewer.isosurface_data["edges"]["colors"],
            ],
        )


def draw_polygon_edges_batch(
    viewer,
    render_edges,
):
    edges_name = viewer.isosurface_data["edges"]["lines_name"]
    if not check_if_lines_should_be_drawn(viewer, render_edges, edges_name):
        return

    viewer.log_lines(
        name=edges_name,
        starts=viewer.isosurface_data["edges"]["starts"],
        ends=viewer.isosurface_data["edges"]["ends"],
        colors=viewer.isosurface_data["edges"]["colors"],
        width=0.0005,
        # hidden=not render_edges,
    )


def draw_tet_mesh_edges(viewer, state_0, hydro_mesh, mesh_id, color, render_tet_mesh_edges):
    if not hydro_mesh.is_soft:
        return
    edges_name = f"/mesh_{mesh_id}_tet_mesh_edges"
    edges_exists = False
    if isinstance(viewer, newton.viewer.ViewerGL):
        edges_exists = edges_name in viewer.lines

    if not edges_exists and not render_tet_mesh_edges:
        return

    num_tets = hydro_mesh.mesh.elements_count
    line_starts = wp.zeros(num_tets * 6, dtype=wp.vec3)
    line_ends = wp.zeros(num_tets * 6, dtype=wp.vec3)

    if render_tet_mesh_edges:
        wp.launch(
            compute_tet_mesh_edges,
            dim=num_tets,
            inputs=[
                state_0.body_q,
                hydro_mesh.body_id,
                hydro_mesh.mesh.indices,
                hydro_mesh.mesh.default_points,
            ],
            outputs=[
                line_starts,
                line_ends,
            ],
        )

    viewer.log_lines(
        name=edges_name,
        starts=line_starts,
        ends=line_ends,
        colors=wp.full(shape=line_starts.shape, value=wp.vec3(*color)),
        width=0.0005,
    )


def render_tet_meshes(viewer, model, state_0, editable_vars):
    with wp.ScopedTimer("render_tet_meshe_edges", print=False):
        num_meshes = len(model.hydro_mesh)
        for i in range(num_meshes):
            render_tet_mesh_edges = False
            if hasattr(editable_vars, "render_tet_mesh_edges"):
                render_tet_mesh_edges = editable_vars.render_tet_mesh_edges
            color = wp.render.bourke_color_map(0, num_meshes - 1, i)
            draw_tet_mesh_edges(viewer, state_0, model.hydro_mesh[i], i, color, render_tet_mesh_edges)
