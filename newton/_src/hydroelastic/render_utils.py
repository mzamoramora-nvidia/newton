import numpy as np
import warp as wp

import newton.viewer
from newton._src.hydroelastic.types import mat43h


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


def render_isosurfaces(viewer, state_0, contacts, editable_vars):
    with wp.ScopedTimer("draw_polygon_normals", print=False):
        for i in range(len(contacts.isosurface)):
            # max_polygons_for_rendering = contacts.isosurface[i].geom_pairs.shape[0]
            max_polygons_for_rendering = 512
            draw_polygon_normals(
                viewer,
                f"/{contacts.isosurface[i].label}",
                max_polygons_for_rendering,
                contacts.isosurface[i].contact_polygon.vertex_counts.numpy(),
                contacts.isosurface[i].contact_polygon.centroids.numpy(),
                contacts.isosurface[i].contact_polygon.normals.numpy(),
                contacts.isosurface[i].contact_polygon.centroid_pressure.numpy(),
                np_vertex_offset=editable_vars.np_vertex_offset,
                render_normals=editable_vars.render_isosurfaces_normals,
            )

    with wp.ScopedTimer("draw_polygon_edges", print=False):
        for i in range(len(contacts.isosurface)):
            # max_polygons_for_rendering = contacts.isosurface[i].geom_pairs.shape[0]
            max_polygons_for_rendering = 512
            draw_polygon_edges(
                viewer,
                f"/{contacts.isosurface[i].label}",
                max_polygons_for_rendering,
                contacts.isosurface[i].contact_polygon.vertex_counts.numpy(),
                contacts.isosurface[i].contact_polygon.vertices.numpy(),
                contacts.isosurface[i].contact_polygon.centroids.numpy(),
                contacts.isosurface[i].contact_polygon.normals.numpy(),
                contacts.isosurface[i].contact_polygon.centroid_pressure.numpy(),
                np_vertex_offset=editable_vars.np_vertex_offset,
                render_edges=editable_vars.render_isosurfaces_edges,
            )


def draw_polygon_normals(
    viewer,
    isosurface_id,
    max_polygons_for_rendering,
    vertex_counts,
    polygon_centers,
    polygon_normals,
    pressure_values,
    np_vertex_offset,
    render_normals,
):
    lines_name = isosurface_id + "_polygon_normals"
    points_name = isosurface_id + "_normal_tips"

    normals_exists = False
    if isinstance(viewer, newton.viewer.ViewerGL):
        normals_exists = lines_name in viewer.lines

    if not normals_exists and not render_normals:
        return

    valid_centers = np.zeros((max_polygons_for_rendering, 3))
    valid_tips = np.zeros((max_polygons_for_rendering, 3))
    colors = np.zeros((max_polygons_for_rendering, 3))

    num_points = 0
    mask = vertex_counts > 0
    if np.any(mask) and render_normals:
        num_points = min(sum(mask), max_polygons_for_rendering)
        valid_centers[0:num_points, :] = polygon_centers[mask][0:num_points] + np_vertex_offset
        valid_tips[0:num_points, :] = valid_centers[0:num_points, :] + 0.01 * polygon_normals[mask][0:num_points]

        valid_pressure_values = pressure_values[mask]
        max_pressure = np.max(valid_pressure_values)
        min_pressure = np.min(valid_pressure_values)
        for i in range(num_points):
            new_color = np.array(wp.render.bourke_color_map(min_pressure, max_pressure, valid_pressure_values[i]))
            colors[i, :] = new_color

    valid_centers_wp = wp.array(valid_centers, dtype=wp.vec3)
    valid_tips_wp = wp.array(valid_tips, dtype=wp.vec3)
    colors_wp = wp.array(colors, dtype=wp.vec3)
    tips_radii_wp = wp.full(shape=(max_polygons_for_rendering,), value=0.00125)

    viewer.log_lines(
        name=lines_name,
        starts=valid_centers_wp,
        ends=valid_tips_wp,
        colors=colors_wp,
        width=0.0005,
    )

    viewer.log_points(
        name=points_name,
        points=valid_tips_wp,
        radii=tips_radii_wp,
        colors=colors_wp,
    )


def draw_polygon_edges(
    viewer,
    isosurface_id,
    max_polygons_for_rendering,
    vertex_counts,
    polygon_vertices,
    polygon_centers,
    polygon_normals,
    pressure_values,
    np_vertex_offset,
    render_edges,
):
    edges_name = isosurface_id + "_polygon_edges"

    edges_exists = False
    if isinstance(viewer, newton.viewer.ViewerGL):
        edges_exists = edges_name in viewer.lines

    if not edges_exists and not render_edges:
        return

    valid_centers = np.zeros((max_polygons_for_rendering, 3))

    valid_edge_starts = np.zeros(((8 + 7) * max_polygons_for_rendering, 3))
    valid_edge_ends = np.zeros(((8 + 7) * max_polygons_for_rendering, 3))
    colors = np.zeros(((8 + 7) * max_polygons_for_rendering, 3))

    num_points = 0
    mask = vertex_counts > 0

    if np.any(mask) and render_edges:
        num_points = min(sum(mask), max_polygons_for_rendering)
        # TODO: Print warning if sum(mask) > max_tet_pairs_found.
        if sum(mask) > max_polygons_for_rendering:
            print(
                f"Warning: sum(mask) > max_polygons_for_rendering in draw_polygon_edges. {sum(mask)} > {max_polygons_for_rendering}. Consider increasing max_polygons_for_rendering."
            )
        valid_centers[0:num_points, :] = polygon_centers[mask][0:num_points] + np_vertex_offset

        valid_pressure_values = pressure_values[mask]
        max_pressure = np.max(valid_pressure_values)
        min_pressure = np.min(valid_pressure_values)

        valid_indices = np.nonzero(vertex_counts)[0]

        # Compute edges
        for i in range(num_points):
            # Get the polygon index.
            polygon_index = valid_indices[i]
            vertex_count = vertex_counts[polygon_index]
            # Block of vertices for the polygon.
            block_start = 8 * polygon_index
            block_end = block_start + vertex_count
            valid_polygon_vertices = polygon_vertices[block_start:block_end] + np_vertex_offset
            # Add edges for the polygon.
            for j in range(vertex_count):
                # Edge from center to vertex of polygon.
                valid_edge_starts[i * 15 + j, :] = valid_centers[i, :]
                valid_edge_ends[i * 15 + j, :] = valid_polygon_vertices[j, :]

                # Edge from vertex to vertex of polygon.
                valid_edge_starts[i * 15 + 8 + j, :] = valid_polygon_vertices[j, :]
                valid_edge_ends[i * 15 + 8 + j, :] = valid_polygon_vertices[(j + 1) % vertex_count, :]

            # Set same color for all edges in the block.
            new_color = wp.render.bourke_color_map(min_pressure, max_pressure, pressure_values[polygon_index])
            colors[i * 15 : (i + 1) * 15, :] = np.array(new_color)

    viewer.log_lines(
        name=edges_name,
        starts=wp.array(valid_edge_starts, dtype=wp.vec3),
        ends=wp.array(valid_edge_ends, dtype=wp.vec3),
        colors=wp.array(colors, dtype=wp.vec3),
        width=0.0005,
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

    num_tets = hydro_mesh.volume_mesh.elements_count
    line_starts = wp.zeros(num_tets * 6, dtype=wp.vec3)
    line_ends = wp.zeros(num_tets * 6, dtype=wp.vec3)

    if render_tet_mesh_edges:
        wp.launch(
            compute_tet_mesh_edges,
            dim=num_tets,
            inputs=[
                state_0.body_q,
                hydro_mesh.body_id,
                hydro_mesh.volume_mesh.indices,
                hydro_mesh.volume_mesh.default_points,
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
