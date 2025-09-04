import numpy as np
import warp as wp
import warp.render


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
    if not editable_vars.render_forces_flag:
        return

    body_q = state_0.body_q.numpy()

    twist_pos = np.zeros((contacts.num_isosurfaces, 3))
    twist_torque = np.zeros((contacts.num_isosurfaces, 3))
    twist_force_n = np.zeros((contacts.num_isosurfaces, 3))
    twist_force_t = np.zeros((contacts.num_isosurfaces, 3))

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
        f"/isosurface_force_n_{i}",
        twist_pos_wp,
        force_n_tip_wp,
        arrow_width,
        (0.90, 0.10, 0.10),
        tips_radii,
        tips_color,
    )
    draw_force_arrows(
        viewer,
        f"/isosurface_force_t_{i}",
        twist_pos_wp,
        force_t_tip_wp,
        arrow_width,
        (0.10, 0.90, 0.00),
        tips_radii,
        tips_color,
    )
    draw_force_arrows(
        viewer,
        f"/isosurface_torque_{i}",
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
    if not editable_vars.render_isosurfaces_flag:
        return

    # # TODO: Fix drawing of normals.
    # # TODO: Draw normals with colors indicating the pressure.
    # with wp.ScopedTimer("draw_polygon_normals", print=False):
    #     for i in range(len(contacts.isosurface)):
    #         max_normals_found = contacts.isosurface[i].geom_pairs.shape[0]
    #         pressure_values = None
    #         draw_polygon_normals(
    #             viewer,
    #             f"/{contacts.isosurface[i].label}",
    #             max_normals_found,
    #             contacts.isosurface[i].contact_polygon.vertex_counts.numpy(),
    #             contacts.isosurface[i].contact_polygon.centroids.numpy(),
    #             contacts.isosurface[i].contact_polygon.normals.numpy(),
    #             pressure_values,
    #             np_vertex_offset=editable_vars.np_vertex_offset,
    #         )


def draw_polygon_normals(
    viewer,
    isosurface_id,
    max_tet_pairs_found,
    vertex_counts,
    polygon_centers,
    polygon_normals,
    pressure_values,
    np_vertex_offset,
):
    final_points = np.zeros((2 * max_tet_pairs_found, 3))
    final_edges = np.zeros((max_tet_pairs_found, 2), dtype=np.int32)
    colors = np.zeros((max_tet_pairs_found, 3))

    num_points = 0
    mask = vertex_counts > 0
    if np.any(mask):
        valid_centers = polygon_centers[mask]
        valid_normals = polygon_normals[mask]
        num_points = min(len(valid_centers), max_tet_pairs_found)

        final_points[0:num_points] = valid_centers[0:num_points] + np_vertex_offset
        final_points[max_tet_pairs_found : max_tet_pairs_found + num_points] = (
            final_points[0:num_points] + 0.01 * valid_normals[0:num_points]
        )

        # [0, num_points, 1, num_points+1, 2, num_points+2, 3, num_points+3, ...]
        final_edges[0:num_points, 0] = np.arange(0, num_points)
        final_edges[0:num_points, 1] = final_edges[0:num_points, 0] + max_tet_pairs_found

        # valid_pressure_values = pressure_values[mask]
        # max_pressure = np.max(valid_pressure_values)
        # min_pressure = np.min(valid_pressure_values)
        # for i in range(num_points):
        #     new_color = np.array(wp.render.bourke_color_map(min_pressure, max_pressure, valid_pressure_values[i]))
        #     colors[i, :] = new_color

    viewer.log_lines(
        name=isosurface_id + "_polygon_normals",
        vertices=final_points,
        indices=final_edges.flatten(),
        color=(0.35, 0.55, 0.9),
        radius=0.0005,
        visible=True,
    )

    viewer.log_points(
        name=isosurface_id + "_normal_tips",
        points=final_points[max_tet_pairs_found : 2 * max_tet_pairs_found],
        radius=0.00125,
        colors=colors,
        visible=True,
        as_spheres=False,
    )
