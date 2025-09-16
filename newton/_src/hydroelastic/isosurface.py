# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""
Hydroelastic Contact Isosurface Computation

This module implements the computation of equal pressure surfaces (isosurfaces)
for hydroelastic contact between tetrahedral meshes. The implementation follows
Drake's approach for computing contact surfaces between compliant bodies.

Key components:
- Geometric utilities for bounding box and distance computations
- Tetrahedron operations for signed volume and matrix construction
- Plane-polygon intersection algorithms
- Polygon clipping and trimming operations
- Main kernel for computing pressure surface triangles
"""

import warp as wp

from newton._src.hydroelastic.types import mat43h
from newton._src.hydroelastic.wrenches import (
    compute_wrench_fun_simple,
    launch_batch_add_wrench_to_body_f,
)

MAX_POLYGON_VERTICES = 8

# Numerical thresholds used throughout the hydroelastic contact computation
# Some values match those used in Drake's mesh_intersection.cc implementation

# Edge intersection tolerances
EDGE_PARALLEL_THRESHOLD = 1e-7  # Tolerance for detecting parallel edges
INTERSECTION_PARAMETER_MAX = 1.0  # Maximum allowed intersection parameter

# Matrix and geometric tolerances
MATRIX_SINGULARITY_THRESHOLD = 1e-9  # 1e-6  # Tolerance for detecting singular matrices
PLANE_NORMAL_THRESHOLD = 1e-12  # Minimum squared length for valid plane normals
PRESSURE_MINIMUM_THRESHOLD = 1e-6  # Minimum pressure value to consider valid

# Vertex deduplication tolerance (squared distance)
# Reference: mesh_intersection.cc line 141 - const double kEpsSquared(1e-14 * 1e-14);
VERTEX_DUPLICATE_THRESHOLD_SQ = 1e-14 * 1e-14  # For nearly duplicate vertex removal

# Surface normal alignment tolerance
TRIANGLE_NORMAL_ALIGNMENT_THRESHOLD = 1e-6  # For checking triangle normal alignment

# Additional thresholds (currently unused but may be useful for future debugging)
MINIMUM_POLYGON_AREA_THRESHOLD = 1e-6  # Minimum area for polygon validity
PRESSURE_COMPARISON_THRESHOLD = 1e-4  # Relative threshold for pressure value comparison

# Face normal alignment threshold for pressure gradient filtering
# Reference: This corresponds to Drake's face normal filtering in IsFaceNormalAlongPressureGradient
NORMAL_ALONG_PRESSURE_GRADIENT_THRESHOLD = 0.4 * wp.pi  # 5.0 * wp.pi / 8.0


VEC3F_BYTE_SIZE_ = wp.types.type_size_in_bytes(wp.vec3f)
INT32_BYTE_SIZE_ = wp.types.type_size_in_bytes(wp.int32)


@wp.func
def get_element_bounding_box(v: mat43h, v_count: wp.int32):
    """
    Compute axis-aligned bounding box of a tetrahedron.

    Args:
        tetrahedron_elements: Indices of the four tetrahedron vertices
        vertex_positions: Array of vertex positions

    Returns:
        Tuple of (min_bounds, max_bounds) defining the bounding box
    """
    min_bounds = v[0]
    max_bounds = v[0]

    # Find min/max bounds across all vertices
    for i in range(1, v_count):
        min_bounds = wp.min(min_bounds, v[i])
        max_bounds = wp.max(max_bounds, v[i])

    return min_bounds, max_bounds


@wp.func
def check_bounding_boxes_overlap(
    min_bounds_a: wp.vec3, max_bounds_a: wp.vec3, min_bounds_b: wp.vec3, max_bounds_b: wp.vec3
) -> bool:
    """
    Check if two axis-aligned bounding boxes overlap.

    Args:
        min_bounds_a, max_bounds_a: Bounds of first bounding box
        min_bounds_b, max_bounds_b: Bounds of second bounding box

    Returns:
        True if bounding boxes overlap, False otherwise
    """
    return (
        min_bounds_a.x <= max_bounds_b.x
        and max_bounds_a.x >= min_bounds_b.x
        and min_bounds_a.y <= max_bounds_b.y
        and max_bounds_a.y >= min_bounds_b.y
        and min_bounds_a.z <= max_bounds_b.z
        and max_bounds_a.z >= min_bounds_b.z
    )


@wp.func
def compute_tetrahedron_signed_volume(
    tetrahedron_elements: wp.vec4i, vertex_positions: wp.array(dtype=wp.vec3)
) -> float:
    """
    Compute signed volume of a tetrahedron to check orientation.

    Args:
        tetrahedron_elements: Indices of the four tetrahedron vertices
        vertex_positions: Array of vertex positions

    Returns:
        Signed volume (positive for correctly oriented tetrahedron)
    """
    v0_idx = tetrahedron_elements[0]
    v1_idx = tetrahedron_elements[1]
    v2_idx = tetrahedron_elements[2]
    v3_idx = tetrahedron_elements[3]

    vertex_0 = vertex_positions[v0_idx]
    vertex_1 = vertex_positions[v1_idx]
    vertex_2 = vertex_positions[v2_idx]
    vertex_3 = vertex_positions[v3_idx]

    return wp.dot(vertex_3 - vertex_0, wp.cross(vertex_1 - vertex_0, vertex_2 - vertex_0))


@wp.func
def compute_signed_distance_to_plane(plane_equation: wp.vec4, point: wp.vec3) -> float:
    """
    Compute signed distance from a point to a plane.

    Args:
        plane_equation: Plane equation in form (nx, ny, nz, d) where n is normal
        point: 3D point position

    Returns:
        Signed distance (positive if point is on normal side of plane)
    """
    plane_normal = wp.vec3(plane_equation.x, plane_equation.y, plane_equation.z)
    # For this to make sense, w needs to be w = - normal * p0, where p0 belongs to the plane
    return wp.dot(plane_normal, point) + plane_equation.w


@wp.func
def compute_edge_plane_intersection(plane_equation: wp.vec4, edge_start: wp.vec3, edge_end: wp.vec3):
    """
    Find intersection point of an edge with a plane.

    Args:
        plane_equation: Plane equation in form (nx, ny, nz, d)
        edge_start: Start point of the edge
        edge_end: End point of the edge

    Returns:
        Intersection point on the edge
    """
    distance_start = compute_signed_distance_to_plane(plane_equation, edge_start)
    distance_end = compute_signed_distance_to_plane(plane_equation, edge_end)

    # TODO: Handle degenerate case where edge is parallel to plane
    # TODO: Make robust.
    # Some artificats might arise depending on this threshold.
    # if wp.abs(distance_start - distance_end) < EDGE_PARALLEL_THRESHOLD:
    #     return edge_start

    # Compute intersection parameter and point
    intersection_parameter = distance_start / (distance_start - distance_end)
    # intersection_parameter = wp.abs(distance_start) / (wp.abs(distance_end-distance_start))
    if wp.isnan(intersection_parameter):
        # wp.printf("Warning: Intersection parameter is NaN. Returning edge_start.\n")
        # wp.util.warn("Warning: Intersection parameter is NaN. Returning edge_start.")
        return edge_start

    if wp.isinf(intersection_parameter):
        # wp.printf("Warning: Intersection parameter is infinite. Returning edge_start.\n")
        # wp.util.warn("Warning: Intersection parameter is infinite. Returning edge_start.")
        return edge_start

    if wp.abs(intersection_parameter) > INTERSECTION_PARAMETER_MAX:
        # wp.printf("Warning: Intersection parameter is out of bounds. Returning edge_start.\n")
        # wp.util.warn("Warning: Intersection parameter is out of bounds. Returning edge_start.")
        return edge_start

    # if wp.abs(intersection_parameter) < 1.0e-12:
    #     wp.printf("Warning: Intersection parameter is too small. Returning edge_start.\n")
    #     return edge_start

    return edge_start + intersection_parameter * (edge_end - edge_start)

    # # This is the same as the code above, but it is more similar to the code in the Drake repo.
    # a = compute_signed_distance_to_plane(plane_equation, edge_start)
    # b = compute_signed_distance_to_plane(plane_equation, edge_end)
    # # Some artificats might arise depending on this threshold.
    # if wp.abs(a - b) < 1e-8:
    #     #wp.printf("Warning: Edge a and b are too close to each other. Returning edge_start.\n")
    #     return edge_start

    # wa = b / (b - a)
    # wb = 1.0 - wa
    # return edge_start * wa + edge_end * wb


@wp.func
def ensure_polygon_is_convex(edge_flags: wp.int32, polygon_vertices: wp.array(dtype=wp.vec3f)):
    """
    Ensure a polygon is convex.
    """
    # This method is intended to be used after building the polygon from the plane-tetrahedron intersection.
    # So, that the polygon has at most 4 edges.
    for i in range(4):
        ib = (i + 1) & 3
        a = (edge_flags >> (4 * i)) & 0x0000000F
        b = (edge_flags >> (4 * ib)) & 0x0000000F
        if (a & b) == 0:
            swap_idx = (ib + 1) & 3

            tmp = polygon_vertices[ib]
            polygon_vertices[ib] = polygon_vertices[swap_idx]
            polygon_vertices[swap_idx] = tmp
            return


@wp.func
def plane_tetrahedron_intersection(
    plane_equation: wp.vec4, tet_vertices: mat43h, polygon_vertices: wp.array(dtype=wp.vec3f)
):
    """
    Build polygon from intersection of plane with tetrahedron.

    This function finds where a plane intersects a tetrahedron and returns
    the resulting polygon vertices.

    Args:
        plane_equation: Plane equation in form (nx, ny, nz, d)
        tetrahedron_elements: Indices of the four tetrahedron vertices
        vertex_positions: Array of vertex positions

    Returns:
        Tuple of (polygon_vertices, vertex_count)
    """
    # TODO: Find a better way to initialize the polygon vertices.
    for i in range(MAX_POLYGON_VERTICES):
        polygon_vertices[i] = wp.vec3f(0.0, 0.0, 0.0)
    vertex_count = 0

    # Determine which vertices are on positive side of the plane
    vertex_signs = wp.vector(False, length=4, dtype=bool)
    for i in range(4):
        vertex_signs[i] = compute_signed_distance_to_plane(plane_equation, tet_vertices[i]) >= 0.0

    # Define tetrahedron edges as pairs of vertex indices
    tetrahedron_edges = wp.matrix(0, shape=(6, 2), dtype=wp.int32)
    tetrahedron_edges[0] = wp.vec2i(0, 1)
    tetrahedron_edges[1] = wp.vec2i(0, 2)
    tetrahedron_edges[2] = wp.vec2i(0, 3)
    tetrahedron_edges[3] = wp.vec2i(1, 2)
    tetrahedron_edges[4] = wp.vec2i(1, 3)
    tetrahedron_edges[5] = wp.vec2i(2, 3)

    # From Tobi: 4 bits per edge. Two bits set per edge. The bits set mark the vertices belonging to that edge.
    edge_flags = wp.int32(0)

    # Find edges that cross the plane and compute intersections
    for i in range(6):
        edge = tetrahedron_edges[i]
        vertex_a_idx = edge.x
        vertex_b_idx = edge.y

        # Check if edge crosses plane (vertices on different sides)
        if vertex_signs[vertex_a_idx] != vertex_signs[vertex_b_idx]:
            vertex_a_position = tet_vertices[vertex_a_idx]
            vertex_b_position = tet_vertices[vertex_b_idx]

            if vertex_count >= MAX_POLYGON_VERTICES:
                wp.printf("Warning: Polygon vertex count exceeds maximum allowed.")

            if vertex_signs[vertex_a_idx] >= 0:
                polygon_vertices[vertex_count] = compute_edge_plane_intersection(
                    plane_equation, vertex_a_position, vertex_b_position
                )
            else:
                polygon_vertices[vertex_count] = compute_edge_plane_intersection(
                    plane_equation, vertex_b_position, vertex_a_position
                )

            edge_flags |= ((1 << edge.x) | (1 << edge.y)) << (4 * vertex_count)
            vertex_count += 1

    if vertex_count > 3:
        ensure_polygon_is_convex(edge_flags, polygon_vertices)

    return vertex_count


@wp.func
def clip_polygon_with_plane(
    clipping_plane: wp.vec4,
    polygon: wp.array(dtype=wp.vec3f),
    vertex_count: int,
):
    """
    Clip polygon with a plane using Sutherland-Hodgman algorithm.

    Args:
        clipping_plane: Plane equation for clipping
        input_polygon: Input polygon vertices
        input_vertex_count: Number of vertices in input polygon

    Returns:
        Tuple of (clipped_polygon, clipped_vertex_count)
    """
    if vertex_count == 0:
        return

    # Temporary storage for clipped polygon
    clipped_polygon = wp.matrix(0.0, shape=(8, 3), dtype=wp.float32)
    clipped_vertex_count = wp.int32(0)

    # Sutherland-Hodgman clipping algorithm
    previous_vertex = polygon[vertex_count - 1]
    previous_inside = compute_signed_distance_to_plane(clipping_plane, previous_vertex) >= 0.0

    for i in range(vertex_count):
        current_vertex = polygon[i]
        current_inside = compute_signed_distance_to_plane(clipping_plane, current_vertex) >= 0.0

        if current_inside:
            if not previous_inside:
                # Entering clipped region: add intersection point
                # Current vertex is inside the clipping plane, but previous vertex is outside.
                # We give priority to the vertex that is inside the clipping plane, if the two vertices
                # are too close to each other, which why passing current_vertex as the first vertex argument to the
                # compute_edge_plane_intersection function.
                intersection = compute_edge_plane_intersection(clipping_plane, current_vertex, previous_vertex)
                if clipped_vertex_count < 8:
                    clipped_polygon[clipped_vertex_count] = intersection
                    clipped_vertex_count += 1

            # Add current vertex (inside clipped region)
            if clipped_vertex_count < 8:
                clipped_polygon[clipped_vertex_count] = current_vertex
                clipped_vertex_count += 1
        elif previous_inside:
            # Exiting clipped region: add intersection point.
            # Current vertex is outside the clipping plane, but previous vertex is inside.
            intersection = compute_edge_plane_intersection(clipping_plane, previous_vertex, current_vertex)
            if clipped_vertex_count < 8:
                clipped_polygon[clipped_vertex_count] = intersection
                clipped_vertex_count += 1

        previous_vertex = current_vertex
        previous_inside = current_inside

    # Copy clipped polygon back to input polygon.
    for i in range(clipped_vertex_count):
        polygon[i] = clipped_polygon[i]

    return clipped_vertex_count


@wp.func
def clip_polygon_with_tetrahedron(
    tet_vertices: mat43h,
    polygon: wp.array(dtype=wp.vec3f),
    vertex_count: int,
):
    """
    Clip polygon with all faces of a tetrahedron.

    This function successively clips the input polygon with each face of
    the tetrahedron, resulting in the portion of the polygon that lies
    inside the tetrahedron.

    Args:
        tetrahedron_elements: Indices of the four tetrahedron vertices
        vertex_positions: Array of vertex positions
        polygon: Input polygon to clip
        vertex_count: Number of vertices in input polygon

    Returns:
        Tuple of (clipped_polygon, clipped_vertex_count)
    """

    # Define tetrahedron faces as triplets of vertex indices
    tetrahedron_faces = wp.matrix(0, shape=(4, 3), dtype=wp.int32)
    tetrahedron_faces[0] = wp.vec3i(0, 1, 2)
    tetrahedron_faces[1] = wp.vec3i(0, 3, 1)
    tetrahedron_faces[2] = wp.vec3i(0, 2, 3)
    tetrahedron_faces[3] = wp.vec3i(1, 3, 2)

    # TODO: Sanity check. Compute centroid of tetrahedron and check that the normals of all the faces are pointing
    # inwards or outwards.

    # Clip polygon with each tetrahedron face
    for face_idx in range(4):
        if vertex_count == 0:
            break

        face = tetrahedron_faces[face_idx]

        # Compute face plane equation
        face_vertex_0 = tet_vertices[face.x]
        face_vertex_1 = tet_vertices[face.y]
        face_vertex_2 = tet_vertices[face.z]

        face_normal = wp.normalize(wp.cross(face_vertex_1 - face_vertex_0, face_vertex_2 - face_vertex_0))
        plane_distance = -wp.dot(face_normal, face_vertex_0)
        face_plane_equation = wp.vec4(face_normal.x, face_normal.y, face_normal.z, plane_distance)

        # Clip polygon with this face
        vertex_count = clip_polygon_with_plane(face_plane_equation, polygon, vertex_count)

    return vertex_count


@wp.func
def compute_polygon_centroid(polygon_vertices: wp.array(dtype=wp.vec3f), vertex_count: int):
    """
    Compute centroid of a polygon.

    Note: This function doesn't have a direct equivalent in Drake's mesh_intersection.cc,
    but centroid computation is used in the mesh building process within the MeshBuilder classes.
    The algorithm uses area-weighted triangle centroids for robust centroid calculation.

    Args:
        polygon_vertices: Polygon vertex positions
        vertex_count: Number of vertices in polygon

    Returns:
        Centroid position as wp.vec3
    """
    # See https://math.stackexchange.com/questions/90463/how-can-i-calculate-the-centroid-of-polygon
    mean_vertex = wp.vec3(0.0, 0.0, 0.0)

    for i in range(vertex_count):
        mean_vertex += polygon_vertices[i]

    mean_vertex = mean_vertex * (1.0 / wp.float32(vertex_count))

    centroid = wp.vec3(0.0, 0.0, 0.0)
    total_area = wp.float32(0.0)
    for i in range(vertex_count):
        vertex_a = mean_vertex
        vertex_b = polygon_vertices[i]
        vertex_c = polygon_vertices[(i + 1) % vertex_count]
        triangle_area = 0.5 * wp.length(wp.cross(vertex_b - vertex_a, vertex_c - vertex_a))
        triangle_centroid = (vertex_a + vertex_b + vertex_c) / 3.0
        centroid += triangle_centroid * triangle_area
        total_area += triangle_area

    centroid = centroid * (1.0 / total_area)

    if wp.isnan(centroid) or wp.isinf(centroid):
        return mean_vertex

    return centroid


@wp.func
def compute_polygon_area(polygon_vertices: wp.array(dtype=wp.vec3f), vertex_count: wp.int32, centroid: wp.vec3):
    """
    Compute area of a polygon.
    """

    area = wp.float32(0.0)
    for i in range(vertex_count):
        vertex_a = centroid
        vertex_b = polygon_vertices[i]
        vertex_c = polygon_vertices[(i + 1) % vertex_count]
        area += wp.length(wp.cross(vertex_b - vertex_a, vertex_c - vertex_a))

    return 0.5 * area


@wp.func
def is_normal_along_pressure_gradient(grad_p_W: wp.vec3, normal: wp.vec3, body_id: wp.int32, tid: wp.int32):
    # Equivalent to IsFaceNormalAlongPressureGradient in mesh_intersection.cc from Drake.
    if wp.length_sq(grad_p_W) < PLANE_NORMAL_THRESHOLD:
        return False

    cos_theta = wp.dot(wp.normalize(grad_p_W), normal)
    cos_threshold = wp.cos(NORMAL_ALONG_PRESSURE_GRADIENT_THRESHOLD)

    aligned = cos_theta >= cos_threshold
    # if not aligned:
    #    wp.printf("Warning: Field gradient not aligned enough with the normal. Body: %d, thread: %d, cosine: %f, threshold: %f\n", body_id, tid, cos_theta, cos_threshold)

    return aligned


@wp.func
def warn_potential_overflow_for_pressure(h: wp.float32, penetration_extent: wp.float32, tid: wp.int32):
    FLT_MAX = 3.4028234663852886e38
    if wp.abs(h) > wp.abs(wp.float32(FLT_MAX) / wp.abs(penetration_extent)):
        wp.printf(
            "Potential overflow: Thread %d: Penetration extent: %f, hydroelastic modulus: %f\n",
            tid,
            penetration_extent,
            h,
        )
        return True

    return False


@wp.func
def warn_degenerate_pressure(pressure: wp.float32, tid: wp.int32):
    if wp.isnan(pressure):
        wp.printf("Warning: Pressure is NaN for pair %d. Pressure: %f\n", tid, pressure)
        return True
    if wp.isinf(pressure):
        wp.printf("Warning: Pressure is infinite for pair %d. Pressure: %f\n", tid, pressure)
        return True

    # if pressure < 0.0:
    #     wp.printf("Warning: Pressure is negative for pair %d. Pressure: %f\n", tid, pressure)
    #     return True

    return False


@wp.func
def warn_degenerate_plane(plane_normal: wp.vec3, tid: wp.int32):
    if wp.isnan(plane_normal):
        wp.printf("Warning: Plane normal is NaN for pair %d. Plane normal: %f\n", tid, plane_normal)
        return True
    if wp.isinf(plane_normal):
        wp.printf("Warning: Plane normal is infinite for pair %d. Plane normal: %f\n", tid, plane_normal)
        return True

    return False


@wp.kernel
def batch_compute_contact_surface_and_wrenches_from_bvh(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    # Inputs from batch of objects/meshes
    points: wp.array(dtype=wp.vec3f, ndim=2),
    elements: wp.array(dtype=wp.int32, ndim=2),
    elements_count: wp.array(dtype=wp.int32),
    element_strides: wp.array(dtype=wp.int32),
    default_tet_transform_inv: wp.array(dtype=wp.mat44, ndim=2),
    p: wp.array(dtype=wp.float32, ndim=2),
    grad_p: wp.array(dtype=wp.vec3f, ndim=2),
    tri_normals: wp.array(dtype=wp.vec3f, ndim=2),
    h_mesh: wp.array(dtype=wp.float32),
    bvh_ids: wp.array(dtype=wp.uint64),
    # Inputs from batch of isosurfaces
    queries_with_mesh_a: wp.array(dtype=wp.bool),
    soft_vs_soft: wp.array(dtype=wp.bool),
    body_a_idx: wp.array(dtype=wp.int32),
    body_b_idx: wp.array(dtype=wp.int32),
    h_combined: wp.array(dtype=wp.float32),  # Combined hydroelastic modulus
    d_combined: wp.array(dtype=wp.float32),  # Combined hunt_crossley_dissipation
    mu_static_combined: wp.array(dtype=wp.float32),
    mu_dynamic_combined: wp.array(dtype=wp.float32),
    quadrature_weights: wp.array(dtype=wp.float32),
    quadrature_coords: wp.array(dtype=wp.vec3f),
    twist_convention: int,
    # outputs
    element_pairs: wp.array(dtype=wp.vec2i, ndim=2),
    cp_vcounts: wp.array(dtype=wp.int32, ndim=2),
    cp_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    cp_centroids: wp.array(dtype=wp.vec3f, ndim=2),
    cp_penetration: wp.array(dtype=wp.vec4f, ndim=2),
    cp_normals: wp.array(dtype=wp.vec3f, ndim=2),
    centroid_pressure: wp.array(dtype=wp.float32, ndim=2),
    force: wp.array(dtype=wp.vec3f),
    torque_a: wp.array(dtype=wp.vec3f),
    torque_b: wp.array(dtype=wp.vec3f),
    torque_a_body: wp.array(dtype=wp.vec3f),
    torque_b_body: wp.array(dtype=wp.vec3f),
    force_n: wp.array(dtype=wp.vec3f),
    force_t: wp.array(dtype=wp.vec3f),
):
    surf_id, tid = wp.tid()
    # wp.printf("surf_id: %d, tid: %d\n", surf_id, tid)

    # Get scalar values for input variables
    body_a = body_a_idx[surf_id]
    body_b = body_b_idx[surf_id]

    elements_count_a = elements_count[body_a]
    elements_count_b = elements_count[body_b]

    query_with_mesh_a = queries_with_mesh_a[surf_id]

    if query_with_mesh_a:
        if tid >= elements_count_a:
            # wp.printf("tid >= elements_count_a: %d, %d\n", tid, elements_count_a)
            return
    else:
        if tid >= elements_count_b:
            # wp.printf("tid >= elements_count_b: %d, %d\n", tid, elements_count_b)
            return

    # Get flag for soft vs soft surface
    soft_vs_soft_surface = soft_vs_soft[surf_id]

    element_a_stride = element_strides[body_a]
    element_b_stride = element_strides[body_b]

    bvh_id = bvh_ids[body_b]
    if not query_with_mesh_a:
        bvh_id = bvh_ids[body_a]

    # Get scalar values for material properties
    h = h_combined[surf_id]
    h_a = h_mesh[body_a]
    h_b = h_mesh[body_b]

    d = d_combined[surf_id]

    mu_static = mu_static_combined[surf_id]
    mu_dynamic = mu_dynamic_combined[surf_id]

    # Get array views for input variables
    # points_a = points[body_a]
    # points_b = points[body_b]

    # elements_a = elements[body_a]
    # elements_b = elements[body_a]

    # default_tet_transform_inv_a = default_tet_transform_inv[body_a]
    # default_tet_transform_inv_b = default_tet_transform_inv[body_b]

    # p_a = p[body_a]
    # p_b = p[body_b]

    # grad_p_a = grad_p[body_a]
    # grad_p_b = grad_p[body_b]

    # tri_normals_b = tri_normals[body_b]

    # # Get array views for output arrays
    # element_pairs = element_pairs_batch[surf_id]
    # cp_vcounts = cp_vcounts_batch[surf_id]
    # cp_vertices = cp_vertices_batch[surf_id]
    # cp_centroids = cp_centroids_batch[surf_id]
    # cp_penetration = cp_penetration_batch[surf_id]
    # cp_normals = cp_normals_batch[surf_id]
    # centroid_pressure = centroid_pressure_batch[surf_id]

    # ==============================================================================================================
    # Initialize variables for the contact surface computation.
    el_a_vpos_W = mat43h(0.0)  # element vertex positions in world frame
    el_b_vpos_W = mat43h(0.0)
    vidx_a = wp.vec4i(0)
    vidx_b = wp.vec4i(0)
    body_q_inv_mat_a = wp.transform_to_matrix(wp.transform_inverse(body_q[body_a]))
    body_q_inv_mat_b = wp.transform_to_matrix(wp.transform_inverse(body_q[body_b]))

    # Initialize variables to accumulate the contact wrenches of multiple element pairs.
    force_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_a_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_b_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_a_i_body = wp.vec3f(0.0, 0.0, 0.0)
    torque_b_i_body = wp.vec3f(0.0, 0.0, 0.0)
    force_n_i = wp.vec3f(0.0, 0.0, 0.0)
    force_t_i = wp.vec3f(0.0, 0.0, 0.0)

    valid_pairs_count = wp.int32(0)

    # Initialize variables to query the BVH.
    # tet or triangle vertex positions in world frame
    tet_vpos_query_W = mat43h(0.0)

    body_id_query = body_a
    body_id_bvh = body_b
    element_stride = element_a_stride
    el_max_pairs = wp.int32(element_pairs.shape[1] / elements_count_a)

    if not query_with_mesh_a:
        body_id_query = body_b
        body_id_bvh = body_a
        element_stride = element_b_stride
        el_max_pairs = wp.int32(element_pairs.shape[1] / elements_count_b)

    for i in range(element_stride):
        if query_with_mesh_a:
            vidx_query = elements[body_a, element_stride * tid + i]
            p_query_W = wp.transform_point(body_q[body_id_query], points[body_a, vidx_query])
        else:
            vidx_query = elements[body_b][element_stride * tid + i]
            p_query_W = wp.transform_point(body_q[body_id_query], points[body_b, vidx_query])

        p_query_rel = p_query_W - wp.transform_get_translation(body_q[body_id_bvh])
        tet_vpos_query_W[i] = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[body_id_bvh]), p_query_rel)

    min_bounds_query, max_bounds_query = get_element_bounding_box(tet_vpos_query_W, element_stride)
    lower = wp.vec3(min_bounds_query.x, min_bounds_query.y, min_bounds_query.z)
    upper = wp.vec3(max_bounds_query.x, max_bounds_query.y, max_bounds_query.z)

    # Setup BVH query with the AABB of the querying mesh.
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    bvh_element = wp.int32(0)
    bvh_counter = wp.int32(0)

    # Initialize loop variables.
    pair_idx = wp.int32(0)

    # Loop over pairs of elements.
    while wp.bvh_query_next(query, bvh_element):
        pair_idx = tid * el_max_pairs + wp.min(bvh_counter, el_max_pairs - 1)

        if query_with_mesh_a:
            el_a_idx = tid
            el_b_idx = bvh_element
        else:
            el_a_idx = bvh_element
            el_b_idx = tid

        # if bvh_counter >= el_max_pairs:
        # wp.printf(
        #     "compute_contact_surface_and_wrenches: Query is overflowing: tid, el_a_idx, el_b_idx, pair_idx, el_max_pairs, bvh_counter: %d, %d, %d, %d, %d, %d\n",
        #     tid,
        #     el_a_idx,
        #     el_b_idx,
        #     pair_idx,
        #     el_max_pairs,
        #     bvh_counter,
        # )

        bvh_counter += 1
        element_pairs[surf_id, pair_idx] = wp.vec2i(el_a_idx, el_b_idx)

        for i in range(element_a_stride):
            vidx_a[i] = elements[body_a, element_a_stride * el_a_idx + i]
            el_a_vpos_W[i] = wp.transform_point(body_q[body_a], points[body_a, vidx_a[i]])

        for i in range(element_b_stride):
            vidx_b[i] = elements[body_b, element_b_stride * el_b_idx + i]
            el_b_vpos_W[i] = wp.transform_point(body_q[body_b], points[body_b, vidx_b[i]])

        # Early exit: Check bounding box overlap
        min_bounds_a, max_bounds_a = get_element_bounding_box(el_a_vpos_W, element_a_stride)
        min_bounds_b, max_bounds_b = get_element_bounding_box(el_b_vpos_W, element_b_stride)

        if not check_bounding_boxes_overlap(min_bounds_a, max_bounds_a, min_bounds_b, max_bounds_b):
            # wp.printf("missed: tid, tet_idx_a, tet_idx_b: %d, %d, %d\n", tid, el_a_idx, el_b_idx)
            continue

        # =================================================================================
        # Compute contact surface elements.
        if soft_vs_soft_surface:
            # wp.printf("element_pairs: %d, %d\n", element_pairs[surf_id, pair_idx][0], element_pairs[surf_id, pair_idx][1])
            is_valid = compute_soft_soft_fun_batch(
                surf_id,
                pair_idx,
                element_pairs,
                body_q,
                body_a,
                body_b,
                el_a_vpos_W,
                el_b_vpos_W,
                vidx_a,
                vidx_b,
                default_tet_transform_inv,
                p,
                grad_p,
                h_a,
                h_b,
                body_q_inv_mat_a,
                body_q_inv_mat_b,
                cp_vcounts,
                cp_vertices,
                cp_centroids,
                cp_penetration,
                cp_normals,
                centroid_pressure,
            )
            # wp.printf("v_counts: %d\n", cp_vcounts[surf_id, pair_idx])

            if not is_valid:
                continue
        else:
            is_valid = compute_soft_hard_fun_batch(
                surf_id,
                pair_idx,
                element_pairs,
                body_q,
                body_a,
                body_b,
                el_a_vpos_W,
                el_b_vpos_W,
                vidx_a,
                default_tet_transform_inv,
                p,
                grad_p,
                h_a,
                body_q_inv_mat_a,
                tri_normals,
                cp_vcounts,
                cp_vertices,
                cp_centroids,
                cp_penetration,
                cp_normals,
                centroid_pressure,
            )

            if not is_valid:
                continue

        valid_pairs_count += 1
        # ================================================================
        # Compute contact wrench
        (
            force_i,
            torque_a_i,
            torque_b_i,
            torque_a_i_body,
            torque_b_i_body,
            force_n_i,
            force_t_i,
        ) = compute_wrench_fun_simple(
            surf_id,
            pair_idx,
            element_pairs,
            body_q,
            body_qd,
            body_a,
            body_b,
            cp_vcounts,
            cp_vertices,
            cp_centroids,
            cp_penetration,
            cp_normals,
            grad_p,
            h,
            d,
            mu_static,
            mu_dynamic,
            h_a,
            quadrature_weights,
            quadrature_coords,
            soft_vs_soft_surface,
            twist_convention,
            force_i,
            torque_a_i,
            torque_b_i,
            torque_a_i_body,
            torque_b_i_body,
            force_n_i,
            force_t_i,
        )

    if valid_pairs_count > 0:
        # wp.printf("surf_id: %d, force_i: %f, %f, %f\n", surf_id, force_i[0], force_i[1], force_i[2])
        wp.atomic_add(force, surf_id, force_i)
        wp.atomic_add(torque_a, surf_id, torque_a_i)
        wp.atomic_add(torque_b, surf_id, torque_b_i)
        wp.atomic_add(torque_a_body, surf_id, torque_a_i_body)
        wp.atomic_add(torque_b_body, surf_id, torque_b_i_body)
        wp.atomic_add(force_n, surf_id, force_n_i)
        wp.atomic_add(force_t, surf_id, force_t_i)


@wp.kernel
def batch_compute_contact_surface_and_wrenches_from_pairs(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    # Inputs from batch of objects/meshes
    points: wp.array(dtype=wp.vec3f, ndim=2),
    elements: wp.array(dtype=wp.int32, ndim=2),
    elements_count: wp.array(dtype=wp.int32),
    element_strides: wp.array(dtype=wp.int32),
    default_tet_transform_inv: wp.array(dtype=wp.mat44, ndim=2),
    p: wp.array(dtype=wp.float32, ndim=2),
    grad_p: wp.array(dtype=wp.vec3f, ndim=2),
    tri_normals: wp.array(dtype=wp.vec3f, ndim=2),
    h_mesh: wp.array(dtype=wp.float32),
    # Inputs from batch of isosurfaces
    queries_with_mesh_a: wp.array(dtype=wp.bool),
    soft_vs_soft: wp.array(dtype=wp.bool),
    body_a_idx: wp.array(dtype=wp.int32),
    body_b_idx: wp.array(dtype=wp.int32),
    h_combined: wp.array(dtype=wp.float32),  # Combined hydroelastic modulus
    d_combined: wp.array(dtype=wp.float32),  # Combined hunt_crossley_dissipation
    mu_static_combined: wp.array(dtype=wp.float32),
    mu_dynamic_combined: wp.array(dtype=wp.float32),
    quadrature_weights: wp.array(dtype=wp.float32),
    quadrature_coords: wp.array(dtype=wp.vec3f),
    twist_convention: int,
    # outputs
    element_pairs: wp.array(dtype=wp.vec2i, ndim=2),
    cp_vcounts: wp.array(dtype=wp.int32, ndim=2),
    cp_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    cp_centroids: wp.array(dtype=wp.vec3f, ndim=2),
    cp_penetration: wp.array(dtype=wp.vec4f, ndim=2),
    cp_normals: wp.array(dtype=wp.vec3f, ndim=2),
    centroid_pressure: wp.array(dtype=wp.float32, ndim=2),
    force: wp.array(dtype=wp.vec3f),
    torque_a: wp.array(dtype=wp.vec3f),
    torque_b: wp.array(dtype=wp.vec3f),
    torque_a_body: wp.array(dtype=wp.vec3f),
    torque_b_body: wp.array(dtype=wp.vec3f),
    force_n: wp.array(dtype=wp.vec3f),
    force_t: wp.array(dtype=wp.vec3f),
):
    surf_id, tid = wp.tid()
    # wp.printf("surf_id: %d, tid: %d\n", surf_id, tid)

    # Get scalar values for input variables
    body_a = body_a_idx[surf_id]
    body_b = body_b_idx[surf_id]

    elements_count_a = elements_count[body_a]
    elements_count_b = elements_count[body_b]

    query_with_mesh_a = queries_with_mesh_a[surf_id]

    if query_with_mesh_a:
        if tid >= elements_count_a:
            # wp.printf("tid >= elements_count_a: %d, %d\n", tid, elements_count_a)
            return
    else:
        if tid >= elements_count_b:
            # wp.printf("tid >= elements_count_b: %d, %d\n", tid, elements_count_b)
            return

    # Get flag for soft vs soft surface
    soft_vs_soft_surface = soft_vs_soft[surf_id]

    element_a_stride = element_strides[body_a]
    element_b_stride = element_strides[body_b]

    # Get scalar values for material properties
    h = h_combined[surf_id]
    h_a = h_mesh[body_a]
    h_b = h_mesh[body_b]

    d = d_combined[surf_id]

    mu_static = mu_static_combined[surf_id]
    mu_dynamic = mu_dynamic_combined[surf_id]

    # ==============================================================================================================
    # Initialize variables for the contact surface computation.
    el_a_vpos_W = mat43h(0.0)  # element vertex positions in world frame
    el_b_vpos_W = mat43h(0.0)
    vidx_a = wp.vec4i(0)
    vidx_b = wp.vec4i(0)
    body_q_inv_mat_a = wp.transform_to_matrix(wp.transform_inverse(body_q[body_a]))
    body_q_inv_mat_b = wp.transform_to_matrix(wp.transform_inverse(body_q[body_b]))

    # Initialize variables to accumulate the contact wrenches of multiple element pairs.
    force_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_a_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_b_i = wp.vec3f(0.0, 0.0, 0.0)
    torque_a_i_body = wp.vec3f(0.0, 0.0, 0.0)
    torque_b_i_body = wp.vec3f(0.0, 0.0, 0.0)
    force_n_i = wp.vec3f(0.0, 0.0, 0.0)
    force_t_i = wp.vec3f(0.0, 0.0, 0.0)

    valid_pairs_count = wp.int32(0)

    el_max_pairs = wp.int32(element_pairs.shape[1] / elements_count_a)
    if not query_with_mesh_a:
        el_max_pairs = wp.int32(element_pairs.shape[1] / elements_count_b)

    # Initialize loop variables.
    counter = wp.int32(0)
    pair_idx = wp.int32(0)

    # Loop over pairs of elements.
    while counter < el_max_pairs:
        pair_idx = tid * el_max_pairs + wp.min(counter, el_max_pairs - 1)

        if element_pairs[surf_id, pair_idx][0] == -1 or element_pairs[surf_id, pair_idx][1] == -1:
            break

        el_a_idx = element_pairs[surf_id, pair_idx][0]
        el_b_idx = element_pairs[surf_id, pair_idx][1]

        for i in range(element_a_stride):
            vidx_a[i] = elements[body_a, element_a_stride * el_a_idx + i]
            el_a_vpos_W[i] = wp.transform_point(body_q[body_a], points[body_a, vidx_a[i]])

        for i in range(element_b_stride):
            vidx_b[i] = elements[body_b, element_b_stride * el_b_idx + i]
            el_b_vpos_W[i] = wp.transform_point(body_q[body_b], points[body_b, vidx_b[i]])

        # Early exit: Check bounding box overlap
        min_bounds_a, max_bounds_a = get_element_bounding_box(el_a_vpos_W, element_a_stride)
        min_bounds_b, max_bounds_b = get_element_bounding_box(el_b_vpos_W, element_b_stride)

        if not check_bounding_boxes_overlap(min_bounds_a, max_bounds_a, min_bounds_b, max_bounds_b):
            # wp.printf("missed: tid, tet_idx_a, tet_idx_b: %d, %d, %d\n", tid, el_a_idx, el_b_idx)
            counter += 1
            continue

        # =================================================================================
        # Compute contact surface elements.
        if soft_vs_soft_surface:
            # wp.printf("element_pairs: %d, %d\n", element_pairs[surf_id, pair_idx][0], element_pairs[surf_id, pair_idx][1])
            is_valid = compute_soft_soft_fun_batch(
                surf_id,
                pair_idx,
                element_pairs,
                body_q,
                body_a,
                body_b,
                el_a_vpos_W,
                el_b_vpos_W,
                vidx_a,
                vidx_b,
                default_tet_transform_inv,
                p,
                grad_p,
                h_a,
                h_b,
                body_q_inv_mat_a,
                body_q_inv_mat_b,
                cp_vcounts,
                cp_vertices,
                cp_centroids,
                cp_penetration,
                cp_normals,
                centroid_pressure,
            )
            # wp.printf("v_counts: %d\n", cp_vcounts[surf_id, pair_idx])

            if not is_valid:
                counter += 1
                continue
        else:
            is_valid = compute_soft_hard_fun_batch(
                surf_id,
                pair_idx,
                element_pairs,
                body_q,
                body_a,
                body_b,
                el_a_vpos_W,
                el_b_vpos_W,
                vidx_a,
                default_tet_transform_inv,
                p,
                grad_p,
                h_a,
                body_q_inv_mat_a,
                tri_normals,
                cp_vcounts,
                cp_vertices,
                cp_centroids,
                cp_penetration,
                cp_normals,
                centroid_pressure,
            )

            if not is_valid:
                counter += 1
                continue

        valid_pairs_count += 1
        # ================================================================
        # Compute contact wrench
        (
            force_i,
            torque_a_i,
            torque_b_i,
            torque_a_i_body,
            torque_b_i_body,
            force_n_i,
            force_t_i,
        ) = compute_wrench_fun_simple(
            surf_id,
            pair_idx,
            element_pairs,
            body_q,
            body_qd,
            body_a,
            body_b,
            cp_vcounts,
            cp_vertices,
            cp_centroids,
            cp_penetration,
            cp_normals,
            grad_p,
            h,
            d,
            mu_static,
            mu_dynamic,
            h_a,
            quadrature_weights,
            quadrature_coords,
            soft_vs_soft_surface,
            twist_convention,
            force_i,
            torque_a_i,
            torque_b_i,
            torque_a_i_body,
            torque_b_i_body,
            force_n_i,
            force_t_i,
        )

        counter += 1

    if valid_pairs_count > 0:
        # wp.printf("surf_id: %d, force_i: %f, %f, %f\n", surf_id, force_i[0], force_i[1], force_i[2])
        wp.atomic_add(force, surf_id, force_i)
        wp.atomic_add(torque_a, surf_id, torque_a_i)
        wp.atomic_add(torque_b, surf_id, torque_b_i)
        wp.atomic_add(torque_a_body, surf_id, torque_a_i_body)
        wp.atomic_add(torque_b_body, surf_id, torque_b_i_body)
        wp.atomic_add(force_n, surf_id, force_n_i)
        wp.atomic_add(force_t, surf_id, force_t_i)


@wp.func
def compute_soft_soft_fun_batch(
    surf_id: wp.int32,
    pair_idx: wp.int32,
    tet_pairs_found: wp.array(dtype=wp.vec2i, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    body_a: wp.int32,
    body_b: wp.int32,
    tet_vpos_a_W: mat43h,
    tet_vpos_b_W: mat43h,
    vidx_a: wp.vec4i,
    vidx_b: wp.vec4i,
    default_tet_transform_inv: wp.array(dtype=wp.mat44, ndim=2),
    p: wp.array(dtype=wp.float32, ndim=2),
    grad_p: wp.array(dtype=wp.vec3f, ndim=2),
    h_a: wp.float32,  # hydroelastic modulus
    h_b: wp.float32,
    body_q_inv_mat_a: wp.mat44,
    body_q_inv_mat_b: wp.mat44,
    # outputs
    cp_vcounts: wp.array(dtype=wp.int32, ndim=2),
    cp_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    cp_centroids: wp.array(dtype=wp.vec3f, ndim=2),
    cp_penetration: wp.array(dtype=wp.vec4f, ndim=2),
    cp_normals: wp.array(dtype=wp.vec3f, ndim=2),
    centroid_pressure: wp.array(dtype=wp.float32, ndim=2),
):
    if tet_pairs_found[surf_id, pair_idx][0] == -1 or tet_pairs_found[surf_id, pair_idx][1] == -1:
        return False

    tet_idx_a = tet_pairs_found[surf_id, pair_idx][0]
    tet_idx_b = tet_pairs_found[surf_id, pair_idx][1]

    # Build pressure field vectors.
    # TODO: Consider building the field vectors when loading the meshes.
    p_a_vec = wp.vec4(p[body_a, vidx_a[0]], p[body_a, vidx_a[1]], p[body_a, vidx_a[2]], p[body_a, vidx_a[3]])
    p_b_vec = wp.vec4(p[body_b, vidx_b[0]], p[body_b, vidx_b[1]], p[body_b, vidx_b[2]], p[body_b, vidx_b[3]])

    inv_mat_a = default_tet_transform_inv[body_a, tet_idx_a] * body_q_inv_mat_a
    inv_mat_b = default_tet_transform_inv[body_b, tet_idx_b] * body_q_inv_mat_b

    # Combine field contributions weighted by material compliance
    homogeneous_to_penetration_map_a = p_a_vec * inv_mat_a
    homogeneous_to_penetration_map_b = p_b_vec * inv_mat_b

    # Compute plane equation.
    max_modulus = wp.max(h_a, h_b)
    scale_factor_a = max_modulus / h_b
    scale_factor_b = max_modulus / h_a
    weighted_field_a = homogeneous_to_penetration_map_a * scale_factor_a
    weighted_field_b = homogeneous_to_penetration_map_b * scale_factor_b

    plane_equation = weighted_field_a - weighted_field_b
    normal_magnitude = wp.length(wp.vec3(plane_equation.x, plane_equation.y, plane_equation.z))

    # Normalizing plane equation.
    plane_equation = (1.0 / normal_magnitude) * plane_equation
    plane_normal = wp.vec3(plane_equation.x, plane_equation.y, plane_equation.z)
    warn_degenerate_plane(plane_normal, pair_idx)

    # Transform the pressure gradient to the world frame.
    grad_p_a_W = wp.transform_vector(body_q[body_a], grad_p[body_a, tet_idx_a])
    grad_p_b_W = -wp.transform_vector(body_q[body_b], grad_p[body_b, tet_idx_b])

    # Check if the plane normal is aligned with the pressure gradient.
    if not is_normal_along_pressure_gradient(grad_p_a_W, plane_normal, body_a, pair_idx):
        return False

    if not is_normal_along_pressure_gradient(grad_p_b_W, plane_normal, body_b, pair_idx):
        return False

    # Create local array for the polygon vertices.
    cp_v = wp.zeros(shape=(MAX_POLYGON_VERTICES,), dtype=wp.vec3f)
    ## Array pointing to block of memory for the vertices of the current contact polygon.
    # ptr = cp_vertices.ptr + wp.uint64(MAX_POLYGON_VERTICES * pair_idx * VEC3F_BYTE_SIZE_)
    # cp_v = wp.array(ptr=ptr, shape=(MAX_POLYGON_VERTICES,), dtype=wp.vec3f)

    # Build initial polygon from plane-tetrahedron intersection
    # Clip polygon with first tetrahedron
    cp_vcounts[surf_id, pair_idx] = plane_tetrahedron_intersection(plane_equation, tet_vpos_a_W, cp_v)

    # Return if the polygon is empty.
    if cp_vcounts[surf_id, pair_idx] == 0:
        return False

    # Clip polygon with second tetrahedron
    cp_vcounts[surf_id, pair_idx] = clip_polygon_with_tetrahedron(tet_vpos_b_W, cp_v, cp_vcounts[surf_id, pair_idx])

    # Return if the polygon is empty.
    if cp_vcounts[surf_id, pair_idx] == 0:
        return False

    # Compute centroid of the final polygon.
    centroid = compute_polygon_centroid(cp_v, cp_vcounts[surf_id, pair_idx])

    # Compute pressure at the centroid.
    c_homogeneous = wp.vec4(centroid.x, centroid.y, centroid.z, 1.0)
    penetration_extent_a = wp.dot(homogeneous_to_penetration_map_a, c_homogeneous)
    pressure_a = h_a * penetration_extent_a
    warn_potential_overflow_for_pressure(h_a, penetration_extent_a, pair_idx)
    warn_degenerate_pressure(pressure_a, pair_idx)

    # Store results.
    for i in range(cp_vcounts[surf_id, pair_idx]):
        cp_vertices[surf_id, MAX_POLYGON_VERTICES * pair_idx + i] = cp_v[i]
    cp_centroids[surf_id, pair_idx] = centroid
    cp_normals[surf_id, pair_idx] = plane_normal
    centroid_pressure[surf_id, pair_idx] = pressure_a
    cp_penetration[surf_id, pair_idx] = homogeneous_to_penetration_map_a

    return True


@wp.func
def compute_soft_hard_fun_batch(
    surf_id: wp.int32,
    pair_idx: wp.int32,
    tet_triangle_pairs: wp.array(dtype=wp.vec2i, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    body_a: wp.int32,
    body_b: wp.int32,
    tet_vpos_a_W: mat43h,
    tri_vpos_b_W: mat43h,
    vidx_a: wp.vec4i,
    default_tet_transform_inv: wp.array(dtype=wp.mat44, ndim=2),
    p: wp.array(dtype=wp.float32, ndim=2),
    grad_p: wp.array(dtype=wp.vec3f, ndim=2),
    h_a: wp.float32,  # hydroelastic modulus
    body_q_inv_mat_a: wp.mat44,
    tri_normals: wp.array(dtype=wp.vec3f, ndim=2),
    # outputs
    cp_vcounts: wp.array(dtype=wp.int32, ndim=2),
    cp_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    cp_centroids: wp.array(dtype=wp.vec3f, ndim=2),
    cp_penetration: wp.array(dtype=wp.vec4f, ndim=2),
    cp_normals: wp.array(dtype=wp.vec3f, ndim=2),
    centroid_pressure: wp.array(dtype=wp.float32, ndim=2),
):
    if tet_triangle_pairs[surf_id][pair_idx][0] == -1 or tet_triangle_pairs[surf_id][pair_idx][1] == -1:
        return False

    tet_id = tet_triangle_pairs[surf_id][pair_idx][0]
    tri_id = tet_triangle_pairs[surf_id][pair_idx][1]

    # Check face normal alignment if requested
    # Equivalent to lines 332-337 in mesh_intersection.cc:
    # if (filter_face_normal_along_field_gradient) {
    #   if (!this->IsFaceNormalAlongPressureGradient(
    #           volume_field_M, surface_N, X_MN_d, tet_index, tri_index)) {
    #     return;
    #   }
    # }

    # Transform field gradient to world space
    soft_grad_W = wp.transform_vector(body_q[body_a], grad_p[body_a][tet_id])

    # Transform triangle normal to world space
    hard_normal_W = wp.transform_vector(body_q[body_b], tri_normals[body_b][tri_id])

    if not is_normal_along_pressure_gradient(soft_grad_W, hard_normal_W, body_a, pair_idx):
        return False

    # Clip triangle by tetrahedron
    # Equivalent to lines 349-353 in mesh_intersection.cc:
    # const std::vector<Vector3<T>>& polygon_vertices_M =
    #     this->ClipTriangleByTetrahedron(tet_index, vol_mesh_M, tri_index,
    #                                     surface_N, X_MN);
    # if (polygon_vertices_M.size() < 3) return;

    ## TODO: Find a better way to initialize the polygon vertices.
    # ptr = cp_vertices.ptr + wp.uint64(8 * pair_idx * VEC3F_BYTE_SIZE_)
    # polygon_vertices = wp.array(ptr=ptr, shape=(8,), dtype=wp.vec3f)

    polygon_vertices = wp.zeros(shape=(MAX_POLYGON_VERTICES,), dtype=wp.vec3f)
    for i in range(3):
        polygon_vertices[i] = tri_vpos_b_W[i]

    cp_vcounts[surf_id, pair_idx] = 3
    cp_vcounts[surf_id, pair_idx] = clip_polygon_with_tetrahedron(
        tet_vpos_a_W, polygon_vertices, cp_vcounts[surf_id][pair_idx]
    )

    if cp_vcounts[surf_id, pair_idx] < 3:
        cp_vcounts[surf_id, pair_idx] = 0
        return False

    # Compute polygon centroid
    centroid = compute_polygon_centroid(polygon_vertices, cp_vcounts[surf_id, pair_idx])

    # Build pressure field vectors.
    # Equivalent to lines 355-361 in mesh_intersection.cc:
    # Add the vertices to the builder (with corresponding field values) and
    # construct index-based polygon representation.
    # polygon_vertex_indices_.clear();
    # for (const auto& p_MV : polygon_vertices_M) {
    #   polygon_vertex_indices_.push_back(builder_M->AddVertex(
    #       p_MV, volume_field_M.EvaluateCartesian(tet_index, p_MV)));
    # }
    p_a_vec = wp.vec4(p[body_a][vidx_a[0]], p[body_a][vidx_a[1]], p[body_a][vidx_a[2]], p[body_a][vidx_a[3]])
    inv_mat = default_tet_transform_inv[body_a][tet_id] * body_q_inv_mat_a
    homogeneous_to_penetration_map = p_a_vec * inv_mat

    # Compute pressure at centroid
    # Equivalent to lines 369-370 in mesh_intersection.cc:
    # const Vector3<T> grad_e_MN_M = volume_field_M.EvaluateGradient(tet_index);
    # Note: In our implementation, pressure is computed as modulus * penetration
    c_homogeneous = wp.vec4(centroid.x, centroid.y, centroid.z, 1.0)
    penetration_extent_a = wp.dot(homogeneous_to_penetration_map, c_homogeneous)
    pressure_a = h_a * penetration_extent_a
    warn_potential_overflow_for_pressure(h_a, penetration_extent_a, pair_idx)

    # Store results
    for i in range(cp_vcounts[surf_id, pair_idx]):
        cp_vertices[surf_id, MAX_POLYGON_VERTICES * pair_idx + i] = polygon_vertices[i]
    cp_centroids[surf_id, pair_idx] = centroid
    cp_normals[surf_id, pair_idx] = hard_normal_W  # Normal (from hard surface)
    cp_penetration[surf_id, pair_idx] = homogeneous_to_penetration_map
    centroid_pressure[surf_id, pair_idx] = pressure_a

    return True


# @wp.func
# def remove_nearly_duplicate_vertices(
#     polygon_vertices: wp.array(dtype=wp.vec3f),
#     vertex_count: wp.array(dtype=wp.int32)
# ):
#     """
#     Remove nearly duplicate vertices from a polygon.

#     Based on Drake's RemoveNearlyDuplicateVertices implementation.

#     Reference: mesh_intersection.cc lines 122-170
#     Drake function: RemoveNearlyDuplicateVertices(std::vector<Vector3<T>>* polygon)

#     Args:
#         polygon_vertices: Polygon vertices (modified in-place)
#         vertex_count: Number of vertices (modified)
#     """
#     # Equivalent to line 132: if (polygon->size() <= 1) return;
#     if vertex_count[0] <= 1:
#         return

#     # Remove consecutive duplicate vertices
#     # Equivalent to lines 149-159 in mesh_intersection.cc:
#     # Remove consecutive vertices that are duplicated in the linear order.  It
#     # will change "A,B,B,C,C,A" to "A,B,C,A". To close the cyclic order, we
#     # will check the first and the last vertices again near the end of the
#     # function.
#     # auto it = std::unique(polygon->begin(), polygon->end(), near);
#     # polygon->resize(it - polygon->begin());
#     write_index = 0
#     for read_index in range(vertex_count[0]):
#         current_vertex = polygon_vertices[read_index]

#         # Check if current vertex is nearly identical to the last written vertex
#         # Equivalent to the lambda function "near" in lines 134-147:
#         # auto near = [](const Vector3<T>& p, const Vector3<T>& q) -> bool {
#         #   const double kEpsSquared(1e-14 * 1e-14);
#         #   return (convert_to_double(p) - convert_to_double(q)).squaredNorm() < kEpsSquared;
#         # };
#         is_duplicate = False
#         if write_index > 0:
#             last_vertex = polygon_vertices[write_index - 1]
#             distance_sq = wp.length_sq(current_vertex - last_vertex)
#             # Using the same threshold as Drake: kEpsSquared(1e-14 * 1e-14)
#             is_duplicate = distance_sq < VERTEX_DUPLICATE_THRESHOLD_SQ

#         if not is_duplicate:
#             polygon_vertices[write_index] = current_vertex
#             write_index += 1

#     vertex_count[0] = write_index

#     # Check first and last vertices for closure
#     # Equivalent to lines 161-167 in mesh_intersection.cc:
#     # if (polygon->size() >= 3) {
#     #   // Check the first and the last vertices in the sequence. For example, given
#     #   // "A,B,C,A", we want "A,B,C".
#     #   if (near((*polygon)[0], *(polygon->rbegin()))) {
#     #     polygon->pop_back();
#     #   }
#     # }
#     if vertex_count[0] >= 3:
#         first_vertex = polygon_vertices[0]
#         last_vertex = polygon_vertices[vertex_count[0] - 1]
#         distance_sq = wp.length_sq(first_vertex - last_vertex)
#         if distance_sq < VERTEX_DUPLICATE_THRESHOLD_SQ:
#             vertex_count[0] -= 1


def launch_batch_compute_contact_polygons_and_wrenches_from_bvh(
    body_q,
    body_qd,
    objects_batch,
    isosurface_batch,
    twist_convention,
):
    isosurface_batch.element_pairs.fill_(-1)
    isosurface_batch.v_counts.zero_()
    isosurface_batch.force.zero_()
    isosurface_batch.torque_a.zero_()
    isosurface_batch.torque_b.zero_()
    isosurface_batch.torque_a_body.zero_()
    isosurface_batch.torque_b_body.zero_()
    isosurface_batch.force_n.zero_()
    isosurface_batch.force_t.zero_()

    # print("Kernel dims: ", (isosurface_batch.element_pairs.shape[0], objects_batch.max_elements_count))

    wp.launch(
        batch_compute_contact_surface_and_wrenches_from_bvh,
        dim=(isosurface_batch.element_pairs.shape[0], objects_batch.max_elements_count),
        inputs=[
            body_q,
            body_qd,
            objects_batch.default_points,
            objects_batch.indices,
            objects_batch.elements_count,
            objects_batch.elements_stride,
            objects_batch.default_tet_transform_inv,
            objects_batch.field,
            objects_batch.field_gradient,
            objects_batch.normals,
            objects_batch.h,
            objects_batch.bvh_ids,
            isosurface_batch.query_with_mesh_a,
            isosurface_batch.soft_vs_soft,
            isosurface_batch.body_a_idx,
            isosurface_batch.body_b_idx,
            isosurface_batch.h_combined,
            isosurface_batch.d_combined,
            isosurface_batch.mu_static_combined,
            isosurface_batch.mu_dynamic_combined,
            isosurface_batch.quadrature_weights,
            isosurface_batch.quadrature_coords,
            twist_convention,
        ],
        outputs=[
            isosurface_batch.element_pairs,
            isosurface_batch.v_counts,
            isosurface_batch.vertices,
            isosurface_batch.centroids,
            isosurface_batch.cartesian_to_penetration,
            isosurface_batch.normals,
            isosurface_batch.centroid_pressure,
            isosurface_batch.force,
            isosurface_batch.torque_a,
            isosurface_batch.torque_b,
            isosurface_batch.torque_a_body,
            isosurface_batch.torque_b_body,
            isosurface_batch.force_n,
            isosurface_batch.force_t,
        ],
    )


def launch_batch_compute_contact_polygons_and_wrenches_from_pairs(
    body_q,
    body_qd,
    objects_batch,
    isosurface_batch,
    twist_convention,
):
    # isosurface_batch.element_pairs.fill_(-1)
    isosurface_batch.v_counts.zero_()
    isosurface_batch.force.zero_()
    isosurface_batch.torque_a.zero_()
    isosurface_batch.torque_b.zero_()
    isosurface_batch.torque_a_body.zero_()
    isosurface_batch.torque_b_body.zero_()
    isosurface_batch.force_n.zero_()
    isosurface_batch.force_t.zero_()

    # print("Kernel dims: ", (isosurface_batch.element_pairs.shape[0], objects_batch.max_elements_count))

    wp.launch(
        batch_compute_contact_surface_and_wrenches_from_pairs,
        dim=(isosurface_batch.element_pairs.shape[0], objects_batch.max_elements_count),
        inputs=[
            body_q,
            body_qd,
            objects_batch.default_points,
            objects_batch.indices,
            objects_batch.elements_count,
            objects_batch.elements_stride,
            objects_batch.default_tet_transform_inv,
            objects_batch.field,
            objects_batch.field_gradient,
            objects_batch.normals,
            objects_batch.h,
            isosurface_batch.query_with_mesh_a,
            isosurface_batch.soft_vs_soft,
            isosurface_batch.body_a_idx,
            isosurface_batch.body_b_idx,
            isosurface_batch.h_combined,
            isosurface_batch.d_combined,
            isosurface_batch.mu_static_combined,
            isosurface_batch.mu_dynamic_combined,
            isosurface_batch.quadrature_weights,
            isosurface_batch.quadrature_coords,
            twist_convention,
        ],
        outputs=[
            isosurface_batch.element_pairs,
            isosurface_batch.v_counts,
            isosurface_batch.vertices,
            isosurface_batch.centroids,
            isosurface_batch.cartesian_to_penetration,
            isosurface_batch.normals,
            isosurface_batch.centroid_pressure,
            isosurface_batch.force,
            isosurface_batch.torque_a,
            isosurface_batch.torque_b,
            isosurface_batch.torque_a_body,
            isosurface_batch.torque_b_body,
            isosurface_batch.force_n,
            isosurface_batch.force_t,
        ],
    )


def batch_compute_contact_surfaces_and_wrenches(solver, state, contacts, twist_convention, update_contact_pairs=True):
    with wp.ScopedTimer("Computation of contact surfaces", print=False):
        if twist_convention == 0:  # newton convention
            print("Newton convention not supported yet")
        elif twist_convention == 1:  # featherstone convention
            if update_contact_pairs:
                launch_batch_compute_contact_polygons_and_wrenches_from_bvh(
                    state.body_q,
                    solver.body_v_s,
                    solver.model.hydro_batch,
                    contacts.isosurface_batch,
                    twist_convention,
                )
            else:
                launch_batch_compute_contact_polygons_and_wrenches_from_pairs(
                    state.body_q,
                    solver.body_v_s,
                    solver.model.hydro_batch,
                    contacts.isosurface_batch,
                    twist_convention,
                )

            launch_batch_add_wrench_to_body_f(
                body_a_idx=contacts.isosurface_batch.body_a_idx,
                body_b_idx=contacts.isosurface_batch.body_b_idx,
                force=contacts.isosurface_batch.force,
                torque_a=contacts.isosurface_batch.torque_a,
                torque_b=contacts.isosurface_batch.torque_b,
                twist_convention=twist_convention,
                body_f=state.body_f,
            )

        elif twist_convention == 2:  # mujoco convention
            print("Mujoco convention not supported yet")
        else:
            raise ValueError("Invalid twist convention")
