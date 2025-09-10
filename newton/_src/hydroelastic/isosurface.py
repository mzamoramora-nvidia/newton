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

from newton._src.hydroelastic.types import mat43h, vec4i
from newton._src.hydroelastic.utils import compute_body_q_inv_mat

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
def get_tetrahedron_bounding_box(tetrahedron_elements: wp.vec4i, vertex_positions: wp.array(dtype=wp.vec3f)):
    """
    Compute axis-aligned bounding box of a tetrahedron.

    Args:
        tetrahedron_elements: Indices of the four tetrahedron vertices
        vertex_positions: Array of vertex positions

    Returns:
        Tuple of (min_bounds, max_bounds) defining the bounding box
    """
    first_vertex_idx = tetrahedron_elements[0]
    min_bounds = vertex_positions[first_vertex_idx]
    max_bounds = vertex_positions[first_vertex_idx]

    # Find min/max bounds across all four vertices
    for i in range(1, 4):
        vertex_idx = tetrahedron_elements[i]
        vertex_position = vertex_positions[vertex_idx]
        min_bounds = wp.min(min_bounds, vertex_position)
        max_bounds = wp.max(max_bounds, vertex_position)

    return min_bounds, max_bounds


@wp.func
def get_tet_bounding_box(v: mat43h):
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

    # Find min/max bounds across all four vertices
    for i in range(1, 4):
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
def find_geom_pairs_bvh_basic(
    body_q: wp.array(dtype=wp.transform),
    body_id_bvh: wp.int32,
    bvh_id: wp.uint64,
    body_id_to_query: wp.int32,
    lowers: wp.array(dtype=wp.vec3f),
    uppers: wp.array(dtype=wp.vec3f),
    query_with_mesh_a: bool,
    # outputs
    geom_pairs_found: wp.array(dtype=wp.vec2i),
):
    tid = wp.tid()  # Each thread should process one geom (tet or triangle) of the querying mesh.

    block = wp.int32(geom_pairs_found.shape[0] / lowers.shape[0])

    # Transform the query AABB to the BVH frame.
    q_inv = wp.transform_inverse(body_q[body_id_bvh])
    xform_aabb_to_bvh = wp.transform_multiply(q_inv, body_q[body_id_to_query])

    lowers_body_frame = wp.transform_point(xform_aabb_to_bvh, lowers[tid])
    uppers_body_frame = wp.transform_point(xform_aabb_to_bvh, uppers[tid])
    query = wp.bvh_query_aabb(bvh_id, lowers_body_frame, uppers_body_frame)
    query_idx = wp.int32(0)
    counter = wp.int32(0)
    while wp.bvh_query_next(query, query_idx):
        if query_with_mesh_a:
            geom_idx_a = tid
            geom_idx_b = query_idx
        else:
            geom_idx_a = query_idx
            geom_idx_b = tid

        idx = tid * block + counter
        if idx >= (tid + 1) * block:
            wp.printf(
                "Query is overflowing: tid, geom_idx_a, geom_idx_b, idx, block, counter: %d, %d, %d, %d, %d, %d\n",
                tid,
                geom_idx_a,
                geom_idx_b,
                idx,
                block,
                counter,
            )
            idx = (tid + 1) * block - 1

        geom_pairs_found[idx] = wp.vec2i(geom_idx_a, geom_idx_b)
        counter += 1
    # wp.printf("tid, counter: %d, %d\n", tid, counter)


@wp.kernel
def find_tet_pairs_bvh(
    body_q: wp.array(dtype=wp.transform),
    body_a: wp.int32,
    body_b: wp.int32,
    points_a: wp.array(dtype=wp.vec3f),
    points_b: wp.array(dtype=wp.vec3f),
    tet_elements_a: wp.array(dtype=wp.vec4i),
    tet_elements_b: wp.array(dtype=wp.vec4i),
    bvh_id_smaller_mesh: wp.uint64,
    lowers_larger_mesh: wp.array(dtype=wp.vec3f),
    uppers_larger_mesh: wp.array(dtype=wp.vec3f),
    mesh_a_is_larger: bool,
    # outputs
    tet_pairs_found: wp.array(dtype=wp.vec2i),
):
    tid = wp.tid()  # Each thread should process one tet of the larger mesh (the mesh with the larger number of tets).

    block = wp.int32(tet_pairs_found.shape[0] / lowers_larger_mesh.shape[0])

    # query Mesh-smaller BVH with larger mesh's AABB
    query = wp.bvh_query_aabb(bvh_id_smaller_mesh, lowers_larger_mesh[tid], uppers_larger_mesh[tid])
    query_idx = wp.int32(0)
    counter = wp.int32(0)
    counter_missed = wp.int32(0)
    while wp.bvh_query_next(query, query_idx):
        if mesh_a_is_larger:
            tet_idx_a = tid
            tet_idx_b = query_idx
        else:
            tet_idx_a = query_idx
            tet_idx_b = tid

        tet_vidx_a = tet_elements_a[tet_idx_a]  # tet vertex indices
        tet_vidx_b = tet_elements_b[tet_idx_b]

        tet_vpos_a_W = mat43h(0.0)  # tet vertex positions in world frame
        tet_vpos_b_W = mat43h(0.0)

        for i in range(4):
            tet_vpos_a_W[i] = wp.transform_point(body_q[body_a], points_a[tet_vidx_a[i]])
            tet_vpos_b_W[i] = wp.transform_point(body_q[body_b], points_b[tet_vidx_b[i]])

        # Early exit: Check bounding box overlap
        min_bounds_a, max_bounds_a = get_tet_bounding_box(tet_vpos_a_W)
        min_bounds_b, max_bounds_b = get_tet_bounding_box(tet_vpos_b_W)

        if not check_bounding_boxes_overlap(min_bounds_a, max_bounds_a, min_bounds_b, max_bounds_b):
            counter_missed += 1
            continue

        idx = tid * block + counter
        if idx >= (tid + 1) * block:
            # wp.printf(
            #     "Query is overflowing: tid, tet_idx_b, idx, block, conter: %d, %d, %d, %d, %d\n",
            #     tid,
            #     tet_idx_b,
            #     idx,
            #     block,
            #     counter,
            # )
            idx = (tid + 1) * block - 1

        tet_pairs_found[idx] = wp.vec2i(tet_idx_a, tet_idx_b)
        counter += 1

    # Can happend if box falls indefinitely.
    # if counter_missed > 0:
    #    wp.printf("tid, counter, counter_missed: %d, %d, %d\n", tid, counter, counter_missed)


@wp.kernel
def compute_soft_soft_contact_surface_elements(
    body_q: wp.array(dtype=wp.transform),
    body_q_inv_mat: wp.array(dtype=wp.mat44),
    body_a: wp.int32,
    body_b: wp.int32,
    tet_pairs_found: wp.array(dtype=wp.vec2i),
    points_a: wp.array(dtype=wp.vec3f),
    points_b: wp.array(dtype=wp.vec3f),
    default_tet_transform_inv_a: wp.array(dtype=wp.mat44),
    default_tet_transform_inv_b: wp.array(dtype=wp.mat44),
    tet_elements_a: wp.array(dtype=wp.vec4i),  # tet vertex indices
    tet_elements_b: wp.array(dtype=wp.vec4i),
    p_a: wp.array(dtype=wp.float32),  # pressure field values
    p_b: wp.array(dtype=wp.float32),
    grad_p_a: wp.array(dtype=wp.vec3f),
    grad_p_b: wp.array(dtype=wp.vec3f),
    h_a: wp.float32,  # hydroelastic modulus
    h_b: wp.float32,
    # outputs
    cp_vertices: wp.array(dtype=wp.vec3f),
    cp_vcounts: wp.array(dtype=wp.int32),
    cp_centroids: wp.array(dtype=wp.vec3f),
    cp_normals: wp.array(dtype=wp.vec3f),
    cp_penetration: wp.array(dtype=wp.vec4f),
    centroid_pressure: wp.array(dtype=wp.float32),
    # Outputs for debugging.
    intermediate_cp_vertices: wp.array(dtype=wp.vec3f),
    intermediate_cp_vcounts: wp.array(dtype=wp.int32),
    intermediate_cp_centroids: wp.array(dtype=wp.vec3f),
):
    """
    Compute hydroelastic contact surface elements between tetrahedron pairs.

    This kernel implements the main algorithm for computing equal pressure
    surfaces between pairs of tetrahedra from two different meshes.
    """
    tid = wp.tid()  # Each thread process one tet pair

    if tet_pairs_found[tid][0] == -1 or tet_pairs_found[tid][1] == -1:
        return

    tet_idx_a = tet_pairs_found[tid][0]
    tet_idx_b = tet_pairs_found[tid][1]

    # wp.printf("tet_idx_a: %d, tet_idx_b: %d\n", tet_idx_a, tet_idx_b)

    tet_vidx_a = tet_elements_a[tet_idx_a]  # tet vertex indices
    tet_vidx_b = tet_elements_b[tet_idx_b]

    tet_vpos_a_W = mat43h(0.0)  # tet vertex positions in world frame
    tet_vpos_b_W = mat43h(0.0)

    for i in range(4):
        tet_vpos_a_W[i] = wp.transform_point(body_q[body_a], points_a[tet_vidx_a[i]])
        tet_vpos_b_W[i] = wp.transform_point(body_q[body_b], points_b[tet_vidx_b[i]])

    # # Early exit: Check bounding box overlap
    # min_bounds_a, max_bounds_a = get_tet_bounding_box(tet_vpos_a_W)
    # min_bounds_b, max_bounds_b = get_tet_bounding_box(tet_vpos_b_W)

    # if not check_bounding_boxes_overlap(min_bounds_a, max_bounds_a, min_bounds_b, max_bounds_b):
    #     # wp.printf("No overlap for tet pair %d. Tet a: %d, Tet b: %d\n", tid, tet_idx_a, tet_idx_b)
    #     return

    # Build pressure field vectors.
    # TODO: Consider building the field vectors when loading the meshes.
    p_a_vec = wp.vec4(p_a[tet_vidx_a[0]], p_a[tet_vidx_a[1]], p_a[tet_vidx_a[2]], p_a[tet_vidx_a[3]])
    p_b_vec = wp.vec4(p_b[tet_vidx_b[0]], p_b[tet_vidx_b[1]], p_b[tet_vidx_b[2]], p_b[tet_vidx_b[3]])

    inv_mat_a = default_tet_transform_inv_a[tet_idx_a] * body_q_inv_mat[body_a]
    inv_mat_b = default_tet_transform_inv_b[tet_idx_b] * body_q_inv_mat[body_b]

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
    warn_degenerate_plane(plane_normal, tid)

    # Transform the pressure gradient to the world frame.
    grad_p_a_W = wp.transform_vector(body_q[body_a], grad_p_a[tet_idx_a])
    grad_p_b_W = -wp.transform_vector(body_q[body_b], grad_p_b[tet_idx_b])

    # Check if the plane normal is aligned with the pressure gradient.
    if not is_normal_along_pressure_gradient(grad_p_a_W, plane_normal, body_a, tid):
        return

    if not is_normal_along_pressure_gradient(grad_p_b_W, plane_normal, body_b, tid):
        return

    # Array pointing to block of memory for the vertices of the current contact polygon.
    ptr = cp_vertices.ptr + wp.uint64(MAX_POLYGON_VERTICES * tid * VEC3F_BYTE_SIZE_)
    cp_v = wp.array(ptr=ptr, shape=(MAX_POLYGON_VERTICES,), dtype=wp.vec3f)

    # Build initial polygon from plane-tetrahedron intersection
    # Clip polygon with first tetrahedron
    cp_vcounts[tid] = plane_tetrahedron_intersection(plane_equation, tet_vpos_a_W, cp_v)

    # Return if the polygon is empty.
    if cp_vcounts[tid] == 0:
        return

    # Store intermediate polygon for debugging.
    for i in range(cp_vcounts[tid]):
        intermediate_cp_vertices[8 * tid + i] = cp_v[i]
    intermediate_cp_centroids[tid] = compute_polygon_centroid(cp_v, cp_vcounts[tid])
    intermediate_cp_vcounts[tid] = cp_vcounts[tid]

    # Clip polygon with second tetrahedron
    cp_vcounts[tid] = clip_polygon_with_tetrahedron(tet_vpos_b_W, cp_v, cp_vcounts[tid])

    # Return if the polygon is empty.
    if cp_vcounts[tid] == 0:
        return

    # TODO: Remove duplcated vertices, based on distance or on the area of the triangles.

    # Compute centroid of the final polygon.
    centroid = compute_polygon_centroid(cp_v, cp_vcounts[tid])

    # Compute pressure at the centroid.
    c_homogeneous = wp.vec4(centroid.x, centroid.y, centroid.z, 1.0)
    penetration_extent_a = wp.dot(homogeneous_to_penetration_map_a, c_homogeneous)
    pressure_a = h_a * penetration_extent_a
    warn_potential_overflow_for_pressure(h_a, penetration_extent_a, tid)

    # Store results
    cp_centroids[tid] = centroid
    cp_normals[tid] = plane_normal
    cp_penetration[tid] = homogeneous_to_penetration_map_a
    centroid_pressure[tid] = pressure_a

    # Check that pressures are well defined e.g.: not NaN or Inf and not negative.
    # TODO: Check that pressures are equal.
    warn_degenerate_pressure(pressure_a, tid)


@wp.kernel
def generate_contact_surface_triangulation(
    valid_polygon_indices: wp.array(dtype=wp.int32),
    tet_pairs: wp.array(dtype=wp.vec2i),
    contact_polygon_vertices: wp.array(dtype=wp.vec3f),
    contact_polygon_centroids: wp.array(dtype=wp.vec3f),
    contact_polygon_vertex_counts: wp.array(dtype=wp.int32),
    contact_surface_normals: wp.array(dtype=wp.vec3f),
    contact_pressure_values: wp.array(dtype=wp.float32),
    vertex_offset_cumulative: wp.array(dtype=wp.int32),
    total_polygon_vertices: wp.int32,
    # outputs
    triangulated_vertices: wp.array(dtype=wp.vec3f),
    triangulated_elements: wp.array(dtype=wp.vec3i),
    per_triangle_pressure: wp.array(dtype=wp.float32),
    visualization_radii: wp.array(dtype=wp.vec2i),
    visualization_edges: wp.array(dtype=vec4i),
    per_triangle_tet_pairs: wp.array(dtype=wp.vec2i),
):
    """
    Generate triangulated surface from contact polygons.

    This kernel creates a triangular mesh representation of the contact surface
    by triangulating each polygon with a fan pattern from its centroid.
    """
    thread_id = wp.tid()

    polygon_idx = valid_polygon_indices[thread_id]

    # Determine starting index for this polygon's vertices
    vertex_start_index = 0
    if thread_id > 0:
        vertex_start_index = vertex_offset_cumulative[polygon_idx - 1]

    # Copy polygon vertices to output array
    polygon_vertex_count = contact_polygon_vertex_counts[polygon_idx]
    for i in range(polygon_vertex_count):
        triangulated_vertices[vertex_start_index + i] = contact_polygon_vertices[8 * polygon_idx + i]

    # Add centroid vertex after all polygon vertices
    centroid_vertex_index = total_polygon_vertices + thread_id
    triangulated_vertices[centroid_vertex_index] = contact_polygon_centroids[polygon_idx]

    # Generate triangular elements using fan pattern from centroid
    for i in range(polygon_vertex_count):
        # Check that the normals are aligned.
        index_a = centroid_vertex_index
        index_b = vertex_start_index + i
        index_c = vertex_start_index + (i + 1) % polygon_vertex_count
        vertex_a = triangulated_vertices[index_a]
        vertex_b = triangulated_vertices[index_b]
        vertex_c = triangulated_vertices[index_c]
        surface_normal = wp.normalize(contact_surface_normals[polygon_idx])
        triangle_normal = wp.normalize(wp.cross(vertex_b - vertex_a, vertex_c - vertex_a))
        if wp.abs(1.0 - wp.dot(triangle_normal, surface_normal)) < TRIANGLE_NORMAL_ALIGNMENT_THRESHOLD:
            # wp.printf("Warning: Normals are not aligned for polygon %d\n", polygon_idx)
            triangulated_elements[vertex_start_index + i] = wp.vec3i(index_a, index_b, index_c)
        else:
            triangulated_elements[vertex_start_index + i] = wp.vec3i(index_a, index_c, index_b)

        per_triangle_pressure[vertex_start_index + i] = contact_pressure_values[polygon_idx]

        # Generate visualization edges (spoke pattern from centroid)
        visualization_radii[vertex_start_index + i] = wp.vec2i(index_a, index_b)

        visualization_edges[vertex_start_index + i] = vec4i(index_a, index_b, index_b, index_c)

        per_triangle_tet_pairs[vertex_start_index + i] = tet_pairs[polygon_idx]


@wp.func
def get_triangle_bounding_box(triangle_vertices: wp.mat33):
    """
    Compute axis-aligned bounding box of a triangle.

    Args:
        triangle_vertices: 3x3 matrix, where each row is a vertex of the triangle.
    Returns:
        Tuple of (min_bounds, max_bounds) defining the bounding box
    """
    v0 = triangle_vertices[0]
    v1 = triangle_vertices[1]
    v2 = triangle_vertices[2]

    min_bounds = wp.min(wp.min(v0, v1), v2)
    max_bounds = wp.max(wp.max(v0, v1), v2)

    return min_bounds, max_bounds


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


@wp.kernel
def compute_soft_hard_contact_surface_elements(
    body_q: wp.array(dtype=wp.transform),
    body_q_inv_mat: wp.array(dtype=wp.mat44),
    body_a: wp.int32,  # Body a is assumed to be soft
    body_b: wp.int32,
    tet_triangle_pairs: wp.array(dtype=wp.vec2i),
    # Soft body (tetrahedra) data
    points_a: wp.array(dtype=wp.vec3f),
    default_tet_transform_inv_a: wp.array(dtype=wp.mat44),
    tet_elements_a: wp.array(dtype=wp.vec4i),  # tet vertex indices
    p_a: wp.array(dtype=wp.float32),  # pressure field values
    grad_p_a: wp.array(dtype=wp.vec3f),
    h_a: wp.float32,
    # Hard body (surface) data
    points_b: wp.array(dtype=wp.vec3f),
    tri_elements_b: wp.array(dtype=wp.int32),  # triangle vertices are stored in a flat array
    tri_normals_b: wp.array(dtype=wp.vec3f),
    # Outputs
    cp_vertices: wp.array(dtype=wp.vec3f),
    cp_vcounts: wp.array(dtype=wp.int32),
    cp_centroids: wp.array(dtype=wp.vec3f),
    cp_normals: wp.array(dtype=wp.vec3f),
    cp_penetration: wp.array(dtype=wp.vec4f),
    centroid_pressure: wp.array(dtype=wp.float32),
):
    """
    Compute hydroelastic contact surface elements between tetrahedra and triangles.

    This kernel implements the soft-hard contact algorithm, where each thread
    processes one tetrahedron-triangle pair.

    Reference: mesh_intersection.cc lines 323-382
    Drake function: CalcContactPolygon(const VolumeMeshFieldLinear<double, double>& volume_field_M,
                                      const TriangleSurfaceMesh<double>& surface_N,
                                      const math::RigidTransform<T>& X_MN,
                                      const math::RigidTransform<double>& X_MN_d,
                                      MeshBuilder* builder_M,
                                      const bool filter_face_normal_along_field_gradient,
                                      const int tet_index, const int tri_index)
    """
    tid = wp.tid()  # Each thread processes one tetrahedron-triangle pair.

    if tet_triangle_pairs[tid][0] == -1 or tet_triangle_pairs[tid][1] == -1:
        return

    tet_id = tet_triangle_pairs[tid][0]
    tri_id = tet_triangle_pairs[tid][1]

    # Get tetrahedron and triangle elements
    tet_vidx_a = tet_elements_a[tet_id]  # tet vertex indices
    tri_vidx_b = wp.vec3i(
        tri_elements_b[3 * tri_id], tri_elements_b[3 * tri_id + 1], tri_elements_b[3 * tri_id + 2]
    )  # triangle vertex indices

    # Transform tetrahedron vertices to world space
    tet_vertices = mat43h(0.0)
    for i in range(4):
        tet_vertices[i] = wp.transform_point(body_q[body_a], points_a[tet_vidx_a[i]])

    # Transform triangle vertices to world space
    triangle_vertices = wp.mat33(0.0)  # wp.matrix(0.0, shape=(3, 3), dtype=wp.float32)
    for i in range(3):
        triangle_vertices[i] = wp.transform_point(body_q[body_b], points_b[tri_vidx_b[i]])

    # Early exit: Check bounding box overlap
    tet_min, tet_max = get_tet_bounding_box(tet_vertices)
    tri_min, tri_max = get_triangle_bounding_box(triangle_vertices)

    if not check_bounding_boxes_overlap(tet_min, tet_max, tri_min, tri_max):
        return

    # Check face normal alignment if requested
    # Equivalent to lines 332-337 in mesh_intersection.cc:
    # if (filter_face_normal_along_field_gradient) {
    #   if (!this->IsFaceNormalAlongPressureGradient(
    #           volume_field_M, surface_N, X_MN_d, tet_index, tri_index)) {
    #     return;
    #   }
    # }

    # Transform field gradient to world space
    soft_grad_W = wp.transform_vector(body_q[body_a], grad_p_a[tet_id])

    # Transform triangle normal to world space
    hard_normal_W = wp.transform_vector(body_q[body_b], tri_normals_b[tri_id])

    if not is_normal_along_pressure_gradient(soft_grad_W, hard_normal_W, body_a, tid):
        return

    # Clip triangle by tetrahedron
    # Equivalent to lines 349-353 in mesh_intersection.cc:
    # const std::vector<Vector3<T>>& polygon_vertices_M =
    #     this->ClipTriangleByTetrahedron(tet_index, vol_mesh_M, tri_index,
    #                                     surface_N, X_MN);
    # if (polygon_vertices_M.size() < 3) return;

    # TODO: Find a better way to initialize the polygon vertices.
    ptr = cp_vertices.ptr + wp.uint64(8 * tid * VEC3F_BYTE_SIZE_)
    polygon_vertices = wp.array(ptr=ptr, shape=(8,), dtype=wp.vec3f)
    for i in range(3):
        polygon_vertices[i] = triangle_vertices[i]

    cp_vcounts[tid] = 3
    cp_vcounts[tid] = clip_polygon_with_tetrahedron(tet_vertices, polygon_vertices, cp_vcounts[tid])

    if cp_vcounts[tid] < 3:
        cp_vcounts[tid] = 0
        return

    # Compute polygon centroid
    centroid = compute_polygon_centroid(polygon_vertices, cp_vcounts[tid])

    # Build pressure field vectors.
    # Equivalent to lines 355-361 in mesh_intersection.cc:
    # Add the vertices to the builder (with corresponding field values) and
    # construct index-based polygon representation.
    # polygon_vertex_indices_.clear();
    # for (const auto& p_MV : polygon_vertices_M) {
    #   polygon_vertex_indices_.push_back(builder_M->AddVertex(
    #       p_MV, volume_field_M.EvaluateCartesian(tet_index, p_MV)));
    # }
    p_a_vec = wp.vec4(p_a[tet_vidx_a[0]], p_a[tet_vidx_a[1]], p_a[tet_vidx_a[2]], p_a[tet_vidx_a[3]])
    inv_mat = default_tet_transform_inv_a[tet_id] * body_q_inv_mat[body_a]
    homogeneous_to_penetration_map = p_a_vec * inv_mat

    # Compute pressure at centroid
    # Equivalent to lines 369-370 in mesh_intersection.cc:
    # const Vector3<T> grad_e_MN_M = volume_field_M.EvaluateGradient(tet_index);
    # Note: In our implementation, pressure is computed as modulus * penetration
    c_homogeneous = wp.vec4(centroid.x, centroid.y, centroid.z, 1.0)
    penetration_extent_a = wp.dot(homogeneous_to_penetration_map, c_homogeneous)
    pressure_a = h_a * penetration_extent_a
    warn_potential_overflow_for_pressure(h_a, penetration_extent_a, tid)

    # Store results
    cp_centroids[tid] = centroid
    cp_normals[tid] = hard_normal_W  # Normal (from hard surface)
    cp_penetration[tid] = homogeneous_to_penetration_map
    centroid_pressure[tid] = pressure_a


def launch_compute_soft_vs_soft_contact_surface(
    body_q, body_q_inv_mat, mesh_a, mesh_b, isosurface, update_contact_pairs=True
):
    # if mesh_a.aabb_low.shape[0] > mesh_b.aabb_low.shape[0]:
    #     wp.launch(
    #         find_tet_pairs_bvh,
    #         dim=mesh_a.volume_mesh.indices.shape[0],
    #         inputs=[
    #             body_q,
    #             isosurface.body_a_wp,
    #             isosurface.body_b_wp,
    #             mesh_a.volume_mesh.default_points,
    #             mesh_b.volume_mesh.default_points,
    #             mesh_a.volume_mesh.indices,
    #             mesh_b.volume_mesh.indices,
    #             mesh_b.bvh.id,
    #             mesh_a.aabb_low,
    #             mesh_a.aabb_high,
    #             True,
    #         ],
    #         outputs=[
    #             isosurface.geom_pairs_found,
    #         ],
    #     )
    # else:
    #     wp.launch(
    #         find_tet_pairs_bvh,
    #         dim=mesh_b.volume_mesh.indices.shape[0],
    #         inputs=[
    #             body_q,
    #             isosurface.body_a_wp,
    #             isosurface.body_b_wp,
    #             mesh_a.volume_mesh.default_points,
    #             mesh_b.volume_mesh.default_points,
    #             mesh_a.volume_mesh.indices,
    #             mesh_b.volume_mesh.indices,
    #             mesh_a.bvh.id,
    #             mesh_b.aabb_low,
    #             mesh_b.aabb_high,
    #             False,
    #         ],
    #         outputs=[
    #             isosurface.geom_pairs_found,
    #         ],
    #     )

    find_geom_pairs(body_q, isosurface, mesh_a, mesh_b, update_contact_pairs)

    wp.launch(
        compute_soft_soft_contact_surface_elements,
        dim=isosurface.geom_pairs_found.shape[0],
        inputs=[
            body_q,
            body_q_inv_mat,
            isosurface.body_a_wp,
            isosurface.body_b_wp,
            isosurface.geom_pairs_found,
            mesh_a.volume_mesh.default_points,
            mesh_b.volume_mesh.default_points,
            mesh_a.volume_mesh.default_tet_transform_inv,
            mesh_b.volume_mesh.default_tet_transform_inv,
            mesh_a.volume_mesh.indices,
            mesh_b.volume_mesh.indices,
            mesh_a.volume_mesh.field,
            mesh_b.volume_mesh.field,
            mesh_a.volume_mesh.field_gradient,
            mesh_b.volume_mesh.field_gradient,
            mesh_a.hydroelastic_modulus,
            mesh_b.hydroelastic_modulus,
        ],
        outputs=[
            isosurface.contact_polygon.vertices,
            isosurface.contact_polygon.vertex_counts,
            isosurface.contact_polygon.centroids,
            isosurface.contact_polygon.normals,
            isosurface.contact_polygon.cartesian_to_penetration,
            isosurface.contact_polygon.centroid_pressure,
            isosurface.intermediate_contact_polygon.vertices,
            isosurface.intermediate_contact_polygon.vertex_counts,
            isosurface.intermediate_contact_polygon.centroids,
        ],
    )


def find_geom_pairs(body_q, isosurface, mesh_a, mesh_b, update_contact_pairs):
    # In this context: geom pairs can be tet_vs_tet pairs or tet_vs_tri pairs.
    if update_contact_pairs:
        if mesh_a.aabb_low.shape[0] > mesh_b.aabb_low.shape[0]:
            wp.launch(
                find_geom_pairs_bvh_basic,
                dim=mesh_a.aabb_low.shape[0],
                inputs=[
                    body_q,
                    mesh_b.body_id,
                    mesh_b.bvh.id,
                    mesh_a.body_id,
                    mesh_a.aabb_low,
                    mesh_a.aabb_high,
                    True,
                ],
                outputs=[
                    isosurface.geom_pairs_found,
                ],
            )
        else:
            wp.launch(
                find_geom_pairs_bvh_basic,
                dim=mesh_b.aabb_low.shape[0],
                inputs=[
                    body_q,
                    mesh_a.body_id,
                    mesh_a.bvh.id,
                    mesh_b.body_id,
                    mesh_b.aabb_low,
                    mesh_b.aabb_high,
                    False,
                ],
                outputs=[
                    isosurface.geom_pairs_found,
                ],
            )


def launch_compute_soft_vs_hard_contact_surface(
    body_q, body_q_inv_mat, mesh_a, mesh_b, isosurface, update_contact_pairs=True
):
    find_geom_pairs(body_q, isosurface, mesh_a, mesh_b, update_contact_pairs)

    wp.launch(
        compute_soft_hard_contact_surface_elements,
        dim=isosurface.geom_pairs_found.shape[0],
        inputs=[
            body_q,
            body_q_inv_mat,
            isosurface.body_a_wp,
            isosurface.body_b_wp,
            isosurface.geom_pairs_found,
            # Soft body data
            mesh_a.volume_mesh.default_points,
            mesh_a.volume_mesh.default_tet_transform_inv,
            mesh_a.volume_mesh.indices,
            mesh_a.volume_mesh.field,
            mesh_a.volume_mesh.field_gradient,
            mesh_a.hydroelastic_modulus,
            # Hard body data
            mesh_b.surface_mesh.points,
            mesh_b.surface_mesh.indices,
            mesh_b.surface_mesh.normals,  # Pre-computed face normals
        ],
        outputs=[
            isosurface.contact_polygon.vertices,
            isosurface.contact_polygon.vertex_counts,
            isosurface.contact_polygon.centroids,
            isosurface.contact_polygon.normals,
            isosurface.contact_polygon.cartesian_to_penetration,
            isosurface.contact_polygon.centroid_pressure,
        ],
    )


def compute_contact_polygons(
    body_q,
    body_q_inv_mat,
    mesh_a,
    mesh_b,
    isosurface,
    update_contact_pairs=True,
):
    """
    Extract isosurface (equal pressure surface) between two tetrahedral meshes.

    This function computes the hydroelastic contact surface between two
    tetrahedral meshes by finding the equal pressure surface where the
    pressure fields of the two meshes are equal.

    Args:
        tetrahedron_pairs: Array of tetrahedron index pairs to test
        vertex_positions_a, vertex_positions_b: Vertex positions for each mesh
        tetrahedron_elements_a, tetrahedron_elements_b: Tetrahedron connectivity
        pressure_field_values_a, pressure_field_values_b: Pressure field values at vertices
        hydroelastic_modulus_a, hydroelastic_modulus_b: Material Young's modulus values

    Returns:
        Tuple containing:
        - is_valid: Boolean indicating if valid surface was found
        - surface_vertices: Triangulated surface vertices
        - surface_elements: Triangulated surface elements
        - visualization_edges: Edges for visualization
        - unused_edges: Placeholder for compatibility
        - surface_normals: Surface normal vectors
        - surface_centroids: Polygon centroid positions
    """

    ## Reset vertex counts only instead of resetting everything.
    ## This means that the vertices and centroids will contain the previous values
    ## but those should be unused.
    # reset_contact_polygon(isosurface.intermediate_contact_polygon)
    # reset_contact_polygon(isosurface.contact_polygon)
    isosurface.contact_polygon.vertex_counts.zero_()
    isosurface.intermediate_contact_polygon.vertex_counts.zero_()
    if update_contact_pairs:
        isosurface.geom_pairs_found.fill_(-1)

    # Compute contact surface elements
    if isosurface.soft_vs_soft:
        launch_compute_soft_vs_soft_contact_surface(
            body_q, body_q_inv_mat, mesh_a, mesh_b, isosurface, update_contact_pairs
        )
    else:
        launch_compute_soft_vs_hard_contact_surface(
            body_q, body_q_inv_mat, mesh_a, mesh_b, isosurface, update_contact_pairs
        )


def compute_contact_surfaces(model, state_0, contacts, body_q_inv_mat, update_contact_pairs=True):
    with wp.ScopedTimer("Computation of contact surfaces", print=False):
        # Compute inverse transform of body_q
        wp.launch(
            compute_body_q_inv_mat,
            dim=model.body_count,
            inputs=[state_0.body_q],
            outputs=[body_q_inv_mat],
        )

        # Compute hydroelastic contact isosurface.
        main_stream = wp.get_stream()
        init_event = main_stream.record_event()
        for isosurface in contacts.isosurface:
            isosurface.stream.wait_event(init_event)
            with wp.ScopedStream(isosurface.stream):
                compute_contact_polygons(
                    state_0.body_q,
                    body_q_inv_mat,
                    model.hydro_mesh[isosurface.body_a],
                    model.hydro_mesh[isosurface.body_b],
                    isosurface,
                    update_contact_pairs=update_contact_pairs,
                )
            isosurface.stream.record_event(isosurface.sync_event)

        for isosurface in contacts.isosurface:
            main_stream.wait_event(isosurface.sync_event)
