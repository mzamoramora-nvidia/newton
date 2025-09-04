import numpy as np
import warp as wp

import newton.solvers
from newton import Axis
from newton._src.hydroelastic.types import Dirs

# TODO: Move all constants to types.py or to a separate file.
PLANE_NORMAL_THRESHOLD = 1e-12  # Minimum squared length for valid plane normals


@wp.kernel
def transform_array(
    v: wp.array(dtype=wp.vec3f),
    T: wp.transform,
    # outputs
    v_transformed: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    v_transformed[tid] = wp.transform_vector(T, v[tid])


@wp.kernel
def update_tet_mesh_vertices(
    i: wp.int32,
    default_tet_mesh_vertices: wp.array(dtype=wp.vec3f),
    state_0: wp.array(dtype=wp.transform),
    # outputs
    tet_mesh_vertices: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()

    tet_mesh_vertices[tid] = wp.transform_point(state_0[i], default_tet_mesh_vertices[tid])


@wp.kernel
def sum_array_int32(
    a: wp.array(dtype=wp.int32),
    # outputs
    sum: wp.array(dtype=wp.int32),
):
    for i in range(a.size):
        sum[0] += a[i]


@wp.kernel
def count_nonzero_array_int32(
    a: wp.array(dtype=wp.int32),
    # outputs
    count: wp.array(dtype=wp.int32),
):
    for i in range(a.shape[0]):
        if a[i] != 0:
            count[0] += 1


@wp.kernel
def compute_body_q_inv_mat(
    body_q: wp.array(dtype=wp.transform),
    # outputs
    body_q_inv_mat: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()

    body_q_inv = wp.transform_inverse(body_q[tid])
    body_q_inv_mat[tid] = wp.transform_to_matrix(body_q_inv)


@wp.func
def compute_default_homogeneous_tet_transform(points: wp.array(dtype=wp.vec3), elements: wp.vec4i):
    """
    Compute homogeneous transformation matrix for a tetrahedron.
    """
    # Build transformation matrix (row-major format)
    # Each column corresponds to one coordinate (x, y, z) plus homogeneous coordinate
    homogeneous_v = wp.mat44(
        points[elements[0]].x,
        points[elements[1]].x,
        points[elements[2]].x,
        points[elements[3]].x,
        points[elements[0]].y,
        points[elements[1]].y,
        points[elements[2]].y,
        points[elements[3]].y,
        points[elements[0]].z,
        points[elements[1]].z,
        points[elements[2]].z,
        points[elements[3]].z,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    return homogeneous_v


@wp.kernel
def compute_default_tet_transform_inv(
    default_points: wp.array(dtype=wp.vec3f),
    indices: wp.array(dtype=wp.vec4i),
    default_tet_transform_inv: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    # Compute the homogeneous transformation matrix for the tetrahedron.
    tet_transform = compute_default_homogeneous_tet_transform(default_points, indices[tid])

    # Compute the inverse of the homogeneous transformation matrix.
    default_tet_transform_inv[tid] = wp.inverse(tet_transform)

    # Check that the determinant is not too small.
    d = wp.determinant(tet_transform)
    if wp.abs(d) < 1.0e-10:
        wp.printf("Warning: Determinant of tet transform is too small: %e, tet_id: %d\n", d, tid)


@wp.func
def compute_triangle_area(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> float:
    """Compute the area of a triangle given its three vertices."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_product = wp.cross(edge1, edge2)
    return 0.5 * wp.length(cross_product)


@wp.func
def compute_triangle_normal(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, tid: wp.int32) -> wp.vec3:
    """Compute the unit normal of a triangle given its three vertices."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = wp.cross(edge1, edge2)
    normal_length = wp.length(normal)

    if normal_length < PLANE_NORMAL_THRESHOLD:
        area = 0.5 * normal_length
        wp.printf("Warning: Triangle is degenerate, Area: %e.  tid: %d. Returning zero normal.\n", area, tid)
        return wp.vec3(0.0, 0.0, 0.0)  # Default normal for degenerate triangles

    return normal / normal_length


@wp.kernel
def compute_face_normals(
    vertices: wp.array(dtype=wp.vec3f),
    flat_indices: wp.array(dtype=wp.int32),
    face_normals: wp.array(dtype=wp.vec3f),
):
    """
    Compute face normals and areas for a surface mesh.

    Args:
        vertices: Vertex positions
        triangles: Triangle connectivity (vertex indices)
        face_normals: Output face normals (unit vectors)
    """
    tid = wp.tid()
    wid = 3 * tid  # Working index.

    # Get the three vertices of the triangle.
    v0 = vertices[flat_indices[wid]]
    v1 = vertices[flat_indices[wid + 1]]
    v2 = vertices[flat_indices[wid + 2]]

    face_normals[tid] = compute_triangle_normal(v0, v1, v2, tid)


def get_dirs(axis: Axis):
    dirs = Dirs()
    if axis == Axis.Y:
        dirs.up = np.array([0, 1, 0])
        dirs.front = np.array([0, 0, 1])
        dirs.right = np.array([1, 0, 0])

    return dirs


def get_twist_convention(solver):
    # For Newton conventions see: https://newton-physics.github.io/newton/concepts/conventions.html
    # Here we use:
    # - 0 is the defualt newton convetion.
    # - 1 is for the Featherstone solver.
    # - 2 is for the MuJoCo solver.
    if isinstance(solver, newton.solvers.SolverXPBD) or isinstance(solver, newton.solvers.SolverSemiImplicit):
        return 0
    elif isinstance(solver, newton.solvers.SolverFeatherstone):
        return 1
    elif isinstance(solver, newton.solvers.SolverMuJoCo):
        return 2
    else:
        raise ValueError(f"Unknown solver: {solver}")
