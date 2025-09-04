import warp as wp


@wp.kernel
def max_of_array(
    arr: wp.array(dtype=wp.float32),
    # outputs
    m: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    wp.atomic_max(m, 0, arr[tid])


@wp.kernel
def compute_distance_to_surface(
    V: wp.array(dtype=wp.vec3f),
    mesh_id: wp.uint64,
    distance_threshold: wp.float32,
    # outputs
    distance_to_surface: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # Compute the closes point on surface Mesh.
    query = wp.mesh_query_point(mesh_id, point=V[tid], max_dist=distance_threshold)
    if not query.result:
        print("No closest point found in kernel: compute_distance_to_surface.")
        return

    closest_point = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    distance_to_surface[tid] = wp.length(V[tid] - closest_point)


# TODO: Find a better name for this kernel.
@wp.kernel
def simple_field_intialization(
    # outputs
    field: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # Assumptions:
    # - The field values are per vertex.
    # - The last vertex is the centroid
    # - The field value is maximum for vertex associated with the centroid,
    # - Field values are zero for all the vertices on the surface.
    # TODO:
    # - We could also pass an argument for min field value and max field value.
    if tid != field.shape[0] - 1:
        field[tid] = 0.0
    else:
        field[tid] = 1.0


@wp.kernel
def init_convex_field(
    V: wp.array(dtype=wp.vec3f),
    distance_to_surface: wp.array(dtype=wp.float32),
    is_on_surface: wp.array(dtype=wp.bool),
    max_distance: wp.array(dtype=wp.float32),
    hydroelastic_modulus: wp.float32,
    margin: wp.float32,
    # outputs
    field: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # Assumptions:
    # - The last vertex is the centroid
    # - The field value is maximum for vertex associated with the centroid,
    # - Field values are zero for all the vertices on the surface.

    # If vertex is on the surface, set field value to 0
    if is_on_surface[tid]:
        field[tid] = 0.0
        return

    # Compute the field value.
    field[tid] = hydroelastic_modulus * (distance_to_surface[tid] - margin) / (max_distance[0] - margin)


@wp.kernel
def compute_field_gradient(
    vertices: wp.array(dtype=wp.vec3f),
    field: wp.array(dtype=wp.float32),
    tetrahedra: wp.array(dtype=wp.vec4i),
    # outputs
    gradient: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    # This implementation should be numerically equivalent to the one in Drake.
    # It has been tested against the one in Drake.

    # Get the tetrahedron vertices
    tet = tetrahedra[tid]
    v0 = vertices[tet[0]]
    v1 = vertices[tet[1]]
    v2 = vertices[tet[2]]
    v3 = vertices[tet[3]]

    # Get the field values at the vertices
    f0 = field[tet[0]]
    f1 = field[tet[1]]
    f2 = field[tet[2]]
    f3 = field[tet[3]]

    # Compute edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Compute the volume of tetrahedron (1/6 of the scalar triple product)
    six_volume = wp.dot(e1, wp.cross(e2, e3))

    # Avoid division by zero
    if wp.abs(six_volume) < 1e-12:
        gradient[tid] = wp.vec3f(0.0, 0.0, 0.0)
        return

    # Compute gradients of shape functions
    # For a tetrahedron, the gradient of the field is:
    # Grad f = (f1-f0) * Grad N1 + (f2-f0) * Grad N2 + (f3-f0) * Grad N3
    # where Grad Ni are the gradients of the shape functions

    # Shape function gradients for tetrahedron
    grad_N1 = wp.cross(e2, e3) / (six_volume)
    grad_N2 = wp.cross(e3, e1) / (six_volume)
    grad_N3 = wp.cross(e1, e2) / (six_volume)

    # Compute the field gradient
    grad_field = (f1 - f0) * grad_N1 + (f2 - f0) * grad_N2 + (f3 - f0) * grad_N3

    # Store the gradient for this tetrahedron
    gradient[tid] = grad_field
