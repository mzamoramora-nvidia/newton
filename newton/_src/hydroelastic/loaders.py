import numpy as np
import warp as wp

import newton
from newton._src.hydroelastic.make_fields import (
    compute_distance_to_surface,
    compute_field_gradient,
    init_convex_field,
    max_of_array,
)
from newton._src.hydroelastic.tetrahedralizer import tetrahedralize
from newton._src.hydroelastic.types import (
    HydroelasticObject,
)
from newton._src.hydroelastic.utils import (
    compute_aabb_elements,
    compute_default_tet_transform_inv,
    compute_face_normals,
    transform_array,
)
from newton._src.hydroelastic.wrenches import combine_material_properties


@wp.kernel
def set_query_with_mesh_a(
    elements_count: wp.array(dtype=wp.int32),
    body_a_idx: wp.array(dtype=wp.int32),
    body_b_idx: wp.array(dtype=wp.int32),
    # outputs
    query_with_mesh_a: wp.array(dtype=wp.bool),
):
    tid = wp.tid()
    body_a = body_a_idx[tid]
    body_b = body_b_idx[tid]
    query_with_mesh_a[tid] = elements_count[body_a] > elements_count[body_b]


@wp.kernel
def set_soft_vs_soft(
    is_soft: wp.array(dtype=wp.bool),
    body_a_idx: wp.array(dtype=wp.int32),
    body_b_idx: wp.array(dtype=wp.int32),
    soft_vs_soft: wp.array(dtype=wp.bool),
):
    tid = wp.tid()
    body_a = body_a_idx[tid]
    body_b = body_b_idx[tid]
    soft_vs_soft[tid] = is_soft[body_a] and is_soft[body_b]


def extract_surface_mesh(vertices, tetrahedrons):
    # Convert to numpy arrays
    vertices = np.array(vertices)
    tetrahedrons = np.array(tetrahedrons)

    # Define faces for a tetrahedron
    faces = np.concatenate(
        [
            tetrahedrons[:, [0, 1, 2]],
            tetrahedrons[:, [0, 1, 3]],
            tetrahedrons[:, [0, 2, 3]],
            tetrahedrons[:, [1, 2, 3]],
        ],
        axis=0,
    )

    # Sort faces by vertices
    faces = np.sort(faces, axis=1)

    # Find unique faces and counts
    faces_tuple = [tuple(face) for face in faces]
    faces_array = np.array(faces_tuple)
    _, idx, counts = np.unique(faces_array, axis=0, return_index=True, return_counts=True)

    surface_faces = faces[idx[counts == 1]]

    centroid = np.mean(vertices, axis=0)
    print("centroid extract_surface_mesh", centroid)
    # Check that the normals are pointing outwards
    for i, face in enumerate(surface_faces):
        normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
        normal = normal / np.linalg.norm(normal)
        triangle_centroid = 1.0 / 3.0 * (vertices[face[0]] + vertices[face[1]] + vertices[face[2]])
        ray_direction = triangle_centroid - centroid
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        if np.dot(normal, ray_direction) < 0:
            surface_faces[i] = [face[0], face[2], face[1]]

    # import trimesh
    # mesh = trimesh.Trimesh(vertices=vertices,
    #                    faces=surface_faces)

    # print("mesh.is_watertight", mesh.is_watertight)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.plot_trisurf(mesh.vertices[:, 0],
    #                 mesh.vertices[:,1],
    #                 mesh.vertices[:,2],
    #                 triangles=mesh.faces,)
    # plt.show()

    return surface_faces


def tet_vol_signed(element, V):
    v0_idx = element[0]
    v1_idx = element[1]
    v2_idx = element[2]
    v3_idx = element[3]
    vertex_0 = V[v0_idx]
    vertex_1 = V[v1_idx]
    vertex_2 = V[v2_idx]
    vertex_3 = V[v3_idx]
    return np.inner(vertex_3 - vertex_0, np.cross(vertex_1 - vertex_0, vertex_2 - vertex_0))


def check_tetrahedron_orientation(elements, V):
    for i in range(elements.shape[0] // 4):
        element = elements[i * 4 : (i + 1) * 4]
        signed_volume = tet_vol_signed(element, V)
        if signed_volume < 0.0:
            # TODO: Swap elements in case of incorrect orientation.
            print(
                f"Warning in function load_drake_mesh. Incorrect tetrahedron orientation. signed_volume: {signed_volume}, element[{i}]: {element}"
            )


def initialize_default_tet_transform_inv(volume_mesh, compute_device):
    num_tets = volume_mesh.elements_count
    volume_mesh.default_tet_transform_inv = wp.zeros(num_tets, dtype=wp.mat44, device=compute_device)
    wp.launch(
        compute_default_tet_transform_inv,
        dim=num_tets,
        inputs=[volume_mesh.default_points, volume_mesh.indices],
        outputs=[volume_mesh.default_tet_transform_inv],
    )


def load_drake_mesh(path: str, Tf, params, compute_device=None):
    if compute_device is None:
        compute_device = wp.get_device()

    import json  # noqa: PLC0415

    with open(path) as f:
        data = json.load(f)

    V = np.array(data["vertices"])
    elements = np.array(data["elements"])

    mean_vertex = np.mean(V, axis=0)
    if params.get("recenter"):
        print("recentering drake mesh", mean_vertex)
        V = V - mean_vertex
        mean_vertex = np.mean(V, axis=0)
    print("Avg of vertices from drake mesh: ", mean_vertex)

    edges = []
    for element in elements:
        for i in range(len(element)):
            j = (i + 1) % len(element)
            edges.append([element[i], element[j]])

    # Check that the tetrahedron orientation is correct.
    check_tetrahedron_orientation(elements.flatten(), V)

    points = wp.array(V, dtype=wp.vec3f, device=compute_device)
    # indices = wp.array(F, dtype=wp.vec3i, device=compute_device)
    points_transformed = wp.zeros(points.shape[0], dtype=wp.vec3f, device=compute_device)
    wp.launch(
        transform_array,
        dim=points.shape[0],
        inputs=[points, Tf],
        outputs=[points_transformed],
    )

    hydroelastic = HydroelasticObject()
    hydroelastic.is_visible = params["is_visible"]
    hydroelastic.hydroelastic_modulus = wp.float32(params["hydroelastic_modulus"])
    # Initialize volume mesh.
    hydroelastic.mesh.default_points = points_transformed
    hydroelastic.mesh.indices = wp.array(elements.flatten(), dtype=wp.int32, device=compute_device)
    hydroelastic.mesh.elements_stride = 4
    hydroelastic.mesh.elements_count = int(hydroelastic.mesh.indices.shape[0] / 4)

    initialize_default_tet_transform_inv(hydroelastic.mesh, compute_device)
    hydroelastic.mesh.edges = wp.array(edges, dtype=wp.vec2i, device=compute_device)

    # Compute AABB.
    num_tets = hydroelastic.mesh.elements_count
    aabb_low = wp.zeros(num_tets, dtype=wp.vec3f, device=compute_device)
    aabb_high = wp.zeros(num_tets, dtype=wp.vec3f, device=compute_device)
    wp.launch(
        compute_aabb_elements,
        dim=num_tets,
        inputs=[
            hydroelastic.mesh.default_points,
            hydroelastic.mesh.indices,
            hydroelastic.mesh.elements_stride,
        ],
        outputs=[
            aabb_low,
            aabb_high,
        ],
    )
    # Initialize BVH.
    hydroelastic.bvh = wp.Bvh(aabb_low, aabb_high)

    # Initialize is_on_surface array.
    # This initiaization ensures that the field value is computed for every vertex.
    # This might generate small field values for the vertices on the surface.
    # TODO: Consider using the distances to the surface together with a margin to evaluate
    # if a vertex is on the surface.
    hydroelastic.mesh.is_on_surface = wp.zeros(V.shape[0], dtype=wp.bool, device=compute_device)

    # Initialize surface mesh.
    surface_faces = extract_surface_mesh(points_transformed.numpy(), elements)
    flat_surface_faces = np.array(surface_faces).flatten()
    indices = wp.array(flat_surface_faces, dtype=wp.int32, device=compute_device)
    hydroelastic.surface_mesh = wp.Mesh(points=points_transformed, indices=indices)

    # TODO: We could also loop over suraface_faces_np to set is_on_surface.

    # Initialize field values.
    init_field_by_distance_to_surface(hydroelastic, compute_device)

    return hydroelastic


def initialize_is_on_surface(num_vertices, compute_device):
    # This assumes that all vertexes are on the surface with exception of the centroid,
    # which is the last vertex.
    is_on_surface = np.ones(num_vertices, dtype=np.bool)
    is_on_surface[-1] = False
    is_on_surface = wp.array(is_on_surface, dtype=wp.bool, device=compute_device)
    return is_on_surface


def generate_equilateral_pyramid(Tf, scale, compute_device):
    # Create an equilateral pyramid.
    v_o = wp.vec3f(1.0, 0.0, 0.0)
    v_p = wp.quat_rotate(wp.quat_from_axis_angle(wp.vec3f(0.0, 1.0, 0.0), 2.0 * wp.pi / 3.0), v_o)
    v_q = wp.quat_rotate(wp.quat_from_axis_angle(wp.vec3f(0.0, 1.0, 0.0), 4.0 * wp.pi / 3.0), v_o)
    v_r = wp.vec3f(0.0, 1.0, 0.0)
    points = scale * wp.array([v_o, v_p, v_q, v_r], dtype=wp.vec3f, device=compute_device)

    points_transformed = wp.zeros(points.shape[0], dtype=wp.vec3f, device=compute_device)
    wp.launch(
        transform_array,
        dim=points.shape[0],
        inputs=[points, Tf],
        outputs=[points_transformed],
    )

    # Writing the faces as an array first for clarity.
    F = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]).flatten()
    indices = wp.array(F, dtype=wp.int32, device=compute_device)
    return points_transformed, indices


def generate_mesh(V, F, params, compute_device=None):
    if compute_device is None:
        compute_device = wp.get_device()

    points = wp.array(V, dtype=wp.vec3f, device=compute_device)
    indices = wp.array(F, dtype=wp.vec3i, device=compute_device)

    if "Tf" in params:
        wp.launch(
            transform_array,
            dim=points.shape[0],
            inputs=[points, params["Tf"]],
            outputs=[points],
        )

    hydroelastic = HydroelasticObject()
    # TODO check that tetrahedralize is working correctly with meshes that are not centered at the origin.
    # Initialize volume mesh.
    (
        hydroelastic.mesh.default_points,
        hydroelastic.mesh.indices,
        hydroelastic.mesh.edges,
    ) = tetrahedralize(points, indices, compute_device)

    hydroelastic.mesh.elements_stride = 4
    hydroelastic.mesh.elements_count = int(hydroelastic.mesh.indices.shape[0] / 4)

    initialize_default_tet_transform_inv(hydroelastic.mesh, compute_device)

    # Compute AABB.
    num_tets = hydroelastic.mesh.elements_count
    aabb_low = wp.zeros(num_tets, dtype=wp.vec3f, device=compute_device)
    aabb_high = wp.zeros(num_tets, dtype=wp.vec3f, device=compute_device)
    wp.launch(
        compute_aabb_elements,
        dim=num_tets,
        inputs=[
            hydroelastic.mesh.default_points,
            hydroelastic.mesh.indices,
            hydroelastic.mesh.elements_stride,
        ],
        outputs=[
            aabb_low,
            aabb_high,
        ],
    )
    # Initialize BVH.
    hydroelastic.bvh = wp.Bvh(aabb_low, aabb_high)

    # Check that the tetrahedron orientation is correct.
    # For now, we only print warnings.
    points_np = hydroelastic.mesh.default_points.numpy()
    indices_np = hydroelastic.mesh.indices.numpy()
    check_tetrahedron_orientation(indices_np, points_np)

    num_vertices = hydroelastic.mesh.default_points.shape[0]
    hydroelastic.mesh.is_on_surface = initialize_is_on_surface(num_vertices, compute_device)
    hydroelastic.hydroelastic_modulus = wp.float32(params["hydroelastic_modulus"])
    # Initialize surface mesh.
    flat_indices_np = indices.numpy().flatten()
    flat_indices_wp = wp.array(flat_indices_np, dtype=wp.int32, device=compute_device)
    hydroelastic.surface_mesh = wp.Mesh(points=hydroelastic.mesh.default_points, indices=flat_indices_wp)
    # Initialize field values.
    init_field_by_distance_to_surface(hydroelastic, compute_device)
    return hydroelastic


def generate_hard_mesh(V, F, params, compute_device=None):
    if compute_device is None:
        compute_device = wp.get_device()

    hydroelastic = HydroelasticObject()
    hydroelastic.is_soft = False
    hydroelastic.hydroelastic_modulus = wp.float32(params["hydroelastic_modulus"])
    # Initialize surface mesh.
    points = wp.array(V, dtype=wp.vec3f, device=compute_device)
    flat_indices_wp = wp.array(F.flatten(), dtype=wp.int32, device=compute_device)
    if "Tf" in params:
        wp.launch(
            transform_array,
            dim=points.shape[0],
            inputs=[points, params["Tf"]],
            outputs=[points],
        )
    elements_count = F.shape[0]
    hydroelastic.mesh.default_points = points
    hydroelastic.mesh.indices = flat_indices_wp
    hydroelastic.mesh.elements_count = elements_count
    hydroelastic.mesh.elements_stride = 3

    # Compute face normals.
    hydroelastic.mesh.normals = wp.zeros(elements_count, dtype=wp.vec3f, device=compute_device)
    wp.launch(
        compute_face_normals,
        dim=elements_count,
        inputs=[points, flat_indices_wp],
        outputs=[hydroelastic.mesh.normals],
    )
    # TODO: Print warning if mesh is not watertight.

    # Compute AABB.
    aabb_low = wp.zeros(elements_count, dtype=wp.vec3f, device=compute_device)
    aabb_high = wp.zeros(elements_count, dtype=wp.vec3f, device=compute_device)
    wp.launch(
        compute_aabb_elements,
        dim=elements_count,
        inputs=[
            hydroelastic.mesh.default_points,
            hydroelastic.mesh.indices,
            hydroelastic.mesh.elements_stride,
        ],
        outputs=[
            aabb_low,
            aabb_high,
        ],
    )
    # Initialize BVH.
    hydroelastic.bvh = wp.Bvh(aabb_low, aabb_high)

    hydroelastic.surface_mesh = wp.Mesh(points=points, indices=flat_indices_wp)
    return hydroelastic


def generate_mesh_from_obj(mesh_path, params, compute_device):
    # Load the mesh
    import trimesh  # noqa: PLC0415

    mesh = trimesh.load(mesh_path)
    V = params["scale"] * np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    # Generate the mesh
    return generate_mesh(V, F, params, compute_device)


def compute_field_gradient_after_init(hydroelastic: HydroelasticObject, compute_device):
    """
    Compute the pressure field gradient after field initialization.
    """
    num_tetrahedra = hydroelastic.mesh.elements_count

    # Initialize gradient array
    hydroelastic.mesh.field_gradient = wp.zeros(num_tetrahedra, dtype=wp.vec3f, device=compute_device)

    # Compute gradients for each tetrahedron
    wp.launch(
        compute_field_gradient,
        dim=num_tetrahedra,
        inputs=[
            hydroelastic.mesh.default_points,
            hydroelastic.mesh.field,
            hydroelastic.mesh.indices,
        ],
        outputs=[hydroelastic.mesh.field_gradient],
    )


def init_field_by_distance_to_surface(hydroelastic: HydroelasticObject, compute_device):
    # Compute centroid and overestimate the max distance to the surface.
    V_np = hydroelastic.mesh.default_points.numpy()
    centroid_np = np.mean(V_np, axis=0)
    distances = np.linalg.norm(V_np - centroid_np, axis=1)
    estimated_max_distance = wp.float32(np.max(distances))
    # print("estimated_max_distance", estimated_max_distance)

    # Compute distance to surface for each vertex.
    num_vertices = hydroelastic.mesh.default_points.shape[0]
    distance_to_surface = wp.zeros(num_vertices, dtype=wp.float32, device=compute_device)
    wp.launch(
        compute_distance_to_surface,
        dim=num_vertices,
        inputs=[hydroelastic.mesh.default_points, hydroelastic.surface_mesh.id, estimated_max_distance],
        outputs=[distance_to_surface],
    )

    # Compute the max distance to the surface.
    max_distance = wp.array([0.0], dtype=wp.float32, device=compute_device)
    wp.launch(
        max_of_array,
        dim=distance_to_surface.shape[0],
        inputs=[distance_to_surface],
        outputs=[max_distance],
    )
    # print("max_distance", max_distance)

    # Initialize field values.
    # This would need to be recomputed if the hydroelastic modulues changes on the fly.
    hydroelastic.mesh.field = wp.zeros(num_vertices, dtype=wp.float32, device=compute_device)
    wp.launch(
        init_convex_field,
        dim=num_vertices,
        inputs=[
            hydroelastic.mesh.default_points,
            distance_to_surface,
            hydroelastic.mesh.is_on_surface,
            max_distance,
            hydroelastic.hydroelastic_modulus,
            hydroelastic.margin,
        ],
        outputs=[hydroelastic.mesh.field],
    )

    # Compute field gradient after field initialization
    compute_field_gradient_after_init(hydroelastic, compute_device)


def add_bodies(builder, init_poses, meshes):
    for init_pose, mesh in zip(init_poses, meshes, strict=False):
        # Add body
        body_id = builder.add_body(
            xform=wp.transform(wp.vec3(init_pose[0:3]), wp.quatf(init_pose[3:7])),
            mass=float(mesh.lumped_properties.mass),
            I_m=mesh.lumped_properties.inertia,
            # I_m=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )

        np_points = mesh.surface_mesh.points.numpy()
        np_indices = mesh.surface_mesh.indices.numpy()
        newton_mesh = newton.Mesh(vertices=np_points, indices=np_indices)

        if mesh.compute_mesh_density:
            from newton._src.geometry.inertia import compute_mesh_inertia  # noqa: PLC0415

            s_, com_, I, vol = compute_mesh_inertia(1.0, np_points, np_indices, is_solid=True)
            mesh.density = mesh.mass / vol
            # print("I", I)

        # Add shape to body
        builder.add_shape_mesh(
            body=body_id,
            mesh=newton_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            cfg=newton.ModelBuilder.ShapeConfig(is_visible=mesh.is_visible, density=mesh.density),
        )


def add_robot(builder, up_axis):
    articulation_builder = newton.ModelBuilder(up_axis=up_axis)

    asset_path = newton.utils.download_asset("franka_description")

    rot_1 = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), -0.5 * wp.pi)
    rot_2 = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), -0.5 * wp.pi)
    T1 = wp.transform(wp.vec3f(0.0, 0.0, 0.0), rot_1)
    T2 = wp.transform(wp.vec3f(0.0, 0.0, 0.0), rot_2)
    Tf = wp.transform_multiply(T1, T2)
    q = wp.transform_get_rotation(Tf)

    newton.utils.parse_urdf(
        str(asset_path / "urdfs" / "fr3_franka_hand.urdf"),
        articulation_builder,
        up_axis=up_axis,
        xform=wp.transform(
            # (-50, -20, 50),
            (-0.0, -0.1, -0.5),
            q,
            # wp.quat_identity(),
        ),
        floating=False,
        scale=1,  # unit: cm
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )
    articulation_builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
    articulation_builder.joint_q[-1] = 0.05
    articulation_builder.joint_q[-2] = 0.05

    xform = wp.transform(wp.vec3(0), wp.quat_identity())
    builder.add_builder(articulation_builder, xform, separate_collision_group=False)


def init_hydro_batch(hydro_objects, hydro_batch, body_q):
    # mesh_batch = HydroelasticBatch()

    # This assumes that each mesh is associated with a different body.
    meshes_count = len(hydro_objects)

    max_points_count = 0
    max_indices_count = 0
    max_elements_count = 0
    for object in hydro_objects:
        max_points_count = max(max_points_count, int(object.mesh.default_points.shape[0]))
        max_indices_count = max(max_indices_count, int(object.mesh.indices.shape[0]))
        max_elements_count = max(max_elements_count, int(object.mesh.elements_count))

    default_points = np.zeros((meshes_count, max_points_count, 3))
    indices = np.zeros((meshes_count, max_indices_count), dtype=np.int32)
    elements_count = np.zeros((meshes_count), dtype=np.int32)
    elements_stride = np.zeros((meshes_count), dtype=np.int32)

    is_soft = np.zeros((meshes_count), dtype=np.bool)

    # Per-vertex fields.
    field = np.zeros((meshes_count, max_points_count), dtype=np.float32)
    # Per-element fields gradient.
    field_gradient = np.zeros((meshes_count, max_elements_count, 3), dtype=np.float32)
    default_tet_transform_inv = np.zeros((meshes_count, max_elements_count, 4, 4), dtype=np.float32)

    normals = np.zeros((meshes_count, max_elements_count, 3), dtype=np.float32)

    h = np.zeros((meshes_count), dtype=np.float32)
    d = np.zeros((meshes_count), dtype=np.float32)
    mu_static = np.zeros((meshes_count), dtype=np.float32)
    mu_dynamic = np.zeros((meshes_count), dtype=np.float32)

    bvh_ids = np.zeros((meshes_count), dtype=np.uint64)

    for i, object in enumerate(hydro_objects):
        points_count = int(object.mesh.default_points.shape[0])
        indices_count = int(object.mesh.indices.shape[0])
        el_count = int(object.mesh.elements_count)

        default_points[i, :points_count, :] = object.mesh.default_points.numpy()
        indices[i, :indices_count] = object.mesh.indices.numpy().flatten()
        elements_count[i] = el_count
        elements_stride[i] = object.mesh.elements_stride

        is_soft[i] = object.is_soft

        if object.is_soft:
            field[i, :points_count] = object.mesh.field.numpy()
            field_gradient[i, :el_count, :] = object.mesh.field_gradient.numpy()
            default_tet_transform_inv[i, :el_count, :, :] = object.mesh.default_tet_transform_inv.numpy()

        if not object.is_soft:
            normals[i, :el_count, :] = object.mesh.normals.numpy()

        h[i] = object.hydroelastic_modulus
        d[i] = object.hunt_crossley_dissipation
        mu_static[i] = object.mu_static
        mu_dynamic[i] = object.mu_dynamic

        bvh_ids[i] = object.bvh.id

    # Initialize the mesh batch.
    hydro_batch.max_elements_count = wp.int32(max_elements_count)
    hydro_batch.default_points = wp.array(default_points, dtype=wp.vec3f)
    hydro_batch.indices = wp.array(indices, dtype=wp.int32)
    hydro_batch.elements_count = wp.array(elements_count, dtype=wp.int32)
    hydro_batch.elements_stride = wp.array(elements_stride, dtype=wp.int32)

    hydro_batch.is_soft = wp.array(is_soft, dtype=wp.bool)

    hydro_batch.field = wp.array(field, dtype=wp.float32)
    hydro_batch.field_gradient = wp.array(field_gradient, dtype=wp.vec3f)
    hydro_batch.default_tet_transform_inv = wp.array(default_tet_transform_inv, dtype=wp.mat44)

    hydro_batch.normals = wp.array(normals, dtype=wp.vec3f)

    hydro_batch.h = wp.array(h, dtype=wp.float32)
    hydro_batch.d = wp.array(d, dtype=wp.float32)
    hydro_batch.mu_static = wp.array(mu_static, dtype=wp.float32)
    hydro_batch.mu_dynamic = wp.array(mu_dynamic, dtype=wp.float32)

    hydro_batch.bvh_ids = wp.array(bvh_ids, dtype=wp.uint64)

    hydro_batch.body_q_inv_mat = wp.zeros(body_q.shape[0], dtype=wp.mat44)


def init_isosurface_batch(isosurface_batch, collision_pairs, hydro_batch, max_element_pairs=-1):
    isosurface_count = len(collision_pairs)
    # ================================================================================================================
    # Find the maximum number of element pairs.
    if max_element_pairs == -1:
        max_element_pairs = 0
        element_counts = hydro_batch.elements_count.numpy()
        for c in collision_pairs:
            body_a = c[0]
            body_b = c[1]
            num_elements_a = element_counts[body_a]
            num_elements_b = element_counts[body_b]
            element_pairs_count = num_elements_a * num_elements_b
            max_element_pairs = max(max_element_pairs, element_pairs_count)
            pair_label = f"body_a: {body_a}, body_b: {body_b}"
            pair_info = f"List of element pairs N: {element_pairs_count}. num_elements_a: {num_elements_a}, num_elements_b: {num_elements_b}"
            print(f"{pair_label} -> {pair_info}")
    print(f"max_element_pairs: {max_element_pairs}")

    # ================================================================================================================
    # Initialize the isosurface batch.
    isosurface_batch.max_element_pairs = wp.int32(max_element_pairs)
    isosurface_batch.element_pairs = wp.zeros((isosurface_count, max_element_pairs), dtype=wp.vec2i)
    isosurface_batch.body_a_idx = wp.zeros(isosurface_count, dtype=wp.int32)
    isosurface_batch.body_b_idx = wp.zeros(isosurface_count, dtype=wp.int32)

    isosurface_batch.v_counts = wp.zeros((isosurface_count, max_element_pairs), dtype=wp.int32)
    isosurface_batch.vertices = wp.zeros((isosurface_count, 8 * max_element_pairs), dtype=wp.vec3f)
    isosurface_batch.centroids = wp.zeros((isosurface_count, max_element_pairs), dtype=wp.vec3f)
    isosurface_batch.cartesian_to_penetration = wp.zeros((isosurface_count, max_element_pairs), dtype=wp.vec4f)
    isosurface_batch.normals = wp.zeros((isosurface_count, max_element_pairs), dtype=wp.vec3f)
    isosurface_batch.centroid_pressure = wp.zeros((isosurface_count, max_element_pairs), dtype=wp.float32)

    isosurface_batch.h_combined = wp.zeros(isosurface_count, dtype=wp.float32)
    isosurface_batch.d_combined = wp.zeros(isosurface_count, dtype=wp.float32)
    isosurface_batch.mu_static_combined = wp.zeros(isosurface_count, dtype=wp.float32)
    isosurface_batch.mu_dynamic_combined = wp.zeros(isosurface_count, dtype=wp.float32)
    isosurface_batch.query_with_mesh_a = wp.zeros(isosurface_count, dtype=wp.bool)
    isosurface_batch.soft_vs_soft = wp.zeros(isosurface_count, dtype=wp.bool)

    isosurface_batch.force = wp.zeros((isosurface_count), dtype=wp.vec3f)
    isosurface_batch.torque_a = wp.zeros((isosurface_count), dtype=wp.vec3f)
    isosurface_batch.torque_b = wp.zeros((isosurface_count), dtype=wp.vec3f)
    isosurface_batch.torque_a_body = wp.zeros((isosurface_count), dtype=wp.vec3f)
    isosurface_batch.torque_b_body = wp.zeros((isosurface_count), dtype=wp.vec3f)
    isosurface_batch.force_n = wp.zeros((isosurface_count), dtype=wp.vec3f)
    isosurface_batch.force_t = wp.zeros((isosurface_count), dtype=wp.vec3f)
    # ================================================================================================================
    # TODO: Collision pairs should represent mesh ids, not body ids.
    # For now, it works as each mesh is associated with a different body.
    body_a_idx = np.zeros((isosurface_count), dtype=np.int32)
    body_b_idx = np.zeros((isosurface_count), dtype=np.int32)

    for i, c in enumerate(collision_pairs):
        # TODO: This should be a better way to get body ides
        # body_a = meshes[c[0]].mesh.body_id
        body_a = c[0]
        body_b = c[1]
        body_a_idx[i] = body_a
        body_b_idx[i] = body_b

    isosurface_batch.body_a_idx.assign(wp.array(body_a_idx, dtype=wp.int32))
    isosurface_batch.body_b_idx.assign(wp.array(body_b_idx, dtype=wp.int32))

    wp.launch(
        combine_material_properties,
        dim=isosurface_count,
        inputs=[
            isosurface_batch.body_a_idx,
            isosurface_batch.body_b_idx,
            hydro_batch.h,
            hydro_batch.d,
            hydro_batch.mu_static,
            hydro_batch.mu_dynamic,
        ],
        outputs=[
            isosurface_batch.h_combined,
            isosurface_batch.d_combined,
            isosurface_batch.mu_static_combined,
            isosurface_batch.mu_dynamic_combined,
        ],
    )

    wp.launch(
        set_query_with_mesh_a,
        dim=isosurface_count,
        inputs=[
            hydro_batch.elements_count,
            isosurface_batch.body_a_idx,
            isosurface_batch.body_b_idx,
        ],
        outputs=[isosurface_batch.query_with_mesh_a],
    )

    wp.launch(
        set_soft_vs_soft,
        dim=isosurface_count,
        inputs=[
            hydro_batch.is_soft,
            isosurface_batch.body_a_idx,
            isosurface_batch.body_b_idx,
        ],
        outputs=[isosurface_batch.soft_vs_soft],
    )

    # Quadrature points and weights.
    isosurface_batch.quadrature_weights = wp.array([1.0], dtype=wp.float32)
    isosurface_batch.quadrature_coords = wp.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=wp.vec3f)

    # # Second order quadrature rule for triangles.
    # isosurface_batch.quadrature_weights = wp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=wp.float32)
    # isosurface_batch.quadrature_coords = wp.array(
    #     [[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]],
    #     dtype=wp.vec3f,
    # )
