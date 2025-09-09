import numpy as np
import warp as wp

import newton
from newton._src.hydroelastic.make_fields import (
    compute_distance_to_surface,
    compute_field_gradient,
    init_convex_field,
    max_of_array,
    simple_field_intialization,
)
from newton._src.hydroelastic.tetrahedralizer import tetrahedralize
from newton._src.hydroelastic.types import HydroelasticMesh, Isosurface
from newton._src.hydroelastic.utils import (
    compute_aabb_tets,
    compute_aabb_tris,
    compute_default_tet_transform_inv,
    compute_face_normals,
    transform_array,
)


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
    for i in range(elements.shape[0]):
        signed_volume = tet_vol_signed(elements[i], V)
        if signed_volume < 0.0:
            # TODO: Swap elements in case of incorrect orientation.
            print(
                f"Warning in function load_drake_mesh. Incorrect tetrahedron orientation. signed_volume: {signed_volume}, element: {elements[i]}"
            )


def initialize_default_tet_transform_inv(volume_mesh, compute_device):
    num_tets = volume_mesh.indices.shape[0]
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
    check_tetrahedron_orientation(elements, V)

    points = wp.array(V, dtype=wp.vec3f, device=compute_device)
    # indices = wp.array(F, dtype=wp.vec3i, device=compute_device)
    points_transformed = wp.zeros(points.shape[0], dtype=wp.vec3f, device=compute_device)
    wp.launch(
        transform_array,
        dim=points.shape[0],
        inputs=[points, Tf],
        outputs=[points_transformed],
    )

    hydroelastic = HydroelasticMesh()
    hydroelastic.is_visible = params["is_visible"]
    hydroelastic.hydroelastic_modulus = wp.float32(params["hydroelastic_modulus"])
    # Initialize volume mesh.
    hydroelastic.volume_mesh.default_points = points_transformed
    hydroelastic.volume_mesh.indices = wp.array(elements, dtype=wp.vec4i, device=compute_device)
    initialize_default_tet_transform_inv(hydroelastic.volume_mesh, compute_device)
    hydroelastic.volume_mesh.edges = wp.array(edges, dtype=wp.vec2i, device=compute_device)

    # Compute AABB.
    num_tets = hydroelastic.volume_mesh.indices.shape[0]
    hydroelastic.aabb_low = wp.zeros(num_tets, dtype=wp.vec3f, device=compute_device)
    hydroelastic.aabb_high = wp.zeros(num_tets, dtype=wp.vec3f, device=compute_device)
    temp_body_q = wp.array(wp.transform_identity(), dtype=wp.transform, device=compute_device)
    body_idx = wp.int32(0)
    wp.launch(
        compute_aabb_tets,
        dim=num_tets,
        inputs=[
            temp_body_q,
            body_idx,
            hydroelastic.volume_mesh.default_points,
            hydroelastic.volume_mesh.indices,
        ],
        outputs=[
            hydroelastic.aabb_low,
            hydroelastic.aabb_high,
        ],
    )
    # Initialize BVH.
    hydroelastic.bvh = wp.Bvh(hydroelastic.aabb_low, hydroelastic.aabb_high)

    # Initialize is_on_surface array.
    # This initiaization ensures that the field value is computed for every vertex.
    # This might generate small field values for the vertices on the surface.
    # TODO: Consider using the distances to the surface together with a margin to evaluate
    # if a vertex is on the surface.
    hydroelastic.volume_mesh.is_on_surface = wp.zeros(V.shape[0], dtype=wp.bool, device=compute_device)

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


def generate_test_tetrahedron(Tf, params, compute_device):
    points, indices = generate_equilateral_pyramid(Tf, scale=params["scale"], compute_device=compute_device)
    num_vertices = points.shape[0]

    hydroelastic = HydroelasticMesh()
    hydroelastic.volume_mesh.default_points = points
    hydroelastic.volume_mesh.indices = wp.array([[0, 1, 2, 3]], dtype=wp.vec4i, device=compute_device)
    hydroelastic.volume_mesh.edges = wp.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=wp.vec2i)
    hydroelastic.volume_mesh.is_on_surface = initialize_is_on_surface(num_vertices, compute_device)
    hydroelastic.hydroelastic_modulus = wp.float32(params["hydroelastic_modulus"])

    hydroelastic.surface_mesh = wp.Mesh(points=hydroelastic.volume_mesh.default_points, indices=indices)

    # Initialize field values.
    hydroelastic.volume_mesh.field = wp.zeros(num_vertices, dtype=wp.float32, device=compute_device)
    wp.launch(
        simple_field_intialization,
        dim=hydroelastic.volume_mesh.field.shape[0],
        outputs=[hydroelastic.volume_mesh.field],
    )

    # Compute field gradient after field initialization
    compute_field_gradient_after_init(hydroelastic, compute_device)

    return hydroelastic


def generate_tetrahedralized_pyramid(Tf, params, compute_device):
    points, indices = generate_equilateral_pyramid(Tf, scale=params["scale"], compute_device=compute_device)
    indices = wp.array(indices, dtype=wp.vec3i, device=compute_device)
    hydroelastic = HydroelasticMesh()
    # TODO check that tetrahedralize is working correctly with meshes that are not centered at the origin.
    # Initialize volume mesh.
    (
        hydroelastic.volume_mesh.default_points,
        hydroelastic.volume_mesh.indices,
        hydroelastic.volume_mesh.edges,
    ) = tetrahedralize(surface_vertices=points, surface_faces=indices, device=compute_device)

    num_vertices = hydroelastic.volume_mesh.default_points.shape[0]
    hydroelastic.volume_mesh.is_on_surface = initialize_is_on_surface(num_vertices, compute_device)
    hydroelastic.hydroelastic_modulus = wp.float32(params["hydroelastic_modulus"])
    # Initialize surface mesh.
    flat_indices_np = indices.numpy().flatten()
    flat_indices_wp = wp.array(flat_indices_np, dtype=wp.int32, device=compute_device)
    hydroelastic.surface_mesh = wp.Mesh(points=hydroelastic.volume_mesh.default_points, indices=flat_indices_wp)
    # Initialize field values.
    init_field_by_distance_to_surface(hydroelastic, compute_device)
    return hydroelastic


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

    hydroelastic = HydroelasticMesh()
    # TODO check that tetrahedralize is working correctly with meshes that are not centered at the origin.
    # Initialize volume mesh.
    (
        hydroelastic.volume_mesh.default_points,
        hydroelastic.volume_mesh.indices,
        hydroelastic.volume_mesh.edges,
    ) = tetrahedralize(points, indices, compute_device)

    initialize_default_tet_transform_inv(hydroelastic.volume_mesh, compute_device)

    # Compute AABB.
    num_tets = hydroelastic.volume_mesh.indices.shape[0]
    hydroelastic.aabb_low = wp.zeros(num_tets, dtype=wp.vec3f, device=compute_device)
    hydroelastic.aabb_high = wp.zeros(num_tets, dtype=wp.vec3f, device=compute_device)
    temp_body_q = wp.array(wp.transform_identity(), dtype=wp.transform, device=compute_device)
    body_idx = wp.int32(0)
    wp.launch(
        compute_aabb_tets,
        dim=num_tets,
        inputs=[
            temp_body_q,
            body_idx,
            hydroelastic.volume_mesh.default_points,
            hydroelastic.volume_mesh.indices,
        ],
        outputs=[
            hydroelastic.aabb_low,
            hydroelastic.aabb_high,
        ],
    )
    # Initialize BVH.
    hydroelastic.bvh = wp.Bvh(hydroelastic.aabb_low, hydroelastic.aabb_high)

    # Check that the tetrahedron orientation is correct.
    # For now, we only print warnings.
    points_np = hydroelastic.volume_mesh.default_points.numpy()
    indices_np = hydroelastic.volume_mesh.indices.numpy()
    check_tetrahedron_orientation(indices_np, points_np)

    num_vertices = hydroelastic.volume_mesh.default_points.shape[0]
    hydroelastic.volume_mesh.is_on_surface = initialize_is_on_surface(num_vertices, compute_device)
    hydroelastic.hydroelastic_modulus = wp.float32(params["hydroelastic_modulus"])
    # Initialize surface mesh.
    flat_indices_np = indices.numpy().flatten()
    flat_indices_wp = wp.array(flat_indices_np, dtype=wp.int32, device=compute_device)
    hydroelastic.surface_mesh = wp.Mesh(points=hydroelastic.volume_mesh.default_points, indices=flat_indices_wp)
    # Initialize field values.
    init_field_by_distance_to_surface(hydroelastic, compute_device)
    return hydroelastic


def generate_hard_mesh(V, F, params, compute_device=None):
    if compute_device is None:
        compute_device = wp.get_device()

    hydroelastic = HydroelasticMesh()
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
    hydroelastic.surface_mesh = wp.Mesh(points=points, indices=flat_indices_wp)
    # Compute face normals.
    faces_count = F.shape[0]
    hydroelastic.surface_mesh.normals = wp.zeros(faces_count, dtype=wp.vec3f, device=compute_device)
    wp.launch(
        compute_face_normals,
        dim=faces_count,
        inputs=[points, flat_indices_wp],
        outputs=[hydroelastic.surface_mesh.normals],
    )
    # TODO: Print warning if mesh is not watertight.

    # Compute AABB.
    num_tris = F.shape[0]
    hydroelastic.aabb_low = wp.zeros(num_tris, dtype=wp.vec3f, device=compute_device)
    hydroelastic.aabb_high = wp.zeros(num_tris, dtype=wp.vec3f, device=compute_device)
    temp_body_q = wp.array(wp.transform_identity(), dtype=wp.transform, device=compute_device)
    body_idx = wp.int32(0)
    wp.launch(
        compute_aabb_tris,
        dim=num_tris,
        inputs=[
            temp_body_q,
            body_idx,
            hydroelastic.surface_mesh.points,
            hydroelastic.surface_mesh.indices,
        ],
        outputs=[
            hydroelastic.aabb_low,
            hydroelastic.aabb_high,
        ],
    )
    # Initialize BVH.
    hydroelastic.bvh = wp.Bvh(hydroelastic.aabb_low, hydroelastic.aabb_high)
    return hydroelastic


def generate_mesh_from_obj(mesh_path, params, compute_device):
    # Load the mesh
    import trimesh  # noqa: PLC0415

    mesh = trimesh.load(mesh_path)
    V = params["scale"] * np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    # Generate the mesh
    return generate_mesh(V, F, params, compute_device)


def compute_field_gradient_after_init(hydroelastic: HydroelasticMesh, compute_device):
    """
    Compute the pressure field gradient after field initialization.
    """
    num_tetrahedra = hydroelastic.volume_mesh.indices.shape[0]

    # Initialize gradient array
    hydroelastic.volume_mesh.field_gradient = wp.zeros(num_tetrahedra, dtype=wp.vec3f, device=compute_device)

    # Compute gradients for each tetrahedron
    wp.launch(
        compute_field_gradient,
        dim=num_tetrahedra,
        inputs=[
            hydroelastic.volume_mesh.default_points,
            hydroelastic.volume_mesh.field,
            hydroelastic.volume_mesh.indices,
        ],
        outputs=[hydroelastic.volume_mesh.field_gradient],
    )


def init_field_by_distance_to_surface(hydroelastic: HydroelasticMesh, compute_device):
    # Compute centroid and overestimate the max distance to the surface.
    V_np = hydroelastic.volume_mesh.default_points.numpy()
    centroid_np = np.mean(V_np, axis=0)
    distances = np.linalg.norm(V_np - centroid_np, axis=1)
    estimated_max_distance = wp.float32(np.max(distances))
    # print("estimated_max_distance", estimated_max_distance)

    # Compute distance to surface for each vertex.
    num_vertices = hydroelastic.volume_mesh.default_points.shape[0]
    distance_to_surface = wp.zeros(num_vertices, dtype=wp.float32, device=compute_device)
    wp.launch(
        compute_distance_to_surface,
        dim=num_vertices,
        inputs=[hydroelastic.volume_mesh.default_points, hydroelastic.surface_mesh.id, estimated_max_distance],
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
    hydroelastic.volume_mesh.field = wp.zeros(num_vertices, dtype=wp.float32, device=compute_device)
    wp.launch(
        init_convex_field,
        dim=num_vertices,
        inputs=[
            hydroelastic.volume_mesh.default_points,
            distance_to_surface,
            hydroelastic.volume_mesh.is_on_surface,
            max_distance,
            hydroelastic.hydroelastic_modulus,
            hydroelastic.margin,
        ],
        outputs=[hydroelastic.volume_mesh.field],
    )

    # Compute field gradient after field initialization
    compute_field_gradient_after_init(hydroelastic, compute_device)


def load_two_tets_vs_one_tet(meshes, scale, compute_device):
    # Create double tetrahedron.
    # ===============================================================================================
    # Define surface mesh
    Tf = wp.transform(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity())
    v_o = wp.vec3f(0.0, 0.0, 0.0)
    v_p = wp.vec3f(0.0, 0.0, 1.0)
    v_q = wp.vec3f(1.0, 0.0, 0.0)
    v_r = wp.vec3f(-1.0, 0.0, 0.0)
    v_s = wp.vec3f(0.0, 1.0, 0.0)
    points = scale * wp.array([v_o, v_p, v_q, v_r, v_s], dtype=wp.vec3f, device=compute_device)

    points_transformed = wp.zeros(points.shape[0], dtype=wp.vec3f, device=compute_device)
    wp.launch(
        transform_array,
        dim=points.shape[0],
        inputs=[points, Tf],
        outputs=[points_transformed],
    )

    # Write the faces as an array first for clarity.
    F = np.array(
        [
            [0, 4, 2],
            [0, 2, 1],
            [1, 2, 4],
            [0, 3, 4],
            [0, 1, 3],
            [1, 4, 3],
        ]
    ).flatten()
    indices = wp.array(F, dtype=wp.int32, device=compute_device)

    # ===============================================================================================
    # Define hydroelastic mesh.
    num_vertices = points_transformed.shape[0]
    hydroelastic = HydroelasticMesh()
    # Define volume mesh.
    hydroelastic.volume_mesh.default_points = points_transformed
    hydroelastic.volume_mesh.indices = wp.array(
        [
            [0, 1, 2, 4],
            [0, 3, 1, 4],
        ],
        dtype=wp.vec4i,
        device=compute_device,
    )
    hydroelastic.volume_mesh.edges = wp.array(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 2],
            [1, 4],
            [2, 4],
            [0, 3],
            [0, 1],
            [0, 4],
            [1, 3],
            [1, 4],
            [3, 4],
        ],
        dtype=wp.vec2i,
    )
    hydroelastic.volume_mesh.is_on_surface = initialize_is_on_surface(num_vertices, compute_device)
    hydroelastic.hydroelastic_modulus = wp.float32(1e5)
    # Define surface mesh.
    hydroelastic.surface_mesh = wp.Mesh(points=hydroelastic.volume_mesh.default_points, indices=indices)
    # Initialize field values.
    hydroelastic.volume_mesh.field = wp.zeros(num_vertices, dtype=wp.float32, device=compute_device)
    wp.launch(
        simple_field_intialization,
        dim=hydroelastic.volume_mesh.field.shape[0],
        outputs=[hydroelastic.volume_mesh.field],
    )

    # Compute field gradient after field initialization
    compute_field_gradient_after_init(hydroelastic, compute_device)

    meshes.append(hydroelastic)
    # ===============================================================================================
    # Create single tetrahedron.
    Tf = wp.transform(wp.vec3f(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), wp.pi))
    v_o = wp.vec3f(0.0, 0.0, 0.0)
    v_p = wp.vec3f(0.0, 0.0, 1.0)
    v_q = wp.vec3f(1.0, 0.0, 0.0)
    v_r = wp.vec3f(0.0, 1.0, 0.0)

    points = scale * wp.array([v_o, v_p, v_q, v_r], dtype=wp.vec3f, device=compute_device)

    points_transformed = wp.zeros(points.shape[0], dtype=wp.vec3f, device=compute_device)
    wp.launch(
        transform_array,
        dim=points.shape[0],
        inputs=[points, Tf],
        outputs=[points_transformed],
    )

    # Write the faces as an array first for clarity.
    F = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 3],
            [1, 2, 3],
        ]
    ).flatten()
    indices = wp.array(F, dtype=wp.int32, device=compute_device)
    # ===============================================================================================
    # Define hydroelastic mesh.
    num_vertices = points_transformed.shape[0]
    hydroelastic = HydroelasticMesh()
    # Define volume mesh.
    hydroelastic.volume_mesh.default_points = points_transformed
    hydroelastic.volume_mesh.indices = wp.array(
        [
            [0, 1, 2, 3],
        ],
        dtype=wp.vec4i,
        device=compute_device,
    )
    hydroelastic.volume_mesh.edges = wp.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=wp.vec2i,
    )
    hydroelastic.volume_mesh.is_on_surface = initialize_is_on_surface(num_vertices, compute_device)
    hydroelastic.hydroelastic_modulus = wp.float32(1e8)
    # Define surface mesh.
    hydroelastic.surface_mesh = wp.Mesh(points=hydroelastic.volume_mesh.default_points, indices=indices)
    # Initialize field values.
    hydroelastic.volume_mesh.field = wp.zeros(num_vertices, dtype=wp.float32, device=compute_device)
    wp.launch(
        simple_field_intialization,
        dim=hydroelastic.volume_mesh.field.shape[0],
        outputs=[hydroelastic.volume_mesh.field],
    )

    # Compute field gradient after field initialization
    compute_field_gradient_after_init(hydroelastic, compute_device)

    meshes.append(hydroelastic)


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


def init_isosurfaces(collision_pairs, isosurfaces, meshes, max_geom_pairs=-1, device=None):
    geom_pairs_count = 0
    if isinstance(max_geom_pairs, int):
        max_geom_pairs = [max_geom_pairs] * len(collision_pairs)
    for i, c in enumerate(collision_pairs):
        body_a = c[0]
        body_b = c[1]
        num_elements_a = meshes[body_a].volume_mesh.indices.shape[0]
        num_elements_b = 0
        if meshes[body_b].is_soft:
            num_elements_b = meshes[body_b].volume_mesh.indices.shape[0]
        else:
            # Surface mesh is a wp.Mesh, where the indices are a flat array of integers.
            num_elements_b = meshes[body_b].surface_mesh.indices.shape[0] // 3
        l = [(x, y) for x in range(num_elements_a) for y in range(num_elements_b)]
        isosurfaces.append(Isosurface(body_a, body_b, l, meshes[body_b].is_soft, max_geom_pairs[i], device))
        geom_pairs_count += len(l)
        print(
            f"{isosurfaces[-1].label} -> list of geom pairs N: {len(l)}. num_elements_a: {num_elements_a}, num_elements_b: {num_elements_b}"
        )
        print(
            f"meshes[body_a].aabb_low.shape: {meshes[body_a].aabb_low.shape}, meshes[body_b].aabb_low.shape: {meshes[body_b].aabb_low.shape}"
        )

    return len(isosurfaces)


def init_streams(num_streams, streams, compute_device):
    for _ in range(num_streams):
        # streams.append(wp.Stream(device=compute_device))
        streams.append(wp.get_stream())
