from enum import Enum

import numpy as np
import warp as wp

# declare a new 8x3 float matrix type:
vec4i = wp.types.vector(length=4, dtype=wp.int32)

hydroelastic_type = wp.float32
vec2h = wp.types.vector(length=2, dtype=hydroelastic_type)
vec3h = wp.types.vector(length=3, dtype=hydroelastic_type)
vec4h = wp.types.vector(length=4, dtype=hydroelastic_type)
mat33h = wp.types.matrix(shape=(3, 3), dtype=hydroelastic_type)
mat44h = wp.types.matrix(shape=(4, 4), dtype=hydroelastic_type)
mat43h = wp.types.matrix(shape=(4, 3), dtype=hydroelastic_type)


@wp.struct
class HydroelasticMesh:
    default_points: wp.array(dtype=wp.vec3f)
    indices: wp.array(dtype=wp.int32)
    elements_count: wp.int32
    elements_stride: wp.int32

    # Only used for soft meshes (Tetrahedral meshes).
    field: wp.array(dtype=wp.float32)
    field_gradient: wp.array(dtype=wp.vec3f)
    default_tet_transform_inv: wp.array(dtype=wp.mat44)

    # Only used for rigid meshes (Triangular meshes).
    normals: wp.array(dtype=wp.vec3f)

    # The folllowing fields are experimental or for debugging purposes.
    edges: wp.array(dtype=wp.vec2i)
    is_on_surface: wp.array(dtype=wp.bool)


@wp.struct
class LumpedProperties:
    mass: wp.float32
    inertia: wp.mat33


class HydroelasticObject:
    def __init__(self):
        self.mesh = HydroelasticMesh()
        self.bvh = None
        self.surface_mesh = None
        self.is_soft = True
        self.hydroelastic_modulus = wp.float32(1.0e3)
        self.mu_static = wp.float32(0.6)
        self.mu_dynamic = wp.float32(0.575)
        self.hunt_crossley_dissipation = wp.float32(3.0)
        self.margin = wp.float32(0.0)
        self.is_visible = True
        self.lumped_properties = LumpedProperties()
        self.compute_mesh_density = False
        self.mass = 0.0
        self.density = 1000.0
        self.body_id = -1
        self.update_aabb = True


@wp.struct
class ContactPolygon:
    vertex_counts: wp.array(dtype=wp.int32)
    vertices: wp.array(dtype=wp.vec3f)
    centroids: wp.array(dtype=wp.vec3f)
    # TODO: Find a better name for this.
    # The pressure is computed as:
    # homogeneous_R = [R; 1]
    # cartesian_to_penetration = field_values * inv(transformation_matrix)
    # penetration_extent = dot(cartesian_to_penetration, homogeneous_R)
    # p = hydroelastic_modulus * penetration_extent
    cartesian_to_penetration: wp.array(dtype=wp.vec4f)
    normals: wp.array(dtype=wp.vec3f)
    centroid_pressure: wp.array(dtype=wp.float32)


def initialize_contact_polygon(num_tet_pairs, contact_polygon, compute_device):
    contact_polygon.vertex_counts = wp.zeros(num_tet_pairs, dtype=wp.int32, device=compute_device)
    contact_polygon.vertices = wp.zeros(8 * num_tet_pairs, dtype=wp.vec3f, device=compute_device)
    contact_polygon.centroids = wp.zeros(num_tet_pairs, dtype=wp.vec3f, device=compute_device)
    contact_polygon.cartesian_to_penetration = wp.zeros(num_tet_pairs, dtype=wp.vec4f, device=compute_device)
    contact_polygon.normals = wp.zeros(num_tet_pairs, dtype=wp.vec3f, device=compute_device)
    contact_polygon.centroid_pressure = wp.zeros(num_tet_pairs, dtype=wp.float32, device=compute_device)


def reset_contact_polygon(contact_polygon):
    contact_polygon.vertex_counts.fill_(0)
    contact_polygon.vertices.fill_(0.0)
    contact_polygon.centroids.fill_(0.0)
    contact_polygon.cartesian_to_penetration.fill_(0.0)
    contact_polygon.normals.fill_(0.0)
    contact_polygon.centroid_pressure.fill_(0.0)


class Isosurface:
    def __init__(
        self, body_a, body_b, geom_pairs, mesh_b_is_soft, max_geom_pairs=-1, query_mesh_a=True, compute_device=None
    ):
        if compute_device is None:
            compute_device = wp.get_device()
        self.stream = wp.Stream(compute_device)
        self.sync_event = wp.Event(compute_device)
        self.body_a = body_a
        self.body_b = body_b
        self.body_a_wp = wp.int32(body_a)
        self.body_b_wp = wp.int32(body_b)  # This is used for the kernel to avoid htod transfers.
        self.label = f"Body_{body_a}_vs_Body_{body_b}"
        self.soft_vs_soft = mesh_b_is_soft
        self.sotf_vs_soft_wp = wp.array([1], dtype=wp.int32) if mesh_b_is_soft else wp.array([0], dtype=wp.int32)
        self.geom_pairs = wp.array(geom_pairs, dtype=wp.vec2i, device=compute_device)
        # num_geom_pairs = self.geom_pairs.shape[0]
        if max_geom_pairs == -1:
            max_geom_pairs = self.geom_pairs.shape[0]
        self.max_geom_pairs = max_geom_pairs
        self.geom_pairs_found = wp.zeros(self.max_geom_pairs, dtype=wp.vec2i, device=compute_device)
        self.query_mesh_a = query_mesh_a

        self.intermediate_contact_polygon = ContactPolygon()
        self.contact_polygon = ContactPolygon()
        initialize_contact_polygon(self.max_geom_pairs, self.intermediate_contact_polygon, compute_device)
        initialize_contact_polygon(self.max_geom_pairs, self.contact_polygon, compute_device)
        # TODO: num_nonzero_polygons is not used anywhere. Remove it.
        self.num_nonzero_polygons = wp.zeros(1, dtype=wp.int32, device=compute_device)
        # Quadrature points and weights.
        self.quadrature_weights = wp.array([1.0], dtype=wp.float32, device=compute_device)
        self.quadrature_coords = wp.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=wp.vec3f, device=compute_device)
        # # Second order quadrature rule for triangles.
        # self.quadrature_weights = wp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=wp.float32, device=compute_device)
        # self.quadrature_coords = wp.array(
        #     [[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]],
        #     dtype=wp.vec3f,
        #     device=compute_device,
        # )
        # Result of integration
        self.force = wp.zeros(1, dtype=wp.vec3f, device=compute_device)
        self.torque_a = wp.zeros(1, dtype=wp.vec3f, device=compute_device)
        self.torque_b = wp.zeros(1, dtype=wp.vec3f, device=compute_device)
        self.torque_a_body = wp.zeros(1, dtype=wp.vec3f, device=compute_device)
        self.torque_b_body = wp.zeros(1, dtype=wp.vec3f, device=compute_device)
        self.force_n = wp.zeros(1, dtype=wp.vec3f, device=compute_device)
        self.force_t = wp.zeros(1, dtype=wp.vec3f, device=compute_device)
        # Offset for visualization.
        self.polygon_normals = None
        self.polygon_centers = None
        self.pressure_values = None
        self.max_triangles_found = 0
        self.max_edges_found = 0
        self.max_normals_found = 0
        self.max_contact_polygons_found = {"intermediate": 0, "contact": 0}
        self.max_tet_pairs_found = 0
        # These variables are used for drawing.
        self.is_valid = False
        self.vertices = None
        self.triangles = None
        self.radii = None  # This one facilitates drawing the isosurface.
        self.edges = None  # This one facilitates drawing the isosurface.
        self.per_triangle_pressure = None  # This one facilitates drawing the isosurface.
        self.per_triangle_tet_pairs = None
        self.is_valid_counter = 0  # Might not be necessary.
        self.vertex_counts_np = 0
        self.valid_polygons = []


class RenderMode(Enum):
    NONE = "none"
    OPENGL = "opengl"
    USD = "usd"

    def __str__(self):
        return self.value


class Dirs:
    up = np.array([0.0, 0.0, 1.0])
    front = np.array([1.0, 0.0, 0.0])
    right = np.array([0.0, 1.0, 0.0])
