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
mat83h = wp.types.matrix(shape=(8, 3), dtype=hydroelastic_type)


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
class HydroelasticBatch:
    max_elements_count: wp.int32
    default_points: wp.array2d(dtype=wp.vec3f)
    indices: wp.array2d(dtype=wp.int32)
    elements_count: wp.array(dtype=wp.int32)
    elements_stride: wp.array(dtype=wp.int32)

    is_soft: wp.array(dtype=wp.bool)

    # Only used for soft meshes (Tetrahedral meshes).
    field: wp.array2d(dtype=wp.float32)
    field_gradient: wp.array2d(dtype=wp.vec3f)
    default_tet_transform_inv: wp.array2d(dtype=wp.mat44)

    # Only used for rigid meshes (Triangular meshes).
    normals: wp.array2d(dtype=wp.vec3f)

    # Material properties
    h: wp.array(dtype=wp.float32)
    d: wp.array(dtype=wp.float32)
    mu_static: wp.array(dtype=wp.float32)
    mu_dynamic: wp.array(dtype=wp.float32)

    # Bvh
    bvh_ids: wp.array(dtype=wp.uint64)


@wp.struct
class IsosurfaceBatch:
    max_element_pairs: wp.int32
    element_pairs: wp.array2d(dtype=wp.vec2i)
    body_a_idx: wp.array(dtype=wp.int32)
    body_b_idx: wp.array(dtype=wp.int32)

    ## Contact polygon data
    v_counts: wp.array2d(dtype=wp.int32)
    vertices: wp.array2d(dtype=wp.vec3f)
    centroids: wp.array2d(dtype=wp.vec3f)
    cartesian_to_penetration: wp.array2d(dtype=wp.vec4f)
    normals: wp.array2d(dtype=wp.vec3f)
    centroid_pressure: wp.array2d(dtype=wp.float32)

    # Combined material properties
    # Hydroelastic modulus
    h_combined: wp.array(dtype=wp.float32)
    # Hunt-Crossley dissipation
    d_combined: wp.array(dtype=wp.float32)
    # Static friction coefficient
    mu_static_combined: wp.array(dtype=wp.float32)
    # Dynamic friction coefficient
    mu_dynamic_combined: wp.array(dtype=wp.float32)

    query_with_mesh_a: wp.array(dtype=wp.bool)
    soft_vs_soft: wp.array(dtype=wp.bool)

    quadrature_weights: wp.array(dtype=wp.float32)
    quadrature_coords: wp.array(dtype=wp.vec3f)

    force: wp.array(dtype=wp.vec3f)
    torque_a: wp.array(dtype=wp.vec3f)
    torque_b: wp.array(dtype=wp.vec3f)
    torque_a_body: wp.array(dtype=wp.vec3f)
    torque_b_body: wp.array(dtype=wp.vec3f)
    force_n: wp.array(dtype=wp.vec3f)
    force_t: wp.array(dtype=wp.vec3f)


class Dirs:
    up = np.array([0.0, 0.0, 1.0])
    front = np.array([1.0, 0.0, 0.0])
    right = np.array([0.0, 1.0, 0.0])


class EditableVars:
    def __init__(self):
        # Drawing vars
        self.np_vertex_offset = np.array([0.0, 0.0, 0.5])

        self.render_isosurfaces_edges = False
        self.render_isosurfaces_normals = False
        self.render_tet_mesh_edges = False
        self.render_forces_flag = True

        self.imgui_isosurfaces_flag = False

        # With a scale of 0.01, an object of 1kg that results in a force of 9.8N, will have an arrow of approx 0.01m = 1cm.
        # The maximum gripping force of Robotiq 2F-140 is 125N. With a scale of 0.001, the arrow will be 0.125m = 12.5cm.
        self.force_scale = 0.1
        # Enabling plotting will make the simulation very slow.
        # TODO: Figure out a way to replace matplotlib with ImGuiPlot via imgui-bundle.
        self.plot_flag = False
