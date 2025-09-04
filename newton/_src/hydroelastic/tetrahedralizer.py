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
Tetrahedral Mesh Generator for Hydroelastic Contact

This module provides functionality to tetrahedralize convex surface meshes
for use in hydroelastic contact simulation. The implementation follows the
Drake approach of creating tetrahedra by connecting each surface triangle
to the mesh centroid.

Key Features:
- Computes volume-weighted centroid of enclosed volume
- Creates tetrahedral elements from surface triangles
- Generates edge connectivity for tetrahedral mesh
- GPU-accelerated using Warp kernels

Usage:
    vertices, elements, edges = tetrahedralize(V_np, F_np, device)

where:
    V_np: numpy array of vertex positions (N x 3)
    F_np: numpy array of face indices (M x 3)
    device: Warp device to run on ("cpu" or "cuda")
"""

import numpy as np
import warp as wp

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Number of edges per tetrahedron (constant for all tetrahedra)
EDGES_PER_TETRAHEDRON = 6

# Tetrahedron edge connectivity pattern (local vertex indices)
TETRAHEDRON_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),  # Edges from vertex 0
    (1, 2),
    (1, 3),  # Edges from vertex 1 (excluding 0-1)
    (2, 3),  # Edge from vertex 2 (excluding previous)
]


# =============================================================================
# WARP KERNELS FOR GEOMETRIC COMPUTATIONS
# =============================================================================


@wp.kernel
def compute_triangle_volume_contributions(
    surface_vertices: wp.array(dtype=wp.vec3f),
    surface_faces: wp.array(dtype=wp.vec3i),
    total_volume: wp.array(dtype=wp.float32),
    weighted_centroid: wp.array(dtype=wp.vec3f),
):
    """
    Compute volume and weighted centroid contributions from each surface triangle.

    Each triangle forms a tetrahedron with the origin, and we accumulate:
    1. The signed volume contribution of each tetrahedron
    2. The volume-weighted centroid contribution

    This follows the Drake implementation for computing mesh centroid.
    """
    triangle_id = wp.tid()

    # Get triangle vertices
    face = surface_faces[triangle_id]
    p = surface_vertices[face[0]]
    q = surface_vertices[face[1]]
    r = surface_vertices[face[2]]

    # Compute signed volume of tetrahedron (origin,p,q,r)
    # Volume = dot(cross(p,q), r) / 6
    # Note: Origin is implicitly at (0,0,0)
    cross_pq = wp.cross(p, q)
    volume_contribution = wp.dot(cross_pq, r) / 6.0

    # Tetrahedron centroid is average of vertices: (p + q + r + origin) / 4
    # Since origin = (0,0,0), this simplifies to (p + q + r) / 4
    tetrahedron_centroid = (p + q + r) * 0.25
    weighted_centroid_contribution = tetrahedron_centroid * volume_contribution

    # Accumulate results atomically (multiple threads may write simultaneously)
    wp.atomic_add(total_volume, 0, volume_contribution)
    wp.atomic_add(weighted_centroid, 0, weighted_centroid_contribution)


@wp.kernel
def finalize_mesh_centroid(
    total_volume: wp.array(dtype=wp.float32),
    weighted_centroid: wp.array(dtype=wp.vec3f),
    final_centroid: wp.array(dtype=wp.vec3f),
):
    """
    Compute final mesh centroid by dividing weighted centroid by total volume.

    This is a separate kernel to ensure all volume contributions are accumulated
    before performing the final division.
    """
    final_centroid[0] = weighted_centroid[0] / total_volume[0]


@wp.kernel
def initialize_tetrahedral_vertices(
    surface_vertices: wp.array(dtype=wp.vec3f),
    mesh_centroid: wp.array(dtype=wp.vec3f),
    tetrahedral_vertices: wp.array(dtype=wp.vec3f),
):
    """
    Initialize tetrahedral mesh vertices.

    The tetrahedral mesh contains:
    - All original surface vertices (indices 0 to N-1)
    - The computed mesh centroid (index N)
    """
    vertex_id = wp.tid()
    num_surface_vertices = surface_vertices.shape[0]

    if vertex_id < num_surface_vertices:
        # Copy surface vertex
        tetrahedral_vertices[vertex_id] = surface_vertices[vertex_id]
    else:
        # Add mesh centroid as final vertex
        tetrahedral_vertices[vertex_id] = mesh_centroid[0]


@wp.kernel
def create_tetrahedral_elements(
    surface_faces: wp.array(dtype=wp.vec3i),
    centroid_vertex_index: wp.int32,
    tetrahedral_elements: wp.array(dtype=wp.vec4i),
):
    """
    Create tetrahedral elements from surface triangles.

    Each surface triangle (p,q,r) becomes a tetrahedron (centroid,p,q,r).
    The centroid is connected to each triangle to form the interior volume.
    """
    triangle_id = wp.tid()
    face = surface_faces[triangle_id]

    # Create tetrahedron: centroid + triangle vertices
    # Order: centroid, vertex_p, vertex_q, vertex_r
    tetrahedral_elements[triangle_id] = wp.vec4i(
        centroid_vertex_index,
        face[0],  # vertex_p
        face[1],  # vertex_q
        face[2],  # vertex_r
    )


@wp.kernel
def generate_tetrahedral_edges(
    tetrahedral_elements: wp.array(dtype=wp.vec4i),
    tetrahedral_edges: wp.array(dtype=wp.vec2i),
):
    """
    Generate edge connectivity for tetrahedral mesh.

    Each tetrahedron has 6 edges connecting its 4 vertices.
    Edge ordering follows the pattern defined in TETRAHEDRON_EDGES.
    """
    tetrahedron_id = wp.tid()
    element = tetrahedral_elements[tetrahedron_id]

    # Base index for this tetrahedron's edges
    edge_base_index = tetrahedron_id * EDGES_PER_TETRAHEDRON

    # Generate all 6 edges for this tetrahedron
    # Edge (0,1): centroid to vertex_p
    tetrahedral_edges[edge_base_index + 0] = wp.vec2i(element[0], element[1])
    # Edge (0,2): centroid to vertex_q
    tetrahedral_edges[edge_base_index + 1] = wp.vec2i(element[0], element[2])
    # Edge (0,3): centroid to vertex_r
    tetrahedral_edges[edge_base_index + 2] = wp.vec2i(element[0], element[3])
    # Edge (1,2): vertex_p to vertex_q
    tetrahedral_edges[edge_base_index + 3] = wp.vec2i(element[1], element[2])
    # Edge (1,3): vertex_p to vertex_r
    tetrahedral_edges[edge_base_index + 4] = wp.vec2i(element[1], element[3])
    # Edge (2,3): vertex_q to vertex_r
    tetrahedral_edges[edge_base_index + 5] = wp.vec2i(element[2], element[3])


# =============================================================================
# MAIN TETRAHEDRALIZATION FUNCTION
# =============================================================================


def tetrahedralize(
    surface_vertices: np.ndarray, surface_faces: np.ndarray, device
) -> tuple[wp.array, wp.array, wp.array]:
    """
    Tetrahedralize a convex surface mesh for hydroelastic contact simulation.

    This function converts a triangular surface mesh into a tetrahedral volume mesh
    by connecting each surface triangle to the volume centroid. The approach follows
    the Drake implementation for hydroelastic contact.

    Args:
        surface_vertices: Vertex positions as numpy array (N x 3)
        surface_faces: Triangle face indices as numpy array (M x 3)
        device: Warp compute device ("cpu" or "cuda")

    Returns:
        Tuple containing:
        - tetrahedral_vertices: Warp array of vertex positions (N+1 x 3)
        - tetrahedral_elements: Warp array of tetrahedron indices (M x 4)
        - tetrahedral_edges: Warp array of edge indices (6*M x 2)

    Raises:
        ValueError: If input arrays have invalid shapes
        RuntimeError: If tetrahedralization fails

    Example:
        >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        >>> tet_verts, tet_elems, tet_edges = tetrahedralize(vertices, faces)
    """
    # TODO(mzamoramora): Allow for warp array as input
    # Validate inputs
    # if not isinstance(surface_vertices, np.ndarray)  or surface_vertices.ndim != 2:
    #     raise ValueError("surface_vertices must be a 2D numpy array")
    # if not isinstance(surface_faces, np.ndarray) or surface_faces.ndim != 2:
    #     raise ValueError("surface_faces must be a 2D numpy array")
    # if surface_vertices.shape[1] != 3:
    #     raise ValueError("surface_vertices must have shape (N, 3)")
    # if surface_faces.shape[1] != 3:
    #     raise ValueError("surface_faces must have shape (M, 3)")

    # Convert numpy arrays to Warp arrays
    warp_vertices = wp.array(surface_vertices, dtype=wp.vec3f, device=device)
    warp_faces = wp.array(surface_faces, dtype=wp.vec3i, device=device)

    num_vertices = warp_vertices.shape[0]
    num_faces = warp_faces.shape[0]

    # Step 1: Compute volume-weighted centroid of enclosed volume
    # --------------------------------------------------------
    total_volume = wp.zeros(1, dtype=wp.float32, device=device)
    weighted_centroid = wp.zeros(1, dtype=wp.vec3f, device=device)
    final_centroid = wp.zeros(1, dtype=wp.vec3f, device=device)

    # Accumulate volume contributions from all triangles
    wp.launch(
        compute_triangle_volume_contributions,
        dim=num_faces,
        inputs=[warp_vertices, warp_faces],
        outputs=[total_volume, weighted_centroid],
        device=device,
    )

    # Compute final centroid
    wp.launch(
        finalize_mesh_centroid, dim=1, inputs=[total_volume, weighted_centroid], outputs=[final_centroid], device=device
    )

    # Step 2: Create tetrahedral mesh vertices
    # ----------------------------------------
    # Tetrahedral mesh has N+1 vertices: original vertices + centroid
    tetrahedral_vertices = wp.zeros(num_vertices + 1, dtype=wp.vec3f, device=device)

    wp.launch(
        initialize_tetrahedral_vertices,
        dim=num_vertices + 1,
        inputs=[warp_vertices, final_centroid],
        outputs=[tetrahedral_vertices],
        device=device,
    )

    # Step 3: Create tetrahedral elements
    # -----------------------------------
    # Each surface triangle becomes one tetrahedron
    centroid_index = num_vertices  # Index of centroid vertex
    tetrahedral_elements = wp.zeros(num_faces, dtype=wp.vec4i, device=device)

    wp.launch(
        create_tetrahedral_elements,
        dim=num_faces,
        inputs=[warp_faces, centroid_index],
        outputs=[tetrahedral_elements],
        device=device,
    )

    # Step 4: Generate edge connectivity
    # ----------------------------------
    # Each tetrahedron contributes 6 edges
    num_edges = num_faces * EDGES_PER_TETRAHEDRON
    tetrahedral_edges = wp.zeros(num_edges, dtype=wp.vec2i, device=device)

    wp.launch(
        generate_tetrahedral_edges,
        dim=num_faces,
        inputs=[tetrahedral_elements],
        outputs=[tetrahedral_edges],
        device=device,
    )

    return tetrahedral_vertices, tetrahedral_elements, tetrahedral_edges


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_mesh_statistics(vertices: wp.array, elements: wp.array, edges: wp.array) -> dict:
    """
    Get statistics about the tetrahedral mesh.

    Args:
        vertices: Tetrahedral vertices array
        elements: Tetrahedral elements array
        edges: Tetrahedral edges array

    Returns:
        Dictionary with mesh statistics
    """
    return {
        "num_vertices": vertices.shape[0],
        "num_tetrahedra": elements.shape[0],
        "num_edges": edges.shape[0],
        "edges_per_tetrahedron": EDGES_PER_TETRAHEDRON,
    }


def validate_tetrahedral_mesh(vertices: wp.array, elements: wp.array, edges: wp.array) -> bool:
    """
    Validate the generated tetrahedral mesh for consistency.

    Args:
        vertices: Tetrahedral vertices array
        elements: Tetrahedral elements array
        edges: Tetrahedral edges array

    Returns:
        True if mesh is valid, False otherwise
    """
    num_vertices = vertices.shape[0]
    num_tetrahedra = elements.shape[0]
    num_edges = edges.shape[0]

    # Check expected relationships
    expected_edges = num_tetrahedra * EDGES_PER_TETRAHEDRON

    if num_edges != expected_edges:
        return False

    # Convert to numpy for validation
    elements_np = elements.numpy()

    # Check that all element indices are valid
    if np.any(elements_np < 0) or np.any(elements_np >= num_vertices):
        return False

    return True
