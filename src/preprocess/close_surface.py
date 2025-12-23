import numpy as np
from collections import defaultdict

def close_surface(faces):
    """
    Close a mesh with a single hole by:
    1. Finding edges that form the hole (edges appearing only once)
    2. Creating triangle strips (zig-zag pattern) to close the hole
    3. Returning refined vertices and faces
    
    Args:
        verts: (T, V, 3) array where T is number of frames, V is number of vertices
        faces: (F, 3) array of face indices (constant across frames)
    
    Returns:
        verts: (T, V, 3) array (unchanged, no new vertices added)
        new_faces: (F', 3) array where F' = F + number of closing faces
    """
    
    # Step 1: Find edges that form the hole (edges appearing only once)
    edge_count = defaultdict(int)
    
    # Count edge occurrences (edges are undirected, so normalize order)
    for face in faces:
        num_verts = len(face)
        for i in range(num_verts):
            v1 = face[i]
            v2 = face[(i + 1) % num_verts]
            # Normalize edge order (smaller index first)
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] += 1
    
    # Find hole edges (edges that appear only once)
    hole_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    if len(hole_edges) == 0:
        # No hole found, return original mesh
        return faces
    
    # Step 2: Find all vertices on the hole boundary
    hole_vertices = set()
    for edge in hole_edges:
        hole_vertices.add(edge[0])
        hole_vertices.add(edge[1])
    
    hole_vertices = list(hole_vertices)
    
    # Create a mapping from hole vertices to their order around the hole
    # We need to order the vertices to form a cycle
    hole_vertex_order = _order_hole_vertices(hole_edges, hole_vertices)
    
    n = len(hole_vertex_order)
    if n < 3:
        # Not enough vertices to form a triangle
        return faces
    
    # Step 3: Generate triangle strip (zig-zag pattern) to close the hole
    new_faces = faces.tolist() if isinstance(faces, np.ndarray) else list(faces)
    
    # Triangle strip pattern: (0, 1, n-1), (n-1, 1, 2), (n-2, n-1, 2), (n-2, 2, 3), ...
    # This creates a fan pattern that closes the hole
    
    v = hole_vertex_order  # Shorthand for ordered vertices
    
    # First triangle: (0, 1, n-1)
    new_faces.append([v[0], v[1], v[n-1]])
    
    # Track indices: end_idx moves from n-1 down, start_idx moves from 1 up
    end_idx = n - 1
    start_idx = 1
    step = 0
    
    # Generate triangles alternating between end and start
    while start_idx < end_idx - 1:
        if step % 2 == 0:
            # Even step: (end-1, end, start) - move end_idx down
            new_faces.append([v[end_idx - 1], v[end_idx], v[start_idx]])
            end_idx -= 1
        else:
            # Odd step: (end, start, start+1) - move start_idx up
            new_faces.append([v[end_idx], v[start_idx], v[start_idx + 1]])
            start_idx += 1
        step += 1
    
    new_faces = np.array(new_faces, dtype=faces.dtype)
    
    return new_faces


def _order_hole_vertices(hole_edges, hole_vertices):
    """
    Order hole vertices to form a cycle by following connected edges.
    Returns ordered list of vertex indices.
    """
    if len(hole_vertices) == 0:
        return []
    
    # Build adjacency list for hole edges
    adj = defaultdict(list)
    for edge in hole_edges:
        v1, v2 = edge
        adj[v1].append(v2)
        adj[v2].append(v1)
    
    # Start from first vertex and follow the cycle
    ordered = []
    visited_edges = set()
    start_vertex = hole_vertices[0]
    current_vertex = start_vertex
    
    # Follow the cycle
    while len(ordered) < len(hole_vertices):
        ordered.append(current_vertex)
        # Find next unvisited edge from current vertex
        next_vertex = None
        for neighbor in adj[current_vertex]:
            edge = tuple(sorted([current_vertex, neighbor]))
            if edge not in visited_edges:
                visited_edges.add(edge)
                next_vertex = neighbor
                break
        
        if next_vertex is None:
            # Cycle complete or disconnected (shouldn't happen for single hole)
            break
        current_vertex = next_vertex
    
    return ordered