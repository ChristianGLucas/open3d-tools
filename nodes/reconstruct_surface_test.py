from gen.messages_pb2 import ReconstructSurfaceInput
from nodes.reconstruct_surface import reconstruct_surface
from nodes._pointcloud import parse_ply_bytes


def test_reconstructs_flat_grid_with_exact_area_and_face_count(ax, plane_grid_cloud):
    # Independent oracle: for a regular k x k grid, Delaunay triangulation
    # always yields exactly 2*(k-1)*(k-1) triangles regardless of diagonal
    # tie-breaking, and since the cloud is perfectly flat (z=0), the true
    # 3D surface area equals the flat footprint area exactly: 2 x 2 = 4.0.
    result = reconstruct_surface(ax, ReconstructSurfaceInput(cloud=plane_grid_cloud))
    assert result.vertex_count == 441
    assert result.face_count == 2 * 20 * 20
    assert abs(result.surface_area - 4.0) < 1e-6
    assert result.mesh_format == "PLY"
    assert len(result.mesh_data) > 0

    # Round-trip: our own PLY parser must recover the same vertex count.
    points, _, _ = parse_ply_bytes(result.mesh_data)
    assert len(points) == 441
