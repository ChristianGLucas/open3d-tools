import pytest

from gen.messages_pb2 import EstimateNormalsInput
from nodes.estimate_normals import estimate_normals
from nodes._pointcloud import PointCloudError


def test_normals_on_flat_plane_point_near_z_axis(ax, plane_grid_cloud):
    # Independent oracle: every point lies exactly on z=0, so the local
    # tangent plane's normal must be (0,0,+-1) for interior points (away
    # from the grid boundary, where the neighborhood is symmetric).
    result = estimate_normals(ax, EstimateNormalsInput(cloud=plane_grid_cloud, k_neighbors=8))
    assert result.point_count == 441
    # Index 220 is the center of the 21x21 grid (row 10, col 10) -- fully
    # interior, symmetric neighborhood.
    center_normal = result.normals[220]
    assert abs(center_normal.z) > 0.99
    assert abs(center_normal.x) < 0.1
    assert abs(center_normal.y) < 0.1


def test_normals_rejects_too_few_neighbors_requested(ax, plane_grid_cloud):
    with pytest.raises(PointCloudError):
        estimate_normals(ax, EstimateNormalsInput(cloud=plane_grid_cloud, k_neighbors=1))


def test_normals_rejects_sparse_cloud_for_radius(ax, unit_cube_cloud):
    # Unit cube corners are 1.0 apart; radius=0.1 leaves every point with 0
    # neighbors (< 3 required for a PCA plane fit).
    with pytest.raises(PointCloudError):
        estimate_normals(ax, EstimateNormalsInput(cloud=unit_cube_cloud, radius=0.1))
