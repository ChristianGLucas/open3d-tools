import pytest

from gen.messages_pb2 import PointCloud
from nodes.get_point_cloud_info import get_point_cloud_info
from nodes._pointcloud import PointCloudError


def test_info_on_plane_grid(ax, plane_grid_cloud):
    result = get_point_cloud_info(ax, plane_grid_cloud)
    assert result.point_count == 441  # 21x21 grid
    assert result.has_colors is False
    assert result.has_normals is False
    assert abs(result.bounds_min.x - (-1.0)) < 1e-9
    assert abs(result.bounds_max.x - 1.0) < 1e-9
    assert abs(result.bounds_min.z) < 1e-9
    assert abs(result.bounds_max.z) < 1e-9
    assert abs(result.centroid.x) < 1e-9
    assert abs(result.centroid.y) < 1e-9


def test_info_from_ply_source_data(ax):
    ply = (
        b"ply\nformat ascii 1.0\nelement vertex 3\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
        b"0 0 0\n1 0 0\n0 1 0\n"
    )
    result = get_point_cloud_info(ax, PointCloud(source_data=ply, format="PLY"))
    assert result.point_count == 3
    assert abs(result.bounds_max.x - 1.0) < 1e-9


def test_info_rejects_empty_cloud(ax):
    with pytest.raises(PointCloudError):
        get_point_cloud_info(ax, PointCloud())
