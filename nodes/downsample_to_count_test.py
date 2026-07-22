import pytest

from gen.messages_pb2 import DownsampleToCountInput, DownsampleMethod
from nodes.downsample_to_count import downsample_to_count
from nodes._pointcloud import PointCloudError


def test_uniform_downsample_reduces_to_target(ax, plane_grid_cloud):
    result = downsample_to_count(
        ax, DownsampleToCountInput(cloud=plane_grid_cloud, target_count=100, method=DownsampleMethod.UNIFORM)
    )
    assert 0 < result.point_count <= 100


def test_random_downsample_is_deterministic_given_seed(ax, plane_grid_cloud):
    a = downsample_to_count(
        ax, DownsampleToCountInput(cloud=plane_grid_cloud, target_count=50, method=DownsampleMethod.RANDOM, seed=7)
    )
    b = downsample_to_count(
        ax, DownsampleToCountInput(cloud=plane_grid_cloud, target_count=50, method=DownsampleMethod.RANDOM, seed=7)
    )
    assert a.point_count == b.point_count == 50
    for pa, pb in zip(a.points, b.points):
        assert pa.x == pb.x and pa.y == pb.y and pa.z == pb.z


def test_downsample_target_larger_than_cloud_returns_full_cloud(ax, unit_cube_cloud):
    result = downsample_to_count(
        ax, DownsampleToCountInput(cloud=unit_cube_cloud, target_count=1000, method=DownsampleMethod.UNIFORM)
    )
    assert result.point_count == 8


def test_downsample_rejects_non_positive_target(ax, unit_cube_cloud):
    with pytest.raises(PointCloudError):
        downsample_to_count(ax, DownsampleToCountInput(cloud=unit_cube_cloud, target_count=0))
