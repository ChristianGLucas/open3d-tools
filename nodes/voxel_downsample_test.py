import pytest

from gen.messages_pb2 import VoxelDownsampleInput
from nodes.voxel_downsample import voxel_downsample
from nodes._pointcloud import PointCloudError


def test_voxel_downsample_merges_cube_corners_into_one(ax, unit_cube_cloud):
    # All 8 unit-cube corners fall in the same voxel (origin=(0,0,0),
    # voxel_size=2 -> every corner's floor((0 or 1)/2) == 0). Oracle: they
    # merge into a single point at the cube's centroid (0.5,0.5,0.5).
    result = voxel_downsample(ax, VoxelDownsampleInput(cloud=unit_cube_cloud, voxel_size=2.0))
    assert result.point_count == 1
    p = result.points[0]
    assert abs(p.x - 0.5) < 1e-9
    assert abs(p.y - 0.5) < 1e-9
    assert abs(p.z - 0.5) < 1e-9


def test_voxel_downsample_no_merge_when_voxel_smaller_than_spacing(ax, unit_cube_cloud):
    # voxel_size=0.5 puts 0 and 1 in different bins on every axis -> all 8
    # corners stay distinct.
    result = voxel_downsample(ax, VoxelDownsampleInput(cloud=unit_cube_cloud, voxel_size=0.5))
    assert result.point_count == 8


def test_voxel_downsample_rejects_non_positive_voxel_size(ax, unit_cube_cloud):
    with pytest.raises(PointCloudError):
        voxel_downsample(ax, VoxelDownsampleInput(cloud=unit_cube_cloud, voxel_size=0.0))
