import pytest

from gen.messages_pb2 import VoxelizeInput
from nodes.voxelize_occupancy_grid import voxelize_occupancy_grid
from nodes._pointcloud import PointCloudError


def test_voxelize_cube_corners_into_one_voxel(ax, unit_cube_cloud):
    result = voxelize_occupancy_grid(ax, VoxelizeInput(cloud=unit_cube_cloud, voxel_size=2.0))
    assert result.occupied_voxel_count == 1
    assert result.grid_dim_x == result.grid_dim_y == result.grid_dim_z == 1
    assert abs(result.grid_origin.x) < 1e-9


def test_voxelize_cube_corners_into_eight_voxels(ax, unit_cube_cloud):
    result = voxelize_occupancy_grid(ax, VoxelizeInput(cloud=unit_cube_cloud, voxel_size=0.5))
    assert result.occupied_voxel_count == 8
    assert result.grid_dim_x == result.grid_dim_y == result.grid_dim_z == 3


def test_rejects_non_positive_voxel_size(ax, unit_cube_cloud):
    with pytest.raises(PointCloudError):
        voxelize_occupancy_grid(ax, VoxelizeInput(cloud=unit_cube_cloud, voxel_size=-1.0))
