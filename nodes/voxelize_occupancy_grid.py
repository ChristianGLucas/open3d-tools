import numpy as np

from gen.messages_pb2 import VoxelizeInput, VoxelizeOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point3_kwargs, PointCloudError


def voxelize_occupancy_grid(ax: AxiomContext, input: VoxelizeInput) -> VoxelizeOutput:
    """Voxelize a point cloud into an occupancy grid and report its summary
    (occupied voxel count, per-axis grid dimensions, grid origin) -- bounded
    well under the transport size cap (never returns the full grid).
    """
    points, _, _ = load_point_cloud(input.cloud)
    voxel_size = input.voxel_size
    if voxel_size <= 0:
        raise PointCloudError("voxel_size must be > 0")

    origin = points.min(axis=0)
    idx = np.floor((points - origin) / voxel_size).astype(np.int64)
    unique_voxels = np.unique(idx, axis=0)
    grid_dims = idx.max(axis=0) - idx.min(axis=0) + 1

    return VoxelizeOutput(
        occupied_voxel_count=len(unique_voxels),
        grid_dim_x=int(grid_dims[0]),
        grid_dim_y=int(grid_dims[1]),
        grid_dim_z=int(grid_dims[2]),
        grid_origin=point3_kwargs(origin),
    )
