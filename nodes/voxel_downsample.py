import numpy as np

from gen.messages_pb2 import VoxelDownsampleInput, PointCloud
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point_cloud_kwargs, PointCloudError


def voxel_downsample(ax: AxiomContext, input: VoxelDownsampleInput) -> PointCloud:
    """Reduce a point cloud by merging all points that fall in the same
    voxel-size grid cell into one point (the mean of their coordinates, and
    of their colors/normals if present). The key data-reduction node for
    dense scans.
    """
    points, colors, normals = load_point_cloud(input.cloud)
    voxel_size = input.voxel_size
    if voxel_size <= 0:
        raise PointCloudError("voxel_size must be > 0")

    origin = points.min(axis=0)
    idx = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n_voxels = len(counts)

    out_points = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(out_points, inverse, points)
    out_points /= counts[:, None]

    out_colors = None
    if colors is not None:
        out_colors = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(out_colors, inverse, colors)
        out_colors /= counts[:, None]

    out_normals = None
    if normals is not None:
        out_normals = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(out_normals, inverse, normals)
        out_normals /= counts[:, None]
        norms = np.linalg.norm(out_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out_normals /= norms

    return PointCloud(**point_cloud_kwargs(out_points, out_colors, out_normals))
