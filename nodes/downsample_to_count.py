import numpy as np

from gen.messages_pb2 import DownsampleToCountInput, PointCloud, DownsampleMethod
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point_cloud_kwargs, PointCloudError


def downsample_to_count(ax: AxiomContext, input: DownsampleToCountInput) -> PointCloud:
    """Downsample a point cloud to a target point count, either by an
    evenly-spaced deterministic stride (UNIFORM) or a seeded random sample
    without replacement (RANDOM).
    """
    points, colors, normals = load_point_cloud(input.cloud)
    n = len(points)
    target = input.target_count
    if target <= 0:
        raise PointCloudError("target_count must be > 0")
    target = min(target, n)

    if input.method == DownsampleMethod.RANDOM:
        rng = np.random.default_rng(input.seed)
        idx = np.sort(rng.choice(n, size=target, replace=False))
    else:
        # UNIFORM: an evenly-spaced stride over the input order. Already
        # deterministic (no seed needed); np.unique may yield slightly
        # fewer than `target` points if the stride rounds to duplicate
        # indices for small n/target ratios.
        idx = np.unique(np.round(np.linspace(0, n - 1, target)).astype(np.int64))

    out_points = points[idx]
    out_colors = colors[idx] if colors is not None else None
    out_normals = normals[idx] if normals is not None else None
    return PointCloud(**point_cloud_kwargs(out_points, out_colors, out_normals))
