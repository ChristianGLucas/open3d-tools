from gen.messages_pb2 import StatisticalOutlierInput, OutlierRemovalOutput, PointCloud
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point_cloud_kwargs, PointCloudError, build_kdtree


def remove_statistical_outliers(ax: AxiomContext, input: StatisticalOutlierInput) -> OutlierRemovalOutput:
    """Remove points whose mean distance to their nb_neighbors nearest
    neighbors exceeds (global mean + std_ratio * global std-dev) —
    sparse-noise removal.
    """
    points, colors, normals = load_point_cloud(input.cloud)
    nb = input.nb_neighbors
    std_ratio = input.std_ratio
    if nb < 1:
        raise PointCloudError("nb_neighbors must be >= 1")
    if std_ratio <= 0:
        raise PointCloudError("std_ratio must be > 0")

    n = len(points)
    tree = build_kdtree(points)
    k = min(nb + 1, n)
    dists, _ = tree.query(points, k=k)
    if k == 1:
        dists = dists.reshape(-1, 1)
    mean_dist = dists[:, 1:].mean(axis=1) if k > 1 else dists[:, 0]

    global_mean = mean_dist.mean()
    global_std = mean_dist.std()
    threshold = global_mean + std_ratio * global_std
    keep = mean_dist <= threshold

    out_points = points[keep]
    out_colors = colors[keep] if colors is not None else None
    out_normals = normals[keep] if normals is not None else None
    return OutlierRemovalOutput(
        cloud=PointCloud(**point_cloud_kwargs(out_points, out_colors, out_normals)),
        removed_count=int((~keep).sum()),
    )
