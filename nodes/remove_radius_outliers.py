import numpy as np

from gen.messages_pb2 import RadiusOutlierInput, OutlierRemovalOutput, PointCloud
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point_cloud_kwargs, PointCloudError, build_kdtree


def remove_radius_outliers(ax: AxiomContext, input: RadiusOutlierInput) -> OutlierRemovalOutput:
    """Remove points with fewer than min_neighbors other points within a
    given radius.
    """
    points, colors, normals = load_point_cloud(input.cloud)
    radius = input.radius
    min_neighbors = input.min_neighbors
    if radius <= 0:
        raise PointCloudError("radius must be > 0")
    if min_neighbors < 0:
        raise PointCloudError("min_neighbors must be >= 0")

    tree = build_kdtree(points)
    neighbor_lists = tree.query_ball_point(points, r=radius)
    counts = np.array([len(nbrs) - 1 for nbrs in neighbor_lists])  # exclude self
    keep = counts >= min_neighbors

    out_points = points[keep]
    out_colors = colors[keep] if colors is not None else None
    out_normals = normals[keep] if normals is not None else None
    return OutlierRemovalOutput(
        cloud=PointCloud(**point_cloud_kwargs(out_points, out_colors, out_normals)),
        removed_count=int((~keep).sum()),
    )
