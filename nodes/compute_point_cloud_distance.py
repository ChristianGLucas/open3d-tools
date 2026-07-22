import numpy as np

from gen.messages_pb2 import PointCloudDistanceInput, PointCloudDistanceOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, build_kdtree


def compute_point_cloud_distance(ax: AxiomContext, input: PointCloudDistanceInput) -> PointCloudDistanceOutput:
    """Compute, for every point in a source cloud, its nearest-neighbor
    distance to a target cloud. Returns the per-point distances plus
    mean/min/max.
    """
    source, _, _ = load_point_cloud(input.source)
    target, _, _ = load_point_cloud(input.target)

    tree = build_kdtree(target)
    dists, _ = tree.query(source, k=1)
    dists = np.atleast_1d(dists)

    return PointCloudDistanceOutput(
        distances=[float(d) for d in dists],
        mean_distance=float(dists.mean()),
        min_distance=float(dists.min()),
        max_distance=float(dists.max()),
    )
