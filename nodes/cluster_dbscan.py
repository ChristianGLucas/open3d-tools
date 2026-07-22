from sklearn.cluster import DBSCAN

from gen.messages_pb2 import ClusterDBSCANInput, ClusterDBSCANOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, PointCloudError


def cluster_dbscan(ax: AxiomContext, input: ClusterDBSCANInput) -> ClusterDBSCANOutput:
    """Cluster points by density using DBSCAN (scikit-learn, BSD-3-Clause).
    Returns a per-point cluster label (-1 = noise) and the number of
    clusters found.
    """
    points, _, _ = load_point_cloud(input.cloud)
    eps = input.eps
    min_points = input.min_points
    if eps <= 0:
        raise PointCloudError("eps must be > 0")
    if min_points < 1:
        raise PointCloudError("min_points must be >= 1")

    labels = DBSCAN(eps=eps, min_samples=min_points).fit(points).labels_
    n_clusters = int(len(set(labels.tolist()) - {-1}))
    n_noise = int((labels == -1).sum())
    return ClusterDBSCANOutput(
        labels=[int(l) for l in labels],
        n_clusters=n_clusters,
        n_noise=n_noise,
    )
