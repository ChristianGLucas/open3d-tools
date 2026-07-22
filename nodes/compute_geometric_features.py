import numpy as np

from gen.messages_pb2 import GeometricFeaturesInput, GeometricFeaturesOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import (
    load_point_cloud,
    PointCloudError,
    build_kdtree,
    neighbor_index_lists,
    local_pca_eigs,
)


def compute_geometric_features(ax: AxiomContext, input: GeometricFeaturesInput) -> GeometricFeaturesOutput:
    """Compute per-point local-shape descriptors (linearity, planarity,
    sphericity, curvature) from the eigenvalues of each point's
    k-neighborhood covariance matrix (Weinmann et al.).
    """
    points, _, _ = load_point_cloud(input.cloud)
    k = input.k_neighbors
    if k < 3:
        raise PointCloudError("k_neighbors must be >= 3")

    tree = build_kdtree(points)
    neighbor_lists = neighbor_index_lists(points, tree, k_neighbors=k)

    n = len(points)
    linearity = np.zeros(n)
    planarity = np.zeros(n)
    sphericity = np.zeros(n)
    curvature = np.zeros(n)

    for i, nbrs in enumerate(neighbor_lists):
        if len(nbrs) < 3:
            raise PointCloudError(
                f"point cloud too sparse for geometric features: point {i} has only "
                f"{len(nbrs)} neighbor(s) (need >= 3)"
            )
        eigvals, _ = local_pca_eigs(points[nbrs])
        l0, l1, l2 = sorted(eigvals, reverse=True)  # l0 >= l1 >= l2 >= 0
        total = l0 + l1 + l2
        if l0 <= 0:
            continue
        linearity[i] = (l0 - l1) / l0
        planarity[i] = (l1 - l2) / l0
        sphericity[i] = l2 / l0
        curvature[i] = l2 / total if total > 0 else 0.0

    return GeometricFeaturesOutput(
        linearity=[float(v) for v in linearity],
        planarity=[float(v) for v in planarity],
        sphericity=[float(v) for v in sphericity],
        curvature=[float(v) for v in curvature],
    )
