import numpy as np

from gen.messages_pb2 import EstimateNormalsInput, PointCloud
from gen.axiom_context import AxiomContext
from nodes._pointcloud import (
    load_point_cloud,
    point_cloud_kwargs,
    PointCloudError,
    build_kdtree,
    neighbor_index_lists,
    local_pca_eigs,
)


def estimate_normals(ax: AxiomContext, input: EstimateNormalsInput) -> PointCloud:
    """Estimate a per-point unit surface normal via PCA on each point's
    k-nearest (or radius) neighborhood — the smallest-eigenvalue
    eigenvector of the local covariance matrix. Normal sign is arbitrary
    (deterministic for a given input, but not globally consistently
    oriented).
    """
    points, colors, _ = load_point_cloud(input.cloud)
    k = input.k_neighbors
    radius = input.radius
    if not (radius and radius > 0) and not (k and k >= 3):
        raise PointCloudError("k_neighbors must be >= 3 (or set radius > 0)")

    tree = build_kdtree(points)
    neighbor_lists = neighbor_index_lists(points, tree, k_neighbors=k, radius=radius)

    out_normals = np.zeros_like(points)
    for i, nbrs in enumerate(neighbor_lists):
        if len(nbrs) < 3:
            raise PointCloudError(
                f"point cloud too sparse for normal estimation: point {i} has only "
                f"{len(nbrs)} neighbor(s) within range (need >= 3)"
            )
        _, eigvecs = local_pca_eigs(points[nbrs])
        out_normals[i] = eigvecs[:, 0]

    return PointCloud(**point_cloud_kwargs(points, colors, out_normals))
