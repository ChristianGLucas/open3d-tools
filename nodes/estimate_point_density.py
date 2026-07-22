from gen.messages_pb2 import PointDensityInput, PointDensityOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, PointCloudError, build_kdtree, axis_aligned_bbox


def estimate_point_density(ax: AxiomContext, input: PointDensityInput) -> PointDensityOutput:
    """Estimate point-cloud density: mean nearest-neighbor distance and
    points per unit volume of the cloud's axis-aligned bounding box.
    """
    points, _, _ = load_point_cloud(input.cloud)
    k = input.k_neighbors
    if k < 1:
        raise PointCloudError("k_neighbors must be >= 1")

    n = len(points)
    tree = build_kdtree(points)
    kk = min(k + 1, n)
    dists, _ = tree.query(points, k=kk)
    if kk == 1:
        mean_nn_distance = 0.0
    else:
        mean_nn_distance = float(dists[:, 1:].mean())

    _, _, _, volume = axis_aligned_bbox(points)
    # A degenerate (flat/collinear/single-point) cloud has zero bounding-box
    # volume, making "points per unit volume" undefined. Report 0.0 rather
    # than a mathematical infinity (which the JSON<->protobuf bridge would
    # otherwise have to round-trip as a non-numeric "Infinity" literal).
    points_per_unit_volume = float(n / volume) if volume > 1e-12 else 0.0

    return PointDensityOutput(
        mean_nn_distance=mean_nn_distance,
        points_per_unit_volume=points_per_unit_volume,
        bbox_volume=volume,
    )
