import random as _random

import pyransac3d as pyrsc

from gen.messages_pb2 import SegmentPlaneInput, SegmentPlaneOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, PointCloudError


def segment_plane_ransac(ax: AxiomContext, input: SegmentPlaneInput) -> SegmentPlaneOutput:
    """Segment the single dominant plane in a point cloud via seeded RANSAC
    (pyransac3d, Apache-2.0) — the "find the floor/wall/table" node.
    Returns the plane coefficients (ax + by + cz + d = 0) and the indices
    of the inlier points.
    """
    points, _, _ = load_point_cloud(input.cloud)
    if len(points) < 3:
        raise PointCloudError("at least 3 points are required for plane segmentation")
    threshold = input.distance_threshold
    if threshold <= 0:
        raise PointCloudError("distance_threshold must be > 0")
    max_iter = input.max_iterations if input.max_iterations > 0 else 1000
    max_iter = max(1, min(max_iter, 10000))

    # pyransac3d.Plane.fit uses Python's `random` module internally with no
    # seed parameter of its own — seed it explicitly here for determinism.
    # This node is single-threaded and stateless per invocation, so seeding
    # the global RNG immediately before the call is safe and reproducible.
    _random.seed(input.seed)
    plane = pyrsc.Plane()
    equation, inliers = plane.fit(points, thresh=threshold, maxIteration=max_iter)

    if len(equation) != 4:
        raise PointCloudError("RANSAC failed to find a plane (degenerate input or insufficient inliers)")

    a, b, c, d = (float(v) for v in equation)
    return SegmentPlaneOutput(
        a=a, b=b, c=c, d=d,
        inlier_indices=[int(i) for i in inliers],
        inlier_count=len(inliers),
    )
