import pytest

from gen.messages_pb2 import SegmentPlaneInput
from nodes.segment_plane_ransac import segment_plane_ransac
from nodes._pointcloud import PointCloudError


def test_recovers_z0_plane(ax, plane_with_outliers_cloud):
    # Independent oracle: 441 points exactly on z=0, plus 5 far outliers.
    # RANSAC must find the z=0 plane (normal (0,0,+-1), d=0) with exactly
    # the 441 plane points as inliers.
    result = segment_plane_ransac(
        ax, SegmentPlaneInput(cloud=plane_with_outliers_cloud, distance_threshold=0.05, max_iterations=500, seed=42)
    )
    assert abs(result.a) < 0.05
    assert abs(result.b) < 0.05
    assert abs(abs(result.c) - 1.0) < 0.05
    assert abs(result.d) < 0.05
    assert result.inlier_count == 441


def test_is_deterministic_given_seed(ax, plane_with_outliers_cloud):
    kwargs = dict(cloud=plane_with_outliers_cloud, distance_threshold=0.05, max_iterations=200, seed=123)
    a = segment_plane_ransac(ax, SegmentPlaneInput(**kwargs))
    b = segment_plane_ransac(ax, SegmentPlaneInput(**kwargs))
    assert a.a == b.a and a.b == b.b and a.c == b.c and a.d == b.d
    assert list(a.inlier_indices) == list(b.inlier_indices)


def test_rejects_invalid_threshold(ax, plane_with_outliers_cloud):
    with pytest.raises(PointCloudError):
        segment_plane_ransac(ax, SegmentPlaneInput(cloud=plane_with_outliers_cloud, distance_threshold=0.0))
