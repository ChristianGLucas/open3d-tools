import pytest

from gen.messages_pb2 import RadiusOutlierInput
from nodes.remove_radius_outliers import remove_radius_outliers
from nodes._pointcloud import PointCloudError


def test_removes_isolated_outliers(ax, plane_with_outliers_cloud):
    # Grid spacing is 0.1; the 5 synthetic outliers are >> 0.2 from any
    # other point, so within radius=0.2 they have 0 neighbors (< 3
    # required) and must be removed. Every interior/edge grid point has
    # several neighbors within 0.2 and must be kept.
    result = remove_radius_outliers(
        ax, RadiusOutlierInput(cloud=plane_with_outliers_cloud, radius=0.2, min_neighbors=3)
    )
    assert result.removed_count >= 5
    for p in result.cloud.points:
        assert abs(p.x) < 10 and abs(p.y) < 10 and abs(p.z) < 10


def test_rejects_invalid_params(ax, plane_with_outliers_cloud):
    with pytest.raises(PointCloudError):
        remove_radius_outliers(ax, RadiusOutlierInput(cloud=plane_with_outliers_cloud, radius=0.0, min_neighbors=3))
