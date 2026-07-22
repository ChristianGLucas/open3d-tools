import pytest

from gen.messages_pb2 import StatisticalOutlierInput
from nodes.remove_statistical_outliers import remove_statistical_outliers
from nodes._pointcloud import PointCloudError


def test_removes_far_outliers_from_plane(ax, plane_with_outliers_cloud):
    result = remove_statistical_outliers(
        ax, StatisticalOutlierInput(cloud=plane_with_outliers_cloud, nb_neighbors=8, std_ratio=2.0)
    )
    assert result.removed_count >= 5
    # None of the 5 synthetic far-away outliers (magnitude > 10 on some
    # axis) should survive.
    for p in result.cloud.points:
        assert abs(p.x) < 10 and abs(p.y) < 10 and abs(p.z) < 10


def test_rejects_invalid_params(ax, plane_with_outliers_cloud):
    with pytest.raises(PointCloudError):
        remove_statistical_outliers(ax, StatisticalOutlierInput(cloud=plane_with_outliers_cloud, nb_neighbors=0, std_ratio=1.0))
    with pytest.raises(PointCloudError):
        remove_statistical_outliers(ax, StatisticalOutlierInput(cloud=plane_with_outliers_cloud, nb_neighbors=8, std_ratio=0.0))
