import pytest

from gen.messages_pb2 import ClusterDBSCANInput
from nodes.cluster_dbscan import cluster_dbscan
from nodes._pointcloud import PointCloudError


def test_finds_two_well_separated_blobs(ax, two_cluster_cloud):
    result = cluster_dbscan(ax, ClusterDBSCANInput(cloud=two_cluster_cloud, eps=0.5, min_points=3))
    assert result.n_clusters == 2
    assert result.n_noise == 0
    assert len(result.labels) == 72
    # The two blobs must get different labels; points within one blob must
    # share the same label.
    labels = list(result.labels)
    assert labels[0] == labels[35]  # both in blob A
    assert labels[36] == labels[71]  # both in blob B
    assert labels[0] != labels[36]


def test_rejects_invalid_params(ax, two_cluster_cloud):
    with pytest.raises(PointCloudError):
        cluster_dbscan(ax, ClusterDBSCANInput(cloud=two_cluster_cloud, eps=0.0, min_points=3))
