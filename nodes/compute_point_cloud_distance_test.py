from gen.messages_pb2 import PointCloudDistanceInput, Point3, PointCloud
from nodes.compute_point_cloud_distance import compute_point_cloud_distance


def test_distance_from_point_above_plane(ax, plane_grid_cloud):
    # (0,0,0) is a grid point (linspace(-1,1,21) includes 0 exactly), so the
    # nearest point to (0,0,5) is exactly 5.0 away.
    source = PointCloud(points=[Point3(x=0.0, y=0.0, z=5.0)])
    result = compute_point_cloud_distance(ax, PointCloudDistanceInput(source=source, target=plane_grid_cloud))
    assert len(result.distances) == 1
    assert abs(result.distances[0] - 5.0) < 1e-9
    assert abs(result.mean_distance - 5.0) < 1e-9
    assert abs(result.min_distance - 5.0) < 1e-9
    assert abs(result.max_distance - 5.0) < 1e-9


def test_distance_of_cloud_to_itself_is_zero(ax, plane_grid_cloud):
    result = compute_point_cloud_distance(ax, PointCloudDistanceInput(source=plane_grid_cloud, target=plane_grid_cloud))
    assert result.mean_distance < 1e-9
    assert result.max_distance < 1e-9
