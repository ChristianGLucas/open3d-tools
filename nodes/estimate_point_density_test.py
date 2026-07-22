from gen.messages_pb2 import PointDensityInput
from nodes.estimate_point_density import estimate_point_density


def test_density_of_unit_cube_corners(ax, unit_cube_cloud):
    # Independent oracle: nearest-neighbor distance between adjacent cube
    # corners is exactly 1.0 (an edge length); bbox volume is exactly 1;
    # points_per_unit_volume = 8 points / 1 volume = 8.
    result = estimate_point_density(ax, PointDensityInput(cloud=unit_cube_cloud, k_neighbors=1))
    assert abs(result.mean_nn_distance - 1.0) < 1e-9
    assert abs(result.bbox_volume - 1.0) < 1e-9
    assert abs(result.points_per_unit_volume - 8.0) < 1e-9


def test_density_of_degenerate_flat_cloud_reports_zero_density(ax, plane_grid_cloud):
    # A perfectly flat cloud has zero bounding-box volume -> density is
    # reported as 0.0 (documented convention), not infinity.
    result = estimate_point_density(ax, PointDensityInput(cloud=plane_grid_cloud, k_neighbors=4))
    assert result.bbox_volume == 0.0
    assert result.points_per_unit_volume == 0.0
