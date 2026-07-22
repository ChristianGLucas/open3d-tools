from gen.messages_pb2 import GeometricFeaturesInput
from nodes.compute_geometric_features import compute_geometric_features


def test_flat_plane_points_are_highly_planar(ax, plane_grid_cloud):
    result = compute_geometric_features(ax, GeometricFeaturesInput(cloud=plane_grid_cloud, k_neighbors=8))
    # Center of the grid (index 220) has a fully symmetric, perfectly
    # planar neighborhood.
    assert result.planarity[220] > 0.9
    assert result.curvature[220] < 0.05


def test_collinear_points_are_highly_linear(ax, collinear_line_cloud):
    result = compute_geometric_features(ax, GeometricFeaturesInput(cloud=collinear_line_cloud, k_neighbors=6))
    # Interior point of a perfectly straight line: linearity ~ 1.
    mid = len(result.linearity) // 2
    assert result.linearity[mid] > 0.95
    assert result.planarity[mid] < 0.05
