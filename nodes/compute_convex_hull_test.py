import pytest

from nodes.compute_convex_hull import compute_convex_hull
from nodes._pointcloud import PointCloudError


def test_hull_of_cube_corners_equals_cube(ax, unit_cube_cloud):
    # Independent oracle: the cube's 8 corners are already convex, so the
    # hull volume/area are the cube's own (1 and 6). All 8 corners are
    # themselves hull vertices, giving 12 triangles = 36 face indices.
    result = compute_convex_hull(ax, unit_cube_cloud)
    assert abs(result.volume - 1.0) < 1e-9
    assert abs(result.area - 6.0) < 1e-9
    assert len(result.hull_points) == 8
    assert len(result.hull_face_indices) == 36
    assert len(result.hull_face_indices) % 3 == 0


def test_hull_rejects_too_few_points(ax, unit_cube_cloud):
    from gen.messages_pb2 import PointCloud, Point3

    with pytest.raises(PointCloudError):
        compute_convex_hull(ax, PointCloud(points=[Point3(x=0, y=0, z=0), Point3(x=1, y=0, z=0)]))
