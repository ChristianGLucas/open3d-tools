import numpy as np

from gen.messages_pb2 import TransformPointCloudInput
from nodes.transform_point_cloud import transform_point_cloud
from nodes.conftest import make_transform


def test_translation_shifts_every_point(ax, unit_cube_cloud):
    transform = make_transform(translation=(1.0, 2.0, 3.0))
    result = transform_point_cloud(ax, TransformPointCloudInput(cloud=unit_cube_cloud, transform=transform))
    for orig, out in zip(unit_cube_cloud.points, result.points):
        assert abs(out.x - (orig.x + 1.0)) < 1e-9
        assert abs(out.y - (orig.y + 2.0)) < 1e-9
        assert abs(out.z - (orig.z + 3.0)) < 1e-9


def test_90deg_z_rotation_of_unit_x_point(ax):
    from gen.messages_pb2 import PointCloud, Point3

    cloud = PointCloud(points=[Point3(x=1.0, y=0.0, z=0.0)])
    transform = make_transform(rz_deg=90.0)
    result = transform_point_cloud(ax, TransformPointCloudInput(cloud=cloud, transform=transform))
    # A 90deg rotation about z maps (1,0,0) -> (0,1,0).
    assert abs(result.points[0].x - 0.0) < 1e-9
    assert abs(result.points[0].y - 1.0) < 1e-9
    assert abs(result.points[0].z - 0.0) < 1e-9
