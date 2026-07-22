from gen.messages_pb2 import CropPointCloudInput, Point3
from nodes.crop_point_cloud import crop_point_cloud


def test_crop_keeps_only_points_inside_box(ax, plane_grid_cloud):
    # Grid: x,y in linspace(-1,1,21) (step 0.1). Cropping to [-0.5,0.5] on
    # both axes keeps exactly the 11 values {-0.5,...,0.5} per axis -> 121
    # points (independent, hand-computable oracle).
    result = crop_point_cloud(
        ax,
        CropPointCloudInput(
            cloud=plane_grid_cloud,
            min=Point3(x=-0.5, y=-0.5, z=-1.0),
            max=Point3(x=0.5, y=0.5, z=1.0),
        ),
    )
    assert result.point_count == 121
    for p in result.points:
        assert -0.5 - 1e-9 <= p.x <= 0.5 + 1e-9
        assert -0.5 - 1e-9 <= p.y <= 0.5 + 1e-9


def test_crop_box_excluding_everything_returns_empty(ax, plane_grid_cloud):
    result = crop_point_cloud(
        ax,
        CropPointCloudInput(cloud=plane_grid_cloud, min=Point3(x=100, y=100, z=100), max=Point3(x=101, y=101, z=101)),
    )
    assert result.point_count == 0
