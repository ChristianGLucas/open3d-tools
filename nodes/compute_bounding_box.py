from gen.messages_pb2 import PointCloud, BoundingBoxOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point3_kwargs, axis_aligned_bbox, oriented_bbox_pca


def compute_bounding_box(ax: AxiomContext, input: PointCloud) -> BoundingBoxOutput:
    """Compute both the axis-aligned bounding box and a PCA-oriented
    bounding box of a point cloud.
    """
    points, _, _ = load_point_cloud(input)
    pmin, pmax, extents, volume = axis_aligned_bbox(points)
    obb_center, R, obb_extents, obb_volume = oriented_bbox_pca(points)
    return BoundingBoxOutput(
        aabb_min=point3_kwargs(pmin),
        aabb_max=point3_kwargs(pmax),
        aabb_extents=point3_kwargs(extents),
        aabb_volume=volume,
        obb_center=point3_kwargs(obb_center),
        obb_rotation=[float(v) for v in R.flatten(order="C")],
        obb_extents=point3_kwargs(obb_extents),
        obb_volume=obb_volume,
    )
