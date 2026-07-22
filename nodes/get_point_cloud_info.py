from gen.messages_pb2 import PointCloud, PointCloudInfoOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point3_kwargs


def get_point_cloud_info(ax: AxiomContext, input: PointCloud) -> PointCloudInfoOutput:
    """Parse a point cloud (inline points, or PLY/PCD/XYZ file bytes) and
    report a structural summary: point count, whether colors and/or normals
    are present, axis-aligned bounding box, and centroid.
    """
    points, colors, normals = load_point_cloud(input)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    centroid = points.mean(axis=0)
    return PointCloudInfoOutput(
        point_count=len(points),
        has_colors=colors is not None,
        has_normals=normals is not None,
        bounds_min=point3_kwargs(pmin),
        bounds_max=point3_kwargs(pmax),
        centroid=point3_kwargs(centroid),
    )
