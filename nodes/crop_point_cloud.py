import numpy as np

from gen.messages_pb2 import CropPointCloudInput, PointCloud
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point_cloud_kwargs, PointCloudError


def crop_point_cloud(ax: AxiomContext, input: CropPointCloudInput) -> PointCloud:
    """Keep only the points of a cloud that fall within an axis-aligned box
    (min <= point <= max on every axis).
    """
    points, colors, normals = load_point_cloud(input.cloud)
    pmin = np.array([input.min.x, input.min.y, input.min.z], dtype=np.float64)
    pmax = np.array([input.max.x, input.max.y, input.max.z], dtype=np.float64)
    if np.any(pmax < pmin):
        raise PointCloudError("crop box `max` must be >= `min` on every axis")

    keep = np.all((points >= pmin) & (points <= pmax), axis=1)
    out_points = points[keep]
    out_colors = colors[keep] if colors is not None else None
    out_normals = normals[keep] if normals is not None else None
    return PointCloud(**point_cloud_kwargs(out_points, out_colors, out_normals))
