import numpy as np

from gen.messages_pb2 import TransformPointCloudInput, PointCloud
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point_cloud_kwargs, apply_transform


def transform_point_cloud(ax: AxiomContext, input: TransformPointCloudInput) -> PointCloud:
    """Apply a caller-supplied 4x4 homogeneous transform (rotate/translate/
    scale) to every point in a cloud. Normals (if present) are transformed
    by the rotation part only and renormalized — exact for rigid or
    uniform-scale transforms.
    """
    points, colors, normals = load_point_cloud(input.cloud)
    out_points, R = apply_transform(points, list(input.transform))

    out_normals = None
    if normals is not None:
        out_normals = normals @ R.T
        norms = np.linalg.norm(out_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out_normals /= norms

    return PointCloud(**point_cloud_kwargs(out_points, colors, out_normals))
