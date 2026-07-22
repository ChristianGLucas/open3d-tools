from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

from gen.messages_pb2 import PointCloud, ConvexHullOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, point3_list, PointCloudError


def compute_convex_hull(ax: AxiomContext, input: PointCloud) -> ConvexHullOutput:
    """Compute the 3D convex hull of a raw, unstructured point cloud
    (distinct from mesh-tools' ComputeConvexHull, which hulls an existing
    triangle mesh). Returns the hull's vertices, triangulated faces,
    volume, and surface area.
    """
    points, _, _ = load_point_cloud(input)
    if len(points) < 4:
        raise PointCloudError("at least 4 (non-coplanar) points are required for a 3D convex hull")
    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        raise PointCloudError(f"failed to compute convex hull (degenerate/coplanar input?): {exc}") from exc

    hull_vertex_indices = hull.vertices
    remap = {int(orig): i for i, orig in enumerate(hull_vertex_indices)}
    hull_points = points[hull_vertex_indices]
    face_indices = []
    for simplex in hull.simplices:
        for orig_idx in simplex:
            face_indices.append(remap[int(orig_idx)])

    return ConvexHullOutput(
        hull_points=point3_list(hull_points),
        hull_face_indices=face_indices,
        volume=float(hull.volume),
        area=float(hull.area),
    )
