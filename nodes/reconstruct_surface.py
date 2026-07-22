import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError

from gen.messages_pb2 import ReconstructSurfaceInput, ReconstructSurfaceOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, encode_ply_ascii, PointCloudError

# ReconstructSurface's output (vertices + triangle faces, ASCII-PLY-encoded)
# must stay well under the ~4 MiB transport cap. A stricter, node-specific
# input cap keeps that true with margin (see the module docstring in
# _pointcloud.py for the shared MAX_POINTS rationale) -- callers with a
# larger scan should downsample first via VoxelDownsample/DownsampleToCount.
_MAX_RECONSTRUCT_POINTS = 20_000


def reconstruct_surface(ax: AxiomContext, input: ReconstructSurfaceInput) -> ReconstructSurfaceOutput:
    """Reconstruct a mesh from a point cloud via 2.5D Delaunay triangulation
    over the cloud's best-fit (PCA) plane -- models terrain, single-view
    depth-camera scans, and any cloud that is a function over a dominant
    plane. NOT a general watertight reconstruction (Poisson/ball-pivoting
    need a library this package's license gate rejects -- see the
    retrospective); it will self-intersect / lose detail on a fully closed
    3D surface such as a sphere scanned from all sides.
    """
    points, _, _ = load_point_cloud(input.cloud)
    if len(points) < 3:
        raise PointCloudError("at least 3 points are required for surface reconstruction")
    if len(points) > _MAX_RECONSTRUCT_POINTS:
        raise PointCloudError(
            f"point cloud too large for surface reconstruction: {len(points)} points "
            f"(max {_MAX_RECONSTRUCT_POINTS}; downsample first with VoxelDownsample or DownsampleToCount)"
        )

    center = points.mean(axis=0)
    centered = points - center
    cov = (centered.T @ centered) / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Two largest-eigenvalue eigenvectors span the dominant plane.
    order = np.argsort(eigvals)[::-1]
    u, v = eigvecs[:, order[0]], eigvecs[:, order[1]]
    proj = np.column_stack([centered @ u, centered @ v])

    try:
        tri = Delaunay(proj)
    except QhullError as exc:
        raise PointCloudError(f"failed to triangulate point cloud (degenerate/collinear input?): {exc}") from exc

    faces = tri.simplices
    vertices = points

    # True 3D surface area (uses actual 3D vertex positions, not the flat
    # projected area) -- reflects slant/relief, not just the footprint.
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    surface_area = float(tri_areas.sum())

    mesh_bytes = encode_ply_ascii(vertices, faces)

    return ReconstructSurfaceOutput(
        mesh_data=mesh_bytes,
        mesh_format="PLY",
        vertex_count=len(vertices),
        face_count=len(faces),
        surface_area=surface_area,
    )
