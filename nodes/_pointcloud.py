"""Shared point-cloud helpers: canonical `PointCloud` message <-> numpy
arrays, file parsing (PLY/PCD/XYZ), and the size/count caps that bound cost
on untrusted input.

LICENSE NOTE — why this package does not wrap Open3D or PCL
-------------------------------------------------------------
Open3D and the Point Cloud Library (PCL) — the two obvious reference
libraries for this domain — were both evaluated and REJECTED at the
package's license-selection gate. Every Open3D wheel (and every PCL build)
statically links Eigen: Eigen is MPL-2.0, and the SimplicialCholesky module
Open3D actually uses is LGPL unless the build defines
`-DEIGEN_MPL2_ONLY` (upstream Open3D's published wheels do not). Both MPL
and LGPL are explicit, unconditional rejections under this package's
copyleft-anywhere-in-the-tree rule — see the package retrospective for the
full finding.

Instead every node here is a thin wrapper around small, individually
license-clean, Eigen-free libraries:
  - numpy / scipy / scikit-learn (BSD-3-Clause) — arrays, k-d trees
    (`scipy.spatial.cKDTree`), convex hull (`scipy.spatial.ConvexHull`,
    wrapping Qhull's permissive custom license), Delaunay triangulation
    (same), and DBSCAN clustering (`sklearn.cluster.DBSCAN`).
  - pyransac3d (Apache-2.0) — owns the RANSAC plane-segmentation algorithm.
  - simpleicp (MIT) — owns the (point-to-plane) ICP registration algorithm.

PLY/PCD/XYZ file parsing is hand-written here rather than via a library:
it is straightforward format decoding (not an algorithmically hard part
any library needs to "own" — the same rationale other Axiom marketplace
packages use for hand-rolled config/log/CAD-file parsers), and hand-writing
it avoids `plyfile` (GPL-3.0) entirely.
"""
from __future__ import annotations

import struct
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Safety caps (input-surface bounds; see the package retrospective).
# ---------------------------------------------------------------------------

# Hard cap on parsed point count. Well under the ~1-2M "dense scan" ceiling
# mentioned as an example upper bound — chosen instead so that a PointCloud
# message carrying points + colors + normals together (3 parallel Point3
# arrays) stays comfortably under the ~4 MiB platform transport cap even in
# the worst case (all three populated).
MAX_POINTS = 100_000

# Hard cap on raw `source_data` file bytes, checked before parsing (defense
# in depth on top of the platform's own message-size limit).
MAX_SOURCE_BYTES = 8 * 1024 * 1024


class PointCloudError(ValueError):
    """Malformed/oversized point-cloud input. Raising this (a ValueError
    subclass) surfaces as a structured NODE_ERROR_USER, never a crash."""


# ---------------------------------------------------------------------------
# PointCloud message <-> numpy
# ---------------------------------------------------------------------------


def load_point_cloud(pc) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Decode a `PointCloud` message into (points, colors, normals) numpy
    arrays (colors/normals are `None` if absent). Raises `PointCloudError`
    on missing/malformed/oversized input. Enforces `MAX_POINTS`."""
    if len(pc.points) > 0:
        points = np.array([[p.x, p.y, p.z] for p in pc.points], dtype=np.float64)
        colors = (
            np.array([[c.x, c.y, c.z] for c in pc.colors], dtype=np.float64)
            if len(pc.colors) > 0
            else None
        )
        normals = (
            np.array([[n.x, n.y, n.z] for n in pc.normals], dtype=np.float64)
            if len(pc.normals) > 0
            else None
        )
        if colors is not None and len(colors) != len(points):
            raise PointCloudError(
                f"colors length ({len(colors)}) must match points length ({len(points)})"
            )
        if normals is not None and len(normals) != len(points):
            raise PointCloudError(
                f"normals length ({len(normals)}) must match points length ({len(points)})"
            )
    elif pc.source_data:
        if len(pc.source_data) > MAX_SOURCE_BYTES:
            raise PointCloudError(
                f"source_data too large: {len(pc.source_data)} bytes (max {MAX_SOURCE_BYTES})"
            )
        fmt = (pc.format or "").strip().upper()
        if fmt == "PLY":
            points, colors, normals = parse_ply_bytes(pc.source_data)
        elif fmt == "PCD":
            points, colors, normals = parse_pcd_bytes(pc.source_data)
        elif fmt == "XYZ":
            points, colors, normals = parse_xyz_bytes(pc.source_data)
        else:
            raise PointCloudError(
                f"unsupported format {pc.format!r} (expected one of PLY, PCD, XYZ)"
            )
    else:
        raise PointCloudError("PointCloud has neither `points` nor `source_data`")

    if len(points) == 0:
        raise PointCloudError("point cloud is empty")
    if len(points) > MAX_POINTS:
        raise PointCloudError(f"point cloud too large: {len(points)} points (max {MAX_POINTS})")
    if not np.all(np.isfinite(points)):
        raise PointCloudError("point cloud contains non-finite coordinates (NaN/Inf)")
    return points, colors, normals


def point3_list(arr: Optional[np.ndarray]):
    if arr is None:
        return []
    return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in arr]


def point_cloud_kwargs(
    points: np.ndarray, colors: Optional[np.ndarray] = None, normals: Optional[np.ndarray] = None
) -> dict:
    """Build the kwargs dict for constructing a `PointCloud` message from
    numpy arrays. Use as `PointCloud(**point_cloud_kwargs(pts, colors, normals))`."""
    return {
        "points": point3_list(points),
        "colors": point3_list(colors),
        "normals": point3_list(normals),
        "point_count": int(len(points)),
    }


def point3_kwargs(v) -> dict:
    """A single Point3's kwargs from a length-3 iterable."""
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


# ---------------------------------------------------------------------------
# PLY parsing (ASCII + binary_little_endian/big_endian). Supports the
# `vertex` element's x,y,z (required), red/green/blue (uchar, optional
# color), nx/ny/nz (optional normals). The `vertex` element must be first
# (universal convention); any trailing elements (e.g. `face`) are ignored —
# this parser only needs point data.
# ---------------------------------------------------------------------------

_PLY_TYPE_MAP = {
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
}


def parse_ply_bytes(data: bytes):
    if not data.startswith(b"ply"):
        raise PointCloudError("not a valid PLY file (missing 'ply' magic header)")
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end == -1:
        raise PointCloudError("malformed PLY: no 'end_header' line found")
    header_text = data[:header_end].decode("ascii", errors="replace")
    body = data[header_end + len(marker):]

    fmt = None
    elements: list[dict] = []
    current = None
    try:
        for line in header_text.splitlines():
            parts = line.split()
            if not parts or parts[0] in ("ply", "comment", "obj_info"):
                continue
            if parts[0] == "format":
                if len(parts) < 2:
                    raise PointCloudError("malformed PLY: 'format' line missing value")
                fmt = parts[1]
            elif parts[0] == "element":
                if len(parts) < 3:
                    raise PointCloudError("malformed PLY: 'element' line malformed")
                current = {"name": parts[1], "count": int(parts[2]), "properties": []}
                elements.append(current)
            elif parts[0] == "property":
                if current is None:
                    raise PointCloudError("malformed PLY: 'property' before any 'element'")
                if parts[1] == "list":
                    if len(parts) < 5:
                        raise PointCloudError("malformed PLY: 'property list' line malformed")
                    current["properties"].append(("list", parts[2], parts[3], parts[4]))
                else:
                    if len(parts) < 3:
                        raise PointCloudError("malformed PLY: 'property' line malformed")
                    current["properties"].append(("scalar", parts[1], parts[2]))
    except PointCloudError:
        raise
    except (ValueError, IndexError) as exc:
        raise PointCloudError(f"malformed PLY header: {exc}") from exc

    if fmt is None:
        raise PointCloudError("malformed PLY: no 'format' line")
    if not elements or elements[0]["name"] != "vertex":
        raise PointCloudError("unsupported PLY: expected 'vertex' as the first element")

    vertex_el = elements[0]
    count = vertex_el["count"]
    if count < 0 or count > MAX_POINTS:
        raise PointCloudError(f"PLY vertex count too large or invalid: {count} (max {MAX_POINTS})")
    props = vertex_el["properties"]
    if any(p[0] == "list" for p in props):
        raise PointCloudError("unsupported PLY: list property in 'vertex' element")
    prop_names = [p[2] for p in props]
    name_to_col = {name: i for i, name in enumerate(prop_names)}

    for req in ("x", "y", "z"):
        if req not in name_to_col:
            raise PointCloudError(f"malformed PLY: 'vertex' element missing property '{req}'")

    has_color = all(n in name_to_col for n in ("red", "green", "blue"))
    has_normal = all(n in name_to_col for n in ("nx", "ny", "nz"))

    if fmt == "ascii":
        text = body.decode("ascii", errors="replace")
        raw_lines = [ln for ln in text.split("\n") if ln.strip() != ""]
        if len(raw_lines) < count:
            raise PointCloudError("malformed PLY: vertex data truncated")
        rows = [ln.split() for ln in raw_lines[:count]]
        for row in rows:
            if len(row) < len(prop_names):
                raise PointCloudError("malformed PLY: vertex row has fewer values than declared properties")
        try:
            cols = [[float(row[i]) for row in rows] for i in range(len(prop_names))]
        except ValueError as exc:
            raise PointCloudError(f"malformed PLY: non-numeric vertex value: {exc}") from exc
    else:
        if fmt == "binary_little_endian":
            endian = "<"
        elif fmt == "binary_big_endian":
            endian = ">"
        else:
            raise PointCloudError(f"unsupported PLY format: {fmt!r}")
        fmt_chars = []
        for _, type_name, _ in props:
            if type_name not in _PLY_TYPE_MAP:
                raise PointCloudError(f"unsupported PLY property type: {type_name!r}")
            fmt_chars.append(_PLY_TYPE_MAP[type_name][0])
        struct_fmt = endian + "".join(fmt_chars)
        stride = struct.calcsize(struct_fmt)
        needed = stride * count
        if len(body) < needed:
            raise PointCloudError("malformed PLY: binary vertex data truncated")
        unpacked = list(struct.iter_unpack(struct_fmt, body[:needed]))
        cols = [[row[i] for row in unpacked] for i in range(len(prop_names))]

    x = np.array(cols[name_to_col["x"]], dtype=np.float64)
    y = np.array(cols[name_to_col["y"]], dtype=np.float64)
    z = np.array(cols[name_to_col["z"]], dtype=np.float64)
    points = np.column_stack([x, y, z])

    colors = None
    if has_color:
        r = np.array(cols[name_to_col["red"]], dtype=np.float64)
        g = np.array(cols[name_to_col["green"]], dtype=np.float64)
        b = np.array(cols[name_to_col["blue"]], dtype=np.float64)
        # uchar (0-255) is by far the most common color encoding; float
        # colors (already 0-1) also occur. Heuristic: if every channel's
        # max is > 1, treat as 0-255 and normalize.
        if max(r.max(initial=0), g.max(initial=0), b.max(initial=0)) > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        colors = np.column_stack([r, g, b])

    normals = None
    if has_normal:
        nx = np.array(cols[name_to_col["nx"]], dtype=np.float64)
        ny = np.array(cols[name_to_col["ny"]], dtype=np.float64)
        nz = np.array(cols[name_to_col["nz"]], dtype=np.float64)
        normals = np.column_stack([nx, ny, nz])

    return points, colors, normals


# ---------------------------------------------------------------------------
# PCD parsing (ascii + binary DATA sections). Supports FIELDS x y z [rgb]
# [normal_x normal_y normal_z], COUNT 1 per field. `binary_compressed`
# (LZF) is explicitly rejected — a clear error, not a silent misparse.
# ---------------------------------------------------------------------------

_PCD_TYPE_MAP = {
    ("F", 4): ("f", 4), ("F", 8): ("d", 8),
    ("U", 1): ("B", 1), ("U", 2): ("H", 2), ("U", 4): ("I", 4),
    ("I", 1): ("b", 1), ("I", 2): ("h", 2), ("I", 4): ("i", 4),
}


def parse_pcd_bytes(data: bytes):
    text_end = data.find(b"\nDATA ")
    if text_end == -1:
        raise PointCloudError("malformed PCD: no 'DATA' line found")
    # Include the DATA line itself in the header text (find its end).
    line_end = data.find(b"\n", text_end + 1)
    if line_end == -1:
        raise PointCloudError("malformed PCD: truncated 'DATA' line")
    header_text = data[:line_end].decode("ascii", errors="replace")
    body = data[line_end + 1:]

    fields = None
    sizes = None
    types = None
    counts = None
    n_points = None
    data_mode = None
    try:
        for line in header_text.splitlines():
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            if key == "FIELDS":
                fields = parts[1:]
            elif key == "SIZE":
                sizes = [int(v) for v in parts[1:]]
            elif key == "TYPE":
                types = parts[1:]
            elif key == "COUNT":
                counts = [int(v) for v in parts[1:]]
            elif key == "POINTS":
                if len(parts) < 2:
                    raise PointCloudError("malformed PCD: 'POINTS' line missing value")
                n_points = int(parts[1])
            elif key == "DATA":
                if len(parts) < 2:
                    raise PointCloudError("malformed PCD: 'DATA' line missing value")
                data_mode = parts[1].lower()
    except PointCloudError:
        raise
    except (ValueError, IndexError) as exc:
        raise PointCloudError(f"malformed PCD header: {exc}") from exc

    if fields is None or sizes is None or types is None:
        raise PointCloudError("malformed PCD: missing FIELDS/SIZE/TYPE header lines")
    if counts is None:
        counts = [1] * len(fields)
    if any(c != 1 for c in counts):
        raise PointCloudError("unsupported PCD: multi-value (COUNT != 1) fields are not supported")
    if n_points is None:
        raise PointCloudError("malformed PCD: missing POINTS header line")
    if n_points < 0 or n_points > MAX_POINTS:
        raise PointCloudError(f"PCD point count too large or invalid: {n_points} (max {MAX_POINTS})")
    if data_mode is None:
        raise PointCloudError("malformed PCD: missing DATA line")
    if data_mode == "binary_compressed":
        raise PointCloudError("unsupported PCD: binary_compressed (LZF) is not supported; use ascii or binary")
    if data_mode not in ("ascii", "binary"):
        raise PointCloudError(f"unsupported PCD DATA mode: {data_mode!r}")

    for req in ("x", "y", "z"):
        if req not in fields:
            raise PointCloudError(f"malformed PCD: FIELDS missing '{req}'")
    if not (len(fields) == len(sizes) == len(types)):
        raise PointCloudError(
            f"malformed PCD: FIELDS ({len(fields)}), SIZE ({len(sizes)}), and TYPE ({len(types)}) "
            "must declare the same number of columns"
        )

    if data_mode == "ascii":
        text = body.decode("ascii", errors="replace")
        raw_lines = [ln for ln in text.split("\n") if ln.strip() != ""]
        if len(raw_lines) < n_points:
            raise PointCloudError("malformed PCD: point data truncated")
        rows = [ln.split() for ln in raw_lines[:n_points]]
        for row in rows:
            if len(row) < len(fields):
                raise PointCloudError("malformed PCD: point row has fewer values than declared fields")
        try:
            cols = {f: np.array([float(row[i]) for row in rows], dtype=np.float64) for i, f in enumerate(fields)}
        except ValueError as exc:
            raise PointCloudError(f"malformed PCD: non-numeric point value: {exc}") from exc
    else:
        fmt_chars = []
        for name, size in zip(types, sizes):
            key = (name.upper(), size)
            if key not in _PCD_TYPE_MAP:
                raise PointCloudError(f"unsupported PCD field type: {name} size {size}")
            fmt_chars.append(_PCD_TYPE_MAP[key][0])
        struct_fmt = "<" + "".join(fmt_chars)
        stride = struct.calcsize(struct_fmt)
        needed = stride * n_points
        if len(body) < needed:
            raise PointCloudError("malformed PCD: binary point data truncated")
        unpacked = list(struct.iter_unpack(struct_fmt, body[:needed]))
        cols = {f: np.array([row[i] for row in unpacked], dtype=np.float64) for i, f in enumerate(fields)}

    points = np.column_stack([cols["x"], cols["y"], cols["z"]])

    colors = None
    if "rgb" in cols or "rgba" in cols:
        key = "rgb" if "rgb" in cols else "rgba"
        # PCL convention: rgb is a float32 whose bit pattern packs 0x00RRGGBB.
        packed = cols[key].astype(np.float32).view(np.uint32)
        r = ((packed >> 16) & 0xFF).astype(np.float64) / 255.0
        g = ((packed >> 8) & 0xFF).astype(np.float64) / 255.0
        b = (packed & 0xFF).astype(np.float64) / 255.0
        colors = np.column_stack([r, g, b])
    elif all(n in cols for n in ("r", "g", "b")):
        r, g, b = cols["r"], cols["g"], cols["b"]
        if max(r.max(initial=0), g.max(initial=0), b.max(initial=0)) > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        colors = np.column_stack([r, g, b])

    normals = None
    if all(n in cols for n in ("normal_x", "normal_y", "normal_z")):
        normals = np.column_stack([cols["normal_x"], cols["normal_y"], cols["normal_z"]])

    return points, colors, normals


# ---------------------------------------------------------------------------
# XYZ parsing — plain text, exactly 3 whitespace/comma-separated columns
# (x y z) per non-empty line. Deliberately does not attempt to guess
# color/normal columns from extra columns (ambiguous without a header) —
# a disclosed scope limitation, see the package retrospective.
# ---------------------------------------------------------------------------


def parse_xyz_bytes(data: bytes):
    text = data.decode("utf-8", errors="replace")
    points = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.replace(",", " ").split()
        if len(tokens) != 3:
            raise PointCloudError(
                f"malformed XYZ at line {lineno}: expected exactly 3 columns (x y z), got {len(tokens)}"
            )
        try:
            points.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        except ValueError as exc:
            raise PointCloudError(f"malformed XYZ at line {lineno}: {exc}") from exc
        if len(points) > MAX_POINTS:
            raise PointCloudError(f"point cloud too large: > {MAX_POINTS} points (max {MAX_POINTS})")
    if not points:
        raise PointCloudError("XYZ file contains no points")
    return np.array(points, dtype=np.float64), None, None


# ---------------------------------------------------------------------------
# PLY encoding (for ReconstructSurface's mesh output).
# ---------------------------------------------------------------------------


def encode_ply_ascii(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """ASCII PLY with `float` x/y/z vertices and triangular faces."""
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v in vertices:
        lines.append(f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")
    for f in faces:
        lines.append(f"3 {int(f[0])} {int(f[1])} {int(f[2])}")
    return ("\n".join(lines) + "\n").encode("ascii")


# ---------------------------------------------------------------------------
# Geometry helpers shared across nodes.
# ---------------------------------------------------------------------------


def axis_aligned_bbox(points: np.ndarray):
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extents = pmax - pmin
    volume = float(np.prod(extents))
    return pmin, pmax, extents, volume


def oriented_bbox_pca(points: np.ndarray):
    """PCA-oriented (not guaranteed minimum-volume, but a standard,
    deterministic, well-fitting) oriented bounding box."""
    center = points.mean(axis=0)
    centered = points - center
    if len(points) < 2:
        cov = np.zeros((3, 3))
    else:
        cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    proj = centered @ R
    local_min = proj.min(axis=0)
    local_max = proj.max(axis=0)
    extents = local_max - local_min
    obb_center = center + R @ ((local_min + local_max) / 2.0)
    volume = float(np.prod(extents))
    return obb_center, R, extents, volume


def build_kdtree(points: np.ndarray) -> cKDTree:
    return cKDTree(points)


def neighbor_index_lists(points: np.ndarray, tree: cKDTree, k_neighbors: int = 0, radius: float = 0.0):
    """Per-point neighbor index lists (each includes the point itself).
    Exactly one of k_neighbors/radius must be positive."""
    n = len(points)
    if radius and radius > 0:
        return list(tree.query_ball_point(points, r=radius))
    if k_neighbors and k_neighbors > 0:
        k = min(int(k_neighbors) + 1, n)
        if k < 1:
            k = 1
        _, idx = tree.query(points, k=k)
        if k == 1:
            idx = idx.reshape(-1, 1)
        return [list(np.atleast_1d(row)) for row in idx]
    raise PointCloudError("either k_neighbors > 0 or radius > 0 is required")


def local_pca_eigs(neighbor_points: np.ndarray):
    """Eigenvalues (ascending) and eigenvectors of a neighborhood's
    covariance matrix. Requires >= 3 points."""
    mean = neighbor_points.mean(axis=0)
    centered = neighbor_points - mean
    cov = (centered.T @ centered) / len(neighbor_points)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvals, eigvecs


def apply_transform(points: np.ndarray, matrix16):
    if len(matrix16) != 16:
        raise PointCloudError(f"transform must have exactly 16 values (4x4 row-major), got {len(matrix16)}")
    M = np.array(matrix16, dtype=np.float64).reshape(4, 4)
    homog = np.hstack([points, np.ones((len(points), 1))])
    transformed = (M @ homog.T).T
    w = transformed[:, 3]
    w_safe = np.where(np.abs(w) < 1e-12, 1.0, w)
    out = transformed[:, :3] / w_safe[:, None]
    return out, M[:3, :3]
