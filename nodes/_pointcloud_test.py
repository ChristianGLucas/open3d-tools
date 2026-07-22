import struct

import numpy as np
import pytest

from nodes._pointcloud import parse_ply_bytes, parse_pcd_bytes, parse_xyz_bytes, PointCloudError


def test_parse_ply_ascii_with_color_and_normals():
    ply = (
        b"ply\nformat ascii 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"property float nx\nproperty float ny\nproperty float nz\n"
        b"end_header\n"
        b"0 0 0 255 0 0 0 0 1\n"
        b"1 1 1 0 255 0 0 1 0\n"
    )
    points, colors, normals = parse_ply_bytes(ply)
    assert points.shape == (2, 3)
    assert colors is not None and np.allclose(colors[0], [1.0, 0.0, 0.0])
    assert normals is not None and np.allclose(normals[1], [0.0, 1.0, 0.0])


def test_parse_ply_binary_little_endian_with_color():
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"end_header\n"
    )
    body = struct.pack("<fffBBB", 1.0, 2.0, 3.0, 10, 20, 30) + struct.pack("<fffBBB", 4.0, 5.0, 6.0, 40, 50, 60)
    points, colors, _ = parse_ply_bytes(header + body)
    assert np.allclose(points[0], [1.0, 2.0, 3.0])
    assert np.allclose(points[1], [4.0, 5.0, 6.0])
    assert colors is not None
    assert np.allclose(colors[0], [10 / 255, 20 / 255, 30 / 255])


def test_parse_ply_rejects_malformed_element_count():
    # A hand-edited/corrupted header (non-numeric vertex count) must raise a
    # clean PointCloudError, not an unguarded ValueError/traceback leak.
    ply = b"ply\nformat ascii 1.0\nelement vertex abc\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    with pytest.raises(PointCloudError):
        parse_ply_bytes(ply)


def test_parse_ply_rejects_non_numeric_vertex_value():
    ply = (
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
        b"oops 0 0\n"
    )
    with pytest.raises(PointCloudError):
        parse_ply_bytes(ply)


def test_parse_pcd_ascii_with_rgb_and_normals():
    pcd = (
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb normal_x normal_y normal_z\n"
        b"SIZE 4 4 4 4 4 4 4\nTYPE F F F F F F F\nCOUNT 1 1 1 1 1 1 1\n"
        b"WIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA ascii\n"
        b"1 2 3 0 0 0 1\n"
    )
    points, colors, normals = parse_pcd_bytes(pcd)
    assert np.allclose(points[0], [1, 2, 3])
    assert normals is not None and np.allclose(normals[0], [0, 0, 1])


def test_parse_pcd_binary():
    header = (
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        b"WIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA binary\n"
    )
    body = struct.pack("<fff", 1.0, 2.0, 3.0) + struct.pack("<fff", 4.0, 5.0, 6.0)
    points, _, _ = parse_pcd_bytes(header + body)
    assert np.allclose(points[1], [4.0, 5.0, 6.0])


def test_parse_pcd_rejects_binary_compressed():
    pcd = b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 1\nDATA binary_compressed\n"
    with pytest.raises(PointCloudError):
        parse_pcd_bytes(pcd)


def test_parse_pcd_rejects_field_size_type_mismatch():
    # FIELDS declares 3 columns but SIZE only declares 2 -- must raise a
    # clean error rather than silently misaligning columns via zip().
    pcd = b"FIELDS x y z\nSIZE 4 4\nTYPE F F\nCOUNT 1 1\nPOINTS 1\nDATA ascii\n0 0 0\n"
    with pytest.raises(PointCloudError):
        parse_pcd_bytes(pcd)


def test_parse_pcd_rejects_malformed_points_line():
    pcd = b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS notanumber\nDATA ascii\n0 0 0\n"
    with pytest.raises(PointCloudError):
        parse_pcd_bytes(pcd)


def test_parse_xyz_with_commas_and_comments():
    text = b"# a comment\n0,0,0\n1, 0, 0\n\n0 1 0\n"
    points, colors, normals = parse_xyz_bytes(text)
    assert points.shape == (3, 3)
    assert colors is None and normals is None
    assert np.allclose(points[1], [1, 0, 0])


def test_parse_xyz_rejects_wrong_column_count():
    with pytest.raises(PointCloudError):
        parse_xyz_bytes(b"0 0 0 0\n")
