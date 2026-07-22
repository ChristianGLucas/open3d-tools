from nodes.compute_bounding_box import compute_bounding_box


def test_aabb_of_unit_cube(ax, unit_cube_cloud):
    result = compute_bounding_box(ax, unit_cube_cloud)
    assert abs(result.aabb_min.x) < 1e-9 and abs(result.aabb_min.y) < 1e-9 and abs(result.aabb_min.z) < 1e-9
    assert abs(result.aabb_max.x - 1.0) < 1e-9
    assert abs(result.aabb_max.y - 1.0) < 1e-9
    assert abs(result.aabb_max.z - 1.0) < 1e-9
    assert abs(result.aabb_volume - 1.0) < 1e-9


def test_obb_of_elongated_box_is_axis_aligned(ax, box_1x1x4_cloud):
    # Non-degenerate PCA (distinct per-axis variances) -> OBB must recover
    # the box's true extents (1,1,4 in some order) and volume 4.
    result = compute_bounding_box(ax, box_1x1x4_cloud)
    extents = sorted([result.obb_extents.x, result.obb_extents.y, result.obb_extents.z])
    assert abs(extents[0] - 1.0) < 1e-6
    assert abs(extents[1] - 1.0) < 1e-6
    assert abs(extents[2] - 4.0) < 1e-6
    assert abs(result.obb_volume - 4.0) < 1e-6
