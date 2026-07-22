import contextlib
import io

import numpy as np
from simpleicp import PointCloud as ICPPointCloud
from simpleicp import SimpleICP
from simpleicp.simpleicp import SimpleICPException

from gen.messages_pb2 import ICPInput, ICPOutput
from gen.axiom_context import AxiomContext
from nodes._pointcloud import load_point_cloud, apply_transform, PointCloudError, build_kdtree


def register_icp(ax: AxiomContext, input: ICPInput) -> ICPOutput:
    """Align a source point cloud onto a target via Iterative Closest Point
    (point-to-plane) registration (simpleicp, MIT) — the flagship
    scan-alignment node. Returns the 4x4 rigid transform, fitness (fraction
    of inliers), and inlier RMSE. Bounded by max_iterations so it always
    returns.
    """
    source, _, _ = load_point_cloud(input.source)
    target, _, _ = load_point_cloud(input.target)
    if len(source) < 6 or len(target) < 6:
        raise PointCloudError("ICP requires at least 6 points in each of source and target")

    max_iter = input.max_iterations if input.max_iterations > 0 else 100
    max_iter = max(1, min(max_iter, 200))
    tolerance = input.tolerance if input.tolerance > 0 else 1.0

    if len(input.initial_transform) == 0:
        initial = np.eye(4)
    elif len(input.initial_transform) == 16:
        initial = np.array(input.initial_transform, dtype=np.float64).reshape(4, 4)
    else:
        raise PointCloudError(
            "initial_transform must have exactly 16 values (4x4 row-major) or be empty, "
            f"got {len(input.initial_transform)}"
        )

    # Pre-apply the initial transform ourselves (numpy), then run simpleicp
    # from a zero initial guess. This avoids needing to reverse-engineer
    # simpleicp's internal euler-angle convention for an arbitrary rotation
    # while still supporting any caller-supplied initial 4x4.
    source_pre, _ = apply_transform(source, initial.flatten(order="C").tolist())

    pc_fix = ICPPointCloud(target, columns=["x", "y", "z"])
    pc_mov = ICPPointCloud(source_pre, columns=["x", "y", "z"])

    # simpleicp's normal-estimation step queries each cloud's own k-d tree for
    # `neighbors` nearest neighbors (default 10). When a cloud has fewer than
    # `neighbors` points, scipy's cKDTree pads missing neighbors with an
    # out-of-range sentinel index equal to the tree size, which simpleicp then
    # indexes into directly — an unhandled IndexError. Clamp to the smaller
    # cloud's point count (minus 1, since a point is its own first neighbor)
    # so every requested neighbor index is always valid. Floor of 3 matches
    # the minimum needed to fit a local plane via PCA.
    neighbors = max(3, min(10, len(source_pre) - 1, len(target) - 1))

    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    try:
        # simpleicp prints progress to stdout; keep node logs clean.
        with contextlib.redirect_stdout(io.StringIO()):
            H_icp, _, _, distance_residuals = icp.run(
                max_iterations=max_iter,
                min_change=tolerance,
                neighbors=neighbors,
            )
    except SimpleICPException as exc:
        raise PointCloudError(f"ICP failed to converge: {exc}") from exc

    H_total = H_icp @ initial
    transformed_source, _ = apply_transform(source, H_total.flatten(order="C").tolist())

    max_corr_dist = input.max_correspondence_distance
    if max_corr_dist <= 0:
        tmin, tmax = target.min(axis=0), target.max(axis=0)
        diag = float(np.linalg.norm(tmax - tmin))
        max_corr_dist = 0.05 * diag if diag > 0 else 1.0

    target_tree = build_kdtree(target)
    nn_dist, _ = target_tree.query(transformed_source, k=1)
    nn_dist = np.atleast_1d(nn_dist)
    inlier_mask = nn_dist <= max_corr_dist
    fitness = float(inlier_mask.mean()) if len(nn_dist) > 0 else 0.0
    inlier_rmse = float(np.sqrt(np.mean(nn_dist[inlier_mask] ** 2))) if inlier_mask.any() else 0.0

    return ICPOutput(
        transform=[float(v) for v in H_total.flatten(order="C")],
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        iterations_used=len(distance_residuals),
    )
