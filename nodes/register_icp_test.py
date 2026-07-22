import numpy as np
import pytest

from gen.messages_pb2 import ICPInput, Point3, PointCloud
from nodes.register_icp import register_icp
from nodes._pointcloud import PointCloudError, apply_transform
from nodes.conftest import make_transform


def _cloud(points):
    return PointCloud(points=[Point3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points])


def test_icp_recovers_small_transform_from_identity_guess(ax, fibonacci_sphere_points):
    # Independent oracle: target = a known small rotation+translation
    # applied to source; with no initial guess (identity), nearest-neighbor
    # correspondence stays locally valid for a small perturbation, so ICP
    # must recover the known transform tightly.
    known = make_transform(rz_deg=8.0, translation=(0.1, 0.05, 0.05))
    target_points, _ = apply_transform(fibonacci_sphere_points, known)

    result = register_icp(
        ax,
        ICPInput(
            source=_cloud(fibonacci_sphere_points),
            target=_cloud(target_points),
            max_iterations=100,
        ),
    )
    H = np.array(result.transform, dtype=np.float64).reshape(4, 4)
    H_known = np.array(known, dtype=np.float64).reshape(4, 4)
    assert np.allclose(H[:3, :3], H_known[:3, :3], atol=1e-3)
    assert np.allclose(H[:3, 3], H_known[:3, 3], atol=1e-3)
    assert result.fitness > 0.99
    assert result.inlier_rmse < 1e-3
    assert result.iterations_used >= 1


def test_icp_refines_large_rotation_from_close_initial_guess(ax, fibonacci_sphere_points):
    # A rotationally-near-symmetric shape (a sphere) makes a LARGE rotation
    # unrecoverable from an identity guess -- nearest-neighbor correspondence
    # is no longer locally valid (this is expected ICP behavior, the same
    # local-convergence limitation every ICP implementation has, not a bug).
    # The realistic use (and the honest oracle here) is a coarse initial
    # alignment refined by ICP -- this also exercises this node's own
    # H_total = H_icp @ initial_transform composition.
    known = make_transform(rz_deg=30.0, translation=(2.0, 1.0, 0.5))
    target_points, _ = apply_transform(fibonacci_sphere_points, known)
    close_initial_guess = make_transform(rz_deg=27.0, translation=(1.9, 0.9, 0.5))

    result = register_icp(
        ax,
        ICPInput(
            source=_cloud(fibonacci_sphere_points),
            target=_cloud(target_points),
            initial_transform=close_initial_guess,
            max_iterations=100,
        ),
    )
    H = np.array(result.transform, dtype=np.float64).reshape(4, 4)
    H_known = np.array(known, dtype=np.float64).reshape(4, 4)
    assert np.allclose(H[:3, :3], H_known[:3, :3], atol=1e-3)
    assert np.allclose(H[:3, 3], H_known[:3, 3], atol=1e-3)
    assert result.fitness > 0.99
    assert result.inlier_rmse < 1e-3


def test_icp_identity_clouds_returns_near_identity(ax, fibonacci_sphere_points):
    cloud = _cloud(fibonacci_sphere_points)
    result = register_icp(ax, ICPInput(source=cloud, target=cloud, max_iterations=50))
    H = np.array(result.transform, dtype=np.float64).reshape(4, 4)
    assert np.allclose(H, np.eye(4), atol=1e-3)
    assert result.fitness > 0.99
    assert result.inlier_rmse < 1e-3


def test_icp_rejects_too_few_points(ax):
    tiny = _cloud(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    with pytest.raises(PointCloudError):
        register_icp(ax, ICPInput(source=tiny, target=tiny))


def test_icp_rejects_malformed_initial_transform(ax, fibonacci_sphere_points):
    cloud = _cloud(fibonacci_sphere_points)
    with pytest.raises(PointCloudError):
        register_icp(ax, ICPInput(source=cloud, target=cloud, initial_transform=[1.0, 2.0, 3.0]))
