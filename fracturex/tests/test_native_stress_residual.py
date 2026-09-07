"""Tests for native divergence/jump residuals and Oswald stress recovery."""
from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TriangleMesh

from fracturex.learn.eval.native_stress_residual import (
    OswaldRecoveredStress,
    compute_native_stress_residual,
    oswald_average_dg_displacement,
)


def _two_triangle_mesh():
    return TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=1, ny=1)


def test_constant_stress_has_zero_native_residual():
    mesh = _two_triangle_mesh()

    def value(bcs, cells):
        out = np.zeros((len(cells), len(bcs), 3), dtype=np.float64)
        out[..., 0] = 2.0
        out[..., 1] = 0.25
        out[..., 2] = 1.0
        return out

    def divergence(bcs, cells):
        return np.zeros((len(cells), len(bcs), 2), dtype=np.float64)

    result = compute_native_stress_residual(mesh, value, divergence, quadrature_order=4)
    assert result.relative_estimator < 1.0e-13
    assert result.max_traction_jump < 1.0e-13


def test_cellwise_stress_jump_is_detected():
    mesh = _two_triangle_mesh()

    def value(bcs, cells):
        out = np.zeros((len(cells), len(bcs), 3), dtype=np.float64)
        out[..., 0] = 1.0 + np.asarray(cells)[:, None]
        return out

    def divergence(bcs, cells):
        return np.zeros((len(cells), len(bcs), 2), dtype=np.float64)

    result = compute_native_stress_residual(mesh, value, divergence, quadrature_order=4)
    assert result.volume_norm == 0.0
    assert result.jump_norm > 0.0
    assert result.relative_jump > 0.0


def test_globally_linear_stress_has_no_spurious_oriented_edge_jump():
    """Opposite cell-edge parameterizations must sample identical physical points."""
    mesh = _two_triangle_mesh()
    node = np.asarray(mesh.entity("node"), dtype=np.float64)
    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)

    def value(bcs, cells):
        points = np.einsum("qv,cvd->cqd", bcs, node[cell[cells]])
        x = points[..., 0]
        y = points[..., 1]
        return np.stack([x, y, x + y], axis=-1)

    def divergence(bcs, cells):
        result = np.empty((len(cells), len(bcs), 2), dtype=np.float64)
        result[..., 0] = 2.0  # d_x sigma_xx + d_y sigma_xy
        result[..., 1] = 1.0  # d_x sigma_xy + d_y sigma_yy
        return result

    result = compute_native_stress_residual(mesh, value, divergence, quadrature_order=4)
    assert result.volume_norm > 0.0
    assert result.relative_jump < 1.0e-13
    assert result.max_traction_jump < 1.0e-13


class _Discretization:
    pass


def _make_affine_dg_discretization():
    mesh = _two_triangle_mesh()
    dg_scalar = LagrangeFESpace(mesh, p=2, ctype="D")
    tensor = TensorFunctionSpace(dg_scalar, shape=(2, -1))
    damage_space = LagrangeFESpace(mesh, p=2, ctype="C")
    discr = _Discretization()
    discr.mesh = mesh
    discr.space_u = tensor
    discr.space_d = damage_space
    discr.state = _Discretization()
    discr.state.u = tensor.interpolate(
        lambda points: bm.stack(
            [points[..., 0] + 2.0 * points[..., 1], -points[..., 0] + points[..., 1]],
            axis=-1,
        )
    )
    discr.state.d = damage_space.function()
    discr.state.d[:] = 0.0
    return discr


def test_oswald_recovery_preserves_affine_displacement_and_zero_residual():
    discr = _make_affine_dg_discretization()
    averaged = oswald_average_dg_displacement(discr)
    points = np.asarray(LagrangeFESpace(discr.mesh, p=2, ctype="C").interpolation_points())
    expected = np.stack([points[:, 0] + 2.0 * points[:, 1], -points[:, 0] + points[:, 1]], axis=-1)
    assert np.allclose(averaged, expected, atol=1.0e-13)

    recovered = OswaldRecoveredStress(
        discr,
        young_modulus=200.0,
        poisson_ratio=0.2,
        plane="strain",
    )
    result = compute_native_stress_residual(
        discr.mesh,
        recovered.value,
        recovered.divergence,
        quadrature_order=4,
    )
    assert result.relative_estimator < 1.0e-12
    assert result.max_traction_jump < 1.0e-10
