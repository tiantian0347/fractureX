"""Convergence tests for the 𝓘₁ / 𝓘₂ interpolation operators.

Locks in the orders observed in docs/operator_learning/m0_interpolation_error.md §2:
  • 𝓘₁ (sample_field_nearest_quad) — relative L² ~ O(h)        on smooth f
  • 𝓘₂ (sample_field_l2_projection, P1) — relative L² ~ O(h²)  on smooth f

Reference field: f(x,y) = x² + y². Domain Ω = [0,1]² with no notch so the
mask covers the whole grid (independent of geometry-side bugs).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fracturex.postprocess.dataset_export import (
    GridSpec,
    sample_field_l2_projection,
    sample_field_nearest_quad,
)
from fracturex.learn.eval.metrics import relative_l2, relative_linf


_GRID = GridSpec(H=128, W=128, bbox=((0.0, 1.0), (0.0, 1.0)))
_NXS = (8, 16, 32, 64)
_Q_ORDER = 5


def _truth_grid() -> np.ndarray:
    xs = np.linspace(0, 1, _GRID.W)
    ys = np.linspace(0, 1, _GRID.H)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return (X**2 + Y**2).astype(np.float32)


def _eval_smooth_qp(mesh) -> tuple[np.ndarray, np.ndarray]:
    qf = mesh.quadrature_formula(_Q_ORDER)
    bcs, _ = qf.get_quadrature_points_and_weights()
    qp = np.asarray(mesh.bc_to_point(bcs))
    fqp = qp[..., 0] ** 2 + qp[..., 1] ** 2
    return fqp, qp


@pytest.mark.parametrize("nx", _NXS)
def test_nearest_quad_smooth_field_first_order(nx):
    from fealpy.mesh import TriangleMesh

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=nx, ny=nx)
    fqp, qp = _eval_smooth_qp(mesh)
    out = sample_field_nearest_quad(fqp, qp, _GRID)
    truth = _truth_grid()
    rel = relative_l2(out, truth, np.ones_like(truth, dtype=bool))
    assert rel < 5.0e-2, f"nx={nx}: 𝓘₁ rel L² unexpectedly high ({rel:.2e})"
    # also check it's not pathologically zero (catch silent all-zero bug)
    assert relative_linf(out, truth, np.ones_like(truth, dtype=bool)) > 0


def test_nearest_quad_first_order_rate():
    """rel L² should roughly halve when h halves (rate ≈ 1)."""
    from fealpy.mesh import TriangleMesh

    rels = []
    for nx in _NXS:
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=nx, ny=nx)
        fqp, qp = _eval_smooth_qp(mesh)
        out = sample_field_nearest_quad(fqp, qp, _GRID)
        truth = _truth_grid()
        rels.append(relative_l2(out, truth, np.ones_like(truth, dtype=bool)))
    # consecutive ratios should be in [1.5, 3.0] for ~O(h) on this regime
    ratios = [rels[i] / rels[i + 1] for i in range(len(rels) - 1)]
    for r in ratios:
        assert 1.5 < r < 3.0, f"𝓘₁ ratio out of band: ratios={ratios}, rels={rels}"


@pytest.mark.parametrize("nx", _NXS)
def test_l2_projection_smooth_field_second_order(nx):
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.mesh import TriangleMesh

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=nx, ny=nx)
    space_d = LagrangeFESpace(mesh, p=1, ctype="C")
    discr = SimpleNamespace(mesh=mesh, space_d=space_d)
    fqp, _ = _eval_smooth_qp(mesh)

    out = sample_field_l2_projection(fqp, discr, _GRID, quadrature_order=_Q_ORDER)
    truth = _truth_grid()
    mask_all = np.ones_like(truth, dtype=bool)
    rel = relative_l2(out, truth, mask_all)
    # P1 projection of x²+y² has L² error ≤ Ch²; on h=1/8 it's ~2e-3.
    expected_upper = 4.0e-3 * (8.0 / nx) ** 2
    assert rel < 3 * expected_upper + 1e-4, (
        f"nx={nx}: 𝓘₂ rel L² {rel:.2e} above 3× expected {expected_upper:.2e}"
    )


def test_l2_projection_second_order_rate():
    """rel L² should drop by ~4× when h halves (rate ≈ 2)."""
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.mesh import TriangleMesh

    rels = []
    for nx in _NXS:
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=nx, ny=nx)
        space_d = LagrangeFESpace(mesh, p=1, ctype="C")
        discr = SimpleNamespace(mesh=mesh, space_d=space_d)
        fqp, _ = _eval_smooth_qp(mesh)
        out = sample_field_l2_projection(fqp, discr, _GRID, quadrature_order=_Q_ORDER)
        rels.append(relative_l2(out, _truth_grid(), np.ones_like(_truth_grid(), dtype=bool)))
    ratios = [rels[i] / rels[i + 1] for i in range(len(rels) - 1)]
    for r in ratios:
        assert 3.0 < r < 5.0, f"𝓘₂ ratio out of band: ratios={ratios}, rels={rels}"


def test_l2_projection_dominates_nearest_quad():
    """At a fixed h, 𝓘₂ should be sharply more accurate than 𝓘₁ on smooth fields."""
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.mesh import TriangleMesh

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=32, ny=32)
    space_d = LagrangeFESpace(mesh, p=1, ctype="C")
    discr = SimpleNamespace(mesh=mesh, space_d=space_d)
    fqp, qp = _eval_smooth_qp(mesh)

    out_i1 = sample_field_nearest_quad(fqp, qp, _GRID)
    out_i2 = sample_field_l2_projection(fqp, discr, _GRID, quadrature_order=_Q_ORDER)
    truth = _truth_grid()
    mask_all = np.ones_like(truth, dtype=bool)
    e1 = relative_l2(out_i1, truth, mask_all)
    e2 = relative_l2(out_i2, truth, mask_all)
    assert e2 < e1 / 10, f"𝓘₂ should crush 𝓘₁ on smooth fields; got e1={e1:.2e}, e2={e2:.2e}"


def test_l2_projection_handles_multichannel():
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.mesh import TriangleMesh

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=16, ny=16)
    space_d = LagrangeFESpace(mesh, p=1, ctype="C")
    discr = SimpleNamespace(mesh=mesh, space_d=space_d)
    qf = mesh.quadrature_formula(_Q_ORDER)
    bcs, _ = qf.get_quadrature_points_and_weights()
    qp = np.asarray(mesh.bc_to_point(bcs))
    g = np.stack([qp[..., 0], qp[..., 1], qp[..., 0] * qp[..., 1]], axis=-1)
    out = sample_field_l2_projection(g, discr, _GRID, quadrature_order=_Q_ORDER)
    assert out.shape == (3, _GRID.H, _GRID.W), out.shape
    assert out.dtype == np.float32


def test_mask_zeros_outside():
    """Outside-Ω mask must blank both 𝓘₁ and 𝓘₂ outputs to zero."""
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.mesh import TriangleMesh

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=8, ny=8)
    space_d = LagrangeFESpace(mesh, p=1, ctype="C")
    discr = SimpleNamespace(mesh=mesh, space_d=space_d)
    fqp, qp = _eval_smooth_qp(mesh)

    mask = np.zeros((_GRID.H, _GRID.W), dtype=np.uint8)
    mask[: _GRID.H // 2, :] = 1
    out1 = sample_field_nearest_quad(fqp, qp, _GRID, mask=mask)
    out2 = sample_field_l2_projection(
        fqp, discr, _GRID, mask=mask, quadrature_order=_Q_ORDER
    )
    assert (out1[~mask.astype(bool)] == 0).all()
    assert (out2[~mask.astype(bool)] == 0).all()


def test_metrics_relative_norms_basic():
    target = np.ones((4, 4))
    pred = np.ones((4, 4)) * 1.1
    mask = np.ones((4, 4), dtype=bool)
    assert abs(relative_l2(pred, target, mask) - 0.1) < 1e-6
    assert abs(relative_linf(pred, target, mask) - 0.1) < 1e-6


def test_metrics_mask_excludes_outside():
    target = np.ones((4, 4))
    pred = target.copy()
    pred[:, 2:] = 99.0  # garbage outside mask must not leak in
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True
    assert relative_l2(pred, target, mask) < 1e-12
    assert relative_linf(pred, target, mask) < 1e-12


def test_metrics_broadcast_mask():
    target = np.ones((3, 4, 4))
    pred = target * 2.0
    mask = np.ones((1, 4, 4), dtype=bool)
    assert abs(relative_l2(pred, target, mask) - 1.0) < 1e-6
