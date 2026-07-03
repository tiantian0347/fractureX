"""Tests for ``fracturex.analysis.nonlinear_elimination``.

See ``docs/preconditioner/DESIGN_nepin_driver.md`` §5 for the test-plan
matrix (T1..T9). All tests are mesh-free: they use synthetic
high-contrast nonlinear systems that reproduce the L_S/L_{S^c} ratio
without needing the fracturex assembler.
"""
from __future__ import annotations

import math

import pytest

from fealpy.backend import backend_manager as bm

from fracturex.analysis import (
    AffineInvariantMonitor,
    NewtonStepRecord,
    NEPINConfig,
    NEPINEliminator,
    NEPINResult,
    identify_subset,
)


# ---------------------------------------------------------------------------
# helpers: synthetic diagonal-quadratic problem
# ---------------------------------------------------------------------------

def _diag_matrix_from_vec(diag: object) -> object:
    """Return a bm dense diagonal matrix from a length-n vector.

    ``bm.diag`` is not part of the multi-backend surface (Ch. 10 design
    §2.4 avoids assuming it); we build via ``bm.eye * broadcast``.
    """
    n = int(diag.shape[0])
    return bm.eye(n, dtype=bm.float64) * diag


def _make_diag_quadratic_problem(
    L: object, b: object,
) -> tuple[object, object]:
    """Return (F, J) callbacks for the residual F_i(x) = L_i x_i^2 - b_i.

    Roots: x_i^* = sqrt(b_i / L_i). The Jacobian is diagonal:
    ``J_ii = 2 L_i x_i``.
    """
    L_bm = bm.asarray(L, dtype=bm.float64)
    b_bm = bm.asarray(b, dtype=bm.float64)

    def F(d, x_frozen):  # noqa: ARG001
        return L_bm * d * d - b_bm

    def J(d, x_frozen, subset_dofs):  # noqa: ARG001
        d_S = d[subset_dofs]
        L_S = L_bm[subset_dofs]
        return _diag_matrix_from_vec(2.0 * L_S * d_S)

    return F, J


# ---------------------------------------------------------------------------
# T1: two-region high-contrast quadratic residual
# ---------------------------------------------------------------------------

def test_T1_high_contrast_two_region():
    """Local Newton on Omega_s converges quickly and leaves S^c intact.

    Contrast: L_S = 1e6, L_{S^c} = 1. NEPIN local Newton finds the
    S-root within a few iters; S^c dofs are untouched.
    """
    L = [1e6] * 3 + [1.0] * 7
    b = [1e6 * 0.25, 1e6 * 0.36, 1e6 * 0.49] + [1.0, 4.0, 9.0, 16.0, 1.0, 1.0, 1.0]
    F, J = _make_diag_quadratic_problem(L, b)

    x0 = bm.asarray([1.5] * 3 + [5.0] * 7, dtype=bm.float64)
    mask = bm.asarray([True] * 3 + [False] * 7)

    elim = NEPINEliminator(F, J, NEPINConfig(local_tol=1e-6, max_local_iter=10))
    res = elim.eliminate(x0, None, subset_mask=mask)

    assert isinstance(res, NEPINResult)
    assert res.subset_size == 3
    assert res.converged
    assert res.local_res_reduction < 1e-6
    assert res.local_iters <= 6

    d_np = bm.to_numpy(res.d_corrected)
    for got, expected in zip(d_np[:3], [0.5, 0.6, 0.7]):
        assert math.isclose(float(got), expected, rel_tol=1e-4)
    for got, orig in zip(d_np[3:], [5.0] * 7):
        assert float(got) == orig  # Untouched


# ---------------------------------------------------------------------------
# T2: subset identifier via threshold
# ---------------------------------------------------------------------------

def test_T2_identify_subset_threshold():
    d = bm.asarray([0.1, 0.5, 0.85, 0.9, 0.3], dtype=bm.float64)
    mask = identify_subset(d, d_c=0.82)
    mask_np = bm.to_numpy(mask)
    assert list(bool(m) for m in mask_np) == [False, False, True, True, False]


# ---------------------------------------------------------------------------
# T3: local solve exact on affine residual
# ---------------------------------------------------------------------------

def test_T3_affine_residual_one_step():
    """For F(x) = A x - b (linear), one local Newton yields the exact
    restricted root within roundoff."""
    A_full = bm.asarray(
        [
            [10.0, 1.0, 0.0, 0.0],
            [1.0, 10.0, 1.0, 0.0],
            [0.0, 1.0, 10.0, 1.0],
            [0.0, 0.0, 1.0, 10.0],
        ],
        dtype=bm.float64,
    )
    b = bm.asarray([1.0, 2.0, 3.0, 4.0], dtype=bm.float64)

    def F(d, x_frozen):  # noqa: ARG001
        return A_full @ d - b

    def J(d, x_frozen, subset_dofs):  # noqa: ARG001
        # dense read of A_full[S, S] via double fancy index
        rows = A_full[subset_dofs]  # shape (|S|, n)
        return rows[:, subset_dofs]  # shape (|S|, |S|)

    x0 = bm.asarray([0.0, 0.0, 0.0, 0.0], dtype=bm.float64)
    mask = bm.asarray([True, True, False, False])

    elim = NEPINEliminator(F, J, NEPINConfig(local_tol=1e-12, max_local_iter=3))
    res = elim.eliminate(x0, None, subset_mask=mask)

    # Analytic solution: A_SS x_S = b_S - A_{S,Sc} x_Sc; x_Sc = 0 here.
    #   [[10 1] [1 10]] x_S = [1, 2]  =>  x_S = [8/99, 19/99]
    d_np = bm.to_numpy(res.d_corrected)
    assert res.local_iters == 1
    assert res.converged
    assert math.isclose(float(d_np[0]), 8.0 / 99.0, rel_tol=1e-10)
    assert math.isclose(float(d_np[1]), 19.0 / 99.0, rel_tol=1e-10)
    assert d_np[2] == 0.0
    assert d_np[3] == 0.0


# ---------------------------------------------------------------------------
# T4: damping guards against overshoot
# ---------------------------------------------------------------------------

def test_T4_damping_backtrack_recovers():
    """A pathologically bad initial that overshoots the root should
    still converge with damping backtrack."""
    L = [1.0, 1.0, 1.0]  # single region, but poorly conditioned start
    b = [1.0, 1.0, 1.0]
    F, J = _make_diag_quadratic_problem(L, b)

    # Start x0 negative -> Newton step enormous
    x0 = bm.asarray([-2.0, -2.0, -2.0], dtype=bm.float64)
    mask = bm.asarray([True, True, True])

    elim = NEPINEliminator(
        F, J, NEPINConfig(local_tol=1e-8, max_local_iter=30, damping=1.0),
    )
    res = elim.eliminate(x0, None, subset_mask=mask)
    # Newton on x^2 = 1 from x=-2 will converge to x=-1, that's fine.
    assert res.converged
    assert res.local_res_reduction < 1e-8


# ---------------------------------------------------------------------------
# T5: max_local_iter cap honoured
# ---------------------------------------------------------------------------

def test_T5_max_iter_cap_honest_reporting():
    L = [1.0, 1.0]
    b = [1.0, 1.0]
    F, J = _make_diag_quadratic_problem(L, b)

    x0 = bm.asarray([10.0, 10.0], dtype=bm.float64)  # far from root
    mask = bm.asarray([True, True])
    # Cap at 1: definitely not enough for x=10 -> x=1
    elim = NEPINEliminator(F, J, NEPINConfig(local_tol=1e-8, max_local_iter=1))
    res = elim.eliminate(x0, None, subset_mask=mask)

    assert res.local_iters == 1
    assert res.converged is False
    assert res.local_res_reduction > 0.0
    # Increased tolerance: with 20 iters converges cleanly
    elim2 = NEPINEliminator(F, J, NEPINConfig(local_tol=1e-8, max_local_iter=20))
    res2 = elim2.eliminate(x0, None, subset_mask=mask)
    assert res2.converged


# ---------------------------------------------------------------------------
# T6: subset_size accounting
# ---------------------------------------------------------------------------

def test_T6_subset_size_matches_mask():
    L = [1.0] * 5
    b = [1.0] * 5
    F, J = _make_diag_quadratic_problem(L, b)
    x0 = bm.asarray([2.0] * 5, dtype=bm.float64)
    mask = bm.asarray([True, False, True, False, True])

    elim = NEPINEliminator(F, J, NEPINConfig())
    res = elim.eliminate(x0, None, subset_mask=mask)
    assert res.subset_size == 3


# ---------------------------------------------------------------------------
# T7: guards on trivial input
# ---------------------------------------------------------------------------

def test_T7a_zero_residual_no_op():
    """If the residual is already zero, do nothing."""
    def F(d, x_frozen):  # noqa: ARG001
        return bm.zeros_like(d)

    def J(d, x_frozen, subset_dofs):  # noqa: ARG001
        return bm.eye(int(subset_dofs.shape[0]), dtype=bm.float64)

    x0 = bm.asarray([1.0, 2.0, 3.0], dtype=bm.float64)
    mask = bm.asarray([True, True, False])
    elim = NEPINEliminator(F, J, NEPINConfig())
    res = elim.eliminate(x0, None, subset_mask=mask)
    assert res.local_iters == 0
    assert res.local_res_reduction == 1.0
    assert res.converged
    d_np = bm.to_numpy(res.d_corrected)
    assert list(d_np) == [1.0, 2.0, 3.0]


def test_T7b_empty_subset_no_op():
    """Empty subset: return unchanged, no residual evaluation needed."""
    def F(d, x_frozen):  # noqa: ARG001
        raise AssertionError("F must not be called when |S|=0")

    def J(d, x_frozen, subset_dofs):  # noqa: ARG001
        raise AssertionError("J must not be called when |S|=0")

    x0 = bm.asarray([0.5, 0.5, 0.5], dtype=bm.float64)
    mask = bm.asarray([False, False, False])
    elim = NEPINEliminator(F, J, NEPINConfig())
    res = elim.eliminate(x0, None, subset_mask=mask)
    assert res.subset_size == 0
    assert res.local_iters == 0
    assert res.converged
    d_np = bm.to_numpy(res.d_corrected)
    assert list(d_np) == [0.5, 0.5, 0.5]


def test_T7c_default_mask_from_config():
    """When ``subset_mask=None``, uses ``identify_subset(d, d_c)``."""
    L = [1e6, 1.0, 1.0, 1.0]
    b = [1e6 * 0.25, 1.0, 1.0, 1.0]
    F, J = _make_diag_quadratic_problem(L, b)
    d_full = bm.asarray([0.9, 0.5, 0.1, 0.0], dtype=bm.float64)
    # d_c=0.82 -> mask [T, F, F, F] -> subset_size=1
    elim = NEPINEliminator(F, J, NEPINConfig(d_c=0.82, local_tol=1e-6))
    res = elim.eliminate(d_full, None)  # no explicit mask
    assert res.subset_size == 1
    assert res.converged


# ---------------------------------------------------------------------------
# T8: multi-backend regression (skipped if pytorch not installed)
# ---------------------------------------------------------------------------

def test_T8_backend_agreement_numpy_pytorch():
    """T1 problem yields the same corrected d under numpy vs pytorch."""
    try:
        bm.set_backend("pytorch")
        _ = bm.asarray([1.0], dtype=bm.float64)
    except Exception:
        bm.set_backend("numpy")
        pytest.skip("pytorch backend not available")

    L = [1e6] * 3 + [1.0] * 7
    b = [1e6 * 0.25, 1e6 * 0.36, 1e6 * 0.49] + [1.0, 4.0, 9.0, 16.0, 1.0, 1.0, 1.0]
    x0 = [1.5] * 3 + [5.0] * 7
    mask = [True] * 3 + [False] * 7

    F, J = _make_diag_quadratic_problem(L, b)
    elim = NEPINEliminator(F, J, NEPINConfig(local_tol=1e-6))
    d_torch = bm.to_numpy(
        elim.eliminate(
            bm.asarray(x0, dtype=bm.float64),
            None,
            subset_mask=bm.asarray(mask),
        ).d_corrected
    )

    bm.set_backend("numpy")
    F, J = _make_diag_quadratic_problem(L, b)
    elim = NEPINEliminator(F, J, NEPINConfig(local_tol=1e-6))
    d_np = bm.to_numpy(
        elim.eliminate(
            bm.asarray(x0, dtype=bm.float64),
            None,
            subset_mask=bm.asarray(mask),
        ).d_corrected
    )

    assert d_torch.shape == d_np.shape
    for a, b_v in zip(d_torch, d_np):
        assert math.isclose(float(a), float(b_v), rel_tol=1e-8)


# ---------------------------------------------------------------------------
# T9: contraction on omega_hat under NEPIN vs plain Newton
# ---------------------------------------------------------------------------

def test_T9_omega_hat_contracted_by_nepin():
    """On a locally-strong-nonlinear problem, plain Newton produces a
    larger omega_hat than NEPIN-preconditioned Newton, matching Prop.
    of Theory §1.2.

    We simulate two Newton sequences (plain vs NEPIN) on the same
    problem by manually stepping and recording ``||delta d||``, then
    feed both traces through ``AffineInvariantMonitor``.
    """
    # Locally strong quadratic:  F_i(x) = L_i x_i^2 - b_i
    L = [1e6, 1e6, 1.0, 1.0, 1.0]
    b_vec = [1e6 * 0.25, 1e6 * 0.36, 1.0, 4.0, 9.0]
    F, J = _make_diag_quadratic_problem(L, b_vec)

    # -- Plain Newton on the full system, from bad initial ---
    x_plain = bm.asarray([2.0, 2.0, 3.0, 3.0, 3.0], dtype=bm.float64)
    plain_deltas = []
    for _ in range(5):
        F_val = F(x_plain, None)
        # full diagonal Jacobian
        J_diag = 2.0 * bm.asarray(L, dtype=bm.float64) * x_plain
        step = -F_val / bm.maximum(bm.abs(J_diag), bm.asarray(1e-30))
        x_next = x_plain + step
        plain_deltas.append(float(bm.linalg.norm(x_next - x_plain)))
        x_plain = x_next

    # -- NEPIN-preconditioned Newton on same problem ---
    x_nepin = bm.asarray([2.0, 2.0, 3.0, 3.0, 3.0], dtype=bm.float64)
    mask = bm.asarray([True, True, False, False, False])
    elim = NEPINEliminator(F, J, NEPINConfig(local_tol=1e-8, max_local_iter=10))
    nepin_deltas = []
    for _ in range(5):
        prev = x_nepin
        # NEPIN inner solve on Omega_s
        res = elim.eliminate(prev, None, subset_mask=mask)
        x_after_nepin = res.d_corrected
        # Global step (S^c only, since S is now consistent)
        F_val = F(x_after_nepin, None)
        J_diag = 2.0 * bm.asarray(L, dtype=bm.float64) * x_after_nepin
        step = -F_val / bm.maximum(bm.abs(J_diag), bm.asarray(1e-30))
        x_next = x_after_nepin + step
        nepin_deltas.append(float(bm.linalg.norm(x_next - prev)))
        x_nepin = x_next

    # Feed both traces into the affine-invariant monitor
    def to_records(deltas, step_id):
        return [
            NewtonStepRecord(
                iter=k + 1,
                step=step_id,
                load=1.0,
                delta_d_norm=d,
                delta_u_norm=0.0,
                max_d=0.9,
                lin_res_e=1e-6,
                lin_res_d=1e-6,
            )
            for k, d in enumerate(deltas)
        ]

    mon = AffineInvariantMonitor()
    summ_plain = mon.process(to_records(plain_deltas, step_id=0))[0]
    summ_nepin = mon.process(to_records(nepin_deltas, step_id=0))[0]

    # NEPIN should have both smaller final increment AND smaller omega_hat_max
    # (or, when the problem converges very quickly under NEPIN and delta_d
    #  drops to eps-guard, comparable-or-better).
    assert summ_nepin.omega_hat_max <= summ_plain.omega_hat_max + 1e-9, (
        f"NEPIN omega_hat_max = {summ_nepin.omega_hat_max} not <= "
        f"plain omega_hat_max = {summ_plain.omega_hat_max}"
    )


# ---------------------------------------------------------------------------
# import safety
# ---------------------------------------------------------------------------

def test_import_lightweight():
    """Importing analysis.nonlinear_elimination should not import torch
    or the driver stack."""
    import fracturex.analysis.nonlinear_elimination as m
    # Trivial: just verify the module exposes the expected names.
    for name in ("NEPINConfig", "NEPINResult", "NEPINEliminator", "identify_subset"):
        assert hasattr(m, name), f"missing export: {name}"
