"""Tests for ``fracturex.analysis.nepin_hook.build_nepin_callbacks``.

Mesh-free: constructs synthetic ``PhaseFieldSystem``-like objects with
scipy sparse ``A`` and residual-form ``F = rhs - A @ d_old`` to verify
that the built callbacks reproduce the NEPIN kernel's expected
behaviour without touching the fracturex assembler.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from fealpy.backend import backend_manager as bm

from fracturex.analysis import (
    NEPINConfig,
    NEPINEliminator,
    build_nepin_callbacks,
)


@dataclass
class _FakePhaseSystem:
    A: Any
    F: Any


def _tridiag(n: int, main: float = 4.0, off: float = -1.0) -> sp.csr_matrix:
    diagonals = [
        np.full(n - 1, off),
        np.full(n, main),
        np.full(n - 1, off),
    ]
    return sp.diags(diagonals, offsets=[-1, 0, 1], format="csr")


# ---------------------------------------------------------------------------
# H1: rhs freezing is correct: R(d_old) == -F
# ---------------------------------------------------------------------------

def test_H1_residual_at_d_old_equals_neg_F():
    """At ``d = d_old`` the callback should return ``-sys.F``, i.e.
    ``A d_old - rhs = A d_old - (F + A d_old) = -F``.
    """
    n = 6
    A = _tridiag(n)
    d_old = np.linspace(0.1, 0.7, n)
    rhs_np = np.arange(1.0, n + 1.0)
    F_np = rhs_np - A @ d_old

    sys_d = _FakePhaseSystem(A=A, F=bm.asarray(F_np, dtype=bm.float64))
    residual, _ = build_nepin_callbacks(sys_d, bm.asarray(d_old, dtype=bm.float64))

    R = bm.to_numpy(residual(bm.asarray(d_old, dtype=bm.float64), None))
    np.testing.assert_allclose(R, -F_np, atol=1e-12)


# ---------------------------------------------------------------------------
# H2: jacobian returns exact sub-block A[S][:, S]
# ---------------------------------------------------------------------------

def test_H2_jacobian_matches_sub_block():
    n = 8
    A = _tridiag(n, main=5.0, off=-2.0)
    d_old = np.zeros(n)
    F_np = np.ones(n)
    sys_d = _FakePhaseSystem(A=A, F=bm.asarray(F_np, dtype=bm.float64))

    _, jacobian = build_nepin_callbacks(sys_d, bm.asarray(d_old, dtype=bm.float64))

    subset = bm.asarray(np.array([1, 3, 5, 7], dtype=np.int64))
    J_SS = bm.to_numpy(jacobian(bm.asarray(d_old, dtype=bm.float64), None, subset))
    expected = A.toarray()[np.ix_([1, 3, 5, 7], [1, 3, 5, 7])]
    np.testing.assert_allclose(J_SS, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# H3: end-to-end with kernel -- local Newton converges + S^c unchanged
# ---------------------------------------------------------------------------

def test_H3_end_to_end_local_newton_converges():
    """Set up a diagonal-heavy A so the local Newton on S converges in
    one step (affine residual). Verify S dofs solve exactly and S^c
    dofs are unchanged.
    """
    n = 5
    A = sp.diags(np.array([10.0, 10.0, 1.0, 1.0, 1.0]), format="csr")
    # Choose rhs so exact solution is [0.1, 0.2, 0.3, 0.4, 0.5]
    x_star = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    rhs_np = A @ x_star

    # Start from d_old with S dofs (indices 0, 1) off, S^c dofs already at x_star
    d_old = np.array([0.0, 0.0, 0.3, 0.4, 0.5])
    F_np = rhs_np - A @ d_old
    sys_d = _FakePhaseSystem(A=A, F=bm.asarray(F_np, dtype=bm.float64))

    residual, jacobian = build_nepin_callbacks(
        sys_d, bm.asarray(d_old, dtype=bm.float64)
    )

    mask = bm.asarray(np.array([True, True, False, False, False]))
    elim = NEPINEliminator(residual, jacobian, NEPINConfig(local_tol=1e-12, max_local_iter=3))
    res = elim.eliminate(
        bm.asarray(d_old, dtype=bm.float64), None, subset_mask=mask
    )

    assert res.converged
    assert res.local_iters == 1  # affine residual -> exact in one Newton step
    assert res.subset_size == 2

    d_corr = bm.to_numpy(res.d_corrected)
    assert math.isclose(d_corr[0], 0.1, rel_tol=1e-10)
    assert math.isclose(d_corr[1], 0.2, rel_tol=1e-10)
    # S^c dofs untouched
    np.testing.assert_allclose(d_corr[2:], d_old[2:], atol=0.0)


# ---------------------------------------------------------------------------
# H4: fealpy CSRTensor path (skipped if fealpy sparse unavailable)
# ---------------------------------------------------------------------------

def test_H4_csrtensor_input_lifted_via_as_scipy_csr():
    """The hook must accept a fealpy CSRTensor via ``as_scipy_csr``; if
    fealpy exposes ``to_scipy`` this exercises that branch.
    """
    try:
        from fealpy.sparse.csr_tensor import CSRTensor  # type: ignore
    except Exception:  # pragma: no cover -- optional path
        pytest.skip("fealpy CSRTensor not importable in this environment")

    n = 4
    A_sci = _tridiag(n)
    # Round-trip via CSRTensor if the constructor accepts scipy.
    try:
        A_ct = CSRTensor.from_scipy(A_sci)  # type: ignore[attr-defined]
    except Exception:
        pytest.skip("CSRTensor.from_scipy not available -- CSRTensor lift path untested")

    d_old = np.linspace(0.0, 0.3, n)
    F_np = np.ones(n) - A_sci @ d_old
    sys_d = _FakePhaseSystem(A=A_ct, F=bm.asarray(F_np, dtype=bm.float64))

    residual, _ = build_nepin_callbacks(sys_d, bm.asarray(d_old, dtype=bm.float64))
    R = bm.to_numpy(residual(bm.asarray(d_old, dtype=bm.float64), None))
    np.testing.assert_allclose(R, -F_np, atol=1e-12)
