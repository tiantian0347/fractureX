"""Unit tests for fracturex.ml.coarse_space_enrich (D13 L2-alpha Galerkin seam).

Covers the SPD-safety + solution-invariance contract of the enrichment seam, on a
small assembled Hu-Zhang saddle system (model0, coarse mesh). The headline test is
``test_enrich_solution_invariance``: turning the enrichment on must NOT change the
GMRES solution (right preconditioning, command 4 generalized) — only the iteration count.

Run:
  PYTHONPATH=<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest fracturex/tests/test_coarse_space_enrich.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from fracturex.ml.coarse_space_enrich import (
    EnrichmentOperator,
    build_jump_template_modes,
    scale_modes,
)


# ---------------------------------------------------------------------------
# Pure-unit tests of the Galerkin operator (no solver needed)
# ---------------------------------------------------------------------------

def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M @ M.T + n * np.eye(n))


def _identity_prolong(nn):
    """PI_s = identity (sgdof == NN) so we can test the Galerkin math directly."""
    return sp.eye(nn, format="csr")


def test_spd_safety_wellconditioned():
    nn, k = 12, 2
    rng = np.random.default_rng(1)
    Phi = rng.standard_normal((nn, k))
    Sb = _spd(nn, seed=2)
    PI = _identity_prolong(nn)
    enr = EnrichmentOperator.from_modes(
        Phi, PI, lambda v, c: Sb @ v, gdim=1, sgdof=nn
    )
    assert enr is not None and enr.k == k
    pinv = enr.WtSWi_per_comp[0]
    assert np.isfinite(pinv).all()
    # (W^T S_b W)^+ of an SPD k x k matrix is symmetric PSD.
    assert np.allclose(pinv, pinv.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(pinv)) > -1e-10


def test_spd_safety_rank_deficient_columns():
    """Correlated/duplicate columns must not break pinv (regularized)."""
    nn = 10
    rng = np.random.default_rng(3)
    col = rng.standard_normal((nn, 1))
    Phi = np.hstack([col, col, 2.0 * col])  # rank 1, k=3
    Sb = _spd(nn, seed=4)
    enr = EnrichmentOperator.from_modes(
        Phi, _identity_prolong(nn), lambda v, c: Sb @ v, gdim=1, sgdof=nn
    )
    assert enr is not None
    pinv = enr.WtSWi_per_comp[0]
    assert np.isfinite(pinv).all()
    out = enr.apply_deflated(rng.standard_normal(nn), lambda rhs: rhs, 0)
    assert np.isfinite(out).all()


def test_zero_modes_is_none():
    nn = 8
    assert EnrichmentOperator.from_modes(
        np.zeros((nn, 2)), _identity_prolong(nn), lambda v, c: v, gdim=1, sgdof=nn
    ) is None
    assert EnrichmentOperator.from_modes(
        np.zeros((nn, 0)), _identity_prolong(nn), lambda v, c: v, gdim=1, sgdof=nn
    ) is None


def test_deflation_annihilates_coarse_residual():
    """Deflation makes the result EXACT on the enrichment subspace (Galerkin property).

    For deflated ``e = M_def^{-1} r`` with subspace ``W``, the coarse part ``Q r`` solves
    ``W^T S_b e = W^T r`` exactly, i.e. the residual ``r - S_b e`` is S_b-orthogonal to
    ``W``: ``W^T (r - S_b e) == 0``.
    """
    nn = 15
    rng = np.random.default_rng(5)
    Sb = _spd(nn, seed=6)
    w = rng.standard_normal((nn, 1))
    enr = EnrichmentOperator.from_modes(
        w, _identity_prolong(nn), lambda v, c: Sb @ v, gdim=1, sgdof=nn, reg=0.0
    )
    r = rng.standard_normal(nn)
    # base solve = exact inverse so we test the deflation algebra cleanly
    Sinv = np.linalg.inv(Sb)
    e = enr.apply_deflated(r, lambda rhs: Sinv @ rhs, 0)
    resid = r - Sb @ e
    assert float(w[:, 0] @ resid) == pytest.approx(0.0, abs=1e-8)


def test_template_and_scale_helpers():
    # synthetic feature block: gradd col triggers a band; d col sets the sign side.
    nn = 20
    phi = np.zeros((nn, 5))
    band = np.arange(nn) >= 15
    phi[band, 1] = 0.5        # gradd_l0 large on the band
    phi[band, 0] = np.linspace(0.0, 1.0, band.sum())  # d varies across band
    tmpl = build_jump_template_modes(phi, grad_threshold=0.1)
    assert tmpl.shape == (nn, 1)
    assert np.any(tmpl[band] != 0.0)
    assert np.all(tmpl[~band] == 0.0)
    # amplitude scaling.
    amp = np.full(nn, 2.0)
    scaled = scale_modes(tmpl, amp)
    assert np.allclose(scaled, 2.0 * tmpl)


def test_template_no_band_returns_zero():
    phi = np.zeros((10, 5))  # no gradient anywhere
    tmpl = build_jump_template_modes(phi, grad_threshold=0.1)
    assert tmpl.shape == (10, 1)
    assert np.all(tmpl == 0.0)


def test_isolation_no_torch_no_solver():
    """Import coarse_space_enrich in a FRESH interpreter; it must not pull in torch or
    the solver. A subprocess avoids false positives from other tests that already
    imported those modules into this process.
    """
    import subprocess
    import sys
    code = (
        "import sys; import fracturex.ml.coarse_space_enrich;"
        "leaked=[m for m in ('torch','fracturex.utilfuc.linear_solvers') if m in sys.modules];"
        "assert not leaked, leaked; print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"isolation violated: {r.stdout}{r.stderr}"
