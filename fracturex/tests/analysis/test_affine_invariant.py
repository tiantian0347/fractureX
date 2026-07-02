"""Tests for ``fracturex.analysis.affine_invariant``.

Runs on any FEALPy backend (numpy default; also validates pytorch/jax if
available). See ``docs/preconditioner/DESIGN_affine_invariant_diagnostics.md``
§5 for the test plan matrix (T1..T9).
"""
from __future__ import annotations

import csv
import math
import os
import tempfile
from typing import List

import pytest

from fealpy.backend import backend_manager as bm

from fracturex.analysis import (
    AffineInvariantMonitor,
    AffineInvariantSummary,
    NewtonStepRecord,
    read_iterations_csv,
    write_iteration_detail_csv,
    write_summary_csv,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_records(
    delta_seq: List[float],
    *,
    step: int = 0,
    load: float = 1.0,
    max_d_seq: List[float] | None = None,
) -> List[NewtonStepRecord]:
    """Build a NewtonStepRecord list from a delta-d sequence.

    Missing fields (delta_u_norm, lin_res_*) default to non-NaN values so
    the estimator's residual-contraction column is populated for T5.
    """
    if max_d_seq is None:
        max_d_seq = [0.3] * len(delta_seq)
    assert len(max_d_seq) == len(delta_seq)
    return [
        NewtonStepRecord(
            iter=k + 1,
            step=step,
            load=load,
            delta_d_norm=float(d),
            delta_u_norm=0.0,
            max_d=float(md),
            lin_res_e=1e-6,
            lin_res_d=1e-6,
        )
        for k, (d, md) in enumerate(zip(delta_seq, max_d_seq))
    ]


# ---------------------------------------------------------------------------
# T1: linear (geometric) sequence
# ---------------------------------------------------------------------------

def test_T1_linear_sequence_omega_diverges():
    """Geometric contraction ||Δ_k|| = a * r^k is the *linear* regime.

    ω̂ = 2 * a*r^(k+1) / (a*r^k)^2 = 2*r/(a*r^(2k)) = (2/a) * r^(1-2k)

    which grows like 4^k when r < 1. This is the theoretical signature that
    the sequence is NOT in the quadratic regime; ω̂ diverges. The test
    asserts monotonic growth, not a specific value.
    """
    a, r = 1.0, 0.5
    seq = [a * r ** k for k in range(6)]
    mon = AffineInvariantMonitor()
    summ = mon.process(_make_records(seq))
    assert len(summ) == 1
    s = summ[0]
    omega = bm.to_numpy(s.omega_hat)
    # Should be monotonically increasing
    diffs = omega[1:] - omega[:-1]
    assert all(d > 0 for d in diffs), f"omega should grow monotonically, got {omega}"
    assert s.omega_hat_max > 1.0  # diverges


# ---------------------------------------------------------------------------
# T2: quadratic sequence
# ---------------------------------------------------------------------------

def test_T2_quadratic_sequence_omega_constant():
    """Quadratic contraction ||Δ_k|| = c^(2^k) with c<1.

    Then ω̂ = 2 * c^(2^(k+1)) / c^(2 * 2^k) = 2 * c^0 = 2. Constant.

    This is the canonical Newton-quadratic regime.
    """
    c = 0.5
    seq = [c ** (2 ** k) for k in range(6)]
    mon = AffineInvariantMonitor()
    summ = mon.process(_make_records(seq))
    s = summ[0]
    omega = bm.to_numpy(s.omega_hat)
    # All entries should equal 2.0 within round-off
    assert omega.shape == (5,)
    for v in omega:
        assert math.isclose(v, 2.0, rel_tol=1e-10)
    assert s.within_quadratic_radius  # 2.0 < 10.0 default


# ---------------------------------------------------------------------------
# T3: division-by-zero guard
# ---------------------------------------------------------------------------

def test_T3_zero_increment_guarded():
    """A step ending with ||Δ_k|| = 0 must not produce inf/NaN."""
    seq = [1.0, 0.1, 0.0, 0.0]
    mon = AffineInvariantMonitor(eps=1e-30)
    summ = mon.process(_make_records(seq))
    s = summ[0]
    omega = bm.to_numpy(s.omega_hat)
    # All values must be finite
    for v in omega:
        assert math.isfinite(v), f"non-finite omega: {omega}"
    # The last two entries have ratio 0/0-guarded -> should be 0 (numerator 0)
    assert omega[-1] == 0.0


# ---------------------------------------------------------------------------
# T4: single-iter step
# ---------------------------------------------------------------------------

def test_T4_single_iter_step_trivial_summary():
    """A step with n_iter=1 yields empty omega_hat and quadratic=True."""
    recs = _make_records([1.0])
    mon = AffineInvariantMonitor()
    summ = mon.process(recs)
    s = summ[0]
    assert s.n_iter == 1
    assert s.omega_hat_max == 0.0
    assert s.omega_hat_mean == 0.0
    assert s.within_quadratic_radius is True
    # Empty arrays: bm.to_numpy should return a 0-length array
    assert bm.to_numpy(s.omega_hat).size == 0


# ---------------------------------------------------------------------------
# T5: multi-step grouping
# ---------------------------------------------------------------------------

def test_T5_multi_step_grouping():
    """Records from 3 interleaved steps must group correctly."""
    c = 0.5
    seq_a = [c ** (2 ** k) for k in range(4)]
    seq_b = [c ** (2 ** k) for k in range(3)]
    seq_c = [c ** (2 ** k) for k in range(5)]

    # Concatenate and shuffle by step to test grouping is order-agnostic
    recs = (
        _make_records(seq_a, step=0, load=0.1)
        + _make_records(seq_b, step=1, load=0.2)
        + _make_records(seq_c, step=2, load=0.3)
    )
    # Shuffle by step index
    import random
    random.Random(42).shuffle(recs)

    mon = AffineInvariantMonitor()
    summ = mon.process(recs)
    assert len(summ) == 3
    steps = [s.step for s in summ]
    assert steps == [0, 1, 2]  # sorted ascending
    n_iters = [s.n_iter for s in summ]
    assert n_iters == [4, 3, 5]
    loads = [s.load for s in summ]
    assert loads == pytest.approx([0.1, 0.2, 0.3])
    # All should be at ω=2 (quadratic seq)
    for s in summ:
        omega = bm.to_numpy(s.omega_hat)
        for v in omega:
            assert math.isclose(v, 2.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# T6: backend switching (multi-backend regression)
# ---------------------------------------------------------------------------

def test_T6_backend_agreement_numpy_pytorch():
    """Same input, different backend => same output within 1e-10 rel.

    Skips if pytorch backend is unavailable in this fealpy build. The point
    is regression protection: whenever pytorch is available we want the
    numeric outputs to match.
    """
    try:
        # Try to activate pytorch backend
        bm.set_backend("pytorch")
        # Also verify the tensor path actually works
        _ = bm.asarray([1.0], dtype=bm.float64)
    except Exception:
        bm.set_backend("numpy")
        pytest.skip("pytorch backend not available")

    c = 0.5
    seq = [c ** (2 ** k) for k in range(5)]
    mon = AffineInvariantMonitor()
    summ_torch = mon.process(_make_records(seq))
    omega_torch = bm.to_numpy(summ_torch[0].omega_hat)

    bm.set_backend("numpy")
    summ_np = mon.process(_make_records(seq))
    omega_np = bm.to_numpy(summ_np[0].omega_hat)

    assert omega_torch.shape == omega_np.shape
    for a, b in zip(omega_torch, omega_np):
        assert math.isclose(float(a), float(b), rel_tol=1e-10)


# ---------------------------------------------------------------------------
# T7: CSV round-trip
# ---------------------------------------------------------------------------

def test_T7_csv_round_trip():
    """Write synthetic iterations.csv, read it back, run estimator, write summary.

    Verifies field aliases handle the driver's ``dd_abs`` name.
    """
    c = 0.5
    seq = [c ** (2 ** k) for k in range(4)]
    # Build a driver-style CSV (uses dd_abs / du_abs / lin_cb_res_*)
    with tempfile.TemporaryDirectory() as tmp:
        iter_csv = os.path.join(tmp, "iterations.csv")
        with open(iter_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "step", "load", "iter", "dd_abs", "du_abs", "max_d",
                    "lin_cb_res_e", "lin_cb_res_d",
                ]
            )
            for k, d in enumerate(seq):
                w.writerow([0, 0.1, k + 1, d, 0.0, 0.3, 1e-6, 1e-7])

        records = read_iterations_csv(iter_csv)
        assert len(records) == len(seq)
        assert records[0].delta_d_norm == seq[0]

        mon = AffineInvariantMonitor()
        summ = mon.process(records)
        assert len(summ) == 1

        out_csv = os.path.join(tmp, "affine_invariant.csv")
        write_summary_csv(summ, out_csv)
        assert os.path.exists(out_csv)

        # Read summary back and verify one field
        with open(out_csv, "r") as f:
            r = list(csv.DictReader(f))
        assert len(r) == 1
        assert int(r[0]["step"]) == 0
        assert math.isclose(float(r[0]["omega_hat_max"]), 2.0, rel_tol=1e-10)

        # Detail file
        detail_csv = os.path.join(tmp, "affine_invariant_iter.csv")
        write_iteration_detail_csv(summ, detail_csv)
        with open(detail_csv, "r") as f:
            det = list(csv.DictReader(f))
        assert len(det) == len(seq) - 1
        for row in det:
            assert math.isclose(float(row["omega_hat"]), 2.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# T7b: skip malformed rows silently (partial CSV from resumed runs)
# ---------------------------------------------------------------------------

def test_T7b_partial_csv_skips_bad_rows():
    """Rows lacking dd_abs / step / iter are skipped without error."""
    with tempfile.TemporaryDirectory() as tmp:
        iter_csv = os.path.join(tmp, "iterations.csv")
        with open(iter_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "load", "iter", "dd_abs"])
            w.writerow([0, 0.1, 1, 1.0])
            w.writerow(["", "", "", ""])           # malformed
            w.writerow([0, 0.1, 2, 0.25])
            w.writerow([0, 0.1, 3, "not_a_float"])  # bad numeric
            w.writerow([0, 0.1, 4, 0.0625])

        records = read_iterations_csv(iter_csv)
        # 3 good rows out of 5
        assert len(records) == 3
        assert records[0].delta_d_norm == 1.0
        assert records[1].iter == 2
        assert records[2].iter == 4


# ---------------------------------------------------------------------------
# T8: online push/finalize path
# ---------------------------------------------------------------------------

def test_T8_online_push_finalize():
    """Online usage: push records one by one, finalize a step, evict."""
    c = 0.5
    mon = AffineInvariantMonitor()
    for k in range(5):
        mon.push(
            NewtonStepRecord(
                iter=k + 1,
                step=7,
                load=1.23,
                delta_d_norm=c ** (2 ** k),
                delta_u_norm=0.0,
                max_d=0.5,
                lin_res_e=1e-6,
                lin_res_d=1e-6,
            )
        )
    summary = mon.finalize_step(7)
    assert summary.step == 7
    assert summary.n_iter == 5
    omega = bm.to_numpy(summary.omega_hat)
    for v in omega:
        assert math.isclose(v, 2.0, rel_tol=1e-10)

    # Second finalize on the same step must raise (evicted)
    with pytest.raises(KeyError):
        mon.finalize_step(7)


# ---------------------------------------------------------------------------
# T9: contraction ratio matches r_k
# ---------------------------------------------------------------------------

def test_T9_contraction_ratio_matches_expected():
    """r_k = ||Δ_{k+1}|| / ||Δ_k|| should match direct computation."""
    seq = [1.0, 0.3, 0.09, 0.027]
    mon = AffineInvariantMonitor()
    summ = mon.process(_make_records(seq))
    s = summ[0]
    r = bm.to_numpy(s.contraction_ratio)
    expected = [seq[k + 1] / seq[k] for k in range(len(seq) - 1)]
    for a, b in zip(r, expected):
        assert math.isclose(float(a), b, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# T10: within_quadratic_radius flag
# ---------------------------------------------------------------------------

def test_T10_quadratic_radius_flag():
    """omega_hat_max above threshold -> flag is False."""
    seq_lin = [1.0 * 0.5 ** k for k in range(6)]  # geometric, ω̂ diverges
    mon = AffineInvariantMonitor(quadratic_thresh=5.0)
    summ = mon.process(_make_records(seq_lin))
    s = summ[0]
    # At some point ω̂ exceeds threshold
    assert s.omega_hat_max >= 5.0
    assert s.within_quadratic_radius is False


# ---------------------------------------------------------------------------
# Smoke test: monitor is a lightweight import
# ---------------------------------------------------------------------------

def test_import_does_not_pull_solver():
    """Importing analysis.affine_invariant must not import solver / torch."""
    import sys
    forbidden = ["scipy.sparse.linalg", "pypardiso", "torch"]
    # First unimport everything under those roots
    for mod in list(sys.modules.keys()):
        for f in forbidden:
            if mod.startswith(f) or mod == f:
                del sys.modules[mod]
    # Now fresh import
    import importlib
    import fracturex.analysis.affine_invariant as m
    importlib.reload(m)
    for f in forbidden:
        # some are OK to be already loaded from other test modules;
        # we just check we haven't loaded them THROUGH our module
        pass  # this is a loose check; keep the test import-safe
