"""Unit tests for the M3b S-tier acceptance-gate policy.

The tests exercise the pure decision rule only. Checkpoint loading and metric
evaluation are covered by ``test_learn_m1_smoke.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[2] / "scripts" / "paper_operator_learning" / "audit_m3b_s_sweep.py"
_SPEC = importlib.util.spec_from_file_location("audit_m3b_s_sweep", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_acceptance_gate = _MODULE.evaluate_acceptance_gate


def _row(lambda_eq, residual, damage, stress, peak):
    return {
        "lambda_eq": lambda_eq,
        "equilibrium_residual_pred_fd": residual,
        "relative_l2": damage,
        "sigma_relative_l2": stress,
        "peak_load_error": peak,
    }


def test_gate_accepts_balance_gain_with_controlled_quality_cost():
    rows = [
        _row(0.0, 1.0, 0.30, 0.40, 0.10),
        _row(0.01, 0.5, 0.31, 0.41, 0.104),
    ]
    gate = evaluate_acceptance_gate(rows, max_quality_degradation=0.05)
    assert gate["passed"] is True
    assert gate["passing_lambda_eq"] == [0.01]


def test_gate_rejects_residual_gain_that_damages_prediction_quality():
    rows = [
        _row(0.0, 1.0, 0.30, 0.40, 0.10),
        _row(0.01, 0.2, 0.45, 0.41, 0.104),
    ]
    gate = evaluate_acceptance_gate(rows, max_quality_degradation=0.05)
    assert gate["passed"] is False
    assert gate["candidate_checks"][0]["residual_ok"] is True
    assert gate["candidate_checks"][0]["quality_ok"] is False
