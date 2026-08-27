"""Tests for the descriptive Schur-coupling correlation diagnostic."""

from __future__ import annotations

import numpy as np

from scripts.paper_solver.analyze_pre_switch_schur_coupling import analyze_records


def _row(load: str, index: int) -> dict[str, str]:
    """Build a finite two-candidate diagnostic row."""
    value = str(index + 1)
    return {
        "label": "test",
        "load": load,
        "pre_switch_external_schur_coupling_mean": value,
        "pre_switch_external_schur_coupling_max": value,
        "pre_switch_external_schur_coupling_local_condition_number": str(10 + index),
        "coupled_condition_number": str(20 + index),
        "survival_factor": str(0.5 + 0.1 * index),
        "coupled_patch_dofs": str(10 + index),
        "total_work": str(100 + 10 * index),
        "krylov_iterations": str(5 + index),
        "physical_residual_evaluations": str(6 + index),
    }


def test_correlation_diagnostic_is_state_local() -> None:
    """Each state yields one correlation record per feature and target."""
    rows = [_row("0.1", 0), _row("0.1", 1), _row("0.2", 0), _row("0.2", 1)]
    records, summary = analyze_records(rows)
    assert len(records) == 2 * 6 * 3
    assert summary["status"] == "diagnostic_only"
    assert summary["uses_post_solve_features"] is False
    assert all(np.isclose(record["pearson_correlation"], 1.0) for record in records)


def test_missing_schur_feature_is_rejected() -> None:
    """Old cost tables without the new diagnostic cannot be misread."""
    rows = [_row("0.1", 0), _row("0.1", 1)]
    rows[0]["pre_switch_external_schur_coupling_mean"] = ""
    try:
        analyze_records(rows)
    except ValueError as error:
        assert "pre_switch_external_schur_coupling_mean" in str(error)
    else:
        raise AssertionError("missing Schur diagnostic should raise ValueError")
