"""Tests for pre-switch cost-model validation."""

from __future__ import annotations

from scripts.paper_solver.validate_pre_switch_cost_model import validate_records


def _row(state: str, patch: str, work: int, offset: float) -> dict[str, str]:
    """Build a compact candidate row with only required pre-switch fields."""
    label, load = state.split("|")
    return {
        "label": label,
        "load": load,
        "patch": patch,
        "region": "damage" if patch.startswith("damage") else "slow",
        "theta": "0.6" if patch.endswith("0") else "0.8",
        "baseline_work": "100",
        "coupled_patch_dofs": str(10 + int(offset)),
        "coupled_condition_number": str(100 + 10 * offset),
        "survival_factor": str(0.6 + 0.01 * offset),
        "online_contraction_estimate": "0.9",
        "online_selected_cell_fraction": "0.2",
        "online_trace_fraction": "0.6",
        "trace_fraction": "0.6",
        "total_work": str(work),
    }


def test_leave_one_state_out_reports_optimal_selection() -> None:
    """A separable synthetic data set should pass the diagnostic gate."""
    rows = [
        _row("a|0.1", "damage_0", 80, 0),
        _row("a|0.1", "slow_1", 120, 10),
        _row("b|0.2", "damage_0", 82, 0),
        _row("b|0.2", "slow_1", 122, 10),
        _row("c|0.3", "damage_0", 84, 0),
        _row("c|0.3", "slow_1", 124, 10),
    ]
    predictions, summary = validate_records(rows)
    assert len(predictions) == len(rows)
    assert summary["status"] == "validated"
    assert summary["selection_accuracy"] == 1.0
    assert summary["top_two_hit_rate"] == 1.0
    assert summary["profitable_state_recall"] == 1.0
    assert summary["operational_gate_state_count"] == 3
    assert summary["operational_false_positive_state_count"] == 0
    assert summary["uses_post_solve_features"] is False


def test_missing_online_feature_is_rejected() -> None:
    """A missing pre-switch indicator must not silently enter the model."""
    rows = [
        _row("a|0.1", "damage_0", 80, 0),
        _row("a|0.1", "slow_1", 120, 10),
        _row("b|0.2", "damage_0", 82, 0),
        _row("b|0.2", "slow_1", 122, 10),
    ]
    rows[0]["online_contraction_estimate"] = ""
    try:
        validate_records(rows)
    except ValueError as error:
        assert "online_contraction_estimate" in str(error)
    else:
        raise AssertionError("missing feature should raise ValueError")
