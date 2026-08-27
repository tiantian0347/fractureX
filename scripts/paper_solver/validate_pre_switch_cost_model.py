#!/usr/bin/env python3
"""Validate a pre-switch surrogate for Reduced-NE candidate cost.

The input is the flat CSV written by ``summarize_cost_model.py``.  Candidate
features are restricted to quantities available before a Reduced-NE solve:
the completed staggered baseline, the local patch geometry, the local coupled
Jacobian condition number, and the offline/online contraction indicators.
Post-solve Krylov counts, residual evaluations, and wall times are used only
as targets for validation and are never used as predictors.

The validator uses leave-one-state-out ridge regression on
``log(total_work / baseline_work)``.  This normalization keeps the target
comparable across meshes and loads while retaining the within-state ranking
needed for candidate selection.  The output is diagnostic: a low selection
accuracy or a nonzero regret is evidence that more topology/load data or a
richer model is required before a cost predictor is promoted into the solver.

Usage
-----
python scripts/paper_solver/validate_pre_switch_cost_model.py \
    --input /path/to/cost_features.csv \
    --output /path/to/pre_switch_cost_validation.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PRE_SWITCH_NUMERIC_FEATURES = (
    "baseline_work",
    "coupled_patch_dofs",
    "coupled_condition_number",
    "survival_factor",
    "online_contraction_estimate",
    "online_selected_cell_fraction",
    "online_trace_fraction",
    "trace_fraction",
    "theta",
)
PRE_SWITCH_CATEGORICAL_FEATURES = ("region",)
TARGET_COLUMN = "total_work"


def _float(row: dict[str, str], key: str) -> float:
    """Parse one finite numeric feature from a CSV row."""
    value = row.get(key, "")
    if key == "coupled_condition_number" and value in (None, "", "None", "nan", "NaN"):
        value = row.get("pre_switch_coupled_condition_number", "")
    if value in (None, "", "None", "nan", "NaN"):
        raise ValueError(f"missing pre-switch feature {key!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite pre-switch feature {key!r}: {value!r}")
    return number


def _state_key(row: dict[str, str]) -> str:
    """Return a stable state identifier independent of candidate patch."""
    return f"{row.get('label', '')}|load={float(row['load']):.12g}"


def _feature_names(rows: Sequence[dict[str, str]]) -> list[str]:
    """Build deterministic numeric and one-hot feature names."""
    regions = sorted({str(row.get("region", "")) for row in rows})
    return [*PRE_SWITCH_NUMERIC_FEATURES, *(f"region={region}" for region in regions)]


def _feature_matrix(
    rows: Sequence[dict[str, str]], names: Sequence[str]
) -> np.ndarray:
    """Construct the raw pre-switch feature matrix for ``rows``."""
    regions = {
        name.split("=", 1)[1]
        for name in names
        if name.startswith("region=")
    }
    matrix = np.empty((len(rows), len(names)), dtype=float)
    for index, row in enumerate(rows):
        values = {name: _float(row, name) for name in PRE_SWITCH_NUMERIC_FEATURES}
        values.update(
            {
                f"region={region}": float(row.get("region", "") == region)
                for region in regions
            }
        )
        matrix[index] = [values[name] for name in names]
    return matrix


def _standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize test data with statistics fitted on the training data."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fit a standardized ridge model with an unpenalized intercept."""
    train_x, test_x, _, _ = _standardize(train_x, test_x)
    design = np.column_stack([np.ones(len(train_x)), train_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + float(ridge) * penalty,
        design.T @ train_y,
    )
    return test_design @ coefficients


def validate_records(
    rows: Iterable[dict[str, str]],
    *,
    ridge: float = 1.0e-2,
    gate_threshold: float = 0.89,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run leave-one-state-out validation and return predictions and summary.

    Parameters
    ----------
    rows : iterable of dict[str, str]
        Rows from ``summarize_cost_model.py --include-rejected``.
    ridge : float, default=1e-2
        Nonnegative ridge penalty.  The value is fixed before the validation
        split and is therefore not tuned on the held-out state.
    gate_threshold : float, default=0.89
        Online contraction estimate above which the second-layer cost model is
        operationally considered.  This does not change the fitted model; it
        only reports metrics on states that pass the first-layer gate.

    Returns
    -------
    predictions, summary : tuple[list[dict[str, Any]], dict[str, Any]]
        Candidate-level predictions and state-level validation statistics.

    Raises
    ------
    ValueError
        If rows are empty, a state has fewer than two candidates, a required
        feature is missing, or the ridge penalty is negative.
    """
    if ridge < 0.0:
        raise ValueError("ridge penalty must be nonnegative")
    if not 0.0 < gate_threshold < 1.0:
        raise ValueError("gate_threshold must lie in (0, 1)")
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("at least one cost-model row is required")
    names = _feature_names(materialized)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in materialized:
        grouped.setdefault(_state_key(row), []).append(row)
    invalid = [key for key, group in grouped.items() if len(group) < 2]
    if invalid:
        raise ValueError(f"each state needs at least two candidates: {invalid}")
    if len(grouped) < 2:
        raise ValueError("leave-one-state-out validation needs at least two states")

    all_x = _feature_matrix(materialized, names)
    all_y = np.array(
        [
            math.log(max(_float(row, TARGET_COLUMN), 1.0e-12) / max(
                _float(row, "baseline_work"), 1.0e-12
            ))
            for row in materialized
        ],
        dtype=float,
    )
    positions = {id(row): index for index, row in enumerate(materialized)}
    predictions: list[dict[str, Any]] = []
    state_summaries: list[dict[str, Any]] = []
    for state, test_rows in grouped.items():
        test_ids = {id(row) for row in test_rows}
        test_indices = [positions[id(row)] for row in test_rows]
        train_indices = [index for index, row in enumerate(materialized) if id(row) not in test_ids]
        predicted_log_ratio = _ridge_predict(
            all_x[train_indices],
            all_y[train_indices],
            all_x[test_indices],
            ridge,
        )
        baseline_work = _float(test_rows[0], "baseline_work")
        predicted_work = baseline_work * np.exp(predicted_log_ratio)
        observed_work = np.array([_float(row, TARGET_COLUMN) for row in test_rows])
        predicted_index = int(np.argmin(predicted_work))
        observed_min = float(np.min(observed_work))
        predicted_actual = float(observed_work[predicted_index])
        relative_regret = (predicted_actual - observed_min) / max(observed_min, 1.0e-12)
        predicted_order = np.argsort(predicted_work)
        top_two = predicted_order[: min(2, len(predicted_order))]
        observed_profitable = observed_work < baseline_work
        predicted_profitable = predicted_work < baseline_work
        profitable_true_positive = int(
            np.count_nonzero(predicted_profitable & observed_profitable)
        )
        observed_profitable_count = int(np.count_nonzero(observed_profitable))
        predicted_profitable_count = int(np.count_nonzero(predicted_profitable))
        top_two_contains_optimum = bool(
            np.any(observed_work[top_two] <= observed_min + 1.0e-12)
        )
        near_optimal = bool(relative_regret <= 0.05)
        state_summaries.append(
            {
                "state": state,
                "candidate_count": len(test_rows),
                "online_contraction_estimate": _float(
                    test_rows[0], "online_contraction_estimate"
                ),
                "predicted_best_patch": str(test_rows[predicted_index]["patch"]),
                "observed_best_patch": str(test_rows[int(np.argmin(observed_work))]["patch"]),
                "predicted_best_work": predicted_actual,
                "observed_best_work": observed_min,
                "relative_regret": relative_regret,
                "selection_optimal": bool(relative_regret <= 1.0e-12),
                "near_optimal_within_5_percent": near_optimal,
                "top_two_contains_optimum": top_two_contains_optimum,
                "observed_profitable_count": observed_profitable_count,
                "predicted_profitable_count": predicted_profitable_count,
                "profitable_candidate_recall": (
                    None
                    if observed_profitable_count == 0
                    else profitable_true_positive / observed_profitable_count
                ),
                "profitable_candidate_precision": (
                    None
                    if predicted_profitable_count == 0
                    else profitable_true_positive / predicted_profitable_count
                ),
            }
        )
        for local_index, row in enumerate(test_rows):
            predictions.append(
                {
                    "state": state,
                    "label": row.get("label", ""),
                    "load": float(row["load"]),
                    "patch": row["patch"],
                    "region": row.get("region", ""),
                    "theta": float(row["theta"]),
                    "observed_total_work": float(observed_work[local_index]),
                    "predicted_total_work": float(predicted_work[local_index]),
                    "observed_rank": int(np.argsort(observed_work).tolist().index(local_index) + 1),
                    "predicted_rank": int(np.argsort(predicted_work).tolist().index(local_index) + 1),
                    "predicted_best_patch": str(test_rows[predicted_index]["patch"]),
                    "observed_profitable": bool(observed_profitable[local_index]),
                    "predicted_profitable": bool(predicted_profitable[local_index]),
                    "relative_regret_if_selected": relative_regret,
                }
            )

    regrets = [float(item["relative_regret"]) for item in state_summaries]
    accuracy = sum(bool(item["selection_optimal"]) for item in state_summaries) / len(state_summaries)
    top_two_accuracy = sum(
        bool(item["top_two_contains_optimum"]) for item in state_summaries
    ) / len(state_summaries)
    near_optimal_accuracy = sum(
        bool(item["near_optimal_within_5_percent"]) for item in state_summaries
    ) / len(state_summaries)
    observed_profit_states = [
        item for item in state_summaries if item["observed_profitable_count"] > 0
    ]
    predicted_profit_states = [
        item for item in state_summaries if item["predicted_profitable_count"] > 0
    ]
    state_profit_recall = (
        None
        if not observed_profit_states
        else sum(
            item["predicted_profitable_count"] > 0 for item in observed_profit_states
        )
        / len(observed_profit_states)
    )
    candidate_recalls = [
        float(item["profitable_candidate_recall"])
        for item in state_summaries
        if item["profitable_candidate_recall"] is not None
    ]
    operational_states = [
        item
        for item in state_summaries
        if item["online_contraction_estimate"] >= gate_threshold
    ]

    def _rate(states: Sequence[dict[str, Any]], key: str) -> float | None:
        """Average a Boolean state metric, returning None for no states."""
        if not states:
            return None
        return sum(bool(item[key]) for item in states) / len(states)

    operational_profit_states = [
        item for item in operational_states if item["observed_profitable_count"] > 0
    ]
    operational_false_positive_states = [
        item
        for item in operational_states
        if item["observed_profitable_count"] == 0
        and item["predicted_profitable_count"] > 0
    ]
    summary = {
        "model": "standardized_ridge_log_work_ratio",
        "status": "validated" if accuracy >= 1.0 and max(regrets, default=0.0) <= 1.0e-12 else "diagnostic_only",
        "uses_post_solve_features": False,
        "target": "log(total_work / baseline_work)",
        "ridge": float(ridge),
        "gate_threshold": float(gate_threshold),
        "feature_names": names,
        "pre_switch_condition_number_rows": sum(
            row.get("pre_switch_coupled_condition_number", "")
            not in (None, "", "None", "nan", "NaN")
            for row in materialized
        ),
        "condition_number_fallback_rows": sum(
            row.get("pre_switch_coupled_condition_number", "")
            in (None, "", "None", "nan", "NaN")
            and row.get("coupled_condition_number", "")
            not in (None, "", "None", "nan", "NaN")
            for row in materialized
        ),
        "state_count": len(state_summaries),
        "candidate_count": len(materialized),
        "selection_accuracy": accuracy,
        "top_two_hit_rate": top_two_accuracy,
        "near_optimal_5_percent_rate": near_optimal_accuracy,
        "profitable_state_recall": state_profit_recall,
        "mean_profitable_candidate_recall": (
            None if not candidate_recalls else float(np.mean(candidate_recalls))
        ),
        "operational_gate_state_count": len(operational_states),
        "operational_selection_accuracy": _rate(
            operational_states, "selection_optimal"
        ),
        "operational_top_two_hit_rate": _rate(
            operational_states, "top_two_contains_optimum"
        ),
        "operational_near_optimal_5_percent_rate": _rate(
            operational_states, "near_optimal_within_5_percent"
        ),
        "operational_profitable_state_recall": (
            None
            if not operational_profit_states
            else sum(
                item["predicted_profitable_count"] > 0
                for item in operational_profit_states
            )
            / len(operational_profit_states)
        ),
        "operational_false_positive_state_count": len(
            operational_false_positive_states
        ),
        "mean_relative_regret": float(np.mean(regrets)),
        "max_relative_regret": float(np.max(regrets)),
        "states": state_summaries,
    }
    return predictions, summary


def _write_outputs(output: Path, predictions: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write candidate predictions and a JSON validation summary."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(predictions[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    with output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def main() -> None:
    """Parse CLI arguments, validate the model, and write both outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    parser.add_argument("--gate-threshold", type=float, default=0.89)
    args = parser.parse_args()
    with args.input.resolve().open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    predictions, summary = validate_records(
        rows, ridge=args.ridge, gate_threshold=args.gate_threshold
    )
    _write_outputs(args.output, predictions, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
