#!/usr/bin/env python3
"""Correlate the pre-switch Schur-coupling diagnostic with solver work.

This script is deliberately descriptive.  It does not fit a cost predictor
and it never uses a post-solve quantity as a feature.  The pre-switch
quantities are correlated with completed-solve work only to decide whether
the proposed Schur-coupling surrogate is worth adding to a later model.

Usage
-----
python scripts/paper_solver/analyze_pre_switch_schur_coupling.py \
    --input results/phasefield_solver/model5_schur_cost_features.csv \
    --output results/phasefield_solver/model5_schur_correlation.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURES = (
    "pre_switch_external_schur_coupling_mean",
    "pre_switch_external_schur_coupling_max",
    "pre_switch_external_schur_coupling_local_condition_number",
    "coupled_condition_number",
    "survival_factor",
    "coupled_patch_dofs",
)
TARGETS = ("total_work", "krylov_iterations", "physical_residual_evaluations")


def _state_key(row: dict[str, str]) -> str:
    """Return a stable label/load state identifier."""
    return f"{row.get('label', '')}|load={float(row['load']):.12g}"


def _finite_values(rows: Iterable[dict[str, str]], key: str) -> np.ndarray:
    """Parse one finite column and reject missing Schur diagnostics."""
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in (None, "", "None", "nan", "NaN"):
            raise ValueError(f"missing diagnostic column {key!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite diagnostic column {key!r}")
        values.append(number)
    return np.asarray(values, dtype=float)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return Pearson correlation, or None for a constant column."""
    if left.size < 2 or np.std(left) < 1.0e-14 or np.std(right) < 1.0e-14:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def analyze_records(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute state-wise feature/target correlations without fitting a model."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("at least one cost-feature row is required")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in materialized:
        grouped.setdefault(_state_key(row), []).append(row)
    records: list[dict[str, Any]] = []
    for state, state_rows in sorted(grouped.items()):
        for feature in FEATURES:
            feature_values = _finite_values(state_rows, feature)
            for target in TARGETS:
                target_values = _finite_values(state_rows, target)
                records.append(
                    {
                        "state": state,
                        "candidate_count": len(state_rows),
                        "feature": feature,
                        "target": target,
                        "pearson_correlation": _pearson(
                            feature_values, target_values
                        ),
                    }
                )
    summary = {
        "status": "diagnostic_only",
        "uses_post_solve_features": False,
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "state_count": len(grouped),
        "candidate_count": len(materialized),
        "records": records,
    }
    return records, summary


def _write_outputs(output: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write flat correlations and a JSON sidecar."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def main() -> None:
    """Parse CLI arguments, analyze correlations, and write outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.resolve().open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    records, summary = analyze_records(rows)
    _write_outputs(args.output, records, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
