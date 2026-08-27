#!/usr/bin/env python3
"""Summarize an online slow-rate switch gate from completed runs.

The gate uses only the weighted contraction estimate produced from recent
staggered increments. It is evaluated before a Reduced-NE solve; accepted
candidate counts are reported afterward only for validation. This script
reads existing JSON summaries and performs no finite-element solve.

Usage
-----
python scripts/paper_solver/summarize_online_switch_gate.py \
    --record model0=/path/to/model0-summary.json \
    --record model5=/path/to/model5-summary.json \
    --output /path/to/online_switch_gate.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: Path) -> dict[str, Any]:
    """Load a verification summary and require an object root."""
    if not path.is_file():
        raise FileNotFoundError(f"online gate summary does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"online gate summary root must be an object: {path}")
    return payload


def _parse_record(text: str) -> tuple[str, Path]:
    """Parse a CLI record in ``label=/path/to/summary.json`` form."""
    label, separator, path_text = text.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError(
            "records must use the form label=/path/to/summary.json"
        )
    return label.strip(), Path(path_text).expanduser()


def _iter_checkpoints(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield all checkpoint objects, including a direct summary if supplied."""
    checkpoints = payload.get("checkpoints")
    if checkpoints is None:
        if "case" not in payload:
            raise ValueError("summary must contain checkpoints or case")
        yield payload
        return
    if not isinstance(checkpoints, list):
        raise ValueError("summary checkpoints must be a list")
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("each checkpoint must be an object")
        yield checkpoint


def summarize_checkpoint(
    label: str,
    checkpoint: dict[str, Any],
    *,
    rho_low: float,
    rho_high: float,
    path_consistent: bool = False,
) -> dict[str, Any]:
    """Summarize one pre-switch estimate and its post-run validation.

    The recommendation is determined solely by ``weighted_contraction_estimate``.
    A missing online estimate is an error instead of a fallback to the offline
    spectral radius, which would invalidate the online-gate test.
    """
    case = checkpoint["case"]
    online = checkpoint.get("online_increment_slow_subspace") or {}
    region = online.get("solver_region") or {}
    if "weighted_contraction_estimate" not in region:
        raise ValueError(
            f"{label} load {case['load']}: online contraction estimate is missing"
        )
    q_hat = float(region["weighted_contraction_estimate"])
    if q_hat < rho_low:
        recommendation = "continue_staggered"
    elif q_hat >= rho_high:
        recommendation = "switch_reduced_ne"
    else:
        recommendation = "defer_decision"

    solver = checkpoint.get("reduced_nonlinear_solver") or {}
    pareto = checkpoint.get("benefit_cost_pareto") or {}
    records = pareto.get("records") or []
    accepted = [
        record
        for record in records
        if bool(record.get("all_acceptance_checks_passed", False))
    ]
    baseline_work = solver.get("baseline_residual_equivalent_evaluations")
    baseline_time = solver.get("baseline_staggered_wall_time_seconds")
    best_work = min(
        (int((solver.get("patches") or {}).get(record["patch"], {}).get(
            "total_residual_equivalent_evaluations", 10**18
        )) for record in accepted),
        default=None,
    )
    best_time = min(
        (float((solver.get("patches") or {}).get(record["patch"], {}).get(
            "total_wall_time_including_warmup_seconds", float("inf")
        )) for record in accepted),
        default=None,
    )
    return {
        "label": label,
        "case": str(case["name"]),
        "mesh_size": float(case["mesh_size"]),
        "load": float(case["load"]),
        "path_consistent": bool(path_consistent),
        "spectral_radius_reference": float(
            (checkpoint.get("slow_mode") or {})["spectral_radius"]
        ),
        "online_contraction_estimate": q_hat,
        "online_rate_relative_error": abs(
            q_hat - float(checkpoint["slow_mode"]["spectral_radius"])
        ) / max(float(checkpoint["slow_mode"]["spectral_radius"]), 1.0e-30),
        "online_dimension": int(region.get("selected_dimension", 0)),
        "online_selected_cell_fraction": float(
            region.get("selected_cell_fraction", float("nan"))
        ),
        "online_trace_fraction": float(
            region.get("reference_trace_fraction", float("nan"))
        ),
        "baseline_iterations": int(
            solver.get("baseline_staggered_iterations", checkpoint["fixed_point"]["iterations"])
        ),
        "baseline_work": None if baseline_work is None else int(baseline_work),
        "baseline_wall_time_seconds": (
            None if baseline_time is None else float(baseline_time)
        ),
        "recommendation": recommendation,
        "pareto_candidate_count": len(records),
        "accepted_candidate_count": len(accepted),
        "best_accepted_work": best_work,
        "best_accepted_wall_time_seconds": best_time,
        "gate_validation": (
            None
            if not records
            else (
                len(accepted) > 0
                if recommendation == "switch_reduced_ne"
                else len(accepted) == 0
            )
        ),
    }


def extract_gate_records(
    label: str,
    payload: dict[str, Any],
    *,
    rho_low: float = 0.87,
    rho_high: float = 0.89,
) -> list[dict[str, Any]]:
    """Extract gate records from every checkpoint in ``payload``."""
    if not 0.0 < rho_low < rho_high < 1.0:
        raise ValueError("require 0 < rho_low < rho_high < 1")
    checkpoints = list(_iter_checkpoints(payload))
    scan = payload.get("scan") or {}
    scan_loads = [float(value) for value in scan.get("loads", [])]
    path_consistent = len(checkpoints) > 1 and (
        scan.get("continuation") == "monotone committed damage and history"
    )
    records = []
    for checkpoint in checkpoints:
        load = float(checkpoint["case"]["load"])
        records.append(
            summarize_checkpoint(
                label,
                checkpoint,
                rho_low=rho_low,
                rho_high=rho_high,
                path_consistent=path_consistent
                and any(abs(value - load) <= 1.0e-8 for value in scan_loads),
            )
        )
    if not records:
        raise ValueError(f"no checkpoints found for {label!r}")
    return records


def _write_outputs(output: Path, records: list[dict[str, Any]]) -> None:
    """Write CSV and JSON sidecar outputs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump({"records": records}, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def main() -> None:
    """Parse CLI arguments and write online switch-gate summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="append", type=_parse_record, required=True)
    parser.add_argument("--rho-low", type=float, default=0.87)
    parser.add_argument("--rho-high", type=float, default=0.89)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    for label, path in args.record:
        records.extend(
            extract_gate_records(
                label,
                _load_json(path.resolve()),
                rho_low=args.rho_low,
                rho_high=args.rho_high,
            )
        )
    records.sort(key=lambda row: (row["label"], row["load"]))
    _write_outputs(args.output, records)


if __name__ == "__main__":
    main()
