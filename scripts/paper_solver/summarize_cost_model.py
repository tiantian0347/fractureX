#!/usr/bin/env python3
"""Flatten solver-cost diagnostics from completed FractureX summaries.

The script joins the accepted threshold-sweep record (which contains the
coupled local Jacobian condition number) with the corresponding reduced
solver record (which contains Krylov work and timing). It only reads existing
JSON files and never runs a finite-element solve.

Usage
-----
python scripts/paper_solver/summarize_cost_model.py \
    --record h005=/path/to/summary.json \
    --record h0035=/path/to/summary.json \
    --output /path/to/cost_model.csv

The JSON sidecar is written next to the CSV. Use ``--load`` one or more times
to restrict the exported checkpoints; without it, every checkpoint containing
a threshold sweep is exported. By default only candidates that pass the
solver's benefit gate are retained; ``--include-rejected`` also keeps
converged candidates that preserve the solution but are correctly rejected
because their cost is not lower.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: Path) -> dict[str, Any]:
    """Load one verification summary and validate its root object.

    Parameters
    ----------
    path : Path
        JSON file produced by ``verify_slow_mode_fracturex.py``.

    Returns
    -------
    dict[str, Any]
        Parsed summary object.

    Raises
    ------
    FileNotFoundError
        If ``path`` is absent.
    ValueError
        If the JSON root is not an object.
    """
    if not path.is_file():
        raise FileNotFoundError(f"cost summary does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"cost summary root must be an object: {path}")
    return payload


def _parse_record(text: str) -> tuple[str, Path]:
    """Parse a CLI record of the form ``label=/path/to/summary.json``."""
    label, separator, path_text = text.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError(
            "records must use the form label=/path/to/summary.json"
        )
    return label.strip(), Path(path_text).expanduser()


def _iter_checkpoints(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield checkpoint objects, accepting a direct single-checkpoint summary."""
    checkpoints = payload.get("checkpoints")
    if checkpoints is not None:
        if not isinstance(checkpoints, list):
            raise ValueError("summary checkpoints must be a list")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                raise ValueError("each checkpoint must be an object")
            yield checkpoint
        return
    if "case" not in payload:
        raise ValueError("summary must contain checkpoints or a case object")
    yield payload


def _path_consistent(payload: dict[str, Any], checkpoint: dict[str, Any]) -> bool:
    """Return whether a checkpoint belongs to a recorded monotone scan path."""
    checkpoints = payload.get("checkpoints")
    scan = payload.get("scan") or {}
    if not isinstance(checkpoints, list) or len(checkpoints) < 2:
        return False
    loads = [float(value) for value in scan.get("loads", [])]
    load = float(checkpoint["case"]["load"])
    return (
        any(abs(value - load) <= 1.0e-8 for value in loads)
        and scan.get("continuation") == "monotone committed damage and history"
    )


def _number(record: dict[str, Any], key: str, default: Any = None) -> Any:
    """Return a numeric field while preserving missing values as ``None``."""
    value = record.get(key, default)
    return None if value is None else float(value)


def _join_candidate(
    label: str,
    payload: dict[str, Any],
    checkpoint: dict[str, Any],
    pareto_record: dict[str, Any],
) -> dict[str, Any]:
    """Join one Pareto record with its reduced-solver candidate.

    The two records are joined by the exact patch name. Missing joins are
    rejected because silently dropping Krylov or timing fields would produce
    an incomplete cost model.
    """
    patch = str(pareto_record["patch"])
    solver = checkpoint.get("reduced_nonlinear_solver") or {}
    patches = solver.get("patches") or {}
    candidate = patches.get(patch)
    if candidate is None:
        raise ValueError(f"missing reduced-solver candidate {patch!r}")

    case = checkpoint["case"]
    subspace = checkpoint.get("coupled_slow_subspace") or {}
    slow_mode = checkpoint.get("slow_mode") or {}
    baseline_work = solver.get("baseline_residual_equivalent_evaluations")
    baseline_time = solver.get("baseline_staggered_wall_time_seconds")
    online = checkpoint.get("online_increment_slow_subspace") or {}
    online_region = online.get("solver_region") or {}
    total_work = candidate.get(
        "total_residual_equivalent_evaluations",
        pareto_record.get("total_residual_equivalent_evaluations"),
    )
    total_time = candidate.get(
        "total_wall_time_including_warmup_seconds",
        pareto_record.get("total_wall_time_seconds"),
    )
    if total_work is None or total_time is None:
        raise ValueError(f"candidate {patch!r} has no total cost fields")

    record = {
        "label": label,
        "mesh_size": float(case["mesh_size"]),
        "cells": int(case["cells"]),
        "scalar_dofs": int(case["scalar_dofs"]),
        "state_size": int(subspace["state_size"])
        if "state_size" in subspace
        else None,
        "load": float(case["load"]),
        "path_consistent": _path_consistent(payload, checkpoint),
        "baseline_iterations": int(
            solver.get("baseline_staggered_iterations", checkpoint["fixed_point"]["iterations"])
        ),
        "baseline_work": None if baseline_work is None else int(baseline_work),
        "baseline_wall_time_seconds": _number(solver, "baseline_staggered_wall_time_seconds"),
        "spectral_radius": float(slow_mode["spectral_radius"]),
        "online_dimension": int(online_region["selected_dimension"])
        if "selected_dimension" in online_region
        else None,
        "online_selected_cell_fraction": _number(
            online_region, "selected_cell_fraction"
        ),
        "online_trace_fraction": _number(online_region, "reference_trace_fraction"),
        "online_contraction_estimate": _number(
            online_region, "weighted_contraction_estimate"
        ),
        "online_construction_seconds": _number(
            online_region, "construction_wall_time_seconds"
        ),
        "region": str(pareto_record["region"]),
        "theta": float(pareto_record["theta"]),
        "patch": patch,
        "coupled_patch_dofs": int(pareto_record["local_patch_dofs"]),
        "phase_patch_dofs": int(pareto_record["phase_patch_dofs"]),
        "coupled_patch_fraction": _number(
            candidate, "local_patch_fraction_of_all_free_state_dofs"
        ),
        "trace_fraction": float(pareto_record["trace_fraction"]),
        "coupled_condition_number": float(
            pareto_record["coupled_local_jacobian_condition_number"]
        ),
        "pre_switch_external_schur_coupling_mean": _number(
            candidate, "pre_switch_external_schur_coupling_mean"
        ),
        "pre_switch_external_schur_coupling_max": _number(
            candidate, "pre_switch_external_schur_coupling_max"
        ),
        "pre_switch_external_schur_coupling_samples": (
            int(candidate["pre_switch_external_schur_coupling_samples"])
            if candidate.get("pre_switch_external_schur_coupling_samples") is not None
            else None
        ),
        "pre_switch_external_schur_coupling_local_condition_number": _number(
            candidate, "pre_switch_external_schur_coupling_local_condition_number"
        ),
        "pre_switch_coupled_condition_number": _number(
            candidate, "pre_switch_external_schur_coupling_local_condition_number"
        ),
        "phase_condition_number": _number(
            pareto_record, "phase_local_jacobian_condition_number"
        ),
        "survival_factor": float(pareto_record["coupled_region_survival_factor"]),
        "local_jacobian_assemblies": int(candidate["local_jacobian_assemblies"]),
        "local_jacobian_setup_seconds": _number(
            candidate, "local_jacobian_assembly_wall_time_seconds"
        ),
        "local_linear_solves": int(candidate["local_linear_solves"]),
        "local_linear_solve_seconds": _number(
            candidate, "local_linear_solve_wall_time_seconds"
        ),
        "physical_residual_evaluations": int(candidate["physical_residual_evaluations"]),
        "jvp_evaluations": int(candidate.get("jvp_evaluations", 0)),
        "krylov_iterations": int(candidate["krylov_iterations"]),
        "preconditioner_applications": int(candidate.get("preconditioner_applications", 0)),
        "total_work": int(total_work),
        "total_wall_time_seconds": float(total_time),
        "work_ratio": (
            None
            if baseline_work is None
            else float(total_work) / float(baseline_work)
        ),
        "wall_time_ratio": (
            None
            if baseline_time is None
            else float(total_time) / float(baseline_time)
        ),
        "same_discrete_solution": bool(
            (candidate.get("acceptance_checks") or {}).get("same_discrete_solution", False)
        ),
        "candidate_converged": bool(candidate.get("converged", False)),
        "candidate_acceptance_checks_passed": bool(
            candidate.get("all_acceptance_checks_passed", False)
        ),
        "pareto_acceptance_checks_passed": bool(
            pareto_record.get("all_acceptance_checks_passed", False)
        ),
        "all_acceptance_checks_passed": bool(
            pareto_record.get("all_acceptance_checks_passed", False)
            and candidate.get("all_acceptance_checks_passed", False)
        ),
        "solution_difference_l2": _number(
            candidate, "full_solution_l2_difference_from_staggered"
        ),
        "projected_residual_norm": _number(candidate, "local_projected_residual_norm"),
    }
    return record


def extract_cost_records(
    label: str,
    payload: dict[str, Any],
    target_loads: Iterable[float] | None = None,
    *,
    include_rejected: bool = False,
) -> list[dict[str, Any]]:
    """Extract all accepted threshold-sweep candidates for selected loads.

    Parameters
    ----------
    label : str
        Mesh or experiment label stored in every output row.
    payload : dict[str, Any]
        Parsed verification summary.
    target_loads : iterable of float, optional
        Loads to retain. ``None`` retains every checkpoint with a Pareto sweep.
    include_rejected : bool, default=False
        Include candidates that converged and preserve the discrete solution but
        failed the lower-cost acceptance checks.

    Returns
    -------
    list[dict[str, Any]]
        Flat, CSV-compatible candidate records.

    Raises
    ------
    ValueError
        If an accepted Pareto record cannot be joined to a reduced-solver
        candidate, or if the requested load is absent.
    """
    requested = None if target_loads is None else [float(value) for value in target_loads]
    records: list[dict[str, Any]] = []
    seen_requested: set[float] = set()
    for checkpoint in _iter_checkpoints(payload):
        load = float(checkpoint["case"]["load"])
        if requested is not None and not any(abs(value - load) <= 1.0e-8 for value in requested):
            continue
        pareto = checkpoint.get("benefit_cost_pareto") or {}
        checkpoint_records = []
        for pareto_record in pareto.get("records", []):
            if (
                not include_rejected
                and not bool(pareto_record.get("all_acceptance_checks_passed", False))
            ):
                continue
            record = _join_candidate(label, payload, checkpoint, pareto_record)
            if not include_rejected and not record["all_acceptance_checks_passed"]:
                continue
            checkpoint_records.append(record)
        records.extend(checkpoint_records)
        if requested is not None and checkpoint_records:
            seen_requested.update(
                value for value in requested if abs(value - load) <= 1.0e-8
            )
    if requested is not None:
        missing = sorted(set(requested) - seen_requested)
        if missing:
            raise ValueError(f"requested loads are absent: {missing}")
    records.sort(key=lambda row: (row["mesh_size"], row["load"], row["region"], row["theta"]))
    if not records:
        raise ValueError(f"no accepted cost records found for {label!r}")
    return records


def _write_outputs(output: Path, records: list[dict[str, Any]]) -> None:
    """Write the flat CSV and a JSON sidecar containing the same records."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump({"records": records}, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def main() -> None:
    """Parse CLI arguments, extract records, and write both output formats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="append", type=_parse_record, required=True)
    parser.add_argument("--load", action="append", type=float, default=None)
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="also export converged candidates rejected by the lower-cost gate",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for label, path in args.record:
        records.extend(
            extract_cost_records(
                label,
                _load_json(path.resolve()),
                args.load,
                include_rejected=args.include_rejected,
            )
        )
    records.sort(key=lambda row: (row["mesh_size"], row["load"], row["region"], row["theta"]))
    _write_outputs(args.output, records)


if __name__ == "__main__":
    main()
