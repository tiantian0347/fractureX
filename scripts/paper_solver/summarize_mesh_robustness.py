#!/usr/bin/env python3
"""Summarize mesh-robustness diagnostics from existing verification JSON files.

The script extracts solver-aware quantities from completed FractureX runs. It
does not run finite-element solves and does not infer missing convergence or
path information. Each input is labeled explicitly so incomparable states can
remain visible without entering the mesh-robustness table unnoticed.

Usage
-----
python scripts/paper_solver/summarize_mesh_robustness.py \
    --record h005=/path/to/summary.json \
    --record h0035=/path/to/summary.json \
    --record h0025=/path/to/summary.json \
    --load 0.1125 \
    --output /path/to/mesh_robustness.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    """Load one finite verification summary.

    Parameters
    ----------
    path : Path
        JSON result written by ``verify_slow_mode_fracturex.py``.

    Returns
    -------
    dict[str, Any]
        Parsed summary. The returned object is newly allocated by ``json``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the root JSON value is not an object.
    """
    if not path.is_file():
        raise FileNotFoundError(f"mesh summary does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"mesh summary root must be an object: {path}")
    return payload


def _select_checkpoint(
    payload: dict[str, Any],
    target_load: float,
) -> tuple[dict[str, Any], bool]:
    """Select the nearest checkpoint and report whether it lies on a scan path.

    The target must match a stored load to numerical roundoff. A single
    checkpoint summary is accepted as a direct-load diagnostic but is marked
    ``path_consistent=False`` because its irreversible loading history is not
    encoded by the summary.
    """
    checkpoints = payload.get("checkpoints")
    if checkpoints:
        selected = min(
            checkpoints,
            key=lambda checkpoint: abs(
                float(checkpoint["case"]["load"]) - target_load
            ),
        )
        selected_load = float(selected["case"]["load"])
        if abs(selected_load - target_load) > 1.0e-8:
            raise ValueError(
                f"target load {target_load:g} is absent from scan; nearest is "
                f"{selected_load:g}"
            )
        scan = payload.get("scan") or {}
        path_consistent = (
            len(checkpoints) > 1
            and target_load in [float(value) for value in scan.get("loads", [])]
            and scan.get("continuation") == "monotone committed damage and history"
        )
        return selected, bool(path_consistent)

    selected_load = float(payload["case"]["load"])
    if abs(selected_load - target_load) > 1.0e-8:
        raise ValueError(
            f"target load {target_load:g} does not match direct summary load "
            f"{selected_load:g}"
        )
    return payload, False


def _get_reduced_patch(
    checkpoint: dict[str, Any],
    region: str,
    theta: float,
) -> dict[str, Any] | None:
    """Return one reduced-solver region record, preferring the requested theta."""
    solver = checkpoint.get("reduced_nonlinear_solver") or {}
    patches = solver.get("patches") or {}
    direct = patches.get(region)
    if direct is not None:
        return direct
    threshold_suffix = f"theta_{round(theta * 10):02d}"
    indexed_suffix = f"theta_{round(theta * 10) - 5:02d}"
    return patches.get(f"{region}_{threshold_suffix}") or patches.get(
        f"{region}_{indexed_suffix}"
    )


def summarize_checkpoint(
    label: str,
    checkpoint: dict[str, Any],
    *,
    target_load: float,
    path_consistent: bool,
    theta: float,
) -> dict[str, Any]:
    """Extract one mesh-robustness record from a verification checkpoint.

    Parameters
    ----------
    label : str
        User-supplied mesh label, such as ``h005``.
    checkpoint : dict[str, Any]
        One checkpoint object from the verification JSON.
    target_load : float
        Requested dimensionless load; must match the checkpoint load.
    path_consistent : bool
        Whether the checkpoint came from a committed monotone scan.
    theta : float
        Trace threshold used for region comparison.

    Returns
    -------
    dict[str, Any]
        Flat, CSV-compatible diagnostics. Missing reduced-solver quantities are
        represented by ``None`` rather than guessed values.
    """
    case = checkpoint["case"]
    slow = checkpoint["slow_mode"]
    subspace = checkpoint["coupled_slow_subspace"]
    localization = checkpoint["localization"]
    spd = checkpoint["spd_patch_calibration"]
    reduced_candidates = {
        region: _get_reduced_patch(checkpoint, region, theta)
        for region in ("slow", "damage", "gradient", "online")
    }
    valid_candidates = {
        region: record
        for region, record in reduced_candidates.items()
        if record is not None and bool(record.get("converged"))
    }
    best_region = None
    best = None
    if valid_candidates:
        best_region, best = min(
            valid_candidates.items(),
            key=lambda item: int(item[1]["total_residual_equivalent_evaluations"]),
        )
    slow_record = reduced_candidates["slow"]
    return {
        "label": label,
        "mesh_size": float(case["mesh_size"]),
        "cells": int(case["cells"]),
        "scalar_dofs": int(case["scalar_dofs"]),
        "state_size": int(subspace["state_size"]),
        "load": float(case["load"]),
        "target_load": float(target_load),
        "path_consistent": bool(path_consistent),
        "baseline_converged": bool(
            checkpoint.get("checks", {}).get("fixed_point_converged", False)
        ),
        "baseline_iterations": int(checkpoint["fixed_point"]["iterations"]),
        "spectral_radius": float(slow["spectral_radius"]),
        "slow_dimension": int(subspace["selected_dimension"]),
        "slow_cell_fraction": float(localization["coupled_selected_cell_fraction"]),
        "slow_patch_dofs": int(spd["slow_patch_dofs"]),
        "slow_chi": float(spd["slow_patch_survival_factor"]),
        "damage_chi": float(spd["damage_patch_survival_factor"]),
        "gradient_chi": float(spd["gradient_patch_survival_factor"]),
        "slow_solver_patch_fraction": (
            None
            if slow_record is None
            else float(slow_record["local_patch_fraction_of_all_free_state_dofs"])
        ),
        "slow_solver_converged": (
            None if slow_record is None else bool(slow_record["converged"])
        ),
        "best_reduced_region": best_region,
        "best_reduced_converged": best is not None,
        "best_reduced_work": (
            None
            if best is None
            else int(best["total_residual_equivalent_evaluations"])
        ),
        "best_reduced_work_ratio": (
            None
            if best is None
            else float(best["residual_equivalent_work_reduction_fraction"])
        ),
        "best_reduced_speedup": (
            None
            if best is None
            else float(best["wall_time_speedup_over_staggered"])
        ),
        "best_reduced_solution_difference": (
            None
            if best is None
            else float(best["full_solution_l2_difference_from_staggered"])
        ),
    }


def _parse_record(text: str) -> tuple[str, Path]:
    """Parse one ``label=summary.json`` CLI value."""
    label, separator, path_text = text.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError(
            "records must use the form label=/path/to/summary.json"
        )
    return label.strip(), Path(path_text).expanduser()


def main() -> None:
    """Build and write the mesh-robustness CSV and JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="append", type=_parse_record, required=True)
    parser.add_argument("--load", type=float, required=True)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.theta < 1.0:
        parser.error("--theta must lie strictly between zero and one")

    records = []
    for label, path in args.record:
        payload = _load_json(path.resolve())
        checkpoint, path_consistent = _select_checkpoint(payload, args.load)
        records.append(
            summarize_checkpoint(
                label,
                checkpoint,
                target_load=args.load,
                path_consistent=path_consistent,
                theta=args.theta,
            )
        )
    records.sort(key=lambda record: record["mesh_size"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "target_load": float(args.load),
        "theta": float(args.theta),
        "records": records,
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
