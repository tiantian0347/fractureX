#!/usr/bin/env python3
"""Replay resolved Model-0 checkpoints and measure the online slow rate.

Purpose
-------
Connect the coarse-grid slow-mode mechanism to the length-scale-resolved
physical path. Each target load is replayed from the preceding converged
report checkpoint with the plain mechanics-then-phase staggered map. The last
five coupled increments provide the same tangent-weighted online contraction
estimate used by the solver gate.

Scope
-----
This script performs an increment diagnostic only. It does not form the full
finite-difference propagation matrix and does not alter the accepted physical
path. Physical-path reaction forces and stabilized continuation iteration
counts are copied from the audited report CSV and kept separate from replay
quantities.

Usage
-----
PYTHONPATH=. python scripts/paper_solver/scan_model0_resolved_online_rate.py \
    --max-sweeps 40 \
    --output-dir ../results/phasefield_solver/model0_resolved_online_rate_h0065
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from fealpy.backend import backend_manager as bm

from fracturex.analysis.staggered_slow_mode import online_increment_subspace
from scripts.paper_solver.run_model0_fine_reference import (
    build_model0_resolved_solver,
)
from scripts.paper_solver.verify_slow_mode_fracturex import FrozenStandardFEStepMap


SCRIPT_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT.parent / "results/phasefield_solver"
DEFAULT_PREFIX_DIR = RESULTS_ROOT / "model0_fine_curve_audit_prefix_h0065"
DEFAULT_POSTPEAK_DIR = RESULTS_ROOT / "model0_fine_curve_audit_unrelaxed_h0065"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "model0_resolved_online_rate_h0065"
DEFAULT_LOADS = (
    "0.0810,0.0832,0.0854,0.0876,0.0898,0.0920,"
    "0.0942,0.0964,0.0986,0.1008,0.1030"
)
CHECKPOINT_PATTERN = re.compile(r"report_(?P<index>\d+)_load_(?P<load>[0-9.]+)\.npz$")


@dataclass(frozen=True)
class CheckpointRecord:
    """Identify one accepted physical-path report checkpoint."""

    index: int
    load: float
    path: Path


def parse_loads(text: str) -> list[float]:
    """Parse a finite, strictly increasing comma-separated load sequence.

    Parameters
    ----------
    text : str
        Comma-separated target loads.

    Returns
    -------
    list[float]
        Strictly increasing target loads.

    Raises
    ------
    ValueError
        If the sequence is empty, non-finite, duplicated, or non-monotone.
    """
    loads = [float(token.strip()) for token in text.split(",") if token.strip()]
    values = np.asarray(loads, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("loads must contain finite values")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("loads must be strictly increasing")
    return values.tolist()


def require_results_directory(
    output_dir: Path, *, results_root: Path = RESULTS_ROOT
) -> Path:
    """Resolve an experiment directory and require it to lie under results.

    Parameters
    ----------
    output_dir : pathlib.Path
        Requested experiment output directory.
    results_root : pathlib.Path
        Root allowed to contain simulation outputs. Defaults to the repository
        ``results/phasefield_solver`` directory; injectable for unit tests.

    Returns
    -------
    pathlib.Path
        Absolute validated output directory.

    Raises
    ------
    ValueError
        If the requested directory lies outside ``results_root``.
    """
    resolved_output = output_dir.resolve()
    resolved_root = results_root.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"simulation outputs must be stored below {resolved_root}"
        ) from error
    return resolved_output


def discover_checkpoints(roots: Iterable[Path]) -> list[CheckpointRecord]:
    """Return unique report checkpoints ordered by report index.

    Parameters
    ----------
    roots : iterable[pathlib.Path]
        Fine-path result directories containing a ``checkpoints`` directory.

    Returns
    -------
    list[CheckpointRecord]
        Checkpoints sorted by report index.

    Raises
    ------
    ValueError
        If roots contain conflicting files for the same report index.
    """
    by_index: dict[int, CheckpointRecord] = {}
    for root in roots:
        for path in sorted((root / "checkpoints").glob("report_*_load_*.npz")):
            match = CHECKPOINT_PATTERN.match(path.name)
            if match is None:
                continue
            record = CheckpointRecord(
                index=int(match.group("index")),
                load=float(match.group("load")),
                path=path.resolve(),
            )
            prior = by_index.get(record.index)
            if prior is not None and (
                not np.isclose(prior.load, record.load, rtol=0.0, atol=1.0e-12)
                or prior.path != record.path
            ):
                raise ValueError(f"conflicting checkpoint for report {record.index}")
            by_index[record.index] = record
    return [by_index[index] for index in sorted(by_index)]


def read_path_rows(csv_path: Path) -> dict[float, dict[str, str]]:
    """Read physical-path reaction and iteration records keyed by load."""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("physical-path CSV is empty")
    return {round(float(row["load"]), 10): row for row in rows}


def _checkpoint_for_load(
    records: list[CheckpointRecord], load: float
) -> CheckpointRecord:
    """Return the unique checkpoint matching ``load`` to roundoff."""
    matches = [
        record
        for record in records
        if np.isclose(record.load, load, rtol=0.0, atol=5.0e-5)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one checkpoint for load {load:.6f}, found {len(matches)}")
    return matches[0]


def _load_checkpoint_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load finite displacement, damage, and history arrays from one restart."""
    with np.load(path) as data:
        displacement = np.asarray(data["uh"], dtype=np.float64).reshape(-1)
        damage = np.asarray(data["d"], dtype=np.float64).reshape(-1)
        history = np.asarray(data["H"], dtype=np.float64).copy()
    if not (
        np.isfinite(displacement).all()
        and np.isfinite(damage).all()
        and np.isfinite(history).all()
    ):
        raise ValueError(f"checkpoint contains non-finite data: {path}")
    return displacement, damage, history


def _tangent_weight(step_map: FrozenStandardFEStepMap) -> np.ndarray:
    """Return the positive diagonal of the captured block tangents."""
    if step_map.last_displacement_matrix is None or step_map.last_phase_matrix is None:
        raise RuntimeError("staggered replay did not capture both tangent blocks")
    weight = np.concatenate(
        (
            np.asarray(step_map.last_displacement_matrix.diagonal(), dtype=np.float64),
            np.asarray(step_map.last_phase_matrix.diagonal(), dtype=np.float64),
        )
    )
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise RuntimeError("captured tangent diagonal must be finite and positive")
    return weight


def _relative_difference(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return ``||candidate-reference||_2/max(1,||reference||_2)``."""
    if candidate.shape != reference.shape:
        raise ValueError("state vectors must have identical shapes")
    return float(
        np.linalg.norm(candidate - reference) / max(1.0, np.linalg.norm(reference))
    )


def replay_transition(
    main: Any,
    *,
    previous: CheckpointRecord,
    target: CheckpointRecord,
    max_sweeps: int,
    atol: float,
    rtol: float,
    window_size: int,
    relative_singular_value: float,
    max_dimension: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay one report transition with the plain staggered map.

    Parameters
    ----------
    main : MainSolve
        Solver built by ``build_model0_resolved_solver``.
    previous, target : CheckpointRecord
        Consecutive accepted physical-path checkpoints.
    max_sweeps : int
        Positive cap on ordinary staggered sweeps.
    atol, rtol : float
        Damage-increment stopping tolerances. Convergence requires
        ``||dd|| <= atol + rtol*max(1, ||d_new||)``.
    window_size : int
        Number of trailing coupled increments used for the online estimate.
    relative_singular_value : float
        Relative singular-value cutoff for the online subspace.
    max_dimension : int or None
        Optional retained online-subspace dimension cap.

    Returns
    -------
    tuple[dict, list[dict]]
        One load-level summary and the per-sweep trace.

    Notes
    -----
    The committed history and damage lower bound come from ``previous``. The
    replay uses the direct displacement and phase solves followed by nodal box
    projection, matching the unaccelerated production block map before path
    stabilization is applied.
    """
    if target.index != previous.index + 1:
        raise ValueError("replay requires consecutive report checkpoints")
    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be positive")

    previous_u, previous_d, previous_h = _load_checkpoint_arrays(previous.path)
    target_u, target_d, _ = _load_checkpoint_arrays(target.path)
    main.load_restart_npz(str(previous.path))
    if np.asarray(bm.to_numpy(main.d[:])).size != previous_d.size:
        raise ValueError("checkpoint phase size does not match the regenerated mesh")

    step_map = FrozenStandardFEStepMap(
        main,
        load=target.load,
        committed_damage=previous_d,
        committed_history=previous_h,
        phase_bound_solver="clip",
    )
    current_damage = previous_d.copy()
    coupled_states = [np.concatenate((previous_u, previous_d))]
    trace_rows: list[dict[str, Any]] = []
    converged = False
    start_time = perf_counter()

    for sweep in range(1, max_sweeps + 1):
        next_damage = step_map(current_damage)
        next_state = step_map.current_full_state()
        coupled_increment = next_state - coupled_states[-1]
        damage_increment_norm = float(np.linalg.norm(next_damage - current_damage))
        coupled_states.append(next_state)
        current_damage = next_damage
        threshold = atol + rtol * max(1.0, float(np.linalg.norm(current_damage)))
        trace_rows.append(
            {
                "sweep": sweep,
                "damage_increment_norm": damage_increment_norm,
                "coupled_increment_norm_euclidean": float(
                    np.linalg.norm(coupled_increment)
                ),
                "damage_stop_threshold": threshold,
                "converged": damage_increment_norm <= threshold,
            }
        )
        if damage_increment_norm <= threshold:
            converged = True
            break

    elapsed = perf_counter() - start_time
    increments = np.diff(np.vstack(coupled_states), axis=0)
    weight = _tangent_weight(step_map)
    weighted_norms = np.sqrt(np.sum(weight[None, :] * increments**2, axis=1))
    weighted_ratios = np.divide(
        weighted_norms[1:],
        weighted_norms[:-1],
        out=np.full(max(weighted_norms.size - 1, 0), np.nan, dtype=np.float64),
        where=weighted_norms[:-1] > np.finfo(np.float64).tiny,
    )
    for index, row in enumerate(trace_rows):
        row["coupled_increment_norm_weighted"] = float(weighted_norms[index])
        row["weighted_increment_ratio"] = (
            float("nan") if index == 0 else float(weighted_ratios[index - 1])
        )

    estimate = online_increment_subspace(
        increments,
        weight,
        window_size=window_size,
        relative_singular_value=relative_singular_value,
        max_dimension=max_dimension,
    )
    replay_state = coupled_states[-1]
    path_state = np.concatenate((target_u, target_d))
    summary = {
        "previous_load": previous.load,
        "load": target.load,
        "plain_replay_converged": converged,
        "plain_replay_sweeps": len(trace_rows),
        "plain_replay_seconds": elapsed,
        "plain_last_damage_increment_norm": trace_rows[-1]["damage_increment_norm"],
        "plain_last_weighted_increment_norm": float(weighted_norms[-1]),
        "rhohat_online": estimate.contraction_estimate,
        "online_window_size": estimate.window_size,
        "online_dimension": estimate.dimension,
        "online_singular_values": estimate.singular_values.tolist(),
        "relative_state_difference_to_physical_path": _relative_difference(
            replay_state, path_state
        ),
        "target_max_damage": float(np.max(target_d)),
    }
    return summary, trace_rows


def _git_commit() -> str | None:
    """Return the current repository commit when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_scan_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat load-level scan records with stable column ordering."""
    fieldnames = (
        "previous_load",
        "load",
        "reaction_force_abs",
        "physical_path_iterations",
        "physical_path_algorithm",
        "plain_replay_converged",
        "plain_replay_sweeps",
        "plain_replay_seconds",
        "plain_last_damage_increment_norm",
        "plain_last_weighted_increment_norm",
        "rhohat_online",
        "online_window_size",
        "online_dimension",
        "relative_state_difference_to_physical_path",
        "target_max_damage",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def run_scan(args: argparse.Namespace) -> Path:
    """Execute all configured checkpoint replays and persist diagnostics."""
    if args.max_sweeps <= 0 or args.window_size <= 1:
        raise ValueError("max-sweeps must be positive and window-size must exceed one")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("atol and rtol must be nonnegative")
    if not 0.0 < args.relative_singular_value <= 1.0:
        raise ValueError("relative-singular-value must lie in (0, 1]")
    max_dimension = None if args.max_dimension == 0 else args.max_dimension
    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("max-dimension must be nonnegative")

    output_dir = require_results_directory(args.output_dir)
    roots = [args.prefix_dir.resolve(), args.postpeak_dir.resolve()]
    checkpoints = discover_checkpoints(roots)
    loads = parse_loads(args.loads)
    path_rows = read_path_rows(args.path_csv.resolve())
    main, material, mesh_stats, unused_nodes = build_model0_resolved_solver(
        hmin=args.hmin
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for load in loads:
        target = _checkpoint_for_load(checkpoints, load)
        previous_matches = [
            record for record in checkpoints if record.index == target.index - 1
        ]
        if len(previous_matches) != 1:
            raise ValueError(f"missing preceding checkpoint for load {load:.6f}")
        summary, trace_rows = replay_transition(
            main,
            previous=previous_matches[0],
            target=target,
            max_sweeps=args.max_sweeps,
            atol=args.atol,
            rtol=args.rtol,
            window_size=args.window_size,
            relative_singular_value=args.relative_singular_value,
            max_dimension=max_dimension,
        )
        path_row = path_rows[round(target.load, 10)]
        summary.update(
            {
                "reaction_force_abs": float(path_row["residual_force_abs"]),
                "physical_path_iterations": int(path_row["staggered_iterations"]),
                "physical_path_algorithm": path_row["algorithm_stage"],
            }
        )
        summaries.append(summary)
        trace_path = output_dir / f"trace_load_{target.load:.4f}.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(trace_rows[0]))
            writer.writeheader()
            writer.writerows(trace_rows)
        write_scan_csv(output_dir / "online_rate_scan.csv", summaries)
        print(
            f"load={target.load:.4f} rhohat={summary['rhohat_online']:.6f} "
            f"plain={summary['plain_replay_sweeps']} "
            f"converged={summary['plain_replay_converged']} "
            f"path_iterations={summary['physical_path_iterations']}",
            flush=True,
        )

    metadata = {
        "script": str(Path(__file__).resolve()),
        "command": shlex.join(sys.argv),
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "fealpy")
        },
        "case": "Model-0 circular-hole plate",
        "source_case": "fracturex/cases/phase_field/model0_example.py",
        "material": material,
        "hmin_target": args.hmin,
        "mesh": mesh_stats,
        "unused_distmesh_nodes_removed": unused_nodes,
        "checkpoint_roots": [str(root) for root in roots],
        "physical_path_csv": str(args.path_csv.resolve()),
        "loads": loads,
        "plain_replay": {
            "map": "mechanics then phase, direct block solves, nodal box projection",
            "max_sweeps": args.max_sweeps,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "online_estimator": {
            "weight": "diagonal of final captured displacement and phase tangents",
            "window_size": args.window_size,
            "relative_singular_value": args.relative_singular_value,
            "max_dimension": max_dimension,
        },
        "random_seed": None,
        "output_dir": str(output_dir),
        "output_csv": str((output_dir / "online_rate_scan.csv").resolve()),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir / "online_rate_scan.csv"


def parse_args() -> argparse.Namespace:
    """Parse reproducible fine-grid online-rate scan parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmin", type=float, default=0.0065)
    parser.add_argument("--loads", default=DEFAULT_LOADS)
    parser.add_argument("--max-sweeps", type=int, default=40)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--relative-singular-value", type=float, default=1.0e-2)
    parser.add_argument("--max-dimension", type=int, default=4)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--postpeak-dir", type=Path, default=DEFAULT_POSTPEAK_DIR)
    parser.add_argument(
        "--path-csv",
        type=Path,
        default=DEFAULT_POSTPEAK_DIR / "residual_force_vs_displacement.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the configured scan and print its primary output path."""
    path = run_scan(parse_args())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
