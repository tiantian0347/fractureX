#!/usr/bin/env python3
"""Measure Model-0 online rates on the accepted internal continuation path.

Purpose
-------
Replay every adjacent pair of accepted length-scale-resolved Model-0 states
with the plain staggered map. The preceding state supplies the exact committed
history and damage lower bound used by the physical continuation path.

Scope
-----
This is a diagnostic replay. It does not alter accepted physical states and it
does not form a global finite-difference propagation matrix. Nonconverged
replays remain observable finite-window rate measurements and are marked in
the output rather than treated as spectral radii.

Usage
-----
PYTHONPATH=. python scripts/paper_solver/scan_model0_internal_path_online_rate.py \
    --manifest ../results/phasefield_solver/model0_resolved_internal_prefix_h0065/accepted_internal_states.csv \
    --manifest ../results/phasefield_solver/model0_resolved_internal_postpeak_h0065/accepted_internal_states.csv \
    --output-dir ../results/phasefield_solver/model0_resolved_internal_online_rate_h0065
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from scripts.paper_solver.run_model0_fine_reference import (
    build_model0_resolved_solver,
)
from scripts.paper_solver.scan_model0_resolved_online_rate import (
    CheckpointRecord,
    replay_transition,
    require_results_directory,
)


SCRIPT_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT.parent / "results/phasefield_solver"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "model0_resolved_internal_online_rate_h0065"


@dataclass(frozen=True)
class AcceptedStateRecord:
    """One accepted physical continuation state and its solver metadata."""

    load: float
    path: Path
    report_index: int
    is_report: bool
    iterations: int
    damage_relaxation: float
    anderson_depth: int
    algorithm_stage: str


def _parse_bool(text: str) -> bool:
    """Parse an explicit CSV Boolean without accepting ambiguous values."""
    normalized = text.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid Boolean value {text!r}")


def read_internal_manifests(paths: Iterable[Path]) -> list[AcceptedStateRecord]:
    """Read and stitch chronological accepted-state manifests.

    Parameters
    ----------
    paths : iterable[pathlib.Path]
        Ordered CSV manifests produced by ``run_model0_fine_reference.py``.
        Adjacent manifests may repeat their common restart state.

    Returns
    -------
    list[AcceptedStateRecord]
        Strictly increasing physical states. At a repeated boundary, the
        second manifest's restart file is retained because it is the exact
        source used by its following transition.

    Raises
    ------
    ValueError
        If a manifest is empty, locally unordered, or incompatible with its
        predecessor.
    FileNotFoundError
        If a referenced restart file is absent.
    """
    stitched: list[AcceptedStateRecord] = []
    manifest_count = 0
    for manifest_path in paths:
        manifest_count += 1
        manifest = manifest_path.resolve()
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"accepted-state manifest is empty: {manifest}")
        indices = np.asarray(
            [int(row["accepted_index"]) for row in rows], dtype=np.int64
        )
        loads = np.asarray([float(row["load"]) for row in rows], dtype=np.float64)
        if not np.array_equal(indices, np.arange(indices.size)):
            raise ValueError(f"manifest indices are not consecutive: {manifest}")
        if not np.isfinite(loads).all() or np.any(np.diff(loads) <= 0.0):
            raise ValueError(f"manifest loads are not strictly increasing: {manifest}")

        local_records: list[AcceptedStateRecord] = []
        for row in rows:
            checkpoint = Path(row["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = manifest.parent / checkpoint
            checkpoint = checkpoint.resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            local_records.append(
                AcceptedStateRecord(
                    load=float(row["load"]),
                    path=checkpoint,
                    report_index=int(row["report_index"]),
                    is_report=_parse_bool(row["is_report"]),
                    iterations=int(row["staggered_iterations"]),
                    damage_relaxation=float(row["damage_relaxation"]),
                    anderson_depth=int(row["anderson_depth"]),
                    algorithm_stage=row["algorithm_stage"],
                )
            )

        if stitched and np.isclose(
            stitched[-1].load,
            local_records[0].load,
            rtol=0.0,
            atol=1.0e-12,
        ):
            # Keep the first manifest's metadata for the shared accepted
            # state: it records how the physical path actually arrived there.
            # Use the second manifest only for transitions after the boundary.
            local_records = local_records[1:]
        if stitched and local_records and local_records[0].load <= stitched[-1].load:
            raise ValueError("accepted-state manifests overlap beyond one boundary")
        stitched.extend(local_records)

    if manifest_count == 0 or len(stitched) < 2:
        raise ValueError("at least two accepted states are required")
    return stitched


def select_load_interval(
    records: list[AcceptedStateRecord],
    *,
    start_load: float,
    end_load: float,
) -> list[AcceptedStateRecord]:
    """Select a closed load interval containing at least one transition."""
    if not np.isfinite(start_load) or not np.isfinite(end_load):
        raise ValueError("load bounds must be finite")
    if end_load <= start_load:
        raise ValueError("end-load must exceed start-load")
    selected = [
        record
        for record in records
        if record.load >= start_load - 1.0e-12
        and record.load <= end_load + 1.0e-12
    ]
    if len(selected) < 2:
        raise ValueError("selected load interval contains fewer than two states")
    if not np.isclose(selected[0].load, start_load, rtol=0.0, atol=5.0e-8):
        raise ValueError("start-load must match an accepted path state")
    if not np.isclose(selected[-1].load, end_load, rtol=0.0, atol=5.0e-8):
        raise ValueError("end-load must match an accepted path state")
    return selected


def _write_scan_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write flat path-consistent replay diagnostics in chronological order."""
    fieldnames = (
        "accepted_index",
        "previous_load",
        "load",
        "step_size",
        "target_is_report",
        "target_report_index",
        "physical_path_algorithm",
        "physical_path_iterations",
        "physical_damage_relaxation",
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
        writer.writerows({name: row[name] for name in fieldnames} for row in rows)


def _git_commit() -> str | None:
    """Return the current FractureX commit when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_scan(args: argparse.Namespace) -> Path:
    """Replay all selected immediate-neighbor continuation transitions."""
    output_dir = require_results_directory(args.output_dir)
    records = select_load_interval(
        read_internal_manifests(args.manifest),
        start_load=args.start_load,
        end_load=args.end_load,
    )
    main, material, mesh_stats, unused_nodes = build_model0_resolved_solver(
        hmin=args.hmin
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for accepted_index in range(1, len(records)):
        previous = records[accepted_index - 1]
        target = records[accepted_index]
        replay_summary, trace_rows = replay_transition(
            main,
            previous=CheckpointRecord(
                index=accepted_index - 1,
                load=previous.load,
                path=previous.path,
            ),
            target=CheckpointRecord(
                index=accepted_index,
                load=target.load,
                path=target.path,
            ),
            max_sweeps=args.max_sweeps,
            atol=args.atol,
            rtol=args.rtol,
            window_size=args.window_size,
            relative_singular_value=args.relative_singular_value,
            max_dimension=(None if args.max_dimension == 0 else args.max_dimension),
        )
        replay_summary.update(
            {
                "accepted_index": accepted_index,
                "step_size": target.load - previous.load,
                "target_is_report": target.is_report,
                "target_report_index": target.report_index,
                "physical_path_algorithm": target.algorithm_stage,
                "physical_path_iterations": target.iterations,
                "physical_damage_relaxation": target.damage_relaxation,
            }
        )
        summaries.append(replay_summary)
        trace_path = output_dir / (
            f"trace_step_{accepted_index:04d}_load_{target.load:.8f}.csv"
        )
        with trace_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(trace_rows[0]))
            writer.writeheader()
            writer.writerows(trace_rows)
        _write_scan_csv(output_dir / "online_rate_internal_path.csv", summaries)
        print(
            f"step={accepted_index:04d} load={target.load:.8f} "
            f"rhohat={replay_summary['rhohat_online']:.6f} "
            f"plain={replay_summary['plain_replay_sweeps']} "
            f"converged={replay_summary['plain_replay_converged']} "
            f"path_iterations={target.iterations}",
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
        "accepted_state_manifests": [
            str(path.resolve()) for path in args.manifest
        ],
        "selected_load_interval": [args.start_load, args.end_load],
        "transition_count": len(summaries),
        "path_contract": (
            "each replay starts from the immediate previous accepted "
            "continuation state with its committed H and damage lower bound"
        ),
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
            "max_dimension": None if args.max_dimension == 0 else args.max_dimension,
            "interpretation": (
                "finite-window increment rate; not labeled as spectral radius"
            ),
        },
        "random_seed": None,
        "output_dir": str(output_dir),
        "output_csv": str(
            (output_dir / "online_rate_internal_path.csv").resolve()
        ),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir / "online_rate_internal_path.csv"


def parse_args() -> argparse.Namespace:
    """Parse path-consistent replay parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="accepted_internal_states.csv; repeat in chronological stage order",
    )
    parser.add_argument("--start-load", type=float, default=0.0788)
    parser.add_argument("--end-load", type=float, default=0.1030)
    parser.add_argument("--hmin", type=float, default=0.0065)
    parser.add_argument("--max-sweeps", type=int, default=40)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--relative-singular-value", type=float, default=1.0e-2)
    parser.add_argument("--max-dimension", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the configured internal-path online-rate scan."""
    path = run_scan(parse_args())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
