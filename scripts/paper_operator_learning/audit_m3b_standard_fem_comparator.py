#!/usr/bin/env python3
"""Audit a genuine continuous-displacement Model-0 recovery comparator.

The script rebuilds the exact resolved standard-FEM mesh used by
``run_model0_fine_reference.py``, loads selected converged report checkpoints,
and evaluates ``g(d) C eps(u)`` directly from its continuous P1 displacement.
It reports native divergence/jump components and independently integrates the
top-boundary reaction, which is compared with the run's recorded reaction.

This smoke validates the standard-FEM recovery against its own discrete path.
It does not compare stress L2 directly with the Hu--Zhang hires data because
the existing paths use different residual-stiffness floors (1e-10 vs 1e-6).

Usage
-----
PYTHONPATH=$PWD python scripts/paper_operator_learning/audit_m3b_standard_fem_comparator.py \
  --run-dir ../results/phasefield_solver/model0_fine_curve_audit_unrelaxed_h0065 \
  --output-dir ../results/learn/m3b_standard_fem_comparator_smoke_v1 \
  --report-indices 21 30
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def load_recorded_reactions(csv_path: Path) -> dict[int, dict[str, float]]:
    """Load report-index keyed displacement/reaction rows from the physical path."""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        int(row["step"]): {
            "load": float(row["load"]),
            "reaction": float(row["residual_force_abs"]),
        }
        for row in rows
    }


def find_report_checkpoint(run_dir: Path, report_index: int) -> Path:
    """Return the unique checkpoint for one report index."""
    matches = sorted((run_dir / "checkpoints").glob(f"report_{report_index:02d}_load_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one checkpoint for report {report_index}; found {len(matches)}"
        )
    return matches[0]


def git_revision(repo_root: Path) -> str:
    """Return the current short Git revision or ``unknown``."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write stable scalar result rows to CSV."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-indices", nargs="+", type=int, default=[21, 30])
    parser.add_argument("--hmin", type=float, default=0.0065)
    parser.add_argument("--quadrature-order", type=int, default=6)
    parser.add_argument("--degradation-floor", type=float, default=1.0e-10)
    parser.add_argument("--reaction-relative-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    from fealpy.backend import backend_manager as bm
    from fracturex.learn.eval.native_stress_residual import (
        ContinuousP1RecoveredStress,
        compute_native_stress_residual,
        integrate_boundary_reaction,
    )
    from scripts.paper_solver.run_model0_fine_reference import (
        build_model0_resolved_solver,
    )

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    meta_path = output_dir / "meta.json"
    meta = {
        "status": "running",
        "command": shlex.join(sys.argv),
        "timestamp_started": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "git_commit": git_revision(repo_root),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "hmin": args.hmin,
        "quadrature_order": args.quadrature_order,
        "degradation_floor": args.degradation_floor,
        "reaction_relative_tolerance": args.reaction_relative_tolerance,
        "comparator": "continuous_P1_standard_FEM_displacement_recovery",
        "hz_hires_degradation_floor": 1.0e-6,
        "direct_hz_stress_l2_comparison_allowed": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    main_solver, material, mesh_stats, unused_nodes = build_model0_resolved_solver(
        hmin=args.hmin
    )
    reactions = load_recorded_reactions(run_dir / "residual_force_vs_displacement.csv")
    rows: list[dict] = []
    for report_index in args.report_indices:
        if report_index not in reactions:
            raise ValueError(f"report index {report_index} is absent from reaction CSV")
        checkpoint_path = find_report_checkpoint(run_dir, report_index)
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            displacement = np.asarray(checkpoint["uh"], dtype=np.float64)
            damage = np.asarray(checkpoint["d"], dtype=np.float64)
        if displacement.size != int(main_solver.tspace.number_of_global_dofs()):
            raise ValueError("checkpoint displacement size does not match rebuilt mesh")
        if damage.size != int(main_solver.space.number_of_global_dofs()):
            raise ValueError("checkpoint damage size does not match rebuilt mesh")
        main_solver.uh[:] = bm.asarray(displacement, dtype=main_solver.uh.dtype)
        main_solver.d[:] = bm.asarray(damage, dtype=main_solver.d.dtype)
        main_solver.pfcm.update_disp(main_solver.uh)
        main_solver.pfcm.update_phase(main_solver.d)

        recovered = ContinuousP1RecoveredStress(
            main_solver.mesh,
            main_solver.space,
            main_solver.tspace,
            displacement,
            damage,
            young_modulus=float(material["E"]),
            poisson_ratio=float(material["nu"]),
            degradation_floor=args.degradation_floor,
        )
        residual = compute_native_stress_residual(
            main_solver.mesh,
            recovered.value,
            recovered.divergence,
            quadrature_order=args.quadrature_order,
        )
        integrated_reaction = abs(integrate_boundary_reaction(
            main_solver.mesh,
            recovered.value,
            lambda points: np.abs(points[:, 1] - 1.0) < 1.0e-12,
            component=1,
            quadrature_order=args.quadrature_order,
        ))
        recorded = reactions[report_index]
        relative_error = abs(integrated_reaction - recorded["reaction"]) / max(
            abs(recorded["reaction"]), 1.0e-30
        )
        row = {
            "report_index": report_index,
            "load": recorded["load"],
            "max_damage": float(np.max(damage)),
            "stress_l2_norm": residual.stress_l2_norm,
            "relative_estimator": residual.relative_estimator,
            "relative_volume": residual.relative_volume,
            "relative_jump": residual.relative_jump,
            "max_traction_jump": residual.max_traction_jump,
            "recorded_reaction": recorded["reaction"],
            "integrated_reaction": integrated_reaction,
            "reaction_relative_error": relative_error,
        }
        rows.append(row)
        print(
            f"[standard-rec] report={report_index} load={recorded['load']:.6f} "
            f"res={residual.relative_estimator:.3e} "
            f"reaction={integrated_reaction:.6e}/{recorded['reaction']:.6e} "
            f"error={relative_error:.3e}",
            flush=True,
        )

    gate_passed = all(
        row["reaction_relative_error"] <= args.reaction_relative_tolerance
        for row in rows
    )
    summary = {
        "n_rows": len(rows),
        "max_reaction_relative_error": max(row["reaction_relative_error"] for row in rows),
        "reaction_gate_passed": gate_passed,
        "interpretation": (
            "Reaction consistency validates stress recovery on its own standard-FEM path. "
            "Native residual remains a non-H(div) source diagnostic."
        ),
    }
    write_csv(output_dir / "rows.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    meta.update({
        "status": "complete",
        "timestamp_finished": datetime.now().astimezone().isoformat(),
        "mesh_stats": mesh_stats,
        "unused_distmesh_nodes_removed": unused_nodes,
        **summary,
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[standard-rec] reaction_gate_passed={gate_passed} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
