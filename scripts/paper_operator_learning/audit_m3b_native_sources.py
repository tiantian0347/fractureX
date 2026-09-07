#!/usr/bin/env python3
"""Audit native HZ and Oswald-recovered stress sources on M3b recorder data.

For each completed dataset sample, the script selects representative converged
load steps, rebuilds the Hu--Zhang spaces from ``mesh.npz``, and evaluates the
native triangle residual (cell divergence + normal-traction jumps) for:

1. the primary Hu--Zhang stress;
2. ``g(d) C eps(u_O)`` where ``u_O`` is the continuous-P2 Oswald average of
   the stored Hu--Zhang P2-DG displacement.

The Oswald field is a controlled continuous reconstruction of the same mixed
solution, not an independently solved standard-FEM comparator. Results are
appended to CSV after every step and accompanied by ``meta.json``.

Usage
-----
PYTHONPATH=$PWD python scripts/paper_operator_learning/audit_m3b_native_sources.py \
    --dataset-dir results/datasets/m3b_hires_pilot \
    --output-dir results/learn/m3b_hires_native_source_audit_v1 \
    --max-steps-per-sample 4
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


def select_converged_steps(
    checkpoint_steps: list[int],
    step_converged: np.ndarray | None,
    max_steps: int,
) -> list[int]:
    """Select evenly spaced converged checkpoint indices, including endpoints."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    candidates = checkpoint_steps
    if step_converged is not None:
        candidates = [
            step for step in checkpoint_steps
            if step < step_converged.size and bool(step_converged[step])
        ]
    if not candidates:
        return []
    if len(candidates) <= max_steps:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, max_steps).round().astype(int)
    return [candidates[index] for index in np.unique(positions)]


def load_material(sample_meta_path: Path, run_meta_path: Path) -> tuple[float, float, str]:
    """Return ``(E, nu, plane)`` from dataset metadata with run fallback."""
    sources = []
    for path in (sample_meta_path, run_meta_path):
        if path.exists():
            sources.append(json.loads(path.read_text()))
    for source in sources:
        material = source.get("material_params", source.get("material", {}))
        if "E" in material and "nu" in material:
            formulation = str(source.get("formulation", source.get("elastic_formulation", "standard")))
            plane = "stress" if formulation.lower() in {"plane_stress", "stress"} else "strain"
            return float(material["E"]), float(material["nu"]), plane
        if "lam" in material and "mu" in material:
            lam = float(material["lam"])
            mu = float(material["mu"])
            young = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
            poisson = lam / (2.0 * (lam + mu))
            return young, poisson, "strain"
    raise ValueError(f"cannot recover E/nu from {sample_meta_path} or {run_meta_path}")


def append_csv(path: Path, row: dict) -> None:
    """Append one stable-schema result row and flush it to disk."""
    write_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--max-steps-per-sample", type=int, default=4)
    parser.add_argument("--quadrature-order", type=int, default=6)
    parser.add_argument("--residual-stiffness", type=float, default=1.0e-6)
    args = parser.parse_args()

    from fealpy.backend import backend_manager as bm
    from fracturex.learn.eval.native_stress_residual import (
        OswaldRecoveredStress,
        compute_native_relative_l2_difference,
        compute_native_stress_residual,
        make_huzhang_stress_callables,
    )
    from fracturex.postprocess.dataset_export import load_discr_from_dir
    from fracturex.postprocess.dataset_export.adapters.huzhang_phasefield import (
        verify_discr_matches_checkpoint,
    )

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    meta_path = output_dir / "meta.json"
    rows_path = output_dir / "native_source_rows.csv"
    started = datetime.now().astimezone()
    meta = {
        "status": "running",
        "command": shlex.join(sys.argv),
        "timestamp_started": started.isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "hostname": socket.gethostname(),
        "git_commit": git_revision(repo_root),
        "quadrature_order": args.quadrature_order,
        "residual_stiffness": args.residual_stiffness,
        "max_steps_per_sample": args.max_steps_per_sample,
        "comparator": "continuous_P2_Oswald_average_of_HZ_P2DG_displacement",
        "independent_standard_fem_comparator": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    samples_dir = dataset_dir / "samples"
    runs_dir = dataset_dir / "runs"
    # A run directory can survive an interrupted solve without a completed
    # exported sample (M3b sample_000006 is exactly such a case). Default to
    # exported npz records so truncated runs never enter the source audit.
    sample_ids = args.samples or sorted(
        path.stem
        for path in samples_dir.glob("sample_*.npz")
        if (runs_dir / path.stem).is_dir()
    )
    rows: list[dict] = []
    for sample_id in sample_ids:
        run_dir = runs_dir / sample_id
        checkpoint_paths = sorted((run_dir / "checkpoints").glob("step_[0-9][0-9][0-9].npz"))
        checkpoint_by_step = {
            int(path.stem.split("_")[-1]): path for path in checkpoint_paths
        }
        sample_npz_path = samples_dir / f"{sample_id}.npz"
        step_converged = None
        loads = None
        if sample_npz_path.exists():
            with np.load(sample_npz_path, allow_pickle=False) as sample_npz:
                if "step_converged" in sample_npz:
                    step_converged = np.asarray(sample_npz["step_converged"], dtype=bool)
                if "load_history" in sample_npz:
                    loads = np.asarray(sample_npz["load_history"], dtype=np.float64).reshape(-1)
        steps = select_converged_steps(
            sorted(checkpoint_by_step), step_converged, args.max_steps_per_sample
        )
        if not steps:
            print(f"[native] skip {sample_id}: no converged checkpoints", flush=True)
            continue

        discr = load_discr_from_dir(run_dir)
        young, poisson, plane = load_material(
            samples_dir / f"{sample_id}.meta.json", run_dir / "meta.json"
        )
        print(
            f"[native] {sample_id}: NC={discr.mesh.number_of_cells()} steps={steps}",
            flush=True,
        )
        for step in steps:
            checkpoint_path = checkpoint_by_step[step]
            verify_discr_matches_checkpoint(discr, checkpoint_path)
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                discr.state.sigma[:] = bm.asarray(checkpoint["sigma"])
                discr.state.u[:] = bm.asarray(checkpoint["u"])
                discr.state.d[:] = bm.asarray(checkpoint["d"])
            hz_value, hz_divergence = make_huzhang_stress_callables(discr)
            recovered = OswaldRecoveredStress(
                discr,
                young_modulus=young,
                poisson_ratio=poisson,
                plane=plane,
                residual_stiffness=args.residual_stiffness,
            )
            hz = compute_native_stress_residual(
                discr.mesh,
                hz_value,
                hz_divergence,
                quadrature_order=args.quadrature_order,
            )
            rec = compute_native_stress_residual(
                discr.mesh,
                recovered.value,
                recovered.divergence,
                quadrature_order=args.quadrature_order,
            )
            rec_to_hz_l2 = compute_native_relative_l2_difference(
                discr.mesh,
                recovered.value,
                hz_value,
                quadrature_order=args.quadrature_order,
            )
            row = {
                "sample_id": sample_id,
                "step": step,
                "load": float(loads[step]) if loads is not None and step < loads.size else float("nan"),
                "max_damage": float(np.max(np.asarray(discr.state.d))),
                "n_cells": int(discr.mesh.number_of_cells()),
                "h_min": float(np.min(np.asarray(discr.mesh.entity_measure("edge")))),
                "h_max": float(np.max(np.asarray(discr.mesh.entity_measure("edge")))),
                "rec_to_hz_relative_l2": rec_to_hz_l2,
                "hz_stress_l2_norm": hz.stress_l2_norm,
                "hz_relative_estimator": hz.relative_estimator,
                "hz_relative_volume": hz.relative_volume,
                "hz_relative_jump": hz.relative_jump,
                "hz_max_traction_jump": hz.max_traction_jump,
                "rec_stress_l2_norm": rec.stress_l2_norm,
                "rec_relative_estimator": rec.relative_estimator,
                "rec_relative_volume": rec.relative_volume,
                "rec_relative_jump": rec.relative_jump,
                "rec_max_traction_jump": rec.max_traction_jump,
            }
            append_csv(rows_path, row)
            rows.append(row)
            print(
                f"[native] {sample_id} step={step} "
                f"hz={hz.relative_estimator:.3e} rec={rec.relative_estimator:.3e} "
                f"rec/hz-L2={rec_to_hz_l2:.3e}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("native source audit produced no rows")
    informative_rows = [row for row in rows if row["hz_stress_l2_norm"] > 1.0e-12]
    if not informative_rows:
        raise RuntimeError("native source audit has no nonzero-stress rows")
    hz = np.asarray([row["hz_relative_estimator"] for row in informative_rows])
    rec = np.asarray([row["rec_relative_estimator"] for row in informative_rows])
    positive_hz = np.maximum(hz, 1.0e-30)
    summary = {
        "n_rows": len(rows),
        "n_informative_rows": len(informative_rows),
        "n_samples": len({row["sample_id"] for row in rows}),
        "hz_better_fraction": float(np.mean(hz < rec)),
        "median_hz_relative_estimator": float(np.median(hz)),
        "median_rec_relative_estimator": float(np.median(rec)),
        "median_rec_to_hz_residual_ratio": float(np.median(rec / positive_hz)),
        "median_rec_to_hz_relative_l2": float(
            np.median([row["rec_to_hz_relative_l2"] for row in informative_rows])
        ),
    }
    summary["source_gate_passed"] = bool(
        summary["hz_better_fraction"] >= 0.9
        and summary["median_rec_to_hz_residual_ratio"] >= 10.0
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    meta.update({
        "status": "complete",
        "timestamp_finished": datetime.now().astimezone().isoformat(),
        **summary,
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[native] source_gate_passed={summary['source_gate_passed']} output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
