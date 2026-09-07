#!/usr/bin/env python3
"""Audit saved M3b S-tier checkpoints and decide the Stage-D acceptance gate.

The script re-evaluates every completed run in an M3b sweep on the held-out
split, including the grid-FD equilibrium residual for the prediction and its
supervision target. It writes a complete JSON/CSV table plus reproducibility
metadata without modifying checkpoints or dataset files.

This audit is deliberately limited to the structured-grid FD diagnostic. It
does not test the native-mesh broken-divergence/normal-jump claim and therefore
cannot by itself establish the paper's H(div) source-level theorem.

Usage
-----
PYTHONPATH=$PWD python scripts/paper_operator_learning/audit_m3b_s_sweep.py \
    --sweep-dir results/learn/m3b_S_hz_sweep \
    --output-dir results/learn/m3b_S_hz_sweep/acceptance_audit
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


QUALITY_METRICS = (
    "relative_l2",
    "sigma_relative_l2",
    "peak_load_error",
)


def load_final_training_metrics(run_dir: Path) -> dict[str, float]:
    """Load the final epoch's scalar metrics from one saved run.

    Parameters
    ----------
    run_dir : Path
        Run directory containing ``metrics.csv``.

    Returns
    -------
    dict[str, float]
        Final CSV row converted to floats where possible. A newly allocated
        dictionary is returned.
    """
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"training metrics are empty: {metrics_path}")
    final: dict[str, float] = {}
    for key, value in rows[-1].items():
        if value not in (None, ""):
            final[key] = float(value)
    return final


def evaluate_acceptance_gate(
    rows: list[dict],
    *,
    max_quality_degradation: float,
) -> dict:
    """Check whether a regularized run improves balance at controlled quality.

    A candidate passes when its held-out grid-FD residual is below the
    unregularized baseline and each headline quality error is at most
    ``1 + max_quality_degradation`` times the baseline. All quantities are
    dimensionless and lower is better.
    """
    baselines = [row for row in rows if float(row["lambda_eq"]) == 0.0]
    if len(baselines) != 1:
        raise ValueError(f"expected exactly one lambda_eq=0 baseline; got {len(baselines)}")
    baseline = baselines[0]
    residual_key = "equilibrium_residual_pred_fd"
    passing: list[float] = []
    candidate_checks: list[dict] = []
    for row in rows:
        lam = float(row["lambda_eq"])
        if lam <= 0.0:
            continue
        quality_ok = all(
            float(row[key]) <= (1.0 + max_quality_degradation) * float(baseline[key])
            for key in QUALITY_METRICS
        )
        residual_ok = float(row[residual_key]) < float(baseline[residual_key])
        passed = bool(quality_ok and residual_ok)
        if passed:
            passing.append(lam)
        candidate_checks.append({
            "lambda_eq": lam,
            "quality_ok": quality_ok,
            "residual_ok": residual_ok,
            "passed": passed,
        })
    return {
        "passed": bool(passing),
        "passing_lambda_eq": passing,
        "max_quality_degradation": max_quality_degradation,
        "quality_metrics": list(QUALITY_METRICS),
        "residual_metric": residual_key,
        "candidate_checks": candidate_checks,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write scalar audit rows to CSV without mutating them."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_revision(repo_root: Path) -> str:
    """Return the current Git revision, or ``unknown`` outside a Git checkout."""
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
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-quality-degradation",
        type=float,
        default=0.05,
        help="maximum relative degradation allowed for each quality error",
    )
    args = parser.parse_args()

    from fracturex.learn.train import evaluate_saved_run

    sweep_dir = args.sweep_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        path.parent for path in sweep_dir.glob("*/config.json")
        if (path.parent / "checkpoints" / "model_final.pt").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"no completed saved runs below {sweep_dir}")

    repo_root = Path(__file__).resolve().parents[2]
    started = datetime.now().astimezone()
    meta = {
        "status": "running",
        "command": shlex.join(sys.argv),
        "timestamp_started": started.isoformat(),
        "output_dir": str(output_dir),
        "sweep_dir": str(sweep_dir),
        "device": args.device,
        "batch_size": args.batch_size,
        "max_quality_degradation": args.max_quality_degradation,
        "hostname": socket.gethostname(),
        "git_commit": git_revision(repo_root),
        "residual_kind": "structured_grid_central_difference",
        "native_fe_jump_residual_included": False,
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    rows: list[dict] = []
    for run_dir in run_dirs:
        saved_config = json.loads((run_dir / "config.json").read_text())
        lam = saved_config.get("lambda_eq")
        lambda_eq = 0.0 if lam is None else float(lam)
        print(f"[audit] evaluating {run_dir.name}: lambda_eq={lambda_eq}", flush=True)
        held_out = evaluate_saved_run(
            run_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        training = load_final_training_metrics(run_dir)
        row = {
            "run": run_dir.name,
            "lambda_eq": lambda_eq,
            "seed": int(saved_config.get("seed", 0)),
            **held_out,
        }
        for key in ("train_l_eq", "train_l_eq_norm", "train_sigma_ref"):
            if key in training:
                row[key] = training[key]
        rows.append(row)

    rows.sort(key=lambda row: float(row["lambda_eq"]))
    gate = evaluate_acceptance_gate(
        rows,
        max_quality_degradation=args.max_quality_degradation,
    )
    payload = {
        "rows": rows,
        "gate": gate,
        "interpretation": (
            "This gate covers held-out grid-FD balance versus prediction quality. "
            "A native-mesh jump/weak-residual audit is still required before "
            "testing the paper's H(div) source-level claim."
        ),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(payload, indent=2))
    write_csv(output_dir / "audit_summary.csv", rows)

    meta.update({
        "status": "complete",
        "timestamp_finished": datetime.now().astimezone().isoformat(),
        "n_runs": len(rows),
        "gate_passed": gate["passed"],
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[audit] gate_passed={gate['passed']} output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
