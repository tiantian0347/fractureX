from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return int(default)


def summarize_timing(
    run_dir: str = "results_model0/epsg_sweep/epsg_1e-06",
):
    """
    Build per-load-step timing decomposition from recorder outputs.

    Expected files in run_dir:
      - history.csv
      - iterations.csv
    """
    history_csv = os.path.join(run_dir, "history.csv")
    iter_csv = os.path.join(run_dir, "iterations.csv")
    out_csv = os.path.join(run_dir, "timing_step_breakdown.csv")
    out_fig = os.path.join(run_dir, "timing_step_breakdown_stacked.png")

    if (not os.path.exists(history_csv)) or (not os.path.exists(iter_csv)):
        raise FileNotFoundError(
            f"Missing history/iterations csv in '{run_dir}'. "
            f"Need both '{history_csv}' and '{iter_csv}'."
        )

    per_step = defaultdict(lambda: {
        "n_iter": 0,
        "t_elastic_assemble_sum": 0.0,
        "t_elastic_solve_sum": 0.0,
        "t_phase_assemble_sum": 0.0,
        "t_phase_solve_sum": 0.0,
        "t_iter_sum": 0.0,
    })

    with open(iter_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = _to_int(row.get("step", 0), 0)
            item = per_step[step]
            item["n_iter"] += 1
            item["t_elastic_assemble_sum"] += _to_float(row.get("t_elastic_assemble_iter_s", 0.0))
            item["t_elastic_solve_sum"] += _to_float(row.get("t_elastic_solve_iter_s", 0.0))
            item["t_phase_assemble_sum"] += _to_float(row.get("t_phase_assemble_iter_s", 0.0))
            item["t_phase_solve_sum"] += _to_float(row.get("t_phase_solve_iter_s", 0.0))
            item["t_iter_sum"] += _to_float(row.get("t_iter_s", 0.0))

    history_by_step = {}
    with open(history_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = _to_int(row.get("step", 0), 0)
            history_by_step[step] = row

    steps = sorted(set(list(per_step.keys()) + list(history_by_step.keys())))
    rows = []
    for s in steps:
        it = per_step[s]
        hist = history_by_step.get(s, {})
        load = _to_float(hist.get("load", 0.0))
        residual_force = abs(_to_float(hist.get("residual_force", 0.0)))
        t_step = _to_float(hist.get("t_step_s", 0.0))
        t_ea = _to_float(hist.get("t_elastic_assemble_s", it["t_elastic_assemble_sum"]))
        t_es = _to_float(hist.get("t_elastic_solve_s", it["t_elastic_solve_sum"]))
        t_pa = _to_float(hist.get("t_phase_assemble_s", it["t_phase_assemble_sum"]))
        t_ps = _to_float(hist.get("t_phase_solve_s", it["t_phase_solve_sum"]))
        t_parts = t_ea + t_es + t_pa + t_ps
        denom = max(t_parts, 1e-30)

        rows.append({
            "step": s,
            "load": load,
            "residual_force_abs": residual_force,
            "n_iter": int(it["n_iter"]),
            "t_step_s": t_step,
            "t_elastic_assemble_s": t_ea,
            "t_elastic_solve_s": t_es,
            "t_phase_assemble_s": t_pa,
            "t_phase_solve_s": t_ps,
            "t_parts_sum_s": t_parts,
            "ratio_elastic_assemble": t_ea / denom,
            "ratio_elastic_solve": t_es / denom,
            "ratio_phase_assemble": t_pa / denom,
            "ratio_phase_solve": t_ps / denom,
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "step", "load", "residual_force_abs", "n_iter",
            "t_step_s", "t_elastic_assemble_s", "t_elastic_solve_s",
            "t_phase_assemble_s", "t_phase_solve_s", "t_parts_sum_s",
            "ratio_elastic_assemble", "ratio_elastic_solve",
            "ratio_phase_assemble", "ratio_phase_solve",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    # stacked timing bars by load step
    if rows:
        xs = [r["step"] for r in rows]
        ea = [r["t_elastic_assemble_s"] for r in rows]
        es = [r["t_elastic_solve_s"] for r in rows]
        pa = [r["t_phase_assemble_s"] for r in rows]
        ps = [r["t_phase_solve_s"] for r in rows]

        plt.figure(figsize=(8.2, 4.6))
        plt.bar(xs, ea, label="elastic assemble")
        plt.bar(xs, es, bottom=ea, label="elastic solve")
        bottom2 = [a + b for a, b in zip(ea, es)]
        plt.bar(xs, pa, bottom=bottom2, label="phase assemble")
        bottom3 = [a + b + c for a, b, c in zip(ea, es, pa)]
        plt.bar(xs, ps, bottom=bottom3, label="phase solve")
        plt.xlabel("Load step")
        plt.ylabel("Time (s)")
        plt.title("Timing breakdown per load step")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_fig, dpi=170)

    print(f"Saved timing breakdown table: {out_csv}")
    print(f"Saved timing breakdown figure: {out_fig}")


if __name__ == "__main__":
    summarize_timing()

