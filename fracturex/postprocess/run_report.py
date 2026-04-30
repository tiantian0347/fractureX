"""Export run summaries (JSON/CSV) and Markdown test reports for Hu–Zhang + phase-field demos."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def export_paper_summary(
    infos,
    *,
    outdir: str,
    total_wall_s: float,
    solver_mode: str,
    history_source: str,
) -> None:
    """Export aggregated run summary in JSON/CSV.

    Inputs:
        infos: Iterable of per-step `StepInfo` objects.
        outdir: Output directory.
        total_wall_s: End-to-end runtime in seconds.
        solver_mode: Human-readable solver mode label.
        history_source: History-driving source label (e.g. `from_u`).
    Output:
        None. Writes `summary.json` and `summary.csv`.
    """
    os.makedirs(outdir, exist_ok=True)
    nstep = max(len(infos), 1)

    step_times = [float(info.meta.get("t_step_s", 0.0)) for info in infos]
    e_assm = [float(info.meta.get("t_elastic_assemble_s", 0.0)) for info in infos]
    e_solve = [float(info.meta.get("t_elastic_solve_s", 0.0)) for info in infos]
    p_assm = [float(info.meta.get("t_phase_assemble_s", 0.0)) for info in infos]
    p_solve = [float(info.meta.get("t_phase_solve_s", 0.0)) for info in infos]
    iters = [int(info.iters) for info in infos]
    converged = [bool(info.converged) for info in infos]
    residual_force = [float(info.meta.get("residual_force", 0.0)) for info in infos]
    max_d = [float(info.meta.get("max_d", info.max_d)) for info in infos]

    e_niters = [int(info.meta.get("linear_niter_elastic", -1)) for info in infos]
    d_niters = [int(info.meta.get("linear_niter_phase", -1)) for info in infos]
    e_convs = [bool(info.meta.get("linear_converged_elastic", False)) for info in infos]
    d_convs = [bool(info.meta.get("linear_converged_phase", False)) for info in infos]
    e_res = [float(info.meta.get("linear_res_elastic", float("nan"))) for info in infos]
    d_res = [float(info.meta.get("linear_res_phase", float("nan"))) for info in infos]

    valid_e_niters = [v for v in e_niters if v >= 0]
    valid_d_niters = [v for v in d_niters if v >= 0]
    valid_e_res = [v for v in e_res if np.isfinite(v)]
    valid_d_res = [v for v in d_res if np.isfinite(v)]

    summary = {
        "solver_mode": solver_mode,
        "history_source": str(history_source),
        "n_load_steps": int(len(infos)),
        "n_converged_steps": int(sum(converged)),
        "step_convergence_rate": float(sum(converged) / nstep),
        "total_wall_s": float(total_wall_s),
        "sum_step_s": float(sum(step_times)),
        "avg_step_s": float(sum(step_times) / nstep),
        "max_step_s": float(max(step_times) if step_times else 0.0),
        "sum_elastic_assemble_s": float(sum(e_assm)),
        "sum_elastic_solve_s": float(sum(e_solve)),
        "sum_phase_assemble_s": float(sum(p_assm)),
        "sum_phase_solve_s": float(sum(p_solve)),
        "avg_elastic_assemble_s_per_step": float(sum(e_assm) / nstep),
        "avg_elastic_solve_s_per_step": float(sum(e_solve) / nstep),
        "avg_phase_assemble_s_per_step": float(sum(p_assm) / nstep),
        "avg_phase_solve_s_per_step": float(sum(p_solve) / nstep),
        "avg_nonlinear_iters": float(sum(iters) / nstep),
        "max_nonlinear_iters": int(max(iters) if iters else 0),
        "avg_linear_niter_elastic": float(sum(valid_e_niters) / max(len(valid_e_niters), 1)),
        "avg_linear_niter_phase": float(sum(valid_d_niters) / max(len(valid_d_niters), 1)),
        "elastic_linear_convergence_rate": float(sum(e_convs) / nstep),
        "phase_linear_convergence_rate": float(sum(d_convs) / nstep),
        "elastic_linear_res_avg": float(sum(valid_e_res) / max(len(valid_e_res), 1)) if valid_e_res else float("nan"),
        "elastic_linear_res_max": float(max(valid_e_res)) if valid_e_res else float("nan"),
        "phase_linear_res_avg": float(sum(valid_d_res) / max(len(valid_d_res), 1)) if valid_d_res else float("nan"),
        "phase_linear_res_max": float(max(valid_d_res)) if valid_d_res else float("nan"),
        "reaction_force_max": float(max(residual_force) if residual_force else 0.0),
        "reaction_force_min": float(min(residual_force) if residual_force else 0.0),
        "reaction_force_final": float(residual_force[-1] if residual_force else 0.0),
        "reaction_force_abs_max": float(max(abs(v) for v in residual_force) if residual_force else 0.0),
        "reaction_force_abs_min": float(min(abs(v) for v in residual_force) if residual_force else 0.0),
        "reaction_force_abs_final": float(abs(residual_force[-1]) if residual_force else 0.0),
        "damage_max_final": float(max_d[-1] if max_d else 0.0),
        "damage_max_peak": float(max(max_d) if max_d else 0.0),
    }

    summary_json = os.path.join(outdir, "summary.json")
    summary_csv = os.path.join(outdir, "summary.csv")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])

    print(f"Saved paper summary JSON: {summary_json}")
    print(f"Saved paper summary CSV : {summary_csv}")


def export_model0_huzhang_test_markdown(
    *,
    outdir: str,
    tag: str,
    solver_mode: str,
    history_source: str,
    eps_g: float,
    hmin: float,
    summary: dict,
    meta: dict,
) -> str:
    """Generate Markdown test report for Model0 HuZhang run.

    Inputs:
        outdir: Output directory.
        tag: Run tag string.
        solver_mode: Solver mode description.
        history_source: History update source description.
        eps_g: Residual degradation floor.
        hmin: Mesh minimum size parameter.
        summary: Aggregated summary dict.
        meta: Run metadata dict.
    Output:
        Path string of generated `TEST_REPORT.md`.
    """
    os.makedirs(outdir, exist_ok=True)
    report_md = os.path.join(outdir, "TEST_REPORT.md")

    mesh = meta.get("mesh", {})
    material = meta.get("material", {})
    lines = [
        "# Model0 Hu-Zhang + Phase-Field Test Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Tag: `{tag}`",
        f"- Output directory: `{outdir}`",
        "",
        "## Run Configuration",
        "",
        f"- Solver mode: `{solver_mode}`",
        f"- Elastic formulation: `{meta.get('elastic_formulation', 'unknown')}`",
        f"- history_source: `{history_source}`",
        f"- eps_g: `{eps_g:.1e}`",
        f"- hmin: `{hmin}`",
        f"- p: `{meta.get('p', 'unknown')}`",
        f"- use_relaxation: `{meta.get('use_relaxation', 'unknown')}`",
        "",
        "## Mesh and Material",
        "",
        f"- Mesh: NN={mesh.get('NN', 'N/A')}, NE={mesh.get('NE', 'N/A')}, NC={mesh.get('NC', 'N/A')}",
        (
            f"- Material: E={material.get('E', 'N/A')}, nu={material.get('nu', 'N/A')}, "
            f"Gc={material.get('Gc', 'N/A')}, l0={material.get('l0', 'N/A')}"
        ),
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n_load_steps | {summary.get('n_load_steps', 'N/A')} |",
        f"| n_converged_steps | {summary.get('n_converged_steps', 'N/A')} |",
        f"| step_convergence_rate | {summary.get('step_convergence_rate', 'N/A')} |",
        f"| total_wall_s | {summary.get('total_wall_s', 'N/A')} |",
        f"| avg_step_s | {summary.get('avg_step_s', 'N/A')} |",
        f"| max_step_s | {summary.get('max_step_s', 'N/A')} |",
        f"| avg_nonlinear_iters | {summary.get('avg_nonlinear_iters', 'N/A')} |",
        f"| max_nonlinear_iters | {summary.get('max_nonlinear_iters', 'N/A')} |",
        f"| reaction_force_final (signed) | {summary.get('reaction_force_final', 'N/A')} |",
        f"| reaction_force_abs_final | {summary.get('reaction_force_abs_final', 'N/A')} |",
        f"| damage_max_final | {summary.get('damage_max_final', 'N/A')} |",
        "",
        "## Timing Breakdown",
        "",
        "| Phase | Total (s) | Avg per step (s) |",
        "|---|---:|---:|",
        (
            f"| Elastic Assemble | {summary.get('sum_elastic_assemble_s', 'N/A')} | "
            f"{summary.get('avg_elastic_assemble_s_per_step', 'N/A')} |"
        ),
        (
            f"| Elastic Solve | {summary.get('sum_elastic_solve_s', 'N/A')} | "
            f"{summary.get('avg_elastic_solve_s_per_step', 'N/A')} |"
        ),
        (
            f"| Phase Assemble | {summary.get('sum_phase_assemble_s', 'N/A')} | "
            f"{summary.get('avg_phase_assemble_s_per_step', 'N/A')} |"
        ),
        (
            f"| Phase Solve | {summary.get('sum_phase_solve_s', 'N/A')} | "
            f"{summary.get('avg_phase_solve_s_per_step', 'N/A')} |"
        ),
        "",
        "## Artifacts",
        "",
        "- `summary.json`: machine-readable aggregate metrics",
        "- `summary.csv`: flat metrics table",
        "- `history.csv`: per-load-step history",
        "- `meta.json`: run metadata",
        "- `residual_force_vs_displacement.csv`: per-step displacement and |residual_force|",
        "- `residual_force_vs_displacement.png`: residual-force/displacement curve",
        "",
        "> This file is auto-generated and overwritten on each run.",
        "",
    ]

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved Markdown report : {report_md}")
    return report_md


def export_residual_force_displacement_curve(
    infos,
    *,
    outdir: str,
    csv_name: str = "residual_force_vs_displacement.csv",
    fig_name: str = "residual_force_vs_displacement.png",
) -> tuple[str, str]:
    """
    Export residual-force/displacement data and plot.

    Inputs:
        infos: Iterable of per-step `StepInfo` objects.
        outdir: Output directory.
        csv_name: CSV filename.
        fig_name: Figure filename.
    Output:
        Tuple `(csv_path, fig_path)`.

    Residual force is exported as absolute value to keep it non-negative.
    """
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, csv_name)
    fig_path = os.path.join(outdir, fig_name)

    disp = np.asarray([float(info.load) for info in infos], dtype=float)
    residual_abs = np.asarray(
        [abs(float(info.meta.get("residual_force", 0.0))) for info in infos],
        dtype=float,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "displacement", "residual_force_abs"])
        for i, (x, y) in enumerate(zip(disp, residual_abs)):
            writer.writerow([i, x, y])

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(disp, residual_abs, "-o", markersize=2.5, linewidth=1.2)
    plt.xlabel("Displacement")
    plt.ylabel("Residual force (absolute)")
    plt.title("Residual force vs displacement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170)
    plt.close()

    print(f"Saved residual curve CSV: {csv_path}")
    print(f"Saved residual curve PNG: {fig_path}")
    return csv_path, fig_path
