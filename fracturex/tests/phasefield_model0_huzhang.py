from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import time
import os
import csv
import json
import matplotlib.pyplot as plt

from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.postprocess.recorder import RunRecorder


@dataclass
class Model0Material:
    E: float = 200.0
    nu: float = 0.2
    Gc: float = 1.0
    l0: float = 0.05

    @property
    def mu(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _export_paper_summary(
    infos,
    *,
    outdir: str,
    total_wall_s: float,
    solver_mode: str,
):
    """Export a compact run summary for tables/figures in papers."""
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


def main():
    performance_mode = True
    use_direct_solver = True  # 基线：双直接法
    release_elastic_iterative_only = False  # 逐项放开：只放开 elastic 线性求解
    use_fast_coarse_test = False  # True 仅用于提速冒烟，不用于结果对比
    run_epsg_sweep = False
    epsg_list = [1e-6]

    mat = Model0Material()

    hmin = 0.04
    case = Model0CircularNotchCase(
        _model=mat,
        hmin=hmin,
        distmesh_maxit=80,
        debug_mesh=False,
    )

    print("\n===== run configuration =====")
    print(f"use_direct_solver   = {use_direct_solver}")
    print(f"release_elastic_iterative_only = {release_elastic_iterative_only}")
    print(f"use_fast_coarse_test= {use_fast_coarse_test}")
    print(f"performance_mode   = {performance_mode}")
    print(f"material l0         = {mat.l0}")
    print(f"mesh hmin           = {hmin}")
    print(f"run_epsg_sweep      = {run_epsg_sweep}")
    print(f"epsg_list           = {epsg_list}")

    sweep_outdir = "results_model0/epsg_sweep"
    os.makedirs(sweep_outdir, exist_ok=True)
    compare_rows = []
    all_curves = []

    target_eps = epsg_list if run_epsg_sweep else [1e-6]
    for eps_g in target_eps:
        tag = f"epsg_{eps_g:.0e}"
        print(f"\n===== running case: {tag} =====")

        discr = HuZhangDiscretization(
            case=case,
            p=3,
            damage_p=2,
            use_relaxation=True,
        ).build()

        damage = PhaseFieldDamageModel(
            density_type="AT2",
            degradation_type="quadratic",
            split="hybrid",
            eps_g=float(eps_g),
            debug=False,
        )
        print(f"damage eps_g        = {damage.eps_g}")

        elastic_assembler = HuZhangElasticAssembler(discr, case, damage)
        phase_assembler = PhaseFieldAssembler(discr, case, damage, debug=(not performance_mode))
        recorder = RunRecorder(
            os.path.join(sweep_outdir, tag),
            save_npz=(not performance_mode),
            save_every=10 if performance_mode else 1,
        )

        if use_direct_solver:
            if release_elastic_iterative_only:
                def elastic_solver(A, F):
                    from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_auxspace

                    x, _ = solve_huzhang_block_gmres_auxspace(
                        A,
                        F,
                        gdof_sigma=discr.gdof_sigma,
                        vspace=discr.space_u,
                        atol=1e-12,
                        rtol=1e-8,
                        restart=20,
                        maxit=200,
                        sstep=3,
                        theta=0.25,
                        q=3,
                        schur_rebuild_interval=5,
                    )
                    return x
            else:
                elastic_solver = HuZhangPhaseFieldStaggeredDriver._default_spsolve
            driver = HuZhangPhaseFieldStaggeredDriver(
                case=case,
                discr=discr,
                damage=damage,
                elastic_assembler=elastic_assembler,
                phase_assembler=phase_assembler,
                tol=1e-5,
                maxit=1000,
                elastic_solver=elastic_solver,
                phase_solver=HuZhangPhaseFieldStaggeredDriver._default_spsolve,
                compute_linear_residual=True,
                debug=(not performance_mode),
                timing=True,
                recorder=recorder,
            )
        else:
            driver = HuZhangPhaseFieldStaggeredDriver(
                case=case,
                discr=discr,
                damage=damage,
                elastic_assembler=elastic_assembler,
                phase_assembler=phase_assembler,
                tol=1e-5,
                maxit=1000,
                debug=(not performance_mode),
                timing=True,
                recorder=recorder,
            )

        loads = np.asarray(case.default_loads(), dtype=float).tolist()
        t0 = time.perf_counter()
        infos = driver.run(loads)
        total_wall_s = float(time.perf_counter() - t0)

        loads_curve = [float(info.load) for info in infos]
        residual_forces = [abs(float(info.meta.get("residual_force", 0.0))) for info in infos]
        all_curves.append((eps_g, loads_curve, residual_forces))

        _export_paper_summary(
            infos,
            outdir=os.path.join(sweep_outdir, tag),
            total_wall_s=total_wall_s,
            solver_mode="direct/spsolve" if use_direct_solver and not release_elastic_iterative_only else "mixed/iterative",
        )
        if not performance_mode:
            driver._save_vtkfile(os.path.join(sweep_outdir, f"model0_huzhang_final_{tag}.vtu"), cell_mode="mean")

        nstep = max(len(infos), 1)
        compare_rows.append({
            "eps_g": float(eps_g),
            "n_load_steps": int(len(infos)),
            "n_converged_steps": int(sum(bool(info.converged) for info in infos)),
            "step_convergence_rate": float(sum(bool(info.converged) for info in infos) / nstep),
            "total_wall_s": float(total_wall_s),
            "avg_step_s": float(sum(float(info.meta.get("t_step_s", 0.0)) for info in infos) / nstep),
            "avg_elastic_assemble_s": float(sum(float(info.meta.get("t_elastic_assemble_s", 0.0)) for info in infos) / nstep),
            "avg_elastic_solve_s": float(sum(float(info.meta.get("t_elastic_solve_s", 0.0)) for info in infos) / nstep),
            "avg_phase_assemble_s": float(sum(float(info.meta.get("t_phase_assemble_s", 0.0)) for info in infos) / nstep),
            "avg_phase_solve_s": float(sum(float(info.meta.get("t_phase_solve_s", 0.0)) for info in infos) / nstep),
            "avg_nonlinear_iters": float(sum(int(info.iters) for info in infos) / nstep),
            "reaction_force_final_abs": float(residual_forces[-1] if residual_forces else 0.0),
            "reaction_force_peak_abs": float(max(residual_forces) if residual_forces else 0.0),
            "damage_max_final": float(infos[-1].max_d if infos else 0.0),
        })

    # Multi-curve residual-force comparison figure (absolute residual force vs load)
    plt.figure(figsize=(7.0, 4.5))
    for eps_g, loads_curve, residual_forces in all_curves:
        plt.plot(
            loads_curve,
            residual_forces,
            "-o",
            markersize=2.5,
            linewidth=1.1,
            label=f"eps_g={eps_g:.0e}",
        )
    plt.xlabel("Load")
    plt.ylabel("Residual force (absolute)")
    plt.title("Residual force vs load (eps_g sweep)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    cmp_fig = os.path.join(sweep_outdir, "residual_force_vs_load_epsg_compare.png")
    plt.tight_layout()
    plt.savefig(cmp_fig, dpi=170)
    print(f"Saved comparison figure: {cmp_fig}")

    # Comparison table CSV
    cmp_csv = os.path.join(sweep_outdir, "epsg_comparison_table.csv")
    if compare_rows:
        fieldnames = list(compare_rows[0].keys())
        with open(cmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(compare_rows)
    print(f"Saved comparison table: {cmp_csv}")


if __name__ == "__main__":
    main()
