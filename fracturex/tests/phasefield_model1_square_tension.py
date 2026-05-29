"""
Hu-Zhang + staggered phase-field: square y-tension (paper model-1 style).

Intact unit-square mesh + phase-field pre-crack (d=1 on y=0.5, x∈[0,0.5]);
no geometric notch mesh (Hu-Zhang corner issues on cut meshes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import time
import os

from scipy.sparse.linalg import gmres as scipy_gmres
from scipy.sparse.linalg import spsolve as scipy_spsolve

from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.postprocess.recorder import RunRecorder
from fracturex.postprocess.run_paths import epsg_tag, phasefield_tag_dir
from fracturex.postprocess.run_report import (
    export_paper_summary,
    export_residual_force_displacement_curve,
)
from fracturex.utilfuc.linear_solvers import (
    KrylovInfo,
    _extract_converged_from_info,
    solve_huzhang_block_gmres_auxspace,
)


def _phase_gmres(
    A,
    F,
    *,
    restart: int = 80,
    maxiter: int = 3000,
    atol: float = 1e-20,
    rtol: float = 1e-10,
    check_rtol: float = 1e-8,
    fallback_to_spsolve: bool = True,
):
    """Phase-field linear solve: unpreconditioned GMRES with optional direct fallback."""
    if hasattr(A, "to_scipy"):
        A_ = A.to_scipy().tocsr()
    else:
        A_ = A.tocsr() if hasattr(A, "tocsr") else A
    F_ = np.asarray(F, dtype=float).reshape(-1)
    residuals: list[float] = []

    def _cb(rk):
        residuals.append(float(rk))

    x, info = scipy_gmres(
        A_,
        F_,
        restart=int(restart),
        maxiter=int(maxiter),
        atol=atol,
        rtol=rtol,
        callback=_cb,
        callback_type="pr_norm",
    )
    x = np.asarray(x, dtype=float).reshape(-1)
    bnorm = max(float(np.linalg.norm(F_)), 1e-30)
    rrel = float(np.linalg.norm(A_ @ x - F_) / bnorm)
    bad = (info != 0) or (not np.isfinite(x).all()) or (not np.isfinite(rrel)) or (rrel > check_rtol)
    niter = int(len(residuals)) if residuals else 0
    if bad:
        print(
            "[phasefield_square_tension] "
            f"gmres unstable/non-converged: info={info}, relres={rrel:.3e}; "
            f"fallback_to_spsolve={fallback_to_spsolve}"
        )
        if fallback_to_spsolve:
            x = scipy_spsolve(A_, F_)
            return x, KrylovInfo(
                solver="spsolve-phase-fallback",
                niter=1,
                converged=True,
                residual_norm=0.0,
                atol=atol,
                rtol=rtol,
            )
    return x, KrylovInfo(
        solver="gmres-phase",
        niter=max(niter, 1),
        converged=bool(_extract_converged_from_info(info)),
        residual_norm=float(residuals[-1]) if residuals else float("nan"),
        atol=atol,
        rtol=rtol,
    )


@dataclass
class PhaseFieldMaterial:
    E: float = 210
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 0.015
    ft: float = 3.0

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


# Backward-compatible name (was a no-precrack wrapper over SquareTensionCase).
SquareTensionPhaseFieldCase = SquareTensionPreCrackCase


def main() -> None:
    performance_mode = True
    # Default: elastic block = aux-space preconditioned GMRES (same recipe as model0 test).
    # Set FRACTUREX_SQUARE_DIRECT_ELASTIC=1 to use sparse direct elastic solve instead.
    use_direct_elastic = os.environ.get("FRACTUREX_SQUARE_DIRECT_ELASTIC", "0") == "1"
    elastic_formulation = "standard"  # "standard" | "effective_stress"
    eps_g = 1e-6
    d_relaxation = HuZhangPhaseFieldStaggeredDriver._resolve_d_relaxation(None)
    benchmark_assembly = True   # always run serial vs parallel comparison
    benchmark_fast = True       # limit to first few load steps

    mat = PhaseFieldMaterial()
    nx = int(os.environ.get("FRACTUREX_NX", "32"))
    ny = int(os.environ.get("FRACTUREX_NY", str(nx)))

    case = SquareTensionPreCrackCase(
        _model=mat,
        nx=nx,
        ny=ny,
        crack_y=0.5,
        crack_length=0.5,
        debug_mesh=False,
    )

    mesh = case.make_mesh()
    h_max = float(mesh.edge_length().max())
    ncell = int(mesh.number_of_cells())

    print("\n===== square tension y-stretch (Hu-Zhang + phase-field) =====")
    print("mesh: intact from_box (no geometric notch)")
    print(
        f"pre-crack: d=1 on y={case.crack_y}, x in [0, {case.crack_length}] "
        f"(phasefield_initial_damage_data)"
    )
    print("BC: bottom y=0 u=0; top y=1 u_y=load")
    print(f"case                = {case.name}")
    print(f"elastic_linear      = {'direct(spsolve)' if use_direct_elastic else 'gmres+auxspace'}")
    print(f"elastic_formulation = {elastic_formulation}")
    print(f"phase_linear        = gmres (no preconditioner)")
    print(f"damage eps_g        = {eps_g}")
    print(f"d_relaxation        = {d_relaxation}")
    print(f"benchmark_assembly  = {benchmark_assembly}")
    print(f"performance_mode    = {performance_mode}")
    print(f"material l0         = {mat.l0}")
    print(f"mesh nx×ny          = {nx}×{ny}, NC={ncell}, h_max={h_max:.5f} (l0/2={mat.l0/2:.5f})")

    run_label = "direct_elastic" if use_direct_elastic else "auxspace_phase_gmres"
    tag = epsg_tag(eps_g)

    def _tag_dir(*extra: str, mkdir: bool = False) -> str:
        return phasefield_tag_dir(case.name, run_label, *extra, eps_g=eps_g, mkdir=mkdir)

    def _build_driver(assembly_parallel: bool, tag_dir_path: str):
        discr = HuZhangDiscretization(
            case=case,
            p=3,
            damage_p=2,
            use_relaxation=True,
        ).build(mesh=mesh)

        damage = PhaseFieldDamageModel(
            density_type="AT2",
            degradation_type="quadratic",
            split="hybrid",
            eps_g=float(eps_g),
            debug=False,
        )

        elastic_assembler = HuZhangElasticAssembler(
            discr,
            case,
            damage,
            formulation=elastic_formulation,
            assembly_parallel=assembly_parallel,
        )
        phase_assembler = PhaseFieldAssembler(
            discr,
            case,
            damage,
            debug=False,
            assembly_parallel=assembly_parallel,
        )
        recorder = RunRecorder(
            tag_dir_path,
            save_npz=(not performance_mode),
            save_every=10 if performance_mode else 1,
        )

        if use_direct_elastic:
            elastic_solver = HuZhangPhaseFieldStaggeredDriver._default_spsolve
        else:

            def elastic_solver(A, F):
                x, _ = solve_huzhang_block_gmres_auxspace(
                    A,
                    F,
                    gdof_sigma=discr.gdof_sigma,
                    vspace=discr.space_u,
                    atol=1e-12,
                    rtol=1e-8,
                    restart=60,
                    maxit=200,
                    sstep=3,
                    theta=0.25,
                    q=3,
                    schur_rebuild_interval=5,
                    coarse_rebuild_interval=5,
                    weighted_aux=True,
                    elastic_formulation=elastic_formulation,
                    damage=damage,
                    state=discr.state,
                    schur_ilu_in_precond=False,
                )
                return x

        driver = HuZhangPhaseFieldStaggeredDriver(
            case=case,
            discr=discr,
            damage=damage,
            elastic_assembler=elastic_assembler,
            phase_assembler=phase_assembler,
            tol=1e-5,
            maxit=1000,
            d_relaxation=d_relaxation,
            elastic_solver=elastic_solver,
            phase_solver=_phase_gmres,
            compute_linear_residual=True,
            debug=False,
            timing=True,
            recorder=recorder,
            output_dir=tag_dir_path,
            save_vtu_per_step=not performance_mode,
        )
        return driver, discr, damage, elastic_assembler, phase_assembler

    loads = np.concatenate([
        np.linspace(0.0, 5e-3, 51, dtype=float),
        np.linspace(5e-3, 6.1e-3, 111, dtype=float)[1:],
    ]).tolist()
    if os.environ.get("FRACTUREX_RUN_SHORT", "0") == "1":
        loads = loads[:3]
        print(f"FRACTUREX_RUN_SHORT=1: using first {len(loads)} load steps")

    def _install_assembly_timer(assembler):
        stats = {"count": 0, "wall_s": 0.0}
        original = assembler.assemble

        def wrapped(load):
            t1 = time.perf_counter()
            out = original(load)
            stats["wall_s"] += float(time.perf_counter() - t1)
            stats["count"] += 1
            return out

        assembler.assemble = wrapped
        return stats

    run_cfgs = [("main", True)] if not benchmark_assembly else [
        ("serial", False),
        ("parallel", True),
    ]
    if benchmark_assembly and benchmark_fast:
        loads = loads[:3]

    bench_results = []
    for run_name, assembly_parallel in run_cfgs:
        bench_extra = () if not benchmark_assembly else (f"bench_{run_name}",)
        tag_dir = _tag_dir(*bench_extra, mkdir=True)
        print(f"\n===== run mode: {run_name} (assembly_parallel={assembly_parallel}) =====")
        print(f"output: {tag_dir}")
        driver, discr, damage, elastic_assembler, phase_assembler = _build_driver(
            assembly_parallel=assembly_parallel,
            tag_dir_path=tag_dir,
        )
        elastic_stats = _install_assembly_timer(elastic_assembler)
        phase_stats = _install_assembly_timer(phase_assembler)

        t0 = time.perf_counter()
        infos = driver.run(loads)
        total_wall_s = float(time.perf_counter() - t0)

        bench_results.append({
            "mode": run_name,
            "assembly_parallel": bool(assembly_parallel),
            "total_wall_s": float(total_wall_s),
            "elastic_assemble_calls": int(elastic_stats["count"]),
            "elastic_assemble_s": float(elastic_stats["wall_s"]),
            "phase_assemble_calls": int(phase_stats["count"]),
            "phase_assemble_s": float(phase_stats["wall_s"]),
        })

        solver_mode = (
            f"{elastic_formulation}:elastic_direct/phase_gmres"
            if use_direct_elastic
            else f"{elastic_formulation}:elastic_auxspace_gmres_parallel/phase_gmres_no_precond"
        )
        export_paper_summary(
            infos,
            outdir=tag_dir,
            total_wall_s=total_wall_s,
            solver_mode=solver_mode,
            history_source=str(getattr(damage, "history_source", "from_u")),
        )
        export_residual_force_displacement_curve(infos, outdir=tag_dir)

        print("\n===== solve summary =====")
        for info in infos:
            print(
                f"step={info.step:02d}, "
                f"load={info.load:.4e}, "
                f"iters={info.iters:02d}, "
                f"conv={info.converged}, "
                f"err_u={info.err_u:.3e}, "
                f"err_d={info.err_d:.3e}, "
                f"max_d={info.max_d:.3e}"
            )

        state = discr.state
        print("\n===== final state =====")
        print("max(d) =", float(np.max(state.d[:])))
        print("max(H) =", float(np.max(state.H[:])))

    if benchmark_assembly:
        print("\n===== assembly benchmark summary =====")
        for row in bench_results:
            print(
                f"[{row['mode']}] total={row['total_wall_s']:.3f}s, "
                f"elastic={row['elastic_assemble_s']:.3f}s/{row['elastic_assemble_calls']} calls, "
                f"phase={row['phase_assemble_s']:.3f}s/{row['phase_assemble_calls']} calls"
            )


if __name__ == "__main__":
    main()
