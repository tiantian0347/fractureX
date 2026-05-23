from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import time
import os
import json

from scipy.sparse.linalg import gmres as scipy_gmres
from scipy.sparse.linalg import spsolve as scipy_spsolve

from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.utilfuc.phasefield_mesh import (
    mesh_h_stats,
    phasefield_h_target,
    resolve_model0_distmesh_hmin,
)
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.postprocess.recorder import RunRecorder
from fracturex.postprocess.run_paths import epsg_tag, phasefield_tag_dir, vtk_dir
from fracturex.postprocess.run_report import (
    export_model0_huzhang_test_markdown,
    export_paper_summary,
    export_residual_force_displacement_curve,
)
from fracturex.utilfuc.linear_solvers import (
    KrylovInfo,
    _extract_converged_from_info,
    solve_huzhang_block_gmres_auxspace,
    solve_huzhang_block_gmres_fast,
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
            "[phasefield_model0_huzhang] "
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
class Model0Material:
    """Model-0 phase-field material parameters.

    Inputs:
        E: Young's modulus.
        nu: Poisson ratio.
        Gc: Fracture toughness.
        l0: Regularization length scale.

    Outputs:
        Material object used by case/discretization/assembler builders.
    """

    E: float = 200.0
    nu: float = 0.2
    Gc: float = 1.0
    # Must match fracturex/cases/phase_field/model0_example.py (l0=0.02, not 0.05).
    # Larger l0 + coarse mesh gives diffuse d→1; wrong peak/softening on force curve.
    l0: float = 0.02

    @property
    def mu(self) -> float:
        """Compute shear modulus.

        Input:
            self: Material parameters containing `E` and `nu`.
        Output:
            Shear modulus `mu`.
        """
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        """Compute first Lamé parameter.

        Input:
            self: Material parameters containing `E` and `nu`.
        Output:
            Lamé parameter `lam`.
        """
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def main() -> None:
    """Run HuZhang mixed-element Model-0 phase-field benchmark.

    Inputs:
        None (all run switches are configured inside this function).
    Outputs:
        Writes result files under ``results/phasefield/<case>/...``, including json/csv/md
        reports and optional VTK output.
    """
    performance_mode = True
    elastic_formulation = "standard"  # "standard" | "effective_stress"
    use_direct_solver = True
    # Default production run: elastic aux-space GMRES + phase GMRES(no preconditioner).
    release_elastic_iterative_only = True
    use_fast_coarse_test = False  # True 仅用于提速冒烟，不用于结果对比
    use_elastic_fast = os.environ.get("FRACTUREX_ELASTIC_FAST", "0") == "1"
    eps_g = 1e-6
    benchmark_assembly = os.environ.get("FRACTUREX_ASSEMBLY_BENCHMARK", "0") == "1"
    benchmark_fast = os.environ.get("FRACTUREX_ASSEMBLY_BENCHMARK_FAST", "1") == "1"
    # Set FRACTUREX_COMPARE_ELASTIC=1 for direct vs aux-space GMRES.
    # Set FRACTUREX_COMPARE_ELASTIC_FAST=1 for direct vs integrated fast Schur-GMRES (default hmin=0.02, full loads).
    compare_elastic_direct_vs_aux = os.environ.get("FRACTUREX_COMPARE_ELASTIC", "0") == "1"
    compare_elastic_direct_vs_fast = os.environ.get("FRACTUREX_COMPARE_ELASTIC_FAST", "0") == "1"
    compare_short = os.environ.get("FRACTUREX_COMPARE_SHORT", "0") == "1"
    assembly_parallel_compare = True
    assembly_parallel_single = True

    if compare_elastic_direct_vs_aux and compare_elastic_direct_vs_fast:
        raise SystemExit("Use only one of FRACTUREX_COMPARE_ELASTIC=1 or FRACTUREX_COMPARE_ELASTIC_FAST=1")

    d_relaxation = HuZhangPhaseFieldStaggeredDriver._resolve_d_relaxation(None)

    # Optional verbose progress inside solve_huzhang_block_gmres_auxspace.
    if os.environ.get("FRACTUREX_AUXSPACE_DEBUG", "0") == "1":
        os.environ["FRACTUREX_AUXSPACE_DEBUG"] = "1"

    mat = Model0Material()
    mesh_info: dict = {}

    # Default: auto distmesh hmin so h_max < FRACTUREX_H_SAFETY * l0/2 (see phasefield_mesh.py).
    # Overrides: FRACTUREX_HMIN; compare_elastic_fast uses coarse hmin=0.02 unless HMIN set;
    # FRACTUREX_ELASTIC_FAST + FRACTUREX_FAST_COARSE_MESH uses hmin=0.05 for quick smoke only.
    if os.environ.get("FRACTUREX_HMIN", "").strip():
        hmin = float(os.environ["FRACTUREX_HMIN"])
    elif compare_elastic_direct_vs_fast:
        hmin = 0.02
        mesh_info["mesh_note"] = "compare_fast: coarse hmin; may exceed l0/2"
    elif use_elastic_fast and os.environ.get("FRACTUREX_FAST_COARSE_MESH", "1") == "1":
        hmin = 0.05
        mesh_info["mesh_note"] = "elastic_fast smoke: coarse hmin; may exceed l0/2"
        print(
            f"FRACTUREX_ELASTIC_FAST: coarser mesh hmin={hmin} "
            "(set FRACTUREX_FAST_COARSE_MESH=0 to auto-resolve for h<l0/2)"
        )
    else:
        hmin, mesh_info = resolve_model0_distmesh_hmin(
            mat.l0,
            lambda h: Model0CircularNotchCase(
                _model=mat, hmin=h, distmesh_maxit=100, debug_mesh=False
            ).make_mesh(),
        )

    case = Model0CircularNotchCase(
        _model=mat,
        hmin=hmin,
        distmesh_maxit=100,
        debug_mesh=False,
    )
    session_mesh = case.make_mesh()
    stats = mesh_h_stats(session_mesh)
    mesh_info = {**stats, **mesh_info}
    mesh_info.setdefault("h_target", phasefield_h_target(mat.l0))
    mesh_info.setdefault("l0", float(mat.l0))
    mesh_info["h_ok"] = float(mesh_info["h_max"]) < float(mesh_info["h_target"])
    mesh_info.setdefault("NC", int(session_mesh.number_of_cells()))
    mesh_info.setdefault("NN", int(session_mesh.number_of_nodes()))

    print("\n===== run configuration =====")
    print(f"compare_elastic_direct_vs_aux  = {compare_elastic_direct_vs_aux}")
    print(f"compare_elastic_direct_vs_fast = {compare_elastic_direct_vs_fast}")
    print(f"use_direct_solver   = {use_direct_solver}")
    print(f"elastic_formulation = {elastic_formulation}")
    print(f"release_elastic_iterative_only = {release_elastic_iterative_only}")
    print(f"use_fast_coarse_test= {use_fast_coarse_test}")
    print(f"use_elastic_fast      = {use_elastic_fast}")
    print(f"baseline_reference   = 1 load step timing only (FRACTUREX_SKIP_BASELINE=1 to skip)")
    if compare_elastic_direct_vs_aux or compare_elastic_direct_vs_fast:
        print("phase_solver = gmres (no preconditioner); elastic/phase assembly_parallel = True")
    else:
        elastic_run = "single_fast" if use_elastic_fast else "single_aux"
        print(f"elastic_run         = {elastic_run} (aux-space GMRES vs integrated fast Schur GMRES)")
        print("phase_solver = GMRES(no preconditioner)")
    print(f"benchmark_assembly = {benchmark_assembly}")
    print(f"performance_mode   = {performance_mode}")
    print(f"material l0         = {mat.l0}")
    print(f"mesh hmin (distmesh)= {hmin}")
    print(
        f"mesh h_max          = {mesh_info.get('h_max', float('nan')):.6f}  "
        f"(target < {mesh_info.get('h_target', phasefield_h_target(mat.l0)):.6f}, "
        f"l0/2={mat.l0/2:.6f}, ok={mesh_info.get('h_ok', False)})"
    )
    print(f"mesh NC             = {mesh_info.get('NC', int(session_mesh.number_of_cells()))}")
    if not mesh_info.get("h_ok", True):
        print("WARNING: h_max >= FRACTUREX_H_SAFETY*l0/2; refine mesh or unset coarse/fast overrides.")
    print(f"eps_g               = {eps_g}")
    print(
        f"d_relaxation        = {d_relaxation} "
        "(FRACTUREX_D_RELAXATION; 1=off, try 0.3–0.7 if staggered oscillates)"
    )

    disable_step_cache = os.environ.get("FRACTUREX_DISABLE_LOADSTEP_CACHE", "0") == "1"
    solver_tag = "direct" if (use_direct_solver and not release_elastic_iterative_only) else "iterative"
    cache_tag = "cache_off" if disable_step_cache else "cache_on"
    def _run_path_prefix() -> tuple[str, ...]:
        if compare_elastic_direct_vs_aux:
            return ("compare_elastic", elastic_formulation, cache_tag)
        if compare_elastic_direct_vs_fast:
            return ("compare_elastic_fast", elastic_formulation, cache_tag)
        if use_elastic_fast:
            return (f"{elastic_formulation}_fast_phase_gmres_{cache_tag}",)
        return (f"{elastic_formulation}_auxspace_phase_gmres_{cache_tag}",)

    def _tag_dir(*extra: str, mkdir: bool = False) -> str:
        return phasefield_tag_dir(case.name, *_run_path_prefix(), *extra, eps_g=eps_g, mkdir=mkdir)

    tag = epsg_tag(eps_g)
    print(f"\n===== running case: {tag} -> results/phasefield/{case.name}/.../{tag} =====")

    def _build_driver(
        assembly_parallel: bool,
        tag_dir: str,
        *,
        mesh=None,
        elastic_mode: str = "legacy",
        d_relaxation: float = d_relaxation,
        save_vtu_per_step: bool = False,
    ):
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
        print(f"damage eps_g        = {damage.eps_g}")

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
            tag_dir,
            save_npz=(not performance_mode),
            save_every=10 if performance_mode else 1,
        )

        phase_solver = (
            _phase_gmres
            if elastic_mode in ("compare_spsolve", "compare_aux", "single_aux", "single_fast")
            else HuZhangPhaseFieldStaggeredDriver._default_lgmres
        )

        if elastic_mode == "compare_spsolve":
            driver = HuZhangPhaseFieldStaggeredDriver(
                case=case,
                discr=discr,
                damage=damage,
                elastic_assembler=elastic_assembler,
                phase_assembler=phase_assembler,
                tol=1e-5,
                maxit=50,
                d_relaxation=d_relaxation,
                elastic_solver=HuZhangPhaseFieldStaggeredDriver._default_spsolve,
                phase_solver=phase_solver,
                compute_linear_residual=True,
                debug=False,
                timing=True,
                recorder=recorder,
                output_dir=tag_dir,
                save_vtu_per_step=save_vtu_per_step,
            )
        elif elastic_mode in ("compare_aux", "single_aux"):

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
                maxit=50,
                d_relaxation=d_relaxation,
                elastic_solver=elastic_solver,
                phase_solver=phase_solver,
                compute_linear_residual=True,
                debug=False,
                timing=True,
                recorder=recorder,
                output_dir=tag_dir,
                save_vtu_per_step=save_vtu_per_step,
            )
        elif elastic_mode == "single_fast":

            def elastic_solver(A, F):
                x, _ = solve_huzhang_block_gmres_fast(
                    A,
                    F,
                    gdof_sigma=discr.gdof_sigma,
                    vspace=discr.space_u,
                    atol=1e-12,
                    rtol=1e-8,
                    restart=60,
                    maxit=500,
                    q=3,
                    cheb_degree=2,
                    elastic_formulation=elastic_formulation,
                    weighted_aux=True,
                    damage=damage,
                    state=discr.state,
                )
                return x

            driver = HuZhangPhaseFieldStaggeredDriver(
                case=case,
                discr=discr,
                damage=damage,
                elastic_assembler=elastic_assembler,
                phase_assembler=phase_assembler,
                tol=1e-5,
                maxit=50,
                d_relaxation=d_relaxation,
                elastic_solver=elastic_solver,
                phase_solver=phase_solver,
                compute_linear_residual=True,
                debug=False,
                timing=True,
                recorder=recorder,
                output_dir=tag_dir,
                save_vtu_per_step=save_vtu_per_step,
            )
        elif use_direct_solver:
            if release_elastic_iterative_only:
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
            else:
                elastic_solver = HuZhangPhaseFieldStaggeredDriver._default_spsolve
            driver = HuZhangPhaseFieldStaggeredDriver(
                case=case,
                discr=discr,
                damage=damage,
                elastic_assembler=elastic_assembler,
                phase_assembler=phase_assembler,
                tol=1e-5,
                maxit=50,
                d_relaxation=d_relaxation,
                elastic_solver=elastic_solver,
                phase_solver=phase_solver,
                compute_linear_residual=True,
                debug=False,
                timing=True,
                recorder=recorder,
                output_dir=tag_dir,
                save_vtu_per_step=save_vtu_per_step,
            )
        else:
            driver = HuZhangPhaseFieldStaggeredDriver(
                case=case,
                discr=discr,
                damage=damage,
                elastic_assembler=elastic_assembler,
                phase_assembler=phase_assembler,
                tol=1e-5,
                maxit=50,
                d_relaxation=d_relaxation,
                phase_solver=phase_solver,
                debug=False,
                timing=True,
                recorder=recorder,
                output_dir=tag_dir,
                save_vtu_per_step=save_vtu_per_step,
            )
        return driver, discr, damage, elastic_assembler, phase_assembler

    # Same end displacement as model0_example.default_loads / is_force, but finer steps
    # after ~70 mm so staggered u–d coupling can resolve localized damage + reaction drop.
    loads = np.concatenate(
        [
            np.linspace(0.0, 70e-3, 6, dtype=float),
            np.linspace(70e-3, 125e-3, 26, dtype=float)[1:],
        ]
    ).tolist()

    # Apply load-list truncation once so baseline + main runs use the same steps.
    if (compare_elastic_direct_vs_aux or compare_elastic_direct_vs_fast) and compare_short:
        loads = loads[:3]
        print(f"FRACTUREX_COMPARE_SHORT=1: using first {len(loads)} load steps only")
    elif not (compare_elastic_direct_vs_aux or compare_elastic_direct_vs_fast):
        if os.environ.get("FRACTUREX_RUN_SHORT", "0") == "1" and not benchmark_assembly:
            loads = loads[:3]
            print(f"FRACTUREX_RUN_SHORT=1: using first {len(loads)} load steps only")
        if benchmark_assembly and benchmark_fast:
            loads = loads[:3]

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

    def _reaction_series(infos):
        return np.array([float(i.meta.get("residual_force", 0.0)) for i in infos], dtype=float)

    def _state_snapshot(discr):
        return {
            "d": np.asarray(discr.state.d[:], dtype=float).copy(),
            "u": np.asarray(discr.state.u[:], dtype=float).copy(),
        }

    def _baseline_single_load(loads_full: list) -> list:
        """One load step for serial/direct timing (not the full load history)."""
        if not loads_full:
            return [0.0]
        raw = os.environ.get("FRACTUREX_BASELINE_LOAD_INDEX", "").strip()
        if raw:
            k = max(0, min(int(raw), len(loads_full) - 1))
        else:
            # Default: first nontrivial step (skip pure zero if present).
            k = 1 if len(loads_full) > 1 and float(loads_full[0]) == 0.0 else 0
        return [float(loads_full[k])]

    bench_results = []

    baseline_tag_dir: Optional[str] = None
    if os.environ.get("FRACTUREX_SKIP_BASELINE", "0") != "1":
        baseline_tag_dir = phasefield_tag_dir(
            case.name,
            "reference_direct_serial",
            f"{elastic_formulation}_{cache_tag}",
            eps_g=eps_g,
            mkdir=True,
        )
        baseline_loads = _baseline_single_load(loads)
        print("\n===== baseline timing: elastic spsolve + serial assembly (ONE load step only) =====")
        print(
            f"load_value={baseline_loads[0]:.6e}  "
            f"(override index: FRACTUREX_BASELINE_LOAD_INDEX; full run uses {len(loads)} steps)"
        )
        print(f"output: {baseline_tag_dir}  (set FRACTUREX_SKIP_BASELINE=1 to skip)")
        driver_b, discr_b, damage_b, ea_b, pa_b = _build_driver(
            False,
            baseline_tag_dir,
            mesh=session_mesh,
            elastic_mode="compare_spsolve",
            save_vtu_per_step=True,
        )
        t0b = time.perf_counter()
        infos_b = driver_b.run(baseline_loads)
        wall_b = float(time.perf_counter() - t0b)
        snap_b = _state_snapshot(discr_b)
        R_b = _reaction_series(infos_b)
        ref_json = {
            "description": "Single load-step wall time: Hu-Zhang elastic scipy.sparse.linalg.spsolve; assembly_parallel=False",
            "hmin": float(hmin),
            "eps_g": float(eps_g),
            "elastic_formulation": elastic_formulation,
            "n_load_steps_baseline": 1,
            "baseline_load_value": float(baseline_loads[0]),
            "full_run_n_load_steps": len(loads),
            "wall_s_total_baseline_run": wall_b,
            "reaction": R_b.tolist(),
            "final_max_d": float(np.max(snap_b["d"])),
            "dvec": snap_b["d"].tolist(),
            "uvec": snap_b["u"].tolist(),
        }
        with open(
            os.path.join(os.path.dirname(baseline_tag_dir), "baseline_reference.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ref_json, f, indent=2)
        export_paper_summary(
            infos_b,
            outdir=baseline_tag_dir,
            total_wall_s=wall_b,
            solver_mode=f"{elastic_formulation}:baseline_elastic_spsolve_serial_asm_1step/phase_gmres_parallel",
            history_source=str(getattr(damage_b, "history_source", "from_u")),
        )
        export_residual_force_displacement_curve(
            infos_b,
            outdir=baseline_tag_dir,
        )
        summary_path_b = os.path.join(baseline_tag_dir, "summary.json")
        meta_path_b = os.path.join(baseline_tag_dir, "meta.json")
        if os.path.isfile(summary_path_b) and os.path.isfile(meta_path_b):
            with open(summary_path_b, "r", encoding="utf-8") as f:
                summary_b = json.load(f)
            with open(meta_path_b, "r", encoding="utf-8") as f:
                meta_b = json.load(f)
            export_model0_huzhang_test_markdown(
                outdir=baseline_tag_dir,
                tag=tag,
                solver_mode=summary_b.get("solver_mode", "unknown"),
                history_source=str(getattr(damage_b, "history_source", "from_u")),
                eps_g=float(eps_g),
                hmin=float(hmin),
                summary=summary_b,
                meta=meta_b,
            )
        print(f"Baseline wall time (1 load step): {wall_b:.3f}s")

    if compare_elastic_direct_vs_aux or compare_elastic_direct_vs_fast:
        mesh = session_mesh
        if compare_elastic_direct_vs_aux:
            compare_pairs = (
                ("elastic_spsolve", "compare_spsolve", "direct_elastic"),
                ("elastic_aux_gmres", "compare_aux", "aux_precond_elastic"),
            )
            compare_kind = "direct_vs_aux"
            cmp_title = "elastic direct (spsolve) vs aux-space GMRES"
        else:
            compare_pairs = (
                ("elastic_spsolve", "compare_spsolve", "direct_elastic"),
                ("elastic_fast_gmres", "single_fast", "fast_precond_elastic"),
            )
            compare_kind = "direct_vs_fast"
            cmp_title = "elastic direct (spsolve) vs integrated fast Schur-GMRES"

        rows = []
        for run_label, elastic_mode, subdir in compare_pairs:
            tag_dir = phasefield_tag_dir(
                case.name, *_run_path_prefix(), subdir, eps_g=eps_g, mkdir=True
            )
            print(f"\n===== compare branch: {run_label} (elastic_mode={elastic_mode}) =====")
            print(f"output: {tag_dir}")
            driver, discr, damage, elastic_assembler, phase_assembler = _build_driver(
                assembly_parallel_compare,
                tag_dir,
                mesh=mesh,
                elastic_mode=elastic_mode,
            )
            elastic_stats = _install_assembly_timer(elastic_assembler)
            phase_stats = _install_assembly_timer(phase_assembler)

            t0 = time.perf_counter()
            infos = driver.run(loads)
            total_wall_s = float(time.perf_counter() - t0)
            snap = _state_snapshot(discr)
            R = _reaction_series(infos)
            rows.append(
                {
                    "label": run_label,
                    "elastic_mode": elastic_mode,
                    "total_wall_s": total_wall_s,
                    "reaction": R.tolist(),
                    "final_max_d": float(np.max(snap["d"])),
                    "dvec": snap["d"].tolist(),
                    "uvec": snap["u"].tolist(),
                }
            )
            bench_results.append(
                {
                    "mode": run_label,
                    "assembly_parallel": True,
                    "total_wall_s": total_wall_s,
                    "elastic_assemble_calls": int(elastic_stats["count"]),
                    "elastic_assemble_s": float(elastic_stats["wall_s"]),
                    "phase_assemble_calls": int(phase_stats["count"]),
                    "phase_assemble_s": float(phase_stats["wall_s"]),
                }
            )

            solver_mode = (
                f"{elastic_formulation}:elastic_spsolve/phase_gmres_parallel"
                if elastic_mode == "compare_spsolve"
                else (
                    f"{elastic_formulation}:elastic_aux_gmres/phase_gmres_parallel"
                    if compare_elastic_direct_vs_aux
                    else f"{elastic_formulation}:elastic_fast_schur_gmres/phase_gmres_parallel"
                )
            )
            export_paper_summary(
                infos,
                outdir=tag_dir,
                total_wall_s=total_wall_s,
                solver_mode=solver_mode,
                history_source=str(getattr(damage, "history_source", "from_u")),
            )
            export_residual_force_displacement_curve(
                infos,
                outdir=tag_dir,
            )
            summary_path = os.path.join(tag_dir, "summary.json")
            meta_path = os.path.join(tag_dir, "meta.json")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            export_model0_huzhang_test_markdown(
                outdir=tag_dir,
                tag=tag,
                solver_mode=summary.get("solver_mode", "unknown"),
                history_source=str(getattr(damage, "history_source", "from_u")),
                eps_g=float(eps_g),
                hmin=float(hmin),
                summary=summary,
                meta=meta,
            )

        d0 = np.asarray(rows[0]["dvec"], dtype=float)
        d1 = np.asarray(rows[1]["dvec"], dtype=float)
        u0 = np.asarray(rows[0]["uvec"], dtype=float)
        u1 = np.asarray(rows[1]["uvec"], dtype=float)
        R0 = np.asarray(rows[0]["reaction"], dtype=float)
        R1 = np.asarray(rows[1]["reaction"], dtype=float)
        nd = float(np.linalg.norm(d0 - d1))
        nu = float(np.linalg.norm(u0 - u1))
        nR = float(np.linalg.norm(R0 - R1))
        rd = nd / max(float(np.linalg.norm(d0)), 1e-30)
        ru = nu / max(float(np.linalg.norm(u0)), 1e-30)
        rR = nR / max(float(np.linalg.norm(R0)), 1e-30)
        step_rel = (
            float(np.max(np.abs(R0 - R1) / np.maximum(np.abs(R0), 1e-30)))
            if R0.size
            else float("nan")
        )

        print(f"\n===== {cmp_title} =====")
        print(f"(same mesh, hmin={hmin}, n_load_steps={len(loads)}, phase GMRES, parallel assembly)")
        if compare_elastic_direct_vs_aux:
            print(f"||d_direct - d_aux||_2 = {nd:.6e}  (relative {rd:.6e})")
            print(f"||u_direct - u_aux||_2 = {nu:.6e}  (relative {ru:.6e})")
            print(f"||R_direct - R_aux||_2 = {nR:.6e}  (relative {rR:.6e})")
        else:
            print(f"||d_direct - d_fast||_2 = {nd:.6e}  (relative {rd:.6e})")
            print(f"||u_direct - u_fast||_2 = {nu:.6e}  (relative {ru:.6e})")
            print(f"||R_direct - R_fast||_2 = {nR:.6e}  (relative {rR:.6e})")
        print(f"max_k |R0[k]-R1[k]|/|R0[k]| = {step_rel:.6e}")

        cmp_tag_dir = _tag_dir(mkdir=True)
        cmp_path = os.path.join(cmp_tag_dir, "elastic_solver_comparison.json")
        with open(cmp_path, "w", encoding="utf-8") as f:
            cmp_payload = {
                "compare_mode": compare_kind,
                "hmin": float(hmin),
                "mesh_note": "single distmesh instance passed to both runs",
                "phase_solver": "gmres",
                "assembly_parallel": True,
                "n_load_steps": len(loads),
                "norm_d_diff": nd,
                "rel_d_diff": rd,
                "norm_u_diff": nu,
                "rel_u_diff": ru,
                "norm_R_diff": nR,
                "rel_R_diff": rR,
                "max_relative_reaction_step": step_rel,
            }
            if baseline_tag_dir is not None:
                cmp_payload["baseline_reference_dir"] = baseline_tag_dir
            json.dump(cmp_payload, f, indent=2)
        print(f"Wrote comparison metrics: {cmp_path}")

    else:
        run_cfgs = (
            [("fast_phase_gmres", assembly_parallel_single, "single_fast")]
            if use_elastic_fast
            else [("auxspace_phase_gmres", assembly_parallel_single, "single_aux")]
        ) if not benchmark_assembly else [
            ("serial", False, "single_fast" if use_elastic_fast else "single_aux"),
            ("parallel", True, "single_fast" if use_elastic_fast else "single_aux"),
        ]

        for run_name, assembly_parallel, elastic_mode in run_cfgs:
            bench_extra = () if not benchmark_assembly else (f"bench_{run_name}",)
            tag_dir = _tag_dir(*bench_extra, mkdir=True)
            print(
                f"\n===== run mode: {run_name} "
                f"(elastic_mode={elastic_mode}, assembly_parallel={assembly_parallel}) ====="
            )
            print(f"output: {tag_dir}")
            driver, discr, damage, elastic_assembler, phase_assembler = _build_driver(
                assembly_parallel=assembly_parallel,
                tag_dir=tag_dir,
                mesh=session_mesh,
                elastic_mode=elastic_mode,
                save_vtu_per_step=not performance_mode,
            )
            elastic_stats = _install_assembly_timer(elastic_assembler)
            phase_stats = _install_assembly_timer(phase_assembler)

            t0 = time.perf_counter()
            infos = driver.run(loads)
            total_wall_s = float(time.perf_counter() - t0)
            bench_results.append(
                {
                    "mode": run_name,
                    "assembly_parallel": bool(assembly_parallel),
                    "total_wall_s": float(total_wall_s),
                    "elastic_assemble_calls": int(elastic_stats["count"]),
                    "elastic_assemble_s": float(elastic_stats["wall_s"]),
                    "phase_assemble_calls": int(phase_stats["count"]),
                    "phase_assemble_s": float(phase_stats["wall_s"]),
                }
            )

            export_paper_summary(
                infos,
                outdir=tag_dir,
                total_wall_s=total_wall_s,
                solver_mode=f"{elastic_formulation}:elastic_{elastic_mode}/phase_gmres_no_precond",
                history_source=str(getattr(damage, "history_source", "from_u")),
            )
            export_residual_force_displacement_curve(
                infos,
                outdir=tag_dir,
            )
            summary_path = os.path.join(tag_dir, "summary.json")
            meta_path = os.path.join(tag_dir, "meta.json")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            export_model0_huzhang_test_markdown(
                outdir=tag_dir,
                tag=tag,
                solver_mode=summary.get("solver_mode", "unknown"),
                history_source=str(getattr(damage, "history_source", "from_u")),
                eps_g=float(eps_g),
                hmin=float(hmin),
                summary=summary,
                meta=meta,
            )
            if not performance_mode and infos:
                os.makedirs(vtk_dir(tag_dir), exist_ok=True)
                driver._save_vtkfile(
                    os.path.join(vtk_dir(tag_dir), "final.vtu"),
                    cell_mode="mean",
                )

    if benchmark_assembly and not compare_elastic_direct_vs_aux and not compare_elastic_direct_vs_fast:
        print("\n===== assembly benchmark summary =====")
        for row in bench_results:
            print(
                f"[{row['mode']}] total={row['total_wall_s']:.3f}s, "
                f"elastic={row['elastic_assemble_s']:.3f}s/{row['elastic_assemble_calls']} calls, "
                f"phase={row['phase_assemble_s']:.3f}s/{row['phase_assemble_calls']} calls"
            )


if __name__ == "__main__":
    main()
