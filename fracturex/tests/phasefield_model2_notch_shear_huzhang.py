"""
Hu-Zhang + staggered phase-field test for paper model-2 (top x-stretch).

Intact unit-square mesh + phase-field pre-crack (d=1 on y=0.5, x∈[0,0.5]);
no ``from_square_domain_with_fracture`` (Hu-Zhang corner issues on cut meshes).

Reference: adaptive_paper.tex (979–982);
  bottom y=0: u_x=u_y=0; top y=1: u_x ramp, u_y=0.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np

from scipy.sparse.linalg import gmres as scipy_gmres
from scipy.sparse.linalg import spsolve as scipy_spsolve

from fracturex.cases.model2_notch_shear import Model2NotchXStretchCase
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
    if hasattr(A, "to_scipy"):
        a_ = A.to_scipy().tocsr()
    else:
        a_ = A.tocsr() if hasattr(A, "tocsr") else A
    f_ = np.asarray(F, dtype=float).reshape(-1)
    residuals: list[float] = []

    def _cb(rk):
        residuals.append(float(rk))

    x, info = scipy_gmres(
        a_,
        f_,
        restart=int(restart),
        maxiter=int(maxiter),
        atol=atol,
        rtol=rtol,
        callback=_cb,
        callback_type="pr_norm",
    )
    x = np.asarray(x, dtype=float).reshape(-1)
    bnorm = max(float(np.linalg.norm(f_)), 1e-30)
    rrel = float(np.linalg.norm(a_ @ x - f_) / bnorm)
    bad = (info != 0) or (not np.isfinite(x).all()) or (not np.isfinite(rrel)) or (rrel > check_rtol)
    niter = int(len(residuals)) if residuals else 0
    if bad:
        print(
            "[phasefield_model2_notch_shear] "
            f"gmres unstable/non-converged: info={info}, relres={rrel:.3e}; "
            f"fallback_to_spsolve={fallback_to_spsolve}"
        )
        if fallback_to_spsolve:
            x = scipy_spsolve(a_, f_)
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
class Model2Material:
    """Material parameters for paper model-1/2 (mm, kN, GPa-equivalent)."""

    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 1.33e-2

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _paper_loads(case: Model2NotchXStretchCase) -> list[float]:
    return np.asarray(case.default_loads(), dtype=float).tolist()


def main() -> None:
    performance_mode = True
    use_direct_elastic = os.environ.get("FRACTUREX_MODEL2_DIRECT_ELASTIC", "0") == "1"
    elastic_formulation = "standard"
    eps_g = 1e-6
    d_relaxation = HuZhangPhaseFieldStaggeredDriver._resolve_d_relaxation(None)

    mat = Model2Material()
    nx = int(os.environ.get("FRACTUREX_NX", "32"))
    ny = int(os.environ.get("FRACTUREX_NY", str(nx)))
    case = Model2NotchXStretchCase(_model=mat, nx=nx, ny=ny, debug_mesh=False)

    mesh = case.make_mesh()
    h_max = float(mesh.edge_length().max())
    ncell = int(mesh.number_of_cells())

    print("\n===== model-2 notch x-stretch (Hu-Zhang + phase-field) =====")
    print("mesh: intact from_box (no geometric notch)")
    print(
        f"pre-crack: d=1 on y={case.crack_y}, x in [0, {case.crack_length}] "
        f"(phasefield_initial_damage_data)"
    )
    print("BC: bottom y=0 u_x=u_y=0; top y=1 u_x=load (x tension), u_y=0")
    print(f"case                = {case.name}")
    print(f"elastic_linear      = {'direct(spsolve)' if use_direct_elastic else 'gmres+auxspace'}")
    print(f"elastic_formulation = {elastic_formulation}")
    print(f"phase_linear        = gmres (no preconditioner)")
    print(f"material E,nu,Gc,l0 = {mat.E}, {mat.nu}, {mat.Gc}, {mat.l0}")
    print(f"mesh nx×ny          = {nx}×{ny}, NC={ncell}, h_max={h_max:.5f} (l0/2={mat.l0/2:.5f})")
    print(f"load: du_x={case.du_x}, steps={case.n_load_steps}, u_x_tot={case.u_x_total}")
    print(f"d_relaxation        = {d_relaxation}")
    print(f"eps_g               = {eps_g}")

    loads = _paper_loads(case)
    if os.environ.get("FRACTUREX_RUN_SHORT", "0") == "1":
        loads = loads[:3]
        print(f"FRACTUREX_RUN_SHORT=1: using first {len(loads)} load steps")
    elif raw := os.environ.get("FRACTUREX_RUN_NSTEPS", "").strip():
        n = max(1, int(raw))
        loads = loads[: n + 1] if loads and float(loads[0]) == 0.0 else loads[:n]
        print(f"FRACTUREX_RUN_NSTEPS={n}: using {len(loads)} load values")

    run_label = (
        "direct_elastic"
        if use_direct_elastic
        else "auxspace_phase_gmres"
    )
    tag_dir = phasefield_tag_dir(case.name, run_label, eps_g=eps_g, mkdir=True)
    tag = epsg_tag(eps_g)

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
            maxit=50,
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
        return driver, discr, damage

    print(f"\n===== running {len(loads)} load steps -> {tag_dir} =====")
    driver, discr, damage = _build_driver(assembly_parallel=True, tag_dir_path=tag_dir)

    t0 = time.perf_counter()
    infos = driver.run(loads)
    total_wall_s = float(time.perf_counter() - t0)

    export_paper_summary(
        infos,
        outdir=tag_dir,
        total_wall_s=total_wall_s,
        solver_mode=(
            f"{elastic_formulation}:elastic_"
            f"{'direct' if use_direct_elastic else 'auxspace'}/phase_gmres"
        ),
        history_source=str(getattr(damage, "history_source", "from_u")),
    )
    export_residual_force_displacement_curve(infos, outdir=tag_dir)

    print("\n===== solve summary (last 5 steps) =====")
    for info in infos[-5:]:
        print(
            f"step={info.step:04d}, load(ux)={info.load:.6e}, "
            f"iters={info.iters:02d}, conv={info.converged}, "
            f"R={info.meta.get('residual_force', 0.0):.6e}, max_d={info.max_d:.4f}"
        )
    print(f"total wall time = {total_wall_s:.3f} s")
    print(f"final max(d)    = {float(np.max(discr.state.d[:])):.6f}")
    print(f"outputs         = {tag_dir}")


if __name__ == "__main__":
    main()
