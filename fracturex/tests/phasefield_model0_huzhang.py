from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import time
import os
import json

from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.postprocess.recorder import RunRecorder
from fracturex.postprocess.run_report import (
    export_model0_huzhang_test_markdown,
    export_paper_summary,
    export_residual_force_displacement_curve,
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
        Writes result files under `results_model0/...`, including json/csv/md
        reports and optional VTK output.
    """
    performance_mode = True
    elastic_formulation = "standard"  # "standard" | "effective_stress"
    use_direct_solver = True  # 保留开关：True 时 phase 用直接法
    # False: 弹性用 scipy.sparse.linalg.spsolve；True: 弹性用 aux-space GMRES
    release_elastic_iterative_only = False
    use_fast_coarse_test = False  # True 仅用于提速冒烟，不用于结果对比
    eps_g = 1e-6
    # Phase subproblem: GMRES/LGMRES without preconditioner (A x = b).
    # Can also be enabled by env: FRACTUREX_PHASE_GMRES_NOPREC=1
    use_phase_gmres_no_precond = os.environ.get("FRACTUREX_PHASE_GMRES_NOPREC", "0") == "1"

    # Verbose progress inside solve_huzhang_block_gmres_auxspace (see linear_solvers._auxspace_log).
    if release_elastic_iterative_only:
        os.environ["FRACTUREX_AUXSPACE_DEBUG"] = "1"

    mat = Model0Material()

    # Keep the same fine mesh scale as model0_example (--h 0.01). Coarser hmin can
    # introduce stronger mesh bias and trigger earlier left/right asymmetry.
    hmin = 0.01
    case = Model0CircularNotchCase(
        _model=mat,
        hmin=hmin,
        distmesh_maxit=100,
        debug_mesh=False,
    )

    print("\n===== run configuration =====")
    print(f"use_direct_solver   = {use_direct_solver}")
    print(f"elastic_formulation = {elastic_formulation}")
    print(f"release_elastic_iterative_only = {release_elastic_iterative_only}")
    print(f"use_fast_coarse_test= {use_fast_coarse_test}")
    print(f"use_phase_gmres_no_precond = {use_phase_gmres_no_precond}")
    print(f"performance_mode   = {performance_mode}")
    print(f"material l0         = {mat.l0}")
    print(f"mesh hmin           = {hmin}")
    print(f"eps_g               = {eps_g}")

    disable_step_cache = os.environ.get("FRACTUREX_DISABLE_LOADSTEP_CACHE", "0") == "1"
    solver_tag = "direct" if (use_direct_solver and not release_elastic_iterative_only) else "iterative"
    cache_tag = "cache_off" if disable_step_cache else "cache_on"
    outdir = f"results_model0/{elastic_formulation}_{solver_tag}_{cache_tag}"
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
        # Use strain from u (same spirit as standard phase-field / model0_example ψ^+(ε(u))).
        # from_sigma_tilde (C^{-1}σ̃) can mis-match mixed effective-stress variables and
        # under-drive H → little damage / no softening on the reaction curve.
        history_source="from_u",
        eps_g=float(eps_g),
        debug=False,
    )
    print(f"damage eps_g        = {damage.eps_g}")

    elastic_assembler = HuZhangElasticAssembler(
        discr,
        case,
        damage,
        formulation=elastic_formulation,
    )
    phase_assembler = PhaseFieldAssembler(discr, case, damage, debug=False)
    recorder = RunRecorder(
        os.path.join(outdir, tag),
        save_npz=(not performance_mode),
        save_every=10 if performance_mode else 1,
    )

    if use_direct_solver:
        if release_elastic_iterative_only:
            def elastic_solver(A, F):
                """Elastic linear solver callback used by the staggered driver.

                Inputs:
                    A: Global sparse stiffness matrix of mixed HuZhang block system.
                    F: Right-hand-side vector of the elastic subproblem.
                Output:
                    Solution vector x for the elastic unknowns.
                """
                from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_auxspace

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
                    d_fun=discr.state.d,
                    degradation_fun=damage.degradation,
                    # ILU triangular solves inside each preconditioner apply can stall GMRES.
                    # Keep ILU factorization for Schur reuse, but disable in-apply correction by default.
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
            maxit=1000,
            elastic_solver=elastic_solver,
            phase_solver=(
                HuZhangPhaseFieldStaggeredDriver._default_lgmres
                if use_phase_gmres_no_precond
                else HuZhangPhaseFieldStaggeredDriver._default_spsolve
            ),
            compute_linear_residual=True,
            debug=False,
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
            phase_solver=(
                HuZhangPhaseFieldStaggeredDriver._default_lgmres
                if use_phase_gmres_no_precond
                else HuZhangPhaseFieldStaggeredDriver._default_spsolve
            ),
            debug=False,
            timing=True,
            recorder=recorder,
        )

    # Same end displacement as model0_example.default_loads / is_force, but finer steps
    # after ~70 mm so staggered u–d coupling can resolve localized damage + reaction drop.
    loads = np.concatenate(
        [
            np.linspace(0.0, 70e-3, 6, dtype=float),
            np.linspace(70e-3, 125e-3, 26, dtype=float)[1:],
        ]
    ).tolist()
    t0 = time.perf_counter()
    # `infos` stores per-load-step solver history and postprocess metrics.
    infos = driver.run(loads)
    total_wall_s = float(time.perf_counter() - t0)

    export_paper_summary(
        infos,
        outdir=os.path.join(outdir, tag),
        total_wall_s=total_wall_s,
        solver_mode=(
            f"{elastic_formulation}:direct/"
            f"{'phase_lgmres_no_precond' if use_phase_gmres_no_precond else 'phase_spsolve'}"
            if use_direct_solver and not release_elastic_iterative_only
            else (
                f"{elastic_formulation}:mixed/"
                f"{'phase_lgmres_no_precond' if use_phase_gmres_no_precond else 'phase_default'}"
            )
        ),
        history_source=str(damage.history_source),
    )
    export_residual_force_displacement_curve(
        infos,
        outdir=os.path.join(outdir, tag),
    )
    summary_path = os.path.join(outdir, tag, "summary.json")
    meta_path = os.path.join(outdir, tag, "meta.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    export_model0_huzhang_test_markdown(
        outdir=os.path.join(outdir, tag),
        tag=tag,
        solver_mode=summary.get("solver_mode", "unknown"),
        history_source=str(damage.history_source),
        eps_g=float(eps_g),
        hmin=float(hmin),
        summary=summary,
        meta=meta,
    )
    if not performance_mode:
        driver._save_vtkfile(os.path.join(outdir, f"model0_huzhang_final_{tag}.vtu"), cell_mode="mean")


if __name__ == "__main__":
    main()
