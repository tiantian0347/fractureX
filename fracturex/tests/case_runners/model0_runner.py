# fracturex/tests/case_runners/model0_runner.py
"""Lightweight, parametrized Model-0 driver entry for dataset generation.

Why this exists:
- ``fracturex/tests/phasefield_model0_huzhang.py`` is the paper experiment
  entry: ~860 lines, lots of env-var switches, benchmark prints, and
  multi-mode comparisons. Refactoring it to accept a CaseArgs would risk
  breaking ongoing P1/P2 runs.
- For operator-learning dataset generation we only need: take a
  ``CaseArgs``, build (case, discr, damage, assemblers), pick **one**
  elastic+phase solver pair, run loads, return a ``recorder_dir``.

Out of scope here: solver mode comparisons, GMRES tuning logs, mesh
auto-resolution against ``l0/2`` (datasets pin ``hmin`` per sample).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)
from fracturex.postprocess.recorder import RunRecorder
from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_auxspace


@dataclass
class Model0Material:
    E: float = 200.0
    nu: float = 0.2
    Gc: float = 1.0
    l0: float = 0.02

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


@dataclass
class Model0RunArgs:
    """Per-sample knobs for a Model-0 dataset run.

    Geometry (case kwargs):
        circle_r:  notch radius.
        circle_cx, circle_cy: notch center.
        hmin: distmesh target edge length (smaller → finer).

    Material (Lamé via E, nu):
        E, nu, Gc, l0 — see Model0Material.

    Discretization:
        p_sigma: HuZhang stress order.
        damage_p: continuous Lagrange order for d.

    Loading:
        loads: optional explicit load schedule. If None, uses
            ``np.linspace(0, 0.07, 6)`` ∪ ``np.linspace(0.07, 0.125, 26)[1:]``
            (same as model0 paper schedule).

    Solver:
        elastic_mode: ``"direct"`` (sparse direct) or ``"aux"`` (block GMRES
            with auxiliary-space preconditioner). ``"direct"`` is recommended
            for small dataset meshes.
        save_every: checkpoint every N steps (1 = every step).

    Output:
        outdir: recorder output directory (created if missing). Reused as-is
            so the caller controls naming / nesting.
    """

    circle_r: float = 0.2
    circle_cx: float = 0.5
    circle_cy: float = 0.5
    hmin: float = 0.05

    E: float = 200.0
    nu: float = 0.2
    Gc: float = 1.0
    l0: float = 0.02

    p_sigma: int = 3
    damage_p: int = 2
    use_relaxation: bool = True

    loads: Optional[list[float]] = None

    elastic_mode: str = "direct"   # 'direct' | 'aux'
    elastic_atol: float = 1e-12
    elastic_rtol: float = 1e-8
    staggered_tol: float = 1e-5
    staggered_maxit: int = 50
    eps_g: float = 1e-6

    save_every: int = 1

    outdir: Path = field(default_factory=lambda: Path("results/operator_learning_runs"))


def _resolve_loads(args: Model0RunArgs) -> np.ndarray:
    if args.loads is not None:
        return np.asarray(args.loads, dtype=float)
    # Same schedule as the paper experiment. Keep here so dataset runs stay
    # self-contained even if phasefield_model0_huzhang.py changes.
    return np.concatenate(
        [
            np.linspace(0.0, 70e-3, 6, dtype=float),
            np.linspace(70e-3, 125e-3, 26, dtype=float)[1:],
        ]
    )


def run_model0_one(args: Model0RunArgs) -> Path:
    """Run one Model-0 sample end-to-end. Returns the recorder directory."""
    args.outdir.mkdir(parents=True, exist_ok=True)

    mat = Model0Material(E=args.E, nu=args.nu, Gc=args.Gc, l0=args.l0)
    case = Model0CircularNotchCase(
        _model=mat,
        circle_cx=args.circle_cx,
        circle_cy=args.circle_cy,
        circle_r=args.circle_r,
        hmin=args.hmin,
        distmesh_maxit=100,
        debug_mesh=False,
    )

    discr = HuZhangDiscretization(
        case=case,
        p=args.p_sigma,
        damage_p=args.damage_p,
        use_relaxation=args.use_relaxation,
    ).build()

    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=float(args.eps_g),
        debug=False,
    )

    elastic_assembler = HuZhangElasticAssembler(
        discr, case, damage, formulation="standard", assembly_parallel=False
    )
    phase_assembler = PhaseFieldAssembler(
        discr, case, damage, debug=False, assembly_parallel=False
    )

    recorder = RunRecorder(
        str(args.outdir),
        save_npz=True,
        save_every=int(args.save_every),
    )

    if args.elastic_mode == "direct":
        elastic_solver = HuZhangPhaseFieldStaggeredDriver._default_spsolve
    elif args.elastic_mode == "aux":
        def elastic_solver(A, F):
            x, _ = solve_huzhang_block_gmres_auxspace(
                A, F,
                gdof_sigma=discr.gdof_sigma,
                vspace=discr.space_u,
                atol=args.elastic_atol,
                rtol=args.elastic_rtol,
            )
            return x
    else:
        raise ValueError(
            f"unknown elastic_mode {args.elastic_mode!r}; expected 'direct' or 'aux'"
        )

    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case,
        discr=discr,
        damage=damage,
        elastic_assembler=elastic_assembler,
        phase_assembler=phase_assembler,
        tol=args.staggered_tol,
        maxit=args.staggered_maxit,
        elastic_solver=elastic_solver,
        phase_solver=HuZhangPhaseFieldStaggeredDriver._default_lgmres,
        compute_linear_residual=False,
        debug=False,
        timing=False,
        recorder=recorder,
        output_dir=str(args.outdir),
        save_vtu_per_step=False,
    )

    loads = _resolve_loads(args)
    driver.run(loads)
    return args.outdir
