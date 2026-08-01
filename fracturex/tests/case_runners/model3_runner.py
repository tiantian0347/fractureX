"""Lightweight Model-3 (L-shape) Hu–Zhang runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.cases.model3_lshape import Model3LShapeCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)
from fracturex.postprocess.recorder import RunRecorder
from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_auxspace


@dataclass
class Model3Material:
    lam: float = 6.16
    mu: float = 10.95
    Gc: float = 8.9e-5
    l0: float = 1.1875
    ft: float = 3.0

    @property
    def E(self) -> float:
        return self.mu * (3.0 * self.lam + 2.0 * self.mu) / (self.lam + self.mu)

    @property
    def nu(self) -> float:
        return self.lam / (2.0 * (self.lam + self.mu))


@dataclass
class Model3RunArgs:
    nx: int = 50
    ny: int = 50
    load_half_width: float = 2.0

    lam: float = 6.16
    mu: float = 10.95
    Gc: float = 8.9e-5
    l0: float = 1.1875

    p_sigma: int = 3
    damage_p: int = 2
    use_relaxation: bool = True  # re-entrant corner needs this for Hu–Zhang

    loads: Optional[list[float]] = None
    # If None: Ambati cyclic 0→0.3→−0.2→1.0; short smoke uses first few.

    elastic_mode: str = "direct"
    elastic_atol: float = 1e-12
    elastic_rtol: float = 1e-8
    staggered_tol: float = 1e-5
    staggered_maxit: int = 50
    eps_g: float = 1e-6

    save_every: int = 100
    save_quadrature_fields: bool = False
    outdir: Path = field(
        default_factory=lambda: Path("results/operator_learning_runs")
    )


def _resolve_loads(args: Model3RunArgs, case: Model3LShapeCase) -> np.ndarray:
    if args.loads is not None:
        return np.asarray(args.loads, dtype=float)
    return np.asarray(case.default_loads(), dtype=float)


def run_model3_one(args: Model3RunArgs) -> Path:
    args.outdir.mkdir(parents=True, exist_ok=True)
    mat = Model3Material(lam=args.lam, mu=args.mu, Gc=args.Gc, l0=args.l0)
    case = Model3LShapeCase(
        _model=mat,
        nx=args.nx,
        ny=args.ny,
        load_half_width=args.load_half_width,
        lam=args.lam,
        mu=args.mu,
        Gc=args.Gc,
        l0=args.l0,
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
        discr, case, damage, formulation="standard"
    )
    phase_assembler = PhaseFieldAssembler(discr, case, damage, debug=False)
    recorder = RunRecorder(
        str(args.outdir),
        save_npz=True,
        save_every=int(args.save_every),
        save_quadrature_fields=bool(args.save_quadrature_fields),
    )
    if args.elastic_mode == "direct":
        elastic_solver = HuZhangPhaseFieldStaggeredDriver._default_spsolve
    elif args.elastic_mode == "aux":
        def elastic_solver(A, F):
            x, _ = solve_huzhang_block_gmres_auxspace(
                A,
                F,
                gdof_sigma=discr.gdof_sigma,
                vspace=discr.space_u,
                atol=args.elastic_atol,
                rtol=args.elastic_rtol,
            )
            return x
    else:
        raise ValueError(f"unknown elastic_mode {args.elastic_mode!r}")

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
        save_vtu_per_step=True,
    )
    driver.run(_resolve_loads(args, case))
    return args.outdir
