"""Lightweight Model-5 (three-point bending) Hu–Zhang runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.cases.model5_three_point_bending import Model5ThreePointBendingCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)
from fracturex.postprocess.recorder import RunRecorder
from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_auxspace


@dataclass
class Model5Material:
    lam: float = 12.0
    mu: float = 8.0
    Gc: float = 5.4e-4
    l0: float = 0.03
    ft: float = 3.0

    @property
    def E(self) -> float:
        return self.mu * (3.0 * self.lam + 2.0 * self.mu) / (self.lam + self.mu)

    @property
    def nu(self) -> float:
        return self.lam / (2.0 * (self.lam + self.mu))


@dataclass
class Model5RunArgs:
    mesh_size: float = 0.1
    length: float = 8.0
    height: float = 2.0
    notch_depth: float = 0.4
    with_geometric_notch: bool = False  # Hu–Zhang: d=1 pre-crack

    lam: float = 12.0
    mu: float = 8.0
    Gc: float = 5.4e-4
    l0: float = 0.03

    p_sigma: int = 3
    damage_p: int = 2
    use_relaxation: bool = True

    loads: Optional[list[float]] = None
    u_y_total: float = 0.1
    n_load_steps: int = 100

    elastic_mode: str = "direct"
    elastic_atol: float = 1e-12
    elastic_rtol: float = 1e-8
    staggered_tol: float = 1e-5
    staggered_maxit: int = 50
    eps_g: float = 1e-6

    save_every: int = 50
    save_quadrature_fields: bool = False
    outdir: Path = field(
        default_factory=lambda: Path("results/operator_learning_runs")
    )


def _resolve_loads(args: Model5RunArgs) -> np.ndarray:
    if args.loads is not None:
        return np.asarray(args.loads, dtype=float)
    n = int(args.n_load_steps)
    return np.linspace(0.0, float(args.u_y_total), n + 1, dtype=float)


def run_model5_one(args: Model5RunArgs) -> Path:
    args.outdir.mkdir(parents=True, exist_ok=True)
    mat = Model5Material(lam=args.lam, mu=args.mu, Gc=args.Gc, l0=args.l0)
    case = Model5ThreePointBendingCase(
        _model=mat,
        mesh_size=args.mesh_size,
        length=args.length,
        height=args.height,
        notch_depth=args.notch_depth,
        with_geometric_notch=bool(args.with_geometric_notch),
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
        save_vtu_per_step=False,
    )
    driver.run(_resolve_loads(args))
    return args.outdir
