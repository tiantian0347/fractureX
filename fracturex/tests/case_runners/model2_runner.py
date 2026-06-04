"""Lightweight, parametrized Model-2 driver entry for dataset generation.

Mirror of [model0_runner.py](model0_runner.py) for the Model-2 notch-shear
case. Geometry: unit square pre-cracked via ``d = 1`` on a horizontal
segment (no geometric notch); load: x-stretch on top edge with bottom
fully fixed. See [model2_notch_shear.py](../../cases/model2_notch_shear.py)
and [phasefield_model2_notch_shear_huzhang.py](../phasefield_model2_notch_shear_huzhang.py)
for the paper experiment counterpart.

Why a separate runner: ``phasefield_model2_notch_shear_huzhang.py`` is the
paper experiment entry (same env-var coupling story as model0); refactoring
it to take CaseArgs would risk breaking traditional-method runs. For
operator-learning dataset generation this runner only needs Model2RunArgs
→ build (case, discr, damage, assemblers) → one fixed solver pair → run
loads → recorder dir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.cases.model2_notch_shear import Model2NotchXStretchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)
from fracturex.postprocess.recorder import RunRecorder
from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_auxspace


@dataclass
class Model2Material:
    """Defaults match scripts/paper_huzhang/run_case.py:Model2Material."""

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


@dataclass
class Model2RunArgs:
    """Per-sample knobs for a Model-2 dataset run.

    Geometry (pre-crack via d=1, no geometric notch):
        crack_y, crack_length: horizontal pre-crack at y=crack_y, x in [0, crack_length].
        nx, ny: TriangleMesh.from_box(...) resolution; NC = 2*nx*ny.

    Material (Lamé via E, nu):
        E, nu, Gc, l0 — see Model2Material.

    Discretization:
        p_sigma: HuZhang stress order (paper default 3).
        damage_p: continuous Lagrange order for d (paper default 2).

    Loading:
        loads: optional explicit load schedule. If None, uses the paper
            schedule ``np.linspace(0, u_x_total, n_load_steps + 1)`` —
            same as ``Model2NotchXStretchCase.default_loads()`` /
            ``phasefield_model2_notch_shear_huzhang.py``.

    Solver:
        elastic_mode: 'direct' (sparse direct) or 'aux' (block GMRES with
            aux-space preconditioner). Paper model-2 traditionally uses
            'direct' (see paper_direct/ runs).
        save_every: checkpoint every N steps; default 100 to mirror the
            existing 1700-step paper run that produced ~17 checkpoints.
        save_quadrature_fields: dump H_qp + xq for operator-learning
            history-channel data (recorder D-B path; off by default).

    Output:
        outdir: recorder output directory (created if missing).
    """

    crack_y: float = 0.5
    crack_length: float = 0.5
    nx: int = 32
    ny: int = 32

    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 1.33e-2

    p_sigma: int = 3
    damage_p: int = 2
    use_relaxation: bool = True

    n_load_steps: int = 1700
    u_x_total: float = 1.7e-2
    loads: Optional[list[float]] = None

    elastic_mode: str = "direct"   # 'direct' | 'aux'
    elastic_atol: float = 1e-12
    elastic_rtol: float = 1e-8
    staggered_tol: float = 1e-5
    staggered_maxit: int = 50
    eps_g: float = 1e-6

    save_every: int = 100
    save_quadrature_fields: bool = False

    outdir: Path = field(default_factory=lambda: Path("results/operator_learning_runs"))


def _resolve_loads(args: Model2RunArgs) -> np.ndarray:
    if args.loads is not None:
        return np.asarray(args.loads, dtype=float)
    n = int(args.n_load_steps)
    return np.linspace(0.0, float(args.u_x_total), n + 1, dtype=float)


def run_model2_one(args: Model2RunArgs) -> Path:
    """Run one Model-2 sample end-to-end. Returns the recorder directory."""
    args.outdir.mkdir(parents=True, exist_ok=True)

    mat = Model2Material(E=args.E, nu=args.nu, Gc=args.Gc, l0=args.l0)
    case = Model2NotchXStretchCase(
        _model=mat,
        nx=args.nx,
        ny=args.ny,
        crack_y=args.crack_y,
        crack_length=args.crack_length,
        n_load_steps=args.n_load_steps,
        u_x_total=args.u_x_total,
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
        save_quadrature_fields=bool(args.save_quadrature_fields),
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
    driver.run(loads.tolist())
    return args.outdir
