"""Lightweight, parametrized Model-4 driver entry for dataset generation.

Mirror of [model0_runner.py](model0_runner.py) / [model2_runner.py](model2_runner.py)
for the Ambati / COMSOL notched plate with an offset hole.

Geometry / BC / material live in
[model4_notched_plate_with_hole.py](../../cases/model4_notched_plate_with_hole.py).
The paper experiment counterpart is
[phasefield_model4_notched_plate_huzhang.py](../phasefield_model4_notched_plate_huzhang.py).

Why a separate runner: the paper script carries env-var switches, VTU /
benchmark prints, and solver-mode labels. Dataset generation only needs
``Model4RunArgs`` → build (case, discr, damage, assemblers) → one fixed
solver pair → run loads → recorder dir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.cases.model4_notched_plate_with_hole import (
    Model4NotchedPlateWithHoleCase,
)
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)
from fracturex.postprocess.recorder import RunRecorder
from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_auxspace


@dataclass
class Model4Material:
    """Ambati §4.6 Lamé parameters (kN–mm). Prefer ``lam``/``mu`` over E/ν."""

    lam: float = 1.94
    mu: float = 2.45
    Gc: float = 2.28e-3
    l0: float = 0.1
    ft: float = 3.0

    @property
    def E(self) -> float:
        return self.mu * (3.0 * self.lam + 2.0 * self.mu) / (self.lam + self.mu)

    @property
    def nu(self) -> float:
        return self.lam / (2.0 * (self.lam + self.mu))


@dataclass
class Model4RunArgs:
    """Per-sample knobs for a Model-4 dataset run.

    Geometry (gmsh plate, mm):
        mesh_size: target edge length (smaller → finer). Ambati ℓ₀=0.1;
            production runs should refine toward ``ℓ₀/2`` near the crack path.
        width, height, notch_*, hole_*, *_pin_*: forwarded to the case.
        with_geometric_notch: False for Hu–Zhang (intact left edge + d=1
            pre-crack); True only for standard FEM / IP-FEM geometric notch.

    Material (Lamé):
        lam, mu, Gc, l0 — see Model4Material.

    Discretization:
        p_sigma: HuZhang stress order (paper default 3).
        damage_p: continuous Lagrange order for d (paper default 2).

    Loading:
        loads: optional explicit schedule. If None, uses Ambati
            ``np.linspace(0, u_y_total, n_load_steps + 1)`` with
            ``Δu = u_y_total / n_load_steps`` (default 1e-3 mm up to 2 mm).

    Solver:
        elastic_mode: ``'direct'`` or ``'aux'``.
        save_every: checkpoint every N steps (default 100 for long runs).

    Output:
        outdir: recorder output directory (created if missing).
    """

    mesh_size: float = 2.0
    width: float = 65.0
    height: float = 120.0
    notch_width: float = 10.0
    notch_height: float = 0.5
    notch_y: float = 65.0
    # Hu–Zhang runner default: no geometric cut (dangling corners).
    with_geometric_notch: bool = False
    hole_center: tuple = (36.5, 51.0)
    hole_radius: float = 10.0
    lower_pin_center: tuple = (20.0, 20.0)
    upper_pin_center: tuple = (20.0, 100.0)
    pin_radius: float = 5.0

    lam: float = 1.94
    mu: float = 2.45
    Gc: float = 2.28e-3
    l0: float = 0.1

    p_sigma: int = 3
    damage_p: int = 2
    use_relaxation: bool = True

    n_load_steps: int = 2000
    u_y_total: float = 2.0
    loads: Optional[list[float]] = None

    elastic_mode: str = "direct"  # 'direct' | 'aux'
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


def _resolve_loads(args: Model4RunArgs) -> np.ndarray:
    if args.loads is not None:
        return np.asarray(args.loads, dtype=float)
    n = int(args.n_load_steps)
    return np.linspace(0.0, float(args.u_y_total), n + 1, dtype=float)


def run_model4_one(args: Model4RunArgs) -> Path:
    """Run one Model-4 sample end-to-end. Returns the recorder directory."""
    args.outdir.mkdir(parents=True, exist_ok=True)

    mat = Model4Material(
        lam=args.lam, mu=args.mu, Gc=args.Gc, l0=args.l0
    )
    case = Model4NotchedPlateWithHoleCase(
        _model=mat,
        mesh_size=args.mesh_size,
        width=args.width,
        height=args.height,
        notch_width=args.notch_width,
        notch_height=args.notch_height,
        notch_y=args.notch_y,
        with_geometric_notch=bool(args.with_geometric_notch),
        hole_center=args.hole_center,
        hole_radius=args.hole_radius,
        lower_pin_center=args.lower_pin_center,
        upper_pin_center=args.upper_pin_center,
        pin_radius=args.pin_radius,
        lam=args.lam,
        mu=args.mu,
        Gc=args.Gc,
        l0=args.l0,
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
        discr, case, damage, formulation="standard"
    )
    phase_assembler = PhaseFieldAssembler(
        discr, case, damage, debug=False
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
                A,
                F,
                gdof_sigma=discr.gdof_sigma,
                vspace=discr.space_u,
                atol=args.elastic_atol,
                rtol=args.elastic_rtol,
            )
            return x
    else:
        raise ValueError(
            f"unknown elastic_mode {args.elastic_mode!r}; "
            "expected 'direct' or 'aux'"
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
