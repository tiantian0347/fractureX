"""Asymmetric notched beam with three holes (Ambati §4.5) — standard FEM.

Displacement-based (non Hu–Zhang) path for model6. Geometric notch is cut
into the mesh. Supports at the bottom; downward ``u_y`` at top centre.

Geometry / material: ``fracturex.cases.model6_asymmetric_notched_beam``.

Run (smoke)
-----------
    PYTHONPATH=fractureX:fealpy python \\
        fracturex/cases/phase_field/model6_asymmetric_beam.py \\
        --mesh-size 0.4 --max-steps 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm

from fracturex.cases.model6_asymmetric_notched_beam import (
    Model6AsymmetricNotchedBeamCase,
)
from fracturex.cases.phase_field._force_disp import write_force_disp
from fracturex.phasefield.main_solve import MainSolve


class Model6StandardFEM:
    """Ambati asymmetric beam wired for ``MainSolve`` (geometric notch)."""

    params = {
        "lam": 12.0,
        "mu": 8.0,
        "Gc": 1.0e-3,
        "l0": 0.01,
    }

    def __init__(self, *, mesh_size: float = 0.2):
        self.mesh_size = float(mesh_size)
        mouth = max(0.05, 0.4 * self.mesh_size)
        hw = max(0.2, self.mesh_size)
        self.case = Model6AsymmetricNotchedBeamCase(
            mesh_size=self.mesh_size,
            with_geometric_notch=True,
            notch_mouth=mouth,
            support_half_width=hw,
            load_half_width=hw,
            bd_tol=max(1e-6, 0.02 * self.mesh_size),
            debug_mesh=True,
        )

    def build_mesh(self):
        return self.case.make_mesh()

    def is_left_support(self, p):
        return self.case._on_left_support(p)

    def is_right_support(self, p):
        return self.case._on_right_support(p)

    def is_load(self, p):
        return self.case._on_load(p)

    def force_schedule(self):
        """Downward midspan ``u_y`` (negative, mm)."""
        return -self.case.default_loads()


def _attach_bcs(ms: MainSolve, model: Model6StandardFEM, disp) -> None:
    ms.add_boundary_condition("force", "Dirichlet", model.is_load, disp, "y")
    ms.add_boundary_condition("displacement", "Dirichlet", model.is_left_support, 0)
    ms.add_boundary_condition(
        "displacement", "Dirichlet", model.is_right_support, 0, "y"
    )


def count_bc_nodes(model: Model6StandardFEM, mesh) -> dict:
    """Count nodal hits on supports and the load patch."""
    node = np.asarray(mesh.entity("node"))
    return {
        "NN": int(mesh.number_of_nodes()),
        "NC": int(mesh.number_of_cells()),
        "n_left": int(np.asarray(model.is_left_support(node)).sum()),
        "n_right": int(np.asarray(model.is_right_support(node)).sum()),
        "n_load": int(np.asarray(model.is_load(node)).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Model6 asymmetric beam — standard FEM")
    parser.add_argument("--degree", default=1, type=int)
    parser.add_argument("--maxit", default=50, type=int)
    parser.add_argument("--mesh-size", default=0.2, type=float)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--u-start", default=None, type=float)
    parser.add_argument("--u-end", default=None, type=float)
    parser.add_argument("--n-steps", default=None, type=int)
    parser.add_argument("--restart-npz", default=None, type=str)
    parser.add_argument("--save-state-npz", default=None, type=str)
    parser.add_argument("--merge-with", default=None, type=str)
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--model_type", default="HybridModel", type=str)
    parser.add_argument(
        "--lin-method",
        default="gmres",
        type=str,
        help="MainSolve linear solver: gmres|direct|auto|pardiso|mumps. "
             "Lab uses gmres: displacement Jacobian has empty Dirichlet rows "
             "that SuperLU/PARDISO reject.",
    )
    parser.add_argument(
        "--outdir", default="results/phasefield/model6_standard_fem", type=str
    )
    parser.add_argument("--vtkname", default="model6_std", type=str)
    parser.add_argument("--save-vtk", action="store_true")
    args = parser.parse_args(argv)

    bm.set_backend(args.backend)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    model = Model6StandardFEM(mesh_size=args.mesh_size)
    mesh = model.build_mesh()
    info = count_bc_nodes(model, mesh)
    print(
        f"[model6-std] NN={info['NN']} NC={info['NC']} "
        f"left={info['n_left']} right={info['n_right']} load={info['n_load']}"
    )
    if info["n_left"] < 1 or info["n_right"] < 1 or info["n_load"] < 1:
        raise SystemExit(f"BC nodes missing: {info}")

    disp = model.force_schedule()
    if args.u_start is not None or args.n_steps is not None:
        u0 = 0.0 if args.u_start is None else float(args.u_start)
        u1 = 0.25 if args.u_end is None else float(args.u_end)
        n = 250 if args.n_steps is None else int(args.n_steps)
        disp = -bm.linspace(u0, u1, n + 1, dtype=bm.float64)
    elif args.max_steps is not None:
        disp = disp[: args.max_steps + 1]

    ms = MainSolve(mesh=mesh, material_params=model.params, model_type=args.model_type)
    _attach_bcs(ms, model, disp)
    if args.save_vtk:
        ms.save_vtkfile(fname=str(out / args.vtkname))

    t0 = time.time()
    ms.solve(
        p=args.degree,
        maxit=args.maxit,
        restart_npz=args.restart_npz,
        linear_solver_options={"method": args.lin_method},
    )
    print(f"[model6-std] wall time {time.time() - t0:.1f}s")

    if args.save_state_npz:
        ms.save_restart_npz(args.save_state_npz)
        print(f"[model6-std] saved state {args.save_state_npz}")

    table = write_force_disp(
        out / f"{args.vtkname}_force_disp.txt",
        bm.to_numpy(disp),
        bm.to_numpy(ms.get_residual_force()),
        merge_with=args.merge_with,
    )
    print(
        f"[model6-std] |R|_max={np.nanmax(np.abs(table[:, 1])):.3e} "
        f"wrote {out / (args.vtkname + '_force_disp.txt')}"
    )


if __name__ == "__main__":
    main()
