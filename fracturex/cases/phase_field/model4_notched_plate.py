"""Notched plate with offset hole (Ambati §4.6) — standard Lagrange FEM.

Displacement-based (non Hu–Zhang) path for model4. Geometric notch is cut
into the mesh. Lower pin is fixed; upper pin has prescribed ``u_y``.

Geometry / material: ``fracturex.cases.model4_notched_plate_with_hole``.

Run (smoke)
-----------
    PYTHONPATH=fractureX:fealpy python \\
        fracturex/cases/phase_field/model4_notched_plate.py \\
        --mesh-size 2 --max-steps 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm

from fracturex.cases.model4_notched_plate_with_hole import (
    Model4NotchedPlateWithHoleCase,
)
from fracturex.cases.phase_field._force_disp import write_force_disp
from fracturex.phasefield.main_solve import MainSolve


class Model4StandardFEM:
    """Ambati holed plate wired for ``MainSolve`` (geometric notch)."""

    params = {
        "lam": 1.94,
        "mu": 2.45,
        "Gc": 2.28e-3,
        "l0": 0.1,
    }

    def __init__(self, *, mesh_size: float = 2.0):
        self.mesh_size = float(mesh_size)
        self.case = Model4NotchedPlateWithHoleCase(
            mesh_size=self.mesh_size,
            with_geometric_notch=True,
            debug_mesh=True,
        )

    def build_mesh(self):
        return self.case.make_mesh()

    def is_lower_pin(self, p):
        return self.case._on_lower_pin(p)

    def is_upper_pin(self, p):
        return self.case._on_upper_pin(p)

    def force_schedule(self):
        """Upward upper-pin ``u_y`` (mm), Ambati Δu=1e-3 to 2 mm."""
        return self.case.default_loads()


def _attach_bcs(ms: MainSolve, model: Model4StandardFEM, disp) -> None:
    ms.add_boundary_condition("force", "Dirichlet", model.is_upper_pin, disp, "y")
    ms.add_boundary_condition("displacement", "Dirichlet", model.is_lower_pin, 0)


def count_bc_nodes(model: Model4StandardFEM, mesh) -> dict:
    """Count nodal hits on pin circles. Used by tests and the driver log."""
    node = np.asarray(mesh.entity("node"))
    return {
        "NN": int(mesh.number_of_nodes()),
        "NC": int(mesh.number_of_cells()),
        "n_lower": int(np.asarray(model.is_lower_pin(node)).sum()),
        "n_upper": int(np.asarray(model.is_upper_pin(node)).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Model4 holed plate — standard FEM")
    parser.add_argument("--degree", default=1, type=int)
    parser.add_argument("--maxit", default=50, type=int)
    parser.add_argument("--mesh-size", default=2.0, type=float)
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
    parser.add_argument("--outdir", default="results/phasefield/model4_standard_fem", type=str)
    parser.add_argument("--vtkname", default="model4_std", type=str)
    parser.add_argument("--save-vtk", action="store_true")
    args = parser.parse_args(argv)

    bm.set_backend(args.backend)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    model = Model4StandardFEM(mesh_size=args.mesh_size)
    mesh = model.build_mesh()
    info = count_bc_nodes(model, mesh)
    print(
        f"[model4-std] NN={info['NN']} NC={info['NC']} "
        f"lower_pin={info['n_lower']} upper_pin={info['n_upper']}"
    )
    if info["n_lower"] < 1 or info["n_upper"] < 1:
        raise SystemExit(f"pin nodes missing: {info}")

    disp = model.force_schedule()
    if args.u_start is not None or args.n_steps is not None:
        u0 = 0.0 if args.u_start is None else float(args.u_start)
        u1 = 2.0 if args.u_end is None else float(args.u_end)
        n = 200 if args.n_steps is None else int(args.n_steps)
        disp = bm.linspace(u0, u1, n + 1, dtype=bm.float64)
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
    print(f"[model4-std] wall time {time.time() - t0:.1f}s")

    if args.save_state_npz:
        ms.save_restart_npz(args.save_state_npz)
        print(f"[model4-std] saved state {args.save_state_npz}")

    table = write_force_disp(
        out / f"{args.vtkname}_force_disp.txt",
        bm.to_numpy(disp),
        bm.to_numpy(ms.get_residual_force()),
        merge_with=args.merge_with,
    )
    print(
        f"[model4-std] |R|_max={np.nanmax(np.abs(table[:, 1])):.3e} "
        f"wrote {out / (args.vtkname + '_force_disp.txt')}"
    )


if __name__ == "__main__":
    main()
