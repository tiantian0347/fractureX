"""Three-point bending (Ambati §4.4) — standard Lagrange FEM via ``MainSolve``.

This is the displacement-based (non Hu–Zhang) path for model5. Supports sit at
the beam ends; top midspan gets prescribed downward ``u_y``.

Geometry / material match ``fracturex.cases.model5_three_point_bending``
(Ambati Fig. 20): beam ``8×2``, notch depth ``0.4``, ``lam=12``, ``mu=8``,
``Gc=5.4e-4``, ``l0=0.03``.

Run (smoke)
-----------
    PYTHONPATH=fractureX:fealpy python \\
        fracturex/cases/phase_field/model5_three_point_bending.py \\
        --mesh-size 0.15 --max-steps 5

Full Ambati-like schedule uses default ``--max-steps`` (None).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.cases.model5_three_point_bending import _build_gmsh_beam
from fracturex.phasefield.main_solve import MainSolve


class Model5StandardFEM:
    """Ambati TPB wired for ``MainSolve`` boundary API."""

    length: float = 8.0
    height: float = 2.0
    notch_depth: float = 0.4
    notch_mouth: float = 0.2
    left_support_x: float = 0.0
    right_support_x: float = 8.0
    support_half_width: float = 0.1
    load_half_width: float = 0.1
    bd_tol: float = 1e-9

    # Ambati §4.4 (kN–mm)
    params = {
        "lam": 12.0,
        "mu": 8.0,
        "Gc": 5.4e-4,
        "l0": 0.03,
    }

    def __init__(self, *, mesh_size: float = 0.15, with_geometric_notch: bool = True):
        self.mesh_size = float(mesh_size)
        self.with_geometric_notch = bool(with_geometric_notch)
        hw = max(float(self.support_half_width), 0.6 * self.mesh_size)
        self._support_hw = hw
        self._load_hw = max(float(self.load_half_width), 0.6 * self.mesh_size)

    def build_mesh(self) -> TriangleMesh:
        return _build_gmsh_beam(
            mesh_size=self.mesh_size,
            length=self.length,
            height=self.height,
            notch_depth=self.notch_depth,
            notch_mouth=self.notch_mouth,
            with_geometric_notch=self.with_geometric_notch,
            left_support_x=self.left_support_x,
            right_support_x=self.right_support_x,
        )

    def build_box_mesh(self, nx: int = 40, ny: int = 10) -> TriangleMesh:
        """Uniform box mesh (no notch) for elastic kinematics checks."""
        return TriangleMesh.from_box(
            [0.0, self.length, 0.0, self.height], nx=nx, ny=ny
        )

    def force_schedule(self):
        """Downward midspan displacement (negative ``u_y``), Ambati-like."""
        return -bm.concatenate(
            (
                bm.linspace(0.0, 0.04, 41, dtype=bm.float64),
                bm.linspace(0.04, 0.1, 601, dtype=bm.float64)[1:],
            )
        )

    def is_force_boundary(self, p):
        mid = 0.5 * self.length
        return (bm.abs(p[..., 1] - self.height) < self.bd_tol) & (
            bm.abs(p[..., 0] - mid) <= self._load_hw
        )

    def is_left_support(self, p):
        return (bm.abs(p[..., 1]) < self.bd_tol) & (
            bm.abs(p[..., 0] - self.left_support_x) <= self._support_hw
        )

    def is_right_support(self, p):
        return (bm.abs(p[..., 1]) < self.bd_tol) & (
            bm.abs(p[..., 0] - self.right_support_x) <= self._support_hw
        )


def _attach_model5_bcs(ms: MainSolve, model: Model5StandardFEM, disp) -> None:
    ms.add_boundary_condition(
        "force", "Dirichlet", model.is_force_boundary, disp, "y"
    )
    # Left pin: u = 0
    ms.add_boundary_condition(
        "displacement", "Dirichlet", model.is_left_support, 0
    )
    # Right roller: u_y = 0
    ms.add_boundary_condition(
        "displacement", "Dirichlet", model.is_right_support, 0, "y"
    )


def run_elastic_kinematics(
    *,
    mesh_size: float = 0.2,
    load: float = 1e-3,
    p: int = 1,
    use_box: bool = True,
) -> dict:
    """One elastic-like step (large Gc) and report midspan kinematics / stiffness."""
    model = Model5StandardFEM(mesh_size=mesh_size, with_geometric_notch=False)
    # Suppress damage for kinematics check
    params = dict(model.params)
    params["Gc"] = 1.0e3
    params["l0"] = 0.2

    mesh = model.build_box_mesh(nx=40, ny=10) if use_box else model.build_mesh()
    node = np.asarray(mesh.entity("node"))
    n_load = int(np.asarray(model.is_force_boundary(node)).sum())
    n_L = int(np.asarray(model.is_left_support(node)).sum())
    n_R = int(np.asarray(model.is_right_support(node)).sum())
    if n_load < 1 or n_L < 1 or n_R < 1:
        raise RuntimeError(
            f"BC nodes missing: load={n_load}, left={n_L}, right={n_R}"
        )

    disp = bm.array([0.0, -float(load)], dtype=bm.float64)
    ms = MainSolve(mesh=mesh, material_params=params, model_type="HybridModel")
    _attach_model5_bcs(ms, model, disp)
    ms.solve(p=p, maxit=30)

    GD = int(mesh.geo_dimension())
    # TensorFunctionSpace(shape=(GD, -1)): component-major → (NN, GD) nodal.
    un = np.asarray(ms.uh[:], dtype=float).reshape(GD, -1).T
    i_load = int(np.argmin(np.abs(node[:, 0] - 4) + np.abs(node[:, 1] - 2)))
    i_bot = int(np.argmin(np.abs(node[:, 0] - 4) + np.abs(node[:, 1] - 0)))
    force = np.asarray(ms.get_residual_force(), dtype=float)
    R = float(force[-1]) if force.size else float("nan")
    uy_load = float(un[i_load, 1])
    uy_bot = float(un[i_bot, 1])
    return {
        "uy_load": uy_load,
        "uy_bot": uy_bot,
        "ratio": uy_bot / uy_load if abs(uy_load) > 1e-30 else 0.0,
        "R": R,
        "k": abs(R) / load if load > 0 else float("nan"),
        "n_load": n_load,
        "n_L": n_L,
        "n_R": n_R,
        "NN": int(mesh.number_of_nodes()),
        "NC": int(mesh.number_of_cells()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Model5 TPB — standard FEM (MainSolve)"
    )
    parser.add_argument("--degree", default=1, type=int)
    parser.add_argument("--maxit", default=50, type=int)
    parser.add_argument("--mesh-size", default=0.15, type=float)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--model_type", default="HybridModel", type=str)
    parser.add_argument(
        "--no-notch",
        action="store_true",
        help="use intact rectangle (no geometric V-notch)",
    )
    parser.add_argument(
        "--outdir",
        default="results/phasefield/model5_standard_fem",
        type=str,
    )
    parser.add_argument("--vtkname", default="model5_std", type=str)
    parser.add_argument("--save-vtk", action="store_true")
    args = parser.parse_args(argv)

    bm.set_backend(args.backend)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    model = Model5StandardFEM(
        mesh_size=args.mesh_size,
        with_geometric_notch=not args.no_notch,
    )
    mesh = model.build_mesh()
    node = np.asarray(mesh.entity("node"))
    n_load = int(np.asarray(model.is_force_boundary(node)).sum())
    if n_load < 1:
        raise SystemExit(f"no load-boundary nodes matched (mesh_size={args.mesh_size})")
    print(
        f"[model5-std] NN={mesh.number_of_nodes()} NC={mesh.number_of_cells()} "
        f"load_nodes={n_load} left={int(model.is_left_support(node).sum())} "
        f"right={int(model.is_right_support(node).sum())}"
    )

    disp = model.force_schedule()
    if args.max_steps is not None:
        disp = disp[: args.max_steps + 1]

    ms = MainSolve(
        mesh=mesh, material_params=model.params, model_type=args.model_type
    )
    _attach_model5_bcs(ms, model, disp)
    if args.save_vtk:
        ms.save_vtkfile(fname=str(out / args.vtkname))

    t0 = time.time()
    ms.solve(p=args.degree, maxit=args.maxit)
    print(f"[model5-std] wall time {time.time() - t0:.1f}s")

    force = bm.to_numpy(ms.get_residual_force())
    disp_np = bm.to_numpy(disp)
    np.savetxt(
        out / f"{args.vtkname}_force_disp.txt",
        np.c_[disp_np, force],
        header="disp(mm)  reaction_force",
        comments="",
    )
    print(
        f"[model5-std] |R|_max={np.nanmax(np.abs(force)):.3e} "
        f"wrote {out / (args.vtkname + '_force_disp.txt')}"
    )


if __name__ == "__main__":
    main()
