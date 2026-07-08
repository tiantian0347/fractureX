"""Model0 2nd-order vs 4th-order phase-field comparison for the ip_fracture paper.

Runs the circular-hole benchmark (model0) with:
  - 4th-order phase-field  via IPFEMPhaseFieldSolver (through the existing case)
  - 2nd-order phase-field  via MainSolve (fracturex.phasefield.main_solve)

Both branches share the same distmesh grid (built once from
SquareWithCircleHoleDomain) and the same load sequence.  For each polynomial
degree ``p in {2, 3}`` we record the residual-force response and dump one CSV
row per load step.  A single force--displacement comparison PNG is produced.

Outputs under ``$FRACTUREX_RESULTS_ROOT/paper_ipfem/model0_order/``:
  - model0_order_compare.csv      : all curves in long format
  - model0_order_compare.png      : overlay figure

Usage
-----
    python scripts/paper_ipfem/model0_order_compare.py
    python scripts/paper_ipfem/model0_order_compare.py --p 2 3 --hmin 0.05
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fealpy.backend import backend_manager as bm  # noqa: E402
from fealpy.mesh import TriangleMesh  # noqa: E402

from fracturex.interior_penalty.cases._domain_hole import (  # noqa: E402
    SquareWithCircleHoleDomain,
)
from fracturex.interior_penalty.cases.model0_circular_hole import (  # noqa: E402
    Model0CircularHoleCase,
)
from fracturex.phasefield.main_solve import MainSolve  # noqa: E402


DEFAULT_LOADS = np.concatenate(
    (
        np.linspace(0, 70e-3, 6),
        np.linspace(70e-3, 125e-3, 26)[1:],
    )
)


def _build_shared_mesh(hmin: float) -> TriangleMesh:
    domain = SquareWithCircleHoleDomain(hmin=hmin)
    return TriangleMesh.from_domain_distmesh(domain, maxit=100)


def _is_force_boundary_2nd(p):
    return np.abs(p[..., 1] - 1) < 1e-12


def _is_dirichlet_2nd(p):
    return np.abs((p[..., 0] - 0.5) ** 2 + np.abs(p[..., 1] - 0.5) ** 2 - 0.04) < 0.001


def run_second_order(mesh: TriangleMesh, loads: np.ndarray,
                     material: dict, p: int, maxit: int = 30) -> np.ndarray:
    """2nd-order phase-field via MainSolve. Returns residual force per step."""
    # MainSolve mutates the mesh via adaptivity; deep copy defensively.
    mesh_copy = TriangleMesh(bm.copy(mesh.node), bm.copy(mesh.cell))
    ms = MainSolve(mesh=mesh_copy, material_params=material)
    ms.add_boundary_condition(
        "force", "Dirichlet", _is_force_boundary_2nd, loads, "y"
    )
    ms.add_boundary_condition(
        "displacement", "Dirichlet", _is_dirichlet_2nd, 0
    )
    ms.add_boundary_condition(
        "phase", "Dirichlet", _is_dirichlet_2nd, 0
    )
    ms.solve(p=p, maxit=maxit)
    force = ms.get_residual_force()
    force = np.asarray(force, dtype=np.float64).reshape(-1)
    # MainSolve force convention: length == len(loads). Absolute value for plot.
    return np.abs(force)


def run_fourth_order(mesh: TriangleMesh, loads: np.ndarray,
                     material_kwargs: dict, p_phase: int,
                     p_disp: int, gamma: float, maxit: int,
                     rtol: float) -> np.ndarray:
    """4th-order via IPFEM. Returns residual force per step (same length as loads)."""
    case = Model0CircularHoleCase(
        E=material_kwargs["E"], nu=material_kwargs["nu"],
        Gc=material_kwargs["Gc"], l0=material_kwargs["l0"],
        hmin=material_kwargs["hmin"], gamma=gamma,
        p_disp=p_disp, p_phase=p_phase,
        load_sequence=loads.copy(),
    )
    # Reuse the shared mesh instead of re-generating with distmesh.
    solver = case.build_solver(mesh=mesh)
    force = np.zeros_like(loads)
    for i in range(len(loads) - 1):
        solver.newton_raphson(loads[i + 1], maxit=maxit, rtol=rtol,
                              verbose=False)
        force[i + 1] = float(solver.force)
    return np.abs(force)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p", type=int, nargs="+", default=[2, 3],
                        help="polynomial degrees to compare")
    parser.add_argument("--hmin", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="fallback γ for the 4th-order side if degree-adaptive is not set")
    parser.add_argument("--gamma-p2", type=float, default=5.0,
                        help="4th-order side γ for p=2 (matches §5.1 fracture experiments)")
    parser.add_argument("--gamma-p3", type=float, default=10.0,
                        help="4th-order side γ for p=3 (matches §5.1 fracture experiments)")
    parser.add_argument("--gamma-p4", type=float, default=20.0,
                        help="4th-order side γ for p=4 (matches §5.1 fracture experiments)")
    parser.add_argument("--maxit", type=int, default=30)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--E", type=float, default=200.0)
    parser.add_argument("--nu", type=float, default=0.2)
    parser.add_argument("--Gc", type=float, default=1.0)
    parser.add_argument("--l0", type=float, default=0.02)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--skip-second", action="store_true")
    parser.add_argument("--skip-fourth", action="store_true")
    args = parser.parse_args()

    bm.set_backend("numpy")

    if args.out is not None:
        out_dir = Path(args.out)
    else:
        results_root = os.environ.get("FRACTUREX_RESULTS_ROOT",
                                      str(_REPO_ROOT / "results"))
        out_dir = Path(results_root) / "paper_ipfem" / "model0_order"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] writing outputs to {out_dir}", flush=True)

    loads = DEFAULT_LOADS.copy()
    material = dict(E=args.E, nu=args.nu, Gc=args.Gc, l0=args.l0)

    t0 = time.time()
    mesh = _build_shared_mesh(args.hmin)
    print(f"[info] shared mesh: NN={mesh.number_of_nodes()} "
          f"NC={mesh.number_of_cells()} in {time.time()-t0:.1f}s", flush=True)

    rows: list[dict] = []
    curves: dict[tuple[str, int], np.ndarray] = {}

    gamma_by_p = {2: args.gamma_p2, 3: args.gamma_p3, 4: args.gamma_p4}
    print(f"[info] 4th-order γ by p: {gamma_by_p}", flush=True)

    for p in args.p:
        if not args.skip_fourth:
            gamma_p = gamma_by_p.get(p, args.gamma)
            print(f"[run] 4th-order p_phase={p} p_disp=1 γ={gamma_p}",
                  flush=True)
            t = time.time()
            f4 = run_fourth_order(
                mesh, loads,
                material_kwargs=dict(**material, hmin=args.hmin),
                p_phase=p, p_disp=1,
                gamma=gamma_p, maxit=args.maxit, rtol=args.rtol,
            )
            print(f"  4th p={p} done in {time.time()-t:.1f}s peak={f4.max():.3e}",
                  flush=True)
            curves[("fourth_order", p)] = f4
            for i, (u, ff) in enumerate(zip(loads, f4)):
                rows.append({"order": "fourth", "p": p, "step": i,
                             "disp": float(u), "force": float(ff)})

        if not args.skip_second:
            print(f"[run] 2nd-order p={p}", flush=True)
            t = time.time()
            f2 = run_second_order(mesh, loads, material,
                                  p=p, maxit=args.maxit)
            # Length of MainSolve force output equals len(loads).
            if len(f2) != len(loads):
                # Pad or trim to align with load sequence.
                f2 = f2[: len(loads)]
                f2 = np.pad(f2, (0, len(loads) - len(f2)))
            print(f"  2nd p={p} done in {time.time()-t:.1f}s peak={f2.max():.3e}",
                  flush=True)
            curves[("second_order", p)] = f2
            for i, (u, ff) in enumerate(zip(loads, f2)):
                rows.append({"order": "second", "p": p, "step": i,
                             "disp": float(u), "force": float(ff)})

    # --- write CSV
    csv_path = out_dir / "model0_order_compare.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["order", "p", "step",
                                                "disp", "force"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[out] wrote {csv_path}", flush=True)

    # --- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping figure.")
        return 0

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    style = {("fourth_order",): "-", ("second_order",): "--"}
    colors = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}
    for (order, p), f in sorted(curves.items()):
        ls = "-" if order == "fourth_order" else "--"
        label = f"{'4th' if order == 'fourth_order' else '2nd'}-order, p={p}"
        ax.plot(loads, f, ls, color=colors.get(p, "black"),
                marker="o", markersize=3, label=label)
    ax.set_xlabel("prescribed displacement u_y")
    ax.set_ylabel("residual force |F|")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    png_path = out_dir / "model0_order_compare.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"[out] wrote {png_path}", flush=True)

    print(f"[done] total {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
