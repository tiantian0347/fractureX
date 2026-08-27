#!/usr/bin/env python3
"""Plot a local-refinement diagnostic state for the V-notch bending case.

Purpose
-------
Rebuild the deterministic Gmsh mesh used by the local-refinement standard-FEM
run, plot one saved nodal phase-field state, and write reproducibility metadata.
This script does not infer a converged load path from a stopped nonlinear run.

Entry point
-----------
    PYTHONPATH=.:../fealpy python \\
      scripts/paper_huzhang/plot_model5_local_refined_diagnostic.py

Numerical contract
------------------
The input ``step_XXXX.npz`` must contain ``node``, ``cell``, and ``d`` arrays;
``d`` is nodal, finite, and clipped to [0, 1] for plotting.  Coordinates and
loads use the kN--mm convention of the V-notch benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_damage_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load one saved ``(node, cell, d, step)`` diagnostic state.

    Returns
    -------
    node : (n_nodes, 2) float64 array
    cell : (n_cells, 3) integer array
    damage : (n_nodes,) float64 array clipped to [0, 1]
    step : integer load-step index
    """
    with np.load(path) as state:
        required = {"node", "cell", "d"}
        missing = required.difference(state.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        node = np.asarray(state["node"], dtype=float)[:, :2]
        cell = np.asarray(state["cell"], dtype=np.int64)
        damage = np.asarray(state["d"], dtype=float).reshape(-1)
        step = int(np.asarray(state["step"]).item()) if "step" in state else -1
    if node.ndim != 2 or node.shape[1] != 2:
        raise ValueError(f"invalid node shape: {node.shape}")
    if cell.ndim != 2 or cell.shape[1] != 3:
        raise ValueError(f"invalid cell shape: {cell.shape}")
    if damage.size != node.shape[0] or not np.isfinite(damage).all():
        raise ValueError("damage must be finite and nodal")
    return node, cell, np.clip(damage, 0.0, 1.0), step


def compute_mesh_metrics(
    node: np.ndarray,
    cell: np.ndarray,
    *,
    center_x: float,
    half_width: float,
    ymax: float,
    length_scale: float,
) -> dict:
    """Compute global and local maximum edge lengths in millimetres."""
    edges = np.stack(
        (
            np.linalg.norm(node[cell[:, 1]] - node[cell[:, 0]], axis=1),
            np.linalg.norm(node[cell[:, 2]] - node[cell[:, 1]], axis=1),
            np.linalg.norm(node[cell[:, 0]] - node[cell[:, 2]], axis=1),
        ),
        axis=1,
    ).max(axis=1)
    barycenter = node[cell].mean(axis=1)
    local = (np.abs(barycenter[:, 0] - center_x) <= half_width) & (
        barycenter[:, 1] <= ymax
    )
    return {
        "global_h_max_mm": float(edges.max()),
        "local_h_max_mm": float(edges[local].max()),
        "local_cell_count": int(local.sum()),
        "cell_count": int(cell.shape[0]),
        "node_count": int(node.shape[0]),
        "length_scale_mm": float(length_scale),
        "local_h_lt_length_scale_over_2": bool(edges[local].max() < length_scale / 2.0),
    }


def plot_damage(
    path: Path,
    node: np.ndarray,
    cell: np.ndarray,
    damage: np.ndarray,
    load: float,
    mesh_metrics: dict,
) -> None:
    """Write a publication-sized phase-field image."""
    triangulation = mtri.Triangulation(node[:, 0], node[:, 1], cell)
    figure, axis = plt.subplots(figsize=(7.2, 3.3))
    field = axis.tripcolor(
        triangulation, damage, shading="gouraud", cmap="YlOrRd", vmin=0.0, vmax=1.0
    )
    axis.set_aspect("equal")
    axis.set_xlim(0.0, 8.0)
    axis.set_ylim(0.0, 2.0)
    axis.set_xlabel(r"$x$ (mm)")
    axis.set_ylabel(r"$y$ (mm)")
    axis.set_title(rf"V-notch beam local-refined phase field ($|u_y|={load:.4f}$ mm)")
    axis.spines[["top", "right"]].set_visible(False)
    colorbar = figure.colorbar(field, ax=axis, fraction=0.036, pad=0.03)
    colorbar.set_label(r"phase field $d$")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate(args: argparse.Namespace) -> dict:
    """Generate diagnostic figures and metadata from one saved state."""
    state_path = args.state.resolve()
    node, cell, damage, step = load_damage_state(state_path)
    metrics = compute_mesh_metrics(
        node,
        cell,
        center_x=args.center_x,
        half_width=args.local_refine_half_width,
        ymax=args.local_refine_ymax,
        length_scale=args.length_scale,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_damage(output / "model5_local_refined_diagnostic_phase_field", node, cell, damage, args.load, metrics)
    metadata = {
        "case": "v_notch_three_point_bending",
        "state_path": str(state_path),
        "step": step,
        "load_magnitude_mm": float(args.load),
        "mesh": metrics,
        "local_refinement": {
            "background_mesh_size_mm": float(args.background_mesh_size),
            "local_mesh_size_target_mm": float(args.local_mesh_size),
            "half_width_mm": float(args.local_refine_half_width),
            "ymax_mm": float(args.local_refine_ymax),
            "transition_mm": float(args.local_refine_transition),
        },
        "material": {"lam": 12.0, "mu": 8.0, "Gc": 5.4e-4, "l0_mm": 0.03},
        "status": args.status,
        "figures": [
            "model5_local_refined_diagnostic_phase_field.png",
            "model5_local_refined_diagnostic_phase_field.pdf",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse reproducible diagnostic plotting parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_ROOT / "results/phasefield/model5_standard_fem/std_local_h001_u0001_u006_direct/d_npz/step_0042.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/phasefield/model5_standard_fem/std_local_h001_diagnostic_figures",
    )
    parser.add_argument("--load", type=float, default=0.043)
    parser.add_argument("--background-mesh-size", type=float, default=0.1)
    parser.add_argument("--local-mesh-size", type=float, default=0.01)
    parser.add_argument("--local-refine-half-width", type=float, default=0.5)
    parser.add_argument("--local-refine-ymax", type=float, default=1.6)
    parser.add_argument("--local-refine-transition", type=float, default=0.15)
    parser.add_argument("--center-x", type=float, default=4.0)
    parser.add_argument("--length-scale", type=float, default=0.03)
    parser.add_argument(
        "--status",
        default="diagnostic_path_stopped_at_staggered_nonconvergence",
        help="status recorded in summary.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = generate(parse_args())
    print(json.dumps(result["mesh"], ensure_ascii=False))
