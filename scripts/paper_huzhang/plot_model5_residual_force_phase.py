#!/usr/bin/env python3
"""Plot the Model-5 standard-FEM reaction curve and final phase field.

The force data and the final phase-field state must come from the same
``model5_three_point_bending.py`` run.  The script rebuilds the deterministic
Gmsh mesh used by that run and attaches the saved nodal phase field to it.

The default input is the existing same-run standard-FEM path ending at
``|u_y|=0.08``.  Outputs are written below the repository-level ``results``
directory, including a normalized CSV and a JSON provenance record.

Example
-------
    PYTHONPATH=. python scripts/paper_huzhang/plot_model5_residual_force_phase.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results/phasefield/model5_standard_fem/std_h010_anim"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results/phasefield_solver/model5_standard_fem_h010_anim_figures"


def _load_force_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read signed displacement/reaction columns and return magnitudes.

    Numerical contract: the returned arrays are finite, have equal length,
    contain at least two samples, and displacement is nondecreasing.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"invalid force curve shape: {data.shape}")
    signed_u = np.asarray(data[:, 0], dtype=float)
    signed_r = np.asarray(data[:, 1], dtype=float)
    displacement = np.abs(signed_u)
    reaction = np.abs(signed_r)
    if not np.isfinite(data[:, :2]).all():
        raise ValueError(f"non-finite force curve values: {path}")
    if np.any(np.diff(displacement) < -1.0e-12):
        raise ValueError("displacement magnitude is not monotone")
    return displacement, reaction


def _build_mesh(mesh_size: float):
    """Build the exact Model-5 mesh and return coordinates, cells, h_max."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from fracturex.cases.phase_field.model5_three_point_bending import Model5StandardFEM

    model = Model5StandardFEM(mesh_size=mesh_size, with_geometric_notch=True)
    mesh = model.build_mesh()
    node = np.asarray(mesh.entity("node"), dtype=float)[:, :2]
    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
    edges = np.concatenate(
        (
            np.linalg.norm(node[cell[:, 1]] - node[cell[:, 0]], axis=1),
            np.linalg.norm(node[cell[:, 2]] - node[cell[:, 1]], axis=1),
            np.linalg.norm(node[cell[:, 0]] - node[cell[:, 2]], axis=1),
        )
    )
    return model, node, cell, float(np.max(edges))


def _load_final_damage(path: Path, node_count: int) -> np.ndarray:
    """Read and clip the final nodal damage state."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as state:
        if "d" not in state:
            raise ValueError(f"state has no d field: {path}")
        damage = np.asarray(state["d"], dtype=float).reshape(-1)
    if damage.size != node_count:
        raise ValueError(
            f"state d has {damage.size} entries but rebuilt mesh has {node_count} nodes"
        )
    if not np.isfinite(damage).all():
        raise ValueError("final damage contains non-finite values")
    return np.clip(damage, 0.0, 1.0)


def _write_force_csv(path: Path, displacement: np.ndarray, reaction: np.ndarray) -> None:
    """Write a self-contained, signed-and-absolute force record."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "displacement",
                "reaction_force_abs",
                "displacement_signed",
                "reaction_force_signed",
            ]
        )
        for step, (u, r) in enumerate(zip(displacement, reaction)):
            writer.writerow([step, f"{u:.16e}", f"{r:.16e}", f"{-u:.16e}", f"{-r:.16e}"])


def _plot_force(path: Path, displacement: np.ndarray, reaction: np.ndarray) -> None:
    """Save a publication-sized residual/reaction force curve."""
    peak = int(np.argmax(reaction))
    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    axis.plot(displacement, reaction, color="#1F6D8F", linewidth=1.8)
    axis.scatter(
        displacement[peak],
        reaction[peak],
        s=38,
        color="#C4513D",
        zorder=4,
    )
    axis.axvline(displacement[peak], color="#C4513D", linewidth=0.8, linestyle="--", alpha=0.6)
    axis.annotate(
        rf"$|R_y|={reaction[peak]:.4f}$ at $\bar{{u}}={displacement[peak]:.4f}$",
        xy=(displacement[peak], reaction[peak]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=8.0,
        color="#8F3025",
    )
    axis.set_xlabel(r"prescribed displacement $\bar u$ (mm)")
    axis.set_ylabel(r"reaction force $|R_y|$ (kN)")
    axis.grid(axis="y", color="#DDE3E8", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_phase(path: Path, node: np.ndarray, cell: np.ndarray, damage: np.ndarray, load: float, h_max: float) -> None:
    """Save the final nodal phase-field distribution."""
    triangulation = mtri.Triangulation(node[:, 0], node[:, 1], cell)
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    field = axis.tripcolor(
        triangulation, damage, shading="gouraud", cmap="YlOrRd", vmin=0.0, vmax=1.0
    )
    axis.set_aspect("equal")
    axis.set_xlim(0.0, 8.0)
    axis.set_ylim(0.0, 2.0)
    axis.set_xlabel(r"$x$ (mm)")
    axis.set_ylabel(r"$y$ (mm)")
    axis.set_title("V-notch beam phase field near the end of continuation")
    axis.spines[["top", "right"]].set_visible(False)
    colorbar = figure.colorbar(field, ax=axis, fraction=0.036, pad=0.03)
    colorbar.set_label(r"phase field $d$")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate(input_dir: Path, output_dir: Path, mesh_size: float) -> dict:
    """Generate figures, normalized data, and provenance metadata."""
    force_path = input_dir / "model5_std_force_disp.txt"
    state_path = input_dir / "model5_std_state.npz"
    displacement, reaction = _load_force_curve(force_path)
    model, node, cell, h_max = _build_mesh(mesh_size)
    damage = _load_final_damage(state_path, node.shape[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_force_csv(output_dir / "residual_force_vs_displacement.csv", displacement, reaction)
    _plot_force(output_dir / "model5_residual_force_curve", displacement, reaction)
    _plot_phase(output_dir / "model5_final_phase_field", node, cell, damage, float(displacement[-1]), h_max)
    shutil.copy2(force_path, output_dir / "source_model5_std_force_disp.txt")
    shutil.copy2(state_path, output_dir / "source_model5_std_state.npz")

    schedule_count = 41 + 600
    metadata = {
        "case": "model5_three_point_bending",
        "source_run": str(force_path.parent),
        "load_path": {
            "complete_default_schedule": bool(displacement.size == schedule_count),
            "steps": int(displacement.size - 1),
            "samples": int(displacement.size),
            "displacement_range_mm": [float(displacement[0]), float(displacement[-1])],
            "default_schedule_end_mm": 0.1,
        },
        "geometry_mm": {
            "length": float(model.length),
            "height": float(model.height),
            "notch_depth": float(model.notch_depth),
            "notch_mouth": float(model.notch_mouth),
        },
        "material": {key: float(value) for key, value in model.params.items()},
        "boundary_conditions": {
            "left_support": "u_x=u_y=0 on bottom-left support segment",
            "right_support": "u_y=0 on bottom-right roller segment",
            "load": "prescribed downward u_y on top midspan segment",
            "support_half_width_mm": float(model.support_half_width),
            "load_half_width_mm": float(model.load_half_width),
        },
        "mesh": {
            "requested_mesh_size_mm": float(mesh_size),
            "max_edge_mm": h_max,
            "nodes": int(node.shape[0]),
            "cells": int(cell.shape[0]),
            "length_scale_resolved_hmax_lt_l0_over_2": bool(h_max < model.params["l0"] / 2.0),
        },
        "peak": {
            "displacement_mm": float(displacement[np.argmax(reaction)]),
            "reaction_force_abs": float(np.max(reaction)),
        },
        "final_phase_field": {
            "max_damage": float(np.max(damage)),
            "min_damage": float(np.min(damage)),
        },
        "figures": [
            "model5_residual_force_curve.png",
            "model5_residual_force_curve.pdf",
            "model5_final_phase_field.png",
            "model5_final_phase_field.pdf",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mesh-size", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = generate(args.input_dir.resolve(), args.output_dir.resolve(), args.mesh_size)
    print(json.dumps(metadata["peak"], ensure_ascii=False))
    print(f"wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
