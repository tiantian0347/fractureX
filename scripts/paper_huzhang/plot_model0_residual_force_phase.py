#!/usr/bin/env python3
"""Plot the audited clamped circular-boundary response and phase field.

The figures use the geometry, material, boundary conditions, and report loads
defined by ``model0_example.py``.  The input data are obtained on a
length-scale-resolved mesh with stabilized report-load continuation.  The input
directory must contain
``residual_force_vs_displacement.csv`` and either ``final.vtu`` or VTU files
under ``vtk/``, with point data named ``damage``.

Usage
-----
python scripts/paper_huzhang/plot_model0_residual_force_phase.py
python scripts/paper_huzhang/plot_model0_residual_force_phase.py \
    --input-dir results/phasefield_solver/model0_fine_curve_audit_unrelaxed_h0065 \
    --output-dir docs/benchmarks/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402


SCRIPT_VERSION = "2.1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_INPUT_DIR = (
    REPOSITORY_ROOT
    / "results/phasefield_solver/model0_fine_curve_audit_unrelaxed_h0065"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs/benchmarks/figures"


def _read_force_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a finite, monotone reaction-force curve from CSV.

    Parameters
    ----------
    path : Path
        CSV with ``displacement`` and ``residual_force_abs`` columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Displacement and absolute residual force arrays in source order.

    Raises
    ------
    ValueError
        If required columns are missing or values are invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"force curve does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "residual_force_abs" not in rows[0]:
        raise ValueError(f"invalid force curve columns in {path}")
    coordinate_field = (
        "displacement" if "displacement" in rows[0] else "load"
    )
    if coordinate_field not in rows[0]:
        raise ValueError(f"force curve needs displacement or load column: {path}")
    displacement = np.asarray([float(row[coordinate_field]) for row in rows])
    force = np.asarray([float(row["residual_force_abs"]) for row in rows])
    if displacement.ndim != 1 or displacement.size < 2:
        raise ValueError("force curve must contain at least two samples")
    if not np.isfinite(displacement).all() or not np.isfinite(force).all():
        raise ValueError("force curve contains non-finite values")
    if np.any(np.diff(displacement) < 0.0):
        raise ValueError("displacement samples must be monotone")
    if np.any(force < 0.0):
        raise ValueError("residual_force_abs must be nonnegative")
    return displacement, force


def _read_damage_vtu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read triangular geometry and pointwise damage from one VTU file."""
    if not path.is_file():
        raise FileNotFoundError(f"VTU file does not exist: {path}")
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("VTK is required to read the final phase field") from exc

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    points = vtk_to_numpy(grid.GetPoints().GetData())[:, :2]
    damage_array = grid.GetPointData().GetArray("damage")
    if damage_array is None:
        raise ValueError(f"point data 'damage' is missing from {path}")
    damage = np.asarray(vtk_to_numpy(damage_array), dtype=float).reshape(-1)
    triangles = np.empty((grid.GetNumberOfCells(), 3), dtype=np.int64)
    for index in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(index)
        if cell.GetNumberOfPoints() != 3:
            raise ValueError("final phase-field plot requires triangular cells")
        triangles[index] = [cell.GetPointId(j) for j in range(3)]
    if not np.isfinite(points).all() or not np.isfinite(damage).all():
        raise ValueError("VTU geometry or damage contains non-finite values")
    if damage.size != points.shape[0] or np.any((damage < 0.0) | (damage > 1.0)):
        raise ValueError("damage must be pointwise and lie in [0, 1]")
    return points, triangles, damage


def _save_force_figure(
    displacement: np.ndarray,
    force: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a publication-sized reaction-force--displacement figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    peak = int(np.argmax(force))
    figure, axis = plt.subplots(figsize=(5.5, 3.7))
    axis.plot(
        displacement,
        force,
        color="#1F6D8F",
        linewidth=2.0,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
        solid_capstyle="round",
        label="standard finite element solution",
    )
    axis.scatter(
        displacement[peak],
        force[peak],
        s=34,
        color="#C4513D",
        zorder=4,
        label=rf"peak: $|R_y|={force[peak]:.2f}$ at $\bar{{u}}={displacement[peak]:.4f}$",
    )
    axis.axvline(displacement[peak], color="#C4513D", linewidth=0.9, linestyle="--", alpha=0.65)
    axis.set_xlabel("prescribed displacement")
    axis.set_ylabel("reaction force")
    axis.grid(axis="y", color="#DDE3E8", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.legend(frameon=False, fontsize=8.3, loc="best")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"model0_residual_force_curve.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _save_phase_figure(
    points: np.ndarray,
    triangles: np.ndarray,
    damage: np.ndarray,
    output_dir: Path,
    load: float,
) -> None:
    """Save the final pointwise phase-field distribution over the mesh."""
    output_dir.mkdir(parents=True, exist_ok=True)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    figure, axis = plt.subplots(figsize=(5.1, 4.2))
    axis.set_facecolor("#F4F5F3")
    field = axis.tripcolor(
        triangulation,
        damage,
        shading="gouraud",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
    )
    axis.add_patch(
        patches.Circle((0.5, 0.5), 0.2, facecolor="#F4F5F3", edgecolor="#4D5357", linewidth=0.8)
    )
    axis.set_aspect("equal")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(
        rf"Clamped circular-boundary plate: final phase field $d$ "
        rf"($\bar u={load:.4f}$)"
    )
    axis.tick_params(direction="out", length=3.0, width=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    colorbar = figure.colorbar(field, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"phase field $d$")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"model0_final_phase_field.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_model0(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Generate both figures and return their PNG paths."""
    displacement, force = _read_force_curve(input_dir / "residual_force_vs_displacement.csv")
    direct_final_vtu = input_dir / "final.vtu"
    if direct_final_vtu.is_file():
        final_vtu = direct_final_vtu
    else:
        vtu_files = sorted((input_dir / "vtk").glob("*.vtu"))
        if not vtu_files:
            raise FileNotFoundError(
                f"no final.vtu or VTU files found under {input_dir}"
            )
        final_vtu = vtu_files[-1]
    points, triangles, damage = _read_damage_vtu(final_vtu)
    load = float(displacement[-1])
    _save_force_figure(displacement, force, output_dir / "loaddisp")
    _save_phase_figure(points, triangles, damage, output_dir / "phasefield", load)
    return (
        output_dir / "loaddisp/model0_residual_force_curve.png",
        output_dir / "phasefield/model0_final_phase_field.png",
    )


def _parse_args() -> argparse.Namespace:
    """Parse deterministic input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Generate clamped circular-boundary force and phase-field figures."""
    args = _parse_args()
    force_png, phase_png = plot_model0(args.input_dir.resolve(), args.output_dir.resolve())
    print(f"[{SCRIPT_VERSION}] wrote {force_png}")
    print(f"[{SCRIPT_VERSION}] wrote {phase_png}")


if __name__ == "__main__":
    main()
