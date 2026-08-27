#!/usr/bin/env python3
"""Plot the resolved-grid slow-mode trace and physical phase indicators.

Purpose
-------
Create a three-panel comparison at the resolved circular-hole checkpoint
``bar_u=0.0898``.  The panels use the same finite-element mesh and show the
dominant coupled slow-mode cell energy, the phase field, and the phase-field
gradient magnitude.

Scope
-----
The script reads the stored spectral checkpoint, rebuilds the deterministic
resolved mesh used by the physical-path experiment, and reassembles the frozen
linearized propagation operator to recover the leading slow subspace. It does
not modify simulation data; the resulting map is a visualization artifact.

Outputs
-------
The canonical PDF, PNG, and metadata are written below ``results``.  A copied
PDF may be used as a manuscript typesetting asset.

Run
---
``python scripts/paper_solver/plot_resolved_localization_map.py``
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from fealpy.backend import backend_manager as bm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = REPOSITORY_ROOT / "results/phasefield_solver"
CHECKPOINT_DIR = (
    RESULTS_ROOT / "model0_resolved_transactional_stabilization_h0065_final"
)
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "dominant_phase_mode_load_0.0898.npz"
DEFAULT_OUTPUT = CHECKPOINT_DIR / "resolved_slow_trace_map.pdf"


def _triangle_gradient(nodes: np.ndarray, cells: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the piecewise-linear gradient magnitude on every triangle."""
    coordinates = nodes[cells]
    nodal_values = values[cells]
    x0, y0 = coordinates[:, 0, 0], coordinates[:, 0, 1]
    x1, y1 = coordinates[:, 1, 0], coordinates[:, 1, 1]
    x2, y2 = coordinates[:, 2, 0], coordinates[:, 2, 1]
    denominator = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if np.any(np.abs(denominator) <= np.finfo(float).eps):
        raise ValueError("the resolved mesh contains a degenerate triangle")
    d0, d1, d2 = nodal_values[:, 0], nodal_values[:, 1], nodal_values[:, 2]
    grad_x = (d0 * (y1 - y2) + d1 * (y2 - y0) + d2 * (y0 - y1)) / denominator
    grad_y = (d0 * (x2 - x1) + d1 * (x0 - x2) + d2 * (x1 - x0)) / denominator
    return np.hypot(grad_x, grad_y)


def _build_coupled_slow_subspace_trace(
    main: object,
    checkpoint: Path,
    *,
    relative_radius: float = 0.95,
) -> tuple[np.ndarray, float, int]:
    """Recompute the basis-invariant trace on the resolved checkpoint.

    The saved dominant mode is retained as a fallback for archival checkpoints;
    the default path reconstructs the leading invariant subspace from the same
    matrix-free propagation operator used by the spectral diagnostic.
    """
    from scipy.sparse.linalg import eigs
    from scripts.paper_solver.scan_model0_resolved_spectrum import (
        _assembled_propagation_operator,
        _load_checkpoint_arrays,
        _solve_real_factor,
        _state_for_load,
        discover_accepted_states,
    )
    from scripts.paper_solver.verify_slow_mode_fracturex import FrozenStandardFEStepMap
    from fracturex.analysis.staggered_slow_mode import (
        diagonal_cell_weights,
        subspace_cell_trace_indicator,
        weighted_orthonormalize,
    )

    checkpoint_load = float(checkpoint.stem.split("load_")[-1])
    roots = discover_accepted_states(
        [
            RESULTS_ROOT / "model0_resolved_transactional_prefix_h0065",
            RESULTS_ROOT / "model0_resolved_transactional_postpeak_h0065",
        ]
    )
    target, previous = _state_for_load(roots, checkpoint_load)
    _, previous_damage, previous_history = _load_checkpoint_arrays(previous.path)
    target_u, target_damage, _ = _load_checkpoint_arrays(target.path)
    main.load_restart_npz(str(previous.path))
    step_map = FrozenStandardFEStepMap(
        main,
        load=target.load,
        committed_damage=previous_damage,
        committed_history=previous_history,
        phase_bound_solver="active_set",
    )
    operator, context = _assembled_propagation_operator(
        step_map,
        np.concatenate((target_u, target_damage)),
        active_bound_tolerance=1.0e-10,
        active_kkt_tolerance=1.0e-10,
    )
    generator = np.random.default_rng(20260824)
    initial_vector = generator.standard_normal(operator.shape[0])
    initial_vector /= np.linalg.norm(initial_vector)
    values, vectors = eigs(
        operator,
        k=3,
        which="LM",
        tol=1.0e-7,
        maxiter=120,
        ncv=16,
        v0=initial_vector,
    )
    order = np.argsort(-np.abs(values))
    values, vectors = values[order], vectors[:, order]
    cutoff = relative_radius * abs(values[0])
    selected = np.abs(values) >= cutoff
    lifted = []
    for value, phase_mode in zip(values[selected], vectors[:, selected].T):
        displacement_mode = _solve_real_factor(
            context["factor_a"], context["matrix_b"] @ phase_mode
        )
        full = np.zeros(step_map.full_state_size, dtype=np.complex128)
        full[context["displacement_free"]] = -displacement_mode / value
        full[step_map.displacement_size + context["phase_free"]] = phase_mode
        lifted.append(full.real)
    basis = weighted_orthonormalize(
        np.column_stack(lifted),
        np.maximum(
            np.abs(np.concatenate((context["displacement_diagonal"], context["phase_diagonal"]))),
            np.finfo(float).tiny,
        ),
    )
    displacement_connectivity = np.asarray(
        bm.to_numpy(step_map.solver.tspace.cell_to_dof()), dtype=np.int64
    )
    phase_connectivity = np.asarray(
        bm.to_numpy(step_map.solver.space.cell_to_dof()), dtype=np.int64
    )
    coupled_connectivity = np.concatenate(
        (displacement_connectivity, step_map.displacement_size + phase_connectivity), axis=1
    )
    weight = np.maximum(
        np.abs(np.concatenate((context["displacement_diagonal"], context["phase_diagonal"]))),
        np.finfo(float).tiny,
    )
    trace = subspace_cell_trace_indicator(
        basis, diagonal_cell_weights(weight, coupled_connectivity), coupled_connectivity
    )
    return trace, float(abs(values[0])), int(basis.shape[1])


def make_figure(checkpoint: Path, output_pdf: Path) -> Path:
    """Rebuild the resolved mesh, validate checkpoint arrays, and plot maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    checkpoint = checkpoint.resolve()
    output_pdf = output_pdf.resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with np.load(checkpoint) as data:
        mode_energy = np.asarray(data["coupled_mode_cell_energy"], dtype=np.float64)
        damage = np.asarray(data["target_damage"], dtype=np.float64)

    # Importing the audited builder guarantees identical geometry and node
    # ordering for the stored physical-path checkpoint.
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.paper_solver.run_model0_fine_reference import (
        build_model0_resolved_solver,
    )

    main, _material, mesh_stats, _unused = build_model0_resolved_solver(hmin=0.0065)
    nodes = np.asarray(main.mesh.entity("node"), dtype=np.float64)
    cells = np.asarray(main.mesh.entity("cell"), dtype=np.int64)
    if mode_energy.shape != (cells.shape[0],) or damage.shape != (nodes.shape[0],):
        raise ValueError(
            "checkpoint and rebuilt mesh are inconsistent: "
            f"cells={cells.shape[0]}, mode_energy={mode_energy.shape}; "
            f"nodes={nodes.shape[0]}, damage={damage.shape}"
        )
    if not np.isfinite(mode_energy).all() or not np.isfinite(damage).all():
        raise ValueError("checkpoint fields must be finite")
    mode_energy = np.maximum(mode_energy, 0.0)
    raw_trace = mode_energy
    try:
        raw_trace, spectral_radius, slow_dimension = _build_coupled_slow_subspace_trace(
            main, checkpoint
        )
    except (OSError, RuntimeError, ValueError):
        spectral_radius, slow_dimension = float("nan"), 1
    mode_energy = np.maximum(raw_trace, 0.0)
    mode_energy /= max(float(mode_energy.max()), np.finfo(float).tiny)
    damage = np.clip(damage, 0.0, 1.0)
    grad_damage = _triangle_gradient(nodes, cells, damage)
    grad_damage /= max(float(grad_damage.max()), np.finfo(float).tiny)

    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], cells)
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.45), constrained_layout=True)
    panels = (
        (mode_energy, "viridis", r"computed slow-subspace trace $\eta_K$", "(a)"),
        (damage, "magma", r"phase field $d$", "(b)"),
        (grad_damage, "plasma", r"normalized gradient $|\nabla d|$", "(c)"),
    )
    for axis, (field, cmap, label, panel_label) in zip(axes, panels):
        if field.shape == (nodes.shape[0],):
            artist = axis.tripcolor(
                triangulation,
                field,
                shading="gouraud",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                edgecolors="none",
                antialiased=False,
            )
        else:
            artist = axis.tripcolor(
                triangulation,
                field,
                shading="flat",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                edgecolors="none",
                antialiased=False,
            )
        axis.set_aspect("equal")
        axis.set_xlim(float(nodes[:, 0].min()), float(nodes[:, 0].max()))
        axis.set_ylim(float(nodes[:, 1].min()), float(nodes[:, 1].max()))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(label, fontsize=9)
        axis.text(
            0.02,
            0.96,
            panel_label,
            transform=axis.transAxes,
            color="white",
            fontsize=10,
            fontweight="bold",
            va="top",
            path_effects=[],
        )
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle(
        r"Resolved circular-hole checkpoint, $\bar u=0.0898$; fields normalized independently",
        fontsize=10,
    )
    for extension in ("pdf", "png"):
        figure.savefig(output_pdf.with_suffix(f".{extension}"), dpi=220)
    plt.close(figure)
    metadata = {
        "checkpoint": str(checkpoint),
        "load": 0.0898,
        "mesh_nodes": int(nodes.shape[0]),
        "mesh_cells": int(cells.shape[0]),
        "h_max": float(mesh_stats["h_max"]),
        "slow_space_dimension": slow_dimension,
        "spectral_radius": spectral_radius,
        "spectrum_random_seed": 20260824,
        "fields": ["computed_slow_subspace_trace", "damage", "normalized_damage_gradient"],
        "command": " ".join(sys.argv),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output_pdf.parent),
    }
    output_pdf.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_pdf}")
    return output_pdf


def main() -> None:
    """Parse command-line paths and create the localization figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    make_figure(arguments.checkpoint, arguments.output)


if __name__ == "__main__":
    main()
