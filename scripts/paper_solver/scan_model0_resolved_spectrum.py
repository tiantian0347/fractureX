#!/usr/bin/env python3
"""Compute matrix-free staggered spectra on the resolved Model-0 path.

Purpose
-------
This diagnostic assembles the coupled Jacobian blocks at accepted states on
the length-scale-resolved physical path.  After fixing the history branch and
KKT active set, it applies ``T = D^{-1} C A^{-1} B`` through sparse block
solves.  Matrix-free Arnoldi therefore avoids forming the dense propagation
matrix while matching the propagation operator used in the paper.

The script uses consecutive accepted *internal* states as the continuation
history.  It therefore measures the local map associated with the actual
resolved path, not a report-point replay with an artificial load jump.

Outputs
-------
For every requested load, the script writes one JSON summary and one NumPy
file containing the leading phase-space eigenvectors.  All files are placed
below ``results/phasefield_solver`` for reproducibility.

Usage
-----
PYTHONPATH=. python scripts/paper_solver/scan_model0_resolved_spectrum.py \
    --loads 0.0876,0.0898,0.1030 \
    --output-dir ../results/phasefield_solver/model0_resolved_transactional_stabilization_h0065
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import LinearOperator, eigs, splu

from fealpy.backend import backend_manager as bm

from scripts.paper_solver.run_model0_fine_reference import (
    build_model0_resolved_solver,
)
from scripts.paper_solver.scan_model0_resolved_online_rate import (
    _load_checkpoint_arrays,
    parse_loads,
    require_results_directory,
)
from fracturex.analysis.staggered_slow_mode import (
    compute_cell_energy_from_diagonal_weight,
    select_bulk_cells,
)
from scripts.paper_solver.verify_slow_mode_fracturex import (
    FrozenStandardFEStepMap,
    _assemble_history_field_coupling_blocks,
)


SCRIPT_VERSION = "2.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT.parent / "results/phasefield_solver"
DEFAULT_PREFIX_DIR = RESULTS_ROOT / "model0_resolved_transactional_prefix_h0065"
DEFAULT_POSTPEAK_DIR = RESULTS_ROOT / "model0_resolved_transactional_postpeak_h0065"
DEFAULT_OUTPUT_DIR = (
    RESULTS_ROOT / "model0_resolved_transactional_stabilization_h0065"
)
DEFAULT_LOADS = "0.0876,0.0898,0.1030"
ACCEPTED_PATTERN = re.compile(
    r"accepted_(?P<index>\d+)_load_(?P<load>[0-9.]+)\.npz$"
)


@dataclass(frozen=True)
class AcceptedState:
    """One accepted internal continuation state."""

    index: int
    load: float
    path: Path
    root: Path


def discover_accepted_states(roots: Iterable[Path]) -> list[AcceptedState]:
    """Return unique accepted internal states ordered by load.

    Parameters
    ----------
    roots : iterable[pathlib.Path]
        Resolved-path result directories containing ``accepted_states``.

    Returns
    -------
    list[AcceptedState]
        States sorted by their physical loading parameter.
    """
    states: list[AcceptedState] = []
    for root in roots:
        root = root.resolve()
        for path in sorted((root / "accepted_states").glob("accepted_*.npz")):
            match = ACCEPTED_PATTERN.match(path.name)
            if match is None:
                continue
            states.append(
                AcceptedState(
                    index=int(match.group("index")),
                    load=float(match.group("load")),
                    path=path.resolve(),
                    root=root,
                )
            )
    states.sort(key=lambda state: (state.load, str(state.path)))
    if len(states) < 2:
        raise ValueError("at least two accepted states are required")
    return states


def _state_for_load(states: list[AcceptedState], load: float) -> tuple[AcceptedState, AcceptedState]:
    """Return the target state and its immediate physical predecessor."""
    matches = [state for state in states if np.isclose(
        state.load, load, rtol=0.0, atol=5.0e-5
    )]
    if not matches:
        raise ValueError(f"expected one accepted state for load {load:.6f}")
    candidates: list[tuple[AcceptedState, AcceptedState]] = []
    for target in matches:
        same_root = sorted(
            (state for state in states if state.root == target.root),
            key=lambda state: state.load,
        )
        prior = [state for state in same_root if state.load < target.load - 5.0e-5]
        if prior:
            candidates.append((target, prior[-1]))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one internally contiguous state for load {load:.6f}; "
            f"found {len(candidates)}"
        )
    target, previous = candidates[0]
    if previous.load >= target.load:
        raise ValueError(f"load {load:.6f} has no preceding accepted state")
    return target, previous


def _git_commit() -> str | None:
    """Return the current repository commit when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _solve_real_factor(factor: Any, right_hand_side: np.ndarray) -> np.ndarray:
    """Apply a real sparse factorization to real or complex data."""
    rhs = np.asarray(right_hand_side)
    if np.iscomplexobj(rhs):
        return factor.solve(rhs.real) + 1j * factor.solve(rhs.imag)
    return factor.solve(rhs)


def parse_coverages(text: str) -> list[float]:
    """Parse unique increasing slow-mode energy coverage fractions."""
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    array = np.asarray(values, dtype=np.float64)
    if (
        array.size == 0
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
        or np.any(array > 1.0)
        or np.any(np.diff(array) <= 0.0)
    ):
        raise ValueError("coverages must be unique, increasing values in (0, 1]")
    return array.tolist()


def _assembled_propagation_operator(
    step_map: FrozenStandardFEStepMap,
    state: np.ndarray,
    *,
    active_bound_tolerance: float,
    active_kkt_tolerance: float,
) -> tuple[LinearOperator, dict[str, Any]]:
    """Assemble and factor the free-block propagation operator.

    Parameters
    ----------
    step_map : FrozenStandardFEStepMap
        Frozen-load map carrying the previous accepted damage and history.
    state : ndarray, shape (n_u+n_d,)
        Accepted state where the block Jacobian is assembled.
    active_bound_tolerance, active_kkt_tolerance : float
        Nonnegative tolerances for identifying lower/upper KKT coordinates.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
        Phase-free action ``D_ff^{-1} C_ff A_ff^{-1} B_ff``.
    dict
        Block matrices, free indices, residual diagnostics, and work counters.
    """
    candidate = np.asarray(state, dtype=np.float64).reshape(-1)
    if candidate.size != step_map.full_state_size:
        raise ValueError("state has an incompatible size")
    if active_bound_tolerance < 0.0 or active_kkt_tolerance < 0.0:
        raise ValueError("active-set tolerances must be nonnegative")

    residual = step_map.assemble_coupled_residual(
        candidate, enforce_phase_box=False
    )
    jacobian_ud, jacobian_du = _assemble_history_field_coupling_blocks(step_map)
    displacement_matrix = step_map.last_displacement_matrix.tocsr()
    phase_matrix = step_map.last_phase_matrix.tocsr()
    displacement_free = np.flatnonzero(~step_map.fixed_displacement_mask)
    damage = candidate[step_map.displacement_size :]
    phase_residual = residual[step_map.displacement_size :]
    lower = step_map.damage_lower_bound
    upper = step_map.damage_upper_bound
    projected_argument = damage - phase_residual
    projected_damage_residual = damage - np.minimum(
        np.maximum(projected_argument, lower), upper
    )
    projected_interior = (
        projected_argument > lower + active_bound_tolerance
    ) & (
        projected_argument < upper - active_bound_tolerance
    )
    active_lower = (~projected_interior) & (
        projected_argument <= lower + active_bound_tolerance
    )
    active_upper = (~projected_interior) & (
        projected_argument >= upper - active_bound_tolerance
    )
    # A tiny physical residual can leave a numerically converged bound row on
    # the interior side of the strict projection test. Use the KKT tolerance
    # only to settle such near-bound coordinates.
    active_lower |= (
        damage <= lower + active_bound_tolerance
    ) & (phase_residual >= -active_kkt_tolerance)
    active_upper |= (
        damage >= upper - active_bound_tolerance
    ) & (phase_residual <= active_kkt_tolerance)
    phase_active = step_map.fixed_damage_mask | active_lower | active_upper
    phase_free = np.flatnonzero(~phase_active)
    if displacement_free.size == 0 or phase_free.size == 0:
        raise RuntimeError("assembled propagation operator has an empty free block")

    matrix_a = displacement_matrix[displacement_free][:, displacement_free].tocsc()
    matrix_d = phase_matrix[phase_free][:, phase_free].tocsc()
    matrix_b = jacobian_ud[displacement_free][:, phase_free].tocsr()
    matrix_c = jacobian_du[phase_free][:, displacement_free].tocsr()
    factor_start = perf_counter()
    factor_a = splu(matrix_a)
    factor_d = splu(matrix_d)
    factor_seconds = perf_counter() - factor_start
    counts = {"matvecs": 0, "seconds": 0.0}

    def matvec(vector: np.ndarray) -> np.ndarray:
        """Apply ``D_ff^{-1} C_ff A_ff^{-1} B_ff``."""
        start = perf_counter()
        direction = np.asarray(vector).reshape(-1)
        if direction.shape != (phase_free.size,):
            raise ValueError("Arnoldi direction has an incompatible size")
        displacement_response = _solve_real_factor(
            factor_a, matrix_b @ direction
        )
        result = _solve_real_factor(
            factor_d, matrix_c @ displacement_response
        )
        counts["matvecs"] += 1
        counts["seconds"] += perf_counter() - start
        return result

    operator = LinearOperator(
        shape=(phase_free.size, phase_free.size),
        matvec=matvec,
        dtype=np.float64,
    )
    free_residual = np.concatenate(
        (residual[displacement_free], projected_damage_residual[phase_free])
    )
    projected_residual_norm = float(
        np.linalg.norm(
            np.concatenate(
                (residual[displacement_free], projected_damage_residual)
            )
        )
    )
    context = {
        "factor_a": factor_a,
        "matrix_a": matrix_a,
        "matrix_d": matrix_d,
        "matrix_b": matrix_b,
        "matrix_c": matrix_c,
        "displacement_free": displacement_free,
        "phase_free": phase_free,
        "phase_active_lower": active_lower,
        "phase_active_upper": active_upper,
        "displacement_diagonal": np.asarray(
            displacement_matrix.diagonal(), dtype=np.float64
        ),
        "phase_diagonal": np.asarray(phase_matrix.diagonal(), dtype=np.float64),
        "full_residual_norm": float(np.linalg.norm(residual)),
        "free_residual_norm": float(np.linalg.norm(free_residual)),
        "projected_residual_norm": projected_residual_norm,
        "factor_seconds": float(factor_seconds),
        "counts": counts,
    }
    return operator, context


def _stabilization_diagnostics(
    step_map: FrozenStandardFEStepMap,
    context: dict[str, Any],
    dominant_coupled: np.ndarray,
    eigenvalue: complex,
    *,
    coverages: list[float],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Apply the exact local-elimination derivative on nested slow regions."""
    displacement_connectivity = np.asarray(
        bm.to_numpy(step_map.solver.tspace.cell_to_dof()), dtype=np.int64
    )
    phase_connectivity = np.asarray(
        bm.to_numpy(step_map.solver.space.cell_to_dof()), dtype=np.int64
    )
    coupled_connectivity = np.concatenate(
        (
            displacement_connectivity,
            step_map.displacement_size + phase_connectivity,
        ),
        axis=1,
    )
    weight = np.concatenate(
        (context["displacement_diagonal"], context["phase_diagonal"])
    )
    weight = np.maximum(np.abs(weight), np.finfo(np.float64).tiny)
    cell_energy = compute_cell_energy_from_diagonal_weight(
        dominant_coupled, weight, coupled_connectivity
    )
    global_free = np.concatenate(
        (
            context["displacement_free"],
            step_map.displacement_size + context["phase_free"],
        )
    )
    global_to_free = np.full(step_map.full_state_size, -1, dtype=np.int64)
    global_to_free[global_free] = np.arange(global_free.size, dtype=np.int64)
    free_mode = dominant_coupled[global_free]
    free_weight = weight[global_free]
    jacobian = bmat(
        [
            [context["matrix_a"], context["matrix_b"]],
            [context["matrix_c"], context["matrix_d"]],
        ],
        format="csr",
    )
    jacobian_mode = jacobian @ free_mode
    denominator = float(np.sum(free_weight * np.abs(free_mode) ** 2))
    rows: list[dict[str, Any]] = []
    for theta in coverages:
        selected_cells = select_bulk_cells(cell_energy, theta=theta)
        patch_global = np.unique(coupled_connectivity[selected_cells].reshape(-1))
        patch = global_to_free[patch_global]
        patch = np.unique(patch[patch >= 0])
        if patch.size == 0:
            raise RuntimeError("selected slow region contains no free coordinates")
        factor_start = perf_counter()
        local_factor = splu(jacobian[patch][:, patch].tocsc())
        local_correction = _solve_real_factor(
            local_factor, jacobian_mode[patch]
        )
        factor_seconds = perf_counter() - factor_start
        projected = free_mode.copy()
        projected[patch] -= local_correction
        numerator = float(np.sum(free_weight * np.abs(projected) ** 2))
        chi = float(np.sqrt(max(0.0, numerator) / denominator))
        composite_factor = float(abs(eigenvalue) * chi)
        rows.append(
            {
                "coverage_target": float(theta),
                "selected_cells": int(np.count_nonzero(selected_cells)),
                "selected_cell_fraction": float(np.mean(selected_cells)),
                "captured_mode_energy": float(
                    np.sum(cell_energy[selected_cells]) / np.sum(cell_energy)
                ),
                "patch_free_dofs": int(patch.size),
                "patch_free_dof_fraction": float(patch.size / global_free.size),
                "chi_omega": chi,
                "lambda_abs_chi": composite_factor,
                "restores_or_strengthens_contraction": bool(composite_factor < 1.0),
                "local_factor_wall_time_seconds": float(factor_seconds),
            }
        )
    return rows, cell_energy


def compute_spectrum(
    main: Any,
    *,
    previous: AcceptedState,
    target: AcceptedState,
    active_bound_tolerance: float,
    active_kkt_tolerance: float,
    nev: int,
    arnoldi_tolerance: float,
    arnoldi_maxiter: int,
    ncv: int,
    seed: int,
    coverages: list[float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute leading eigenpairs of the assembled block propagation."""
    _, previous_d, previous_h = _load_checkpoint_arrays(previous.path)
    target_u, target_d, target_h = _load_checkpoint_arrays(target.path)
    main.load_restart_npz(str(previous.path))
    step_map = FrozenStandardFEStepMap(
        main,
        load=target.load,
        committed_damage=previous_d,
        committed_history=previous_h,
        phase_bound_solver="active_set",
    )
    target_state = np.concatenate((target_u, target_d))
    operator, context = _assembled_propagation_operator(
        step_map,
        target_state,
        active_bound_tolerance=active_bound_tolerance,
        active_kkt_tolerance=active_kkt_tolerance,
    )
    start = perf_counter()
    generator = np.random.default_rng(seed)
    initial_vector = generator.standard_normal(operator.shape[0])
    initial_vector /= np.linalg.norm(initial_vector)
    values, vectors = eigs(
        operator,
        k=nev,
        which="LM",
        tol=arnoldi_tolerance,
        maxiter=arnoldi_maxiter,
        ncv=ncv,
        v0=initial_vector,
        return_eigenvectors=True,
    )
    elapsed = perf_counter() - start
    order = np.argsort(-np.abs(values))
    values = values[order]
    vectors = vectors[:, order]
    eigen_residuals = [
        float(
            np.linalg.norm(operator @ vectors[:, index] - values[index] * vectors[:, index])
            / max(float(np.linalg.norm(vectors[:, index])), np.finfo(np.float64).tiny)
        )
        for index in range(values.size)
    ]
    dominant_free = vectors[:, 0]
    dominant_phase = np.zeros(step_map.damage_size, dtype=np.complex128)
    dominant_phase[context["phase_free"]] = dominant_free
    displacement_response = _solve_real_factor(
        context["factor_a"], context["matrix_b"] @ dominant_free
    )
    dominant_displacement = np.zeros(
        step_map.displacement_size, dtype=np.complex128
    )
    dominant_displacement[context["displacement_free"]] = (
        -displacement_response / values[0]
    )
    dominant_coupled = np.concatenate(
        (dominant_displacement, dominant_phase)
    )
    tangent_diagonal = np.concatenate(
        (context["displacement_diagonal"], context["phase_diagonal"])
    )
    positive_weight = np.maximum(
        np.abs(tangent_diagonal), np.finfo(np.float64).tiny
    )
    weighted_norm = float(
        np.sqrt(np.sum(positive_weight * np.abs(dominant_coupled) ** 2))
    )
    dominant_coupled /= weighted_norm
    dominant_phase /= weighted_norm
    dominant_displacement /= weighted_norm
    stabilization, cell_energy = _stabilization_diagnostics(
        step_map,
        context,
        dominant_coupled,
        values[0],
        coverages=coverages,
    )
    assembled_history = np.asarray(
        bm.to_numpy(step_map.solver.H), dtype=np.float64
    )
    return {
        "load": float(target.load),
        "previous_load": float(previous.load),
        "target_path": str(target.path),
        "previous_path": str(previous.path),
        "damage_dofs": int(step_map.damage_size),
        "fixed_damage_dofs": int(np.count_nonzero(step_map.fixed_damage_mask)),
        "free_displacement_dofs": int(context["displacement_free"].size),
        "free_phase_dofs": int(context["phase_free"].size),
        "active_lower_phase_dofs": int(
            np.count_nonzero(context["phase_active_lower"])
        ),
        "active_upper_phase_dofs": int(
            np.count_nonzero(context["phase_active_upper"])
        ),
        "full_residual_norm": context["full_residual_norm"],
        "free_residual_norm": context["free_residual_norm"],
        "projected_residual_norm": context["projected_residual_norm"],
        "history_relative_difference": float(
            np.linalg.norm(assembled_history - target_h)
            / max(1.0, float(np.linalg.norm(target_h)))
        ),
        "target_history_norm": float(np.linalg.norm(target_h)),
        "eigenvalues_real": [float(value.real) for value in values],
        "eigenvalues_imag": [float(value.imag) for value in values],
        "eigenvalue_moduli": [float(abs(value)) for value in values],
        "spectral_radius": float(np.max(np.abs(values))),
        "eigenpair_residual_norms": eigen_residuals,
        "arnoldi_wall_time_seconds": float(elapsed),
        "factor_wall_time_seconds": context["factor_seconds"],
        "active_bound_tolerance": float(active_bound_tolerance),
        "active_kkt_tolerance": float(active_kkt_tolerance),
        "arnoldi_tolerance": float(arnoldi_tolerance),
        "arnoldi_maxiter": int(arnoldi_maxiter),
        "arnoldi_ncv": int(ncv),
        "random_seed": int(seed),
        "matvecs": context["counts"],
        "stabilization": stabilization,
        "dominant_vector_file": f"dominant_phase_mode_load_{target.load:.4f}.npz",
    }, {
        "dominant_phase_mode": dominant_phase,
        "dominant_displacement_mode": dominant_displacement,
        "dominant_coupled_mode": dominant_coupled,
        "target_damage": target_d,
        "phase_free_dofs": context["phase_free"],
        "displacement_free_dofs": context["displacement_free"],
        "coupled_mode_cell_energy": cell_energy,
    }


def parse_args() -> argparse.Namespace:
    """Parse reproducible resolved-spectrum parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmin", type=float, default=0.0065)
    parser.add_argument("--loads", default=DEFAULT_LOADS)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--postpeak-dir", type=Path, default=DEFAULT_POSTPEAK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--active-bound-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--active-kkt-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--nev", type=int, default=2)
    parser.add_argument("--arnoldi-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--arnoldi-maxiter", type=int, default=80)
    parser.add_argument("--ncv", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--coverages", default="0.5,0.6,0.7")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    """Run the requested resolved-path spectrum measurements."""
    loads = parse_loads(args.loads)
    coverages = parse_coverages(args.coverages)
    if args.nev <= 0 or args.nev >= args.ncv - 1:
        raise ValueError("nev must be positive and smaller than ncv-1")
    if args.ncv <= 2 or args.arnoldi_maxiter <= 0:
        raise ValueError("ncv and arnoldi-maxiter must be positive")
    if args.active_bound_tolerance < 0.0 or args.active_kkt_tolerance < 0.0:
        raise ValueError("active-set tolerances must be nonnegative")
    output_dir = require_results_directory(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = discover_accepted_states(
        [args.prefix_dir.resolve(), args.postpeak_dir.resolve()]
    )
    main, material, mesh_stats, unused_nodes = build_model0_resolved_solver(
        hmin=args.hmin
    )
    rows: list[dict[str, Any]] = []
    for load in loads:
        target, previous = _state_for_load(states, load)
        summary, arrays = compute_spectrum(
            main,
            previous=previous,
            target=target,
            active_bound_tolerance=args.active_bound_tolerance,
            active_kkt_tolerance=args.active_kkt_tolerance,
            nev=args.nev,
            arnoldi_tolerance=args.arnoldi_tolerance,
            arnoldi_maxiter=args.arnoldi_maxiter,
            ncv=args.ncv,
            seed=args.seed,
            coverages=coverages,
        )
        np.savez_compressed(
            output_dir / summary["dominant_vector_file"], **arrays
        )
        rows.append(summary)
        (output_dir / f"spectrum_load_{target.load:.4f}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"load={target.load:.4f} rho={summary['spectral_radius']:.6f} "
            f"free_residual={summary['free_residual_norm']:.3e} "
            f"matvecs={summary['matvecs']['matvecs']}",
            flush=True,
        )

    csv_path = output_dir / "resolved_spectrum.csv"
    coverage_columns = tuple(
        f"coverage_{int(round(100.0 * theta))}_lambda_chi"
        for theta in coverages
    )
    fieldnames = (
        "load",
        "previous_load",
        "spectral_radius",
        "projected_residual_norm",
        "free_residual_norm",
        "free_phase_dofs",
        "active_lower_phase_dofs",
        "arnoldi_wall_time_seconds",
        "matvecs",
        "eigenpair_residual_norm",
    ) + coverage_columns
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat_row = {
                    "load": row["load"],
                    "previous_load": row["previous_load"],
                    "spectral_radius": row["spectral_radius"],
                    "projected_residual_norm": row["projected_residual_norm"],
                    "free_residual_norm": row["free_residual_norm"],
                    "free_phase_dofs": row["free_phase_dofs"],
                    "active_lower_phase_dofs": row["active_lower_phase_dofs"],
                    "arnoldi_wall_time_seconds": row["arnoldi_wall_time_seconds"],
                    "matvecs": row["matvecs"]["matvecs"],
                    "eigenpair_residual_norm": row["eigenpair_residual_norms"][0],
            }
            for column, diagnostic in zip(
                coverage_columns, row["stabilization"]
            ):
                flat_row[column] = diagnostic["lambda_abs_chi"]
            writer.writerow(flat_row)
    metadata = {
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "command": shlex.join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "fealpy")
        },
        "case": "Model-0 circular-hole plate",
        "source_case": "fracturex/cases/phase_field/model0_example.py",
        "material": material,
        "hmin_target": args.hmin,
        "mesh": mesh_stats,
        "unused_distmesh_nodes_removed": unused_nodes,
        "states": [str(path) for path in (args.prefix_dir, args.postpeak_dir)],
        "loads": loads,
        "operator": "matrix-free assembled D_ff^{-1} C_ff A_ff^{-1} B_ff",
        "history_branch": "previous accepted history with target-state active branch",
        "active_bound_tolerance": args.active_bound_tolerance,
        "active_kkt_tolerance": args.active_kkt_tolerance,
        "slow_mode_energy_coverages": coverages,
        "arnoldi": {
            "nev": args.nev,
            "tolerance": args.arnoldi_tolerance,
            "maxiter": args.arnoldi_maxiter,
            "ncv": args.ncv,
            "random_seed": args.seed,
        },
        "output_csv": str(csv_path.resolve()),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path


def main() -> None:
    """Run the configured spectrum diagnostic."""
    path = run(parse_args())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
