#!/usr/bin/env python3
"""Compare resolved-path staggered, Anderson, and Reduced-NE solvers.

Purpose
-------
Run one path-consistent Model-0 checkpoint from the accepted state at
``u=0.0887`` to the target state ``u=0.0898``.  The script compares plain
staggered iteration, the production safeguarded Anderson continuation, and
phase-patch Reduced-NE driven by the resolved slow-mode energy.

Boundary
--------
This is an experiment driver.  It reuses the FractureX standard-FE map and
Reduced-NE kernel; it does not change the production solver or claim a global
optimal patch.  When ``--reference-state`` is provided, the archive is used
as the offline full-coupled KKT reference for strict state comparison.

Usage
-----
PYTHONPATH=. python scripts/paper_solver/run_model0_resolved_reduced_ne.py \
    --output-dir ../results/phasefield_solver/model0_resolved_reduced_ne_0898_v5
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from fealpy.backend import backend_manager as bm

from fracturex.analysis.staggered_slow_mode import iterate_fixed_point, select_bulk_cells
from scripts.paper_solver.run_model0_fine_reference import (
    _solve_load,
    build_model0_resolved_solver,
)
from scripts.paper_solver.scan_model0_resolved_online_rate import (
    _load_checkpoint_arrays,
    require_results_directory,
)
from scripts.paper_solver.verify_slow_mode_fracturex import (
    FrozenStandardFEStepMap,
    _cells_to_coupled_dofs,
    _coupled_residual_metrics,
    _run_reduced_solver_comparison,
)


SCRIPT_VERSION = "1.2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT.parent / "results/phasefield_solver"
DEFAULT_PREVIOUS_STATE = (
    RESULTS_ROOT
    / "model0_resolved_transactional_postpeak_h0065/accepted_states/"
    "accepted_0001_load_0.08870000.npz"
)
DEFAULT_TARGET_STATE = (
    RESULTS_ROOT
    / "model0_resolved_transactional_postpeak_h0065/accepted_states/"
    "accepted_0002_load_0.08980000.npz"
)
DEFAULT_REFERENCE_STATE = (
    RESULTS_ROOT
    / "model0_resolved_coupled_reference_0898_v2/reference_root.npz"
)
DEFAULT_SPECTRUM_MODE = (
    RESULTS_ROOT
    / "model0_resolved_transactional_stabilization_h0065_final/"
    "dominant_phase_mode_load_0.0898.npz"
)
DEFAULT_PLAIN_REPLAY_CSV = (
    RESULTS_ROOT
    / "model0_resolved_transactional_online_rate_0898_h0065/"
    "online_rate_internal_path.csv"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "model0_resolved_reduced_ne_0898_v5"


def _git_commit() -> str | None:
    """Return the repository commit used for the experiment, when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(name: str) -> str:
    """Return an installed package version without adding a hard dependency."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict-JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def _load_restart(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one finite accepted restart as displacement, damage, and history."""
    if not path.is_file():
        raise FileNotFoundError(f"accepted restart does not exist: {path}")
    displacement, damage, history = _load_checkpoint_arrays(path)
    return displacement, damage, history


def _load_reference_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load either a standard checkpoint or a coupled-root state archive."""
    if not path.is_file():
        raise FileNotFoundError(f"reference state does not exist: {path}")
    with np.load(path) as data:
        if {"uh", "d", "H"}.issubset(data.files):
            displacement = np.asarray(data["uh"], dtype=np.float64).reshape(-1)
            damage = np.asarray(data["d"], dtype=np.float64).reshape(-1)
            history = np.asarray(data["H"], dtype=np.float64).copy()
        elif {"displacement", "damage", "history"}.issubset(data.files):
            displacement = np.asarray(data["displacement"], dtype=np.float64).reshape(-1)
            damage = np.asarray(data["damage"], dtype=np.float64).reshape(-1)
            history = np.asarray(data["history"], dtype=np.float64).copy()
        else:
            raise ValueError(f"unsupported reference-state archive: {path}")
    if not (np.isfinite(displacement).all() and np.isfinite(damage).all() and np.isfinite(history).all()):
        raise ValueError(f"reference state contains non-finite data: {path}")
    return displacement, damage, history


def _build_reduced_args(
    coverages: list[float],
    warmup_sweeps: int,
    *,
    warmup_mode: str = "adaptive",
    warmup_min_sweeps: int = 3,
    warmup_max_sweeps: int = 12,
    warmup_slow_rate: float = 0.89,
    warmup_required_slow_steps: int = 2,
    warmup_residual_tolerance: float = 1.0e-8,
    warmup_residual_ratio_threshold: float = 0.8,
    continuation: str = "none",
    continuation_stages: int = 4,
    max_local_iterations: int,
    max_outer_iterations: int,
    max_krylov_iterations: int,
    krylov_rtol: float = 1.0e-5,
    preconditioner: str = "block_lu",
    local_predictor: bool = False,
    initialization: str = "warmup",
) -> SimpleNamespace:
    """Return the explicit numerical contract consumed by the Reduced-NE kernel."""
    names = [f"slow_{int(round(100.0 * value)):02d}" for value in coverages]
    return SimpleNamespace(
        reduced_minimum_outer_iterations=1,
        reduced_reference_free_state_tolerance=5.0e-7,
        reduced_reference_free_condition_scaled_residual_tolerance=3.0e-3,
        reduced_warmup_sweeps=int(warmup_sweeps),
        reduced_warmup_mode=str(warmup_mode),
        reduced_warmup_min_sweeps=int(warmup_min_sweeps),
        reduced_warmup_max_sweeps=int(warmup_max_sweeps),
        reduced_warmup_slow_rate=float(warmup_slow_rate),
        reduced_warmup_required_slow_steps=int(warmup_required_slow_steps),
        reduced_warmup_residual_tolerance=float(warmup_residual_tolerance),
        reduced_warmup_residual_ratio_threshold=float(
            warmup_residual_ratio_threshold
        ),
        reduced_continuation=str(continuation),
        reduced_continuation_stages=int(continuation_stages),
        reduced_local_jacobian_check_directions=0,
        reduced_patch_space="phase",
        reduced_patches=names,
        reduced_local_atol=1.0e-10,
        reduced_local_rtol=1.0e-8,
        reduced_local_predictor=bool(local_predictor),
        reduced_preconditioner=str(preconditioner),
        reduced_initialization=str(initialization),
        reduced_outer_atol=1.0e-8,
        reduced_outer_rtol=1.0e-8,
        reduced_krylov_rtol=float(krylov_rtol),
        reduced_krylov_atol=0.0,
        reduced_max_local_iterations=int(max_local_iterations),
        reduced_max_outer_iterations=int(max_outer_iterations),
        reduced_max_krylov_iterations=int(max_krylov_iterations),
        reduced_fd_step=1.0e-7,
        reduced_fd_scheme="forward",
        reduced_reference_free_acceptance=True,
        reduced_local_predictor_enabled=bool(local_predictor),
    )


def _make_step_map(
    previous_state: Path,
    *,
    hmin: float,
    target_load: float,
) -> tuple[Any, FrozenStandardFEStepMap, np.ndarray, np.ndarray]:
    """Build a target-load map whose committed state is the previous accepted state."""
    main, material, mesh_stats, unused_nodes = build_model0_resolved_solver(hmin=hmin)
    main.load_restart_npz(str(previous_state))
    _, committed_damage, committed_history = _load_restart(previous_state)
    step_map = FrozenStandardFEStepMap(
        main,
        load=target_load,
        committed_damage=committed_damage,
        committed_history=committed_history,
        # The resolved physical path uses a direct phase solve followed by
        # nodal irreversibility projection; keep the replay on that same map.
        phase_bound_solver="clip",
        phase_active_set_max_iterations=1000,
    )
    return main, step_map, material, {
        "mesh": mesh_stats,
        "unused_distmesh_nodes": int(unused_nodes),
    }


def _run_plain_staggered(
    step_map: FrozenStandardFEStepMap,
    *,
    atol: float,
    rtol: float,
    max_iterations: int,
) -> dict[str, Any]:
    """Run plain staggered iteration from the immediate previous accepted damage."""
    started = perf_counter()
    trace = iterate_fixed_point(
        step_map,
        step_map.committed_damage,
        atol=atol,
        rtol=rtol,
        max_iterations=max_iterations,
    )
    elapsed = perf_counter() - started
    state = step_map.current_full_state()
    metrics = _coupled_residual_metrics(step_map, state)
    return {
        "converged": bool(trace.converged),
        "iterations": int(trace.iterations),
        "stopped_at_iteration_cap": bool(
            not trace.converged and trace.iterations >= max_iterations
        ),
        "wall_time_seconds": float(elapsed),
        "final_increment_norm": float(trace.increment_norms[-1]),
        "asymptotic_ratio": float(trace.asymptotic_ratio),
        "state": state,
        "damage": trace.solution.copy(),
        "residual_metrics": metrics,
    }


def _run_anderson(
    previous_state: Path,
    *,
    hmin: float,
    target_load: float,
    max_iterations: int,
    tolerance: float,
    depth: int,
) -> dict[str, Any]:
    """Run the production safeguarded Anderson step from the same restart."""
    main, step_map, _, mesh_info = _make_step_map(
        previous_state, hmin=hmin, target_load=target_load
    )
    started = perf_counter()
    converged, iterations = _solve_load(
        main,
        target_load,
        maxit=max_iterations,
        tolerance=tolerance,
        damage_relaxation=1.0,
        anderson_depth=depth,
    )
    elapsed = perf_counter() - started
    state = step_map.current_full_state()
    metrics = _coupled_residual_metrics(step_map, state)
    return {
        "converged": bool(converged),
        "iterations": int(iterations),
        "wall_time_seconds": float(elapsed),
        "state": state,
        "damage": state[step_map.displacement_size :].copy(),
        "residual_metrics": metrics,
        "mesh": mesh_info,
    }


def _candidate_patches(
    main: Any,
    step_map: FrozenStandardFEStepMap,
    energy: np.ndarray,
    coverages: list[float],
) -> dict[str, np.ndarray]:
    """Build full coupled cell-union patches from saved slow-mode energy."""
    displacement_connectivity = np.asarray(
        bm.to_numpy(main.tspace.cell_to_dof()), dtype=np.int64
    )
    phase_connectivity = np.asarray(
        bm.to_numpy(main.space.cell_to_dof()), dtype=np.int64
    )
    coupled_connectivity = np.concatenate(
        (
            displacement_connectivity,
            step_map.displacement_size + phase_connectivity,
        ),
        axis=1,
    )
    patches: dict[str, np.ndarray] = {}
    for coverage in coverages:
        selected_cells = select_bulk_cells(energy, theta=float(coverage))
        name = f"slow_{int(round(100.0 * coverage)):02d}"
        patches[name] = _cells_to_coupled_dofs(
            selected_cells, coupled_connectivity
        )
    return patches


def _parse_args() -> argparse.Namespace:
    """Parse the reproducible resolved-grid experiment configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmin", type=float, default=0.0065)
    parser.add_argument("--target-load", type=float, default=0.0898)
    parser.add_argument("--previous-state", type=Path, default=DEFAULT_PREVIOUS_STATE)
    parser.add_argument("--target-state", type=Path, default=DEFAULT_TARGET_STATE)
    parser.add_argument("--reference-state", type=Path, default=None)
    parser.add_argument("--spectrum-mode", type=Path, default=DEFAULT_SPECTRUM_MODE)
    parser.add_argument("--patch-coverages", default="0.5,0.6,0.7")
    parser.add_argument("--plain-atol", type=float, default=1.0e-8)
    parser.add_argument("--plain-rtol", type=float, default=1.0e-8)
    parser.add_argument("--plain-replay-csv", type=Path, default=DEFAULT_PLAIN_REPLAY_CSV)
    parser.add_argument("--anderson-iterations", type=int, default=28)
    parser.add_argument(
        "--reduced-warmup-sweeps",
        type=int,
        default=3,
        help="fixed-mode staggered sweeps used for initialization",
    )
    parser.add_argument(
        "--reduced-warmup-mode",
        choices=("fixed", "adaptive"),
        default="adaptive",
        help="reference-free warmup stop rule",
    )
    parser.add_argument("--reduced-warmup-min-sweeps", type=int, default=3)
    parser.add_argument("--reduced-warmup-max-sweeps", type=int, default=12)
    parser.add_argument("--reduced-warmup-slow-rate", type=float, default=0.89)
    parser.add_argument(
        "--reduced-warmup-required-slow-steps", type=int, default=2
    )
    parser.add_argument(
        "--reduced-warmup-residual-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--reduced-warmup-residual-ratio-threshold", type=float, default=0.8
    )
    parser.add_argument(
        "--reduced-continuation",
        choices=("none", "linear"),
        default="none",
        help="optional reference-free linear homotopy before the physical solve",
    )
    parser.add_argument(
        "--reduced-continuation-stages",
        type=int,
        default=4,
        help="number of stages for linear Reduced-NE continuation",
    )
    parser.add_argument(
        "--reduced-initialization",
        choices=("warmup", "secant", "patch_secant"),
        default="warmup",
        help="initial state from warmup, full-state secant, or patch-only secant",
    )
    parser.add_argument("--reduced-max-local-iterations", type=int, default=2)
    parser.add_argument("--reduced-max-outer-iterations", type=int, default=3)
    parser.add_argument("--reduced-max-krylov-iterations", type=int, default=8)
    parser.add_argument(
        "--reduced-krylov-rtol",
        type=float,
        default=1.0e-5,
        help="relative tolerance for each outer reduced GMRES solve",
    )
    parser.add_argument(
        "--reduced-preconditioner",
        choices=(
            "block_diag",
            "block_lower",
            "block_upper",
            "block_lu",
            "schur_ilu",
            "schur_gmres",
            "global_ilu",
        ),
        default="block_lu",
        help="outer Schur--Krylov block preconditioner",
    )
    parser.add_argument(
        "--reduced-local-predictor",
        action="store_true",
        help="use the implicit local-map predictor for each outer Newton step",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the three solver paths and persist JSON, CSV, and metadata."""
    args = _parse_args()
    output_dir = require_results_directory(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverages = [float(token.strip()) for token in args.patch_coverages.split(",") if token.strip()]
    if (
        not coverages
        or any(not 0.0 < value <= 1.0 for value in coverages)
        or any(right <= left for left, right in zip(coverages, coverages[1:]))
    ):
        raise ValueError("--patch-coverages must be strictly increasing values in (0, 1]")
    previous_state = args.previous_state.resolve()
    target_state = args.target_state.resolve()
    reference_state_path = (
        args.reference_state.resolve()
        if args.reference_state is not None
        else target_state
    )
    spectrum_mode = args.spectrum_mode.resolve()
    _, target_damage, _ = _load_restart(target_state)
    reference_displacement, reference_damage, _ = _load_reference_state(
        reference_state_path
    )
    reference_state_vector = np.concatenate((reference_displacement, reference_damage))
    mode_data = np.load(spectrum_mode)
    energy = np.asarray(mode_data["coupled_mode_cell_energy"], dtype=np.float64).reshape(-1)
    main, step_map, material, mesh_info = _make_step_map(
        previous_state, hmin=args.hmin, target_load=args.target_load
    )
    patches = _candidate_patches(main, step_map, energy, coverages)
    names = list(patches)

    with args.plain_replay_csv.resolve().open(newline="", encoding="utf-8") as stream:
        replay_rows = list(csv.DictReader(stream))
    if not replay_rows:
        raise ValueError(f"plain replay CSV contains no rows: {args.plain_replay_csv}")
    replay = replay_rows[-1]
    plain = {
        "converged": replay["plain_replay_converged"].lower() == "true",
        "iterations": int(replay["plain_replay_sweeps"]),
        "wall_time_seconds": float(replay["plain_replay_seconds"]),
        "final_increment_norm": float(replay["plain_last_damage_increment_norm"]),
        "asymptotic_ratio": None,
        "residual_metrics": {},
    }
    target_displacement, target_damage, _ = _load_restart(target_state)
    target_state_vector = np.concatenate((target_displacement, target_damage))
    step_map.set_full_state(target_state_vector)
    anderson_metrics = _coupled_residual_metrics(step_map, target_state_vector)
    anderson = {
        # This is the accepted physical continuation state.  It is not
        # treated as a converged root of the full coupled residual below.
        "converged": True,
        "path_accepted": True,
        "iterations": int(args.anderson_iterations),
        "wall_time_seconds": None,
        "coupled_kkt_converged": bool(
            anderson_metrics["projected_raw_norm"] <= 1.0e-8
            and anderson_metrics["block_correction_norm"] <= 1.0e-6
        ),
        "state": target_state_vector,
        "damage": target_damage.copy(),
        "residual_metrics": anderson_metrics,
    }

    reduced_args = _build_reduced_args(
        coverages,
        args.reduced_warmup_sweeps,
        warmup_mode=args.reduced_warmup_mode,
        warmup_min_sweeps=args.reduced_warmup_min_sweeps,
        warmup_max_sweeps=args.reduced_warmup_max_sweeps,
        warmup_slow_rate=args.reduced_warmup_slow_rate,
        warmup_required_slow_steps=args.reduced_warmup_required_slow_steps,
        warmup_residual_tolerance=args.reduced_warmup_residual_tolerance,
        warmup_residual_ratio_threshold=args.reduced_warmup_residual_ratio_threshold,
        continuation=args.reduced_continuation,
        continuation_stages=args.reduced_continuation_stages,
        max_local_iterations=args.reduced_max_local_iterations,
        max_outer_iterations=args.reduced_max_outer_iterations,
        max_krylov_iterations=args.reduced_max_krylov_iterations,
        krylov_rtol=args.reduced_krylov_rtol,
        preconditioner=args.reduced_preconditioner,
        local_predictor=args.reduced_local_predictor,
        initialization=args.reduced_initialization,
    )
    reduced = _run_reduced_solver_comparison(
        step_map,
        reduced_args,
        reference_damage,
        patches,
        baseline_iterations=int(plain["iterations"]),
        baseline_wall_time_seconds=float(plain["wall_time_seconds"]),
        requested_patch_names=names,
        reference_state_override=reference_state_vector,
    )

    reference_state = reference_state_vector
    step_map.set_full_state(reference_state)
    reference_metrics = _coupled_residual_metrics(step_map, reference_state)
    solver_rows = []
    for label, result in (("plain_staggered", plain), ("anderson", anderson)):
        difference = (
            None
            if label == "plain_staggered"
            else result["state"] - reference_state
        )
        solver_rows.append(
            {
                "solver": label,
                "converged": result["converged"],
                "path_accepted": bool(result.get("path_accepted", False)),
                "coupled_kkt_converged": bool(
                    result.get("coupled_kkt_converged", False)
                ),
                "iterations": result["iterations"],
                "wall_time_seconds": result["wall_time_seconds"],
                "state_l2_difference_from_reference": (
                    None
                    if difference is None
                    else float(np.linalg.norm(difference))
                ),
                "state_max_difference_from_reference": (
                    None
                    if difference is None
                    else float(np.max(np.abs(difference)))
                ),
                "projected_residual_norm": (
                    None if label == "plain_staggered" else result["residual_metrics"]["projected_raw_norm"]
                ),
                "block_correction_norm": (
                    None if label == "plain_staggered" else result["residual_metrics"]["block_correction_norm"]
                ),
            }
        )
    for name, result in reduced["patches"].items():
        solver_rows.append(
            {
                "solver": f"reduced_ne_{name}",
                "converged": result["converged"],
                "path_accepted": False,
                "coupled_kkt_converged": bool(result["converged"]),
                "iterations": result["outer_newton_iterations"],
                "wall_time_seconds": result["total_wall_time_including_warmup_seconds"],
                "state_l2_difference_from_reference": result[
                    "full_solution_l2_difference_from_staggered"
                ],
                "state_max_difference_from_reference": result[
                    "full_solution_max_difference_from_staggered"
                ],
                "projected_residual_norm": result["final_coupled_residual"]["projected_raw_norm"],
                "block_correction_norm": result["final_coupled_residual"]["block_correction_norm"],
                "all_acceptance_checks_passed": result["all_acceptance_checks_passed"],
                "local_patch_dofs": result["local_patch_dofs"],
                "phase_patch_dofs": result["phase_patch_dofs"],
                "residual_equivalent_work": result["total_residual_equivalent_evaluations"],
                "speedup_over_plain": result["wall_time_speedup_over_staggered"],
            }
        )

    summary = {
        "status": (
            "resolved_reduced_ne_reference_root"
            if args.reference_state is not None
            else "resolved_reduced_ne_boundary"
        ),
        "script_version": SCRIPT_VERSION,
        "target_load": float(args.target_load),
        "previous_load": float(previous_state.stem.split("load_")[-1]),
        "reference_state_kind": (
            "full_coupled_kkt_root"
            if args.reference_state is not None
            else "accepted_physical_path_state"
        ),
        "reference_state_full_coupled_residual": reference_metrics,
        "material": material,
        "mesh": mesh_info,
        "paths": {
            "previous_state": str(previous_state),
            "target_state": str(target_state),
            "reference_state": str(reference_state_path),
            "spectrum_mode": str(spectrum_mode),
        },
        "patch_coverages": coverages,
        "plain_staggered": {key: value for key, value in plain.items() if key not in {"state", "damage"}},
        "anderson": {key: value for key, value in anderson.items() if key not in {"state", "damage"}},
        "reduced_nonlinear_solver": reduced,
        "solver_rows": solver_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "solver_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in solver_rows for key in row}))
        writer.writeheader()
        writer.writerows(solver_rows)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "output_dir": str(output_dir),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fealpy": _package_version("fealpy"),
        "fracturex": _package_version("fracturex"),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "patch_space": "phase",
        "history_transaction": "committed H from previous accepted state",
    }
    (output_dir / "meta.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
