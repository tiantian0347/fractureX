#!/usr/bin/env python3
"""Construct a fixed-history full coupled KKT reference root for Model-0.

Purpose
-------
Solve the resolved Model-0 checkpoint at ``u=0.0898`` with a direct coupled
residual/Jacobian definition.  The previous accepted history field and phase
lower bound are held fixed, so the result is a reference root for the same
discrete KKT problem used by Reduced-NE.

Boundary
--------
This driver does not modify the physical continuation path.  It is an offline
reference-root construction used to validate solver equivalence and cost.

Usage
-----
PYTHONPATH=. python scripts/paper_solver/run_model0_resolved_coupled_reference.py \
    --output-dir ../results/phasefield_solver/model0_resolved_coupled_reference_0898_v2
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fracturex.analysis.coupled_newton_solver import (
    CoupledNewtonConfig,
    solve_coupled_newton,
)
from scripts.paper_solver.run_model0_fine_reference import (
    build_model0_resolved_solver,
)
from scripts.paper_solver.scan_model0_resolved_online_rate import (
    _load_checkpoint_arrays,
    require_results_directory,
)
from scripts.paper_solver.verify_slow_mode_fracturex import (
    FrozenStandardFEStepMap,
    _assemble_history_field_coupling_blocks,
    _coupled_residual_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT.parent / "results/phasefield_solver"
DEFAULT_PREVIOUS_STATE = (
    RESULTS_ROOT
    / "model0_resolved_transactional_postpeak_h0065/accepted_states/"
    "accepted_0001_load_0.08870000.npz"
)
DEFAULT_INITIAL_STATE = (
    RESULTS_ROOT
    / "model0_resolved_transactional_postpeak_h0065/accepted_states/"
    "accepted_0002_load_0.08980000.npz"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "model0_resolved_coupled_reference_0898_v2"


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
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


def _git_commit() -> str | None:
    """Return the FractureX commit used for the reference-root run."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(name: str) -> str:
    """Return one installed package version without adding a dependency."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _load_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load displacement, phase, and history arrays from one accepted state."""
    if not path.is_file():
        raise FileNotFoundError(f"state file does not exist: {path}")
    displacement, damage, history = _load_checkpoint_arrays(path)
    return displacement, damage, history


def _make_step_map(
    previous_state: Path,
    *,
    hmin: float,
    target_load: float,
) -> tuple[Any, FrozenStandardFEStepMap, dict[str, Any], dict[str, Any]]:
    """Build the fixed-history FE map and return material/mesh metadata."""
    main, material, mesh_stats, unused_nodes = build_model0_resolved_solver(
        hmin=hmin
    )
    main.load_restart_npz(str(previous_state))
    _, committed_damage, committed_history = _load_state(previous_state)
    step_map = FrozenStandardFEStepMap(
        main,
        load=target_load,
        committed_damage=committed_damage,
        committed_history=committed_history,
        phase_bound_solver="clip",
        phase_active_set_max_iterations=1000,
    )
    return main, step_map, material, {
        "mesh": mesh_stats,
        "unused_distmesh_nodes": int(unused_nodes),
    }


def _bounds(step_map: FrozenStandardFEStepMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-state lower/upper bounds and Dirichlet mask."""
    nu = step_map.displacement_size
    lower = np.full(step_map.full_state_size, -np.inf, dtype=np.float64)
    upper = np.full(step_map.full_state_size, np.inf, dtype=np.float64)
    lower[nu:] = step_map.damage_lower_bound
    upper[nu:] = step_map.damage_upper_bound
    return lower, upper, step_map.fixed_state_mask()


def _build_residual_jacobian_callback(step_map: FrozenStandardFEStepMap):
    """Return a direct FE residual and analytic four-block Jacobian callback."""

    def callback(state: np.ndarray):
        residual = step_map.assemble_coupled_residual(
            state, enforce_phase_box=False
        )
        if (
            step_map.last_displacement_matrix is None
            or step_map.last_phase_matrix is None
        ):
            raise RuntimeError("direct residual assembly did not capture diagonal blocks")
        jacobian_ud, jacobian_du = _assemble_history_field_coupling_blocks(step_map)
        return (
            residual,
            step_map.last_displacement_matrix.copy(),
            jacobian_ud,
            jacobian_du,
            step_map.last_phase_matrix.copy(),
        )

    return callback


def _parse_args() -> argparse.Namespace:
    """Parse explicit reference-root and Newton--GMRES controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmin", type=float, default=0.0065)
    parser.add_argument("--target-load", type=float, default=0.0898)
    parser.add_argument("--previous-state", type=Path, default=DEFAULT_PREVIOUS_STATE)
    parser.add_argument("--initial-state", type=Path, default=DEFAULT_INITIAL_STATE)
    parser.add_argument("--residual-atol", type=float, default=1.0e-10)
    parser.add_argument("--residual-rtol", type=float, default=1.0e-8)
    parser.add_argument("--gmres-rtol", type=float, default=1.0e-7)
    parser.add_argument("--gmres-max-iterations", type=int, default=80)
    parser.add_argument("--max-newton-iterations", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Construct and persist the fixed-history coupled reference root."""
    args = _parse_args()
    output_dir = require_results_directory(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_state = args.previous_state.resolve()
    initial_state_path = args.initial_state.resolve()
    displacement, damage, _ = _load_state(initial_state_path)
    initial_state = np.concatenate((displacement, damage))
    main_solver, step_map, material, mesh_info = _make_step_map(
        previous_state, hmin=args.hmin, target_load=args.target_load
    )
    lower, upper, fixed = _bounds(step_map)
    initial_metrics = _coupled_residual_metrics(step_map, initial_state)
    callback = _build_residual_jacobian_callback(step_map)
    result = solve_coupled_newton(
        callback,
        initial_state,
        displacement_size=step_map.displacement_size,
        lower_bound=lower,
        upper_bound=upper,
        fixed_mask=fixed,
        config=CoupledNewtonConfig(
            residual_atol=args.residual_atol,
            residual_rtol=args.residual_rtol,
            gmres_rtol=args.gmres_rtol,
            gmres_max_iterations=args.gmres_max_iterations,
            max_newton_iterations=args.max_newton_iterations,
        ),
    )
    final_metrics = _coupled_residual_metrics(step_map, result.state)
    final_displacement = result.state[: step_map.displacement_size]
    final_damage = result.state[step_map.displacement_size :]
    np.savez_compressed(
        output_dir / "reference_root.npz",
        displacement=final_displacement,
        damage=final_damage,
        history=np.asarray(step_map.committed_history, dtype=np.float64),
        load=float(args.target_load),
    )
    summary = {
        "status": "passed" if result.converged else "reference_root_not_converged",
        "target_load": float(args.target_load),
        "previous_load": float(previous_state.stem.split("load_")[-1]),
        "material": material,
        "mesh": mesh_info,
        "paths": {
            "previous_state": str(previous_state),
            "initial_state": str(initial_state_path),
            "reference_root": str(output_dir / "reference_root.npz"),
        },
        "history_policy": "fixed committed H from previous accepted state",
        "phase_lower_bound": "previous accepted damage",
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "newton": {
            "converged": result.converged,
            "termination_reason": result.termination_reason,
            "newton_iterations": result.newton_iterations,
            "gmres_iterations": result.gmres_iterations,
            "residual_jacobian_evaluations": result.residual_jacobian_evaluations,
            "preconditioner_factorizations": result.preconditioner_factorizations,
            "projected_residual_norms": result.projected_residual_norms,
            "krylov_residual_norms": result.krylov_residual_norms,
            "wall_time_seconds": result.wall_time_seconds,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "output_dir": str(output_dir),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fealpy": _package_version("fealpy"),
        "fracturex": _package_version("fracturex"),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "meta.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
