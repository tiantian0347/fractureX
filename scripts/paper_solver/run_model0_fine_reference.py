#!/usr/bin/env python3
"""Run an audited Model-0 path on a length-scale-resolved mesh.

Purpose
-------
Reproduce the boundary conditions, material parameters, and report loads from
``fracturex/cases/phase_field/model0_example.py`` while enforcing
``h_max < l0/2``.  Small continuation substeps are inserted only between the
31 report loads so the fine-grid crack branch remains in the converged basin.

Scope
-----
This script validates the physical standard-FE path.  T7 slow-space and
Reduced-NE diagnostics are evaluated separately from its converged states.
Every accepted continuation state is saved by default so diagnostics can
replay the exact immediate-previous-state transitions used by the path.

Usage
-----
python scripts/paper_solver/run_model0_fine_reference.py \
    --hmin 0.0065 \
    --continuation-step 0.0011 \
    --maxit 400 \
    --anderson-start-load 0.0854 \
    --output-dir results/phasefield_solver/model0_example_fine_reference_h0065
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT.parent / "results/phasefield_solver"
DEFAULT_OUTPUT_DIR = (
    RESULTS_ROOT / "model0_example_fine_reference_h0065"
)
REPORT_CHECKPOINT_PATTERN = re.compile(
    r"report_(?P<index>\d+)_load_(?P<load>[0-9.]+)\.npz$"
)


def require_results_directory(
    output_dir: Path, *, results_root: Path = RESULTS_ROOT
) -> Path:
    """Resolve a simulation directory and require it below ``results``.

    Parameters
    ----------
    output_dir : pathlib.Path
        Requested directory for simulation states and summaries.
    results_root : pathlib.Path
        Allowed result root; injectable for unit tests.

    Returns
    -------
    pathlib.Path
        Absolute validated output directory.

    Raises
    ------
    ValueError
        If ``output_dir`` lies outside ``results_root``.
    """
    resolved_output = output_dir.resolve()
    resolved_root = results_root.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"simulation outputs must be stored below {resolved_root}"
        ) from error
    return resolved_output


def build_report_loads() -> np.ndarray:
    """Return the 31 report loads used by ``model0_example.py``.

    Returns
    -------
    numpy.ndarray
        Strictly increasing float64 displacement values, shape ``(31,)``,
        ranging from ``0`` to ``0.125``.
    """
    return np.concatenate(
        (
            np.linspace(0.0, 70.0e-3, 6, dtype=np.float64),
            np.linspace(70.0e-3, 125.0e-3, 26, dtype=np.float64)[1:],
        )
    )


def _restore_state(main, snapshot: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Restore displacement, phase, and history after a rejected trial load."""
    from fealpy.backend import backend_manager as bm

    displacement, damage, history = snapshot
    main.uh[:] = bm.asarray(displacement, dtype=main.uh.dtype)
    main.d[:] = bm.asarray(damage, dtype=main.d.dtype)
    main.H = bm.asarray(history, dtype=main.d.dtype)
    main.pfcm.update_disp(main.uh)
    main.pfcm.update_phase(main.d)
    main.pfcm.update_historical_field(main.H)


def _restore_history_snapshot(main, history: np.ndarray) -> None:
    """Restore one committed history field before a staggered trial sweep."""
    from fealpy.backend import backend_manager as bm

    main.H = bm.asarray(history, dtype=main.d.dtype).copy()
    main.pfcm.update_historical_field(main.H)


def _solve_load(
    main,
    load: float,
    maxit: int,
    tolerance: float,
    damage_relaxation: float,
    anderson_depth: int,
) -> tuple[bool, int]:
    """Attempt one prescribed-displacement step with finite-state validation.

    Returns
    -------
    tuple[bool, int]
        Convergence flag and number of staggered iterations performed. A false
        return leaves state restoration to the caller.
    """
    from fealpy.backend import backend_manager as bm
    from fracturex.drivers.anderson_acceleration import AndersonAccelerator

    main._currt_force_value = float(load)
    committed_history = np.asarray(
        bm.to_numpy(main.H), dtype=np.float64
    ).copy()
    accelerator = (
        AndersonAccelerator(
            depth=anderson_depth,
            beta=1.0,
            omega=1.0,
            restart_patience=3,
            blowup_factor=2.0,
            tr_factor=20.0,
            restart_omega=1.6,
        )
        if anderson_depth > 0
        else None
    )
    reference_u = None
    reference_d = None
    for iteration in range(1, maxit + 1):
        # Every nonlinear trial starts from H_{n-1}. Intermediate staggered
        # iterates must not irreversibly modify the accepted load-step history.
        _restore_history_snapshot(main, committed_history)
        residual_u = float(np.asarray(bm.to_numpy(main.solve_displacement())))
        previous_damage = np.asarray(bm.to_numpy(main.d[:]), dtype=np.float64).copy()
        main.solve_phase_field()
        candidate_damage = np.asarray(bm.to_numpy(main.d[:]), dtype=np.float64)
        relaxed_damage = previous_damage + damage_relaxation * (
            candidate_damage - previous_damage
        )
        plain_damage = np.clip(
            np.maximum(previous_damage, relaxed_damage), 0.0, 1.0
        )
        fixed_point_residual = float(np.linalg.norm(plain_damage - previous_damage))
        if accelerator is None:
            next_damage = plain_damage
        else:
            accelerated_damage = accelerator.step(previous_damage, relaxed_damage)
            next_damage = np.clip(
                np.maximum(previous_damage, accelerated_damage), 0.0, 1.0
            )
        main.d[:] = bm.asarray(next_damage, dtype=main.d.dtype)
        main.pfcm.update_phase(main.d)
        displacement = np.asarray(bm.to_numpy(main.uh[:]), dtype=np.float64)
        damage = np.asarray(bm.to_numpy(main.d[:]), dtype=np.float64)
        history = np.asarray(bm.to_numpy(main.H), dtype=np.float64)
        if not (
            np.isfinite(residual_u)
            and np.isfinite(fixed_point_residual)
            and np.isfinite(displacement).all()
            and np.isfinite(damage).all()
            and np.isfinite(history).all()
        ):
            return False, iteration
        if iteration == 1:
            reference_u = max(abs(residual_u), np.finfo(np.float64).tiny)
            reference_d = max(fixed_point_residual, np.finfo(np.float64).tiny)
        error = max(
            residual_u / reference_u,
            fixed_point_residual / reference_d,
        )
        if error <= tolerance:
            return True, iteration
    return False, maxit


def _write_report_csv(
    csv_path: Path,
    report_loads: list[float],
    report_reactions: list[float],
    report_iterations: list[int],
    algorithm_stage: str,
) -> None:
    """Atomically persist all converged report points collected so far."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "step",
                "load",
                "residual_force_abs",
                "staggered_iterations",
                "algorithm_stage",
            ),
        )
        writer.writeheader()
        for step, (load, reaction, iterations) in enumerate(
            zip(report_loads, report_reactions, report_iterations)
        ):
            writer.writerow(
                {
                    "step": step,
                    "load": float(load),
                    "residual_force_abs": abs(float(reaction)),
                    "staggered_iterations": int(iterations),
                    "algorithm_stage": algorithm_stage,
                }
            )
    temporary_path.replace(csv_path)


def _read_resume_report_rows(
    csv_path: Path,
    checkpoint_path: Path,
) -> list[dict[str, str]]:
    """Read the report prefix ending at a named report checkpoint.

    ``restart_latest.npz`` uses the full supplied CSV. A checkpoint named
    ``report_<index>_load_<value>.npz`` truncates a longer CSV at its unique
    matching load so the committed state and reported path cannot be mixed.
    """
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("resume report CSV contains no completed rows")
    checkpoint_match = REPORT_CHECKPOINT_PATTERN.match(checkpoint_path.name)
    if checkpoint_match is None:
        return rows
    checkpoint_load = float(checkpoint_match.group("load"))
    matching_rows = [
        index
        for index, row in enumerate(rows)
        if np.isclose(float(row["load"]), checkpoint_load, rtol=0.0, atol=5.0e-5)
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            "resume report CSV does not uniquely contain the checkpoint load"
        )
    return rows[: matching_rows[0] + 1]


def _write_internal_state_csv(
    csv_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Atomically write the chronological accepted-state manifest.

    Parameters
    ----------
    csv_path : pathlib.Path
        Destination manifest path.
    rows : list[dict]
        Chronological accepted states. Loads must be finite and strictly
        increasing after the initial state. ``checkpoint`` paths identify
        restart NPZ files without modifying the supplied rows.

    Raises
    ------
    ValueError
        If rows are empty, indices are not consecutive, or loads are invalid.
    """
    if not rows:
        raise ValueError("internal-state manifest requires at least one row")
    indices = np.asarray([int(row["accepted_index"]) for row in rows], dtype=int)
    loads = np.asarray([float(row["load"]) for row in rows], dtype=np.float64)
    if not np.array_equal(indices, np.arange(indices.size)):
        raise ValueError("accepted-state indices must start at zero and be consecutive")
    if not np.isfinite(loads).all() or np.any(np.diff(loads) <= 0.0):
        raise ValueError("accepted-state loads must be finite and strictly increasing")

    fieldnames = (
        "accepted_index",
        "load",
        "previous_load",
        "step_size",
        "report_index",
        "is_report",
        "staggered_iterations",
        "damage_relaxation",
        "anderson_depth",
        "algorithm_stage",
        "checkpoint",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fieldnames} for row in rows)
    temporary_path.replace(csv_path)


def _save_internal_state(
    main,
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    load: float,
    previous_load: float | None,
    report_index: int,
    is_report: bool,
    iterations: int,
    damage_relaxation: float,
    anderson_depth: int,
    algorithm_stage: str,
) -> None:
    """Save one accepted continuation state and refresh its manifest.

    The restart contains displacement, phase field, and committed history at
    one accepted path state. Report states are saved after reaction
    re-equilibration because that is the state used by the next continuation
    step.
    """
    accepted_index = len(rows)
    checkpoint_dir = output_dir / "accepted_states"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / (
        f"accepted_{accepted_index:04d}_load_{load:.8f}.npz"
    )
    main.save_restart_npz(str(checkpoint))
    rows.append(
        {
            "accepted_index": accepted_index,
            "load": float(load),
            "previous_load": "" if previous_load is None else float(previous_load),
            "step_size": 0.0 if previous_load is None else float(load - previous_load),
            "report_index": int(report_index),
            "is_report": bool(is_report),
            "staggered_iterations": int(iterations),
            "damage_relaxation": float(damage_relaxation),
            "anderson_depth": int(anderson_depth),
            "algorithm_stage": algorithm_stage,
            "checkpoint": str(checkpoint.resolve()),
        }
    )
    _write_internal_state_csv(output_dir / "accepted_internal_states.csv", rows)


def _git_commit() -> str | None:
    """Return the current FractureX commit when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _equilibrate_and_read_reaction(main) -> tuple[float, float]:
    """Re-equilibrate displacement for the accepted damage and read reaction.

    ``MainSolve.solve_displacement`` evaluates ``_Rfu`` before applying its
    displacement correction. The first call therefore synchronizes the
    displacement with the accepted phase field; the second call reads the
    reaction at that synchronized state.

    Returns
    -------
    tuple[float, float]
        Absolute boundary reaction and the second displacement residual norm.
    """
    from fealpy.backend import backend_manager as bm

    main.solve_displacement()
    residual = float(np.asarray(bm.to_numpy(main.solve_displacement())))
    reaction = abs(float(np.asarray(bm.to_numpy(main._Rfu))))
    if not np.isfinite(reaction) or not np.isfinite(residual):
        raise RuntimeError("non-finite reaction during report-point re-equilibration")
    return reaction, residual


def build_model0_resolved_solver(
    *, hmin: float
) -> tuple[object, dict[str, float], dict[str, float], int]:
    """Build the audited length-scale-resolved Model-0 discretization.

    Parameters
    ----------
    hmin : float
        Positive DistMesh target size. The realized mesh must satisfy
        ``h_max < l0/2`` for ``l0=0.02``.

    Returns
    -------
    tuple
        Initialized ``MainSolve`` instance, material dictionary, mesh-size
        statistics, and the number of unused DistMesh nodes removed.

    Raises
    ------
    ValueError
        If ``hmin`` is not finite and positive.
    RuntimeError
        If the realized mesh does not resolve the phase-field length scale.

    Notes
    -----
    Geometry, material parameters, boundary conditions, polynomial order, and
    quadrature are identical to the fine physical-path experiment. Keeping
    this construction in one function prevents diagnostic replays from
    silently changing the discrete problem.
    """
    from fealpy.backend import backend_manager as bm
    from fealpy.mesh import TriangleMesh
    from fealpy.old.geometry.domain_2d import SquareWithCircleHoleDomain
    from fracturex.phasefield.main_solve import MainSolve
    from fracturex.utilfuc.phasefield_mesh import mesh_h_stats

    if not np.isfinite(hmin) or hmin <= 0.0:
        raise ValueError("hmin must be finite and positive")

    bm.set_backend("numpy")
    report_loads = build_report_loads()
    domain = SquareWithCircleHoleDomain(hmin=float(hmin))
    mesh = TriangleMesh.from_domain_distmesh(domain, maxit=100, ftype=bm.float64)
    raw_nodes = np.asarray(bm.to_numpy(mesh.entity("node")), dtype=np.float64)
    raw_cells = np.asarray(bm.to_numpy(mesh.entity("cell")), dtype=np.int64)
    used_nodes = np.zeros(raw_nodes.shape[0], dtype=bool)
    used_nodes[raw_cells.reshape(-1)] = True
    unused_node_count = int(np.count_nonzero(~used_nodes))
    if unused_node_count:
        old_to_new = np.full(raw_nodes.shape[0], -1, dtype=np.int64)
        old_to_new[used_nodes] = np.arange(np.count_nonzero(used_nodes))
        mesh = TriangleMesh(
            bm.asarray(raw_nodes[used_nodes], dtype=bm.float64),
            bm.asarray(old_to_new[raw_cells], dtype=bm.int64),
        )

    mesh_stats = {
        key: float(value) for key, value in mesh_h_stats(mesh).items()
    }
    length_scale = 0.02
    if mesh_stats["h_max"] >= 0.5 * length_scale:
        raise RuntimeError(
            "realized mesh is not length-scale resolved: "
            f"h_max={mesh_stats['h_max']:.8f}, l0/2={0.5 * length_scale:.8f}"
        )

    def on_top(points):
        """Select the loaded top edge."""
        return bm.abs(points[..., 1] - 1.0) < 1.0e-12

    def on_inner_circle(points):
        """Select the circular-hole boundary used by the production case."""
        return (
            bm.abs(
                (points[..., 0] - 0.5) ** 2
                + bm.abs(points[..., 1] - 0.5) ** 2
                - 0.04
            )
            < 0.001
        )

    material = {"E": 200.0, "nu": 0.2, "Gc": 1.0, "l0": length_scale}
    main = MainSolve(mesh=mesh, material_params=material, model_type="HybridModel")
    main.add_boundary_condition("force", "Dirichlet", on_top, report_loads, "y")
    main.add_boundary_condition("displacement", "Dirichlet", on_inner_circle, 0)
    main.add_boundary_condition("phase", "Dirichlet", on_inner_circle, 0)
    main._method = "lfem"
    main.initialize_settings(p=1)
    main._initialize_force_boundary()
    quadrature = mesh.quadrature_formula(int(main.q), "cell")
    _, quadrature_weights = quadrature.get_quadrature_points_and_weights()
    main.H = bm.zeros(
        (int(mesh.number_of_cells()), int(quadrature_weights.shape[0])),
        dtype=main.d.dtype,
    )
    main.pfcm.update_historical_field(main.H)
    main.set_linear_solver_options(method="direct")
    return main, material, mesh_stats, unused_node_count


def run_fine_reference(
    *,
    hmin: float,
    continuation_step: float,
    maxit: int,
    damage_relaxation: float,
    relaxation_start_load: float,
    anderson_depth: int,
    anderson_start_load: float,
    final_report_load: float,
    resume_from: Path | None,
    resume_report_csv: Path | None,
    output_dir: Path,
    save_internal_states: bool = True,
) -> Path:
    """Run the fine-grid standard FE path and export its report-point response.

    Parameters
    ----------
    hmin : float
        DistMesh target size. The realized ``h_max`` must be below ``l0/2``.
    continuation_step : float
        Maximum internal displacement increment between report loads.
    maxit : int
        Maximum MainSolve staggered iterations per expanded load.
    damage_relaxation : float
        Phase update factor used at and above ``relaxation_start_load``.
    relaxation_start_load : float
        Load at which phase under-relaxation is activated.
    anderson_depth : int
        Safeguarded Anderson window size; zero disables acceleration.
    anderson_start_load : float
        Last report load solved without Anderson. Acceleration is active only
        for strictly larger trial loads.
    final_report_load : float
        Last requested report load. It must be one of the original 31 values.
    resume_from : pathlib.Path or None
        Existing MainSolve restart file at a completed report load.
    resume_report_csv : pathlib.Path or None
        Partial report CSV matching ``resume_from``.
    output_dir : pathlib.Path
        Directory receiving ``residual_force_vs_displacement.csv``,
        ``final.vtu``, and ``meta.json``.
    save_internal_states : bool
        Save every accepted continuation state and its chronological manifest.
        Enabled by default for path-consistent solver diagnostics.

    Returns
    -------
    pathlib.Path
        Path to the report-point reaction CSV.

    Raises
    ------
    RuntimeError
        If the realized mesh violates ``h_max < l0/2``.
    """
    from fealpy.backend import backend_manager as bm
    if maxit <= 0:
        raise ValueError("maxit must be positive")
    if not 0.0 < damage_relaxation <= 1.0:
        raise ValueError("damage_relaxation must lie in (0, 1]")
    if anderson_depth < 0:
        raise ValueError("anderson_depth must be nonnegative")
    if not np.isfinite(anderson_start_load) or anderson_start_load < 0.0:
        raise ValueError("anderson_start_load must be finite and nonnegative")
    if not np.isfinite(final_report_load):
        raise ValueError("final_report_load must be finite")
    if (resume_from is None) != (resume_report_csv is None):
        raise ValueError("resume_from and resume_report_csv must be supplied together")
    output_dir = require_results_directory(output_dir)

    all_report_loads = build_report_loads()
    terminal_matches = np.flatnonzero(
        np.isclose(all_report_loads, final_report_load, rtol=0.0, atol=1.0e-12)
    )
    if terminal_matches.size != 1:
        raise ValueError("final_report_load must be one of the Model-0 report loads")
    report_loads = all_report_loads[: int(terminal_matches[0]) + 1]
    main, material, mesh_stats, unused_node_count = build_model0_resolved_solver(
        hmin=hmin
    )
    length_scale = float(material["l0"])

    csv_path = output_dir / "residual_force_vs_displacement.csv"
    if resume_from is None:
        completed_loads = [0.0]
        report_reactions = [0.0]
        report_iterations = [0]
        current_load = 0.0
        accepted_internal_count = 1
    else:
        main.load_restart_npz(str(resume_from))
        rows = _read_resume_report_rows(resume_report_csv, resume_from)
        completed_loads = [float(row["load"]) for row in rows]
        report_reactions = [float(row["residual_force_abs"]) for row in rows]
        report_iterations = [int(row["staggered_iterations"]) for row in rows]
        current_load = completed_loads[-1]
        expected_loads = report_loads[: len(completed_loads)]
        if not np.allclose(completed_loads, expected_loads, rtol=0.0, atol=1.0e-12):
            raise ValueError("resume report loads do not match the Model-0 load path")
        accepted_internal_count = len(completed_loads)
        if resume_from.name == "restart_latest.npz":
            progress_path = resume_from.parent / "progress.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                accepted_internal_count = int(
                    progress.get("accepted_internal_load_count", accepted_internal_count)
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    internal_rows: list[dict[str, object]] = []
    if save_internal_states:
        _save_internal_state(
            main,
            output_dir=output_dir,
            rows=internal_rows,
            load=current_load,
            previous_load=None,
            report_index=len(completed_loads) - 1,
            is_report=True,
            iterations=0,
            damage_relaxation=1.0,
            anderson_depth=0,
            algorithm_stage="resume_state" if resume_from is not None else "initial_state",
        )

    for report_index in range(len(completed_loads), len(report_loads)):
        report_load = float(report_loads[report_index])
        nominal_step = (
            float(report_load - current_load)
            if report_load <= 70.0e-3 + 1.0e-14
            else float(continuation_step)
        )
        step = nominal_step
        minimum_step = nominal_step / 64.0
        total_iterations = 0
        report_state_record: dict[str, object] | None = None
        while current_load < float(report_load) - 1.0e-14:
            trial_load = min(float(report_load), current_load + step)
            previous_load = current_load
            active_relaxation = (
                damage_relaxation
                if trial_load >= relaxation_start_load
                else 1.0
            )
            active_anderson_depth = (
                anderson_depth
                if trial_load > anderson_start_load + 1.0e-14
                else 0
            )
            snapshot = (
                np.asarray(bm.to_numpy(main.uh[:]), dtype=np.float64).copy(),
                np.asarray(bm.to_numpy(main.d[:]), dtype=np.float64).copy(),
                np.asarray(bm.to_numpy(main.H), dtype=np.float64).copy(),
            )
            converged, iterations = _solve_load(
                main,
                trial_load,
                maxit=maxit,
                tolerance=1.0e-5,
                damage_relaxation=active_relaxation,
                anderson_depth=active_anderson_depth,
            )
            if not converged:
                _restore_state(main, snapshot)
                step *= 0.5
                if step < minimum_step:
                    raise RuntimeError(
                        "fine-grid continuation failed: "
                        f"accepted_load={current_load:.8f}, "
                        f"target={report_load:.8f}, trial={trial_load:.8f}"
                    )
                continue
            current_load = trial_load
            accepted_internal_count += 1
            total_iterations += iterations
            algorithm_stage = (
                "safeguarded_anderson"
                if active_anderson_depth > 0
                else "plain_staggered"
            )
            if current_load < report_load - 1.0e-14:
                if save_internal_states:
                    _save_internal_state(
                        main,
                        output_dir=output_dir,
                        rows=internal_rows,
                        load=current_load,
                        previous_load=previous_load,
                        report_index=report_index,
                        is_report=False,
                        iterations=iterations,
                        damage_relaxation=active_relaxation,
                        anderson_depth=active_anderson_depth,
                        algorithm_stage=algorithm_stage,
                    )
            else:
                report_state_record = {
                    "previous_load": previous_load,
                    "iterations": iterations,
                    "damage_relaxation": active_relaxation,
                    "anderson_depth": active_anderson_depth,
                    "algorithm_stage": algorithm_stage,
                }
            step = min(nominal_step, 1.5 * step)
        reaction, equilibrium_residual = _equilibrate_and_read_reaction(main)
        if save_internal_states:
            if report_state_record is None:
                raise RuntimeError("missing accepted report-state record")
            _save_internal_state(
                main,
                output_dir=output_dir,
                rows=internal_rows,
                load=report_load,
                previous_load=float(report_state_record["previous_load"]),
                report_index=report_index,
                is_report=True,
                iterations=int(report_state_record["iterations"]),
                damage_relaxation=float(report_state_record["damage_relaxation"]),
                anderson_depth=int(report_state_record["anderson_depth"]),
                algorithm_stage=str(report_state_record["algorithm_stage"]),
            )
        report_reactions.append(reaction)
        report_iterations.append(total_iterations)
        max_damage = float(np.max(np.asarray(bm.to_numpy(main.d[:]), dtype=float)))
        print(
            f"report {report_index:02d}/{len(report_loads) - 1:02d} "
            f"load={report_load:.6f} "
            f"iterations={total_iterations} max_d={max_damage:.6f} "
            f"reaction={report_reactions[-1]:.6f} "
            f"equilibrium_residual={equilibrium_residual:.3e}",
            flush=True,
        )
        main.save_restart_npz(str(output_dir / "restart_latest.npz"))
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        main.save_restart_npz(
            str(checkpoint_dir / f"report_{report_index:02d}_load_{report_load:.4f}.npz")
        )
        _write_report_csv(
            csv_path,
            completed_loads + [float(report_load)],
            report_reactions,
            report_iterations,
            "standard_fe_stabilized_continuation",
        )
        completed_loads.append(float(report_load))
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "last_report_index": report_index,
                    "last_report_load": float(report_load),
                    "last_reaction": float(report_reactions[-1]),
                    "last_max_damage": max_damage,
                    "last_equilibrium_residual": equilibrium_residual,
                    "accepted_internal_load_count": accepted_internal_count,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_report_csv(
        csv_path,
        [float(load) for load in report_loads],
        report_reactions,
        report_iterations,
        "standard_fe_stabilized_continuation",
    )

    vtk_path = output_dir / "final.vtu"
    main._save_vtkfile(str(vtk_path))
    path_strategy = []
    if relaxation_start_load > float(report_loads[0]):
        path_strategy.append(
            f"plain staggered continuation below load {relaxation_start_load:g}"
        )
    if damage_relaxation < 1.0:
        path_strategy.append(
            "damage under-relaxation with factor "
            f"{damage_relaxation:g} from load {relaxation_start_load:g}"
        )
    else:
        path_strategy.append("unrelaxed phase-field updates")
    if anderson_depth > 0:
        path_strategy.append(
            "safeguarded Anderson after the converged report load "
            f"{anderson_start_load:g}"
        )
    path_strategy.append(
        "displacement re-equilibration before each reported reaction"
    )
    path_strategy.append(
        "transactional trial history reset to the previous accepted state "
        "before every staggered sweep"
    )

    meta = {
        "script": str(Path(__file__).resolve()),
        "command": shlex.join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "fealpy")
        },
        "source": "fracturex/cases/phase_field/model0_example.py",
        "algorithm": (
            "standard MainSolve with report-load continuation, safeguarded "
            "Anderson, and re-equilibrated reactions"
        ),
        "material": material,
        "hmin_target": float(hmin),
        "mesh": mesh_stats,
        "unused_distmesh_nodes_removed": unused_node_count,
        "length_scale_resolution": {
            "l0": length_scale,
            "l0_over_2": 0.5 * length_scale,
            "h_max_less_than_l0_over_2": True,
        },
        "report_load_count": int(report_loads.size),
        "expanded_load_count": int(accepted_internal_count),
        "saved_internal_state_count": int(len(internal_rows)),
        "internal_states_saved": bool(save_internal_states),
        "internal_state_manifest": (
            str((output_dir / "accepted_internal_states.csv").resolve())
            if save_internal_states
            else None
        ),
        "all_requested_report_loads_converged": True,
        "continuation_step_max": float(continuation_step),
        "damage_relaxation": float(damage_relaxation),
        "relaxation_start_load": float(relaxation_start_load),
        "anderson_depth": int(anderson_depth),
        "anderson_activation_after_report_load": float(anderson_start_load),
        "history_transaction": (
            "reset H to the previous accepted load-step snapshot before each "
            "staggered trial; commit only the converged trial history"
        ),
        "path_strategy": path_strategy,
        "maxit": int(maxit),
        "final_load": float(report_loads[-1]),
        "final_max_damage": float(np.max(np.asarray(bm.to_numpy(main.d[:]), dtype=float))),
        "output_dir": str(output_dir),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path


def parse_args() -> argparse.Namespace:
    """Parse reproducible fine-reference parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmin", type=float, default=0.0065)
    parser.add_argument("--continuation-step", type=float, default=0.0011)
    parser.add_argument("--maxit", type=int, default=400)
    parser.add_argument("--damage-relaxation", type=float, default=0.5)
    parser.add_argument("--relaxation-start-load", type=float, default=0.08)
    parser.add_argument("--anderson-depth", type=int, default=5)
    parser.add_argument("--anderson-start-load", type=float, default=0.0854)
    parser.add_argument("--final-report-load", type=float, default=0.125)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-report-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--save-internal-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save every accepted continuation state and its CSV manifest",
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured fine-grid reference experiment."""
    args = parse_args()
    path = run_fine_reference(
        hmin=args.hmin,
        continuation_step=args.continuation_step,
        maxit=args.maxit,
        damage_relaxation=args.damage_relaxation,
        relaxation_start_load=args.relaxation_start_load,
        anderson_depth=args.anderson_depth,
        anderson_start_load=args.anderson_start_load,
        final_report_load=args.final_report_load,
        resume_from=args.resume_from.resolve() if args.resume_from else None,
        resume_report_csv=(
            args.resume_report_csv.resolve() if args.resume_report_csv else None
        ),
        output_dir=args.output_dir.resolve(),
        save_internal_states=args.save_internal_states,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
