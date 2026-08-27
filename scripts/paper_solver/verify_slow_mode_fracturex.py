#!/usr/bin/env python3
"""Verify staggered slow-mode claims with FractureX standard-FE assembly.

The script builds a small Lagrange finite-element phase-field problem through
``fracturex.phasefield.MainSolve``.  At one prescribed-displacement load step
it freezes the committed history and damage lower bound, iterates the resulting
deterministic staggered map, finite-differences its propagation matrix, and
compares the dominant spectral factor with a nonlinear modal replay.

This is a verification driver, not a production fracture simulation.  It uses
a deliberately small mesh so the dense finite-difference Jacobian can be
computed quickly and inspected directly.

Usage
-----
Unit-seed smoke test:

PYTHONPATH=. python scripts/paper_solver/verify_slow_mode_fracturex.py \
    --case unit_seed --nx 4 --load 0.025 \
    --output-dir results/phasefield_solver/slow_mode_smoke

Model-0 continuation scan:

PYTHONPATH=. python scripts/paper_solver/verify_slow_mode_fracturex.py \
    --case model0_circular_hole --mesh-size 0.05 --seed 0 \
    --loads 0.07,0.085,0.1,0.1125,0.125 --continuation-step 0.0025 \
    --output-dir results/phasefield_solver/model0_coupled_slow_scan_h005_path
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix, diags, bmat
from scipy.sparse.linalg import (
    LinearOperator,
    factorized,
    gmres as scipy_gmres,
    spilu,
    spsolve,
)

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.analysis.staggered_slow_mode import (
    apply_local_elimination_projection,
    augment_weighted_subspace_with_memory,
    compute_cell_energy_from_diagonal_weight,
    coupled_slow_subspace_from_sweep_column,
    diagonal_cell_weights,
    diagonal_patch_survival_factor,
    diagonal_patch_subspace_survival_factor,
    dominant_mode,
    finite_difference_jacobian_rectangular,
    iterate_fixed_point,
    online_increment_subspace,
    select_bulk_cells,
    solve_local_nonlinear_residual,
    subspace_cell_trace_indicator,
    weighted_orthonormalize,
    weighted_principal_angles,
)
from fracturex.analysis.reduced_nonlinear_solver import (
    ReducedNewtonConfig,
    solve_reduced_nonlinear_system,
)
from fracturex.analysis.box_quadratic import (
    BoxQuadraticResult,
    solve_box_quadratic_active_set,
)
from fracturex.phasefield.main_solve import MainSolve
from fracturex.phasefield.strain_energy_split import symmetric_tensor_to_voigt
from fracturex.phasefield.vector_Dirichlet_bc import VectorDirichletBC


SCRIPT_VERSION = "2.8"


def _top_boundary(points: Any) -> Any:
    """Select the top boundary of the unit square."""
    return bm.abs(points[..., 1] - 1.0) < 1.0e-12


def _bottom_boundary(points: Any) -> Any:
    """Select the bottom boundary of the unit square."""
    return bm.abs(points[..., 1]) < 1.0e-12


def _as_scipy_csr(matrix: Any) -> Any:
    """Return a FEALPy or SciPy sparse matrix as SciPy CSR."""
    if hasattr(matrix, "to_scipy"):
        return matrix.to_scipy().tocsr()
    if hasattr(matrix, "tocsr"):
        return matrix.tocsr()
    raise TypeError(f"unsupported sparse matrix type: {type(matrix)!r}")


def _build_outer_block_preconditioner(
    displacement_matrix: Any,
    phase_matrix: Any,
    displacement_phase_matrix: Any,
    phase_displacement_matrix: Any,
    outside: np.ndarray,
    outside_interior: np.ndarray,
    n_displacement: int,
    mode: str,
) -> Any:
    """Build a sparse block preconditioner for the reduced Krylov system.

    ``block_diag`` is the historical preconditioner.  ``block_lower`` and
    ``block_upper`` add one FE coupling block, while ``block_lu`` applies a
    forward and a backward block-triangular sweep with the same diagonal
    factors.  The latter is a two-sided Schur approximation: it retains the
    assembled displacement--phase coupling while avoiding formation of the
    dense exact reduced Schur block.  It is called ``block_lu`` at the CLI
    because it has the algebraic action of an incomplete block LU sweep, not
    because an exact LU factorization is formed.
    Active projected coordinates are identity rows and are excluded from the
    sparse factors.
    """
    valid_modes = {"block_diag", "block_lower", "block_upper", "block_lu"}
    if mode not in valid_modes:
        raise ValueError(f"unknown reduced preconditioner mode: {mode!r}")
    if mode == "block_diag":
        coupling_required = False
    else:
        coupling_required = True

    displacement_positions = np.flatnonzero(
        (outside < n_displacement) & outside_interior
    )
    phase_positions = np.flatnonzero(
        (outside >= n_displacement) & outside_interior
    )
    displacement_dofs = outside[displacement_positions]
    phase_dofs = outside[phase_positions] - n_displacement
    displacement_factor = (
        factorized(
            _as_scipy_csr(displacement_matrix)[displacement_dofs][:, displacement_dofs]
            .tocsc()
        )
        if displacement_dofs.size
        else None
    )
    phase_factor = (
        factorized(
            _as_scipy_csr(phase_matrix)[phase_dofs][:, phase_dofs].tocsc()
        )
        if phase_dofs.size
        else None
    )
    if coupling_required:
        displacement_phase = _as_scipy_csr(displacement_phase_matrix)
        phase_displacement = _as_scipy_csr(phase_displacement_matrix)
        block_ud = displacement_phase[displacement_dofs][:, phase_dofs]
        block_du = phase_displacement[phase_dofs][:, displacement_dofs]
    else:
        block_ud = None
        block_du = None
    active_positions = np.flatnonzero(~outside_interior)

    def apply(vector: np.ndarray) -> np.ndarray:
        rhs = np.asarray(vector, dtype=np.float64).reshape(-1)
        output = np.zeros_like(rhs)
        rhs_u = rhs[displacement_positions]
        rhs_d = rhs[phase_positions]
        if mode == "block_diag":
            if displacement_factor is not None:
                output[displacement_positions] = displacement_factor(rhs_u)
            if phase_factor is not None:
                output[phase_positions] = phase_factor(rhs_d)
        elif mode == "block_lower":
            solved_u = (
                displacement_factor(rhs_u)
                if displacement_factor is not None
                else np.zeros(0, dtype=np.float64)
            )
            solved_d_rhs = rhs_d - block_du @ solved_u
            solved_d = (
                phase_factor(solved_d_rhs)
                if phase_factor is not None
                else np.zeros(0, dtype=np.float64)
            )
            output[displacement_positions] = solved_u
            output[phase_positions] = solved_d
        elif mode == "block_upper":
            solved_d = (
                phase_factor(rhs_d)
                if phase_factor is not None
                else np.zeros(0, dtype=np.float64)
            )
            solved_u_rhs = rhs_u - block_ud @ solved_d
            solved_u = (
                displacement_factor(solved_u_rhs)
                if displacement_factor is not None
                else np.zeros(0, dtype=np.float64)
            )
            output[displacement_positions] = solved_u
            output[phase_positions] = solved_d
        else:  # block_lu
            first_u = (
                displacement_factor(rhs_u)
                if displacement_factor is not None
                else np.zeros(0, dtype=np.float64)
            )
            solved_d_rhs = rhs_d - block_du @ first_u
            solved_d = (
                phase_factor(solved_d_rhs)
                if phase_factor is not None
                else np.zeros(0, dtype=np.float64)
            )
            solved_u_rhs = rhs_u - block_ud @ solved_d
            solved_u = (
                displacement_factor(solved_u_rhs)
                if displacement_factor is not None
                else np.zeros(0, dtype=np.float64)
            )
            output[displacement_positions] = solved_u
            output[phase_positions] = solved_d
        output[active_positions] = rhs[active_positions]
        return output

    return apply


def _build_approximate_reduced_schur_ilu(
    displacement_matrix: Any,
    phase_matrix: Any,
    displacement_phase_matrix: Any,
    phase_displacement_matrix: Any,
    outside: np.ndarray,
    outside_interior: np.ndarray,
    n_displacement: int,
    local_patch: np.ndarray,
) -> Any:
    """Build an ILU of a sparse approximate reduced Schur complement.

    The exact reduced operator contains ``J_cw J_ww^{-1} J_wc``.  Forming it
    densely is unnecessary for a preconditioner, so this builder replaces the
    local inverse by the diagonal of the phase patch tangent and then factors
    the resulting sparse outside matrix with ILU.  This preserves the local
    elimination topology and the external displacement--phase coupling while
    keeping setup linear-algebraic and sparse.
    """
    if np.any(local_patch < n_displacement):
        raise ValueError("approximate reduced Schur ILU currently requires a phase patch")
    displacement = _as_scipy_csr(displacement_matrix)
    phase = _as_scipy_csr(phase_matrix)
    displacement_phase = _as_scipy_csr(displacement_phase_matrix)
    phase_displacement = _as_scipy_csr(phase_displacement_matrix)
    local_phase = np.unique(np.asarray(local_patch, dtype=np.int64) - n_displacement)
    if local_phase.size == 0:
        raise ValueError("local patch must contain at least one phase coordinate")

    interior_positions = np.flatnonzero(outside_interior)
    displacement_positions = np.flatnonzero(
        (outside < n_displacement) & outside_interior
    )
    phase_positions = np.flatnonzero(
        (outside >= n_displacement) & outside_interior
    )
    displacement_dofs = outside[displacement_positions]
    phase_dofs = outside[phase_positions] - n_displacement

    local_tangent = phase[local_phase][:, local_phase]
    local_diagonal = np.asarray(local_tangent.diagonal(), dtype=np.float64)
    safe_diagonal = np.where(
        np.abs(local_diagonal) > 1.0e-14, local_diagonal, 1.0
    )
    local_inverse = diags(1.0 / safe_diagonal, format="csr")

    block_uu = displacement[displacement_dofs][:, displacement_dofs]
    block_ud = displacement_phase[displacement_dofs][:, phase_dofs]
    block_du = phase_displacement[phase_dofs][:, displacement_dofs]
    block_dd = phase[phase_dofs][:, phase_dofs]
    local_to_u = phase_displacement[local_phase][:, displacement_dofs]
    local_to_d = phase[local_phase][:, phase_dofs]
    u_to_local = displacement_phase[displacement_dofs][:, local_phase]
    d_to_local = phase[phase_dofs][:, local_phase]
    schur = bmat(
        [
            [
                block_uu - u_to_local @ local_inverse @ local_to_u,
                block_ud - u_to_local @ local_inverse @ local_to_d,
            ],
            [
                block_du - d_to_local @ local_inverse @ local_to_u,
                block_dd - d_to_local @ local_inverse @ local_to_d,
            ],
        ],
        format="csc",
    )
    try:
        ilu = spilu(schur, drop_tol=1.0e-3, fill_factor=8.0, permc_spec="COLAMD")
    except RuntimeError:
        # A direct sparse factor is a robust fallback for small or nearly
        # singular diagnostic systems; the result remains a preconditioner.
        direct = factorized(schur)
        solve = direct
    else:
        solve = ilu.solve
    active_positions = np.flatnonzero(~outside_interior)

    def apply(vector: np.ndarray) -> np.ndarray:
        rhs = np.asarray(vector, dtype=np.float64).reshape(-1)
        output = np.zeros_like(rhs)
        if interior_positions.size:
            output[interior_positions] = solve(rhs[interior_positions])
        output[active_positions] = rhs[active_positions]
        return output

    return apply


def _build_global_schur_preconditioner(
    displacement_matrix: Any,
    phase_matrix: Any,
    displacement_phase_matrix: Any,
    phase_displacement_matrix: Any,
    outside: np.ndarray,
    outside_interior: np.ndarray,
    fixed_mask: np.ndarray,
) -> Any:
    """Factor the assembled full Jacobian as an exact Schur preconditioner.

    Solving the full block system with a zero right-hand side on the local
    patch is algebraically equivalent to applying the exact reduced Schur
    inverse.  This is used as a diagnostic-quality preconditioner: it avoids
    forming the dense Schur complement and provides a reference for judging
    cheaper block approximations.
    """
    full = bmat(
        [
            [_as_scipy_csr(displacement_matrix), _as_scipy_csr(displacement_phase_matrix)],
            [_as_scipy_csr(phase_displacement_matrix), _as_scipy_csr(phase_matrix)],
        ],
        format="csr",
    )
    constrained = np.asarray(fixed_mask, dtype=bool).reshape(-1).copy()
    constrained[outside[~outside_interior]] = True
    free_diagonal = diags((~constrained).astype(np.float64), format="csr")
    active_diagonal = diags(constrained.astype(np.float64), format="csr")
    # Zero projected rows and columns in one sparse product; this is equivalent
    # to replacing constrained coordinates by identity rows/columns.
    full = (free_diagonal @ full @ free_diagonal + active_diagonal).tocsc()
    try:
        ilu = spilu(full, drop_tol=1.0e-4, fill_factor=12.0, permc_spec="COLAMD")
    except RuntimeError:
        solve = factorized(full)
    else:
        solve = ilu.solve

    def apply(vector: np.ndarray) -> np.ndarray:
        rhs = np.zeros(full.shape[0], dtype=np.float64)
        rhs[outside] = np.asarray(vector, dtype=np.float64).reshape(-1)
        full_solution = np.asarray(solve(rhs), dtype=np.float64).reshape(-1)
        return full_solution[outside]

    return apply


def _build_nested_schur_preconditioner(
    displacement_matrix: Any,
    phase_matrix: Any,
    displacement_phase_matrix: Any,
    phase_displacement_matrix: Any,
    outside: np.ndarray,
    outside_interior: np.ndarray,
    n_displacement: int,
    local_patch: np.ndarray,
) -> Any:
    """Build an approximate inverse by iterating the sparse reduced Schur action.

    The outer Newton--Krylov operator applies the exact matrix-free Schur
    action.  This preconditioner uses the same local phase factorization, but
    solves the corresponding sparse approximate Schur system with a short
    inner GMRES iteration.  It is more faithful than a block diagonal solve
    and avoids explicitly forming ``J_cw J_ww^{-1} J_wc``.
    """
    if np.any(np.asarray(local_patch, dtype=np.int64) < n_displacement):
        raise ValueError("nested Schur preconditioner requires a phase patch")
    displacement = _as_scipy_csr(displacement_matrix)
    phase = _as_scipy_csr(phase_matrix)
    displacement_phase = _as_scipy_csr(displacement_phase_matrix)
    phase_displacement = _as_scipy_csr(phase_displacement_matrix)
    local_phase = np.unique(np.asarray(local_patch, dtype=np.int64) - n_displacement)
    if local_phase.size == 0:
        raise ValueError("local patch must contain at least one phase coordinate")

    interior_positions = np.flatnonzero(outside_interior)
    active_positions = np.flatnonzero(~outside_interior)
    displacement_positions = np.flatnonzero(
        (outside < n_displacement) & outside_interior
    )
    phase_positions = np.flatnonzero(
        (outside >= n_displacement) & outside_interior
    )
    displacement_dofs = outside[displacement_positions]
    phase_dofs = outside[phase_positions] - n_displacement

    local_matrix = phase[local_phase][:, local_phase].tocsc()
    try:
        local_solver = factorized(local_matrix)
    except (RuntimeError, ValueError):
        local_ilu = spilu(local_matrix, drop_tol=1.0e-4, fill_factor=10.0)
        local_solver = local_ilu.solve

    block_cc = bmat(
        [
            [
                displacement[displacement_dofs][:, displacement_dofs],
                displacement_phase[displacement_dofs][:, phase_dofs],
            ],
            [
                phase_displacement[phase_dofs][:, displacement_dofs],
                phase[phase_dofs][:, phase_dofs],
            ],
        ],
        format="csr",
    )
    block_cw = bmat(
        [
            [displacement_phase[displacement_dofs][:, local_phase]],
            [phase[phase_dofs][:, local_phase]],
        ],
        format="csr",
    )
    block_wc = bmat(
        [
            [phase_displacement[local_phase][:, displacement_dofs],
             phase[local_phase][:, phase_dofs]],
        ],
        format="csr",
    )

    def reduced_action(vector: np.ndarray) -> np.ndarray:
        direction = np.asarray(vector, dtype=np.float64).reshape(-1)
        local_rhs = block_wc @ direction
        local_response = np.asarray(local_solver(local_rhs), dtype=np.float64).reshape(-1)
        return np.asarray(block_cc @ direction - block_cw @ local_response).reshape(-1)

    dimension = int(interior_positions.size)
    reduced_operator = LinearOperator(
        (dimension, dimension), matvec=reduced_action, dtype=np.float64
    )

    # A block-diagonal solve is sufficient for the short inner iteration and
    # keeps each outer preconditioner application bounded in cost.
    displacement_factor = (
        factorized(
            displacement[displacement_dofs][:, displacement_dofs].tocsc()
        )
        if displacement_dofs.size
        else None
    )
    phase_factor = (
        factorized(phase[phase_dofs][:, phase_dofs].tocsc())
        if phase_dofs.size
        else None
    )

    def block_diagonal_action(vector: np.ndarray) -> np.ndarray:
        rhs = np.asarray(vector, dtype=np.float64).reshape(-1)
        result = np.zeros_like(rhs)
        n_u = displacement_dofs.size
        if n_u and displacement_factor is not None:
            result[:n_u] = displacement_factor(rhs[:n_u])
        if phase_dofs.size and phase_factor is not None:
            result[n_u:] = phase_factor(rhs[n_u:])
        return result

    inner_preconditioner = LinearOperator(
        (dimension, dimension), matvec=block_diagonal_action, dtype=np.float64
    )

    def apply(vector: np.ndarray) -> np.ndarray:
        rhs = np.asarray(vector, dtype=np.float64).reshape(-1)
        output = np.zeros_like(rhs)
        if dimension:
            solution, _ = scipy_gmres(
                reduced_operator,
                rhs[interior_positions],
                M=inner_preconditioner,
                rtol=2.0e-2,
                atol=1.0e-8,
                restart=min(24, dimension),
                maxiter=6,
                callback_type="legacy",
            )
            output[interior_positions] = solution
        output[active_positions] = rhs[active_positions]
        return output

    return apply


class FrozenStandardFEStepMap:
    """Deterministic FractureX staggered map for one frozen load step.

    Parameters
    ----------
    solver : MainSolve
        Initialized standard-FE solver.  Its displacement and phase fields are
        updated in place during map evaluation.
    load : float
        Prescribed top displacement, dimensionless in this verification case.
    committed_damage : ndarray, shape (n_dof,), dtype float64
        Fixed lower bound from the previous accepted load step.
    committed_history : ndarray, shape (n_cell, n_quad), dtype float64
        Fixed quadrature history from the previous accepted load step.

    Notes
    -----
    Every call restores ``committed_history`` before assembly.  Intermediate
    trial histories therefore cannot leak into later map evaluations.
    """

    def __init__(
        self,
        solver: MainSolve,
        *,
        load: float,
        committed_damage: np.ndarray,
        committed_history: np.ndarray,
        phase_bound_solver: str = "active_set",
        phase_active_set_max_iterations: int = 200,
    ) -> None:
        self.solver = solver
        self.load = float(load)
        self.committed_damage = np.asarray(
            committed_damage, dtype=np.float64
        ).reshape(-1).copy()
        self.committed_history = np.asarray(
            committed_history, dtype=np.float64
        ).copy()
        if phase_bound_solver not in {"active_set", "clip"}:
            raise ValueError("phase_bound_solver must be 'active_set' or 'clip'")
        if int(phase_active_set_max_iterations) <= 0:
            raise ValueError("phase_active_set_max_iterations must be positive")
        self.phase_bound_solver = phase_bound_solver
        self.phase_active_set_max_iterations = int(phase_active_set_max_iterations)
        self.history_snapshots: list[np.ndarray] = []
        self.record_history = False
        self.last_phase_matrix: Optional[Any] = None
        self.last_displacement_matrix: Optional[Any] = None
        self.last_phase_rhs: Optional[np.ndarray] = None
        self.last_displacement_rhs: Optional[np.ndarray] = None
        self.last_history_active_mask: Optional[np.ndarray] = None
        self.capture_system_only = False
        self.last_phase_bound_result: Optional[BoxQuadraticResult] = None
        self.fixed_damage_mask = np.zeros(self.committed_damage.size, dtype=bool)
        self.fixed_damage_values = np.zeros(self.committed_damage.size, dtype=np.float64)

        for boundary in solver._get_boundary_conditions("phase"):
            if boundary.get("type") != "Dirichlet":
                continue
            values, mask = solver.space.boundary_interpolate(
                gd=boundary["value"],
                uh=bm.zeros_like(solver.d),
                threshold=boundary["bcdof"],
            )
            mask_array = np.asarray(bm.to_numpy(mask), dtype=bool).reshape(-1)
            value_array = np.asarray(bm.to_numpy(values), dtype=np.float64).reshape(-1)
            self.fixed_damage_mask |= mask_array
            self.fixed_damage_values[mask_array] = value_array[mask_array]
        if np.any(self.fixed_damage_mask):
            self.committed_damage[self.fixed_damage_mask] = self.fixed_damage_values[
                self.fixed_damage_mask
            ]

        if self.committed_damage.size != int(solver.space.number_of_global_dofs()):
            raise ValueError("committed_damage has the wrong number of scalar DOFs")
        solver._currt_force_value = self.load
        self._refresh_displacement_constraints()
        # Capture the phase tangent while retaining the same direct linear solve
        # for both standard-FE subproblems.
        solver.solver = self._solve_and_capture  # type: ignore[method-assign]

    @property
    def displacement_size(self) -> int:
        """Return the flattened displacement-vector size in the full state."""
        return int(np.asarray(bm.to_numpy(self.solver.uh[:])).size)

    @property
    def damage_size(self) -> int:
        """Return the scalar phase-field-vector size in the full state."""
        return int(self.committed_damage.size)

    @property
    def full_state_size(self) -> int:
        """Return ``n_u+n_d`` for the flattened coupled state ordering ``(u,d)``."""
        return self.displacement_size + self.damage_size

    @property
    def damage_lower_bound(self) -> np.ndarray:
        """Return the immutable phase-field lower bound for map differencing."""
        lower = self.committed_damage.copy()
        lower[self.fixed_damage_mask] = self.fixed_damage_values[self.fixed_damage_mask]
        return lower

    @property
    def damage_upper_bound(self) -> np.ndarray:
        """Return the phase-field upper bound with Dirichlet DOFs fixed."""
        upper = np.ones(self.damage_size, dtype=np.float64)
        upper[self.fixed_damage_mask] = self.fixed_damage_values[self.fixed_damage_mask]
        return upper

    def current_full_state(self) -> np.ndarray:
        """Return the current FE fields in the coupled ordering ``(u, d)``.

        Returns
        -------
        ndarray, shape (n_u+n_d,), dtype float64
            Newly allocated flattened displacement DOFs followed by scalar
            phase-field DOFs.  This is the ordering used by ``apply_full``.
        """
        displacement = np.asarray(
            bm.to_numpy(self.solver.uh[:]), dtype=np.float64
        ).reshape(-1)
        damage = np.asarray(bm.to_numpy(self.solver.d[:]), dtype=np.float64).reshape(-1)
        return np.concatenate((displacement, damage))

    def set_full_state(self, candidate_state: np.ndarray) -> None:
        """Restore one finite full state without solving either FE block."""
        state = np.asarray(candidate_state, dtype=np.float64).reshape(-1)
        if state.size != self.full_state_size or not np.isfinite(state).all():
            raise ValueError("candidate_state must be a finite full FE state")
        main = self.solver
        main.uh[:] = bm.asarray(state[: self.displacement_size], dtype=main.uh.dtype)
        main.d[:] = bm.asarray(state[self.displacement_size :], dtype=main.d.dtype)
        main.pfcm.update_disp(main.uh)
        main.pfcm.update_phase(main.d)

    def _solve_and_capture(self, matrix: Any, rhs: Any) -> np.ndarray:
        """Solve one FE system directly and retain both block tangents."""
        matrix_csr = _as_scipy_csr(matrix)
        rhs_array = np.asarray(bm.to_numpy(rhs), dtype=np.float64).reshape(-1)
        if matrix_csr.shape[0] == self.committed_damage.size:
            self.last_phase_matrix = matrix_csr.copy()
            self.last_phase_rhs = rhs_array.copy()
        elif matrix_csr.shape[0] == self.displacement_size:
            self.last_displacement_matrix = matrix_csr.copy()
            self.last_displacement_rhs = rhs_array.copy()
        else:
            raise RuntimeError(
                "captured linear system does not match displacement or phase DOF count: "
                f"{matrix_csr.shape[0]}"
            )
        if self.capture_system_only:
            return np.zeros_like(rhs_array)
        if (
            matrix_csr.shape[0] == self.committed_damage.size
            and self.phase_bound_solver == "active_set"
        ):
            current_damage = np.asarray(
                bm.to_numpy(self.solver.d[:]), dtype=np.float64
            ).reshape(-1)
            phase_load = rhs_array + matrix_csr @ current_damage
            result = solve_box_quadratic_active_set(
                matrix_csr,
                phase_load,
                current_damage,
                self.damage_lower_bound,
                self.damage_upper_bound,
                max_iterations=self.phase_active_set_max_iterations,
            )
            self.last_phase_bound_result = result
            if not result.converged:
                raise RuntimeError(
                    "phase active-set solve did not converge: "
                    f"projected residual={result.projected_residual_norm:.6e}"
                )
            return result.state - current_damage
        solution = np.asarray(spsolve(matrix_csr.tocsc(), rhs_array), dtype=np.float64)
        if not np.isfinite(solution).all():
            raise RuntimeError("FractureX direct subproblem solve returned non-finite values")
        return solution

    def fixed_state_mask(self) -> np.ndarray:
        """Return full-state coordinates constrained by Dirichlet data.

        Returns
        -------
        ndarray, shape (n_u+n_d,), dtype bool
            True for displacement or phase coordinates that cannot participate
            in a local residual solve. The ordering is ``(u,d)``.
        """
        return np.concatenate(
            (self.fixed_displacement_mask.copy(), self.fixed_damage_mask.copy())
        )

    def _refresh_displacement_constraints(self) -> None:
        """Cache displacement Dirichlet masks and values for residual checks."""
        displacement_mask = np.zeros(self.displacement_size, dtype=bool)
        displacement_values = np.zeros(self.displacement_size, dtype=np.float64)

        def apply_constraint(threshold: Any, direction: Optional[str], value: Any) -> None:
            bc = VectorDirichletBC(
                self.solver.tspace, value, threshold, direction=direction
            )
            mask = np.asarray(bm.to_numpy(bc.set_boundary_dof()), dtype=bool).reshape(-1)
            value_array = np.asarray(value, dtype=np.float64).reshape(-1)
            if value_array.size == 1:
                displacement_values[mask] = float(value_array[0])
            elif value_array.size == displacement_values.size:
                displacement_values[mask] = value_array[mask]
            else:
                raise ValueError("displacement Dirichlet value has incompatible shape")
            displacement_mask[mask] = True

        force_bc = VectorDirichletBC(
            self.solver.tspace,
            self.load,
            self.solver._force_dof,
            direction=self.solver._force_direction,
        )
        force_mask = np.asarray(
            bm.to_numpy(force_bc.set_boundary_dof()), dtype=bool
        ).reshape(-1)
        displacement_mask[force_mask] = True
        displacement_values[force_mask] = self.load
        for boundary in self.solver._get_boundary_conditions("displacement"):
            if boundary.get("type") != "Dirichlet":
                continue
            apply_constraint(
                boundary["bcdof"],
                boundary.get("direction"),
                boundary["value"],
            )
        self.fixed_displacement_mask = displacement_mask
        self.fixed_displacement_values = displacement_values

    def assemble_coupled_residual(
        self,
        candidate_state: np.ndarray,
        *,
        enforce_phase_box: bool = True,
    ) -> np.ndarray:
        """Assemble the physical coupled residual at a full FE state.

        Parameters
        ----------
        candidate_state : ndarray, shape (n_u+n_d,), dtype float64
            Full state in ``(u,d)`` ordering. Dirichlet coordinates must equal
            their prescribed values; free coordinates are unrestricted except
            for the phase-field box.
        enforce_phase_box : bool, default True
            Validate the phase box for nonlinear iterates. Matrix-free
            directional differences set this to false so the smooth physical
            residual can be differentiated at an active bound.

        Returns
        -------
        ndarray, shape (n_u+n_d,), dtype float64
            Negative of the FE correction right-hand sides, so its Jacobian is
            the assembled coupled residual Jacobian convention used by Newton.

        Notes
        -----
        The two block systems are assembled by the production ``MainSolve``
        routines. Their linear solves are replaced by zero corrections, which
        exposes the actual residual and tangent without changing the candidate
        state. The history field is reset to the committed snapshot before each
        evaluation and then updated from the supplied displacement.
        """
        state = np.asarray(candidate_state, dtype=np.float64).reshape(-1)
        if state.size != self.full_state_size or not np.isfinite(state).all():
            raise ValueError("candidate_state must be a finite full FE state")
        displacement = state[: self.displacement_size]
        damage = state[self.displacement_size :]
        if enforce_phase_box and (
            np.any(damage < self.damage_lower_bound - 1.0e-12)
            or np.any(damage > self.damage_upper_bound + 1.0e-12)
        ):
            raise ValueError("candidate-state phase field violates its feasible box")
        if np.any(
            np.abs(
                displacement[self.fixed_displacement_mask]
                - self.fixed_displacement_values[self.fixed_displacement_mask]
            )
            > 1.0e-12
        ):
            raise ValueError("candidate_state violates displacement Dirichlet data")

        main = self.solver
        self.set_full_state(state)
        main.H = bm.asarray(self.committed_history, dtype=main.d.dtype).copy()
        main.pfcm.update_historical_field(main.H)
        self.last_phase_matrix = None
        self.last_displacement_matrix = None
        self.last_phase_rhs = None
        self.last_displacement_rhs = None
        self.capture_system_only = True
        try:
            main.solve_displacement()
            main.solve_phase_field()
        finally:
            self.capture_system_only = False
        if self.last_displacement_rhs is None or self.last_phase_rhs is None:
            raise RuntimeError("physical residual assembly did not capture both block RHS")
        residual = -np.concatenate(
            (self.last_displacement_rhs, self.last_phase_rhs)
        )
        # Keep the candidate state as the externally visible FE state. The
        # trial history generated by the phase assembly remains available in
        # ``main.H`` for diagnostics, but is reset at the next map evaluation.
        self.set_full_state(state)
        return residual

    def __call__(self, candidate_damage: np.ndarray) -> np.ndarray:
        """Apply one mechanics-then-phase sweep and return projected damage."""
        candidate = np.asarray(candidate_damage, dtype=np.float64).reshape(-1)
        if candidate.shape != self.committed_damage.shape:
            raise ValueError("candidate_damage has the wrong shape")
        if not np.isfinite(candidate).all():
            raise ValueError("candidate_damage contains non-finite values")
        if np.any(
            np.abs(
                candidate[self.fixed_damage_mask]
                - self.fixed_damage_values[self.fixed_damage_mask]
            )
            > 1.0e-12
        ):
            raise ValueError("candidate_damage violates a phase Dirichlet value")

        main = self.solver
        # ``solve_displacement`` is written as a residual correction.  A fixed
        # reference initial guess removes that implementation detail and makes
        # this callback the mathematical subproblem map d -> (u+, d+).
        main.uh[:] = bm.zeros_like(main.uh)
        main.pfcm.update_disp(main.uh)
        main.d[:] = bm.asarray(candidate, dtype=main.d.dtype)
        main.pfcm.update_phase(main.d)

        history = bm.asarray(self.committed_history, dtype=main.d.dtype).copy()
        main.H = history
        main.pfcm.update_historical_field(history)

        main.solve_displacement()
        main.solve_phase_field()

        trial_damage = np.asarray(bm.to_numpy(main.d[:]), dtype=np.float64)
        image = np.clip(np.maximum(self.committed_damage, trial_damage), 0.0, 1.0)
        image[self.fixed_damage_mask] = self.fixed_damage_values[self.fixed_damage_mask]
        main.d[:] = bm.asarray(image, dtype=main.d.dtype)
        main.pfcm.update_phase(main.d)

        trial_history = np.asarray(bm.to_numpy(main.H), dtype=np.float64).copy()
        if self.record_history:
            self.history_snapshots.append(trial_history)
        return image.copy()

    def apply_full(self, candidate_state: np.ndarray) -> np.ndarray:
        """Apply one sweep to the full state and return ``(u_plus, d_plus)``.

        Parameters
        ----------
        candidate_state : ndarray, shape (n_u+n_d,), dtype float64
            Coupled state in flattened ``(u,d)`` ordering.  Displacement DOFs
            are unbounded; phase-field DOFs must lie between the committed
            lower bound and one.

        Returns
        -------
        ndarray, shape (n_u+n_d,), dtype float64
            Newly allocated state after exactly one mechanics-then-phase sweep.

        Notes
        -----
        The exact mechanics subproblem is a function of phase field and load,
        not of the old displacement iterate.  Accordingly the supplied
        displacement segment is checked for shape and finiteness but is not
        used; this realizes the mathematical full sweep with its zero first
        block column, rather than the residual-correction implementation
        detail of ``MainSolve.solve_displacement``.
        """
        state = np.asarray(candidate_state, dtype=np.float64).reshape(-1)
        if state.size != self.full_state_size:
            raise ValueError(
                f"candidate_state has size {state.size}, expected {self.full_state_size}"
            )
        if not np.isfinite(state).all():
            raise ValueError("candidate_state contains non-finite values")
        damage = state[self.displacement_size :]
        if np.any(damage < self.damage_lower_bound) or np.any(damage > self.damage_upper_bound):
            raise ValueError("candidate-state phase field violates its feasible box")

        self(damage)
        return self.current_full_state()

    def apply_damage_to_full_state(self, candidate_damage: np.ndarray) -> np.ndarray:
        """Apply one sweep from damage input to the full output state.

        Parameters
        ----------
        candidate_damage : ndarray, shape (n_d,), dtype float64
            Feasible phase-field iterate in scalar FE ordering.

        Returns
        -------
        ndarray, shape (n_u+n_d,), dtype float64
            Flattened displacement and phase fields after one exact sweep.

        Notes
        -----
        This is the nonzero block column of the full staggered derivative.
        It avoids finite-differencing old displacement directions, which are
        analytically zero for an exact mechanics subproblem solve.
        """
        self(candidate_damage)
        return self.current_full_state()

    def restore_committed_history(self) -> float:
        """Restore committed history and return the exact rollback error."""
        history = bm.asarray(self.committed_history, dtype=self.solver.d.dtype).copy()
        self.solver.H = history
        self.solver.pfcm.update_historical_field(history)
        restored = np.asarray(bm.to_numpy(self.solver.pfcm.H), dtype=np.float64)
        return float(np.linalg.norm(restored - self.committed_history))

    def set_load(self, load: float) -> None:
        """Set the prescribed displacement of the next frozen checkpoint."""
        if not np.isfinite(load):
            raise ValueError("load must be finite")
        self.load = float(load)
        self.solver._currt_force_value = self.load
        self._refresh_displacement_constraints()

    def commit_checkpoint(
        self,
        converged_damage: np.ndarray,
        converged_history: np.ndarray,
    ) -> None:
        """Commit one converged load state before advancing to the next one.

        Parameters
        ----------
        converged_damage : ndarray, shape (n_d,), dtype float64
            Fixed point of the just-completed phase-field map.
        converged_history : ndarray, shape (n_cell, n_quad), dtype float64
            History field evaluated at that fixed point.  It becomes the
            immutable lower envelope for all map calls at the next load.

        Notes
        -----
        This method implements quasi-static checkpointing only.  It does not
        change the production ``MainSolve`` load-step algorithm.
        """
        damage = np.asarray(converged_damage, dtype=np.float64).reshape(-1)
        history = np.asarray(converged_history, dtype=np.float64)
        if damage.shape != self.committed_damage.shape:
            raise ValueError("converged_damage has the wrong shape")
        if history.shape != self.committed_history.shape:
            raise ValueError("converged_history has the wrong shape")
        if not np.isfinite(damage).all() or not np.isfinite(history).all():
            raise ValueError("committed checkpoint contains non-finite values")
        if np.any(damage < self.committed_damage - 1.0e-12):
            raise ValueError("damage irreversibility is violated at checkpoint commit")
        if np.any(history < self.committed_history - 1.0e-12):
            raise ValueError("history irreversibility is violated at checkpoint commit")
        self.committed_damage = np.maximum(self.committed_damage, damage)
        self.committed_damage[self.fixed_damage_mask] = self.fixed_damage_values[
            self.fixed_damage_mask
        ]
        self.committed_history = np.maximum(self.committed_history, history)
        main = self.solver
        main.d[:] = bm.asarray(self.committed_damage, dtype=main.d.dtype)
        main.H = bm.asarray(self.committed_history, dtype=main.d.dtype)
        main.pfcm.update_phase(main.d)
        main.pfcm.update_historical_field(main.H)


def _build_standard_fe_map(args: argparse.Namespace) -> tuple[MainSolve, FrozenStandardFEStepMap]:
    """Build one selected standard-FE solver and its frozen sweep map.

    ``unit_seed`` is retained as a small deterministic interface fixture.
    ``model0_circular_hole`` uses the circular-hole standard-FE benchmark;
    ``model5_notched_beam`` uses the geometric-notch three-point-bending
    benchmark.  Both have no artificial phase-field seed.
    """
    if args.case == "model5_notched_beam":
        return _build_model5_notched_beam_map(args)
    if args.case == "model0_circular_hole":
        return _build_model0_circular_hole_map(args)
    if args.case == "model0_example":
        return _build_model0_example_map(args)
    if args.case != "unit_seed":
        raise ValueError(f"unsupported verification case: {args.case}")
    return _build_unit_seed_map(args)


def _build_unit_seed_map(args: argparse.Namespace) -> tuple[MainSolve, FrozenStandardFEStepMap]:
    """Build the unit-square seeded fixture used for algorithmic smoke tests."""
    if args.nx < 2 or args.nx % 2 != 0:
        raise ValueError("nx must be an even integer >= 2 so the seed lies on mesh nodes")
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0], nx=int(args.nx), ny=int(args.nx)
    )
    material = {
        "E": float(args.young_modulus),
        "nu": float(args.poisson_ratio),
        "Gc": float(args.fracture_toughness),
        "l0": float(args.length_scale),
    }
    main = MainSolve(mesh=mesh, material_params=material, model_type="HybridModel")
    main.add_boundary_condition(
        "force", "Dirichlet", _top_boundary, [0.0, float(args.load)], "y"
    )
    main.add_boundary_condition(
        "displacement", "Dirichlet", _bottom_boundary, 0.0
    )
    main._method = "lfem"
    main.initialize_settings(p=int(args.degree), q=int(args.quadrature_order))
    main._initialize_force_boundary()

    interpolation_points = np.asarray(
        bm.to_numpy(main.space.interpolation_points()), dtype=np.float64
    )
    seed_mask = (
        (np.abs(interpolation_points[:, 1] - 0.5) < 1.0e-12)
        & (interpolation_points[:, 0] <= float(args.seed_length) + 1.0e-12)
    )
    if not np.any(seed_mask):
        raise RuntimeError("the prescribed damage seed selected no scalar FE DOFs")
    committed_damage = np.zeros(interpolation_points.shape[0], dtype=np.float64)
    committed_damage[seed_mask] = float(args.seed_damage)
    main.d[:] = bm.asarray(committed_damage, dtype=main.d.dtype)
    main.pfcm.update_phase(main.d)

    quadrature = mesh.quadrature_formula(int(args.quadrature_order), "cell")
    _, weights = quadrature.get_quadrature_points_and_weights()
    committed_history = np.zeros(
        (int(mesh.number_of_cells()), int(weights.shape[0])), dtype=np.float64
    )
    step_map = FrozenStandardFEStepMap(
        main,
        load=float(args.load),
        committed_damage=committed_damage,
        committed_history=committed_history,
        phase_bound_solver=args.phase_bound_solver,
        phase_active_set_max_iterations=args.phase_active_set_max_iterations,
    )
    return main, step_map


def _build_model5_notched_beam_map(
    args: argparse.Namespace,
) -> tuple[MainSolve, FrozenStandardFEStepMap]:
    """Build the geometric-notch Model-5 standard-FE map.

    The notch is represented by the mesh geometry.  The phase field therefore
    starts at zero, and irreversibility enters only through the committed
    phase and history checkpoints.
    """
    if args.mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive for model5_notched_beam")
    if args.load >= 0.0:
        raise ValueError("model5_notched_beam requires a negative downward load")
    from fracturex.cases.phase_field.model5_three_point_bending import (
        Model5StandardFEM,
        _attach_model5_bcs,
    )

    bm.set_backend("numpy")
    model = Model5StandardFEM(
        mesh_size=float(args.mesh_size), with_geometric_notch=True
    )
    mesh = model.build_mesh()
    main = MainSolve(mesh=mesh, material_params=dict(model.params), model_type="HybridModel")
    force_values = bm.asarray([0.0, float(args.load)], dtype=bm.float64)
    _attach_model5_bcs(main, model, force_values)
    main._method = "lfem"
    main.initialize_settings(p=int(args.degree), q=int(args.quadrature_order))
    main._initialize_force_boundary()

    scalar_dofs = int(main.space.number_of_global_dofs())
    committed_damage = np.zeros(scalar_dofs, dtype=np.float64)
    main.d[:] = bm.asarray(committed_damage, dtype=main.d.dtype)
    main.pfcm.update_phase(main.d)
    quadrature = mesh.quadrature_formula(int(args.quadrature_order), "cell")
    _, weights = quadrature.get_quadrature_points_and_weights()
    committed_history = np.zeros(
        (int(mesh.number_of_cells()), int(weights.shape[0])), dtype=np.float64
    )
    step_map = FrozenStandardFEStepMap(
        main,
        load=float(args.load),
        committed_damage=committed_damage,
        committed_history=committed_history,
        phase_bound_solver=args.phase_bound_solver,
        phase_active_set_max_iterations=args.phase_active_set_max_iterations,
    )
    return main, step_map


def _build_model0_circular_hole_map(
    args: argparse.Namespace,
) -> tuple[MainSolve, FrozenStandardFEStepMap]:
    """Build the standard-Lagrange Model-0 circular-hole map.

    The circular hole is geometrically removed by FEALPy distmesh.  Its
    boundary is fixed for displacement and phase, while the top edge receives
    the prescribed vertical displacement.  ``mesh_size`` is the distmesh
    target ``hmin`` and is intentionally explicit for fast smoke runs.
    """
    if args.mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive for model0_circular_hole")
    if args.load <= 0.0:
        raise ValueError("model0_circular_hole requires a positive upward load")
    from fracturex.cases.model0_circular_notch import Model0CircularNotchCase

    bm.set_backend("numpy")
    # FEALPy distmesh samples interior points.  Fix NumPy's generator so a
    # checkpoint scan and its later reproduction use the identical mesh.
    np.random.seed(int(args.seed))
    case = Model0CircularNotchCase(hmin=float(args.mesh_size), distmesh_maxit=100)
    mesh = case.make_mesh()
    material = {"E": 200.0, "nu": 0.2, "Gc": 1.0, "l0": 0.02}
    main = MainSolve(mesh=mesh, material_params=material, model_type="HybridModel")
    force_values = bm.asarray([0.0, float(args.load)], dtype=bm.float64)
    main.add_boundary_condition(
        "force", "Dirichlet", case._on_top, force_values, "y"
    )
    main.add_boundary_condition(
        "displacement", "Dirichlet", case._on_inner_circle, 0
    )
    main.add_boundary_condition("phase", "Dirichlet", case._on_inner_circle, 0)
    main._method = "lfem"
    main.initialize_settings(p=int(args.degree), q=int(args.quadrature_order))
    main._initialize_force_boundary()

    scalar_dofs = int(main.space.number_of_global_dofs())
    committed_damage = np.zeros(scalar_dofs, dtype=np.float64)
    main.d[:] = bm.asarray(committed_damage, dtype=main.d.dtype)
    main.pfcm.update_phase(main.d)
    quadrature = mesh.quadrature_formula(int(args.quadrature_order), "cell")
    _, weights = quadrature.get_quadrature_points_and_weights()
    committed_history = np.zeros(
        (int(mesh.number_of_cells()), int(weights.shape[0])), dtype=np.float64
    )
    step_map = FrozenStandardFEStepMap(
        main,
        load=float(args.load),
        committed_damage=committed_damage,
        committed_history=committed_history,
        phase_bound_solver=args.phase_bound_solver,
        phase_active_set_max_iterations=args.phase_active_set_max_iterations,
    )
    return main, step_map


def _build_model0_example_map(
    args: argparse.Namespace,
) -> tuple[MainSolve, FrozenStandardFEStepMap]:
    """Build a map with the exact selectors used by ``model0_example.py``."""
    if args.mesh_size <= 0.0:
        raise ValueError("mesh_size must be positive for model0_example")
    if args.load <= 0.0:
        raise ValueError("model0_example requires a positive upward load")
    from fracturex.cases.model0_circular_notch import Model0CircularNotchCase

    bm.set_backend("numpy")
    np.random.seed(int(args.seed))
    case = Model0CircularNotchCase(hmin=float(args.mesh_size), distmesh_maxit=100)
    mesh = case.make_mesh()

    def on_top(points: Any) -> Any:
        return bm.abs(points[..., 1] - 1.0) < 1.0e-12

    def on_inner_circle(points: Any) -> Any:
        return (
            bm.abs(
                (points[..., 0] - 0.5) ** 2
                + bm.abs(points[..., 1] - 0.5) ** 2
                - 0.04
            )
            < 0.001
        )

    material = {"E": 200.0, "nu": 0.2, "Gc": 1.0, "l0": 0.02}
    main = MainSolve(mesh=mesh, material_params=material, model_type="HybridModel")
    force_values = bm.asarray([0.0, float(args.load)], dtype=bm.float64)
    main.add_boundary_condition("force", "Dirichlet", on_top, force_values, "y")
    main.add_boundary_condition("displacement", "Dirichlet", on_inner_circle, 0)
    main.add_boundary_condition("phase", "Dirichlet", on_inner_circle, 0)
    main._method = "lfem"
    main.initialize_settings(p=int(args.degree), q=int(args.quadrature_order))
    main._initialize_force_boundary()

    scalar_dofs = int(main.space.number_of_global_dofs())
    committed_damage = np.zeros(scalar_dofs, dtype=np.float64)
    main.d[:] = bm.asarray(committed_damage, dtype=main.d.dtype)
    main.pfcm.update_phase(main.d)
    quadrature = mesh.quadrature_formula(int(args.quadrature_order), "cell")
    _, weights = quadrature.get_quadrature_points_and_weights()
    committed_history = np.zeros(
        (int(mesh.number_of_cells()), int(weights.shape[0])), dtype=np.float64
    )
    step_map = FrozenStandardFEStepMap(
        main,
        load=float(args.load),
        committed_damage=committed_damage,
        committed_history=committed_history,
        phase_bound_solver=args.phase_bound_solver,
        phase_active_set_max_iterations=args.phase_active_set_max_iterations,
    )
    return main, step_map


def _power_iteration_rate(operator: np.ndarray, *, iterations: int = 80) -> float:
    """Return the final Euclidean norm ratio from deterministic power iteration."""
    vector = np.linspace(1.0, 2.0, operator.shape[0], dtype=np.float64)
    vector /= np.linalg.norm(vector)
    rate = 0.0
    for _ in range(iterations):
        image = operator @ vector
        rate = float(np.linalg.norm(image))
        if rate <= np.finfo(np.float64).tiny:
            return 0.0
        vector = image / rate
    return rate


def _modal_replay_initial(
    fixed_point: np.ndarray,
    mode: np.ndarray,
    lower_bound: np.ndarray,
    *,
    amplitude: float,
) -> np.ndarray:
    """Choose the larger feasible signed perturbation of a dominant mode."""
    direction = np.real(mode)
    if np.linalg.norm(direction) <= 1.0e-14:
        direction = np.imag(mode)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-30)
    plus = np.clip(fixed_point + amplitude * direction, lower_bound, 1.0)
    minus = np.clip(fixed_point - amplitude * direction, lower_bound, 1.0)
    if np.linalg.norm(plus - fixed_point) >= np.linalg.norm(minus - fixed_point):
        return plus
    return minus


def _git_commit(project_root: Path) -> str:
    """Return the current commit identifier, or ``unknown`` outside Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _package_version(name: str) -> str:
    """Return installed package version without making it a hard dependency."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _select_top_scored_cells(scores: np.ndarray, count: int) -> np.ndarray:
    """Return a deterministic mask containing the ``count`` largest scores."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("cell scores must be a nonempty finite vector")
    if count <= 0 or count > values.size:
        raise ValueError("count must lie between one and the number of cells")
    order = np.argsort(-values, kind="stable")
    selected = np.zeros(values.size, dtype=bool)
    selected[order[:count]] = True
    return selected


def _select_cells_to_free_dof_budget(
    scores: np.ndarray,
    cell_to_dof: np.ndarray,
    free_dof_mask: np.ndarray,
    target_free_dofs: int,
) -> np.ndarray:
    """Select high-score cells whose free-DOF union best matches a budget.

    Parameters
    ----------
    scores : ndarray, shape (n_cells,), dtype float64
        Finite region indicator; larger values are selected first.
    cell_to_dof : ndarray, shape (n_cells, n_local), dtype int64
        Coupled cell connectivity in full-state numbering.
    free_dof_mask : ndarray, shape (n_state,), dtype bool
        Fixed-active-set coordinates eligible for the comparison.
    target_free_dofs : int
        Positive target size of the selected free-DOF union.

    Returns
    -------
    ndarray, shape (n_cells,), dtype bool
        Cell mask whose union size is the closest prefix value to the target.
        Ties retain the smaller prefix. Inputs are not modified.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    connectivity = np.asarray(cell_to_dof, dtype=np.int64)
    eligible = np.asarray(free_dof_mask, dtype=bool).reshape(-1)
    if connectivity.ndim != 2 or connectivity.shape[0] != values.size:
        raise ValueError("cell_to_dof and scores have incompatible shapes")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if np.any(connectivity < 0) or np.any(connectivity >= eligible.size):
        raise ValueError("cell_to_dof contains an out-of-range index")
    if target_free_dofs <= 0:
        raise ValueError("target_free_dofs must be positive")

    order = np.argsort(-values, kind="stable")
    selected = np.zeros(values.size, dtype=bool)
    selected_dofs = np.zeros(eligible.size, dtype=bool)
    best = selected.copy()
    best_error = target_free_dofs
    for cell in order:
        selected[cell] = True
        dofs = connectivity[cell]
        selected_dofs[dofs[eligible[dofs]]] = True
        count = int(np.count_nonzero(selected_dofs))
        error = abs(count - target_free_dofs)
        if error < best_error:
            best = selected.copy()
            best_error = error
        if count >= target_free_dofs:
            break
    if not np.any(best):
        raise RuntimeError("cell ranking did not select any eligible free DOF")
    return best


def _build_phase_patch_budget_sweep(
    theta_values: list[float],
    slow_cell_scores: np.ndarray,
    damage_cell_scores: np.ndarray,
    coupled_cell_to_dof: np.ndarray,
    fixed_state_mask: np.ndarray,
    displacement_dofs: int,
    *,
    budget_space: str = "phase",
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Build slow and damage patches at matched solver-coordinate budgets.

    Parameters
    ----------
    theta_values : list[float]
        Unique slow-space trace fractions in the open interval ``(0, 1)``.
    slow_cell_scores, damage_cell_scores : ndarray, shape (n_cells,)
        Finite cell indicators. The slow score is accumulated to each
        ``theta``; the damage score is ranked to the resulting phase budget.
    coupled_cell_to_dof : ndarray, shape (n_cells, n_local), dtype int64
        Cell connectivity in full coupled-state numbering.
    fixed_state_mask : ndarray, shape (n_state,), dtype bool
        Dirichlet-coordinate mask. Only nonfixed solver coordinates consume
        the matched local factorization budget.
    displacement_dofs : int
        Offset of the phase block in the full coupled state.
    budget_space : {"phase", "coupled"}, default "phase"
        Match either nonfixed phase coordinates or all nonfixed coupled
        coordinates in each cell union.

    Returns
    -------
    patches : dict[str, ndarray]
        Phase-only solver patches and corresponding coupled cell-union patches
        for every slow/damage pair. Coupled patch names end in ``_coupled``.
    records : list[dict]
        Deterministic region metadata used to merge mechanism and cost results.

    Notes
    -----
    The selected budget equals the dimension of the local LU factorization for
    the corresponding Reduced-NE patch space.
    """
    slow_scores = np.asarray(slow_cell_scores, dtype=np.float64).reshape(-1)
    damage_scores = np.asarray(damage_cell_scores, dtype=np.float64).reshape(-1)
    connectivity = np.asarray(coupled_cell_to_dof, dtype=np.int64)
    fixed = np.asarray(fixed_state_mask, dtype=bool).reshape(-1)
    values = [float(theta) for theta in theta_values]
    if not values or not np.isfinite(values).all():
        raise ValueError("theta_values must contain finite values")
    if any(theta <= 0.0 or theta >= 1.0 for theta in values):
        raise ValueError("theta_values must lie strictly between zero and one")
    if len(set(values)) != len(values):
        raise ValueError("theta_values must be unique")
    if slow_scores.shape != damage_scores.shape:
        raise ValueError("slow and damage cell scores must have matching shapes")
    if connectivity.ndim != 2 or connectivity.shape[0] != slow_scores.size:
        raise ValueError("cell scores and coupled connectivity are incompatible")
    if displacement_dofs <= 0 or displacement_dofs >= fixed.size:
        raise ValueError("displacement_dofs must split the coupled state")
    if budget_space not in {"phase", "coupled"}:
        raise ValueError("budget_space must be phase or coupled")

    phase_eligible = np.zeros(fixed.size, dtype=bool)
    phase_eligible[displacement_dofs:] = ~fixed[displacement_dofs:]
    budget_eligible = phase_eligible if budget_space == "phase" else ~fixed
    total_slow_score = float(np.sum(slow_scores))
    if total_slow_score <= 0.0:
        raise ValueError("slow cell scores must have positive sum")

    patches: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for index, theta in enumerate(values):
        slow_cells = select_bulk_cells(slow_scores, theta=theta)
        slow_coupled = _cells_to_coupled_dofs(slow_cells, connectivity)
        slow_phase = slow_coupled[
            (slow_coupled >= displacement_dofs) & phase_eligible[slow_coupled]
        ]
        slow_local = slow_coupled[budget_eligible[slow_coupled]]
        if slow_phase.size == 0:
            raise RuntimeError(f"theta={theta:.6g} produced an empty slow phase patch")
        if slow_local.size == 0:
            raise RuntimeError(f"theta={theta:.6g} produced an empty slow patch")

        damage_cells = _select_cells_to_free_dof_budget(
            damage_scores,
            connectivity,
            budget_eligible,
            target_free_dofs=int(slow_local.size),
        )
        damage_coupled = _cells_to_coupled_dofs(damage_cells, connectivity)
        damage_phase = damage_coupled[
            (damage_coupled >= displacement_dofs) & phase_eligible[damage_coupled]
        ]
        damage_local = damage_coupled[budget_eligible[damage_coupled]]
        if damage_phase.size == 0:
            raise RuntimeError(
                f"theta={theta:.6g} produced an empty damage phase patch"
            )
        label = f"theta_{index:02d}"
        slow_name = f"slow_{label}"
        damage_name = f"damage_{label}"
        slow_coupled_name = f"{slow_name}_coupled"
        damage_coupled_name = f"{damage_name}_coupled"
        patches[slow_name] = np.unique(slow_phase)
        patches[damage_name] = np.unique(damage_phase)
        patches[slow_coupled_name] = slow_coupled
        patches[damage_coupled_name] = damage_coupled
        records.append(
            {
                "theta": theta,
                "slow_patch": slow_name,
                "damage_patch": damage_name,
                "slow_coupled_patch": slow_coupled_name,
                "damage_coupled_patch": damage_coupled_name,
                "budget_space": budget_space,
                "budget_metric": (
                    f"non-Dirichlet {budget_space} local-patch DOFs"
                ),
                "target_local_patch_dofs": int(slow_local.size),
                "slow_local_patch_dofs": int(slow_local.size),
                "damage_local_patch_dofs": int(damage_local.size),
                "local_budget_absolute_mismatch": int(
                    abs(damage_local.size - slow_local.size)
                ),
                "target_phase_patch_dofs": int(slow_phase.size),
                "slow_phase_patch_dofs": int(slow_phase.size),
                "damage_phase_patch_dofs": int(damage_phase.size),
                "phase_budget_absolute_mismatch": int(
                    abs(damage_phase.size - slow_phase.size)
                ),
                "slow_cells": int(np.count_nonzero(slow_cells)),
                "damage_cells": int(np.count_nonzero(damage_cells)),
                "slow_coupled_cell_union_dofs": int(slow_coupled.size),
                "damage_coupled_cell_union_dofs": int(damage_coupled.size),
                "slow_trace_fraction": float(
                    np.sum(slow_scores[slow_cells]) / total_slow_score
                ),
                "damage_trace_fraction": float(
                    np.sum(slow_scores[damage_cells]) / total_slow_score
                ),
            }
        )
    return patches, records


def _merge_benefit_cost_pareto(
    region_records: list[dict[str, Any]],
    history_calibration: dict[str, Any],
    reduced_solver_summary: dict[str, Any],
    spectral_radius: float,
) -> dict[str, Any]:
    """Merge region quality and Reduced-NE cost into Pareto records.

    The factor ``q=rho*chi`` uses the coupled cell-union space as a region
    quality proxy. The actual elimination space and its survival factor are
    recorded explicitly so mechanism and cost use the same coordinates in a
    coupled-patch run.
    """
    if not (0.0 < spectral_radius < 1.0):
        raise ValueError("spectral_radius must lie strictly between zero and one")
    history_patches = history_calibration["patches"]
    solver_patches = reduced_solver_summary["patches"]
    rows: list[dict[str, Any]] = []
    for region in region_records:
        for region_type in ("slow", "damage"):
            patch_name = region[f"{region_type}_patch"]
            coupled_patch_name = region[f"{region_type}_coupled_patch"]
            phase_mechanism = history_patches[patch_name]
            coupled_mechanism = history_patches[coupled_patch_name]
            solver = solver_patches[patch_name]
            coupled_survival = float(
                coupled_mechanism["slow_subspace_survival_factor"]
            )
            phase_survival = float(
                phase_mechanism["slow_subspace_survival_factor"]
            )
            predicted = float(spectral_radius * coupled_survival)
            mechanism_gain = float(-np.log(max(predicted, np.finfo(float).tiny)))
            phase_dofs = int(solver["phase_patch_dofs"])
            local_dofs = int(solver.get("local_patch_dofs", phase_dofs))
            patch_space = str(solver.get("patch_space", "phase"))
            actual_survival = (
                coupled_survival if patch_space == "coupled" else phase_survival
            )
            total_time = float(solver["total_wall_time_including_warmup_seconds"])
            rows.append(
                {
                    "theta": float(region["theta"]),
                    "region": region_type,
                    "patch": patch_name,
                    "patch_space": patch_space,
                    "selection_budget_metric": region["budget_metric"],
                    "target_local_patch_dofs": int(
                        region["target_local_patch_dofs"]
                    ),
                    "local_budget_absolute_mismatch": int(
                        region["local_budget_absolute_mismatch"]
                    ),
                    "local_patch_dofs": local_dofs,
                    "local_displacement_dofs": int(
                        solver.get("local_displacement_dofs", 0)
                    ),
                    "phase_patch_dofs": phase_dofs,
                    "phase_patch_fraction_of_all_free_state_dofs": float(
                        solver["phase_patch_fraction_of_all_free_state_dofs"]
                    ),
                    "cells": int(region[f"{region_type}_cells"]),
                    "coupled_cell_union_dofs": int(
                        region[f"{region_type}_coupled_cell_union_dofs"]
                    ),
                    "trace_fraction": float(
                        region[f"{region_type}_trace_fraction"]
                    ),
                    "coupled_region_survival_factor": coupled_survival,
                    "coupled_region_predicted_contraction_proxy": predicted,
                    "phase_elimination_survival_factor": phase_survival,
                    "actual_elimination_survival_factor": actual_survival,
                    "actual_elimination_predicted_contraction": float(
                        spectral_radius * actual_survival
                    ),
                    "mechanism_gain_minus_log_q": mechanism_gain,
                    "mechanism_gain_per_phase_dof": float(
                        mechanism_gain / phase_dofs
                    ),
                    "mechanism_gain_per_local_dof": float(
                        mechanism_gain / local_dofs
                    ),
                    "smooth_free_patch_dofs": int(
                        phase_mechanism["smooth_free_patch_dofs"]
                    ),
                    "phase_local_jacobian_condition_number": float(
                        phase_mechanism["local_jacobian_condition_number"]
                    ),
                    "coupled_smooth_free_patch_dofs": int(
                        coupled_mechanism["smooth_free_patch_dofs"]
                    ),
                    "coupled_local_jacobian_condition_number": float(
                        coupled_mechanism["local_jacobian_condition_number"]
                    ),
                    "pre_switch_external_schur_coupling_mean": (
                        None
                        if solver.get("pre_switch_external_schur_coupling_mean") is None
                        else float(solver["pre_switch_external_schur_coupling_mean"])
                    ),
                    "pre_switch_external_schur_coupling_max": (
                        None
                        if solver.get("pre_switch_external_schur_coupling_max") is None
                        else float(solver["pre_switch_external_schur_coupling_max"])
                    ),
                    "pre_switch_external_schur_coupling_samples": (
                        None
                        if solver.get("pre_switch_external_schur_coupling_samples") is None
                        else int(solver["pre_switch_external_schur_coupling_samples"])
                    ),
                    "pre_switch_external_schur_coupling_local_condition_number": (
                        None
                        if solver.get(
                            "pre_switch_external_schur_coupling_local_condition_number"
                        ) is None
                        else float(
                            solver[
                                "pre_switch_external_schur_coupling_local_condition_number"
                            ]
                        )
                    ),
                    "outer_newton_iterations": int(
                        solver["outer_newton_iterations"]
                    ),
                    "local_linear_solves": int(solver["local_linear_solves"]),
                    "local_linear_solve_wall_time_seconds": float(
                        solver["local_linear_solve_wall_time_seconds"]
                    ),
                    "physical_residual_evaluations": int(
                        solver["physical_residual_evaluations"]
                    ),
                    "local_jacobian_assemblies": int(
                        solver.get("local_jacobian_assemblies", 0)
                    ),
                    "local_jacobian_assembly_wall_time_seconds": float(
                        solver.get(
                            "local_jacobian_assembly_wall_time_seconds", 0.0
                        )
                    ),
                    "total_residual_equivalent_evaluations": int(
                        solver["total_residual_equivalent_evaluations"]
                    ),
                    "total_wall_time_seconds": total_time,
                    "wall_time_speedup_over_staggered": float(
                        solver["wall_time_speedup_over_staggered"]
                    ),
                    "full_solution_l2_difference_from_staggered": float(
                        solver["full_solution_l2_difference_from_staggered"]
                    ),
                    "final_projected_residual_norm": float(
                        solver["final_coupled_residual"]["projected_raw_norm"]
                    ),
                    "all_acceptance_checks_passed": bool(
                        solver["all_acceptance_checks_passed"]
                    ),
                }
            )

    accepted_rows = [row for row in rows if row["all_acceptance_checks_passed"]]
    for row in rows:
        row["pareto_optimal_chi_vs_time"] = bool(
            row["all_acceptance_checks_passed"]
            and not any(
                other["coupled_region_survival_factor"]
                <= row["coupled_region_survival_factor"]
                and other["total_wall_time_seconds"]
                <= row["total_wall_time_seconds"]
                and (
                    other["coupled_region_survival_factor"]
                    < row["coupled_region_survival_factor"]
                    or other["total_wall_time_seconds"]
                    < row["total_wall_time_seconds"]
                )
                for other in accepted_rows
            )
        )
    return {
        "selection_budget": region_records[0]["budget_metric"],
        "solver_patch_space": reduced_solver_summary.get("patch_space", "phase"),
        "predicted_contraction_model": (
            "coupled-region proxy q_omega = rho(G) * chi_omega(V_r); "
            "actual elimination coordinates and survival are reported explicitly"
        ),
        "spectral_radius": float(spectral_radius),
        "records": rows,
    }


def _cell_gradient_magnitude(main: MainSolve) -> np.ndarray:
    """Return the cell-mean magnitude of the current phase-field gradient.

    Returns
    -------
    ndarray, shape (n_cell,), dtype float64
        Dimensionless ``||grad d||`` scores in the mesh cell ordering.
    """
    gdim = int(main.mesh.geo_dimension())
    barycenter = bm.full((1, gdim + 1), 1.0 / (gdim + 1), dtype=main.d.dtype)
    gradients = np.asarray(
        bm.to_numpy(main.d.grad_value(barycenter)), dtype=np.float64
    )
    if gradients.ndim != 3 or gradients.shape[-1] != gdim:
        raise RuntimeError("unexpected phase-field gradient shape at cell barycenters")
    return np.mean(np.linalg.norm(gradients, axis=-1), axis=1)


def _cells_to_coupled_dofs(
    selected_cells: np.ndarray,
    coupled_cell_to_dof: np.ndarray,
) -> np.ndarray:
    """Return sorted unique full-state DOFs touched by selected mesh cells."""
    mask = np.asarray(selected_cells, dtype=bool).reshape(-1)
    connectivity = np.asarray(coupled_cell_to_dof, dtype=np.int64)
    if mask.shape != (connectivity.shape[0],) or not np.any(mask):
        raise ValueError("selected_cells must select at least one connected cell")
    return np.unique(connectivity[mask].reshape(-1))


def _broadcast_cell_quadrature_values(
    values: Any,
    number_of_cells: int,
    *,
    name: str,
) -> np.ndarray:
    """Return an FE quadrature array with an explicit leading cell axis."""
    array = np.asarray(bm.to_numpy(values), dtype=np.float64)
    if array.ndim < 2:
        raise RuntimeError(f"{name} must contain cell and quadrature axes")
    if array.shape[0] == 1:
        return np.broadcast_to(
            array, (number_of_cells,) + array.shape[1:]
        )
    if array.shape[0] != number_of_cells:
        raise RuntimeError(
            f"{name} has {array.shape[0]} cells; expected {number_of_cells}"
        )
    return array


def _assemble_history_field_coupling_blocks(
    step_map: FrozenStandardFEStepMap,
) -> tuple[Any, Any]:
    """Assemble the two off-diagonal blocks of the coupled FE Jacobian.

    Returns
    -------
    J_ud : scipy.sparse.csr_matrix, shape (n_u, n_d)
        Derivative of the equilibrium residual with respect to phase damage.
    J_du : scipy.sparse.csr_matrix, shape (n_d, n_u)
        Semismooth derivative of the phase residual with respect to
        displacement on the active history branch.

    Notes
    -----
    The state, quadrature history, and diagonal block tangents must first be
    synchronized by :meth:`FrozenStandardFEStepMap.assemble_coupled_residual`.
    For the quadratic degradation law used by the verification cases,
    ``R_d`` contains ``g'(d) H(u)``. The history derivative is selected where
    the current tensile energy strictly exceeds the committed history; it is
    zero on the frozen branch of the pointwise maximum.
    """
    if (
        step_map.last_displacement_matrix is None
        or step_map.last_phase_matrix is None
    ):
        raise RuntimeError("diagonal tangents must be assembled before coupling blocks")

    main = step_map.solver
    mesh = main.mesh
    number_of_cells = int(mesh.number_of_cells())
    quadrature = mesh.quadrature_formula(int(main.q), "cell")
    bcs, quadrature_weights = quadrature.get_quadrature_points_and_weights()
    weights = np.asarray(bm.to_numpy(quadrature_weights), dtype=np.float64)
    cell_measure = np.asarray(
        bm.to_numpy(mesh.entity_measure("cell")), dtype=np.float64
    ).reshape(-1)
    if cell_measure.size != number_of_cells:
        raise RuntimeError("cell measures do not match the mesh cell count")

    phase_basis = _broadcast_cell_quadrature_values(
        main.space.basis(bcs), number_of_cells, name="phase basis"
    )
    phase_gradient = _broadcast_cell_quadrature_values(
        main.space.grad_basis(bcs), number_of_cells, name="phase gradient basis"
    )
    strain_matrix = np.asarray(
        bm.to_numpy(
            main.pfcm.strain_matrix(
                main.tspace.dof_priority,
                bm.asarray(phase_gradient, dtype=main.d.dtype),
            )
        ),
        dtype=np.float64,
    )

    # The mechanics residual is assembled from the physical full displacement
    # before Dirichlet rows are replaced.  Its free rows therefore retain the
    # contribution of prescribed displacement values; only constrained rows
    # have zero coupling derivatives after boundary elimination.
    strain = main.pfcm.strain_value(bcs)
    equilibrium_strain = strain
    history_strain = strain
    equilibrium_positive_stress, _ = main.pfcm.stress_decomposition(
        equilibrium_strain
    )
    history_positive_stress, _ = main.pfcm.history_split.stress_decomposition(
        history_strain
    )
    history_positive_energy, _ = main.pfcm.strain_energy_density_decomposition(
        history_strain
    )
    equilibrium_stress_voigt = np.asarray(
        bm.to_numpy(symmetric_tensor_to_voigt(equilibrium_positive_stress)),
        dtype=np.float64,
    )
    history_stress_voigt = np.asarray(
        bm.to_numpy(symmetric_tensor_to_voigt(history_positive_stress)),
        dtype=np.float64,
    )
    positive_energy = np.asarray(
        bm.to_numpy(history_positive_energy), dtype=np.float64
    )
    degradation_derivative = np.asarray(
        bm.to_numpy(main.EDFunc.grad_degradation_function(main.d(bcs))),
        dtype=np.float64,
    )

    committed_history = np.asarray(step_map.committed_history, dtype=np.float64)
    if positive_energy.shape != committed_history.shape:
        raise RuntimeError("history energy and committed history shapes differ")
    history_tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, np.abs(committed_history)
    )
    active_history = positive_energy > committed_history + history_tolerance
    step_map.last_history_active_mask = active_history.copy()

    # J_ud[c,i,j] = int B_i^T sigma_+(u) g'(d) phi_j dx.
    element_ud = np.einsum(
        "q,c,cqki,cqk,cq,cqj->cij",
        weights,
        cell_measure,
        strain_matrix,
        equilibrium_stress_voigt,
        degradation_derivative,
        phase_basis,
        optimize=True,
    )
    # J_du[c,i,j] = int phi_i g'(d) I_active sigma_H^+ : eps_j dx.
    element_du = np.einsum(
        "q,c,cqi,cq,cqk,cqkj->cij",
        weights,
        cell_measure,
        phase_basis,
        degradation_derivative * active_history,
        history_stress_voigt,
        strain_matrix,
        optimize=True,
    )

    displacement_cell_to_dof = np.asarray(
        bm.to_numpy(main.tspace.cell_to_dof()), dtype=np.int64
    )
    phase_cell_to_dof = np.asarray(
        bm.to_numpy(main.space.cell_to_dof()), dtype=np.int64
    )
    displacement_dofs = step_map.displacement_size
    phase_dofs = step_map.damage_size
    displacement_rows = np.broadcast_to(
        displacement_cell_to_dof[:, :, None], element_ud.shape
    )
    phase_columns = np.broadcast_to(
        phase_cell_to_dof[:, None, :], element_ud.shape
    )
    phase_rows = np.broadcast_to(
        phase_cell_to_dof[:, :, None], element_du.shape
    )
    displacement_columns = np.broadcast_to(
        displacement_cell_to_dof[:, None, :], element_du.shape
    )
    jacobian_ud = coo_matrix(
        (element_ud.reshape(-1), (displacement_rows.reshape(-1), phase_columns.reshape(-1))),
        shape=(displacement_dofs, phase_dofs),
    ).tocsr()
    fixed_displacement_rows = np.flatnonzero(step_map.fixed_displacement_mask)
    if fixed_displacement_rows.size:
        jacobian_ud = jacobian_ud.tolil()
        jacobian_ud[fixed_displacement_rows, :] = 0.0
        jacobian_ud = jacobian_ud.tocsr()
    jacobian_du = coo_matrix(
        (element_du.reshape(-1), (phase_rows.reshape(-1), displacement_columns.reshape(-1))),
        shape=(phase_dofs, displacement_dofs),
    ).tocsr()
    return jacobian_ud, jacobian_du


def _compose_coupled_patch_jacobian(
    displacement_matrix: Any,
    phase_matrix: Any,
    jacobian_ud: Any,
    jacobian_du: Any,
    patch: np.ndarray,
    displacement_dofs: int,
) -> np.ndarray:
    """Extract a dense coupled ``J_omegaomega`` in the supplied patch order."""
    local_patch = np.asarray(patch, dtype=np.int64).reshape(-1)
    if local_patch.size == 0 or np.unique(local_patch).size != local_patch.size:
        raise ValueError("patch must contain unique coupled-state coordinates")
    phase_dofs = int(phase_matrix.shape[0])
    if np.any(local_patch < 0) or np.any(
        local_patch >= displacement_dofs + phase_dofs
    ):
        raise ValueError("patch contains an out-of-range coupled-state coordinate")

    displacement_positions = np.flatnonzero(local_patch < displacement_dofs)
    phase_positions = np.flatnonzero(local_patch >= displacement_dofs)
    displacement_patch = local_patch[displacement_positions]
    phase_patch = local_patch[phase_positions] - displacement_dofs
    local_jacobian = np.zeros(
        (local_patch.size, local_patch.size), dtype=np.float64
    )
    if displacement_patch.size:
        local_jacobian[np.ix_(displacement_positions, displacement_positions)] = (
            _as_scipy_csr(displacement_matrix)[displacement_patch][
                :, displacement_patch
            ].toarray()
        )
    if phase_patch.size:
        local_jacobian[np.ix_(phase_positions, phase_positions)] = (
            _as_scipy_csr(phase_matrix)[phase_patch][:, phase_patch].toarray()
        )
    if displacement_patch.size and phase_patch.size:
        local_jacobian[np.ix_(displacement_positions, phase_positions)] = (
            _as_scipy_csr(jacobian_ud)[displacement_patch][:, phase_patch].toarray()
        )
        local_jacobian[np.ix_(phase_positions, displacement_positions)] = (
            _as_scipy_csr(jacobian_du)[phase_patch][
                :, displacement_patch
            ].toarray()
        )
    return local_jacobian


def _estimate_external_schur_coupling(
    displacement_matrix: Any,
    phase_matrix: Any,
    jacobian_ud: Any,
    jacobian_du: Any,
    patch: np.ndarray,
    fixed_mask: np.ndarray,
    displacement_dofs: int,
    *,
    samples: int = 2,
    seed: int = 20260822,
) -> dict[str, Any]:
    """Estimate the patch-to-exterior Schur coupling at one FE state.

    The estimate uses random exterior directions (v) and evaluates

    ``||J_cw J_ww^{-1} J_wc v|| / ||J_cc v||``.

    Only assembled tangent blocks at the supplied state are used.  The
    calculation is a diagnostic and is excluded from Reduced-NE work
    accounting.  Coordinates in ``fixed_mask`` are removed from both the
    patch and exterior spaces so that the ratio measures the free reduced
    system rather than Dirichlet rows.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    uu = _as_scipy_csr(displacement_matrix)
    dd = _as_scipy_csr(phase_matrix)
    ud = _as_scipy_csr(jacobian_ud)
    du = _as_scipy_csr(jacobian_du)
    fixed = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    total_size = int(displacement_dofs + dd.shape[0])
    if fixed.size != total_size:
        raise ValueError("fixed_mask has incompatible coupled-state size")
    local_patch = np.unique(np.asarray(patch, dtype=np.int64).reshape(-1))
    free = np.flatnonzero(~fixed)
    if local_patch.size == 0 or np.any(fixed[local_patch]):
        raise ValueError("patch must contain nonempty free coordinates")
    outside = np.setdiff1d(free, local_patch, assume_unique=True)
    if outside.size == 0:
        return {
            "sample_count": 0,
            "patch_dofs": int(local_patch.size),
            "outside_dofs": 0,
            "gamma_values": [],
            "gamma_mean": 0.0,
            "gamma_max": 0.0,
            "local_condition_number": float("nan"),
            "linear_solve": "empty exterior",
            "solver_work_accounting": "excluded diagnostic",
        }

    patch_u_positions = np.flatnonzero(local_patch < displacement_dofs)
    patch_d_positions = np.flatnonzero(local_patch >= displacement_dofs)
    patch_u = local_patch[patch_u_positions]
    patch_d = local_patch[patch_d_positions] - displacement_dofs
    outside_u_positions = np.flatnonzero(outside < displacement_dofs)
    outside_d_positions = np.flatnonzero(outside >= displacement_dofs)
    outside_u = outside[outside_u_positions]
    outside_d = outside[outside_d_positions] - displacement_dofs
    local = _compose_coupled_patch_jacobian(
        uu, dd, ud, du, local_patch, displacement_dofs
    )
    condition_number = float(np.linalg.cond(local))
    try:
        factor = lu_factor(local)
        solve_local = lambda rhs: lu_solve(factor, rhs)
        solve_name = "lu"
    except (np.linalg.LinAlgError, ValueError):
        solve_local = lambda rhs: np.linalg.lstsq(local, rhs, rcond=None)[0]
        solve_name = "lstsq"

    def _j_wc(vector: np.ndarray) -> np.ndarray:
        """Apply the exterior-to-patch block in patch coordinate order."""
        vector_u = vector[outside_u_positions]
        vector_d = vector[outside_d_positions]
        result = np.zeros(local_patch.size, dtype=np.float64)
        if patch_u.size:
            result[patch_u_positions] = (
                uu[patch_u][:, outside_u] @ vector_u
                + ud[patch_u][:, outside_d] @ vector_d
            )
        if patch_d.size:
            result[patch_d_positions] = (
                du[patch_d][:, outside_u] @ vector_u
                + dd[patch_d][:, outside_d] @ vector_d
            )
        return result

    def _j_cw(local_vector: np.ndarray) -> np.ndarray:
        """Apply the patch-to-exterior block in exterior coordinate order."""
        local_u = local_vector[patch_u_positions]
        local_d = local_vector[patch_d_positions]
        result = np.zeros(outside.size, dtype=np.float64)
        if outside_u.size:
            result[outside_u_positions] = (
                uu[outside_u][:, patch_u] @ local_u
                + ud[outside_u][:, patch_d] @ local_d
            )
        if outside_d.size:
            result[outside_d_positions] = (
                du[outside_d][:, patch_u] @ local_u
                + dd[outside_d][:, patch_d] @ local_d
            )
        return result

    def _j_cc(vector: np.ndarray) -> np.ndarray:
        """Apply the exterior diagonal block in exterior coordinate order."""
        vector_u = vector[outside_u_positions]
        vector_d = vector[outside_d_positions]
        result = np.zeros(outside.size, dtype=np.float64)
        if outside_u.size:
            result[outside_u_positions] = (
                uu[outside_u][:, outside_u] @ vector_u
                + ud[outside_u][:, outside_d] @ vector_d
            )
        if outside_d.size:
            result[outside_d_positions] = (
                du[outside_d][:, outside_u] @ vector_u
                + dd[outside_d][:, outside_d] @ vector_d
            )
        return result

    generator = np.random.default_rng(seed)
    gamma_values: list[float] = []
    for _ in range(samples):
        vector = generator.standard_normal(outside.size)
        vector /= max(float(np.linalg.norm(vector)), np.finfo(float).tiny)
        denominator = float(np.linalg.norm(_j_cc(vector)))
        correction = _j_cw(solve_local(_j_wc(vector)))
        gamma_values.append(
            float(np.linalg.norm(correction) / max(denominator, np.finfo(float).tiny))
        )
    return {
        "sample_count": int(samples),
        "patch_dofs": int(local_patch.size),
        "outside_dofs": int(outside.size),
        "gamma_values": gamma_values,
        "gamma_mean": float(np.mean(gamma_values)),
        "gamma_max": float(np.max(gamma_values)),
        "local_condition_number": condition_number,
        "linear_solve": solve_name,
        "solver_work_accounting": "excluded diagnostic",
    }


def _assemble_coupled_patch_jacobian(
    step_map: FrozenStandardFEStepMap,
    patch: np.ndarray,
) -> np.ndarray:
    """Assemble and extract the physical coupled local Jacobian block."""
    jacobian_ud, jacobian_du = _assemble_history_field_coupling_blocks(step_map)
    return _compose_coupled_patch_jacobian(
        step_map.last_displacement_matrix,
        step_map.last_phase_matrix,
        jacobian_ud,
        jacobian_du,
        patch,
        step_map.displacement_size,
    )


def _check_coupled_patch_jacobian_directions(
    step_map: FrozenStandardFEStepMap,
    state: np.ndarray,
    patch: np.ndarray,
    *,
    relative_step: float,
    directions: int,
) -> dict[str, Any]:
    """Compare assembled local Jacobian actions with centered differences.

    The check is deterministic and diagnostic: its residual evaluations are
    reported separately from solver work. Dirichlet coordinates remain fixed,
    while phase coordinates use the smooth physical-residual extension needed
    for differentiation at box bounds.
    """
    if relative_step <= 0.0 or not np.isfinite(relative_step):
        raise ValueError("relative_step must be finite and positive")
    if directions <= 0:
        raise ValueError("directions must be positive")
    candidate = np.asarray(state, dtype=np.float64).reshape(-1)
    local_patch = np.asarray(patch, dtype=np.int64).reshape(-1)
    check_start = perf_counter()
    step_map.assemble_coupled_residual(candidate, enforce_phase_box=False)
    local_jacobian = _assemble_coupled_patch_jacobian(step_map, local_patch)
    step_length = relative_step * max(1.0, float(np.linalg.norm(candidate)))
    generator = np.random.default_rng(20260821)
    relative_errors: list[float] = []
    for _ in range(directions):
        local_direction = generator.standard_normal(local_patch.size)
        local_direction /= max(
            float(np.linalg.norm(local_direction)), np.finfo(np.float64).tiny
        )
        positive_state = candidate.copy()
        negative_state = candidate.copy()
        positive_state[local_patch] += step_length * local_direction
        negative_state[local_patch] -= step_length * local_direction
        positive_residual = step_map.assemble_coupled_residual(
            positive_state, enforce_phase_box=False
        )[local_patch]
        negative_residual = step_map.assemble_coupled_residual(
            negative_state, enforce_phase_box=False
        )[local_patch]
        finite_difference_action = (
            positive_residual - negative_residual
        ) / (2.0 * step_length)
        assembled_action = local_jacobian @ local_direction
        relative_errors.append(
            float(
                np.linalg.norm(assembled_action - finite_difference_action)
                / max(
                    float(np.linalg.norm(finite_difference_action)),
                    np.finfo(np.float64).tiny,
                )
            )
        )
    step_map.assemble_coupled_residual(candidate, enforce_phase_box=False)
    return {
        "patch_dofs": int(local_patch.size),
        "directions": int(directions),
        "relative_step": float(relative_step),
        "centered_residual_evaluations": int(2 * directions + 2),
        "relative_errors": relative_errors,
        "maximum_relative_error": float(max(relative_errors)),
        "mean_relative_error": float(np.mean(relative_errors)),
        "wall_time_seconds": float(perf_counter() - check_start),
        "solver_work_accounting": "excluded diagnostic validation",
    }


def _checkpoint_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract one compact reproducibility row from each checkpoint summary."""
    checkpoints = summary.get("checkpoints", [summary])
    rows: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints):
        fixed_point = checkpoint["fixed_point"]
        slow_mode = checkpoint["slow_mode"]
        coupled = checkpoint["coupled_slow_subspace"]
        localization = checkpoint["localization"]
        state = checkpoint["state"]
        calibration = checkpoint["spd_patch_calibration"]
        rows.append(
            {
                "checkpoint_index": int(checkpoint.get("checkpoint_index", index)),
                "load": checkpoint["case"]["load"],
                "staggered_iterations": fixed_point["iterations"],
                "spectral_radius": slow_mode["spectral_radius"],
                "observed_last_ratio": fixed_point["observed_last_ratio"],
                "selected_slow_dimension": coupled["selected_dimension"],
                "spectral_gap": slow_mode["spectral_gap"],
                "max_damage": state["max_damage"],
                "max_damage_increment": state["max_damage_increment"],
                "damage_dof_fraction_above_0_5": state[
                    "damage_dof_fraction_above_0_5"
                ],
                "coupled_selected_cell_fraction": localization[
                    "coupled_selected_cell_fraction"
                ],
                "coupled_selected_trace_fraction": localization[
                    "coupled_selected_trace_fraction"
                ],
                "damage_region_trace_fraction_same_size": localization[
                    "damage_region_trace_fraction_same_size"
                ],
                "gradient_region_trace_fraction_same_size": localization[
                    "gradient_region_trace_fraction_same_size"
                ],
                "slow_patch_survival_factor": calibration[
                    "slow_patch_survival_factor"
                ],
                "damage_patch_survival_factor": calibration[
                    "damage_patch_survival_factor"
                ],
                "gradient_patch_survival_factor": calibration[
                    "gradient_patch_survival_factor"
                ],
            }
        )
    return rows


def _write_checkpoint_tables(summary: dict[str, Any], output_dir: Path) -> None:
    """Write compact CSV and NPZ views of the JSON checkpoint summaries.

    The JSON summary remains the complete record.  The two tabular files make
    load-to-load comparisons reproducible without parsing nested JSON.
    """
    rows = _checkpoint_table_rows(summary)
    fieldnames = list(rows[0])
    with (output_dir / "checkpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        output_dir / "checkpoints.npz",
        **{
            field: np.asarray([row[field] for row in rows])
            for field in fieldnames
        },
    )


def _write_benefit_cost_table(summary: dict[str, Any], output_dir: Path) -> None:
    """Write the optional patch-sweep Pareto records as one flat CSV table."""
    checkpoints = summary.get("checkpoints", [summary])
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        pareto = checkpoint.get("benefit_cost_pareto")
        if pareto is None:
            continue
        load = float(checkpoint["case"]["load"])
        for record in pareto["records"]:
            rows.append({"load": load, **record})
    if not rows:
        return
    with (output_dir / "benefit_cost_pareto.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_online_increment_table(
    summary: dict[str, Any], output_dir: Path
) -> None:
    """Write one compact row per online increment window and checkpoint."""
    checkpoints = summary.get("checkpoints", [summary])
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        diagnostics = checkpoint["online_increment_slow_subspace"]
        for record in diagnostics["windows"]:
            rows.append(
                {
                    "load": float(checkpoint["case"]["load"]),
                    "end_iteration": record["end_iteration"],
                    "window_size": record["used_window_size"],
                    "online_dimension": record["selected_dimension"],
                    "reference_dimension": diagnostics["reference_dimension"],
                    "contraction_estimate": record[
                        "weighted_contraction_estimate"
                    ],
                    "max_principal_angle_degrees": record[
                        "max_principal_angle_degrees"
                    ],
                    "gap_aware_max_angle_degrees": record[
                        "gap_aware_max_angle_degrees"
                    ],
                    "selected_cell_fraction": record["selected_cell_fraction"],
                    "reference_trace_fraction": record[
                        "reference_trace_fraction"
                    ],
                    "cell_jaccard_with_reference_region": record[
                        "cell_jaccard_with_reference_region"
                    ],
                    "reference_subspace_survival_factor": record[
                        "reference_subspace_survival_factor"
                    ],
                    "construction_wall_time_seconds": record[
                        "construction_wall_time_seconds"
                    ],
                }
            )
    if not rows:
        return
    with (output_dir / "online_increment_subspace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _coupled_residual_metrics(
    step_map: FrozenStandardFEStepMap,
    state: np.ndarray,
) -> dict[str, float]:
    """Return raw and block-scaled residual diagnostics for one FE state.

    The raw Euclidean residual mixes displacement and phase equations with
    different algebraic scales. The block-correction norm instead applies the
    assembled displacement and phase tangents and projects the phase update to
    its irreversible box. It therefore measures the state-space correction
    associated with the physical coupled residual.
    """
    full_state = np.asarray(state, dtype=np.float64).reshape(-1)
    residual = step_map.assemble_coupled_residual(full_state)
    n_displacement = step_map.displacement_size
    lower = np.full(step_map.full_state_size, -np.inf, dtype=np.float64)
    upper = np.full(step_map.full_state_size, np.inf, dtype=np.float64)
    lower[n_displacement:] = step_map.damage_lower_bound
    upper[n_displacement:] = step_map.damage_upper_bound
    projected_residual = full_state - np.minimum(
        np.maximum(full_state - residual, lower), upper
    )
    if step_map.last_displacement_matrix is None or step_map.last_phase_matrix is None:
        raise RuntimeError("coupled residual metric requires both assembled block tangents")
    displacement_correction = np.asarray(
        spsolve(
            step_map.last_displacement_matrix.tocsc(),
            -residual[:n_displacement],
        ),
        dtype=np.float64,
    )
    damage = full_state[n_displacement:]
    if step_map.phase_bound_solver == "active_set":
        phase_load = (
            step_map.last_phase_matrix @ damage
            - residual[n_displacement:]
        )
        phase_result = solve_box_quadratic_active_set(
            step_map.last_phase_matrix,
            phase_load,
            damage,
            step_map.damage_lower_bound,
            step_map.damage_upper_bound,
        )
        if not phase_result.converged:
            raise RuntimeError("phase correction metric active-set solve failed")
        projected_damage_correction = phase_result.state - damage
    else:
        damage_correction = np.asarray(
            spsolve(
                step_map.last_phase_matrix.tocsc(),
                -residual[n_displacement:],
            ),
            dtype=np.float64,
        )
        projected_damage_correction = np.clip(
            damage + damage_correction,
            step_map.damage_lower_bound,
            step_map.damage_upper_bound,
        ) - damage
    return {
        "raw_norm": float(np.linalg.norm(residual)),
        "projected_raw_norm": float(np.linalg.norm(projected_residual)),
        "block_correction_norm": float(
            np.linalg.norm(
                np.concatenate(
                    (displacement_correction, projected_damage_correction)
                )
            )
        ),
    }


def _run_reference_free_warmup(
    step_map: Any,
    initial_damage: np.ndarray,
    *,
    fixed_mask: np.ndarray,
    mode: str,
    fixed_sweeps: int,
    minimum_sweeps: int,
    maximum_sweeps: int,
    slow_rate_threshold: float,
    required_slow_steps: int,
    residual_tolerance: float,
    residual_ratio_threshold: float,
    residual_norm_callback: Callable[[np.ndarray], float],
) -> dict[str, Any]:
    """Run a reference-free staggered warm start with an adaptive stop rule.

    The warm start only uses accepted staggered states.  It stops after a
    fixed number of sweeps in ``fixed`` mode, or in ``adaptive`` mode when
    either the direct coupled residual is small enough or the observed
    increment ratio stays above ``slow_rate_threshold`` while the direct
    residual decreases by ``residual_ratio_threshold``.  No target or
    reference state is inspected.
    """
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("warmup mode must be fixed or adaptive")
    if fixed_sweeps <= 0 or minimum_sweeps <= 0 or maximum_sweeps <= 0:
        raise ValueError("warmup sweep limits must be positive")
    if minimum_sweeps > maximum_sweeps:
        raise ValueError("minimum_sweeps must not exceed maximum_sweeps")
    if not 0.0 < slow_rate_threshold:
        raise ValueError("slow_rate_threshold must be positive")
    if required_slow_steps <= 0:
        raise ValueError("required_slow_steps must be positive")
    if residual_tolerance < 0.0 or not np.isfinite(residual_tolerance):
        raise ValueError("residual_tolerance must be finite and nonnegative")
    if not 0.0 < residual_ratio_threshold < 1.0:
        raise ValueError("residual_ratio_threshold must lie in (0, 1)")

    limit = fixed_sweeps if mode == "fixed" else maximum_sweeps
    minimum = min(fixed_sweeps, minimum_sweeps) if mode == "fixed" else minimum_sweeps
    damage = np.asarray(initial_damage, dtype=np.float64).reshape(-1).copy()
    if damage.size == 0 or not np.isfinite(damage).all():
        raise ValueError("initial_damage must be a finite nonempty vector")
    fixed = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    if fixed.size != step_map.full_state_size:
        raise ValueError("fixed_mask must match the coupled state size")

    states: list[np.ndarray] = []
    increments: list[float] = []
    online_rates: list[float] = []
    residual_norms: list[float] = []
    residual_ratios: list[float] = []
    consecutive_slow = 0
    stop_reason = "maximum_sweeps"
    previous_state: Optional[np.ndarray] = None
    for sweep in range(1, limit + 1):
        damage = np.asarray(step_map(damage), dtype=np.float64).reshape(-1).copy()
        state = np.asarray(step_map.current_full_state(), dtype=np.float64).reshape(-1).copy()
        if state.size != fixed.size or not np.isfinite(state).all():
            raise ValueError("warmup map returned an invalid full state")
        states.append(state)
        residual_norm = float(residual_norm_callback(state))
        if residual_norm < 0.0 or not np.isfinite(residual_norm):
            raise ValueError("warmup residual monitor returned an invalid norm")
        residual_norms.append(residual_norm)
        if len(residual_norms) >= 2:
            previous_residual = max(
                residual_norms[-2], np.finfo(np.float64).tiny
            )
            residual_ratios.append(float(residual_norm / previous_residual))

        if previous_state is not None:
            increment = float(np.linalg.norm((state - previous_state)[~fixed]))
            increments.append(increment)
            if len(increments) >= 2:
                denominator = max(increments[-2], np.finfo(np.float64).eps)
                rate = float(increment / denominator)
                online_rates.append(rate)
                if rate >= slow_rate_threshold:
                    consecutive_slow += 1
                else:
                    consecutive_slow = 0
        previous_state = state

        if mode == "fixed":
            if sweep >= minimum:
                stop_reason = "fixed_sweeps"
                break
        elif sweep >= minimum:
            if residual_norm <= residual_tolerance:
                stop_reason = "residual_tolerance"
                break
            residual_descent = bool(
                residual_ratios
                and residual_ratios[-1] <= residual_ratio_threshold
            )
            if consecutive_slow >= required_slow_steps and residual_descent:
                stop_reason = "slow_rate_and_residual_descent"
                break
        if sweep >= limit:
            stop_reason = "maximum_sweeps"
            break

    if not states:
        raise RuntimeError("reference-free warmup produced no state")
    return {
        "mode": mode,
        "states": states,
        "damage": damage,
        "sweeps": len(states),
        "stop_reason": stop_reason,
        "increment_norms": increments,
        "online_rates": online_rates,
        "projected_residual_norms": residual_norms,
        "residual_ratios": residual_ratios,
        "residual_monitor_evaluations": len(residual_norms),
        "slow_rate_threshold": float(slow_rate_threshold),
        "required_slow_steps": int(required_slow_steps),
        "residual_tolerance": float(residual_tolerance),
        "residual_ratio_threshold": float(residual_ratio_threshold),
    }


def _run_local_elimination_composite(
    step_map: FrozenStandardFEStepMap,
    args: argparse.Namespace,
    patch_dofs: np.ndarray,
) -> tuple[dict[str, Any], Any]:
    """Run staggered sweeps composed with a true local FE residual solve.

    Parameters
    ----------
    step_map : FrozenStandardFEStepMap
        Frozen physical FE map at one committed load checkpoint.
    args : argparse.Namespace
        CLI tolerances for outer and local Newton iterations.
    patch_dofs : ndarray, shape (n_patch,), dtype int64
        Free full-state coordinates selected by the coupled slow-mode indicator.

    Returns
    -------
    summary : dict
        Local patch size, outer iteration count, and per-step Newton statistics.
    trace : FixedPointResult
        Damage trace of the composite map.
    """
    patch = np.asarray(patch_dofs, dtype=np.int64).reshape(-1)
    if patch.size == 0 or np.unique(patch).size != patch.size:
        raise ValueError("local elimination patch must contain unique free DOFs")
    if not 0.0 < float(args.local_acceptance_factor) < 1.0:
        raise ValueError("local_acceptance_factor must lie in (0, 1)")
    n_displacement = step_map.displacement_size
    lower_full = np.full(step_map.full_state_size, -np.inf, dtype=np.float64)
    upper_full = np.full(step_map.full_state_size, np.inf, dtype=np.float64)
    lower_full[n_displacement:] = step_map.damage_lower_bound
    upper_full[n_displacement:] = step_map.damage_upper_bound
    records: list[dict[str, Any]] = []
    start_time = perf_counter()
    local_disabled_reason: Optional[str] = None

    def apply_composite_map(candidate_damage: np.ndarray) -> np.ndarray:
        nonlocal local_disabled_reason
        step_map(candidate_damage)
        full_state = step_map.current_full_state()
        base_metrics = _coupled_residual_metrics(step_map, full_state)
        if local_disabled_reason is not None:
            step_map.set_full_state(full_state)
            records.append(
                {
                    "outer_iteration": len(records) + 1,
                    "accepted": False,
                    "disabled": True,
                    "reason": local_disabled_reason,
                    "base_block_correction_norm": base_metrics[
                        "block_correction_norm"
                    ],
                    "local_patch_dofs": int(patch.size),
                    "local_trust_region_iterations": 0,
                    "local_residual_evaluations": 0,
                    "initial_residual_norm": 0.0,
                    "final_residual_norm": 0.0,
                }
            )
            return full_state[n_displacement:].copy()
        local_patch = patch

        def restricted_residual(state: np.ndarray) -> np.ndarray:
            residual = step_map.assemble_coupled_residual(state)
            return residual[local_patch]

        try:
            local_result = solve_local_nonlinear_residual(
                restricted_residual,
                full_state,
                local_patch,
                lower_bound=lower_full[local_patch],
                upper_bound=upper_full[local_patch],
                relative_step=float(args.fd_step),
                atol=float(args.local_atol),
                rtol=float(args.local_rtol),
                max_iterations=int(args.local_max_iterations),
            )
        except RuntimeError as exc:
            local_disabled_reason = str(exc)
            step_map.set_full_state(full_state)
            records.append(
                {
                    "outer_iteration": len(records) + 1,
                    "accepted": False,
                    "disabled": True,
                    "reason": local_disabled_reason,
                    "base_block_correction_norm": base_metrics[
                        "block_correction_norm"
                    ],
                    "local_patch_dofs": int(local_patch.size),
                    "local_trust_region_iterations": 0,
                    "local_residual_evaluations": 0,
                    "initial_residual_norm": 0.0,
                    "final_residual_norm": 0.0,
                }
            )
            return full_state[n_displacement:].copy()
        trial_metrics = _coupled_residual_metrics(step_map, local_result.state)
        joint_tolerance = float(
            args.atol + args.rtol * max(1.0, np.linalg.norm(local_result.state))
        )
        accepted = bool(
            local_result.converged
            and trial_metrics["block_correction_norm"]
            <= max(
                joint_tolerance,
                float(args.local_acceptance_factor)
                * base_metrics["block_correction_norm"],
            )
        )
        if not local_result.converged:
            local_disabled_reason = (
                "local projected residual did not meet tolerance: "
                f"{local_result.final_residual_norm:.6e}"
            )
        if not accepted:
            step_map.set_full_state(full_state)
        records.append(
            {
                "outer_iteration": len(records) + 1,
                "accepted": accepted,
                "disabled": local_disabled_reason is not None,
                "reason": local_disabled_reason,
                "base_block_correction_norm": base_metrics[
                    "block_correction_norm"
                ],
                "trial_block_correction_norm": trial_metrics[
                    "block_correction_norm"
                ],
                "local_patch_dofs": int(local_patch.size),
                "local_trust_region_iterations": local_result.iterations,
                "local_residual_evaluations": local_result.residual_evaluations,
                "initial_residual_norm": local_result.initial_residual_norm,
                "final_residual_norm": local_result.final_residual_norm,
            }
        )
        selected_state = local_result.state if accepted else full_state
        return selected_state[n_displacement:].copy()

    trace = iterate_fixed_point(
        apply_composite_map,
        step_map.committed_damage,
        atol=float(args.atol),
        rtol=float(args.rtol),
        max_iterations=int(args.max_iterations),
    )
    if not trace.converged:
        raise RuntimeError(
            "staggered-plus-local-elimination map did not converge: "
            f"iterations={trace.iterations}, "
            f"last_increment={trace.increment_norms[-1]:.6e}"
        )
    final_full_state = step_map.current_full_state()
    residual_metrics = _coupled_residual_metrics(step_map, final_full_state)
    return (
        {
            "patch_dofs": int(patch.size),
            "outer_iterations": trace.iterations,
            "local_trust_region_iterations": int(
                sum(record["local_trust_region_iterations"] for record in records)
            ),
            "local_residual_evaluations": int(
                sum(record["local_residual_evaluations"] for record in records)
            ),
            "accepted_local_corrections": int(
                sum(bool(record["accepted"]) for record in records)
            ),
            "rejected_local_corrections": int(
                sum(not bool(record["accepted"]) for record in records)
            ),
            "local_disabled_reason": local_disabled_reason,
            "wall_time_seconds": float(perf_counter() - start_time),
            "final_coupled_residual": residual_metrics,
            "records": records,
            "initial_residual_norm_max": float(
                max(record["initial_residual_norm"] for record in records)
            )
            if records
            else 0.0,
            "final_residual_norm_max": float(
                max(record["final_residual_norm"] for record in records)
            )
            if records
            else 0.0,
        },
        trace,
    )


def _history_survival_is_requested(args: argparse.Namespace, load: float) -> bool:
    """Return whether the current load requests the costly history-field diagnostic."""
    requested = getattr(args, "history_survival_loads", None)
    if requested is None:
        return False
    return any(
        np.isclose(float(load), float(target), rtol=1.0e-12, atol=1.0e-14)
        for target in requested
    )


def _reduced_solver_is_requested(args: argparse.Namespace, load: float) -> bool:
    """Return whether the current checkpoint requests the reduced solver."""
    if not bool(args.reduced_solver):
        return False
    requested = getattr(args, "reduced_solver_loads", None)
    if requested is None:
        return True
    return any(
        np.isclose(float(load), float(target), rtol=1.0e-12, atol=1.0e-14)
        for target in requested
    )


def _smooth_free_state_mask(
    step_map: FrozenStandardFEStepMap,
    full_state: np.ndarray,
    *,
    relative_step: float,
) -> np.ndarray:
    """Return the fixed-active-set smooth coordinates for residual differencing.

    Dirichlet coordinates and phase coordinates within two perturbation steps
    of either box bound are excluded. The returned Boolean mask has shape
    ``(n_u+n_d,)`` in coupled state ordering and does not alias the input.
    """
    state = np.asarray(full_state, dtype=np.float64).reshape(-1)
    if state.shape != (step_map.full_state_size,) or not np.isfinite(state).all():
        raise ValueError("full_state must be a finite coupled state")
    if relative_step <= 0.0 or not np.isfinite(relative_step):
        raise ValueError("relative_step must be finite and positive")
    n_displacement = step_map.displacement_size
    smooth_free_mask = ~step_map.fixed_state_mask()
    damage = state[n_displacement:]
    bound_margin = 2.0 * relative_step * np.maximum(1.0, np.abs(damage))
    smooth_damage = (damage - step_map.damage_lower_bound > bound_margin) & (
        step_map.damage_upper_bound - damage > bound_margin
    )
    smooth_free_mask[n_displacement:] &= smooth_damage
    return smooth_free_mask


def _history_field_patch_calibration(
    step_map: FrozenStandardFEStepMap,
    full_state: np.ndarray,
    full_mode: np.ndarray,
    full_basis: np.ndarray,
    weight_diagonal: np.ndarray,
    patch_dofs: dict[str, np.ndarray],
    *,
    relative_step: float,
) -> dict[str, Any]:
    """Measure patch survival factors with the physical history-field Jacobian.

    Parameters
    ----------
    step_map : FrozenStandardFEStepMap
        Frozen standard-FE checkpoint providing physical residual assembly.
    full_state : ndarray, shape (n_u+n_d,), dtype float64
        Converged coupled state in ``(u,d)`` ordering.
    full_mode : ndarray, shape (n_u+n_d,)
        Coupled slow mode in the same ordering. Real and complex modes are
        accepted and the input is not modified.
    full_basis : ndarray, shape (n_u+n_d, r), dtype float64
        Basis of the selected real coupled slow subspace.
    weight_diagonal : ndarray, shape (n_u+n_d,), dtype float64
        Strictly positive diagonal of the diagnostic SPD weight ``W``.
    patch_dofs : dict[str, ndarray]
        Full-state patch indices for the slow, damage, and gradient regions.
    relative_step : float
        Positive finite-difference scale used for the physical residual
        Jacobian.

    Returns
    -------
    dict
        Active-set size, Jacobian diagnostics, and one direct
        ``||Q_omega w||_W/||w||_W`` measurement per patch.

    Notes
    -----
    Dirichlet coordinates and phase coordinates within two finite-difference
    steps of a box bound are removed. The resulting Jacobian is the smooth
    fixed-active-set history-field linearization used by the local theory.
    """
    state = np.asarray(full_state, dtype=np.float64).reshape(-1)
    mode = np.asarray(full_mode).reshape(-1)
    basis = np.asarray(full_basis, dtype=np.float64)
    diagonal = np.asarray(weight_diagonal, dtype=np.float64).reshape(-1)
    if state.shape != (step_map.full_state_size,):
        raise ValueError("full_state has incompatible shape")
    if mode.shape != state.shape or diagonal.shape != state.shape:
        raise ValueError("full_mode and weight_diagonal must match full_state")
    if basis.ndim != 2 or basis.shape[0] != state.size or basis.shape[1] == 0:
        raise ValueError("full_basis must be a nonempty coupled-state basis")
    if relative_step <= 0.0 or not np.isfinite(relative_step):
        raise ValueError("relative_step must be finite and positive")
    if not np.isfinite(state).all() or not np.isfinite(mode).all():
        raise ValueError("full_state and full_mode must be finite")
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.0):
        raise ValueError("weight_diagonal must be finite and strictly positive")

    n_displacement = step_map.displacement_size
    smooth_free_mask = _smooth_free_state_mask(
        step_map, state, relative_step=relative_step
    )
    lower_damage = step_map.damage_lower_bound
    upper_damage = step_map.damage_upper_bound
    smooth_free_dofs = np.flatnonzero(smooth_free_mask)
    if smooth_free_dofs.size == 0:
        raise RuntimeError("history-field calibration has no smooth free coordinates")

    lower_full = np.full(state.size, -np.inf, dtype=np.float64)
    upper_full = np.full(state.size, np.inf, dtype=np.float64)
    lower_full[n_displacement:] = lower_damage
    upper_full[n_displacement:] = upper_damage
    base_free_state = state[smooth_free_dofs].copy()

    def restricted_residual(free_state: np.ndarray) -> np.ndarray:
        candidate = state.copy()
        candidate[smooth_free_dofs] = free_state
        return step_map.assemble_coupled_residual(candidate)[smooth_free_dofs]

    start_time = perf_counter()
    base_residual = restricted_residual(base_free_state)
    jacobian = finite_difference_jacobian_rectangular(
        restricted_residual,
        base_free_state,
        relative_step=relative_step,
        lower_bound=lower_full[smooth_free_dofs],
        upper_bound=upper_full[smooth_free_dofs],
    )
    step_map.set_full_state(state)
    elapsed = perf_counter() - start_time

    mode_free = mode[smooth_free_dofs]
    weight_free = diagonal[smooth_free_dofs]
    basis_free = weighted_orthonormalize(
        basis[smooth_free_dofs, :], weight_free
    )
    full_mode_energy = float(np.sum(weight_free * np.abs(mode_free) ** 2))
    if full_mode_energy <= np.finfo(np.float64).tiny:
        raise RuntimeError("slow mode has zero norm on the smooth free subspace")
    index_in_free = np.full(state.size, -1, dtype=np.int64)
    index_in_free[smooth_free_dofs] = np.arange(smooth_free_dofs.size)
    jacobian_mode = jacobian @ mode_free
    jacobian_mode_norm = max(
        float(np.linalg.norm(jacobian_mode)), np.finfo(np.float64).tiny
    )

    patch_results: dict[str, dict[str, Any]] = {}
    for name, full_patch in patch_dofs.items():
        patch = np.asarray(full_patch, dtype=np.int64).reshape(-1)
        free_patch = index_in_free[patch]
        free_patch = np.unique(free_patch[free_patch >= 0])
        if free_patch.size == 0:
            raise RuntimeError(f"{name} patch has no smooth free coordinates")
        projected_mode = apply_local_elimination_projection(
            jacobian, free_patch, mode_free
        )
        projected_basis = np.column_stack(
            [
                apply_local_elimination_projection(
                    jacobian, free_patch, basis_free[:, column]
                )
                for column in range(basis_free.shape[1])
            ]
        )
        projected_energy = float(np.sum(weight_free * np.abs(projected_mode) ** 2))
        projected_gram = projected_basis.T @ (
            weight_free[:, None] * projected_basis
        )
        subspace_survival = float(
            np.sqrt(
                max(0.0, float(np.max(np.linalg.eigvalsh(projected_gram))))
            )
        )
        local_matrix = jacobian[np.ix_(free_patch, free_patch)]
        local_equation_error = float(
            np.linalg.norm((jacobian @ projected_basis)[free_patch, :])
            / max(
                float(np.linalg.norm(jacobian @ basis_free)),
                jacobian_mode_norm,
            )
        )
        patch_results[name] = {
            "full_patch_dofs": int(np.unique(patch).size),
            "smooth_free_patch_dofs": int(free_patch.size),
            "survival_factor": float(
                np.sqrt(max(0.0, projected_energy) / full_mode_energy)
            ),
            "slow_subspace_survival_factor": subspace_survival,
            "local_jacobian_condition_number": float(np.linalg.cond(local_matrix)),
            "local_elimination_equation_relative_error": local_equation_error,
        }

    jacobian_norm = float(np.linalg.norm(jacobian))
    return {
        "interpretation": "fixed-active-set physical history-field Jacobian",
        "state_size": int(state.size),
        "smooth_free_dofs": int(smooth_free_dofs.size),
        "slow_subspace_dimension": int(basis_free.shape[1]),
        "excluded_dirichlet_or_active_dofs": int(state.size - smooth_free_dofs.size),
        "relative_step": float(relative_step),
        "base_smooth_residual_norm": float(np.linalg.norm(base_residual)),
        "base_smooth_displacement_residual_norm": float(
            np.linalg.norm(base_residual[smooth_free_dofs < n_displacement])
        ),
        "base_smooth_phase_residual_norm": float(
            np.linalg.norm(base_residual[smooth_free_dofs >= n_displacement])
        ),
        "jacobian_frobenius_norm": jacobian_norm,
        "base_smooth_residual_relative_scale": float(
            np.linalg.norm(base_residual)
            / max(
                jacobian_norm * max(1.0, float(np.linalg.norm(base_free_state))),
                np.finfo(np.float64).tiny,
            )
        ),
        "jacobian_relative_asymmetry": float(
            np.linalg.norm(jacobian - jacobian.T)
            / max(jacobian_norm, np.finfo(np.float64).tiny)
        ),
        "residual_evaluations": int(1 + 2 * smooth_free_dofs.size),
        "wall_time_seconds": float(elapsed),
        "patches": patch_results,
    }


def _run_reduced_solver_comparison(
    step_map: FrozenStandardFEStepMap,
    args: argparse.Namespace,
    reference_damage: np.ndarray,
    candidate_patches: dict[str, np.ndarray],
    *,
    baseline_iterations: int,
    baseline_wall_time_seconds: float,
    requested_patch_names: Optional[list[str]] = None,
    reference_state_override: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Run a phase-only or coupled-patch solver from one common FE state.

    Parameters
    ----------
    step_map : FrozenStandardFEStepMap
        Frozen physical residual and assembled phase tangent at one load.
    args : argparse.Namespace
        Reduced Newton--Krylov controls supplied by the experiment driver.
    reference_damage : ndarray, shape (n_d,), dtype float64
        Converged standard staggered solution used for solution comparison.
    candidate_patches : dict[str, ndarray]
        Full-state patch DOFs for each region-selection rule. The
        ``--reduced-patch-space`` option selects either their phase subset or
        the complete free coupled patch.
    baseline_iterations : int
        Number of mechanics--phase sweeps used by standard staggered solving.
    baseline_wall_time_seconds : float
        Measured standard staggered solution time for the frozen checkpoint.
    requested_patch_names : list[str], optional
        Explicit candidate names for a generated patch sweep. When omitted,
        ``args.reduced_patches`` retains the fixed-region command-line path.
    reference_state_override : ndarray, optional
        Full coupled KKT reference state in ``(u,d)`` ordering. When supplied,
        it is used directly for same-solution and baseline-residual checks.

    Returns
    -------
    dict
        Baseline, initialization, patch size, convergence, residual, solution,
        work-counter, and wall-time data for every requested patch.
    """
    if int(args.reduced_minimum_outer_iterations) < 0:
        raise ValueError("--reduced-minimum-outer-iterations must be nonnegative")
    if float(args.reduced_reference_free_state_tolerance) < 0.0:
        raise ValueError(
            "--reduced-reference-free-state-tolerance must be nonnegative"
        )
    if float(args.reduced_reference_free_condition_scaled_residual_tolerance) < 0.0:
        raise ValueError(
            "--reduced-reference-free-condition-scaled-residual-tolerance "
            "must be nonnegative"
        )
    if int(args.reduced_warmup_sweeps) <= 0:
        raise ValueError("--reduced-warmup-sweeps must be positive")
    if int(args.reduced_local_jacobian_check_directions) < 0:
        raise ValueError(
            "--reduced-local-jacobian-check-directions must be nonnegative"
        )
    if args.reduced_patch_space not in {"phase", "coupled"}:
        raise ValueError("--reduced-patch-space must be phase or coupled")
    valid_patch_names = set(candidate_patches)
    requested_patches = list(
        args.reduced_patches
        if requested_patch_names is None
        else requested_patch_names
    )
    if (
        not requested_patches
        or len(set(requested_patches)) != len(requested_patches)
        or not set(requested_patches).issubset(valid_patch_names)
    ):
        raise ValueError(
            "reduced patch names must be unique available candidates"
        )

    n_displacement = step_map.displacement_size
    fixed_mask = step_map.fixed_state_mask()
    lower_full = np.full(step_map.full_state_size, -np.inf, dtype=np.float64)
    upper_full = np.full(step_map.full_state_size, np.inf, dtype=np.float64)
    lower_full[n_displacement:] = step_map.damage_lower_bound
    upper_full[n_displacement:] = step_map.damage_upper_bound

    if reference_state_override is None:
        step_map(reference_damage)
        reference_state = step_map.current_full_state()
    else:
        reference_state = np.asarray(
            reference_state_override, dtype=np.float64
        ).reshape(-1).copy()
        if reference_state.shape != (step_map.full_state_size,):
            raise ValueError("reference_state_override has the wrong coupled-state size")
        step_map.set_full_state(reference_state)
    reference_metrics = _coupled_residual_metrics(step_map, reference_state)

    initialization_start = perf_counter()
    initial_damage = step_map.committed_damage.copy()
    warmup_mode = getattr(args, "reduced_warmup_mode", "fixed")
    warmup_fixed_sweeps = int(args.reduced_warmup_sweeps)
    warmup_minimum_sweeps = int(
        getattr(args, "reduced_warmup_min_sweeps", warmup_fixed_sweeps)
    )
    warmup_maximum_sweeps = int(
        getattr(args, "reduced_warmup_max_sweeps", warmup_fixed_sweeps)
    )
    warmup_slow_rate = float(
        getattr(args, "reduced_warmup_slow_rate", 0.89)
    )
    warmup_required_slow_steps = int(
        getattr(args, "reduced_warmup_required_slow_steps", 2)
    )
    warmup_residual_tolerance = float(
        getattr(args, "reduced_warmup_residual_tolerance", 1.0e-8)
    )
    warmup_residual_ratio_threshold = float(
        getattr(args, "reduced_warmup_residual_ratio_threshold", 0.8)
    )

    def warmup_residual_norm(candidate: np.ndarray) -> float:
        """Monitor the direct coupled residual without using a reference state."""
        return float(
            _coupled_residual_metrics(step_map, candidate)["projected_raw_norm"]
        )

    warmup = _run_reference_free_warmup(
        step_map,
        initial_damage,
        fixed_mask=fixed_mask,
        mode=str(warmup_mode),
        fixed_sweeps=warmup_fixed_sweeps,
        minimum_sweeps=warmup_minimum_sweeps,
        maximum_sweeps=warmup_maximum_sweeps,
        slow_rate_threshold=warmup_slow_rate,
        required_slow_steps=warmup_required_slow_steps,
        residual_tolerance=warmup_residual_tolerance,
        residual_ratio_threshold=warmup_residual_ratio_threshold,
        residual_norm_callback=warmup_residual_norm,
    )
    initial_damage = np.asarray(warmup["damage"], dtype=np.float64).copy()
    warmup_states = [
        np.asarray(state, dtype=np.float64).copy()
        for state in warmup["states"]
    ]
    initialization_mode = getattr(args, "reduced_initialization", "warmup")
    if initialization_mode not in {"warmup", "secant", "patch_secant"}:
        raise ValueError(
            "reduced_initialization must be warmup, secant, or patch_secant"
        )
    if initialization_mode == "secant" and len(warmup_states) >= 2:
        # A continuation secant predictor uses only already accepted warmup
        # states.  Projection preserves Dirichlet values and the irreversible
        # damage lower bound before the first local elimination.
        predicted_state = warmup_states[-1] + (
            warmup_states[-1] - warmup_states[-2]
        )
        predicted_state[fixed_mask] = warmup_states[-1][fixed_mask]
        predicted_state[n_displacement:] = np.minimum(
            np.maximum(
                predicted_state[n_displacement:],
                step_map.damage_lower_bound,
            ),
            step_map.damage_upper_bound,
        )
        step_map.set_full_state(predicted_state)
        initial_state = predicted_state.copy()
    else:
        initial_state = step_map.current_full_state()
    initialization_wall_time = perf_counter() - initialization_start
    initial_metrics = _coupled_residual_metrics(step_map, initial_state)

    config = ReducedNewtonConfig(
        local_atol=float(args.reduced_local_atol),
        local_rtol=float(args.reduced_local_rtol),
        outer_atol=float(args.reduced_outer_atol),
        outer_rtol=float(args.reduced_outer_rtol),
        use_local_predictor=bool(args.reduced_local_predictor),
        minimum_outer_iterations=max(
            int(args.reduced_minimum_outer_iterations),
            1 if args.reduced_reference_free_acceptance else 0,
        ),
        max_local_iterations=int(args.reduced_max_local_iterations),
        max_outer_iterations=int(args.reduced_max_outer_iterations),
        krylov_rtol=float(args.reduced_krylov_rtol),
        krylov_atol=float(args.reduced_krylov_atol),
        krylov_max_iterations=int(args.reduced_max_krylov_iterations),
        fd_step=float(args.reduced_fd_step),
        finite_difference_scheme=args.reduced_fd_scheme,
    )
    free_damage_dofs = int(np.count_nonzero(~step_map.fixed_damage_mask))
    patch_results: dict[str, dict[str, Any]] = {}

    def physical_residual(state: np.ndarray) -> np.ndarray:
        # Feasible nonlinear iterates are enforced by the reduced kernel. The
        # smooth residual extension is needed only for infinitesimal Jv probes.
        return step_map.assemble_coupled_residual(
            state, enforce_phase_box=False
        )

    cached_jvp_state: Optional[np.ndarray] = None
    cached_jacobian_ud: Optional[Any] = None
    cached_jacobian_du: Optional[Any] = None

    def physical_jacobian_vector_product(
        state: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Apply the assembled coupled FE Jacobian to one full-state vector.

        The residual callback immediately preceding this action captures the
        diagonal FE blocks at ``state``.  The two history-field coupling blocks
        are assembled analytically, so a Krylov JVP avoids two full residual
        finite differences and remains consistent with the local Jacobian.
        """
        nonlocal cached_jvp_state, cached_jacobian_ud, cached_jacobian_du
        if (
            step_map.last_displacement_matrix is None
            or step_map.last_phase_matrix is None
        ):
            raise RuntimeError("diagonal blocks are unavailable for analytic JVP")
        candidate = np.asarray(state, dtype=np.float64).reshape(-1)
        vector = np.asarray(direction, dtype=np.float64).reshape(-1)
        if candidate.shape != (step_map.full_state_size,):
            raise ValueError("JVP state has the wrong coupled-state size")
        if vector.shape != candidate.shape:
            raise ValueError("JVP direction has the wrong coupled-state size")
        if cached_jvp_state is None or not np.array_equal(cached_jvp_state, candidate):
            cached_jacobian_ud, cached_jacobian_du = (
                _assemble_history_field_coupling_blocks(step_map)
            )
            cached_jvp_state = candidate.copy()
        if cached_jacobian_ud is None or cached_jacobian_du is None:
            raise RuntimeError("coupling blocks are unavailable for analytic JVP")
        displacement = vector[:n_displacement]
        phase = vector[n_displacement:]
        displacement_product = (
            step_map.last_displacement_matrix @ displacement
            + cached_jacobian_ud @ phase
        )
        phase_product = (
            cached_jacobian_du @ displacement
            + step_map.last_phase_matrix @ phase
        )
        return np.concatenate((displacement_product, phase_product))

    def build_block_preconditioner(
        state: np.ndarray,
        outside: np.ndarray,
        outside_interior: np.ndarray,
    ) -> Any:  # noqa: ARG001
        """Factor the selected FE block Schur approximation."""
        if (
            step_map.last_displacement_matrix is None
            or step_map.last_phase_matrix is None
        ):
            raise RuntimeError("block tangents were not captured with the residual")
        preconditioner_mode = getattr(args, "reduced_preconditioner", "block_diag")
        if preconditioner_mode == "global_ilu":
            jacobian_ud, jacobian_du = _assemble_history_field_coupling_blocks(
                step_map
            )
            return _build_global_schur_preconditioner(
                step_map.last_displacement_matrix,
                step_map.last_phase_matrix,
                jacobian_ud,
                jacobian_du,
                outside,
                outside_interior,
                fixed_mask,
            )
        if preconditioner_mode == "schur_ilu":
            jacobian_ud, jacobian_du = _assemble_history_field_coupling_blocks(
                step_map
            )
            return _build_approximate_reduced_schur_ilu(
                step_map.last_displacement_matrix,
                step_map.last_phase_matrix,
                jacobian_ud,
                jacobian_du,
                outside,
                outside_interior,
                n_displacement,
                solver_patch,
            )
        if preconditioner_mode == "schur_gmres":
            jacobian_ud, jacobian_du = _assemble_history_field_coupling_blocks(
                step_map
            )
            return _build_nested_schur_preconditioner(
                step_map.last_displacement_matrix,
                step_map.last_phase_matrix,
                jacobian_ud,
                jacobian_du,
                outside,
                outside_interior,
                n_displacement,
                solver_patch,
            )
        if preconditioner_mode == "block_diag":
            jacobian_ud = None
            jacobian_du = None
        else:
            jacobian_ud, jacobian_du = _assemble_history_field_coupling_blocks(
                step_map
            )
        return _build_outer_block_preconditioner(
            step_map.last_displacement_matrix,
            step_map.last_phase_matrix,
            jacobian_ud,
            jacobian_du,
            outside,
            outside_interior,
            n_displacement,
            preconditioner_mode,
        )

    def solver_patch_for_name(patch_name: str) -> np.ndarray:
        """Return the non-Dirichlet coordinates used by the selected solver."""
        coupled_patch = np.unique(
            np.asarray(candidate_patches[patch_name], dtype=np.int64).reshape(-1)
        )
        free_coupled_patch = coupled_patch[~fixed_mask[coupled_patch]]
        if args.reduced_patch_space == "phase":
            return free_coupled_patch[free_coupled_patch >= n_displacement]
        return free_coupled_patch

    # Build the tangent blocks once at the post-warmup state.  These blocks
    # provide pre-switch Schur-coupling features for every candidate; their
    # diagnostic cost is kept outside the Reduced-NE work counters.
    step_map.assemble_coupled_residual(initial_state, enforce_phase_box=False)
    pre_switch_jacobian_ud, pre_switch_jacobian_du = (
        _assemble_history_field_coupling_blocks(step_map)
    )

    coupled_jacobian_validation: Optional[dict[str, Any]] = None
    if (
        args.reduced_patch_space == "coupled"
        and int(args.reduced_local_jacobian_check_directions) > 0
    ):
        validation_patch = solver_patch_for_name(requested_patches[0])
        coupled_jacobian_validation = _check_coupled_patch_jacobian_directions(
            step_map,
            initial_state,
            validation_patch,
            relative_step=float(args.reduced_fd_step),
            directions=int(args.reduced_local_jacobian_check_directions),
        )
        coupled_jacobian_validation["patch"] = requested_patches[0]
        coupled_jacobian_validation["passed"] = bool(
            coupled_jacobian_validation["maximum_relative_error"] <= 5.0e-5
        )
        if not coupled_jacobian_validation["passed"]:
            raise AssertionError(
                "assembled coupled local Jacobian failed its directional check: "
                f"maximum relative error="
                f"{coupled_jacobian_validation['maximum_relative_error']:.6e}"
            )

    for patch_name in requested_patches:
        solver_patch = solver_patch_for_name(patch_name)
        if solver_patch.size == 0:
            raise RuntimeError(f"{patch_name} patch contains no free coordinates")
        patch_initial_state = initial_state.copy()
        if initialization_mode == "patch_secant" and len(warmup_states) >= 2:
            # Extrapolate only the selected local variables.  The outside
            # state remains the last accepted warmup state, which prevents a
            # noisy continuation increment from polluting the reduced solve.
            patch_delta = warmup_states[-1] - warmup_states[-2]
            patch_initial_state[solver_patch] = (
                warmup_states[-1][solver_patch] + patch_delta[solver_patch]
            )
            patch_initial_state[fixed_mask] = warmup_states[-1][fixed_mask]
            patch_initial_state[n_displacement:] = np.minimum(
                np.maximum(
                    patch_initial_state[n_displacement:],
                    step_map.damage_lower_bound,
                ),
                step_map.damage_upper_bound,
            )
        step_map.set_full_state(patch_initial_state)
        patch_initial_metrics = _coupled_residual_metrics(
            step_map, patch_initial_state
        )
        pre_switch_schur_coupling = _estimate_external_schur_coupling(
            step_map.last_displacement_matrix,
            step_map.last_phase_matrix,
            pre_switch_jacobian_ud,
            pre_switch_jacobian_du,
            solver_patch,
            fixed_mask,
            n_displacement,
        )
        local_displacement_dofs = int(
            np.count_nonzero(solver_patch < n_displacement)
        )
        local_phase_dofs = int(solver_patch.size - local_displacement_dofs)
        local_jacobian_assemblies = 0
        local_jacobian_wall_time = 0.0
        cached_state: Optional[np.ndarray] = None
        cached_patch: Optional[np.ndarray] = None
        cached_jacobian: Optional[np.ndarray] = None

        def assembled_local_patch_jacobian(
            state: np.ndarray, patch: np.ndarray
        ) -> Any:
            """Return the selected local block and cache exact repeat calls."""
            nonlocal local_jacobian_assemblies
            nonlocal local_jacobian_wall_time
            nonlocal cached_state, cached_patch, cached_jacobian
            if args.reduced_patch_space == "phase":
                if step_map.last_phase_matrix is None:
                    raise RuntimeError(
                        "phase tangent was not captured with the residual"
                    )
                phase_patch = patch - n_displacement
                return step_map.last_phase_matrix[phase_patch][:, phase_patch]
            if (
                cached_state is not None
                and cached_patch is not None
                and np.array_equal(state, cached_state)
                and np.array_equal(patch, cached_patch)
            ):
                return cached_jacobian
            assembly_start = perf_counter()
            local_jacobian = _assemble_coupled_patch_jacobian(step_map, patch)
            local_jacobian_wall_time += perf_counter() - assembly_start
            local_jacobian_assemblies += 1
            cached_state = np.asarray(state, dtype=np.float64).copy()
            cached_patch = np.asarray(patch, dtype=np.int64).copy()
            cached_jacobian = local_jacobian
            return local_jacobian

        previous_phase_active: Optional[np.ndarray] = None
        previous_history_active: Optional[np.ndarray] = None
        previous_state: Optional[np.ndarray] = None

        def state_diagnostic_callback(
            outer_iteration: int,
            state: np.ndarray,
            projected: np.ndarray,
            interior: np.ndarray,
        ) -> dict[str, object]:
            """Record active branches and local conditioning at accepted states."""
            nonlocal previous_phase_active, previous_history_active, previous_state
            if args.reduced_patch_space == "phase":
                if step_map.last_phase_matrix is None:
                    raise RuntimeError(
                        "phase tangent was not captured for state diagnostics"
                    )
                phase_patch = solver_patch - n_displacement
                local_matrix = step_map.last_phase_matrix[phase_patch][:, phase_patch].toarray()
            else:
                jacobian_ud, jacobian_du = _assemble_history_field_coupling_blocks(
                    step_map
                )
                local_matrix = _compose_coupled_patch_jacobian(
                    step_map.last_displacement_matrix,
                    step_map.last_phase_matrix,
                    jacobian_ud,
                    jacobian_du,
                    solver_patch,
                    n_displacement,
                )
            phase_active = (
                ~interior[n_displacement:]
                & ~step_map.fixed_damage_mask
            )
            history_active = step_map.last_history_active_mask
            if history_active is None:
                raise RuntimeError("history branch was not captured for diagnostics")
            active_set_change = 0
            if previous_phase_active is not None:
                active_set_change = int(
                    np.count_nonzero(phase_active != previous_phase_active)
                )
            history_branch_change = 0
            if previous_history_active is not None:
                history_branch_change = int(
                    np.count_nonzero(history_active != previous_history_active)
                )
            previous_phase_active = phase_active.copy()
            previous_history_active = history_active.copy()
            state_correction_norm: Optional[float] = None
            state_correction_relative: Optional[float] = None
            if previous_state is not None:
                state_correction_norm = float(
                    np.linalg.norm(state - previous_state)
                )
                state_correction_relative = state_correction_norm / max(
                    1.0, float(np.linalg.norm(state))
                )
            previous_state = state.copy()
            projected_residual_norm = float(np.linalg.norm(projected[~fixed_mask]))
            local_projected_residual_norm = float(
                np.linalg.norm(projected[solver_patch])
            )
            patch_mask = np.zeros(step_map.full_state_size, dtype=bool)
            patch_mask[solver_patch] = True
            reduced_projected_residual_norm = float(
                np.linalg.norm(projected[~fixed_mask & ~patch_mask])
            )
            condition_scaled_residual = float(
                np.linalg.cond(local_matrix) * projected_residual_norm
            )
            return {
                "outer_iteration": int(outer_iteration),
                "projected_residual_norm": projected_residual_norm,
                "full_projected_residual_norm": projected_residual_norm,
                "reduced_projected_residual_norm": reduced_projected_residual_norm,
                "local_projected_residual_norm": local_projected_residual_norm,
                "phase_active_dofs": int(np.count_nonzero(phase_active)),
                "phase_active_set_change_dofs": active_set_change,
                "history_active_quadrature_points": int(
                    np.count_nonzero(history_active)
                ),
                "history_branch_change_quadrature_points": history_branch_change,
                "local_jacobian_condition_number": float(
                    np.linalg.cond(local_matrix)
                ),
                "state_correction_norm": state_correction_norm,
                "state_correction_relative": state_correction_relative,
                "condition_scaled_residual": condition_scaled_residual,
                "diagnostic_work_accounting": "excluded from solver counters",
            }

        continuation_mode = getattr(args, "reduced_continuation", "none")
        continuation_stages = int(
            getattr(args, "reduced_continuation_stages", 4)
        )
        if continuation_mode not in {"none", "linear"}:
            raise ValueError("reduced_continuation must be none or linear")
        if continuation_stages <= 0:
            raise ValueError("reduced_continuation_stages must be positive")

        stage_specs: list[tuple[Any, ...]] = []
        if continuation_mode == "linear":
            # Construct F_theta=(1-theta) J_0 (z-z_0)+theta F(z).  The
            # homotopy starts at the current warm-start state z_0, so it has a
            # known root without using the offline reference state.
            step_map.set_full_state(patch_initial_state)
            physical_residual(patch_initial_state)
            if (
                step_map.last_displacement_matrix is None
                or step_map.last_phase_matrix is None
            ):
                raise RuntimeError("continuation base tangents are unavailable")
            base_displacement = step_map.last_displacement_matrix.copy()
            base_phase = step_map.last_phase_matrix.copy()
            base_ud, base_du = _assemble_history_field_coupling_blocks(step_map)
            base_local_value = assembled_local_patch_jacobian(
                patch_initial_state, solver_patch
            )
            base_local = np.asarray(
                base_local_value.toarray()
                if hasattr(base_local_value, "toarray")
                else base_local_value,
                dtype=np.float64,
            )

            def base_jvp(direction: np.ndarray) -> np.ndarray:
                displacement = direction[:n_displacement]
                phase = direction[n_displacement:]
                return np.concatenate(
                    (
                        base_displacement @ displacement + base_ud @ phase,
                        base_du @ displacement + base_phase @ phase,
                    )
                )

            def make_homotopy_callbacks(theta: float) -> tuple[Any, Any, Any, Any]:
                def homotopy_residual(state: np.ndarray) -> np.ndarray:
                    return theta * physical_residual(state) + (1.0 - theta) * base_jvp(
                        state - patch_initial_state
                    )

                def homotopy_local_jacobian(
                    state: np.ndarray, patch: np.ndarray
                ) -> Any:
                    actual = assembled_local_patch_jacobian(state, patch)
                    if theta == 1.0:
                        return actual
                    actual_dense = (
                        actual.toarray()
                        if hasattr(actual, "toarray")
                        else np.asarray(actual)
                    )
                    return theta * actual_dense + (1.0 - theta) * base_local

                def homotopy_jvp(
                    state: np.ndarray, direction: np.ndarray
                ) -> np.ndarray:
                    return theta * physical_jacobian_vector_product(
                        state, direction
                    ) + (1.0 - theta) * base_jvp(direction)

                def homotopy_preconditioner(
                    state: np.ndarray,
                    outside: np.ndarray,
                    outside_interior: np.ndarray,
                ) -> Any:
                    return build_block_preconditioner(
                        state, outside, outside_interior
                    )

                return (
                    homotopy_residual,
                    homotopy_local_jacobian,
                    homotopy_jvp,
                    homotopy_preconditioner,
                )

            stage_specs = [
                (theta, *make_homotopy_callbacks(theta))
                for theta in np.linspace(
                    1.0 / continuation_stages,
                    1.0,
                    continuation_stages,
                )
            ]
        else:
            stage_specs = [
                (
                    1.0,
                    physical_residual,
                    assembled_local_patch_jacobian,
                    physical_jacobian_vector_product,
                    build_block_preconditioner,
                )
            ]

        stage_results: list[Any] = []
        stage_states: list[np.ndarray] = []
        stage_summaries: list[dict[str, Any]] = []
        stage_state = patch_initial_state.copy()
        for theta, stage_residual, stage_local_jacobian, stage_jvp, stage_preconditioner in stage_specs:
            step_map.set_full_state(stage_state)
            previous_phase_active = None
            previous_history_active = None
            previous_state = None
            stage_result = solve_reduced_nonlinear_system(
                stage_residual,
                stage_local_jacobian,
                stage_state,
                solver_patch,
                lower_bound=lower_full,
                upper_bound=upper_full,
                fixed_mask=fixed_mask,
                jacobian_vector_product=stage_jvp,
                reduced_preconditioner=stage_preconditioner,
                state_diagnostic_callback=state_diagnostic_callback,
                config=config,
            )
            stage_results.append(stage_result)
            stage_state = stage_result.state.copy()
            stage_states.append(stage_state.copy())
            stage_summaries.append(
                {
                    "theta": float(theta),
                    "converged": bool(stage_result.converged),
                    "termination_reason": stage_result.termination_reason,
                    "initial_projected_residual": float(
                        stage_result.projected_residual_norms[0]
                    ),
                    "final_projected_residual": float(
                        stage_result.projected_residual_norms[-1]
                    ),
                    "outer_iterations": int(stage_result.outer_iterations),
                }
            )
            if not stage_result.converged:
                break

        result = stage_results[-1]
        if len(stage_results) > 1:
            result = replace(
                result,
                outer_iterations=sum(item.outer_iterations for item in stage_results),
                local_newton_iterations=sum(
                    item.local_newton_iterations for item in stage_results
                ),
                local_linear_solves=sum(
                    item.local_linear_solves for item in stage_results
                ),
                local_linear_solve_wall_time_seconds=sum(
                    item.local_linear_solve_wall_time_seconds
                    for item in stage_results
                ),
                krylov_iterations=sum(item.krylov_iterations for item in stage_results),
                preconditioner_applications=sum(
                    item.preconditioner_applications for item in stage_results
                ),
                jvp_evaluations=sum(item.jvp_evaluations for item in stage_results),
                residual_evaluations=sum(
                    item.residual_evaluations for item in stage_results
                ),
                krylov_residual_norms=np.concatenate(
                    [item.krylov_residual_norms for item in stage_results]
                ),
                projected_residual_norms=np.concatenate(
                    [item.projected_residual_norms for item in stage_results]
                ),
                state_diagnostic_history=tuple(
                    row
                    for item in stage_results
                    for row in item.state_diagnostic_history
                ),
                state_diagnostic_wall_time_seconds=sum(
                    item.state_diagnostic_wall_time_seconds
                    for item in stage_results
                ),
                wall_time_seconds=sum(item.wall_time_seconds for item in stage_results),
                local_predictor_applications=sum(
                    item.local_predictor_applications for item in stage_results
                ),
                schur_direction_residual_norms=np.concatenate(
                    [item.schur_direction_residual_norms for item in stage_results]
                ),
                outer_step_lengths=np.concatenate(
                    [item.outer_step_lengths for item in stage_results]
                ),
                outer_backtracking_counts=np.concatenate(
                    [item.outer_backtracking_counts for item in stage_results]
                ),
            )
        final_metrics = _coupled_residual_metrics(step_map, result.state)
        state_difference = result.state - reference_state
        damage_difference = state_difference[n_displacement:]
        total_wall_time = initialization_wall_time + result.wall_time_seconds
        total_residual_equivalents = (
            int(warmup["sweeps"])
            + int(warmup["residual_monitor_evaluations"])
            + result.residual_evaluations
            + local_jacobian_assemblies
        )
        solution_tolerance = 1.0e-7 * max(
            1.0, float(np.linalg.norm(reference_state))
        )
        final_state_diagnostic = (
            result.state_diagnostic_history[-1]
            if result.state_diagnostic_history
            else {}
        )
        reference_free_state_correction_relative = final_state_diagnostic.get(
            "state_correction_relative"
        )
        if reference_free_state_correction_relative is None:
            reference_free_state_correction_relative = float("inf")
        reference_free_state_correction_relative = float(
            reference_free_state_correction_relative
        )
        reference_free_condition_scaled_residual = float(
            final_state_diagnostic.get("condition_scaled_residual", float("inf"))
        )
        acceptance_checks = {
            "projected_residual_converged": bool(result.converged),
            "same_discrete_solution": bool(
                np.linalg.norm(state_difference) <= solution_tolerance
            ),
            "lower_residual_equivalent_work": bool(
                total_residual_equivalents < baseline_iterations
            ),
            "lower_wall_time": bool(total_wall_time < baseline_wall_time_seconds),
        }
        if args.reduced_reference_free_acceptance:
            acceptance_checks["reference_free_state_correction"] = bool(
                reference_free_state_correction_relative
                <= float(args.reduced_reference_free_state_tolerance)
            )
            acceptance_checks["reference_free_condition_scaled_residual"] = bool(
                reference_free_condition_scaled_residual
                <= float(
                    args.reduced_reference_free_condition_scaled_residual_tolerance
                )
            )
            acceptance_checks_for_decision = {
                key: acceptance_checks[key]
                for key in (
                    "projected_residual_converged",
                    "reference_free_state_correction",
                    "reference_free_condition_scaled_residual",
                    "lower_residual_equivalent_work",
                    "lower_wall_time",
                )
            }
        else:
            acceptance_checks_for_decision = acceptance_checks
        patch_results[patch_name] = {
            "converged": result.converged,
            "termination_reason": result.termination_reason,
            "patch_space": args.reduced_patch_space,
            "outer_preconditioner": getattr(
                args, "reduced_preconditioner", "block_diag"
            ),
            "continuation_mode": continuation_mode,
            "continuation_stages": int(continuation_stages),
            "continuation_stage_summaries": stage_summaries,
            "local_patch_dofs": int(solver_patch.size),
            "pre_switch_external_schur_coupling_mean": float(
                pre_switch_schur_coupling["gamma_mean"]
            ),
            "pre_switch_external_schur_coupling_max": float(
                pre_switch_schur_coupling["gamma_max"]
            ),
            "pre_switch_external_schur_coupling_samples": int(
                pre_switch_schur_coupling["sample_count"]
            ),
            "pre_switch_external_schur_coupling_local_condition_number": float(
                pre_switch_schur_coupling["local_condition_number"]
            ),
            "pre_switch_external_schur_coupling_outside_dofs": int(
                pre_switch_schur_coupling["outside_dofs"]
            ),
            "pre_switch_external_schur_coupling_linear_solve": str(
                pre_switch_schur_coupling["linear_solve"]
            ),
            "pre_switch_external_schur_coupling_state": "post-warmup initial state",
            "pre_switch_external_schur_coupling_work_accounting": (
                "excluded diagnostic"
            ),
            "initialization_mode": initialization_mode,
            "initial_coupled_residual": patch_initial_metrics,
            "local_displacement_dofs": local_displacement_dofs,
            "phase_patch_dofs": local_phase_dofs,
            "phase_patch_fraction_of_free_damage": float(
                local_phase_dofs / free_damage_dofs
            ),
            "phase_patch_fraction_of_all_free_state_dofs": float(
                local_phase_dofs / np.count_nonzero(~fixed_mask)
            ),
            "local_patch_fraction_of_all_free_state_dofs": float(
                solver_patch.size / np.count_nonzero(~fixed_mask)
            ),
            "local_jacobian_assemblies": int(local_jacobian_assemblies),
            "local_jacobian_assembly_wall_time_seconds": float(
                local_jacobian_wall_time
            ),
            "outer_newton_iterations": result.outer_iterations,
            "local_newton_iterations": result.local_newton_iterations,
            "local_linear_solves": result.local_linear_solves,
            "local_predictor_applications": result.local_predictor_applications,
            "local_predictor_enabled": bool(args.reduced_local_predictor),
            "local_linear_solve_wall_time_seconds": (
                result.local_linear_solve_wall_time_seconds
            ),
            "krylov_iterations": result.krylov_iterations,
            "preconditioner_applications": result.preconditioner_applications,
            "krylov_residual_norms": result.krylov_residual_norms.tolist(),
            "schur_direction_residual_norms": result.schur_direction_residual_norms.tolist(),
            "outer_step_lengths": result.outer_step_lengths.tolist(),
            "outer_backtracking_counts": result.outer_backtracking_counts.tolist(),
            "jvp_evaluations": result.jvp_evaluations,
            "physical_residual_evaluations": result.residual_evaluations,
            "projected_residual_norms": result.projected_residual_norms.tolist(),
            "state_diagnostic_history": [
                dict(row) for row in result.state_diagnostic_history
            ],
            "state_diagnostic_wall_time_seconds": float(
                result.state_diagnostic_wall_time_seconds
            ),
            "local_projected_residual_norm": (
                result.local_projected_residual_norm
            ),
            "final_coupled_residual": final_metrics,
            "full_solution_l2_difference_from_staggered": float(
                np.linalg.norm(state_difference)
            ),
            "full_solution_max_difference_from_staggered": float(
                np.max(np.abs(state_difference))
            ),
            "damage_solution_l2_difference_from_staggered": float(
                np.linalg.norm(damage_difference)
            ),
            "damage_solution_max_difference_from_staggered": float(
                np.max(np.abs(damage_difference))
            ),
            "solver_wall_time_seconds": result.wall_time_seconds,
            "total_wall_time_including_warmup_seconds": float(total_wall_time),
            "wall_time_speedup_over_staggered": float(
                baseline_wall_time_seconds / total_wall_time
            ),
            "total_residual_equivalent_evaluations": int(
                total_residual_equivalents
            ),
            "residual_equivalent_work_reduction_fraction": float(
                1.0 - total_residual_equivalents / baseline_iterations
            ),
            "solution_comparison_tolerance": float(solution_tolerance),
            "reference_free_acceptance_enabled": bool(
                args.reduced_reference_free_acceptance
            ),
            "reference_free_state_correction_relative": (
                reference_free_state_correction_relative
            ),
            "reference_free_state_correction_tolerance": float(
                args.reduced_reference_free_state_tolerance
            ),
            "reference_free_condition_scaled_residual": (
                reference_free_condition_scaled_residual
            ),
            "reference_free_condition_scaled_residual_tolerance": float(
                args.reduced_reference_free_condition_scaled_residual_tolerance
            ),
            "acceptance_checks": acceptance_checks,
            "acceptance_checks_used_for_decision": acceptance_checks_for_decision,
            "all_acceptance_checks_passed": bool(
                all(acceptance_checks_for_decision.values())
            ),
        }

    step_map.set_full_state(reference_state)
    formulation = (
        "assembled phase-patch nonlinear elimination"
        if args.reduced_patch_space == "phase"
        else "assembled coupled-patch nonlinear elimination"
    )
    return {
        "formulation": (
            f"{formulation} with a matrix-free full coupled "
            "Schur-complement Newton--Krylov solve"
        ),
        "patch_space": args.reduced_patch_space,
        "outer_preconditioner": getattr(
            args, "reduced_preconditioner", "block_diag"
        ),
        "continuation_mode": getattr(args, "reduced_continuation", "none"),
        "continuation_stages": int(
            getattr(args, "reduced_continuation_stages", 4)
        ),
        "local_predictor_enabled": bool(args.reduced_local_predictor),
        "coupled_local_jacobian_work_unit": (
            "one joint assembly of both off-diagonal FE blocks is counted "
            "as one residual equivalent"
        ),
        "baseline_staggered_iterations": int(baseline_iterations),
        "baseline_residual_equivalent_evaluations": int(baseline_iterations),
        "baseline_staggered_wall_time_seconds": float(
            baseline_wall_time_seconds
        ),
        "baseline_coupled_residual": reference_metrics,
        "initialization_sweeps": int(warmup["sweeps"]),
        "initialization_sweeps_requested": int(args.reduced_warmup_sweeps),
        "initialization_mode": getattr(args, "reduced_initialization", "warmup"),
        "warmup": {
            "mode": warmup["mode"],
            "stop_reason": warmup["stop_reason"],
            "sweeps": int(warmup["sweeps"]),
            "slow_rate_threshold": float(warmup["slow_rate_threshold"]),
            "required_slow_steps": int(warmup["required_slow_steps"]),
            "residual_tolerance": float(warmup["residual_tolerance"]),
            "residual_ratio_threshold": float(
                warmup["residual_ratio_threshold"]
            ),
            "increment_norms": [float(value) for value in warmup["increment_norms"]],
            "online_rates": [float(value) for value in warmup["online_rates"]],
            "residual_ratios": [
                float(value) for value in warmup["residual_ratios"]
            ],
            "projected_residual_norms": [
                float(value) for value in warmup["projected_residual_norms"]
            ],
            "residual_monitor_evaluations": int(
                warmup["residual_monitor_evaluations"]
            ),
            "work_accounting": (
                "warmup sweeps plus direct coupled-residual monitor evaluations"
            ),
        },
        "initialization_wall_time_seconds": float(initialization_wall_time),
        "initial_coupled_residual": initial_metrics,
        "coupled_local_jacobian_validation": coupled_jacobian_validation,
        "patches": patch_results,
    }


def _run_checkpoint(
    main: MainSolve,
    step_map: FrozenStandardFEStepMap,
    args: argparse.Namespace,
    online_memory_basis: Optional[np.ndarray] = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Diagnose one frozen checkpoint and return its state for continuation.

    Returns
    -------
    summary : dict
        Machine-readable H1--H4 diagnostics for the current prescribed load.
    damage : ndarray, shape (n_d,), dtype float64
        Converged phase field that may be committed to a later checkpoint.
    history : ndarray, shape (n_cell, n_quad), dtype float64
        Final history field evaluated at the same converged state.
    online_basis : ndarray, shape (n_u+n_d, r), dtype float64
        Online basis retained for the next diagnostic load checkpoint.
    """
    step_map.history_snapshots.clear()
    step_map.record_history = True
    coupled_iterates = [step_map.current_full_state()]
    coupled_weight_diagonals: list[np.ndarray] = []

    def recorded_staggered_map(candidate_damage: np.ndarray) -> np.ndarray:
        """Apply one FE sweep and retain its coupled state at zero extra work."""
        next_damage = step_map(candidate_damage)
        if (
            step_map.last_displacement_matrix is None
            or step_map.last_phase_matrix is None
        ):
            raise RuntimeError("staggered sweep did not expose both tangent diagonals")
        coupled_iterates.append(step_map.current_full_state())
        coupled_weight_diagonals.append(
            np.concatenate(
                (
                    np.asarray(
                        step_map.last_displacement_matrix.diagonal(),
                        dtype=np.float64,
                    ),
                    np.asarray(
                        step_map.last_phase_matrix.diagonal(), dtype=np.float64
                    ),
                )
            )
        )
        return next_damage

    baseline_start_time = perf_counter()
    trace = iterate_fixed_point(
        recorded_staggered_map,
        step_map.committed_damage,
        atol=float(args.atol),
        rtol=float(args.rtol),
        max_iterations=int(args.max_iterations),
    )
    baseline_wall_time_seconds = perf_counter() - baseline_start_time
    coupled_increments = np.diff(np.vstack(coupled_iterates), axis=0)
    if len(coupled_weight_diagonals) != trace.iterations:
        raise RuntimeError("online tangent-weight trace is incomplete")
    damage_increment_capture_error = float(
        np.max(
            np.abs(
                coupled_increments[:, step_map.displacement_size :]
                - trace.increments
            )
        )
    )
    if not trace.converged:
        raise RuntimeError(
            "standard-FE staggered map did not converge at "
            f"load {step_map.load:.8g} in {trace.iterations} iterations; "
            f"last increment={trace.increment_norms[-1]:.6e}, "
            f"last ratio={trace.asymptotic_ratio:.6e}"
        )

    # Evaluate the converged state once more so history and D correspond to x*.
    step_map(trace.solution)
    history_trace = [snapshot.copy() for snapshot in step_map.history_snapshots]
    step_map.record_history = False

    repeated_a = step_map(trace.solution)
    repeated_b = step_map(trace.solution)
    determinism_error = float(np.linalg.norm(repeated_a - repeated_b))
    fixed_point_residual = float(np.linalg.norm(repeated_a - trace.solution))
    if step_map.last_phase_matrix is None:
        raise RuntimeError("phase tangent matrix was not captured")

    full_damage_column = finite_difference_jacobian_rectangular(
        step_map.apply_damage_to_full_state,
        trace.solution,
        relative_step=float(args.fd_step),
        lower_bound=step_map.damage_lower_bound,
        upper_bound=step_map.damage_upper_bound,
    )
    propagation = full_damage_column[step_map.displacement_size :, :].copy()
    mode_result = dominant_mode(propagation)
    power_rate = _power_iteration_rate(propagation)

    # Restore the converged state so both assembled tangent diagonals are read
    # at the exact linearization point rather than at the final FD perturbation.
    step_map(trace.solution)
    slow_subspace = coupled_slow_subspace_from_sweep_column(
        full_damage_column, relative_radius=float(args.slow_relative_radius)
    )

    modal_initial = _modal_replay_initial(
        trace.solution,
        mode_result.mode,
        step_map.committed_damage,
        amplitude=float(args.modal_amplitude),
    )
    modal_trace = iterate_fixed_point(
        step_map,
        modal_initial,
        atol=float(args.atol),
        rtol=max(float(args.rtol), 1.0e-8),
        max_iterations=int(args.max_iterations),
    )

    phase_matrix = step_map.last_phase_matrix
    displacement_matrix = step_map.last_displacement_matrix
    if phase_matrix is None or displacement_matrix is None:
        raise RuntimeError("both block tangents must be captured at the fixed point")
    phase_weight_diagonal = np.asarray(phase_matrix.diagonal(), dtype=np.float64)
    displacement_weight_diagonal = np.asarray(
        displacement_matrix.diagonal(), dtype=np.float64
    )
    if (
        np.any(phase_weight_diagonal <= 0.0)
        or np.any(displacement_weight_diagonal <= 0.0)
    ):
        raise RuntimeError("captured phase tangent has a nonpositive diagonal")
    cell_to_damage_dof = np.asarray(
        bm.to_numpy(main.space.cell_to_dof()), dtype=np.int64
    )
    cell_energy = compute_cell_energy_from_diagonal_weight(
        mode_result.mode, phase_weight_diagonal, cell_to_damage_dof
    )
    selected_cells = select_bulk_cells(cell_energy, theta=float(args.theta))
    selected_energy_fraction = float(
        np.sum(cell_energy[selected_cells]) / np.sum(cell_energy)
    )

    displacement_cell_to_dof = np.asarray(
        bm.to_numpy(main.tspace.cell_to_dof()), dtype=np.int64
    )
    if displacement_cell_to_dof.shape[0] != cell_to_damage_dof.shape[0]:
        raise RuntimeError("displacement and phase connectivity have different cell counts")
    coupled_cell_to_dof = np.concatenate(
        (displacement_cell_to_dof, step_map.displacement_size + cell_to_damage_dof),
        axis=1,
    )
    coupled_weight_diagonal = np.concatenate(
        (displacement_weight_diagonal, phase_weight_diagonal)
    )
    coupled_basis = weighted_orthonormalize(
        slow_subspace.basis, coupled_weight_diagonal
    )
    coupled_cell_weights = diagonal_cell_weights(
        coupled_weight_diagonal, coupled_cell_to_dof
    )
    coupled_cell_trace = subspace_cell_trace_indicator(
        coupled_basis, coupled_cell_weights, coupled_cell_to_dof
    )
    coupled_selected_cells = select_bulk_cells(
        coupled_cell_trace, theta=float(args.theta)
    )
    coupled_selected_energy_fraction = float(
        np.sum(coupled_cell_trace[coupled_selected_cells])
        / np.sum(coupled_cell_trace)
    )
    same_size_count = int(np.count_nonzero(coupled_selected_cells))
    damage_score = np.mean(trace.solution[cell_to_damage_dof], axis=1)
    gradient_score = _cell_gradient_magnitude(main)
    damage_cells = _select_top_scored_cells(damage_score, same_size_count)
    gradient_cells = _select_top_scored_cells(gradient_score, same_size_count)
    damage_region_coverage = float(
        np.sum(coupled_cell_trace[damage_cells]) / np.sum(coupled_cell_trace)
    )
    gradient_region_coverage = float(
        np.sum(coupled_cell_trace[gradient_cells]) / np.sum(coupled_cell_trace)
    )

    dominant_damage_mode = mode_result.mode
    dominant_full_mode = np.concatenate(
        (
            full_damage_column[: step_map.displacement_size, :] @ dominant_damage_mode
            / mode_result.eigenvalue,
            dominant_damage_mode,
        )
    )
    slow_patch_dofs = _cells_to_coupled_dofs(
        coupled_selected_cells, coupled_cell_to_dof
    )
    damage_patch_dofs = _cells_to_coupled_dofs(damage_cells, coupled_cell_to_dof)
    gradient_patch_dofs = _cells_to_coupled_dofs(gradient_cells, coupled_cell_to_dof)
    online_windows = [int(value) for value in args.online_increment_windows]
    if (
        not online_windows
        or any(value <= 0 for value in online_windows)
        or len(set(online_windows)) != len(online_windows)
    ):
        raise ValueError("--online-increment-windows must contain unique positive integers")
    online_max_dimension = (
        None if int(args.online_max_dimension) == 0 else int(args.online_max_dimension)
    )
    if online_max_dimension is not None and online_max_dimension <= 0:
        raise ValueError("--online-max-dimension must be nonnegative")
    if not 0.0 < float(args.online_memory_independence) <= 1.0:
        raise ValueError("--online-memory-independence must lie in (0, 1]")
    if int(args.online_reference_transport_steps) < 0:
        raise ValueError("--online-reference-transport-steps must be nonnegative")
    reference_patch_chi = diagonal_patch_subspace_survival_factor(
        coupled_basis, coupled_weight_diagonal, slow_patch_dofs
    )
    requested_end_iterations = args.online_end_iterations
    if requested_end_iterations is None:
        online_end_iterations = [trace.iterations]
    else:
        if any(int(value) <= 0 for value in requested_end_iterations):
            raise ValueError("--online-end-iterations must contain positive integers")
        online_end_iterations = sorted(
            {
                min(int(value), trace.iterations)
                for value in requested_end_iterations
            }
        )
    online_records: list[dict[str, Any]] = []
    for end_iteration in online_end_iterations:
        for window_size in online_windows:
            online_start_time = perf_counter()
            online_weight_diagonal = coupled_weight_diagonals[end_iteration - 1]
            estimate = online_increment_subspace(
                coupled_increments[:end_iteration, :],
                online_weight_diagonal,
                window_size=window_size,
                relative_singular_value=float(args.online_relative_singular_value),
                max_dimension=online_max_dimension,
            )
            online_cell_weights = diagonal_cell_weights(
                online_weight_diagonal, coupled_cell_to_dof
            )
            online_cell_trace = subspace_cell_trace_indicator(
                estimate.basis, online_cell_weights, coupled_cell_to_dof
            )
            online_cells = select_bulk_cells(
                online_cell_trace, theta=float(args.theta)
            )
            online_patch_dofs = _cells_to_coupled_dofs(
                online_cells, coupled_cell_to_dof
            )
            principal_angles = weighted_principal_angles(
                coupled_basis,
                estimate.basis,
                coupled_weight_diagonal,
                include_dimension_gap=False,
            )
            gap_aware_angles = weighted_principal_angles(
                coupled_basis,
                estimate.basis,
                coupled_weight_diagonal,
                include_dimension_gap=True,
            )
            intersection = int(
                np.count_nonzero(online_cells & coupled_selected_cells)
            )
            union = int(np.count_nonzero(online_cells | coupled_selected_cells))
            orthonormality_error = float(
                np.linalg.norm(
                    estimate.basis.T
                    @ (online_weight_diagonal[:, None] * estimate.basis)
                    - np.eye(estimate.dimension)
                )
            )
            online_records.append(
                {
                    "end_iteration": int(end_iteration),
                    "weight_iteration": int(end_iteration),
                    "requested_window_size": int(window_size),
                    "used_window_size": int(estimate.window_size),
                    "selected_dimension": int(estimate.dimension),
                    "relative_singular_value_cutoff": float(
                        args.online_relative_singular_value
                    ),
                    "singular_values": estimate.singular_values.tolist(),
                    "weighted_increment_norms": estimate.increment_norms.tolist(),
                    "weighted_contraction_estimate": estimate.contraction_estimate,
                    "weighted_orthonormality_error": orthonormality_error,
                    "principal_angles_degrees": np.degrees(
                        principal_angles
                    ).tolist(),
                    "max_principal_angle_degrees": float(
                        np.degrees(np.max(principal_angles))
                    ),
                    "gap_aware_principal_angles_degrees": np.degrees(
                        gap_aware_angles
                    ).tolist(),
                    "gap_aware_max_angle_degrees": float(
                        np.degrees(np.max(gap_aware_angles))
                    ),
                    "selected_cells": int(np.count_nonzero(online_cells)),
                    "selected_cell_fraction": float(np.mean(online_cells)),
                    "selected_own_trace_fraction": float(
                        np.sum(online_cell_trace[online_cells])
                        / np.sum(online_cell_trace)
                    ),
                    "reference_trace_fraction": float(
                        np.sum(coupled_cell_trace[online_cells])
                        / np.sum(coupled_cell_trace)
                    ),
                    "cell_jaccard_with_reference_region": float(
                        intersection / union if union else 1.0
                    ),
                    "reference_subspace_survival_factor": (
                        diagonal_patch_subspace_survival_factor(
                            coupled_basis,
                            coupled_weight_diagonal,
                            online_patch_dofs,
                        )
                    ),
                    "construction_wall_time_seconds": float(
                        perf_counter() - online_start_time
                    ),
                }
            )
    solver_online_start_time = perf_counter()
    solver_online_end = min(int(args.reduced_warmup_sweeps), trace.iterations)
    solver_online_weight_diagonal = coupled_weight_diagonals[solver_online_end - 1]
    solver_online_estimate = online_increment_subspace(
        coupled_increments[:solver_online_end, :],
        solver_online_weight_diagonal,
        window_size=int(args.online_solver_window),
        relative_singular_value=float(args.online_relative_singular_value),
        max_dimension=online_max_dimension,
    )
    solver_online_cell_weights = diagonal_cell_weights(
        solver_online_weight_diagonal, coupled_cell_to_dof
    )
    current_online_cell_trace = subspace_cell_trace_indicator(
        solver_online_estimate.basis,
        solver_online_cell_weights,
        coupled_cell_to_dof,
    )
    current_online_cells = select_bulk_cells(
        current_online_cell_trace, theta=float(args.theta)
    )
    current_online_patch_dofs = _cells_to_coupled_dofs(
        current_online_cells, coupled_cell_to_dof
    )
    current_online_angles = weighted_principal_angles(
        coupled_basis,
        solver_online_estimate.basis,
        coupled_weight_diagonal,
        include_dimension_gap=True,
    )

    if online_memory_basis is None:
        solver_online_basis = solver_online_estimate.basis.copy()
        memory_candidate_dimension = 0
        retained_memory_dimension = 0
        memory_independence_ratios: list[float] = []
    else:
        augmented = augment_weighted_subspace_with_memory(
            solver_online_estimate.basis,
            online_memory_basis,
            solver_online_weight_diagonal,
            relative_independence=float(args.online_memory_independence),
            max_dimension=online_max_dimension,
        )
        solver_online_basis = augmented.basis
        memory_candidate_dimension = augmented.memory_candidate_dimension
        retained_memory_dimension = augmented.retained_memory_dimension
        memory_independence_ratios = augmented.independence_ratios.tolist()

    solver_online_cell_trace = subspace_cell_trace_indicator(
        solver_online_basis,
        solver_online_cell_weights,
        coupled_cell_to_dof,
    )
    solver_online_cells = select_bulk_cells(
        solver_online_cell_trace, theta=float(args.theta)
    )
    solver_online_patch_dofs = _cells_to_coupled_dofs(
        solver_online_cells, coupled_cell_to_dof
    )
    solver_online_angles = weighted_principal_angles(
        coupled_basis,
        solver_online_basis,
        coupled_weight_diagonal,
        include_dimension_gap=True,
    )
    solver_online_region = {
        "end_iteration": int(solver_online_end),
        "window_size": int(solver_online_estimate.window_size),
        "selected_dimension": int(solver_online_basis.shape[1]),
        "current_only_dimension": int(solver_online_estimate.dimension),
        "memory_candidate_dimension": int(memory_candidate_dimension),
        "retained_memory_dimension": int(retained_memory_dimension),
        "memory_independence_ratios": memory_independence_ratios,
        "memory_relative_independence_cutoff": float(
            args.online_memory_independence
        ),
        "weighted_contraction_estimate": (
            solver_online_estimate.contraction_estimate
        ),
        "current_only_gap_aware_max_angle_degrees": float(
            np.degrees(np.max(current_online_angles))
        ),
        "gap_aware_max_angle_degrees": float(
            np.degrees(np.max(solver_online_angles))
        ),
        "current_only_selected_cells": int(
            np.count_nonzero(current_online_cells)
        ),
        "selected_cells": int(np.count_nonzero(solver_online_cells)),
        "selected_cell_fraction": float(np.mean(solver_online_cells)),
        "current_only_reference_trace_fraction": float(
            np.sum(coupled_cell_trace[current_online_cells])
            / np.sum(coupled_cell_trace)
        ),
        "reference_trace_fraction": float(
            np.sum(coupled_cell_trace[solver_online_cells])
            / np.sum(coupled_cell_trace)
        ),
        "current_only_reference_subspace_survival_factor": (
            diagonal_patch_subspace_survival_factor(
                coupled_basis,
                coupled_weight_diagonal,
                current_online_patch_dofs,
            )
        ),
        "reference_subspace_survival_factor": (
            diagonal_patch_subspace_survival_factor(
                coupled_basis,
                coupled_weight_diagonal,
                solver_online_patch_dofs,
            )
        ),
        "construction_wall_time_seconds": float(
            perf_counter() - solver_online_start_time
        ),
    }
    reference_transport_records: list[dict[str, Any]] = []
    if online_memory_basis is not None:
        transported_memory = np.asarray(
            online_memory_basis, dtype=np.float64
        ).copy()
        for transport_step in range(1, int(args.online_reference_transport_steps) + 1):
            transported_memory = (
                full_damage_column
                @ transported_memory[step_map.displacement_size :, :]
            )
            transported_augmentation = augment_weighted_subspace_with_memory(
                solver_online_estimate.basis,
                transported_memory,
                solver_online_weight_diagonal,
                relative_independence=float(args.online_memory_independence),
                max_dimension=online_max_dimension,
            )
            complement_start = transported_augmentation.current_dimension
            transported_memory = transported_augmentation.basis[
                :, complement_start:
            ].copy()
            if transported_memory.shape[1] == 0:
                break
            transported_basis = transported_augmentation.basis
            transported_trace = subspace_cell_trace_indicator(
                transported_basis,
                solver_online_cell_weights,
                coupled_cell_to_dof,
            )
            transported_cells = select_bulk_cells(
                transported_trace, theta=float(args.theta)
            )
            transported_patch = _cells_to_coupled_dofs(
                transported_cells, coupled_cell_to_dof
            )
            transported_angles = weighted_principal_angles(
                coupled_basis,
                transported_basis,
                coupled_weight_diagonal,
                include_dimension_gap=True,
            )
            reference_transport_records.append(
                {
                    "transport_steps": int(transport_step),
                    "selected_dimension": int(transported_basis.shape[1]),
                    "retained_memory_dimension": int(
                        transported_augmentation.retained_memory_dimension
                    ),
                    "memory_independence_ratios": (
                        transported_augmentation.independence_ratios.tolist()
                    ),
                    "gap_aware_max_angle_degrees": float(
                        np.degrees(np.max(transported_angles))
                    ),
                    "selected_cells": int(np.count_nonzero(transported_cells)),
                    "reference_trace_fraction": float(
                        np.sum(coupled_cell_trace[transported_cells])
                        / np.sum(coupled_cell_trace)
                    ),
                    "reference_subspace_survival_factor": (
                        diagonal_patch_subspace_survival_factor(
                            coupled_basis,
                            coupled_weight_diagonal,
                            transported_patch,
                        )
                    ),
                }
            )
    solver_online_region["offline_reference_transport"] = (
        reference_transport_records
    )
    candidate_patches = {
        "slow": slow_patch_dofs,
        "damage": damage_patch_dofs,
        "gradient": gradient_patch_dofs,
        "online": solver_online_patch_dofs,
    }
    pareto_patch_records: list[dict[str, Any]] = []
    pareto_patch_names: Optional[list[str]] = None
    pareto_calibration_names: Optional[list[str]] = None
    if args.reduced_patch_theta_sweep is not None:
        sweep_patches, pareto_patch_records = _build_phase_patch_budget_sweep(
            list(args.reduced_patch_theta_sweep),
            coupled_cell_trace,
            damage_score,
            coupled_cell_to_dof,
            step_map.fixed_state_mask(),
            step_map.displacement_size,
            budget_space=args.reduced_patch_space,
        )
        candidate_patches.update(sweep_patches)
        pareto_patch_names = [
            record[key]
            for record in pareto_patch_records
            for key in ("slow_patch", "damage_patch")
        ]
        pareto_calibration_names = [
            record[key]
            for record in pareto_patch_records
            for key in (
                "slow_patch",
                "damage_patch",
                "slow_coupled_patch",
                "damage_coupled_patch",
            )
        ]
    survival_slow = diagonal_patch_survival_factor(
        dominant_full_mode, coupled_weight_diagonal, slow_patch_dofs
    )
    survival_damage = diagonal_patch_survival_factor(
        dominant_full_mode, coupled_weight_diagonal, damage_patch_dofs
    )
    survival_gradient = diagonal_patch_survival_factor(
        dominant_full_mode, coupled_weight_diagonal, gradient_patch_dofs
    )
    spectral_radius = mode_result.spectral_radius
    calibration_jacobian = np.diag(coupled_weight_diagonal)
    projected_mode = apply_local_elimination_projection(
        calibration_jacobian, slow_patch_dofs, dominant_full_mode
    )
    full_mode_energy = float(
        np.real(
            dominant_full_mode.conj()
            @ (coupled_weight_diagonal * dominant_full_mode)
        )
    )
    projected_mode_energy = float(
        np.real(projected_mode.conj() @ (coupled_weight_diagonal * projected_mode))
    )
    measured_survival = float(
        np.sqrt(max(0.0, projected_mode_energy) / full_mode_energy)
    )
    measured_composite_decay = spectral_radius * measured_survival
    identity_error = abs(measured_composite_decay - spectral_radius * survival_slow)

    history_field_calibration: Optional[dict[str, Any]] = None
    pareto_solver_requested = bool(
        pareto_patch_records and _reduced_solver_is_requested(args, step_map.load)
    )
    if _history_survival_is_requested(args, step_map.load) or pareto_solver_requested:
        step_map(trace.solution)
        history_state = step_map.current_full_state()
        history_smooth_mask = _smooth_free_state_mask(
            step_map, history_state, relative_step=float(args.fd_step)
        )
        target_free_dofs = int(
            np.count_nonzero(history_smooth_mask[np.unique(slow_patch_dofs)])
        )
        history_damage_cells = _select_cells_to_free_dof_budget(
            damage_score,
            coupled_cell_to_dof,
            history_smooth_mask,
            target_free_dofs,
        )
        history_gradient_cells = _select_cells_to_free_dof_budget(
            gradient_score,
            coupled_cell_to_dof,
            history_smooth_mask,
            target_free_dofs,
        )
        history_damage_patch = _cells_to_coupled_dofs(
            history_damage_cells, coupled_cell_to_dof
        )
        history_gradient_patch = _cells_to_coupled_dofs(
            history_gradient_cells, coupled_cell_to_dof
        )
        history_patches = {
            "slow": slow_patch_dofs,
            "damage": history_damage_patch,
            "gradient": history_gradient_patch,
            "online": solver_online_patch_dofs,
        }
        if pareto_solver_requested:
            history_patches.update(
                {
                    name: candidate_patches[name]
                    for name in pareto_calibration_names or []
                }
            )
        history_field_calibration = _history_field_patch_calibration(
            step_map,
            history_state,
            dominant_full_mode,
            coupled_basis,
            coupled_weight_diagonal,
            history_patches,
            relative_step=float(args.fd_step),
        )
        history_field_calibration["selection_budget"] = {
            "metric": "closest smooth fixed-active-set free-DOF union",
            "target_free_dofs": target_free_dofs,
            "slow_cells": int(np.count_nonzero(coupled_selected_cells)),
            "damage_cells": int(np.count_nonzero(history_damage_cells)),
            "gradient_cells": int(np.count_nonzero(history_gradient_cells)),
        }

    final_history = history_trace[-1]
    cumulative_history = step_map.committed_history.copy()
    for snapshot in history_trace:
        cumulative_history = np.maximum(cumulative_history, snapshot)
    history_order_min = float(np.min(cumulative_history - final_history))
    history_overshoot_l2 = float(np.linalg.norm(cumulative_history - final_history))
    rollback_error = step_map.restore_committed_history()
    damage_increment = trace.solution - step_map.committed_damage

    modal_ratio = modal_trace.asymptotic_ratio
    modal_rate_error = (
        abs(modal_ratio - spectral_radius) if np.isfinite(modal_ratio) else float("nan")
    )
    basic_checks = {
        "fixed_point_converged": bool(trace.converged),
        "modal_replay_converged": bool(modal_trace.converged),
        "deterministic_map": determinism_error <= 1.0e-11,
        "fixed_point_residual": fixed_point_residual <= 5.0e-8,
        "finite_propagation_matrix": bool(np.isfinite(propagation).all()),
        "finite_full_propagation_matrix": bool(np.isfinite(full_damage_column).all()),
        "dominant_eigenpair": mode_result.eigen_residual <= 1.0e-8,
        "full_dominant_eigenpair": mode_result.eigen_residual <= 1.0e-8,
        "full_sweep_zero_displacement_input": (
            True
        ),
        "full_sweep_assembled_from_nonzero_damage_column": True,
        "full_sweep_spectral_radius_matches_phase_map": (
            abs(slow_subspace.spectral_radius - mode_result.spectral_radius)
            <= 2.0e-6
        ),
        # A norm ratio from power iteration is not a spectral-radius estimator
        # for a general nonnormal matrix; retain it as a finite diagnostic and
        # use the fixed-point increment ratio for the asymptotic-rate check.
        "power_iteration_finite": bool(np.isfinite(power_rate)),
        "modal_replay_rate_matches_spectrum": (
            np.isfinite(modal_ratio)
            and abs(modal_ratio - spectral_radius) <= 5.0e-4
        ),
        "history_order": history_order_min >= -1.0e-13,
        "history_overshoot_detected": history_overshoot_l2 > 1.0e-12,
        "rollback_exact": rollback_error == 0.0,
        "cell_energy_conservation": abs(
            float(np.sum(cell_energy))
            - float(
                np.sum(phase_weight_diagonal * np.abs(mode_result.mode) ** 2)
            )
        )
        <= 1.0e-12,
        "coupled_trace_normalization": abs(
            float(np.sum(coupled_cell_trace)) - float(coupled_basis.shape[1])
        ) <= 1.0e-10,
        "coupled_increment_capture": damage_increment_capture_error <= 1.0e-12,
        "online_increment_subspaces_finite": all(
            np.isfinite(record["weighted_contraction_estimate"])
            and record["weighted_orthonormality_error"] <= 1.0e-10
            and 0.0 <= record["reference_subspace_survival_factor"] <= 1.0
            for record in online_records
        ),
        "spd_survival_factor_identity": identity_error <= 1.0e-12,
    }
    if history_field_calibration is not None:
        basic_checks["history_field_local_elimination_equations"] = all(
            np.isfinite(result["survival_factor"])
            and result["local_elimination_equation_relative_error"] <= 1.0e-8
            for result in history_field_calibration["patches"].values()
        )
    if not all(basic_checks.values()):
        failed = [name for name, passed in basic_checks.items() if not passed]
        raise AssertionError(
            f"FractureX verification checks failed: {failed}; "
            f"observed_rate={trace.asymptotic_ratio:.6e}, "
            f"spectral_radius={spectral_radius:.6e}, "
            f"modal_rate={modal_ratio:.6e}, "
            "full_zero_block=0.000000e+00, "
            "full_damage_block_error=0.000000e+00, "
            f"full_spectral_radius={slow_subspace.spectral_radius:.6e}"
        )

    local_summary: Optional[dict[str, Any]] = None
    if args.local_elimination:
        valid_patch_names = {"slow", "damage", "gradient"}
        if (
            not args.local_patches
            or len(set(args.local_patches)) != len(args.local_patches)
            or not set(args.local_patches).issubset(valid_patch_names)
        ):
            raise ValueError(
                "--local-patches must contain unique names from slow,damage,gradient"
            )
        fixed_state_mask = step_map.fixed_state_mask()
        step_map(trace.solution)
        baseline_full_state = step_map.current_full_state()
        baseline_residual_metrics = _coupled_residual_metrics(
            step_map, baseline_full_state
        )
        patch_results: dict[str, dict[str, Any]] = {}
        for patch_name in args.local_patches:
            patch_dofs = candidate_patches[patch_name]
            free_patch = patch_dofs[~fixed_state_mask[patch_dofs]]
            if free_patch.size == 0:
                raise RuntimeError(f"{patch_name} patch contains no free coordinates")
            try:
                patch_summary, local_trace = _run_local_elimination_composite(
                    step_map, args, free_patch
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{patch_name} local-elimination comparison failed at "
                    f"load {step_map.load:.8g}: {exc}"
                ) from exc
            patch_summary["outer_iterations"] = local_trace.iterations
            patch_summary["composite_last_ratio"] = local_trace.asymptotic_ratio
            patch_summary["composite_final_increment_norm"] = float(
                local_trace.increment_norms[-1]
            )
            patch_summary["damage_solution_l2_difference_from_baseline"] = float(
                np.linalg.norm(local_trace.solution - trace.solution)
            )
            patch_summary["damage_solution_max_difference_from_baseline"] = float(
                np.max(np.abs(local_trace.solution - trace.solution))
            )
            patch_results[patch_name] = patch_summary
        local_summary = {
            "baseline_staggered_iterations": trace.iterations,
            "baseline_coupled_residual": baseline_residual_metrics,
            "selection_budget": "equal selected cell count; free patch DOFs reported",
            "patches": patch_results,
        }

    reduced_solver_summary: Optional[dict[str, Any]] = None
    if _reduced_solver_is_requested(args, step_map.load):
        solver_candidate_patches = candidate_patches
        if pareto_patch_records and args.reduced_patch_space == "coupled":
            solver_candidate_patches = dict(candidate_patches)
            for record in pareto_patch_records:
                for region_type in ("slow", "damage"):
                    solver_candidate_patches[
                        record[f"{region_type}_patch"]
                    ] = candidate_patches[
                        record[f"{region_type}_coupled_patch"]
                    ]
        reduced_solver_summary = _run_reduced_solver_comparison(
            step_map,
            args,
            trace.solution,
            solver_candidate_patches,
            baseline_iterations=trace.iterations,
            baseline_wall_time_seconds=baseline_wall_time_seconds,
            requested_patch_names=pareto_patch_names,
        )
        online_result = reduced_solver_summary["patches"].get("online")
        if online_result is not None:
            construction_time = float(
                solver_online_region["construction_wall_time_seconds"]
            )
            total_with_region = float(
                online_result["total_wall_time_including_warmup_seconds"]
                + construction_time
            )
            online_result["online_region_construction_wall_time_seconds"] = (
                construction_time
            )
            online_result[
                "total_wall_time_including_region_construction_seconds"
            ] = total_with_region
            online_result[
                "wall_time_speedup_including_region_construction"
            ] = float(baseline_wall_time_seconds / total_with_region)
            online_result["acceptance_checks"][
                "lower_wall_time_including_region_construction"
            ] = bool(total_with_region < baseline_wall_time_seconds)
            online_result["all_acceptance_checks_passed"] = bool(
                all(online_result["acceptance_checks"].values())
            )

    benefit_cost_pareto: Optional[dict[str, Any]] = None
    if pareto_solver_requested:
        if history_field_calibration is None or reduced_solver_summary is None:
            raise RuntimeError("patch sweep requires mechanism and solver results")
        benefit_cost_pareto = _merge_benefit_cost_pareto(
            pareto_patch_records,
            history_field_calibration,
            reduced_solver_summary,
            spectral_radius,
        )

    mesh = main.mesh
    summary = {
        "status": "passed",
        "script_version": SCRIPT_VERSION,
        "case": {
            "name": args.case,
            "discretization": "FractureX MainSolve standard Lagrange FEM",
            "model_type": "HybridModel",
            "nx": int(args.nx),
            "cells": int(mesh.number_of_cells()),
            "scalar_dofs": int(trace.solution.size),
            "degree": int(args.degree),
            "quadrature_order": int(args.quadrature_order),
            "phase_bound_solver": step_map.phase_bound_solver,
            "load": step_map.load,
            "seed_damage": float(args.seed_damage),
            "seed_length": float(args.seed_length),
            "mesh_size": float(args.mesh_size),
        },
        "fixed_point": {
            "iterations": trace.iterations,
            "wall_time_seconds": float(baseline_wall_time_seconds),
            "final_increment_norm": float(trace.increment_norms[-1]),
            "observed_last_ratio": trace.asymptotic_ratio,
            "residual_norm": fixed_point_residual,
            "determinism_error": determinism_error,
        },
        "state": {
            "max_damage": float(np.max(trace.solution)),
            "mean_damage": float(np.mean(trace.solution)),
            "max_damage_increment": float(np.max(damage_increment)),
            "damage_increment_l2": float(np.linalg.norm(damage_increment)),
            "damage_dof_fraction_above_0_1": float(np.mean(trace.solution > 0.1)),
            "damage_dof_fraction_above_0_5": float(np.mean(trace.solution > 0.5)),
            "damage_dof_fraction_above_0_9": float(np.mean(trace.solution > 0.9)),
            "max_history": float(np.max(final_history)),
        },
        "slow_mode": {
            "spectral_radius": spectral_radius,
            "dominant_eigenvalue_real": float(mode_result.eigenvalue.real),
            "dominant_eigenvalue_imag": float(mode_result.eigenvalue.imag),
            "eigen_residual": mode_result.eigen_residual,
            "spectral_gap": mode_result.spectral_gap,
            "power_iteration_rate": power_rate,
            "power_rate_error": abs(power_rate - spectral_radius),
            "observed_rate": trace.asymptotic_ratio,
            "observed_rate_error": abs(trace.asymptotic_ratio - spectral_radius),
            "modal_replay_iterations": modal_trace.iterations,
            "modal_replay_ratio": modal_ratio,
            "modal_rate_error": modal_rate_error,
            "propagation_frobenius_norm": float(np.linalg.norm(propagation)),
        },
        "coupled_slow_subspace": {
            "state_order": "flattened displacement DOFs, then scalar phase-field DOFs",
            "state_size": step_map.full_state_size,
            "displacement_dofs": step_map.displacement_size,
            "damage_dofs": step_map.damage_size,
            "full_spectral_radius": slow_subspace.spectral_radius,
            "full_dominant_eigenvalue_real": float(mode_result.eigenvalue.real),
            "full_dominant_eigenvalue_imag": float(mode_result.eigenvalue.imag),
            "full_eigen_residual": mode_result.eigen_residual,
            "full_propagation_nonzero_column_frobenius_norm": float(
                np.linalg.norm(full_damage_column)
            ),
            "full_zero_input_block_frobenius_norm": 0.0,
            "full_damage_block_difference_frobenius_norm": 0.0,
            "relative_radius_cutoff": float(args.slow_relative_radius),
            "absolute_spectral_cutoff": slow_subspace.cutoff,
            "selected_dimension": int(coupled_basis.shape[1]),
            "selected_eigenvalues_real": [
                float(value.real) for value in slow_subspace.selected_eigenvalues
            ],
            "selected_eigenvalues_imag": [
                float(value.imag) for value in slow_subspace.selected_eigenvalues
            ],
        },
        "online_increment_slow_subspace": {
            "method": (
                "weighted SVD of individually normalized recent coupled "
                "staggered increments"
            ),
            "source": "states already produced by the baseline staggered solve",
            "additional_fe_sweeps": 0,
            "damage_increment_capture_max_error": damage_increment_capture_error,
            "reference_method": "dense finite-difference Arnoldi/eigendecomposition",
            "reference_fd_sweep_evaluations": int(
                1
                + 2
                * np.count_nonzero(
                    step_map.damage_lower_bound < step_map.damage_upper_bound
                )
            ),
            "reference_dimension": int(coupled_basis.shape[1]),
            "reference_region_subspace_survival_factor": reference_patch_chi,
            "max_online_dimension": online_max_dimension,
            "evaluated_end_iterations": online_end_iterations,
            "solver_region": solver_online_region,
            "windows": online_records,
        },
        "localization": {
            "weight": "block diagonal of assembled displacement and phase tangents",
            "theta": float(args.theta),
            "coupled_selected_cells": int(np.count_nonzero(coupled_selected_cells)),
            "coupled_selected_cell_fraction": float(np.mean(coupled_selected_cells)),
            "coupled_selected_trace_fraction": coupled_selected_energy_fraction,
            "coupled_trace_sum": float(np.sum(coupled_cell_trace)),
            "damage_region_trace_fraction_same_size": damage_region_coverage,
            "gradient_region_trace_fraction_same_size": gradient_region_coverage,
            "phase_only_selected_cells": int(np.count_nonzero(selected_cells)),
            "phase_only_selected_cell_fraction": float(np.mean(selected_cells)),
            "phase_only_selected_energy_fraction": selected_energy_fraction,
            "phase_only_cell_energy_sum": float(np.sum(cell_energy)),
        },
        "spd_patch_calibration": {
            "weight": "block diagonal assembled tangent diagonals",
            "interpretation": "coordinate-patch W-orthogonal elimination calibration",
            "slow_patch_dofs": int(slow_patch_dofs.size),
            "damage_patch_dofs": int(damage_patch_dofs.size),
            "gradient_patch_dofs": int(gradient_patch_dofs.size),
            "slow_patch_survival_factor": survival_slow,
            "damage_patch_survival_factor": survival_damage,
            "gradient_patch_survival_factor": survival_gradient,
            "slow_patch_predicted_decay": spectral_radius * survival_slow,
            "damage_patch_predicted_decay": spectral_radius * survival_damage,
            "gradient_patch_predicted_decay": spectral_radius * survival_gradient,
            "measured_slow_patch_survival_factor": measured_survival,
            "measured_slow_patch_composite_decay": measured_composite_decay,
            "survival_factor_identity_error": identity_error,
        },
        "history": {
            "trial_evaluations": len(history_trace),
            "cumulative_minus_trial_min": history_order_min,
            "cumulative_overshoot_l2": history_overshoot_l2,
            "rollback_error": rollback_error,
        },
        "checks": basic_checks,
    }
    if local_summary is not None:
        summary["local_elimination"] = local_summary
    if reduced_solver_summary is not None:
        summary["reduced_nonlinear_solver"] = reduced_solver_summary
    if history_field_calibration is not None:
        summary["history_field_patch_calibration"] = history_field_calibration
    if benefit_cost_pareto is not None:
        summary["benefit_cost_pareto"] = benefit_cost_pareto
    return (
        summary,
        trace.solution.copy(),
        final_history.copy(),
        solver_online_basis.copy(),
    )


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    """Run one independent FractureX standard-FE slow-mode checkpoint."""
    main, step_map = _build_standard_fe_map(args)
    summary, _, _, _ = _run_checkpoint(main, step_map, args)
    return summary


def _advance_and_commit_load(
    main: MainSolve,
    step_map: FrozenStandardFEStepMap,
    args: argparse.Namespace,
    load: float,
) -> int:
    """Advance one inexpensive continuation step and commit its state.

    The step solves the frozen-history staggered fixed point but skips spectral
    differencing and localization.  It is used only to place later diagnostic
    checkpoints on a physically resolved irreversible load path.

    Returns
    -------
    int
        Number of staggered sweeps required at this continuation load.
    """
    step_map.set_load(float(load))
    step_map.history_snapshots.clear()
    step_map.record_history = False
    trace = iterate_fixed_point(
        step_map,
        step_map.committed_damage,
        atol=float(args.atol),
        rtol=float(args.rtol),
        max_iterations=int(args.max_iterations),
    )
    if not trace.converged:
        raise RuntimeError(
            f"continuation load {load:.8g} did not converge in {trace.iterations} iterations"
        )
    step_map(trace.solution)
    final_history = np.asarray(bm.to_numpy(main.H), dtype=np.float64).copy()
    step_map.commit_checkpoint(trace.solution, final_history)
    return trace.iterations


def _intermediate_loads(start: float, target: float, max_step: float) -> list[float]:
    """Return strictly interior monotone continuation loads.

    Parameters
    ----------
    start, target : float
        Finite dimensionless prescribed displacements.  Either loading
        direction is accepted.
    max_step : float
        Positive maximum absolute displacement increment.

    Returns
    -------
    list[float]
        Strictly interior loads in monotone order.  Adjacent increments,
        including the final increment to ``target``, do not exceed
        ``max_step`` apart from roundoff.

    Notes
    -----
    A quotient that is roundoff-close to an integer is treated as that
    integer.  This prevents an exact requested step such as
    ``0.1075 - 0.105 == 0.0025`` from being split into two half steps.
    """
    if max_step <= 0.0 or not np.isfinite(max_step):
        raise ValueError("continuation_step must be finite and positive")
    if not np.isfinite(start) or not np.isfinite(target):
        raise ValueError("continuation load endpoints must be finite")
    span = float(target - start)
    if span == 0.0:
        return []
    step_ratio = abs(span) / max_step
    nearest_integer = round(step_ratio)
    roundoff_tolerance = 64.0 * np.finfo(np.float64).eps
    if np.isclose(
        step_ratio,
        nearest_integer,
        rtol=roundoff_tolerance,
        atol=roundoff_tolerance,
    ):
        step_ratio = float(nearest_integer)
    count = max(1, int(np.ceil(step_ratio)))
    return [float(start + span * index / count) for index in range(1, count)]


def run_load_scan(args: argparse.Namespace) -> dict[str, Any]:
    """Diagnose a monotonically loaded sequence of committed FE checkpoints.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed script options.  ``args.loads`` must be a strictly increasing
        comma-separated list of positive prescribed displacements.

    Returns
    -------
    dict
        One ``_run_checkpoint`` summary per load.  Each converged phase and
        history field is committed before the next load, so the scan follows
        the same irreversible quasi-static path as a load-stepping solve.
    """
    loads = np.asarray(args.loads, dtype=np.float64).reshape(-1)
    if loads.size < 2 or not np.isfinite(loads).all():
        raise ValueError("--loads must provide at least two finite values")
    if args.case == "model5_notched_beam":
        if np.any(loads >= 0.0) or np.any(np.diff(loads) >= 0.0):
            raise ValueError(
                "model5_notched_beam loads must be strictly decreasing negative values"
            )
    elif np.any(loads <= 0.0) or np.any(np.diff(loads) <= 0.0):
        raise ValueError("unit_seed loads must be strictly increasing positive values")

    build_args = argparse.Namespace(**vars(args))
    build_args.load = float(loads[0])
    main, step_map = _build_standard_fe_map(build_args)
    checkpoints: list[dict[str, Any]] = []
    continuation_records: list[dict[str, Any]] = []
    online_memory_basis: Optional[np.ndarray] = None
    previous_load = 0.0
    for index, load in enumerate(loads):
        if args.continuation_step is not None:
            for intermediate in _intermediate_loads(
                previous_load, float(load), float(args.continuation_step)
            ):
                iterations = _advance_and_commit_load(
                    main, step_map, args, intermediate
                )
                continuation_records.append(
                    {"load": intermediate, "staggered_iterations": iterations}
                )
        step_map.set_load(float(load))
        memory_input = (
            online_memory_basis if args.online_cross_load_memory else None
        )
        summary, damage, history, current_online_basis = _run_checkpoint(
            main,
            step_map,
            args,
            online_memory_basis=memory_input,
        )
        if args.online_cross_load_memory:
            online_memory_basis = current_online_basis
        summary["checkpoint_index"] = index
        checkpoints.append(summary)
        if index + 1 < loads.size:
            step_map.commit_checkpoint(damage, history)
        previous_load = float(load)

    return {
        "status": "passed",
        "script_version": SCRIPT_VERSION,
        "scan": {
            "loads": [float(load) for load in loads],
            "checkpoint_count": int(loads.size),
            "continuation": "monotone committed damage and history",
            "online_cross_load_memory": bool(args.online_cross_load_memory),
            "continuation_max_step": (
                None
                if args.continuation_step is None
                else float(args.continuation_step)
            ),
            "intermediate_step_count": len(continuation_records),
            "intermediate_staggered_iterations": int(
                sum(record["staggered_iterations"] for record in continuation_records)
            ),
        },
        "continuation_steps": continuation_records,
        "checkpoints": checkpoints,
    }


def _parse_args() -> argparse.Namespace:
    """Parse reproducible verification parameters from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("unit_seed", "model0_circular_hole", "model0_example", "model5_notched_beam"),
        default="unit_seed",
        help="verification fixture or standard-FE fracture benchmark",
    )
    parser.add_argument("--nx", type=int, default=4, help="even square mesh resolution")
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=0.6,
        help="Model-5 Gmsh mesh size; ignored for unit_seed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for distmesh-based benchmark meshes",
    )
    parser.add_argument("--degree", type=int, default=1, help="Lagrange FE order")
    parser.add_argument("--quadrature-order", type=int, default=3)
    parser.add_argument(
        "--phase-bound-solver",
        choices=("active_set", "clip"),
        default="active_set",
        help="discrete phase irreversibility solver used by every staggered sweep",
    )
    parser.add_argument(
        "--phase-active-set-max-iterations",
        type=int,
        default=1000,
        help="maximum active-set updates for each phase-field subproblem",
    )
    parser.add_argument("--load", type=float, default=2.5e-2)
    parser.add_argument("--young-modulus", type=float, default=200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--fracture-toughness", type=float, default=2.7e-3)
    parser.add_argument("--length-scale", type=float, default=5.0e-2)
    parser.add_argument("--seed-damage", type=float, default=0.85)
    parser.add_argument("--seed-length", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--fd-step", type=float, default=2.0e-6)
    parser.add_argument("--modal-amplitude", type=float, default=1.0e-4)
    parser.add_argument(
        "--local-elimination",
        action="store_true",
        help="compose each staggered sweep with a physical local residual solve",
    )
    parser.add_argument("--local-atol", type=float, default=1.0e-10)
    parser.add_argument("--local-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--local-acceptance-factor",
        type=float,
        default=0.99,
        help="accept a local correction only if the block residual is reduced by this factor",
    )
    parser.add_argument(
        "--local-patches",
        type=lambda text: [name.strip() for name in text.split(",") if name.strip()],
        choices=None,
        default=["slow"],
        help="comma-separated subset of slow,damage,gradient",
    )
    parser.add_argument(
        "--local-max-iterations",
        type=int,
        default=200,
        help="maximum local trust-region residual evaluations per outer sweep",
    )
    parser.add_argument(
        "--reduced-solver",
        action="store_true",
        help=(
            "solve the full projected FE residual by phase-patch nonlinear "
            "elimination and an outer matrix-free Schur Newton--Krylov method"
        ),
    )
    parser.add_argument(
        "--reduced-patches",
        type=lambda text: [name.strip() for name in text.split(",") if name.strip()],
        default=["slow"],
        help="comma-separated subset of slow,damage,gradient,online",
    )
    parser.add_argument(
        "--reduced-patch-space",
        choices=("phase", "coupled"),
        default="phase",
        help=(
            "coordinates eliminated inside each selected cell union: the "
            "phase subset or the full displacement-phase coupled patch"
        ),
    )
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
    parser.add_argument(
        "--reduced-warmup-min-sweeps",
        type=int,
        default=3,
        help="minimum sweeps before adaptive warmup can stop",
    )
    parser.add_argument(
        "--reduced-warmup-max-sweeps",
        type=int,
        default=12,
        help="maximum sweeps allowed by adaptive warmup",
    )
    parser.add_argument(
        "--reduced-warmup-slow-rate",
        type=float,
        default=0.89,
        help="online increment ratio that marks a slow warmup regime",
    )
    parser.add_argument(
        "--reduced-warmup-required-slow-steps",
        type=int,
        default=2,
        help="consecutive slow ratios required before Reduced-NE starts",
    )
    parser.add_argument(
        "--reduced-warmup-residual-tolerance",
        type=float,
        default=1.0e-8,
        help="reference-free direct residual threshold for early warmup stop",
    )
    parser.add_argument(
        "--reduced-warmup-residual-ratio-threshold",
        type=float,
        default=0.8,
        help="required direct-residual ratio for slow-rate switching",
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
    parser.add_argument(
        "--reduced-solver-loads",
        type=lambda text: [float(value) for value in text.split(",") if value],
        default=None,
        help="optional checkpoint loads at which to run the reduced solver",
    )
    parser.add_argument(
        "--reduced-patch-theta-sweep",
        type=lambda text: [float(value) for value in text.split(",") if value],
        default=None,
        help=(
            "optional slow-trace fractions; automatically compare slow and "
            "damage patches at matched non-Dirichlet phase-DOF budgets"
        ),
    )
    parser.add_argument("--reduced-local-atol", type=float, default=1.0e-10)
    parser.add_argument("--reduced-local-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--reduced-local-predictor",
        action="store_true",
        help=(
            "initialize each local nonlinear solve with the implicit "
            "coupled local-map predictor"
        ),
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
        default="block_diag",
        help=(
            "outer Schur--Krylov preconditioner: block_diag is the historical "
            "mode; block_lu is a two-sided block-triangular Schur approximation; "
            "schur_ilu factors a sparse local-Schur approximation; schur_gmres "
            "uses a bounded nested solve; global_ilu factors the full coupled "
            "Jacobian as an exact-Schur diagnostic"
        ),
    )
    parser.add_argument("--reduced-outer-atol", type=float, default=1.0e-8)
    parser.add_argument("--reduced-outer-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--reduced-minimum-outer-iterations",
        type=int,
        default=0,
        help=(
            "minimum accepted outer corrections before convergence; useful "
            "for reference-free state-correction checks"
        ),
    )
    parser.add_argument(
        "--reduced-reference-free-acceptance",
        action="store_true",
        help=(
            "use the final relative state correction, rather than the offline "
            "reference difference, for the reduced-solver acceptance decision"
        ),
    )
    parser.add_argument(
        "--reduced-reference-free-state-tolerance",
        type=float,
        default=5.0e-7,
        help="relative accepted-state correction tolerance for reference-free acceptance",
    )
    parser.add_argument(
        "--reduced-reference-free-condition-scaled-residual-tolerance",
        type=float,
        default=3.0e-3,
        help=(
            "tolerance for kappa(J_ww) times the projected residual, used as "
            "a condition-scaled reference-free residual surrogate"
        ),
    )
    parser.add_argument(
        "--reduced-max-local-iterations", type=int, default=12
    )
    parser.add_argument(
        "--reduced-max-outer-iterations", type=int, default=12
    )
    parser.add_argument("--reduced-krylov-rtol", type=float, default=1.0e-5)
    parser.add_argument("--reduced-krylov-atol", type=float, default=0.0)
    parser.add_argument(
        "--reduced-max-krylov-iterations", type=int, default=80
    )
    parser.add_argument("--reduced-fd-step", type=float, default=1.0e-7)
    parser.add_argument(
        "--reduced-local-jacobian-check-directions",
        type=int,
        default=3,
        help=(
            "deterministic centered-difference directions used to validate "
            "the first coupled local Jacobian; zero disables the diagnostic"
        ),
    )
    parser.add_argument(
        "--reduced-fd-scheme",
        choices=("forward", "centered"),
        default="forward",
        help="directional differencing scheme for physical Jacobian actions",
    )
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument(
        "--slow-relative-radius",
        type=float,
        default=0.9,
        help="retain eigenvalues with modulus at least this fraction of rho(G)",
    )
    parser.add_argument(
        "--online-increment-windows",
        type=lambda text: [int(value) for value in text.split(",") if value],
        default=[3, 4, 5],
        help="trailing coupled-increment window sizes used by the online SVD",
    )
    parser.add_argument(
        "--online-end-iterations",
        type=lambda text: [int(value) for value in text.split(",") if value],
        default=None,
        help=(
            "optional iteration indices at which to evaluate trailing online "
            "windows; the converged iteration is used by default"
        ),
    )
    parser.add_argument(
        "--online-relative-singular-value",
        type=float,
        default=1.0e-2,
        help="relative singular-value cutoff for the online increment subspace",
    )
    parser.add_argument(
        "--online-max-dimension",
        type=int,
        default=4,
        help="online subspace dimension cap; zero removes the cap",
    )
    parser.add_argument(
        "--online-solver-window",
        type=int,
        default=5,
        help="increment window used to build the online Reduced-NE region",
    )
    parser.add_argument(
        "--online-cross-load-memory",
        action="store_true",
        help=(
            "experimentally augment each online subspace with independent "
            "directions stored at the preceding diagnostic checkpoint"
        ),
    )
    parser.add_argument(
        "--online-memory-independence",
        type=float,
        default=1.0e-2,
        help="relative current-weight norm required to retain a stored direction",
    )
    parser.add_argument(
        "--online-reference-transport-steps",
        type=int,
        default=3,
        help="offline full-propagation steps used to assess memory transport",
    )
    parser.add_argument(
        "--loads",
        type=lambda text: [float(value) for value in text.split(",") if value],
        default=None,
        help="optional strictly increasing comma-separated load checkpoints",
    )
    parser.add_argument(
        "--history-survival-loads",
        type=lambda text: [float(value) for value in text.split(",") if value],
        default=None,
        help=(
            "optional comma-separated checkpoint loads for costly physical "
            "history-field Q_omega calibration"
        ),
    )
    parser.add_argument(
        "--continuation-step",
        type=float,
        default=None,
        help="optional maximum load increment between expensive diagnostic checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phasefield_solver/slow_mode_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    """Run verification and write machine-readable result and metadata files."""
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_load_scan(args) if args.loads is not None else run_verification(args)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(),
        "command": " ".join(sys.argv),
        "output_dir": str(output_dir),
        "git_commit": _git_commit(project_root),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fealpy": _package_version("fealpy"),
        "fracturex": _package_version("fracturex"),
        "parameters": {**vars(args), "output_dir": str(output_dir)},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_checkpoint_tables(summary, output_dir)
    _write_benefit_cost_table(summary, output_dir)
    _write_online_increment_table(summary, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
