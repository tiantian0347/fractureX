"""Matrix-free semismooth Newton for coupled block residuals.

Purpose
-------
Solve a coupled nonlinear system with a block Jacobian
``[[J_uu, J_ud], [J_du, J_dd]]`` while enforcing box constraints through the
natural projected residual.  The full Jacobian is never assembled as a dense
matrix; GMRES applies block products and uses the diagonal blocks as a sparse
preconditioner.

Boundary
--------
Finite-element residual and Jacobian assembly remains in the caller.  This
module only owns the Newton globalization, projected active-set rows, and
iteration accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, factorized, gmres


ResidualJacobianCallback = Callable[
    [np.ndarray], tuple[np.ndarray, Any, Any, Any, Any]
]


@dataclass(frozen=True)
class CoupledNewtonConfig:
    """Controls for projected coupled Newton--Krylov iterations.

    All tolerances use the residual norm on non-Dirichlet coordinates.  The
    callback is evaluated at every accepted or line-search trial state.
    """

    residual_atol: float = 1.0e-10
    residual_rtol: float = 1.0e-8
    gmres_rtol: float = 1.0e-7
    gmres_atol: float = 0.0
    gmres_max_iterations: int = 80
    max_newton_iterations: int = 12
    minimum_step_length: float = 1.0e-6
    armijo_slope: float = 1.0e-4


@dataclass(frozen=True)
class CoupledNewtonResult:
    """Stable result record for one coupled projected Newton solve."""

    state: np.ndarray
    converged: bool
    termination_reason: str
    newton_iterations: int
    gmres_iterations: int
    residual_jacobian_evaluations: int
    preconditioner_factorizations: int
    krylov_residual_norms: np.ndarray
    projected_residual_norms: np.ndarray
    wall_time_seconds: float


def _projected_residual(
    state: np.ndarray,
    residual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fixed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the natural box residual and rows treated as smooth interior."""
    argument = state - residual
    projected = state - np.minimum(np.maximum(argument, lower), upper)
    projected[fixed_mask] = 0.0
    interior = (argument > lower) & (argument < upper) & ~fixed_mask
    interior[np.isneginf(lower) & np.isposinf(upper) & ~fixed_mask] = True
    return projected, interior


def _as_sparse_block(block: Any, shape: tuple[int, int], name: str) -> csr_matrix:
    """Validate and copy one sparse Jacobian block."""
    matrix = block if issparse(block) else csr_matrix(np.asarray(block))
    matrix = csr_matrix(matrix, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix.data).all():
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return matrix


def solve_coupled_newton(
    residual_and_jacobian: ResidualJacobianCallback,
    initial_state: object,
    *,
    displacement_size: int,
    lower_bound: Optional[object] = None,
    upper_bound: Optional[object] = None,
    fixed_mask: Optional[object] = None,
    config: Optional[CoupledNewtonConfig] = None,
) -> CoupledNewtonResult:
    """Solve a box-constrained coupled residual with block Newton--Krylov.

    Parameters
    ----------
    residual_and_jacobian : callable
        ``callback(state)`` returns ``(F, J_uu, J_ud, J_du, J_dd)`` in the
        full ordering ``(u, d)``.  The returned blocks are sparse or dense and
        have dimensions ``(n_u,n_u)``, ``(n_u,n_d)``, ``(n_d,n_u)``, and
        ``(n_d,n_d)``.
    initial_state : array-like, shape ``(n_u+n_d,)``
        Feasible initial coupled state.  The input is copied.
    displacement_size : int
        Number of leading displacement coordinates in the coupled ordering.
    lower_bound, upper_bound, fixed_mask : array-like, optional
        Natural box and fixed-coordinate constraints. Infinite bounds are
        accepted; omitted values mean unbounded or no fixed coordinates.
    config : CoupledNewtonConfig, optional
        Newton, GMRES, and line-search controls.

    Returns
    -------
    CoupledNewtonResult
        Final state, convergence status, residual histories, work counters,
        and wall time. Arrays in the result are newly allocated float64 data.
    """
    cfg = config or CoupledNewtonConfig()
    if displacement_size <= 0:
        raise ValueError("displacement_size must be positive")
    if min(cfg.gmres_max_iterations, cfg.max_newton_iterations) <= 0:
        raise ValueError("iteration limits must be positive")
    if not (
        cfg.residual_atol >= 0.0
        and cfg.residual_rtol >= 0.0
        and cfg.gmres_rtol >= 0.0
        and cfg.gmres_atol >= 0.0
        and 0.0 < cfg.minimum_step_length <= 1.0
        and 0.0 < cfg.armijo_slope < 1.0
    ):
        raise ValueError("invalid Newton tolerance or line-search control")

    state = np.asarray(initial_state, dtype=np.float64).reshape(-1).copy()
    n_state = state.size
    n_damage = n_state - int(displacement_size)
    if n_damage <= 0 or not np.isfinite(state).all():
        raise ValueError("initial_state must contain finite coupled coordinates")
    lower = (
        np.full(n_state, -np.inf, dtype=np.float64)
        if lower_bound is None
        else np.asarray(lower_bound, dtype=np.float64).reshape(-1).copy()
    )
    upper = (
        np.full(n_state, np.inf, dtype=np.float64)
        if upper_bound is None
        else np.asarray(upper_bound, dtype=np.float64).reshape(-1).copy()
    )
    fixed = (
        np.zeros(n_state, dtype=bool)
        if fixed_mask is None
        else np.asarray(fixed_mask, dtype=bool).reshape(-1).copy()
    )
    if lower.shape != state.shape or upper.shape != state.shape or fixed.shape != state.shape:
        raise ValueError("bounds and fixed_mask must match initial_state")
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("bounds must be ordered and must not contain NaN")
    if np.any(state < lower) or np.any(state > upper):
        raise ValueError("initial_state violates the supplied box")

    free = np.flatnonzero(~fixed)
    start = perf_counter()
    callback_evaluations = 0
    preconditioner_factorizations = 0
    gmres_iterations = 0
    krylov_residual_norms: list[float] = []
    projected_norms: list[float] = []

    def evaluate(candidate: np.ndarray):
        nonlocal callback_evaluations
        callback_evaluations += 1
        values = residual_and_jacobian(candidate.copy())
        if len(values) != 5:
            raise ValueError("residual_and_jacobian must return five values")
        residual = np.asarray(values[0], dtype=np.float64).reshape(-1)
        if residual.shape != candidate.shape or not np.isfinite(residual).all():
            raise ValueError("coupled residual has an invalid shape or value")
        nu = int(displacement_size)
        blocks = (
            _as_sparse_block(values[1], (nu, nu), "J_uu"),
            _as_sparse_block(values[2], (nu, n_damage), "J_ud"),
            _as_sparse_block(values[3], (n_damage, nu), "J_du"),
            _as_sparse_block(values[4], (n_damage, n_damage), "J_dd"),
        )
        projected, interior = _projected_residual(
            candidate, residual, lower, upper, fixed
        )
        return residual, blocks, projected, interior

    residual, blocks, projected, interior = evaluate(state)
    initial_norm = float(np.linalg.norm(projected[free]))
    tolerance = cfg.residual_atol + cfg.residual_rtol * max(1.0, initial_norm)
    projected_norms.append(initial_norm)
    if initial_norm <= tolerance:
        return CoupledNewtonResult(
            state=state,
            converged=True,
            termination_reason="projected residual converged",
            newton_iterations=0,
            gmres_iterations=0,
            residual_jacobian_evaluations=callback_evaluations,
            preconditioner_factorizations=0,
            krylov_residual_norms=np.asarray(krylov_residual_norms),
            projected_residual_norms=np.asarray(projected_norms),
            wall_time_seconds=perf_counter() - start,
        )

    termination_reason = "maximum Newton iterations reached"
    completed_newton = 0
    for newton_iteration in range(1, cfg.max_newton_iterations + 1):
        completed_newton = newton_iteration
        nu, nd = int(displacement_size), n_damage
        juu, jud, jdu, jdd = blocks
        active_rows = ~interior

        def apply_jacobian(direction: np.ndarray) -> np.ndarray:
            vector = np.asarray(direction, dtype=np.float64).reshape(-1)
            product = np.concatenate(
                (
                    juu @ vector[:nu] + jud @ vector[nu:],
                    jdu @ vector[:nu] + jdd @ vector[nu:],
                )
            )
            product[active_rows] = vector[active_rows]
            return product

        operator = LinearOperator(
            (free.size, free.size),
            matvec=lambda direction: apply_jacobian(
                np.bincount(
                    free,
                    weights=np.asarray(direction, dtype=np.float64),
                    minlength=n_state,
                )
            )[free],
            dtype=np.float64,
        )

        interior_displacement = free[(free < nu) & interior[free]]
        interior_damage = free[(free >= nu) & interior[free]] - nu
        displacement_factor = (
            factorized(juu[interior_displacement][:, interior_displacement].tocsc())
            if interior_displacement.size
            else None
        )
        damage_factor = (
            factorized(jdd[interior_damage][:, interior_damage].tocsc())
            if interior_damage.size
            else None
        )
        preconditioner_factorizations += int(
            bool(interior_displacement.size) + bool(interior_damage.size)
        )

        interior_positions = np.flatnonzero(interior[free])
        active_positions = np.flatnonzero(~interior[free])

        def apply_preconditioner(direction: np.ndarray) -> np.ndarray:
            output = np.zeros_like(direction, dtype=np.float64)
            if displacement_factor is not None:
                positions = np.flatnonzero((free < nu) & interior[free])
                output[positions] = displacement_factor(direction[positions])
            if damage_factor is not None:
                positions = np.flatnonzero((free >= nu) & interior[free])
                output[positions] = damage_factor(direction[positions])
            output[active_positions] = direction[active_positions]
            return output

        preconditioner = LinearOperator(
            (free.size, free.size), matvec=apply_preconditioner, dtype=np.float64
        )

        def count_iteration(value: object) -> None:
            nonlocal gmres_iterations
            gmres_iterations += 1
            krylov_residual_norms.append(float(np.asarray(value)))

        correction, info = gmres(
            operator,
            -projected[free],
            M=preconditioner,
            rtol=cfg.gmres_rtol,
            atol=cfg.gmres_atol,
            restart=min(free.size, cfg.gmres_max_iterations),
            maxiter=cfg.gmres_max_iterations,
            callback=count_iteration,
            callback_type="legacy",
        )
        if info != 0 or not np.isfinite(correction).all():
            termination_reason = f"coupled GMRES did not converge (info={info})"
            break

        base_norm = projected_norms[-1]
        accepted = False
        step_length = 1.0
        while step_length >= cfg.minimum_step_length:
            trial = state.copy()
            trial[free] = np.minimum(
                np.maximum(state[free] + step_length * correction, lower[free]),
                upper[free],
            )
            trial_residual, trial_blocks, trial_projected, trial_interior = evaluate(trial)
            trial_norm = float(np.linalg.norm(trial_projected[free]))
            if trial_norm <= (1.0 - cfg.armijo_slope * step_length) * base_norm:
                state = trial
                residual, blocks = trial_residual, trial_blocks
                projected, interior = trial_projected, trial_interior
                projected_norms.append(trial_norm)
                accepted = True
                break
            step_length *= 0.5
        if not accepted:
            termination_reason = "coupled Newton line search failed"
            break
        if projected_norms[-1] <= tolerance:
            termination_reason = "projected residual converged"
            return CoupledNewtonResult(
                state=state.copy(),
                converged=True,
                termination_reason=termination_reason,
                newton_iterations=completed_newton,
                gmres_iterations=gmres_iterations,
                residual_jacobian_evaluations=callback_evaluations,
                preconditioner_factorizations=preconditioner_factorizations,
                krylov_residual_norms=np.asarray(krylov_residual_norms),
                projected_residual_norms=np.asarray(projected_norms),
                wall_time_seconds=perf_counter() - start,
            )

    return CoupledNewtonResult(
        state=state.copy(),
        converged=False,
        termination_reason=termination_reason,
        newton_iterations=completed_newton,
        gmres_iterations=gmres_iterations,
        residual_jacobian_evaluations=callback_evaluations,
        preconditioner_factorizations=preconditioner_factorizations,
        krylov_residual_norms=np.asarray(krylov_residual_norms),
        projected_residual_norms=np.asarray(projected_norms),
        wall_time_seconds=perf_counter() - start,
    )


__all__ = [
    "CoupledNewtonConfig",
    "CoupledNewtonResult",
    "solve_coupled_newton",
]
