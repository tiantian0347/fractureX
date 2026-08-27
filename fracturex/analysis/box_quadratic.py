"""Active-set solver for sparse box-constrained quadratic subproblems.

The phase-field verification uses this kernel to solve the assembled convex
quadratic subproblem with nodal irreversibility bounds.  It enforces the KKT
conditions of the discrete problem, unlike clipping an unconstrained solution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class BoxQuadraticResult:
    """State and KKT diagnostics for a box-constrained quadratic solve."""

    state: np.ndarray
    converged: bool
    iterations: int
    projected_residual_norm: float
    active_lower_dofs: int
    active_upper_dofs: int


def _natural_residual(
    state: np.ndarray,
    gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return ``x - projection(x-gradient)`` for a box constraint."""
    return state - np.minimum(np.maximum(state - gradient, lower), upper)


def solve_box_quadratic_active_set(
    matrix: object,
    right_hand_side: object,
    initial_state: object,
    lower_bound: object,
    upper_bound: object,
    *,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-10,
    max_iterations: int = 200,
) -> BoxQuadraticResult:
    """Minimize ``0.5*x.T*A*x-b.T*x`` over a finite or infinite box.

    Parameters
    ----------
    matrix : sparse or dense array, shape (n, n)
        Finite symmetric positive-definite Hessian.
    right_hand_side : array-like, shape (n,)
        Linear load vector ``b``.
    initial_state : array-like, shape (n,)
        Feasible starting point; it is copied.
    lower_bound, upper_bound : array-like, shape (n,)
        Ordered component bounds. Infinite endpoints are accepted.
    atol, rtol : float
        Natural-residual stopping controls. The tolerance is
        ``atol + rtol*max(1, ||b||_2)``.
    max_iterations : int
        Maximum working-set updates.

    Returns
    -------
    BoxQuadraticResult
        Feasible state and KKT convergence data.

    Notes
    -----
    Each working-set iteration solves the exact principal free system. Bound
    violations are activated; multipliers with the wrong sign are released.
    For an SPD Hessian the accepted state is the unique constrained minimizer.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if (
        not np.isfinite(atol)
        or not np.isfinite(rtol)
        or atol < 0.0
        or rtol < 0.0
    ):
        raise ValueError("atol and rtol must be finite and nonnegative")
    hessian = matrix.tocsr() if issparse(matrix) else np.asarray(matrix, dtype=float)
    state = np.asarray(initial_state, dtype=np.float64).reshape(-1).copy()
    load = np.asarray(right_hand_side, dtype=np.float64).reshape(-1).copy()
    lower = np.asarray(lower_bound, dtype=np.float64).reshape(-1).copy()
    upper = np.asarray(upper_bound, dtype=np.float64).reshape(-1).copy()
    size = state.size
    if size == 0 or hessian.shape != (size, size):
        raise ValueError("matrix and state dimensions are incompatible")
    matrix_values = hessian.data if issparse(hessian) else hessian
    if not np.isfinite(matrix_values).all():
        raise ValueError("matrix must contain only finite values")
    if load.shape != state.shape or lower.shape != state.shape or upper.shape != state.shape:
        raise ValueError("vectors and bounds must match initial_state")
    if (
        not np.isfinite(state).all()
        or not np.isfinite(load).all()
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.any(lower > upper)
    ):
        raise ValueError("quadratic data must be finite with ordered bounds")
    if np.any(state < lower) or np.any(state > upper):
        raise ValueError("initial_state lies outside the feasible box")

    state = np.minimum(np.maximum(state, lower), upper)
    tolerance = atol + rtol * max(1.0, float(np.linalg.norm(load)))
    equality = np.isfinite(lower) & np.isfinite(upper) & (
        np.abs(upper - lower) <= tolerance
    )
    status = np.zeros(size, dtype=np.int8)
    status[equality] = 2
    state[equality] = lower[equality]

    gradient = np.asarray(hessian @ state - load, dtype=np.float64).reshape(-1)
    at_lower = np.isfinite(lower) & (state <= lower + tolerance)
    at_upper = np.isfinite(upper) & (state >= upper - tolerance)
    status[at_lower & (gradient >= 0.0) & ~equality] = -1
    status[at_upper & (gradient <= 0.0) & ~equality] = 1

    for iteration in range(max_iterations + 1):
        active_lower = (status == -1) | equality
        active_upper = status == 1
        state[active_lower] = lower[active_lower]
        state[active_upper] = upper[active_upper]
        free = status == 0
        if np.any(free):
            active = ~free
            free_rhs = load[free] - np.asarray(
                hessian[free][:, active] @ state[active], dtype=np.float64
            ).reshape(-1)
            free_matrix = hessian[free][:, free]
            free_solution = (
                np.asarray(spsolve(free_matrix.tocsc(), free_rhs), dtype=np.float64)
                if issparse(free_matrix)
                else np.linalg.solve(free_matrix, free_rhs)
            )
            if not np.isfinite(free_solution).all():
                raise RuntimeError("free active-set solve returned non-finite values")
            free_dofs = np.flatnonzero(free)
            below = free_solution < lower[free] - tolerance
            above = free_solution > upper[free] + tolerance
            state[free] = np.minimum(
                np.maximum(free_solution, lower[free]), upper[free]
            )
            if np.any(below) or np.any(above):
                status[free_dofs[below]] = -1
                status[free_dofs[above]] = 1
                continue

        gradient = np.asarray(hessian @ state - load, dtype=np.float64).reshape(-1)
        projected = _natural_residual(state, gradient, lower, upper)
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm <= tolerance:
            return BoxQuadraticResult(
                state=state.copy(),
                converged=True,
                iterations=iteration,
                projected_residual_norm=projected_norm,
                active_lower_dofs=int(np.count_nonzero(status == -1)),
                active_upper_dofs=int(np.count_nonzero(status == 1)),
            )

        wrong_lower = np.flatnonzero((status == -1) & (gradient < -tolerance))
        wrong_upper = np.flatnonzero((status == 1) & (gradient > tolerance))
        if wrong_lower.size == 0 and wrong_upper.size == 0:
            break
        candidates = np.concatenate((wrong_lower, wrong_upper))
        release = candidates[np.argmax(np.abs(gradient[candidates]))]
        status[release] = 0

    gradient = np.asarray(hessian @ state - load, dtype=np.float64).reshape(-1)
    projected_norm = float(
        np.linalg.norm(_natural_residual(state, gradient, lower, upper))
    )
    return BoxQuadraticResult(
        state=state.copy(),
        converged=False,
        iterations=max_iterations,
        projected_residual_norm=projected_norm,
        active_lower_dofs=int(np.count_nonzero(status == -1)),
        active_upper_dofs=int(np.count_nonzero(status == 1)),
    )


__all__ = ["BoxQuadraticResult", "solve_box_quadratic_active_set"]
