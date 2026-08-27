"""Tests for the box-constrained quadratic active-set solver."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from fracturex.analysis import solve_box_quadratic_active_set


def test_unconstrained_solution_is_recovered() -> None:
    """Inactive bounds reproduce the exact linear-system solution."""
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    exact = np.array([0.25, 0.75])
    result = solve_box_quadratic_active_set(
        matrix,
        matrix @ exact,
        np.zeros(2),
        np.zeros(2),
        np.ones(2),
    )

    assert result.converged
    np.testing.assert_allclose(result.state, exact, atol=1.0e-12)
    assert result.projected_residual_norm < 1.0e-12


def test_coupled_bound_solution_satisfies_kkt_not_clipped_linear_solution() -> None:
    """The active-set solution accounts for off-diagonal Hessian coupling."""
    matrix = csr_matrix([[2.0, 1.0], [1.0, 2.0]])
    load = np.array([-1.0, 1.0])
    unconstrained = np.linalg.solve(matrix.toarray(), load)
    clipped = np.clip(unconstrained, 0.0, 1.0)
    result = solve_box_quadratic_active_set(
        matrix,
        load,
        np.zeros(2),
        np.zeros(2),
        np.ones(2),
    )

    assert result.converged
    np.testing.assert_allclose(result.state, np.array([0.0, 0.5]), atol=1.0e-12)
    assert not np.allclose(result.state, clipped)
    gradient = matrix @ result.state - load
    assert gradient[0] >= 0.0
    assert abs(gradient[1]) < 1.0e-12


def test_equality_bound_remains_fixed() -> None:
    """A Dirichlet-like equality bound is never released."""
    matrix = csr_matrix([[3.0, -1.0], [-1.0, 2.0]])
    result = solve_box_quadratic_active_set(
        matrix,
        np.array([4.0, 1.0]),
        np.array([0.5, 0.0]),
        np.array([0.5, -np.inf]),
        np.array([0.5, np.inf]),
    )

    assert result.converged
    np.testing.assert_allclose(result.state, np.array([0.5, 0.75]), atol=1.0e-12)
