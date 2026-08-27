"""Tests for the generic coupled projected Newton--Krylov kernel."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from fracturex.analysis.coupled_newton_solver import (
    CoupledNewtonConfig,
    solve_coupled_newton,
)


def test_coupled_newton_solves_small_nonlinear_block_system() -> None:
    """A smooth two-variable coupled system reaches its exact root."""

    def residual_and_jacobian(state: np.ndarray):
        u, d = state
        residual = np.array(
            [u + 0.2 * d**2 - 1.0, d + 0.1 * u - 0.5], dtype=np.float64
        )
        juu = csr_matrix([[1.0]])
        jud = csr_matrix([[0.4 * d]])
        jdu = csr_matrix([[0.1]])
        jdd = csr_matrix([[1.0]])
        return residual, juu, jud, jdu, jdd

    result = solve_coupled_newton(
        residual_and_jacobian,
        np.array([0.0, 0.0]),
        displacement_size=1,
        config=CoupledNewtonConfig(
            residual_atol=1.0e-12,
            residual_rtol=1.0e-10,
            gmres_max_iterations=8,
            max_newton_iterations=8,
        ),
    )

    assert result.converged
    assert result.termination_reason == "projected residual converged"
    assert np.linalg.norm(result.state - np.array([0.96747752, 0.40325225])) < 1.0e-7
    assert result.projected_residual_norms[-1] < 1.0e-10


def test_coupled_newton_respects_fixed_and_box_coordinates() -> None:
    """Active phase bounds and Dirichlet rows remain feasible."""

    def residual_and_jacobian(state: np.ndarray):
        residual = np.array([state[0] - 1.0, state[1] - 0.2])
        identity = csr_matrix(np.eye(2))
        return residual, identity[:1, :1], identity[:1, 1:], identity[1:, :1], identity[1:, 1:]

    result = solve_coupled_newton(
        residual_and_jacobian,
        np.array([0.0, 0.0]),
        displacement_size=1,
        lower_bound=np.array([-np.inf, 0.0]),
        upper_bound=np.array([np.inf, 0.1]),
        fixed_mask=np.array([True, False]),
        config=CoupledNewtonConfig(max_newton_iterations=4, gmres_max_iterations=4),
    )

    assert result.state[0] == 0.0
    assert 0.0 <= result.state[1] <= 0.1
    assert result.converged
