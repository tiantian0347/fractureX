"""Tests for the reduced nonlinear elimination solver.

The tests isolate three numerical contracts needed by the fracture solver:
exact Schur reduction for affine systems, preservation of a coupled nonlinear
root, and consistency of the natural residual at an active damage bound.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from fracturex.analysis import ReducedNewtonConfig, solve_reduced_nonlinear_system


def _tight_config() -> ReducedNewtonConfig:
    """Return deterministic tolerances for small dense verification systems."""
    return ReducedNewtonConfig(
        local_atol=1.0e-12,
        local_rtol=1.0e-12,
        outer_atol=1.0e-11,
        outer_rtol=1.0e-11,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        fd_step=1.0e-7,
    )


def test_affine_schur_reduction_recovers_full_root() -> None:
    """An affine coupled system is solved to its unreduced root."""
    matrix = np.array(
        [
            [6.0, 1.0, 1.0, 0.0],
            [1.0, 5.0, 0.0, 1.0],
            [1.0, 0.0, 4.0, 1.0],
            [0.0, 1.0, 1.0, 3.0],
        ]
    )
    exact = np.array([0.25, -0.5, 1.25, 0.75])
    right_hand_side = matrix @ exact

    def residual(state: np.ndarray) -> np.ndarray:
        return matrix @ state - right_hand_side

    def local_jacobian(
        state: np.ndarray, patch: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return matrix[np.ix_(patch, patch)]

    def build_preconditioner(
        state: np.ndarray,
        outside: np.ndarray,
        outside_interior: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:  # noqa: ARG001
        diagonal = np.diag(matrix)[outside]
        return lambda vector: vector / diagonal

    result = solve_reduced_nonlinear_system(
        residual,
        local_jacobian,
        np.zeros(4),
        np.array([0, 1]),
        jacobian_vector_product=lambda state, direction: matrix @ direction,
        reduced_preconditioner=build_preconditioner,
        config=_tight_config(),
    )

    assert result.converged
    np.testing.assert_allclose(result.state, exact, rtol=0.0, atol=1.0e-10)
    assert np.linalg.norm(residual(result.state)) < 1.0e-10
    assert result.outer_iterations == 1
    assert result.local_linear_solves >= 2
    assert result.preconditioner_applications > 0
    assert result.schur_direction_residual_norms.size == result.outer_iterations
    assert result.outer_step_lengths.size == result.outer_iterations
    assert result.outer_backtracking_counts.size == result.outer_iterations


def test_nonlinear_elimination_preserves_coupled_root() -> None:
    """Inner elimination plus the outer solve converges to the full root."""
    def residual(state: np.ndarray) -> np.ndarray:
        local, outside = state
        return np.array(
            [local * local + outside - 2.0, local + 2.0 * outside - 3.0]
        )

    def local_jacobian(state: np.ndarray, patch: np.ndarray) -> np.ndarray:
        assert patch.tolist() == [0]
        return np.array([[2.0 * state[0]]])

    def jacobian_vector_product(
        state: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [
                2.0 * state[0] * direction[0] + direction[1],
                direction[0] + 2.0 * direction[1],
            ]
        )

    result = solve_reduced_nonlinear_system(
        residual,
        local_jacobian,
        np.array([0.8, 0.5]),
        np.array([0]),
        jacobian_vector_product=jacobian_vector_product,
        config=_tight_config(),
    )

    assert result.converged
    np.testing.assert_allclose(result.state, np.ones(2), rtol=0.0, atol=1.0e-9)
    assert result.projected_residual_norms[-1] < 1.0e-10
    assert result.local_projected_residual_norm < 1.0e-11


def test_active_lower_bound_satisfies_natural_residual() -> None:
    """A damage coordinate can converge at its irreversible lower bound."""
    target = np.array([-1.0, 2.0])

    def residual(state: np.ndarray) -> np.ndarray:
        return state - target

    def local_jacobian(
        state: np.ndarray, patch: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return np.eye(patch.size)

    result = solve_reduced_nonlinear_system(
        residual,
        local_jacobian,
        np.zeros(2),
        np.array([0]),
        lower_bound=np.array([0.0, -np.inf]),
        jacobian_vector_product=lambda state, direction: direction,
        config=_tight_config(),
    )

    assert result.converged
    np.testing.assert_allclose(result.state, np.array([0.0, 2.0]), atol=1.0e-11)
    assert residual(result.state)[0] > 0.0
    assert result.projected_residual_norms[-1] < 1.0e-10


def test_matrix_free_finite_difference_path() -> None:
    """The default finite-difference Jv path solves the affine system."""
    matrix = np.array([[4.0, 1.0], [2.0, 3.0]])
    exact = np.array([0.5, -0.25])
    right_hand_side = matrix @ exact

    def residual(state: np.ndarray) -> np.ndarray:
        return matrix @ state - right_hand_side

    def local_jacobian(
        state: np.ndarray, patch: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return matrix[np.ix_(patch, patch)]

    result = solve_reduced_nonlinear_system(
        residual,
        local_jacobian,
        np.zeros(2),
        np.array([0]),
        config=_tight_config(),
    )

    assert result.converged
    np.testing.assert_allclose(result.state, exact, rtol=0.0, atol=1.0e-8)
    assert result.jvp_evaluations > 0
    assert result.residual_evaluations > result.outer_iterations


def test_state_diagnostic_callback_records_accepted_outer_states() -> None:
    """Audit callbacks observe states without changing the numerical result."""
    matrix = np.array([[4.0, 1.0], [2.0, 3.0]])
    exact = np.array([0.5, -0.25])
    right_hand_side = matrix @ exact
    records: list[tuple[int, float, int]] = []

    def residual(state: np.ndarray) -> np.ndarray:
        return matrix @ state - right_hand_side

    def local_jacobian(
        state: np.ndarray, patch: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return matrix[np.ix_(patch, patch)]

    def callback(
        outer_iteration: int,
        state: np.ndarray,
        projected: np.ndarray,
        interior: np.ndarray,
    ) -> dict[str, object]:
        records.append(
            (
                outer_iteration,
                float(np.linalg.norm(projected)),
                int(np.count_nonzero(interior)),
            )
        )
        return {"outer_iteration": outer_iteration, "state_norm": float(np.linalg.norm(state))}

    result = solve_reduced_nonlinear_system(
        residual,
        local_jacobian,
        np.zeros(2),
        np.array([0]),
        jacobian_vector_product=lambda state, direction: matrix @ direction,
        state_diagnostic_callback=callback,
        config=_tight_config(),
    )

    assert result.converged
    assert len(records) == result.outer_iterations + 1
    assert len(result.state_diagnostic_history) == len(records)
    assert result.state_diagnostic_wall_time_seconds >= 0.0
    np.testing.assert_allclose(result.state, exact, rtol=0.0, atol=1.0e-8)


def test_minimum_outer_iterations_support_reference_free_checks() -> None:
    """A forced outer correction is executed even under a loose residual test."""
    matrix = np.array([[4.0, 1.0], [2.0, 3.0]])
    exact = np.array([0.5, -0.25])
    right_hand_side = matrix @ exact

    def residual(state: np.ndarray) -> np.ndarray:
        return matrix @ state - right_hand_side

    def local_jacobian(
        state: np.ndarray, patch: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return matrix[np.ix_(patch, patch)]

    loose = ReducedNewtonConfig(
        outer_atol=1.0e2,
        outer_rtol=0.0,
        local_atol=1.0e-12,
        local_rtol=1.0e-12,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        minimum_outer_iterations=1,
    )
    result = solve_reduced_nonlinear_system(
        residual,
        local_jacobian,
        np.zeros(2),
        np.array([0]),
        jacobian_vector_product=lambda state, direction: matrix @ direction,
        config=loose,
    )

    assert result.converged
    assert result.outer_iterations >= 1
    np.testing.assert_allclose(result.state, exact, rtol=0.0, atol=1.0e-10)


def test_local_predictor_uses_implicit_map_derivative() -> None:
    """The local predictor preserves the root and reduces inner iterations."""
    def residual(state: np.ndarray) -> np.ndarray:
        local, outside = state
        return np.array(
            [np.exp(local) + outside - 3.0, local + 2.0 * outside - 2.0]
        )

    def local_jacobian(state: np.ndarray, patch: np.ndarray) -> np.ndarray:
        assert patch.tolist() == [0]
        return np.array([[np.exp(state[0])]])

    def jacobian_vector_product(
        state: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [
                np.exp(state[0]) * direction[0] + direction[1],
                direction[0] + 2.0 * direction[1],
            ]
        )

    common = dict(
        local_atol=1.0e-12,
        local_rtol=1.0e-12,
        outer_atol=1.0e-11,
        outer_rtol=1.0e-11,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        max_local_iterations=20,
    )
    baseline = solve_reduced_nonlinear_system(
        residual,
        local_jacobian,
        np.array([0.1, 0.1]),
        np.array([0]),
        jacobian_vector_product=jacobian_vector_product,
        config=ReducedNewtonConfig(**common),
    )
    predicted = solve_reduced_nonlinear_system(
        residual,
        local_jacobian,
        np.array([0.1, 0.1]),
        np.array([0]),
        jacobian_vector_product=jacobian_vector_product,
        config=ReducedNewtonConfig(**common, use_local_predictor=True),
    )

    assert baseline.converged and predicted.converged
    np.testing.assert_allclose(predicted.state, baseline.state, atol=1.0e-10)
    assert predicted.local_predictor_applications == predicted.outer_iterations
    assert predicted.local_newton_iterations < baseline.local_newton_iterations
