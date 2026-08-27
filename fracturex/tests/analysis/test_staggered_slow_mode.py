"""Unit tests for the staggered slow-mode analysis utilities.

These tests validate numerical identities only.  The production standard-FE
integration is exercised by ``scripts/paper_solver/verify_slow_mode_fracturex.py``.
"""
from __future__ import annotations

import numpy as np

from fracturex.analysis.staggered_slow_mode import (
    apply_local_elimination_projection,
    augment_weighted_subspace_with_memory,
    coupled_mode_lift,
    coupled_slow_subspace_from_sweep_column,
    compute_cell_energy_from_diagonal_weight,
    diagonal_cell_weights,
    diagonal_patch_survival_factor,
    diagonal_patch_subspace_survival_factor,
    dominant_mode,
    finite_difference_jacobian,
    finite_difference_jacobian_rectangular,
    iterate_fixed_point,
    local_elimination_projection,
    online_increment_subspace,
    select_bulk_cells,
    solve_local_nonlinear_residual,
    spectral_slow_subspace,
    subspace_cell_trace_indicator,
    weighted_orthonormalize,
    weighted_principal_angles,
    weighted_survival_factor,
)


def test_fixed_point_jacobian_and_dominant_rate() -> None:
    """Recover a known affine propagation matrix and its asymptotic rate."""
    propagation = np.array([[0.45, 0.08], [0.0, 0.20]], dtype=np.float64)
    root = np.array([0.3, 0.4], dtype=np.float64)

    def apply_map(vector: np.ndarray) -> np.ndarray:
        return root + propagation @ (vector - root)

    trace = iterate_fixed_point(
        apply_map,
        np.array([1.0, -0.5]),
        atol=1.0e-15,
        rtol=1.0e-9,
        max_iterations=60,
    )
    jacobian = finite_difference_jacobian(apply_map, root, relative_step=1.0e-6)
    mode = dominant_mode(jacobian)

    assert trace.converged
    np.testing.assert_allclose(jacobian, propagation, rtol=1.0e-10, atol=1.0e-10)
    assert abs(mode.spectral_radius - 0.45) < 1.0e-10
    assert abs(trace.asymptotic_ratio - mode.spectral_radius) < 1.0e-7
    assert mode.eigen_residual < 1.0e-12


def test_bounded_difference_and_cell_energy_conservation() -> None:
    """Use feasible one-sided differences and conserve additive cell energy."""
    lower = np.array([0.0, 0.0, 1.0])
    upper = np.ones(3)
    point = np.array([0.0, 0.5, 1.0])
    diagonal = np.array([0.2, 0.3, 0.4])

    def apply_map(vector: np.ndarray) -> np.ndarray:
        return diagonal * vector

    jacobian = finite_difference_jacobian(
        apply_map,
        point,
        relative_step=1.0e-6,
        lower_bound=lower,
        upper_bound=upper,
    )
    np.testing.assert_allclose(np.diag(jacobian), [0.2, 0.3, 0.0], atol=1.0e-10)

    mode = np.array([1.0, 2.0, 3.0])
    weights = np.array([2.0, 1.0, 0.5])
    cell_to_dof = np.array([[0, 1], [1, 2]], dtype=np.int64)
    energy = compute_cell_energy_from_diagonal_weight(mode, weights, cell_to_dof)
    expected_total = float(np.sum(weights * mode**2))
    assert abs(float(np.sum(energy)) - expected_total) < 1.0e-14

    selected = select_bulk_cells(energy, theta=0.6)
    assert selected.dtype == np.bool_
    assert int(np.count_nonzero(selected)) == 1


def test_rectangular_finite_difference_jacobian() -> None:
    """Recover a known map with more output components than inputs."""
    operator = np.array(
        [[0.4, 0.1], [-0.2, 0.3], [1.2, -0.7]], dtype=np.float64
    )
    offset = np.array([0.1, -0.4, 0.3], dtype=np.float64)

    def apply_map(vector: np.ndarray) -> np.ndarray:
        return offset + operator @ vector

    jacobian = finite_difference_jacobian_rectangular(
        apply_map,
        np.array([0.0, 0.5]),
        relative_step=1.0e-6,
        lower_bound=np.array([0.0, -np.inf]),
        upper_bound=np.array([1.0, np.inf]),
    )
    np.testing.assert_allclose(jacobian, operator, atol=1.0e-10, rtol=1.0e-10)


def test_coupled_mode_lifting_matches_full_sweep() -> None:
    """The lifted mode must be an eigenvector of the complete sweep."""
    matrix_a = np.array([[2.0, 0.1], [0.1, 1.5]], dtype=np.float64)
    matrix_d = np.diag([1.2, 1.0, 1.4])
    matrix_b = np.array([[0.9, 0.25, 0.0], [0.0, 0.75, 0.15]])
    matrix_c = matrix_b.T
    propagation = np.linalg.solve(
        matrix_d, matrix_c @ np.linalg.solve(matrix_a, matrix_b)
    )
    values, vectors = np.linalg.eig(propagation)
    index = int(np.argmax(np.abs(values)))
    eigenvalue = complex(values[index])
    damage_mode = vectors[:, index]
    lifted = coupled_mode_lift(matrix_a, matrix_b, damage_mode, eigenvalue)
    full_sweep = np.block(
        [
            [np.zeros((2, 2)), -np.linalg.solve(matrix_a, matrix_b)],
            [np.zeros((3, 2)), propagation],
        ]
    )
    np.testing.assert_allclose(
        full_sweep @ lifted, eigenvalue * lifted, atol=1.0e-12, rtol=1.0e-12
    )

    damage_to_full = full_sweep[:, 2:]
    slow = coupled_slow_subspace_from_sweep_column(
        damage_to_full, relative_radius=0.9
    )
    assert slow.basis.shape[0] == full_sweep.shape[0]
    residual = full_sweep @ slow.basis - slow.basis @ (
        slow.basis.T @ full_sweep @ slow.basis
    )
    assert np.linalg.norm(residual) < 1.0e-12


def test_local_elimination_projection_is_a_root_linearization() -> None:
    """The local elimination derivative annihilates patch directions."""
    jacobian = np.array([[3.0, 0.4, 0.2], [0.4, 2.0, 0.1], [0.2, 0.1, 1.5]])
    patch = np.array([0, 2], dtype=np.int64)
    projection = local_elimination_projection(jacobian, patch)
    selector = np.eye(3)[:, patch]
    np.testing.assert_allclose(projection @ selector, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(projection @ projection, projection, atol=1.0e-12)
    np.testing.assert_allclose(
        jacobian @ projection, projection.T @ jacobian, atol=1.0e-12
    )
    vector = np.array([0.3, -0.4, 0.8])
    np.testing.assert_allclose(
        apply_local_elimination_projection(jacobian, patch, vector),
        projection @ vector,
        atol=1.0e-12,
    )


def test_subspace_trace_indicator_is_basis_invariant() -> None:
    """A trace indicator must not depend on the orthonormal subspace basis."""
    basis = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64
    )
    basis /= np.linalg.norm(basis, axis=0)
    local_weights = np.array(
        [
            [[2.0, 0.1], [0.1, 1.0]],
            [[1.0, 0.0], [0.0, 3.0]],
        ],
        dtype=np.float64,
    )
    connectivity = np.array([[0, 1], [2, 3]], dtype=np.int64)
    indicator = subspace_cell_trace_indicator(
        basis, local_weights, connectivity
    )
    rotation = np.array(
        [[np.cos(0.37), -np.sin(0.37)], [np.sin(0.37), np.cos(0.37)]]
    )
    rotated = subspace_cell_trace_indicator(
        basis @ rotation, local_weights, connectivity
    )
    np.testing.assert_allclose(indicator, rotated, atol=1.0e-12)


def test_general_survival_factor_identity() -> None:
    """The weighted survival factor is valid without a symmetric Jacobian."""
    projection = np.array(
        [[1.0, -0.2, 0.1], [0.0, 0.8, 0.3], [0.0, 0.0, 1.0]]
    )
    mode = np.array([0.4, -0.7, 0.2])
    weight = np.diag([2.0, 1.0, 3.0])
    measured = weighted_survival_factor(projection, mode, weight)
    reduced = projection @ mode
    expected = np.sqrt((reduced @ weight @ reduced) / (mode @ weight @ mode))
    np.testing.assert_allclose(measured, expected, atol=1.0e-14)


def test_diagonal_patch_survival_factor_supports_complex_modes() -> None:
    """The SPD coordinate-patch factor equals the uncaptured modal energy."""
    mode = np.array([1.0 + 2.0j, -0.5j, 2.0 - 1.0j])
    weight = np.array([2.0, 3.0, 0.5])
    patch = np.array([0, 2], dtype=np.int64)
    survival = diagonal_patch_survival_factor(mode, weight, patch)
    energy = weight * np.abs(mode) ** 2
    expected = np.sqrt(energy[1] / np.sum(energy))
    np.testing.assert_allclose(survival, expected, atol=1.0e-14)


def test_local_nonlinear_residual_eliminates_only_selected_coordinates() -> None:
    """A nonlinear local residual reaches its root without moving the exterior."""
    initial = np.array([0.4, -3.0, 0.8], dtype=np.float64)
    patch = np.array([0, 2], dtype=np.int64)

    def residual(state: np.ndarray) -> np.ndarray:
        return np.array([state[0] ** 2 - 1.0, state[2] - 0.25])

    result = solve_local_nonlinear_residual(
        residual,
        initial,
        patch,
        lower_bound=np.array([0.0, -1.0]),
        upper_bound=np.array([2.0, 1.0]),
        atol=1.0e-12,
        rtol=1.0e-10,
        max_iterations=12,
    )
    assert result.converged
    np.testing.assert_allclose(result.state, [1.0, -3.0, 0.25], atol=1.0e-9)
    assert result.final_residual_norm <= 1.0e-10
    assert result.iterations >= 1


def test_local_nonlinear_residual_respects_active_lower_bound() -> None:
    """The projected residual recognizes a valid lower-bound KKT state."""
    initial = np.array([0.0, 2.0], dtype=np.float64)

    def residual(state: np.ndarray) -> np.ndarray:
        return np.array([state[0] + 1.0])

    result = solve_local_nonlinear_residual(
        residual,
        initial,
        np.array([0], dtype=np.int64),
        lower_bound=np.array([0.0]),
        upper_bound=np.array([1.0]),
    )
    assert result.converged
    assert result.iterations == 0
    np.testing.assert_allclose(result.state, initial, atol=0.0)


def test_weighted_subspace_trace_indicator_uses_additive_cell_weights() -> None:
    """A weighted real slow subspace has a basis-invariant cell trace."""
    propagation = np.array(
        [[0.90, -0.20, 0.0], [0.20, 0.90, 0.0], [0.0, 0.0, 0.25]],
        dtype=np.float64,
    )
    slow = spectral_slow_subspace(propagation, relative_radius=0.9)
    assert slow.basis.shape == (3, 2)
    np.testing.assert_allclose(
        np.abs(slow.selected_eigenvalues),
        [np.sqrt(0.85), np.sqrt(0.85)],
        atol=1.0e-12,
    )

    weight = np.array([2.0, 3.0, 5.0])
    basis = weighted_orthonormalize(slow.basis, weight)
    np.testing.assert_allclose(
        basis.T @ (weight[:, None] * basis), np.eye(2), atol=1.0e-12
    )
    connectivity = np.array([[0, 1], [1, 2]], dtype=np.int64)
    local_weights = diagonal_cell_weights(weight, connectivity)
    indicator = subspace_cell_trace_indicator(basis, local_weights, connectivity)
    np.testing.assert_allclose(
        np.sum(indicator),
        np.trace(basis.T @ (weight[:, None] * basis)),
        atol=1.0e-12,
    )


def test_online_increment_subspace_recovers_weighted_two_mode_span() -> None:
    """Normalized recent increments recover their two-dimensional slow span."""
    weight = np.array([2.0, 3.0, 5.0])
    first = np.array([1.0 / np.sqrt(weight[0]), 0.0, 0.0])
    second = np.array([0.0, 1.0 / np.sqrt(weight[1]), 0.0])
    negligible = np.array([0.0, 0.0, 1.0 / np.sqrt(weight[2])])
    increments = np.vstack(
        [
            0.90**iteration * first
            + (-0.72) ** iteration * second
            + 1.0e-10 * negligible
            for iteration in range(7)
        ]
    )

    result = online_increment_subspace(
        increments,
        weight,
        window_size=5,
        relative_singular_value=1.0e-6,
        max_dimension=2,
    )
    reference = np.column_stack((first, second))
    angles = weighted_principal_angles(reference, result.basis, weight)

    assert result.dimension == 2
    assert result.window_size == 5
    assert np.all(result.singular_values[:-1] >= result.singular_values[1:])
    np.testing.assert_allclose(
        result.basis.T @ (weight[:, None] * result.basis),
        np.eye(2),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(angles, 0.0, atol=1.0e-8)
    assert 0.7 < result.contraction_estimate < 1.0


def test_weighted_principal_angles_reports_a_missing_dimension() -> None:
    """A contained rank-one estimate still exposes a rank-two reference gap."""
    weight = np.array([2.0, 3.0, 4.0])
    reference = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64
    )
    candidate = reference[:, :1]
    angles = weighted_principal_angles(reference, candidate, weight)
    np.testing.assert_allclose(angles, [0.0, 0.5 * np.pi], atol=1.0e-12)


def test_diagonal_patch_subspace_survival_matches_rank_one_factor() -> None:
    """The subspace definition reduces to the existing modal factor at rank one."""
    mode = np.array([1.0, -2.0, 0.5])
    weight = np.array([2.0, 1.5, 3.0])
    patch = np.array([0, 2], dtype=np.int64)
    subspace_factor = diagonal_patch_subspace_survival_factor(
        mode[:, None], weight, patch
    )
    modal_factor = diagonal_patch_survival_factor(mode, weight, patch)
    np.testing.assert_allclose(subspace_factor, modal_factor, atol=1.0e-14)


def test_memory_augmentation_retains_only_independent_directions() -> None:
    """Current directions have priority over repeated load-step memory."""
    weight = np.array([2.0, 3.0, 4.0])
    current = np.array([[1.0], [0.0], [0.0]])
    memory = np.array(
        [[1.0, 0.0], [1.0e-4, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    result = augment_weighted_subspace_with_memory(
        current,
        memory,
        weight,
        relative_independence=1.0e-2,
        max_dimension=2,
    )

    assert result.current_dimension == 1
    assert result.memory_candidate_dimension == 2
    assert result.retained_memory_dimension == 1
    assert result.dimension == 2
    assert result.independence_ratios[0] < 1.0e-2
    np.testing.assert_allclose(result.independence_ratios[1], 1.0, atol=1.0e-14)
    np.testing.assert_allclose(
        result.basis.T @ (weight[:, None] * result.basis),
        np.eye(2),
        atol=1.0e-12,
    )
