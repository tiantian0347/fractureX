"""Test pure region-budget helpers used by the paper verification driver.

The tests cover deterministic cell ranking and fixed-active-set free-DOF
accounting. Physical finite-element residual assembly is exercised by the
driver smoke runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.paper_solver.verify_slow_mode_fracturex import (
    _build_nested_schur_preconditioner,
    _build_outer_block_preconditioner,
    _build_phase_patch_budget_sweep,
    _compose_coupled_patch_jacobian,
    _estimate_external_schur_coupling,
    _merge_benefit_cost_pareto,
    _run_reference_free_warmup,
    _select_cells_to_free_dof_budget,
)


def test_outer_block_lu_preconditioner_handles_active_rows() -> None:
    """Block-LU keeps projected active coordinates as identity rows."""
    uu = csr_matrix(np.asarray([[4.0, 0.2], [0.2, 3.0]]))
    dd = csr_matrix(np.asarray([[5.0, 0.1], [0.1, 2.0]]))
    ud = csr_matrix(np.asarray([[0.3, 0.0], [0.0, 0.4]]))
    du = csr_matrix(np.asarray([[0.2, 0.0], [0.0, 0.5]]))
    preconditioner = _build_outer_block_preconditioner(
        uu,
        dd,
        ud,
        du,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.asarray([True, False, True, True]),
        2,
        "block_lu",
    )
    output = preconditioner(np.ones(4))
    assert np.allclose(output[1], 1.0)
    assert np.isfinite(output).all()


def test_nested_schur_preconditioner_returns_finite_action() -> None:
    """The local-factor nested Schur action is finite on a small block system."""
    uu = csr_matrix(np.asarray([[4.0, 0.2], [0.2, 3.0]]))
    dd = csr_matrix(np.asarray([[5.0, 0.1], [0.1, 2.0]]))
    ud = csr_matrix(np.asarray([[0.3, 0.0], [0.0, 0.4]]))
    du = csr_matrix(np.asarray([[0.2, 0.0], [0.0, 0.5]]))
    preconditioner = _build_nested_schur_preconditioner(
        uu,
        dd,
        ud,
        du,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.ones(4, dtype=bool),
        2,
        np.asarray([2], dtype=np.int64),
    )
    output = preconditioner(np.ones(4))
    assert output.shape == (4,)
    assert np.isfinite(output).all()


def test_reference_free_warmup_stops_on_persistent_slow_rate() -> None:
    """Adaptive warmup uses only accepted increments and a direct residual."""

    class FakeStepMap:
        full_state_size = 2

        def __init__(self) -> None:
            self.current = np.zeros(2, dtype=np.float64)
            self.values = iter((1.0, 1.95, 2.89, 3.82, 4.74))

        def __call__(self, damage: np.ndarray) -> np.ndarray:
            value = next(self.values)
            self.current = np.asarray([0.0, value], dtype=np.float64)
            return self.current[1:].copy()

        def current_full_state(self) -> np.ndarray:
            return self.current.copy()

    residual_counter = {"count": 0}

    def residual_norm(_state: np.ndarray) -> float:
        residual_counter["count"] += 1
        return 0.7 ** (residual_counter["count"] - 1)

    warmup = _run_reference_free_warmup(
        FakeStepMap(),
        np.zeros(1),
        fixed_mask=np.zeros(2, dtype=bool),
        mode="adaptive",
        fixed_sweeps=3,
        minimum_sweeps=2,
        maximum_sweeps=5,
        slow_rate_threshold=0.89,
        required_slow_steps=2,
        residual_tolerance=1.0e-12,
        residual_ratio_threshold=0.8,
        residual_norm_callback=residual_norm,
    )

    assert warmup["stop_reason"] == "slow_rate_and_residual_descent"
    assert warmup["sweeps"] == 4
    assert len(warmup["online_rates"]) == 2
    assert all(rate >= 0.89 for rate in warmup["online_rates"])


def test_reference_free_fixed_warmup_uses_requested_sweep_count() -> None:
    """Fixed warmup must not be truncated by the adaptive minimum setting."""

    class FakeStepMap:
        full_state_size = 1

        def __init__(self) -> None:
            self.current = np.zeros(1, dtype=np.float64)

        def __call__(self, damage: np.ndarray) -> np.ndarray:
            self.current = self.current + 1.0
            return self.current.copy()

        def current_full_state(self) -> np.ndarray:
            return self.current.copy()

    warmup = _run_reference_free_warmup(
        FakeStepMap(),
        np.zeros(1),
        fixed_mask=np.zeros(1, dtype=bool),
        mode="fixed",
        fixed_sweeps=4,
        minimum_sweeps=2,
        maximum_sweeps=8,
        slow_rate_threshold=0.89,
        required_slow_steps=2,
        residual_tolerance=1.0e-12,
        residual_ratio_threshold=0.8,
        residual_norm_callback=lambda _state: 1.0,
    )

    assert warmup["stop_reason"] == "fixed_sweeps"
    assert warmup["sweeps"] == 4


def test_cell_prefix_matches_union_dof_budget() -> None:
    """Shared cell DOFs are counted once when matching the target size."""
    scores = np.asarray([3.0, 2.0, 1.0])
    connectivity = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    free = np.ones(4, dtype=bool)

    selected = _select_cells_to_free_dof_budget(
        scores, connectivity, free, target_free_dofs=3
    )

    assert np.array_equal(selected, np.asarray([True, True, False]))


def test_cell_budget_ignores_fixed_or_active_coordinates() -> None:
    """Only coordinates in the supplied smooth free mask consume budget."""
    scores = np.asarray([3.0, 2.0, 1.0])
    connectivity = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    free = np.asarray([False, True, True, True])

    selected = _select_cells_to_free_dof_budget(
        scores, connectivity, free, target_free_dofs=2
    )

    assert np.array_equal(selected, np.asarray([True, True, False]))


def test_invalid_cell_budget_is_rejected() -> None:
    """A nonpositive free-DOF target violates the selection contract."""
    with pytest.raises(ValueError, match="must be positive"):
        _select_cells_to_free_dof_budget(
            np.asarray([1.0]),
            np.asarray([[0]], dtype=np.int64),
            np.asarray([True]),
            target_free_dofs=0,
        )


def test_phase_patch_sweep_matches_actual_elimination_budget() -> None:
    """Damage prefixes match the phase-only local LU dimension."""
    connectivity = np.asarray(
        [
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [3, 0, 7, 8],
        ],
        dtype=np.int64,
    )
    patches, records = _build_phase_patch_budget_sweep(
        [0.5, 0.8],
        np.asarray([4.0, 3.0, 2.0, 1.0]),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        connectivity,
        np.zeros(9, dtype=bool),
        displacement_dofs=4,
    )

    assert len(patches) == 8
    phase_patches = [
        patch for name, patch in patches.items() if not name.endswith("_coupled")
    ]
    assert all(np.all(patch >= 4) for patch in phase_patches)
    assert [record["phase_budget_absolute_mismatch"] for record in records] == [0, 0]
    assert [record["slow_phase_patch_dofs"] for record in records] == [3, 4]
    assert [record["damage_phase_patch_dofs"] for record in records] == [3, 4]


def test_coupled_patch_sweep_matches_actual_local_lu_dimension() -> None:
    """Coupled selection budgets include both displacement and phase DOFs."""
    connectivity = np.asarray(
        [
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [3, 0, 7, 8],
        ],
        dtype=np.int64,
    )
    patches, records = _build_phase_patch_budget_sweep(
        [0.5, 0.8],
        np.asarray([4.0, 3.0, 2.0, 1.0]),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        connectivity,
        np.zeros(9, dtype=bool),
        displacement_dofs=4,
        budget_space="coupled",
    )

    assert [record["budget_space"] for record in records] == [
        "coupled",
        "coupled",
    ]
    assert [record["local_budget_absolute_mismatch"] for record in records] == [
        0,
        0,
    ]
    for record in records:
        assert record["slow_local_patch_dofs"] == len(
            patches[record["slow_coupled_patch"]]
        )
        assert record["damage_local_patch_dofs"] == len(
            patches[record["damage_coupled_patch"]]
        )


def test_duplicate_patch_sweep_threshold_is_rejected() -> None:
    """Each threshold must identify one unambiguous candidate pair."""
    with pytest.raises(ValueError, match="unique"):
        _build_phase_patch_budget_sweep(
            [0.5, 0.5],
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([[0, 1]], dtype=np.int64),
            np.zeros(2, dtype=bool),
            displacement_dofs=1,
        )


def test_coupled_patch_jacobian_preserves_requested_coordinate_order() -> None:
    """The four FE blocks are extracted into an arbitrary mixed patch order."""
    jacobian_uu = csr_matrix(np.asarray([[2.0, 0.5], [0.5, 3.0]]))
    jacobian_dd = csr_matrix(np.asarray([[5.0, 1.0], [1.0, 7.0]]))
    jacobian_ud = csr_matrix(np.asarray([[11.0, 12.0], [13.0, 14.0]]))
    jacobian_du = csr_matrix(np.asarray([[21.0, 22.0], [23.0, 24.0]]))
    full = np.block(
        [[jacobian_uu.toarray(), jacobian_ud.toarray()],
         [jacobian_du.toarray(), jacobian_dd.toarray()]]
    )
    patch = np.asarray([3, 0, 2], dtype=np.int64)

    local = _compose_coupled_patch_jacobian(
        jacobian_uu,
        jacobian_dd,
        jacobian_ud,
        jacobian_du,
        patch,
        displacement_dofs=2,
    )

    assert np.array_equal(local, full[np.ix_(patch, patch)])


def test_external_schur_coupling_uses_free_patch_and_exterior_blocks() -> None:
    """The coupling diagnostic returns finite ratios for a mixed patch."""
    jacobian_uu = csr_matrix(np.asarray([[4.0, 0.2], [0.2, 3.0]]))
    jacobian_dd = csr_matrix(np.asarray([[5.0, 0.1], [0.1, 2.0]]))
    jacobian_ud = csr_matrix(np.asarray([[0.3, 0.0], [0.0, 0.4]]))
    jacobian_du = csr_matrix(np.asarray([[0.2, 0.0], [0.0, 0.5]]))

    result = _estimate_external_schur_coupling(
        jacobian_uu,
        jacobian_dd,
        jacobian_ud,
        jacobian_du,
        np.asarray([0, 2], dtype=np.int64),
        np.zeros(4, dtype=bool),
        displacement_dofs=2,
        samples=3,
    )

    assert result["sample_count"] == 3
    assert result["patch_dofs"] == 2
    assert result["outside_dofs"] == 2
    assert len(result["gamma_values"]) == 3
    assert np.isfinite(result["gamma_mean"])
    assert np.isfinite(result["local_condition_number"])


def test_benefit_cost_merge_marks_tradeoff_as_pareto_optimal() -> None:
    """A faster but weaker region and a slower stronger region both survive."""
    region_records = [
        {
            "theta": 0.5,
            "slow_patch": "slow_theta_00",
            "damage_patch": "damage_theta_00",
            "slow_coupled_patch": "slow_theta_00_coupled",
            "damage_coupled_patch": "damage_theta_00_coupled",
            "budget_metric": "non-Dirichlet phase local-patch DOFs",
            "target_local_patch_dofs": 3,
            "local_budget_absolute_mismatch": 0,
            "slow_cells": 2,
            "damage_cells": 2,
            "slow_coupled_cell_union_dofs": 8,
            "damage_coupled_cell_union_dofs": 8,
            "slow_trace_fraction": 0.55,
            "damage_trace_fraction": 0.45,
        }
    ]
    history = {
        "patches": {
            "slow_theta_00": {
                "slow_subspace_survival_factor": 1.0,
                "smooth_free_patch_dofs": 3,
                "local_jacobian_condition_number": 10.0,
            },
            "damage_theta_00": {
                "slow_subspace_survival_factor": 1.0,
                "smooth_free_patch_dofs": 3,
                "local_jacobian_condition_number": 9.0,
            },
            "slow_theta_00_coupled": {
                "slow_subspace_survival_factor": 0.5,
                "smooth_free_patch_dofs": 6,
                "local_jacobian_condition_number": 20.0,
            },
            "damage_theta_00_coupled": {
                "slow_subspace_survival_factor": 0.7,
                "smooth_free_patch_dofs": 6,
                "local_jacobian_condition_number": 18.0,
            },
        }
    }

    def solver_record(total_time: float) -> dict[str, object]:
        return {
            "phase_patch_dofs": 3,
            "phase_patch_fraction_of_all_free_state_dofs": 0.1,
            "outer_newton_iterations": 2,
            "local_linear_solves": 4,
            "local_linear_solve_wall_time_seconds": 0.01,
            "physical_residual_evaluations": 8,
            "total_residual_equivalent_evaluations": 12,
            "total_wall_time_including_warmup_seconds": total_time,
            "wall_time_speedup_over_staggered": 2.0 / total_time,
            "full_solution_l2_difference_from_staggered": 1.0e-9,
            "final_coupled_residual": {"projected_raw_norm": 1.0e-10},
            "all_acceptance_checks_passed": True,
        }

    merged = _merge_benefit_cost_pareto(
        region_records,
        history,
        {
            "patches": {
                "slow_theta_00": solver_record(1.5),
                "damage_theta_00": solver_record(1.0),
            }
        },
        spectral_radius=0.9,
    )

    rows = merged["records"]
    assert np.isclose(rows[0]["coupled_region_predicted_contraction_proxy"], 0.45)
    assert rows[0]["phase_elimination_survival_factor"] == 1.0
    assert [row["pareto_optimal_chi_vs_time"] for row in rows] == [True, True]
