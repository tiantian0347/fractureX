"""Test extraction of solver-aware mesh-robustness records."""

from __future__ import annotations

from scripts.paper_solver.summarize_mesh_robustness import summarize_checkpoint


def test_summary_marks_path_and_selects_cost_aware_candidate() -> None:
    """The lowest valid reduced work is selected and path status is preserved."""
    checkpoint = {
        "case": {"mesh_size": 0.05, "cells": 10, "scalar_dofs": 8, "load": 0.1},
        "checks": {"fixed_point_converged": True},
        "fixed_point": {"iterations": 20},
        "slow_mode": {"spectral_radius": 0.9},
        "coupled_slow_subspace": {"state_size": 24, "selected_dimension": 2},
        "localization": {"coupled_selected_cell_fraction": 0.2},
        "spd_patch_calibration": {
            "slow_patch_dofs": 5,
            "slow_patch_survival_factor": 0.5,
            "damage_patch_survival_factor": 0.7,
            "gradient_patch_survival_factor": 0.8,
        },
        "reduced_nonlinear_solver": {
            "patches": {
                "slow": {
                    "converged": True,
                    "local_patch_fraction_of_all_free_state_dofs": 0.2,
                    "total_residual_equivalent_evaluations": 18,
                    "residual_equivalent_work_reduction_fraction": 0.1,
                    "wall_time_speedup_over_staggered": 1.1,
                    "full_solution_l2_difference_from_staggered": 1e-9,
                },
                "damage": {
                    "converged": True,
                    "local_patch_fraction_of_all_free_state_dofs": 0.2,
                    "total_residual_equivalent_evaluations": 16,
                    "residual_equivalent_work_reduction_fraction": 0.2,
                    "wall_time_speedup_over_staggered": 1.2,
                    "full_solution_l2_difference_from_staggered": 2e-9,
                },
            }
        },
    }

    record = summarize_checkpoint(
        "h005",
        checkpoint,
        target_load=0.1,
        path_consistent=True,
        theta=0.7,
    )

    assert record["path_consistent"] is True
    assert record["best_reduced_region"] == "damage"
    assert record["best_reduced_work"] == 16
    assert record["slow_dimension"] == 2
