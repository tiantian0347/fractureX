"""Test extraction of solver-aware cost-model records."""

from __future__ import annotations

from scripts.paper_solver.summarize_cost_model import extract_cost_records


def test_cost_record_joins_condition_number_and_krylov_work() -> None:
    """The Pareto and reduced-solver records are joined by patch name."""
    checkpoint = {
        "case": {"mesh_size": 0.05, "cells": 10, "scalar_dofs": 8, "load": 0.1},
        "fixed_point": {"iterations": 20},
        "coupled_slow_subspace": {"state_size": 24},
        "slow_mode": {"spectral_radius": 0.9},
        "benefit_cost_pareto": {
            "records": [
                {
                    "patch": "damage_theta_00",
                    "region": "damage",
                    "theta": 0.6,
                    "local_patch_dofs": 12,
                    "phase_patch_dofs": 5,
                    "trace_fraction": 0.6,
                    "coupled_local_jacobian_condition_number": 120.0,
                    "phase_local_jacobian_condition_number": 8.0,
                    "coupled_region_survival_factor": 0.7,
                    "all_acceptance_checks_passed": True,
                }
            ]
        },
        "reduced_nonlinear_solver": {
            "baseline_staggered_iterations": 20,
            "baseline_residual_equivalent_evaluations": 20,
            "baseline_staggered_wall_time_seconds": 2.0,
            "patches": {
                "damage_theta_00": {
                    "all_acceptance_checks_passed": True,
                    "acceptance_checks": {"same_discrete_solution": True},
                        "local_patch_fraction_of_all_free_state_dofs": 0.2,
                        "pre_switch_external_schur_coupling_mean": 0.25,
                        "pre_switch_external_schur_coupling_max": 0.3,
                        "pre_switch_external_schur_coupling_samples": 2,
                        "pre_switch_external_schur_coupling_local_condition_number": 120.0,
                    "local_jacobian_assemblies": 3,
                    "local_jacobian_assembly_wall_time_seconds": 0.03,
                    "local_linear_solves": 4,
                    "local_linear_solve_wall_time_seconds": 0.01,
                    "physical_residual_evaluations": 6,
                    "jvp_evaluations": 5,
                    "krylov_iterations": 4,
                    "preconditioner_applications": 4,
                    "total_residual_equivalent_evaluations": 15,
                    "total_wall_time_including_warmup_seconds": 1.5,
                    "full_solution_l2_difference_from_staggered": 1e-9,
                    "local_projected_residual_norm": 1e-10,
                }
            },
        },
    }

    records = extract_cost_records("h005", {"checkpoints": [checkpoint], "scan": {}})

    assert len(records) == 1
    assert records[0]["coupled_condition_number"] == 120.0
    assert records[0]["pre_switch_external_schur_coupling_mean"] == 0.25
    assert records[0]["pre_switch_coupled_condition_number"] == 120.0
    assert records[0]["krylov_iterations"] == 4
    assert records[0]["work_ratio"] == 0.75
    assert records[0]["same_discrete_solution"] is True
    assert records[0]["all_acceptance_checks_passed"] is True


def test_requested_load_must_be_present() -> None:
    """A missing requested checkpoint is reported instead of silently omitted."""
    payload = {
        "checkpoints": [
            {
                "case": {"mesh_size": 0.05, "cells": 1, "scalar_dofs": 1, "load": 0.1},
                "fixed_point": {"iterations": 1},
                "slow_mode": {"spectral_radius": 0.5},
                "coupled_slow_subspace": {"state_size": 2},
                "benefit_cost_pareto": {"records": []},
            }
        ]
    }

    try:
        extract_cost_records("h005", payload, [0.2])
    except ValueError as error:
        assert "requested loads are absent" in str(error)
    else:
        raise AssertionError("missing requested load should raise ValueError")


def test_rejected_candidate_can_be_retained_with_online_features() -> None:
    """Rejected but converged candidates remain available for gate calibration."""
    checkpoint = {
        "case": {"mesh_size": 0.3, "cells": 10, "scalar_dofs": 8, "load": -0.1},
        "fixed_point": {"iterations": 20},
        "coupled_slow_subspace": {"state_size": 24},
        "slow_mode": {"spectral_radius": 0.85},
        "online_increment_slow_subspace": {
            "solver_region": {
                "selected_dimension": 1,
                "selected_cell_fraction": 0.3,
                "reference_trace_fraction": 0.7,
                "weighted_contraction_estimate": 0.8,
                "construction_wall_time_seconds": 0.02,
            }
        },
        "benefit_cost_pareto": {
            "records": [
                {
                    "patch": "slow_theta_00",
                    "region": "slow",
                    "theta": 0.6,
                    "local_patch_dofs": 12,
                    "phase_patch_dofs": 5,
                    "trace_fraction": 0.6,
                    "coupled_local_jacobian_condition_number": 120.0,
                    "coupled_region_survival_factor": 0.7,
                    "all_acceptance_checks_passed": False,
                }
            ]
        },
        "reduced_nonlinear_solver": {
            "baseline_staggered_iterations": 20,
            "baseline_residual_equivalent_evaluations": 20,
            "baseline_staggered_wall_time_seconds": 2.0,
            "patches": {
                "slow_theta_00": {
                    "converged": True,
                    "all_acceptance_checks_passed": False,
                    "acceptance_checks": {
                        "same_discrete_solution": True,
                        "lower_residual_equivalent_work": False,
                        "lower_wall_time": False,
                    },
                    "local_patch_fraction_of_all_free_state_dofs": 0.2,
                    "local_jacobian_assemblies": 3,
                    "local_jacobian_assembly_wall_time_seconds": 0.03,
                    "local_linear_solves": 4,
                    "local_linear_solve_wall_time_seconds": 0.01,
                    "physical_residual_evaluations": 6,
                    "jvp_evaluations": 5,
                    "krylov_iterations": 4,
                    "preconditioner_applications": 4,
                    "total_residual_equivalent_evaluations": 25,
                    "total_wall_time_including_warmup_seconds": 2.5,
                    "full_solution_l2_difference_from_staggered": 1e-9,
                    "local_projected_residual_norm": 1e-10,
                }
            },
        },
    }

    records = extract_cost_records(
        "model5", {"checkpoints": [checkpoint], "scan": {}}, include_rejected=True
    )

    assert len(records) == 1
    assert records[0]["all_acceptance_checks_passed"] is False
    assert records[0]["same_discrete_solution"] is True
    assert records[0]["online_dimension"] == 1
    assert records[0]["online_contraction_estimate"] == 0.8
