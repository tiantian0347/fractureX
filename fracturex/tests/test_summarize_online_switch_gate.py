"""Test the online slow-rate switch gate extraction."""

from __future__ import annotations

from scripts.paper_solver.summarize_online_switch_gate import summarize_checkpoint


def _checkpoint(q_hat: float, *, accepted: bool) -> dict:
    """Build a minimal checkpoint for gate tests."""
    solver_patch = {
        "total_residual_equivalent_evaluations": 10,
        "total_wall_time_including_warmup_seconds": 1.0,
    }
    record = {
        "patch": "slow_theta_00",
        "all_acceptance_checks_passed": accepted,
    }
    return {
        "case": {"name": "test", "mesh_size": 0.1, "load": 0.1},
        "slow_mode": {"spectral_radius": q_hat},
        "fixed_point": {"iterations": 20},
        "online_increment_slow_subspace": {
            "solver_region": {
                "weighted_contraction_estimate": q_hat,
                "selected_dimension": 1,
                "selected_cell_fraction": 0.2,
                "reference_trace_fraction": 0.7,
            }
        },
        "benefit_cost_pareto": {"records": [record]},
        "reduced_nonlinear_solver": {
            "baseline_staggered_iterations": 20,
            "baseline_residual_equivalent_evaluations": 20,
            "baseline_staggered_wall_time_seconds": 2.0,
            "patches": {"slow_theta_00": solver_patch},
        },
    }


def test_low_online_rate_keeps_staggered_solver() -> None:
    """A low online rate does not switch when no candidate is beneficial."""
    record = summarize_checkpoint(
        "model5", _checkpoint(0.846, accepted=False), rho_low=0.87, rho_high=0.89
    )
    assert record["recommendation"] == "continue_staggered"
    assert record["gate_validation"] is True


def test_high_online_rate_allows_reduced_solver() -> None:
    """A high online rate agrees with an accepted reduced candidate."""
    record = summarize_checkpoint(
        "model0", _checkpoint(0.91, accepted=True), rho_low=0.87, rho_high=0.89
    )
    assert record["recommendation"] == "switch_reduced_ne"
    assert record["accepted_candidate_count"] == 1
    assert record["gate_validation"] is True
