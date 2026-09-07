"""Tests for configurable residual stiffness in standard-FEM degradation laws."""
from __future__ import annotations

import numpy as np
import pytest

from fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction
from fracturex.phasefield.main_solve import MainSolve


def test_quadratic_default_preserves_historical_additive_floor():
    law = EnergyDegradationFunction()
    damage = np.array([0.0, 0.25, 1.0])
    expected = (1.0 - damage) ** 2 + 1.0e-10
    np.testing.assert_allclose(law.degradation_function(damage), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(law.grad_degradation_function(damage), -2.0 * (1.0 - damage))
    assert law.grad_grad_degradation_function(0.3) == 2.0
    assert law.grad_degradation_function_constant_coef() == -2.0


def test_quadratic_convex_floor_matches_huzhang_convention_and_derivatives():
    stiffness = 1.0e-6
    law = EnergyDegradationFunction(
        residual_stiffness=stiffness,
        floor_mode="convex",
    )
    damage = np.array([0.0, 0.3, 1.0])
    expected = (1.0 - stiffness) * (1.0 - damage) ** 2 + stiffness
    np.testing.assert_allclose(law.degradation_function(damage), expected)
    np.testing.assert_allclose(
        law.grad_degradation_function(damage),
        -2.0 * (1.0 - stiffness) * (1.0 - damage),
    )
    assert law.degradation_function(0.0) == 1.0
    assert law.degradation_function(1.0) == stiffness
    assert law.grad_grad_degradation_function(0.3) == 2.0 * (1.0 - stiffness)
    assert law.grad_degradation_function_constant_coef() == -2.0 * (1.0 - stiffness)


def test_main_solve_forwards_degradation_parameters():
    solver = MainSolve(
        mesh=None,
        material_params={"E": 200.0, "nu": 0.2, "Gc": 1.0, "l0": 0.02},
        model_type="HybridModel",
    )
    solver.set_energy_degradation(
        degradation_type="quadratic",
        residual_stiffness=1.0e-6,
        floor_mode="convex",
    )
    assert solver.EDFunc.residual_stiffness == 1.0e-6
    assert solver.EDFunc.floor_mode == "convex"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residual_stiffness": 0.0},
        {"residual_stiffness": 1.1},
        {"floor_mode": "unknown"},
    ],
)
def test_invalid_quadratic_floor_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        EnergyDegradationFunction(**kwargs)
