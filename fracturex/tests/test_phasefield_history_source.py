"""Unit checks for displacement- and stress-driven phase-field history sources.

The test covers the plane-strain compliance recovery only; coupled fracture-path
effects are evaluated by the paper experiment runner rather than this unit test.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fealpy.backend import backend_manager as bm

from fracturex.damage.phasefield_damage import PhaseFieldDamageModel


class _QuadratureField:
    """Return one fixed quadrature array for the minimal damage-model contract."""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.float64)

    def __call__(self, bcs, index=None):
        return self.values


def test_from_sigma_recovers_prescribed_strain_with_degradation() -> None:
    """Check ``A(d)sigma`` against an exactly constructed plane-strain state."""
    # This test builds its reference tensors with NumPy, so declare the backend
    # explicitly and keep the result independent of pytest execution order.
    bm.set_backend("numpy")
    lam = 2.0
    mu = 3.0
    damage_value = 0.25
    degradation = (1.0 - damage_value) ** 2
    strain = np.array([[[[0.02, 0.01], [0.01, -0.005]]]], dtype=np.float64)
    trace = np.trace(strain, axis1=-2, axis2=-1)
    stress = degradation * (
        lam * trace[..., None, None] * np.eye(2) + 2.0 * mu * strain
    )
    stress_voigt = stress[..., (0, 0, 1), (0, 1, 1)]

    damage = PhaseFieldDamageModel(
        history_source="from_sigma",
        eps_g=1.0e-12,
        lam=lam,
        mu=mu,
    )
    damage._gfun = SimpleNamespace(
        degradation_function=lambda d: (1.0 - d) ** 2
    )
    state = SimpleNamespace(
        sigma=_QuadratureField(stress_voigt),
        d=_QuadratureField(np.full((1, 1), damage_value)),
    )

    recovered = np.asarray(
        damage._recover_strain_from_sigma(state, np.array([[1 / 3] * 3]))
    )
    np.testing.assert_allclose(recovered, strain, rtol=1.0e-12, atol=1.0e-12)
