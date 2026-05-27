# fracturex/utilfuc/recover_strain.py
"""Recover strain from Hu-Zhang stress: ε^h = A(d) σ.

See docs/plan_operator_learning.md §3.3' (history field discretization).
"""
from __future__ import annotations
from typing import Optional

import numpy as np

SCHEMA_VERSION = "0.1"


def recover_strain_from_sigma(
    sigma_qp: np.ndarray,
    d_qp: np.ndarray,
    *,
    lam: float,
    mu: float,
    eta: float = 1e-9,
    formulation: str = "standard",
) -> np.ndarray:
    """Quadrature-level ε^h = A(d) σ.

    Args:
        sigma_qp: (NC, NQ, 2, 2) symmetric stress at quadrature points.
        d_qp:     (NC, NQ) damage at the same quadrature points.
        lam, mu:  Lamé parameters.
        eta:      g(d) = (1-d)^2 + eta.
        formulation: 'standard' uses A(d) = (g(d) C)^{-1};
                     'effective_stress' uses A_eff(d) = C^{-1} (no damage in compliance).

    Returns:
        eps_qp: (NC, NQ, 2, 2) recovered symmetric strain.

    Implementation note (plan §3.3'): in Hu-Zhang the discrete displacement
    u_h is L^2 and cannot be differentiated; the history field driver must
    use the stress-recovered strain here, not ∇u_h.
    """
    raise NotImplementedError("M0 task: implement recover_strain_from_sigma")


def positive_strain_energy_density(
    eps_qp: np.ndarray,
    *,
    lam: float,
    mu: float,
    split: str = "miehe_spectral",
) -> np.ndarray:
    """ψ⁺(ε): driver for the phase-field history field 𝓗.

    Args:
        eps_qp: (NC, NQ, 2, 2) strain tensor.
        lam, mu: Lamé parameters.
        split: 'miehe_spectral' | 'amor_volumetric' | 'no_split'.

    Returns:
        psi_plus: (NC, NQ) energy density at each quadrature point.

    Reference: plan §3.0' equation for ψ⁺/ψ⁻ with Miehe spectral split.
    """
    raise NotImplementedError("M0 task: implement positive_strain_energy_density")
