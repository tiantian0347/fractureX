# fracturex/utilfuc/recover_strain.py
"""Recover strain from Hu-Zhang stress: ε^h = A(d) σ.

See docs/operator_learning/plan_operator_learning.md §3.3' (history field discretization).
"""
from __future__ import annotations
from typing import Optional

import numpy as np

SCHEMA_VERSION = "0.1"


def _compliance_apply(sigma_qp: np.ndarray, lam: float, mu: float) -> np.ndarray:
    """Apply C^{-1} to (..., 2, 2) symmetric stress (plane-strain Lamé).

    σ = λ tr(ε) I + 2μ ε  ⇒  ε = (1/(2μ)) (σ - (λ/(2λ+2μ)) tr(σ) I).
    """
    sigma_qp = np.asarray(sigma_qp, dtype=np.float64)
    if sigma_qp.shape[-2:] != (2, 2):
        raise ValueError(
            f"sigma_qp last two dims must be (2,2); got {sigma_qp.shape}"
        )
    tr_sigma = sigma_qp[..., 0, 0] + sigma_qp[..., 1, 1]
    coef = lam / (2.0 * (lam + mu))
    eye = np.eye(2, dtype=np.float64)
    eps = (sigma_qp - coef * tr_sigma[..., None, None] * eye) / (2.0 * mu)
    return eps


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
        lam, mu:  Lamé parameters (plane-strain convention).
        eta:      g(d) = (1-d)^2 + eta floor.
        formulation: 'standard' uses A(d) = (g(d) C)^{-1};
                     'effective_stress' uses A_eff(d) = C^{-1} (no damage in compliance).

    Returns:
        eps_qp: (NC, NQ, 2, 2) recovered symmetric strain.

    Implementation note (plan §3.3'): in Hu-Zhang the discrete displacement
    u_h is L^2 and cannot be differentiated; the history field driver must
    use the stress-recovered strain here, not ∇u_h.
    """
    sigma_qp = np.asarray(sigma_qp, dtype=np.float64)
    d_qp = np.asarray(d_qp, dtype=np.float64)
    if sigma_qp.ndim != 4 or sigma_qp.shape[-2:] != (2, 2):
        raise ValueError(
            f"sigma_qp must have shape (NC,NQ,2,2); got {sigma_qp.shape}"
        )
    if d_qp.shape != sigma_qp.shape[:2]:
        raise ValueError(
            f"d_qp must have shape (NC,NQ)={sigma_qp.shape[:2]}; got {d_qp.shape}"
        )

    eps = _compliance_apply(sigma_qp, lam=lam, mu=mu)

    if formulation == "effective_stress":
        return eps
    if formulation != "standard":
        raise ValueError(
            f"Unknown formulation {formulation!r}; expected 'standard' or 'effective_stress'."
        )

    d_clip = np.clip(d_qp, 0.0, 1.0)
    g = (1.0 - d_clip) ** 2 + float(eta)
    return eps / g[..., None, None]


def _macaulay(x: np.ndarray):
    """Macaulay 括号：返回 ``(⟨x⟩₊, ⟨x⟩₋)`` 即正部与负部。输入标量或数组 ``x``。"""
    ax = np.abs(x)
    return 0.5 * (x + ax), 0.5 * (x - ax)


def _spectral_pm(eps: np.ndarray):
    """Return (eps_plus, eps_minus) via eigendecomposition (Miehe split)."""
    w, v = np.linalg.eigh(eps)
    wp, wm = _macaulay(w)
    GD = eps.shape[-1]
    sp = np.zeros_like(eps)
    sm = np.zeros_like(eps)
    for i in range(GD):
        ni = v[..., i]
        sp = sp + (wp[..., i, None] * ni)[..., None] * ni[..., None, :]
        sm = sm + (wm[..., i, None] * ni)[..., None] * ni[..., None, :]
    return sp, sm


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
        split: 'miehe_spectral' | 'no_split'.

    Returns:
        psi_plus: (NC, NQ) energy density at each quadrature point.

    Reference: plan §3.0' equation for ψ⁺/ψ⁻ with Miehe spectral split.
    """
    eps_qp = np.asarray(eps_qp, dtype=np.float64)
    if eps_qp.ndim < 2 or eps_qp.shape[-2:] != (2, 2):
        raise ValueError(
            f"eps_qp last two dims must be (2,2); got {eps_qp.shape}"
        )
    tr = eps_qp[..., 0, 0] + eps_qp[..., 1, 1]

    if split == "no_split":
        e2 = np.einsum("...ij,...ij->...", eps_qp, eps_qp)
        psi = 0.5 * lam * tr * tr + mu * e2
        return np.maximum(psi, 0.0)

    if split == "miehe_spectral":
        eps_p, _ = _spectral_pm(eps_qp)
        tr_p, _ = _macaulay(tr)
        epp2 = np.einsum("...ii", eps_p @ eps_p)
        psi_plus = 0.5 * lam * tr_p * tr_p + mu * epp2
        return np.maximum(psi_plus, 0.0)

    raise ValueError(
        f"Unknown split {split!r}; expected 'miehe_spectral' or 'no_split'."
    )
