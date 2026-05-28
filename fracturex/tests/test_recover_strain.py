"""Unit tests for fracturex.utilfuc.recover_strain.

Analytical elastic test cases for:
  - recover_strain_from_sigma: round-trip σ = C(ε) → ε̂ = C^{-1} σ ≈ ε
    in plane-strain Lamé form.
  - positive_strain_energy_density: closed-form ψ⁺ for special states.
"""
from __future__ import annotations

import numpy as np
import pytest

from fracturex.utilfuc.recover_strain import (
    SCHEMA_VERSION,
    recover_strain_from_sigma,
    positive_strain_energy_density,
)


# Plane-strain Hooke: σ = λ tr(ε) I + 2μ ε
def _hooke(eps: np.ndarray, lam: float, mu: float) -> np.ndarray:
    tr = eps[..., 0, 0] + eps[..., 1, 1]
    eye = np.eye(2)
    return lam * tr[..., None, None] * eye + 2.0 * mu * eps


# Materials matching paper_direct_h1: E=200, nu=0.2  →  λ=55.555..., μ=83.333...
LAM = 55.55555555555556
MU = 83.33333333333334


def _strain_pure_uniaxial(eps0: float = 1e-3) -> np.ndarray:
    e = np.zeros((1, 1, 2, 2))
    e[0, 0, 0, 0] = eps0
    return e


def _strain_pure_shear(g0: float = 5e-4) -> np.ndarray:
    e = np.zeros((1, 1, 2, 2))
    e[0, 0, 0, 1] = g0
    e[0, 0, 1, 0] = g0
    return e


def _strain_pure_volumetric(ev: float = 1e-3) -> np.ndarray:
    e = np.zeros((1, 1, 2, 2))
    e[0, 0, 0, 0] = ev
    e[0, 0, 1, 1] = ev
    return e


class TestRecoverStrain:
    """ε̂ = A(d) σ should round-trip σ = C(ε) when d=0 (g=1)."""

    @pytest.mark.parametrize(
        "eps_factory",
        [_strain_pure_uniaxial, _strain_pure_shear, _strain_pure_volumetric],
    )
    @pytest.mark.parametrize("formulation", ["standard", "effective_stress"])
    def test_roundtrip_d0(self, eps_factory, formulation):
        eps = eps_factory()
        sigma = _hooke(eps, LAM, MU)
        d = np.zeros(eps.shape[:2])
        eps_hat = recover_strain_from_sigma(
            sigma, d, lam=LAM, mu=MU, eta=1e-9, formulation=formulation
        )
        # With d=0, η=1e-9: eps_hat = eps / (1 + η) ≈ eps for 'standard'.
        # For 'effective_stress', eps_hat = eps exactly.
        atol = 5e-12 if formulation == "effective_stress" else 5e-9
        np.testing.assert_allclose(eps_hat, eps, atol=atol, rtol=1e-9)

    def test_standard_scales_with_g(self):
        """With damage d, 'standard' formulation gives ε̂ = ε / g(d)."""
        eps = _strain_pure_uniaxial()
        sigma = _hooke(eps, LAM, MU)
        d_val = 0.3
        d = np.full(eps.shape[:2], d_val)
        eps_hat = recover_strain_from_sigma(
            sigma, d, lam=LAM, mu=MU, eta=0.0, formulation="standard"
        )
        g = (1.0 - d_val) ** 2
        np.testing.assert_allclose(eps_hat, eps / g, atol=1e-12, rtol=1e-12)

    def test_effective_stress_ignores_d(self):
        eps = _strain_pure_shear()
        sigma = _hooke(eps, LAM, MU)
        d = np.full(eps.shape[:2], 0.7)
        eps_hat = recover_strain_from_sigma(
            sigma, d, lam=LAM, mu=MU, formulation="effective_stress"
        )
        np.testing.assert_allclose(eps_hat, eps, atol=1e-12, rtol=1e-12)

    def test_batch_shape(self):
        rng = np.random.default_rng(0)
        NC, NQ = 7, 9
        eps = rng.standard_normal((NC, NQ, 2, 2)) * 1e-3
        eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))  # symmetrize
        sigma = _hooke(eps, LAM, MU)
        d = rng.uniform(0.0, 0.5, size=(NC, NQ))
        eps_hat = recover_strain_from_sigma(
            sigma, d, lam=LAM, mu=MU, eta=0.0, formulation="standard"
        )
        g = (1.0 - d) ** 2
        np.testing.assert_allclose(eps_hat * g[..., None, None], eps,
                                   atol=1e-12, rtol=1e-10)

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            recover_strain_from_sigma(
                np.zeros((4, 5, 3, 3)), np.zeros((4, 5)), lam=LAM, mu=MU
            )
        with pytest.raises(ValueError):
            recover_strain_from_sigma(
                np.zeros((4, 5, 2, 2)), np.zeros((4, 6)), lam=LAM, mu=MU
            )
        with pytest.raises(ValueError):
            recover_strain_from_sigma(
                np.zeros((4, 5, 2, 2)), np.zeros((4, 5)),
                lam=LAM, mu=MU, formulation="banana"
            )


class TestPositiveStrainEnergyDensity:

    def test_pure_volumetric_tension(self):
        ev = 1e-3
        eps = _strain_pure_volumetric(ev)
        # tr(ε)=2ev>0; eigenvalues both ev>0; ε₊=ε.
        # ψ⁺ = 0.5 λ (2ev)^2 + μ tr(ε²) = 0.5 λ (2ev)^2 + μ (2 ev²).
        psi_expected = 0.5 * LAM * (2 * ev) ** 2 + MU * 2 * ev ** 2
        psi = positive_strain_energy_density(eps, lam=LAM, mu=MU,
                                             split="miehe_spectral")
        np.testing.assert_allclose(psi[0, 0], psi_expected, rtol=1e-10)

    def test_pure_volumetric_compression(self):
        ev = -1e-3
        eps = _strain_pure_volumetric(ev)
        # tr(ε)=2ev<0 ⇒ ⟨tr⟩₊=0; eigenvalues both ev<0 ⇒ ε₊=0.
        # Therefore ψ⁺ = 0.
        psi = positive_strain_energy_density(eps, lam=LAM, mu=MU,
                                             split="miehe_spectral")
        np.testing.assert_allclose(psi[0, 0], 0.0, atol=1e-20)

    def test_pure_shear(self):
        # eps = [[0,g],[g,0]] has eigenvalues +g,-g; tr=0.
        # ε₊ projects onto +g eigenvector ⇒ tr(ε₊²) = g².
        # ψ⁺ = 0.5 λ ⟨0⟩₊² + μ g² = μ g².
        g0 = 5e-4
        eps = _strain_pure_shear(g0)
        psi = positive_strain_energy_density(eps, lam=LAM, mu=MU,
                                             split="miehe_spectral")
        np.testing.assert_allclose(psi[0, 0], MU * g0 ** 2, rtol=1e-10)

    def test_uniaxial_tension(self):
        eps0 = 1e-3
        eps = _strain_pure_uniaxial(eps0)
        # eigenvalues eps0, 0; tr=eps0>0; ε₊=ε.
        # ψ⁺ = 0.5 λ eps0² + μ eps0².
        psi_expected = 0.5 * LAM * eps0 ** 2 + MU * eps0 ** 2
        psi = positive_strain_energy_density(eps, lam=LAM, mu=MU,
                                             split="miehe_spectral")
        np.testing.assert_allclose(psi[0, 0], psi_expected, rtol=1e-10)

    def test_no_split_matches_full_energy(self):
        # Pure volumetric tension: 'no_split' = isotropic full ψ.
        ev = 1e-3
        eps = _strain_pure_volumetric(ev)
        psi_full = 0.5 * LAM * (2 * ev) ** 2 + MU * 2 * ev ** 2
        psi = positive_strain_energy_density(eps, lam=LAM, mu=MU,
                                             split="no_split")
        np.testing.assert_allclose(psi[0, 0], psi_full, rtol=1e-10)

    def test_no_split_compression_is_positive(self):
        # 'no_split' is unsplit ψ ≥ 0; compression also stores energy.
        ev = -1e-3
        eps = _strain_pure_volumetric(ev)
        psi = positive_strain_energy_density(eps, lam=LAM, mu=MU,
                                             split="no_split")
        assert psi[0, 0] > 0.0

    def test_batch_shape(self):
        rng = np.random.default_rng(1)
        eps = rng.standard_normal((3, 4, 2, 2)) * 1e-3
        eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))
        psi = positive_strain_energy_density(eps, lam=LAM, mu=MU,
                                             split="miehe_spectral")
        assert psi.shape == (3, 4)
        assert np.all(psi >= 0.0)


def test_schema_version_pinned():
    assert SCHEMA_VERSION == "0.1"
