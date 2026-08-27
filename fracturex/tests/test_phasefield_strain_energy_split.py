"""Unit tests for backend-neutral phase-field strain-energy splits.

Run with:
``pytest -q fracturex/tests/test_phasefield_strain_energy_split.py``
from the FractureX repository root.
"""

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction
from fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory
from fracturex.phasefield.strain_energy_split import (
    StrainEnergySplitFactory,
    engineering_strain_basis,
    symmetric_tensor_to_voigt,
)


BACKENDS = ("numpy", "pytorch", "jax")
SPLIT_NAMES = ("isotropic", "lancioni", "amor", "spectral")
MODEL_NAMES = (
    "IsotropicModel",
    "AnisotropicModel",
    "DeviatoricModel",
    "SpectralModel",
    "HybridModel",
)
MATERIAL = {"lam": 12.0, "mu": 7.0}


def _sample_strain(gd: int) -> np.ndarray:
    """Return two deterministic symmetric strain states away from switch points."""
    grid = np.arange(1, gd * gd + 1, dtype=np.float64).reshape(gd, gd)
    first = 1.0e-2 * (grid + grid.T) / (2.0 * gd)
    first[np.diag_indices(gd)] += np.linspace(-0.015, 0.025, gd)
    second = -0.6 * first
    second[np.diag_indices(gd)] -= 0.004
    return np.stack((first, second), axis=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("gd", (2, 3, 4))
@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_split_recombines_intact_elasticity(backend, gd, split_name):
    """Every split must reproduce intact energy, stress, and tangent on recombination."""
    bm.set_backend(backend)
    strain = bm.tensor(_sample_strain(gd), dtype=bm.float64)
    split = StrainEnergySplitFactory.create(split_name, MATERIAL["lam"], MATERIAL["mu"])

    energy_positive, energy_negative = split.energy_density_decomposition(strain)
    stress_positive, stress_negative = split.stress_decomposition(strain)
    tangent_positive, tangent_negative = split.tangent_decomposition(strain)

    np.testing.assert_allclose(
        bm.to_numpy(energy_positive + energy_negative),
        bm.to_numpy(split.elastic_energy_density(strain)),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        bm.to_numpy(stress_positive + stress_negative),
        bm.to_numpy(split.elastic_stress(strain)),
        rtol=2.0e-11,
        atol=2.0e-12,
    )

    _, _, symmetric_identity = split._identity_tensors(strain)
    reference_tangent = (
        MATERIAL["lam"]
        * bm.einsum(
            "ij,kl->ijkl",
            bm.eye(gd, **bm.context(strain)),
            bm.eye(gd, **bm.context(strain)),
        )
        + 2.0 * MATERIAL["mu"] * symmetric_identity
    )
    reference_tangent = bm.broadcast_to(
        reference_tangent, strain.shape[:-2] + reference_tangent.shape
    )
    np.testing.assert_allclose(
        bm.to_numpy(tangent_positive + tangent_negative),
        bm.to_numpy(reference_tangent),
        rtol=5.0e-10,
        atol=5.0e-11,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("gd", (2, 3))
def test_unilateral_splits_do_not_degrade_hydrostatic_compression(backend, gd):
    """Amor and Miehe tensile energies vanish under hydrostatic compression."""
    bm.set_backend(backend)
    strain = -0.02 * bm.eye(gd, dtype=bm.float64)

    for split_name in ("amor", "spectral"):
        split = StrainEnergySplitFactory.create(
            split_name, MATERIAL["lam"], MATERIAL["mu"]
        )
        energy_positive, _ = split.energy_density_decomposition(strain)
        stress_positive, _ = split.stress_decomposition(strain)
        np.testing.assert_allclose(bm.to_numpy(energy_positive), 0.0, atol=1.0e-14)
        np.testing.assert_allclose(bm.to_numpy(stress_positive), 0.0, atol=1.0e-14)


@pytest.mark.parametrize("backend", BACKENDS)
def test_spectral_split_is_rotation_invariant(backend):
    """Spectral energies and stresses transform objectively under a 3-D rotation."""
    bm.set_backend(backend)
    strain_numpy = np.array(
        [[0.021, 0.004, -0.002], [0.004, -0.013, 0.003], [-0.002, 0.003, 0.008]],
        dtype=np.float64,
    )
    angle = 0.37
    rotation_numpy = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    strain = bm.tensor(strain_numpy, dtype=bm.float64)
    rotation = bm.tensor(rotation_numpy, dtype=bm.float64)
    rotated_strain = bm.einsum("ik,kl,jl->ij", rotation, strain, rotation)
    split = StrainEnergySplitFactory.create("spectral", MATERIAL["lam"], MATERIAL["mu"])

    energy = split.energy_density_decomposition(strain)
    rotated_energy = split.energy_density_decomposition(rotated_strain)
    stress = split.stress_decomposition(strain)
    rotated_stress = split.stress_decomposition(rotated_strain)

    for value, rotated_value in zip(energy, rotated_energy):
        np.testing.assert_allclose(
            bm.to_numpy(value), bm.to_numpy(rotated_value), rtol=2.0e-11, atol=2.0e-12
        )
    for value, rotated_value in zip(stress, rotated_stress):
        expected = bm.einsum("ik,kl,jl->ij", rotation, value, rotation)
        np.testing.assert_allclose(
            bm.to_numpy(rotated_value),
            bm.to_numpy(expected),
            rtol=2.0e-10,
            atol=2.0e-11,
        )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("gd", (2, 3))
def test_degraded_tangent_matches_stress_finite_difference(model_name, gd):
    """Material Voigt tangents must be derivatives of degraded stresses."""
    bm.set_backend("numpy")
    strain = bm.tensor(_sample_strain(gd)[0], dtype=bm.float64)
    phase = 0.27
    material = PhaseFractureMaterialFactory.create(
        model_name, MATERIAL, EnergyDegradationFunction()
    )
    tangent = bm.to_numpy(material.elastic_matrix_from_strain(strain, phase))
    strain_basis = bm.to_numpy(engineering_strain_basis(strain))

    step = 2.0e-7
    finite_difference = np.empty_like(tangent)
    for column, direction in enumerate(strain_basis):
        positive = material.stress_from_strain(strain + step * direction, phase)
        negative = material.stress_from_strain(strain - step * direction, phase)
        finite_difference[:, column] = bm.to_numpy(
            symmetric_tensor_to_voigt(positive - negative)
        ) / (2.0 * step)

    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-7, atol=2.0e-8)


def test_hybrid_uses_isotropic_equilibrium_and_spectral_history():
    """Hybrid response equals Bourdin mechanically and Miehe as crack driver."""
    bm.set_backend("numpy")
    strain = bm.tensor(_sample_strain(3), dtype=bm.float64)
    phase = bm.tensor([0.2, 0.4], dtype=bm.float64)
    degradation = EnergyDegradationFunction()
    hybrid = PhaseFractureMaterialFactory.create("HybridModel", MATERIAL, degradation)
    isotropic = PhaseFractureMaterialFactory.create("Bourdin", MATERIAL, degradation)
    spectral = PhaseFractureMaterialFactory.create("Miehe", MATERIAL, degradation)

    np.testing.assert_allclose(
        bm.to_numpy(hybrid.stress_from_strain(strain, phase)),
        bm.to_numpy(isotropic.stress_from_strain(strain, phase)),
    )
    hybrid_energy = hybrid.strain_energy_density_decomposition(strain)
    spectral_energy = spectral.strain_energy_density_decomposition(strain)
    for hybrid_part, spectral_part in zip(hybrid_energy, spectral_energy):
        np.testing.assert_allclose(bm.to_numpy(hybrid_part), bm.to_numpy(spectral_part))


def test_history_field_is_pointwise_irreversible():
    """A lower subsequent tensile energy cannot reduce stored history values."""
    bm.set_backend("numpy")
    material = PhaseFractureMaterialFactory.create(
        "SpectralModel", MATERIAL, EnergyDegradationFunction()
    )
    first_strain = bm.tensor(_sample_strain(2), dtype=bm.float64)
    first_history = bm.to_numpy(material.maximum_historical_field_from_strain(first_strain)).copy()
    second_history = bm.to_numpy(
        material.maximum_historical_field_from_strain(0.25 * first_strain)
    )
    np.testing.assert_allclose(second_history, first_history)


def test_factory_reports_unknown_model():
    """Invalid factory input must fail observably with the supplied name."""
    with pytest.raises(ValueError, match="not-a-model"):
        PhaseFractureMaterialFactory.create(
            "not-a-model", MATERIAL, EnergyDegradationFunction()
        )
