"""Backend-neutral strain-energy splits for brittle phase-field fracture.

This module implements the Bourdin isotropic, Lancioni deviatoric/volumetric,
Amor volumetric-deviatoric, and Miehe spectral splits stated in the thesis.
The implementations operate only on symmetric strain tensors and do not depend
on a mesh, finite-element degree, or spatial discretization.

Notes
-----
- Strain tensors have shape ``(..., GD, GD)`` for any ``GD >= 2``.
- Fourth-order tangents use the index order ``(..., i, j, k, l)``.
- Voigt vectors store engineering shear strain and tensorial shear stress.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


def voigt_pairs(gd: int) -> Tuple[Tuple[int, int], ...]:
    """Return the off-diagonal Voigt ordering for spatial dimension ``gd``.

    The ordering groups pairs by index distance.  It gives ``(xy,)`` in 2-D
    and ``(xy, yz, xz)`` in 3-D, preserving FractureX's existing public order.

    Parameters
    ----------
    gd : int
        Geometric dimension. Must be at least two.

    Returns
    -------
    tuple of tuple of int
        Off-diagonal index pairs following the normal components.
    """
    if gd < 2:
        raise ValueError(f"The strain-energy splits require GD >= 2, got {gd}.")
    return tuple(
        (i, i + offset)
        for offset in range(1, gd)
        for i in range(gd - offset)
    )


def symmetric_tensor_to_voigt(tensor: TensorLike) -> TensorLike:
    """Flatten symmetric stress-like tensors into FractureX Voigt order.

    Parameters
    ----------
    tensor : TensorLike, shape (..., GD, GD)
        Symmetric tensor. Off-diagonal entries are stored without scaling,
        which is the stress-like Voigt convention.

    Returns
    -------
    TensorLike, shape (..., GD*(GD+1)/2)
        Newly allocated Voigt vectors.
    """
    gd = _validate_strain_shape(tensor)
    components = [tensor[..., i, i] for i in range(gd)]
    components.extend(tensor[..., i, j] for i, j in voigt_pairs(gd))
    return bm.stack(components, axis=-1)


def engineering_strain_basis(reference: TensorLike) -> TensorLike:
    """Build symmetric tensor bases for unit engineering Voigt strains.

    Parameters
    ----------
    reference : TensorLike, shape (..., GD, GD)
        Provides ``GD``, dtype, and device context. Leading dimensions are
        ignored.

    Returns
    -------
    TensorLike, shape (GD*(GD+1)/2, GD, GD)
        Basis tensors. Diagonal modes contain one; shear modes contain one
        half in both symmetric locations so that engineering shear equals one.
    """
    gd = _validate_strain_shape(reference)
    nvoigt = gd * (gd + 1) // 2
    basis = bm.zeros((nvoigt, gd, gd), **bm.context(reference))
    for i in range(gd):
        basis = bm.set_at(basis, (i, i, i), 1.0)
    for cursor, (i, j) in enumerate(voigt_pairs(gd), start=gd):
        basis = bm.set_at(basis, (cursor, i, j), 0.5)
        basis = bm.set_at(basis, (cursor, j, i), 0.5)
    return basis


def fourth_order_tensor_to_voigt(tangent: TensorLike) -> TensorLike:
    """Convert a fourth-order tangent to an engineering-strain Voigt matrix.

    Parameters
    ----------
    tangent : TensorLike, shape (..., GD, GD, GD, GD)
        Tensor satisfying ``d_sigma_ij = C_ijkl d_epsilon_kl``.

    Returns
    -------
    TensorLike, shape (..., NVoigt, NVoigt)
        Matrix mapping engineering Voigt strain increments to tensorial Voigt
        stress increments. A newly allocated tensor is returned.
    """
    if len(tangent.shape) < 4:
        raise ValueError(
            "A fourth-order tangent must have at least four axes; "
            f"got shape {tangent.shape}."
        )
    gd = tangent.shape[-1]
    if tangent.shape[-4:] != (gd, gd, gd, gd):
        raise ValueError(
            "The last four tangent axes must all have size GD; "
            f"got shape {tangent.shape}."
        )
    reference = tangent[..., 0, 0, :, :]
    basis = engineering_strain_basis(reference)
    response = bm.einsum("...ijkl,bkl->...bij", tangent, basis)
    direction_by_component = symmetric_tensor_to_voigt(response)
    return bm.swapaxes(direction_by_component, -2, -1)


def macaulay_brackets(alpha: TensorLike) -> Tuple[TensorLike, TensorLike]:
    """Return positive and negative Macaulay parts of ``alpha``."""
    magnitude = bm.abs(alpha)
    return 0.5 * (alpha + magnitude), 0.5 * (alpha - magnitude)


def heaviside(alpha: TensorLike, tolerance: float = 1.0e-12) -> TensorLike:
    """Evaluate a backend-neutral Heaviside function with ``H(0)=1/2``."""
    half = bm.ones_like(alpha) * 0.5
    return bm.where(
        alpha > tolerance,
        bm.ones_like(alpha),
        bm.where(alpha < -tolerance, bm.zeros_like(alpha), half),
    )


class StrainEnergySplit(ABC):
    """Uniform interface for positive/negative elastic-energy decompositions."""

    name = "base"

    def __init__(self, lame_lambda: float, shear_modulus: float) -> None:
        """Store Lamé parameters used by every split.

        Parameters
        ----------
        lame_lambda : float
            First Lamé parameter, in stress units.
        shear_modulus : float
            Shear modulus ``mu > 0``, in stress units.
        """
        if shear_modulus <= 0:
            raise ValueError("The shear modulus mu must be positive.")
        self.lam = lame_lambda
        self.mu = shear_modulus

    def bulk_modulus(self, gd: int) -> float:
        """Return the ``GD``-dimensional bulk modulus ``lambda + 2*mu/GD``."""
        kappa = self.lam + 2.0 * self.mu / gd
        if kappa <= 0:
            raise ValueError(
                "The material is not volumetrically stable in dimension "
                f"GD={gd}: lambda + 2*mu/GD = {kappa}."
            )
        return kappa

    @abstractmethod
    def energy_density_decomposition(
        self, strain: TensorLike
    ) -> Tuple[TensorLike, TensorLike]:
        """Return ``(psi_plus, psi_minus)`` with shape ``strain.shape[:-2]``."""

    @abstractmethod
    def stress_decomposition(
        self, strain: TensorLike
    ) -> Tuple[TensorLike, TensorLike]:
        """Return positive/negative stresses with the same shape as ``strain``."""

    @abstractmethod
    def tangent_decomposition(
        self, strain: TensorLike
    ) -> Tuple[TensorLike, TensorLike]:
        """Return positive/negative fourth-order tangents ``(..., GD, GD, GD, GD)``."""

    def elastic_energy_density(self, strain: TensorLike) -> TensorLike:
        """Return unsplit isotropic elastic energy density."""
        _validate_strain_shape(strain)
        trace = bm.einsum("...ii->...", strain)
        squared_norm = bm.einsum("...ij,...ij->...", strain, strain)
        return 0.5 * self.lam * trace**2 + self.mu * squared_norm

    def elastic_stress(self, strain: TensorLike) -> TensorLike:
        """Return unsplit isotropic Cauchy stress with shape ``(..., GD, GD)``."""
        gd = _validate_strain_shape(strain)
        identity = bm.eye(gd, **bm.context(strain))
        trace = bm.einsum("...ii->...", strain)
        return self.lam * trace[..., None, None] * identity + 2.0 * self.mu * strain

    def _identity_tensors(
        self, strain: TensorLike
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Return ``I``, ``I⊗I``, and the symmetric fourth-order identity."""
        gd = _validate_strain_shape(strain)
        identity = bm.eye(gd, **bm.context(strain))
        volumetric = bm.einsum("ij,kl->ijkl", identity, identity)
        symmetric = 0.5 * (
            bm.einsum("ik,jl->ijkl", identity, identity)
            + bm.einsum("il,jk->ijkl", identity, identity)
        )
        return identity, volumetric, symmetric

    @staticmethod
    def _broadcast_tangent(tangent: TensorLike, strain: TensorLike) -> TensorLike:
        """Broadcast a constant tangent over the strain batch dimensions."""
        return bm.broadcast_to(tangent, strain.shape[:-2] + tangent.shape)


class IsotropicEnergySplit(StrainEnergySplit):
    """Bourdin split: all elastic energy is degraded."""

    name = "isotropic"

    def energy_density_decomposition(self, strain):
        energy = self.elastic_energy_density(strain)
        return energy, bm.zeros_like(energy)

    def stress_decomposition(self, strain):
        stress = self.elastic_stress(strain)
        return stress, bm.zeros_like(stress)

    def tangent_decomposition(self, strain):
        _, volumetric, symmetric = self._identity_tensors(strain)
        positive = self.lam * volumetric + 2.0 * self.mu * symmetric
        positive = self._broadcast_tangent(positive, strain)
        return positive, bm.zeros_like(positive)


class LancioniEnergySplit(StrainEnergySplit):
    """Lancioni split: degrade deviatoric energy and retain volumetric energy."""

    name = "anisotropic"

    def energy_density_decomposition(self, strain):
        gd = _validate_strain_shape(strain)
        trace = bm.einsum("...ii->...", strain)
        identity = bm.eye(gd, **bm.context(strain))
        deviatoric = strain - trace[..., None, None] * identity / gd
        positive = self.mu * bm.einsum("...ij,...ij->...", deviatoric, deviatoric)
        negative = 0.5 * self.bulk_modulus(gd) * trace**2
        return positive, negative

    def stress_decomposition(self, strain):
        gd = _validate_strain_shape(strain)
        trace = bm.einsum("...ii->...", strain)
        identity = bm.eye(gd, **bm.context(strain))
        deviatoric = strain - trace[..., None, None] * identity / gd
        positive = 2.0 * self.mu * deviatoric
        negative = self.bulk_modulus(gd) * trace[..., None, None] * identity
        return positive, negative

    def tangent_decomposition(self, strain):
        gd = _validate_strain_shape(strain)
        _, volumetric, symmetric = self._identity_tensors(strain)
        positive = 2.0 * self.mu * (symmetric - volumetric / gd)
        negative = self.bulk_modulus(gd) * volumetric
        return (
            self._broadcast_tangent(positive, strain),
            self._broadcast_tangent(negative, strain),
        )


class VolumetricDeviatoricEnergySplit(StrainEnergySplit):
    """Amor split: degrade tensile volumetric and all deviatoric energy."""

    name = "volumetric_deviatoric"

    def energy_density_decomposition(self, strain):
        gd = _validate_strain_shape(strain)
        trace = bm.einsum("...ii->...", strain)
        trace_positive, trace_negative = macaulay_brackets(trace)
        identity = bm.eye(gd, **bm.context(strain))
        deviatoric = strain - trace[..., None, None] * identity / gd
        deviatoric_energy = self.mu * bm.einsum(
            "...ij,...ij->...", deviatoric, deviatoric
        )
        kappa = self.bulk_modulus(gd)
        positive = 0.5 * kappa * trace_positive**2 + deviatoric_energy
        negative = 0.5 * kappa * trace_negative**2
        return positive, negative

    def stress_decomposition(self, strain):
        gd = _validate_strain_shape(strain)
        trace = bm.einsum("...ii->...", strain)
        trace_positive, trace_negative = macaulay_brackets(trace)
        identity = bm.eye(gd, **bm.context(strain))
        deviatoric = strain - trace[..., None, None] * identity / gd
        kappa = self.bulk_modulus(gd)
        positive = (
            kappa * trace_positive[..., None, None] * identity
            + 2.0 * self.mu * deviatoric
        )
        negative = kappa * trace_negative[..., None, None] * identity
        return positive, negative

    def tangent_decomposition(self, strain):
        gd = _validate_strain_shape(strain)
        trace = bm.einsum("...ii->...", strain)
        _, volumetric, symmetric = self._identity_tensors(strain)
        kappa = self.bulk_modulus(gd)
        deviatoric_tangent = 2.0 * self.mu * (symmetric - volumetric / gd)
        h_positive = heaviside(trace)[..., None, None, None, None]
        h_negative = heaviside(-trace)[..., None, None, None, None]
        positive = deviatoric_tangent + kappa * h_positive * volumetric
        negative = kappa * h_negative * volumetric
        return positive, negative


class SpectralEnergySplit(StrainEnergySplit):
    """Miehe spectral split using principal strains and divided differences."""

    name = "spectral"

    def principal_strains(self, strain: TensorLike):
        """Return eigenvalues/eigenvectors of symmetric ``strain`` tensors."""
        _validate_strain_shape(strain)
        return bm.linalg.eigh(strain)

    def strain_decomposition(self, strain):
        """Return positive and negative spectral strain tensors."""
        eigenvalues, eigenvectors = self.principal_strains(strain)
        positive_values, negative_values = macaulay_brackets(eigenvalues)
        positive = bm.einsum(
            "...ia,...a,...ja->...ij", eigenvectors, positive_values, eigenvectors
        )
        negative = bm.einsum(
            "...ia,...a,...ja->...ij", eigenvectors, negative_values, eigenvectors
        )
        return positive, negative

    def energy_density_decomposition(self, strain):
        eigenvalues, _ = self.principal_strains(strain)
        positive_values, negative_values = macaulay_brackets(eigenvalues)
        trace = bm.einsum("...ii->...", strain)
        trace_positive, trace_negative = macaulay_brackets(trace)
        positive = (
            0.5 * self.lam * trace_positive**2
            + self.mu * bm.einsum("...i,...i->...", positive_values, positive_values)
        )
        negative = (
            0.5 * self.lam * trace_negative**2
            + self.mu * bm.einsum("...i,...i->...", negative_values, negative_values)
        )
        return positive, negative

    def stress_decomposition(self, strain):
        gd = _validate_strain_shape(strain)
        identity = bm.eye(gd, **bm.context(strain))
        trace = bm.einsum("...ii->...", strain)
        trace_positive, trace_negative = macaulay_brackets(trace)
        strain_positive, strain_negative = self.strain_decomposition(strain)
        positive = (
            self.lam * trace_positive[..., None, None] * identity
            + 2.0 * self.mu * strain_positive
        )
        negative = (
            self.lam * trace_negative[..., None, None] * identity
            + 2.0 * self.mu * strain_negative
        )
        return positive, negative

    def tangent_decomposition(self, strain):
        _, volumetric, _ = self._identity_tensors(strain)
        trace = bm.einsum("...ii->...", strain)
        eigenvalues, eigenvectors = self.principal_strains(strain)
        positive_values, negative_values = macaulay_brackets(eigenvalues)
        positive_projector = self._spectral_derivative(
            eigenvalues, eigenvectors, positive_values, sign=1.0
        )
        negative_projector = self._spectral_derivative(
            eigenvalues, eigenvectors, negative_values, sign=-1.0
        )
        positive = (
            self.lam
            * heaviside(trace)[..., None, None, None, None]
            * volumetric
            + 2.0 * self.mu * positive_projector
        )
        negative = (
            self.lam
            * heaviside(-trace)[..., None, None, None, None]
            * volumetric
            + 2.0 * self.mu * negative_projector
        )
        return positive, negative

    @staticmethod
    def _spectral_derivative(
        eigenvalues: TensorLike,
        eigenvectors: TensorLike,
        split_values: TensorLike,
        sign: float,
        tolerance: float = 1.0e-12,
    ) -> TensorLike:
        """Return the fourth-order derivative of a spectral Macaulay map.

        Repeated eigenvalues use the continuous divided-difference limit. At a
        zero eigenvalue the selected generalized derivative is one half.
        """
        difference = eigenvalues[..., :, None] - eigenvalues[..., None, :]
        value_difference = split_values[..., :, None] - split_values[..., None, :]
        is_distinct = bm.abs(difference) > tolerance
        safe_difference = bm.where(is_distinct, difference, bm.ones_like(difference))
        quotient = value_difference / safe_difference
        diagonal_derivative = heaviside(sign * eigenvalues)
        repeated_limit = 0.5 * (
            diagonal_derivative[..., :, None] + diagonal_derivative[..., None, :]
        )
        divided_difference = bm.where(is_distinct, quotient, repeated_limit)
        derivative = bm.einsum(
            "...ab,...ia,...jb,...ka,...lb->...ijkl",
            divided_difference,
            eigenvectors,
            eigenvectors,
            eigenvectors,
            eigenvectors,
        )
        # Strain increments are symmetric. Explicit minor symmetrization gives
        # the canonical fourth-order representation used by FE Voigt assembly.
        return 0.5 * (derivative + bm.swapaxes(derivative, -1, -2))


class StrainEnergySplitFactory:
    """Create thesis strain-energy splits from concise or literature names."""

    _ALIASES = {
        "isotropic": IsotropicEnergySplit,
        "bourdin": IsotropicEnergySplit,
        "anisotropic": LancioniEnergySplit,
        "lancioni": LancioniEnergySplit,
        "deviatoric": VolumetricDeviatoricEnergySplit,
        "volumetricdeviatoric": VolumetricDeviatoricEnergySplit,
        "amor": VolumetricDeviatoricEnergySplit,
        "spectral": SpectralEnergySplit,
        "miehe": SpectralEnergySplit,
    }

    @classmethod
    def create(
        cls, name: str, lame_lambda: float, shear_modulus: float
    ) -> StrainEnergySplit:
        """Create a split by model, author, or decomposition name."""
        key = _normalize_name(name)
        split_type = cls._ALIASES.get(key)
        if split_type is None:
            available = ", ".join(sorted(cls._ALIASES))
            raise ValueError(f"Unknown strain-energy split '{name}'. Available aliases: {available}.")
        return split_type(lame_lambda, shear_modulus)


def _normalize_name(name: str) -> str:
    """Normalize factory names while preserving descriptive error handling."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A non-empty strain-energy split name is required.")
    key = "".join(character for character in name.lower() if character.isalnum())
    return key[:-5] if key.endswith("model") else key


def _validate_strain_shape(strain: TensorLike) -> int:
    """Validate square strain axes and return ``GD``."""
    if len(strain.shape) < 2 or strain.shape[-1] != strain.shape[-2]:
        raise ValueError(
            "Strain tensors must have shape (..., GD, GD); "
            f"got {strain.shape}."
        )
    gd = strain.shape[-1]
    if gd < 2:
        raise ValueError(f"The strain-energy splits require GD >= 2, got {gd}.")
    return gd


__all__ = [
    "StrainEnergySplit",
    "IsotropicEnergySplit",
    "LancioniEnergySplit",
    "VolumetricDeviatoricEnergySplit",
    "SpectralEnergySplit",
    "StrainEnergySplitFactory",
    "engineering_strain_basis",
    "fourth_order_tensor_to_voigt",
    "heaviside",
    "macaulay_brackets",
    "symmetric_tensor_to_voigt",
    "voigt_pairs",
]
