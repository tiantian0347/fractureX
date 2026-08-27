"""Finite-element integration of phase-field fracture strain-energy splits.

The material classes combine a mesh-independent strain-energy split with a
degradation function, finite-element displacement/phase fields, and an
irreversible history field. All standard models expose the same stress,
tangent, energy, and history interfaces.

Notes
-----
- Tensor APIs accept strain arrays with shape ``(..., GD, GD)`` for ``GD >= 2``.
- FE APIs accept arbitrary Lagrange degree; the material does not depend on ``p``.
- ``HybridModel`` uses isotropic degraded equilibrium and spectral tensile
  energy for the crack-driving history field, following Ambati et al.
"""

from typing import Dict, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric
from fealpy.functionspace.utils import flatten_indices
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.typing import TensorLike

from fracturex.phasefield.strain_energy_split import (
    StrainEnergySplitFactory,
    fourth_order_tensor_to_voigt,
    symmetric_tensor_to_voigt,
    voigt_pairs,
)


class BasedPhaseFractureMaterial(LinearElasticMaterial):
    """Common material adapter for a mechanical split and a history split.

    The class deliberately owns no mesh or polynomial-degree assumptions. A
    solver may either call the pure tensor methods directly or set FE fields
    with :meth:`update_disp` and :meth:`update_phase` before using the
    barycentric coefficient methods.
    """

    mechanical_split_name = "isotropic"
    history_split_name: Optional[str] = None

    def __init__(self, material: Dict, energy_degradation_fun) -> None:
        """Initialize material constants, energy splits, and mutable FE state.

        Parameters
        ----------
        material : dict
            Contains either ``lam`` and ``mu`` or ``E`` and ``nu``. Parameters
            use consistent stress units.
        energy_degradation_fun : object
            Provides ``degradation_function(d)`` and, for the phase equation,
            the derivative methods used by ``MainSolve``.
        """
        self._gd = energy_degradation_fun
        lame_lambda, shear_modulus, _, _ = _elastic_parameters(material)
        super().__init__(
            name=self.__class__.__name__,
            lame_lambda=lame_lambda,
            shear_modulus=shear_modulus,
        )

        self.mechanical_split = StrainEnergySplitFactory.create(
            self.mechanical_split_name, self.lam, self.mu
        )
        history_name = self.history_split_name or self.mechanical_split_name
        self.history_split = StrainEnergySplitFactory.create(
            history_name, self.lam, self.mu
        )

        self.uh = None
        self.d = None
        self.H = None

        # Backward-compatible aliases used by a few early FractureX scripts.
        self._uh = None
        self._d = None

    def update_disp(self, uh) -> None:
        """Set the current FE displacement field without copying it."""
        self.uh = uh
        self._uh = uh

    def update_phase(self, d) -> None:
        """Set the current FE phase field without copying it."""
        self.d = d
        self._d = d

    def update_historical_field(self, history: TensorLike) -> None:
        """Replace the stored quadrature-point history field ``H``."""
        self.H = history

    @barycentric
    def strain_value(self, bc=None) -> TensorLike:
        """Evaluate symmetric small strain at barycentric points.

        Parameters
        ----------
        bc : TensorLike
            Barycentric quadrature coordinates accepted by ``uh.grad_value``.

        Returns
        -------
        TensorLike, shape (NC, NQ, GD, GD)
            Symmetric small-strain tensors in cell/quadrature ordering.
        """
        if self.uh is None:
            raise RuntimeError("Set the displacement field with update_disp() first.")
        displacement_gradient = self.uh.grad_value(bc)
        return 0.5 * (
            displacement_gradient + bm.swapaxes(displacement_gradient, -2, -1)
        )

    def mechanical_energy_density_decomposition(
        self, strain: TensorLike
    ) -> Tuple[TensorLike, TensorLike]:
        """Return the equilibrium split ``(psi_plus, psi_minus)``."""
        return self.mechanical_split.energy_density_decomposition(strain)

    def strain_energy_density_decomposition(
        self, strain: TensorLike
    ) -> Tuple[TensorLike, TensorLike]:
        """Return the crack-driving split ``(psi_plus, psi_minus)``.

        For every non-hybrid material this equals the mechanical split. For the
        hybrid material it is spectral while equilibrium remains isotropic.
        """
        return self.history_split.energy_density_decomposition(strain)

    def stress_decomposition(
        self, strain: TensorLike
    ) -> Tuple[TensorLike, TensorLike]:
        """Return undegraded positive and negative stress tensors."""
        return self.mechanical_split.stress_decomposition(strain)

    def tangent_decomposition(
        self, strain: TensorLike
    ) -> Tuple[TensorLike, TensorLike]:
        """Return undegraded positive and negative fourth-order tangents."""
        return self.mechanical_split.tangent_decomposition(strain)

    def effective_stress_from_strain(self, strain: TensorLike) -> TensorLike:
        """Return intact isotropic stress for prescribed strain tensors."""
        return self.mechanical_split.elastic_stress(strain)

    def stress_from_strain(self, strain: TensorLike, phase: TensorLike) -> TensorLike:
        """Return degraded stress for prescribed strain and phase values.

        Parameters
        ----------
        strain : TensorLike, shape (..., GD, GD)
            Symmetric, dimensionless small strain.
        phase : TensorLike, shape broadcastable to strain.shape[:-2]
            Damage phase field, conventionally in ``[0, 1]``.

        Returns
        -------
        TensorLike, shape (..., GD, GD)
            ``g(phase)*sigma_plus + sigma_minus`` in stress units.
        """
        phase = _phase_tensor(phase, strain)
        positive, negative = self.stress_decomposition(strain)
        degradation = self._gd.degradation_function(phase)
        return degradation[..., None, None] * positive + negative

    def tangent_from_strain(self, strain: TensorLike, phase: TensorLike) -> TensorLike:
        """Return the degraded fourth-order algorithmic tangent tensor."""
        phase = _phase_tensor(phase, strain)
        positive, negative = self.tangent_decomposition(strain)
        degradation = self._gd.degradation_function(phase)
        return degradation[..., None, None, None, None] * positive + negative

    def elastic_matrix_from_strain(
        self, strain: TensorLike, phase: TensorLike
    ) -> TensorLike:
        """Return degraded engineering-Voigt tangents for prescribed states.

        Returns
        -------
        TensorLike, shape (..., NVoigt, NVoigt)
            Matrices with ``NVoigt=GD*(GD+1)/2``. They map engineering shear
            strains to tensorial shear stresses.
        """
        return fourth_order_tensor_to_voigt(self.tangent_from_strain(strain, phase))

    @barycentric
    def effective_stress(self, bc=None) -> TensorLike:
        """Evaluate intact elastic stress at FE quadrature points."""
        return self.effective_stress_from_strain(self.strain_value(bc))

    @barycentric
    def stress_value(self, bc=None) -> TensorLike:
        """Evaluate degraded stress at FE quadrature points."""
        strain = self.strain_value(bc)
        return self.stress_from_strain(strain, self._phase_value(bc))

    @barycentric
    def linear_elastic_matrix(self, bc=None) -> TensorLike:
        """Evaluate the intact isotropic engineering-Voigt tangent."""
        strain = self.strain_value(bc)
        positive, negative = self.tangent_decomposition(strain)
        return fourth_order_tensor_to_voigt(positive + negative)

    @barycentric
    def elastic_matrix(self, bc=None) -> TensorLike:
        """Evaluate degraded engineering-Voigt tangents at FE quadrature points."""
        strain = self.strain_value(bc)
        return self.elastic_matrix_from_strain(strain, self._phase_value(bc))

    def positive_stress_func(self, guh: TensorLike) -> TensorLike:
        """Map displacement gradients to positive Voigt stresses.

        This kernel is used by FEALPy's automatic nonlinear elastic integrator.
        Input shape is ``(..., GD, GD)`` and output shape is
        ``(..., GD*(GD+1)/2)``.
        """
        strain = 0.5 * (guh + bm.swapaxes(guh, -2, -1))
        positive, _ = self.stress_decomposition(strain)
        return symmetric_tensor_to_voigt(positive)

    def negative_stress_func(self, guh: TensorLike) -> TensorLike:
        """Map displacement gradients to negative Voigt stresses."""
        strain = 0.5 * (guh + bm.swapaxes(guh, -2, -1))
        _, negative = self.stress_decomposition(strain)
        return symmetric_tensor_to_voigt(negative)

    def maximum_historical_field_from_strain(self, strain: TensorLike) -> TensorLike:
        """Update and return ``H=max(H, psi_plus(strain))`` pointwise."""
        positive_energy, _ = self.strain_energy_density_decomposition(strain)
        if self.H is None:
            self.H = positive_energy
        else:
            if self.H.shape != positive_energy.shape:
                raise ValueError(
                    "Historical field shape does not match the current quadrature layout: "
                    f"H{self.H.shape} versus psi_plus{positive_energy.shape}."
                )
            self.H = bm.maximum(self.H, positive_energy)
        return self.H

    @barycentric
    def maximum_historical_field(self, bc) -> TensorLike:
        """Update the irreversible crack-driving history at FE quadrature points."""
        return self.maximum_historical_field_from_strain(self.strain_value(bc))

    def strain_matrix(
        self,
        dof_priority: bool,
        gphi: TensorLike,
        shear_order=None,
        correction=None,
        **kwargs,
    ) -> TensorLike:
        """Build a dimension-generic engineering strain-displacement matrix.

        Parameters
        ----------
        dof_priority : bool
            Tensor-space degree-of-freedom ordering used by FEALPy.
        gphi : TensorLike, shape (NC, NQ, LDOF, GD)
            Physical gradients of scalar basis functions.

        Returns
        -------
        TensorLike, shape (NC, NQ, GD*(GD+1)/2, GD*LDOF)
            Newly allocated matrix in the same Voigt order as
            :func:`symmetric_tensor_to_voigt`.
        """
        if correction is not None:
            raise NotImplementedError(
                "Phase-fracture strain_matrix currently supports the standard "
                "small-strain operator only; correction must be None."
            )
        ldof, gd = gphi.shape[-2:]
        expected_pairs = voigt_pairs(gd)
        if shear_order is not None:
            requested_pairs = _parse_shear_order(shear_order, gd)
            if requested_pairs != expected_pairs:
                raise ValueError(
                    f"shear_order={shear_order} conflicts with the material Voigt "
                    f"order {expected_pairs}."
                )

        if dof_priority:
            indices = flatten_indices((ldof, gd), (1, 0))
        else:
            indices = flatten_indices((ldof, gd), (0, 1))

        nvoigt = gd * (gd + 1) // 2
        matrix = bm.zeros(
            gphi.shape[:-2] + (nvoigt, gd * ldof), **bm.context(gphi)
        )
        for i in range(gd):
            matrix = bm.set_at(matrix, (..., i, indices[:, i]), gphi[..., :, i])
        for row, (i, j) in enumerate(expected_pairs, start=gd):
            matrix = bm.set_at(matrix, (..., row, indices[:, i]), gphi[..., :, j])
            matrix = bm.set_at(matrix, (..., row, indices[:, j]), gphi[..., :, i])
        return matrix

    def _phase_value(self, bc) -> TensorLike:
        """Evaluate a callable FE phase field or return an already sampled tensor."""
        if self.d is None:
            raise RuntimeError("Set the phase field with update_phase() first.")
        return self.d(bc) if callable(self.d) else self.d


class IsotropicModel(BasedPhaseFractureMaterial):
    """Bourdin isotropic model: degrade the full elastic response."""

    mechanical_split_name = "isotropic"


class AnisotropicModel(BasedPhaseFractureMaterial):
    """Lancioni model: degrade deviatoric response, retain volumetric response."""

    mechanical_split_name = "anisotropic"


class DeviatoricModel(BasedPhaseFractureMaterial):
    """Amor volumetric-deviatoric split with unilateral volumetric response."""

    mechanical_split_name = "volumetric_deviatoric"


class VolumetricDeviatoricModel(DeviatoricModel):
    """Descriptive alias for :class:`DeviatoricModel`."""


class SpectralModel(BasedPhaseFractureMaterial):
    """Miehe spectral tensile/compressive energy split."""

    mechanical_split_name = "spectral"

    def strain_pm_eig_decomposition(self, strain: TensorLike):
        """Backward-compatible access to positive/negative spectral strains."""
        return self.mechanical_split.strain_decomposition(strain)

    @staticmethod
    def macaulay_operation(alpha: TensorLike):
        """Backward-compatible positive/negative Macaulay operation."""
        from fracturex.phasefield.strain_energy_split import macaulay_brackets

        return macaulay_brackets(alpha)

    @staticmethod
    def heaviside(alpha: TensorLike):
        """Backward-compatible Heaviside convention with ``H(0)=1/2``."""
        from fracturex.phasefield.strain_energy_split import heaviside

        return heaviside(alpha)


class HybridModel(BasedPhaseFractureMaterial):
    """Ambati hybrid: isotropic equilibrium and spectral history driving force."""

    mechanical_split_name = "isotropic"
    history_split_name = "spectral"


class PhaseFractureMaterialFactory:
    """Create phase-field material adapters from model or literature names."""

    _MODELS = {
        "isotropic": IsotropicModel,
        "bourdin": IsotropicModel,
        "anisotropic": AnisotropicModel,
        "lancioni": AnisotropicModel,
        "deviatoric": DeviatoricModel,
        "volumetricdeviatoric": VolumetricDeviatoricModel,
        "amor": VolumetricDeviatoricModel,
        "spectral": SpectralModel,
        "miehe": SpectralModel,
        "hybrid": HybridModel,
        "ambati": HybridModel,
    }

    @classmethod
    def create(cls, model_type, material, energy_degradation_fun):
        """Instantiate a model with a uniform constructor and interface."""
        key = _normalize_model_name(model_type)
        model_class = cls._MODELS.get(key)
        if model_class is None:
            available = ", ".join(sorted(cls._MODELS))
            raise ValueError(
                f"Unknown phase-fracture model '{model_type}'. Available aliases: {available}."
            )
        return model_class(material, energy_degradation_fun)


def _elastic_parameters(material: Dict):
    """Validate and return ``(lambda, mu, E, nu)`` from a material mapping."""
    if "lam" in material and "mu" in material:
        lame_lambda = material["lam"]
        shear_modulus = material["mu"]
        denominator = lame_lambda + shear_modulus
        if denominator == 0:
            raise ValueError("lam + mu must be nonzero when deriving E and nu.")
        elastic_modulus = shear_modulus * (
            3.0 * lame_lambda + 2.0 * shear_modulus
        ) / denominator
        poisson_ratio = lame_lambda / (2.0 * denominator)
    elif "E" in material and "nu" in material:
        elastic_modulus = material["E"]
        poisson_ratio = material["nu"]
        if elastic_modulus <= 0:
            raise ValueError("Young's modulus E must be positive.")
        if not -1.0 < poisson_ratio < 0.5:
            raise ValueError("Poisson ratio nu must satisfy -1 < nu < 0.5.")
        lame_lambda = (
            elastic_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )
        shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
    else:
        raise ValueError("Material parameters require either {'lam','mu'} or {'E','nu'}.")

    if shear_modulus <= 0:
        raise ValueError("The shear modulus mu must be positive.")
    return lame_lambda, shear_modulus, elastic_modulus, poisson_ratio


def _phase_tensor(phase, strain: TensorLike) -> TensorLike:
    """Convert scalar phase input to the active backend and strain context."""
    if bm.is_tensor(phase):
        return phase
    return bm.tensor(phase, **bm.context(strain))


def _normalize_model_name(model_type: str) -> str:
    """Normalize ``HybridModel``, ``hybrid-model``, and similar names."""
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("A non-empty phase-fracture model name is required.")
    key = "".join(character for character in model_type.lower() if character.isalnum())
    return key[:-5] if key.endswith("model") else key


def _parse_shear_order(shear_order, gd: int):
    """Translate optional 2-D/3-D symbolic shear names to index pairs."""
    if gd == 2:
        mapping = {"xy": (0, 1)}
    elif gd == 3:
        mapping = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}
    else:
        raise ValueError("Symbolic shear_order is supported only for GD=2 or GD=3.")
    try:
        return tuple(mapping[name] for name in shear_order)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid shear_order={shear_order} for GD={gd}.") from error


__all__ = [
    "BasedPhaseFractureMaterial",
    "IsotropicModel",
    "AnisotropicModel",
    "DeviatoricModel",
    "VolumetricDeviatoricModel",
    "SpectralModel",
    "HybridModel",
    "PhaseFractureMaterialFactory",
]
