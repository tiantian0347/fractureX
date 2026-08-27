"""Finite-element integration tests for standard phase-fracture materials.

These smoke tests exercise the same ``stress_value`` and ``elastic_matrix``
entry points used by ``MainSolve``. They cover two/three dimensions and
Lagrange degrees one through three.
"""

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm, LinearElasticIntegrator
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TetrahedronMesh, TriangleMesh

from fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction
from fracturex.phasefield.main_solve import MainSolve
from fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory


MATERIAL = {"E": 210.0, "nu": 0.27, "Gc": 1.0, "l0": 0.1}
MODEL_NAMES = (
    "IsotropicModel",
    "AnisotropicModel",
    "DeviatoricModel",
    "SpectralModel",
    "HybridModel",
)


def _affine_displacement(points):
    """Return a nontrivial affine field in the point-cloud dimension."""
    gd = points.shape[-1]
    matrix = bm.eye(gd, **bm.context(points)) * 0.02
    if gd >= 2:
        matrix = bm.set_at(matrix, (0, 1), 0.006)
        matrix = bm.set_at(matrix, (1, 0), -0.002)
    if gd >= 3:
        matrix = bm.set_at(matrix, (1, 2), 0.004)
    return bm.einsum("...j,ij->...i", points, matrix)


def _constant_phase(points):
    """Return the spatially constant phase value ``d=0.23``."""
    return 0.23 * bm.ones_like(points[..., 0])


def _make_mesh(gd):
    """Build one simplex in the requested geometric dimension."""
    if gd == 2:
        return TriangleMesh.from_one_triangle(meshtype="iso")
    if gd == 3:
        return TetrahedronMesh.from_one_tetrahedron(meshtype="equ")
    raise ValueError(f"Unsupported smoke-test dimension {gd}.")


def _interpolate_fields(scalar_space, vector_space):
    """Interpolate affine displacement and constant phase without backend wrappers."""
    points = scalar_space.interpolation_points()
    displacement_values = _affine_displacement(points)
    if vector_space.dof_priority:
        displacement_values = bm.swapaxes(displacement_values, -1, -2)
    displacement = vector_space.function(displacement_values.reshape(-1))
    phase = scalar_space.function(_constant_phase(points))
    return displacement, phase


@pytest.mark.parametrize("backend", ("numpy", "pytorch", "jax"))
def test_standard_material_fe_values_are_backend_consistent(backend):
    """All models evaluate finite 2-D quadratic FE coefficients on each backend."""
    bm.set_backend(backend)
    mesh = _make_mesh(2)
    scalar_space = LagrangeFESpace(mesh, p=2)
    vector_space = TensorFunctionSpace(scalar_space, shape=(2, -1))
    displacement, phase = _interpolate_fields(scalar_space, vector_space)
    quadrature = mesh.quadrature_formula(4, "cell")
    bcs, _ = quadrature.get_quadrature_points_and_weights()

    for model_name in MODEL_NAMES:
        material = PhaseFractureMaterialFactory.create(
            model_name, MATERIAL, EnergyDegradationFunction()
        )
        material.update_disp(displacement)
        material.update_phase(phase)

        strain = material.strain_value(bcs)
        stress = material.stress_value(bcs)
        tangent = material.elastic_matrix(bcs)
        history = material.maximum_historical_field(bcs)

        assert strain.shape[-2:] == (2, 2)
        assert stress.shape == strain.shape
        assert tangent.shape == strain.shape[:-2] + (3, 3)
        assert history.shape == strain.shape[:-2]
        assert np.all(np.isfinite(bm.to_numpy(stress)))
        assert np.all(np.isfinite(bm.to_numpy(tangent)))
        assert np.all(bm.to_numpy(history) >= -1.0e-14)


@pytest.mark.parametrize("gd,p", ((2, 1), (2, 3), (3, 2)))
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_standard_voigt_assembly_supports_dimension_and_degree(gd, p, model_name):
    """The standard FE material path assembles for 2-D/3-D and p=1/2/3."""
    bm.set_backend("numpy")
    mesh = _make_mesh(gd)
    scalar_space = LagrangeFESpace(mesh, p=p)
    vector_space = TensorFunctionSpace(scalar_space, shape=(gd, -1))
    displacement, phase = _interpolate_fields(scalar_space, vector_space)
    material = PhaseFractureMaterialFactory.create(
        model_name, MATERIAL, EnergyDegradationFunction()
    )
    material.update_disp(displacement)
    material.update_phase(phase)

    form = BilinearForm(vector_space)
    form.add_integrator(
        LinearElasticIntegrator(material, q=p + 2, method="voigt")
    )
    matrix = form.assembly()

    gdof = vector_space.number_of_global_dofs()
    assert matrix.shape == (gdof, gdof)
    assert np.all(np.isfinite(matrix.to_scipy().data))


def test_material_strain_matrix_is_dimension_generic():
    """The material strain operator also supports dimensions beyond FE mesh classes."""
    bm.set_backend("numpy")
    material = PhaseFractureMaterialFactory.create(
        "SpectralModel", MATERIAL, EnergyDegradationFunction()
    )
    gphi = bm.tensor(np.arange(10, dtype=np.float64).reshape(1, 1, 2, 5))
    strain_matrix = material.strain_matrix(dof_priority=True, gphi=gphi)
    assert strain_matrix.shape == (1, 1, 15, 10)
    assert np.all(np.isfinite(bm.to_numpy(strain_matrix)))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_standard_staggered_solver_smoke_for_every_split(model_name):
    """One standard ``MainSolve`` step integrates every material model end to end."""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=1, ny=1)
    solver = MainSolve(mesh, MATERIAL, model_type=model_name)
    solver.add_boundary_condition(
        "force",
        "Dirichlet",
        lambda points: bm.abs(points[..., 1] - 1.0) < 1.0e-12,
        [0.0, 1.0e-5],
        "y",
    )
    solver.add_boundary_condition(
        "displacement",
        "Dirichlet",
        lambda points: bm.abs(points[..., 1]) < 1.0e-12,
        0.0,
    )
    solver.solve(p=1, q=3, maxit=1, linear_solver_options={"method": "direct"})

    assert solver.H.shape == (mesh.number_of_cells(), 6)
    assert np.all(np.isfinite(bm.to_numpy(solver.H)))
    assert np.all(np.isfinite(bm.to_numpy(solver.d)))
