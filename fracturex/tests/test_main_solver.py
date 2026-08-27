"""Smoke tests for the current standard phase-field staggered solver API.

The test intentionally runs one load step only. It verifies solver setup,
boundary-condition registration, material integration, and phase-field update
without starting a long fracture simulation.
"""

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.phasefield.main_solve import MainSolve


def _top_boundary(points):
    """Select the top edge for prescribed vertical displacement."""
    return bm.abs(points[..., 1] - 1.0) < 1.0e-12


def _bottom_boundary(points):
    """Select the bottom edge for the zero-displacement constraint."""
    return bm.abs(points[..., 1]) < 1.0e-12


@pytest.mark.parametrize("backend", ("numpy", "pytorch"))
def test_main_solve_one_load_step(backend):
    """Run one HybridModel load step on each standard CPU backend."""
    bm.set_backend(backend)
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=1, ny=1)
    material = {"E": 200.0, "nu": 0.3, "Gc": 1.0, "l0": 0.1}
    solver = MainSolve(mesh=mesh, material_params=material, model_type="HybridModel")

    solver.add_boundary_condition(
        "force", "Dirichlet", _top_boundary, [0.0, 1.0e-5], "y"
    )
    solver.add_boundary_condition(
        "displacement", "Dirichlet", _bottom_boundary, 0.0
    )
    solver.solve(
        method="lfem",
        p=1,
        q=3,
        maxit=1,
        linear_solver_options={"method": "direct"},
    )

    assert solver.H.shape == (mesh.number_of_cells(), 6)
    assert np.all(np.isfinite(bm.to_numpy(solver.H)))
    assert np.all(np.isfinite(bm.to_numpy(solver.d)))

