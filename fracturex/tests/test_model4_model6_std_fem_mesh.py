"""Mesh + pin/support node counts for model4/model6 standard FEM drivers.

Does not run Newton steps. Requires gmsh (same as the CaseBase mesh builders).
"""
from __future__ import annotations

from fracturex.cases.phase_field.model4_notched_plate import (
    Model4StandardFEM,
    count_bc_nodes as count_m4,
)
from fracturex.cases.phase_field.model6_asymmetric_beam import (
    Model6StandardFEM,
    count_bc_nodes as count_m6,
)


def test_model4_std_fem_pin_nodes():
    model = Model4StandardFEM(mesh_size=4.0)
    info = count_m4(model, model.build_mesh())
    assert info["n_lower"] >= 3, info
    assert info["n_upper"] >= 3, info
    assert info["NC"] > 20, info


def test_model6_std_fem_support_and_load_nodes():
    model = Model6StandardFEM(mesh_size=0.8)
    info = count_m6(model, model.build_mesh())
    assert info["n_left"] >= 2, info
    assert info["n_right"] >= 2, info
    assert info["n_load"] >= 2, info
    assert info["NC"] > 20, info
