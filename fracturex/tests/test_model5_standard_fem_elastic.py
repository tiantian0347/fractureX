"""Standard FEM (MainSolve) elastic kinematics for Ambati three-point bending.

Hu–Zhang currently fails beam bending; this checks that the displacement-based
Lagrange path produces midspan bending and Euler-order stiffness.
"""
from __future__ import annotations

import numpy as np

from fracturex.cases.phase_field.model5_three_point_bending import (
    run_elastic_kinematics,
)


def test_standard_fem_tpb_bending_kinematics():
    info = run_elastic_kinematics(mesh_size=0.2, load=1e-3, p=1, use_box=True)
    print("[std-fem TPB]", info)
    assert info["uy_load"] < -0.5e-3, f"load uy too small: {info}"
    assert info["uy_bot"] < -0.2e-3, f"bottom midspan not bending: {info}"
    assert info["ratio"] > 0.2, f"bending ratio too small: {info}"
    # Euler ~1.3 kN/mm for this beam; allow a wide band for coarse P1 mesh.
    assert 0.2 < info["k"] < 20.0, f"stiffness out of band: {info}"


if __name__ == "__main__":
    test_standard_fem_tpb_bending_kinematics()
    print("standard FEM TPB kinematics OK")
