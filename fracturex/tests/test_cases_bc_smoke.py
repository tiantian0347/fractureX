"""Smoke: CaseBase mesh + BC masks + short Hu–Zhang staggered run.

Focus: boundary selectors are non-empty / consistent, and a few load steps
converge with finite reaction (proves Dirichlet pieces are actually applied).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.boundarycondition.huzhang_boundary_condition import (
    build_isNedge_from_isD,
)
from fracturex.cases.model2_notch_shear import Model2NotchXStretchCase
from fracturex.cases.model3_lshape import Model3LShapeCase
from fracturex.cases.model4_notched_plate_with_hole import (
    Model4NotchedPlateWithHoleCase,
)
from fracturex.cases.model5_three_point_bending import Model5ThreePointBendingCase
from fracturex.cases.model6_asymmetric_notched_beam import (
    Model6AsymmetricNotchedBeamCase,
)
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)
from fracturex.postprocess.recorder import RunRecorder


@dataclass
class _Mat:
    lam: float
    mu: float
    Gc: float
    l0: float
    ft: float = 3.0


def _bd_barycenters(mesh):
    is_bd = mesh.boundary_edge_flag()
    eids = bm.where(is_bd)[0]
    bc = mesh.entity_barycenter("edge", index=eids)
    return eids, bc


def _count(mask) -> int:
    return int(np.sum(np.asarray(mask, dtype=bool)))


def _assert_bc_masks(case, mesh, *, checks: List[tuple]):
    """checks: list of (name, callable(bc)->bool, min_count)."""
    _, bc = _bd_barycenters(mesh)
    isD = case.isD_bd(bc)
    nD = _count(isD)
    assert nD > 0, f"{case.name}: isD_bd empty on boundary edges"

    # Neumann complement should exist for nontrivial free boundary
    isN = build_isNedge_from_isD(mesh, case.isD_bd)
    nN = _count(isN)
    assert nN > 0, f"{case.name}: isNedge empty (all-Dirichlet? unexpected)"

    reported = {"isD": nD, "isN": nN}
    for name, fn, nmin in checks:
        n = _count(fn(bc))
        reported[name] = n
        assert n >= nmin, (
            f"{case.name}: BC '{name}' matched {n} edges, need >={nmin}"
        )

    # Dirichlet pieces must include a load tag and fix tag(s)
    pcs = case.dirichlet_pieces(1e-3)
    tags = [getattr(p, "tag", None) for p in pcs]
    assert any(t == "load" for t in tags), f"{case.name}: missing tag='load'"
    assert any(t != "load" for t in tags), f"{case.name}: missing fix piece"

    # load piece threshold must hit ≥1 boundary edge
    lp = case.load_dirichlet_piece(1e-3)
    n_load = _count(lp.threshold(bc))
    assert n_load >= 1, f"{case.name}: load threshold empty (n={n_load})"

    # prescribed value at load edges has the expected component
    vals = np.asarray(lp.value(bc[np.asarray(lp.threshold(bc), dtype=bool)]))
    assert vals.ndim == 2 and vals.shape[1] >= 2
    direction = (lp.direction or "y").lower()
    comp = 0 if direction == "x" else 1
    assert np.allclose(vals[:, comp], 1e-3) or np.allclose(
        vals[:, comp], -1e-3
    ), (
        f"{case.name}: load Dirichlet value component {comp} not ±1e-3, "
        f"got {vals[:, comp]}"
    )

    return reported


def _run_short(case, *, outdir: Path, loads, p_sigma=3, damage_p=2):
    outdir.mkdir(parents=True, exist_ok=True)
    discr = HuZhangDiscretization(
        case=case, p=p_sigma, damage_p=damage_p, use_relaxation=True
    ).build()
    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=1e-6,
        debug=False,
    )
    elastic = HuZhangElasticAssembler(
        discr, case, damage, formulation="standard"
    )
    phase = PhaseFieldAssembler(discr, case, damage, debug=False)
    recorder = RunRecorder(
        str(outdir), save_npz=True, save_every=1, save_quadrature_fields=False
    )
    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case,
        discr=discr,
        damage=damage,
        elastic_assembler=elastic,
        phase_assembler=phase,
        tol=1e-5,
        maxit=30,
        elastic_solver=HuZhangPhaseFieldStaggeredDriver._default_spsolve,
        phase_solver=HuZhangPhaseFieldStaggeredDriver._default_lgmres,
        compute_linear_residual=False,
        debug=False,
        timing=False,
        recorder=recorder,
        output_dir=str(outdir),
        save_vtu_per_step=False,
    )
    infos = driver.run(np.asarray(loads, dtype=float))
    return discr, infos


def _check_solve_ok(case_name: str, infos, *, expect_reaction_grows: bool = True):
    assert len(infos) >= 2
    for info in infos:
        assert info.converged, f"{case_name}: step {info.step} not converged"
        R = float(info.meta.get("residual_force", info.meta.get("R", np.nan)))
        assert np.isfinite(R), f"{case_name}: non-finite reaction {R}"

    # nonzero load → nonzero reaction in linear regime
    R0 = float(infos[0].meta.get("residual_force", infos[0].meta.get("R")))
    R1 = float(infos[1].meta.get("residual_force", infos[1].meta.get("R")))
    assert abs(R1) > 1e-14, (
        f"{case_name}: reaction at first nonzero load is ~0 ({R1}); "
        "load BC likely not applied"
    )
    if expect_reaction_grows and abs(float(infos[1].load)) > abs(float(infos[0].load)):
        # |R| should increase roughly with |load| while still elastic
        assert abs(R1) > abs(R0) * 0.5 or abs(R0) < 1e-16


# ---------------------------------------------------------------------------
# per-case BC + short solve
# ---------------------------------------------------------------------------


def test_model2_bc_and_smoke(tmp_path):
    case = Model2NotchXStretchCase(
        _model=_Mat(121.15, 80.77, 2.7e-3, 0.015), nx=8, ny=8
    )
    mesh = case.make_mesh()
    rep = _assert_bc_masks(
        case,
        mesh,
        checks=[
            ("bottom_fix", case._on_y0, 1),
            ("top_load", case._on_y1, 1),
        ],
    )
    print("[model2 BC]", rep)
    # precrack DOFs
    space = LagrangeFESpace(mesh, p=1)
    n_pc = _count(case._on_precrack(np.asarray(space.interpolation_points())))
    assert n_pc > 0, "model2 precrack empty"
    _, infos = _run_short(
        case, outdir=tmp_path / "m2", loads=[0.0, 1e-5, 2e-5]
    )
    _check_solve_ok("model2", infos)


def test_model3_bc_and_smoke(tmp_path):
    case = Model3LShapeCase(
        _model=_Mat(6.16, 10.95, 8.9e-5, 1.1875), nx=16, ny=16
    )
    mesh = case.make_mesh()
    rep = _assert_bc_masks(
        case,
        mesh,
        checks=[
            ("bottom_fix", case._on_bottom, 1),
            ("load_point", case._on_load, 1),
        ],
    )
    print("[model3 BC]", rep)
    _, infos = _run_short(
        case, outdir=tmp_path / "m3", loads=[0.0, 5e-4, 1e-3]
    )
    _check_solve_ok("model3", infos)


def test_model4_bc_and_smoke(tmp_path):
    case = Model4NotchedPlateWithHoleCase(
        _model=_Mat(1.94, 2.45, 2.28e-3, 0.1),
        mesh_size=4.0,
        with_geometric_notch=False,
        debug_mesh=False,
    )
    mesh = case.make_mesh()
    rep = _assert_bc_masks(
        case,
        mesh,
        checks=[
            ("lower_pin", case._on_lower_pin, 1),
            ("upper_pin", case._on_upper_pin, 1),
        ],
    )
    # pins disjoint
    _, bc = _bd_barycenters(mesh)
    both = case._on_lower_pin(bc) & case._on_upper_pin(bc)
    assert _count(both) == 0
    # precrack DOFs on embedded line
    space = LagrangeFESpace(mesh, p=2)
    n_pc = _count(case._on_precrack(np.asarray(space.interpolation_points())))
    assert n_pc > 0, f"model4 precrack empty (P2 DOFs={n_pc})"
    print("[model4 BC]", rep, f"precrack_P2={n_pc}")
    _, infos = _run_short(
        case, outdir=tmp_path / "m4", loads=[0.0, 1e-3, 2e-3]
    )
    _check_solve_ok("model4", infos)


def test_model5_bc_and_smoke(tmp_path):
    case = Model5ThreePointBendingCase(
        _model=_Mat(12.0, 8.0, 5.4e-4, 0.03),
        mesh_size=0.35,
        with_geometric_notch=False,
    )
    mesh = case.make_mesh()
    rep = _assert_bc_masks(
        case,
        mesh,
        checks=[
            ("left_support", case._on_left_support, 1),
            ("right_support", case._on_right_support, 1),
            ("top_load", case._on_load, 1),
        ],
    )
    space = LagrangeFESpace(mesh, p=2)
    n_pc = _count(case._on_precrack(np.asarray(space.interpolation_points())))
    assert n_pc > 0, "model5 precrack empty"
    print("[model5 BC]", rep, f"precrack_P2={n_pc}")
    # load Dirichlet is downward: u_y = -load
    lp = case.load_dirichlet_piece(1e-3)
    _, bc = _bd_barycenters(mesh)
    mask = np.asarray(lp.threshold(bc), dtype=bool)
    vals = np.asarray(lp.value(bc[mask]))
    assert np.allclose(vals[:, 1], -1e-3), vals[:, 1]
    _, infos = _run_short(
        case, outdir=tmp_path / "m5", loads=[0.0, 5e-4, 1e-3]
    )
    _check_solve_ok("model5", infos)


def test_model6_bc_and_smoke(tmp_path):
    case = Model6AsymmetricNotchedBeamCase(
        _model=_Mat(12.0, 8.0, 1e-3, 0.01),
        mesh_size=0.7,
        with_geometric_notch=False,
    )
    mesh = case.make_mesh()
    rep = _assert_bc_masks(
        case,
        mesh,
        checks=[
            ("left_support", case._on_left_support, 1),
            ("right_support", case._on_right_support, 1),
            ("top_load", case._on_load, 1),
        ],
    )
    space = LagrangeFESpace(mesh, p=2)
    n_pc = _count(case._on_precrack(np.asarray(space.interpolation_points())))
    assert n_pc > 0, "model6 precrack empty"
    print("[model6 BC]", rep, f"precrack_P2={n_pc}")
    _, infos = _run_short(
        case, outdir=tmp_path / "m6", loads=[0.0, 5e-4, 1e-3]
    )
    _check_solve_ok("model6", infos)


def test_model4_geometric_notch_bc_only():
    """FEM/IPFEM path: geometric notch mesh still captures pin BCs."""
    case = Model4NotchedPlateWithHoleCase(
        _model=_Mat(1.94, 2.45, 2.28e-3, 0.1),
        mesh_size=4.0,
        with_geometric_notch=True,
    )
    mesh = case.make_mesh()
    rep = _assert_bc_masks(
        case,
        mesh,
        checks=[
            ("lower_pin", case._on_lower_pin, 1),
            ("upper_pin", case._on_upper_pin, 1),
        ],
    )
    assert case.phasefield_initial_damage_data(0.0) is None
    print("[model4 geometric BC]", rep)


if __name__ == "__main__":
    # Allow: PYTHONPATH=../fealpy:. python fracturex/tests/test_cases_bc_smoke.py
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="fx_bc_smoke_"))
    print("outdir", root)
    test_model2_bc_and_smoke(root)
    test_model3_bc_and_smoke(root)
    test_model4_bc_and_smoke(root)
    test_model5_bc_and_smoke(root)
    test_model6_bc_and_smoke(root)
    test_model4_geometric_notch_bc_only()
    print("ALL BC SMOKE PASSED")
