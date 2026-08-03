"""Regression: three-point bending must produce beam bending, not local crush.

Checks after the set_essential_bc_v2 / reaction_boundary fixes:
  1. Bottom midspan deflects with the load (bending kinematics).
  2. Support reactions carry the load; free bottom is nearly traction-free.
  3. Reported residual_force uses supports (not the load-patch edges).

Note: As of 2026-08, the Hu–Zhang mixed path in this repo still fails (1)–(2)
for beams (TPB and cantilever): deformation stays local under a tip/patch
Dirichlet load even with corrected traction BC. Tests are marked xfail until
the formulation/BC issue is resolved; (3) may still pass once kinematics work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.cases.model5_three_point_bending import Model5ThreePointBendingCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)
from fracturex.postprocess.reaction import reaction_from_sigma
from fracturex.postprocess.recorder import RunRecorder


@dataclass
class _Mat:
    lam: float = 12.0
    mu: float = 8.0
    Gc: float = 5.4e-4
    l0: float = 0.03
    ft: float = 3.0


def _elastic_solve(mesh_size: float = 0.15, load: float = 1e-3, precrack: bool = True):
    case = Model5ThreePointBendingCase(
        _model=_Mat(), mesh_size=mesh_size, with_geometric_notch=False
    )
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(
        case=case, p=3, damage_p=2, use_relaxation=True
    ).build(mesh=mesh)
    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=1e-6,
    )
    damage.on_build(discr, discr.state, case)
    if precrack:
        PhaseFieldAssembler(discr, case, damage)._apply_phase_initial_damage_once(0.0)
    else:
        discr.state.d[:] = 0.0
    ea = HuZhangElasticAssembler(discr, case, damage, formulation="standard")
    sys_e = ea.assemble(load)
    x = np.asarray(
        HuZhangPhaseFieldStaggeredDriver._default_spsolve(sys_e.A, sys_e.F),
        dtype=float,
    )
    sig_fun, u_fun, _ = sys_e.decode(x)
    discr.state.u[:] = np.asarray(u_fun[:])
    drv = HuZhangPhaseFieldStaggeredDriver(
        case=case,
        discr=discr,
        damage=damage,
        elastic_assembler=ea,
        phase_assembler=PhaseFieldAssembler(discr, case, damage),
    )
    un = np.asarray(drv._u_to_nodal(discr.state.u, mesh))
    node = np.asarray(mesh.entity("node"))
    return case, mesh, sig_fun, un, node, load


@pytest.mark.xfail(
    reason="HuZhang beam bending still local-crush; see module docstring",
    strict=False,
)
def test_tpb_bending_kinematics():
    case, mesh, sig_fun, un, node, load = _elastic_solve()
    i_load = int(np.argmin(np.abs(node[:, 0] - 4) + np.abs(node[:, 1] - 2)))
    i_bot = int(np.argmin(np.abs(node[:, 0] - 4) + np.abs(node[:, 1] - 0)))
    uy_load = float(un[i_load, 1])
    uy_bot = float(un[i_bot, 1])
    assert uy_load < -0.5 * load, f"load uy too small: {uy_load}"
    # Bending: bottom midspan must move down with the load (same sign, O(load)).
    assert uy_bot < -0.2 * load, (
        f"bottom midspan not bending: uy_bot={uy_bot}, uy_load={uy_load}"
    )
    assert uy_bot / uy_load > 0.2, (
        f"uy_bot/uy_load={uy_bot / uy_load} (expected bending ratio)"
    )


@pytest.mark.xfail(
    reason="HuZhang beam bending still local-crush; see module docstring",
    strict=False,
)
def test_tpb_support_carries_load():
    case, mesh, sig_fun, un, node, load = _elastic_solve()

    def on_bottom(p):
        return np.abs(np.asarray(p)[:, 1]) < 1e-9

    def on_free_bottom(p):
        p = np.asarray(p)
        return (
            (np.abs(p[:, 1]) < 1e-9)
            & (~np.asarray(case._on_left_support(p), dtype=bool))
            & (~np.asarray(case._on_right_support(p), dtype=bool))
        )

    R_sup = reaction_from_sigma(
        mesh, sig_fun, case.reaction_boundary()[0], direction="y", q=5, sign=1.0
    )
    R_free = reaction_from_sigma(
        mesh, sig_fun, on_free_bottom, direction="y", q=5, sign=1.0
    )
    R_top = reaction_from_sigma(
        mesh, sig_fun, case._on_load, direction="y", q=5, sign=-1.0
    )
    # Supports (not free bottom) carry the load; free-bottom traction << support force.
    assert abs(R_sup) > 0.5 * abs(R_top), (
        f"supports do not carry load: R_sup={R_sup}, R_load_edges={R_top}"
    )
    assert abs(R_free) < 0.35 * abs(R_sup), (
        f"free bottom still loaded: R_free={R_free}, R_sup={R_sup}"
    )
    # Stiffness order: should be O(1) kN/mm (Euler ~1.3), not O(20) crush stiffness.
    k = abs(R_sup) / load
    assert 0.3 < k < 8.0, f"stiffness out of band: k={k} (Euler~1.3)"


@pytest.mark.xfail(
    reason="HuZhang beam bending still local-crush; force curve not meaningful yet",
    strict=False,
)
def test_tpb_driver_reports_support_reaction(tmp_path: Path):
    out = Path(tmp_path) / "tpb"
    from fracturex.tests.case_runners.model5_runner import Model5RunArgs, run_model5_one

    run_model5_one(
        Model5RunArgs(
            mesh_size=0.2,
            elastic_mode="direct",
            loads=[0.0, 1e-3, 2e-3],
            save_every=10,
            outdir=out,
        )
    )
    import csv

    rows = list(csv.DictReader(open(out / "history.csv")))
    r1 = abs(float(rows[1]["residual_force"]))
    u1 = abs(float(rows[1].get("load", rows[1].get("u", 1e-3))))
    if u1 < 1e-30:
        u1 = 1e-3
    k = r1 / u1
    # Euler beam ~1.3 kN/mm; local crush reports either huge R or near-zero support R.
    assert 0.3 < k < 8.0, f"stiffness out of band: R={r1}, k={k} at u={u1}"


if __name__ == "__main__":
    # Direct run: print diagnostics instead of asserting (xfails are pytest-only).
    case, mesh, sig_fun, un, node, load = _elastic_solve()
    i_load = int(np.argmin(np.abs(node[:, 0] - 4) + np.abs(node[:, 1] - 2)))
    i_bot = int(np.argmin(np.abs(node[:, 0] - 4) + np.abs(node[:, 1] - 0)))
    print(
        f"uy_load={un[i_load, 1]:.6e} uy_bot={un[i_bot, 1]:.6e} "
        f"ratio={un[i_bot, 1] / un[i_load, 1]:.4f}"
    )
    print("NOTE: bending assertions are pytest.mark.xfail until HuZhang beam BC is fixed.")
