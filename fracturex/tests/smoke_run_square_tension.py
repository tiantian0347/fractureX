# fracturex/tests/smoke_run_square_tension.py
"""Quick smoke: Hu-Zhang + local nodal damage, square y-tension with phase-field pre-crack."""
from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from fealpy.backend import backend_manager as bm

from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_damage_staggered import HuZhangLocalDamageStaggeredDriver
from fracturex.damage.base import DamageModelBase, DamageStateView
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.damage.local_node_damage import LocalNodeDamage


@dataclass
class SimpleMaterial:
    E: float = 210
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 0.07

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def sigma_c(self):
        return 9 / 16 * np.sqrt(self.E * self.Gc / (6 * self.l0))

    @property
    def ft(self):
        return np.sqrt(self.E * self.Gc / self.l0)

    def characteristic_length(self):
        return self.Gc * self.E / (self.ft**2)

    def moderation_parameter(self):
        lch = self.characteristic_length()
        return self.l0 / (2 * lch + self.l0)


class NoDamage(DamageModelBase):
    name = "no_damage"

    def coef_bary(self, state: DamageStateView, bcs, index=None):
        return 1.0

    def update_after_elastic(self, discr, state: DamageStateView, case):
        return


def _apply_precrack_damage(case, discr, *, load: float = 0.0, value: float = 1.0) -> int:
    """One-shot d=value on ``phasefield_initial_damage_data`` mask (shared with phase-field tests)."""
    if not hasattr(case, "phasefield_initial_damage_data"):
        return 0
    bcdata = case.phasefield_initial_damage_data(load)
    if bcdata is None:
        return 0
    if isinstance(bcdata, dict):
        bcdata = [bcdata]

    space = discr.space_d
    state = discr.state
    ip = space.interpolation_points()
    darr = bm.asarray(state.d[:]).copy()
    n_set = 0
    for item in bcdata:
        thr = item["bcdof"]
        val = item.get("value", value)
        mask = bm.asarray(thr(ip)).astype(bm.bool)
        idx = bm.where(mask)[0]
        if len(idx) == 0:
            continue
        v = val(ip[idx]) if callable(val) else val
        darr = bm.set_at(darr, idx, v)
        n_set += int(len(idx))
    state.d[:] = darr
    return n_set


def main():
    mat = SimpleMaterial()
    print(f"Material sigma_c: {mat.sigma_c:.6f} KN/mm²")
    print(f"Material ft: {mat.ft:.6f} KN/mm²")
    print(f"Material characteristic length: {mat.characteristic_length():.6f} mm")
    print(f"Material moderation parameter Hd: {mat.moderation_parameter():.6f}")

    damage = LocalNodeDamage(
        ft=mat.ft,
        Hd=mat.moderation_parameter(),
        criterion="rankine",
        lam=mat.lam,
        mu=mat.mu,
    )

    nx = int(os.environ.get("FRACTUREX_NX", "16"))
    ny = int(os.environ.get("FRACTUREX_NY", str(nx)))
    case = SquareTensionPreCrackCase(
        _model=damage,
        nx=nx,
        ny=ny,
        crack_y=0.5,
        crack_length=0.5,
        debug_mesh=False,
    )
    mesh = case.make_mesh()
    print(
        f"\nmesh: intact from_box {nx}×{ny}, NC={mesh.number_of_cells()}, "
        f"no geometric notch"
    )
    print(f"pre-crack: y={case.crack_y}, x∈[0,{case.crack_length}]")

    discr = HuZhangDiscretization(case, p=3, use_relaxation=True).build(mesh=mesh)
    n_precrack = _apply_precrack_damage(case, discr)
    print(f"initial damage DOFs on pre-crack line: {n_precrack}")

    assembler = HuZhangElasticAssembler(discr, case, damage=damage)

    driver = HuZhangLocalDamageStaggeredDriver(
        case=case,
        discr=discr,
        damage=damage,
        assembler=assembler,
        tol=1e-8,
        maxit=50,
        debug=True,
    )

    loads = np.concatenate(
        [
            np.linspace(0.0, 5e-3, 51, dtype=float),
            np.linspace(5e-3, 6.1e-3, 111, dtype=float)[1:],
        ]
    )
    if os.environ.get("FRACTUREX_RUN_SHORT", "0") == "1":
        loads = loads[:3]
        print(f"FRACTUREX_RUN_SHORT=1: {len(loads)} load steps")

    result = driver.run(loads.tolist())

    from fracturex.postprocess.run_paths import phasefield_tag_dir, vtk_dir

    tag_dir = phasefield_tag_dir(case.name, "smoke_damage", eps_g=1e-6, mkdir=True)
    vtk_path = os.path.join(vtk_dir(tag_dir), "final.vtu")
    driver._save_vtkfile(vtk_path, cell_mode="mean")
    print(f"VTK saved to {vtk_path}")

    damages = [step_info.meta.get("max_d", step_info.max_d) for step_info in result]
    plt.plot(loads[: len(damages)], damages, label="Damage evolution")
    plt.xlabel("Load (mm)")
    plt.ylabel("max(d)")
    plt.title("Damage vs Load")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
