# fracturex/tests/smoke_run_square_tension.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from fracturex.cases.square_tension import SquareTensionCase
from fracturex.cases.base import debug_isNedge_on_crack
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_damage_staggered import HuZhangLocalDamageStaggeredDriver
from fracturex.damage.base import DamageModelBase, DamageStateView

from fealpy.backend import backend_manager as bm

@dataclass
class SimpleMaterial:
    E: float = 10.0
    nu: float = 0.3
    ft: float = 0.01
    Gc: float = 1.0
    l0: float = 0.02

    @property
    def mu(self):
        return self.E/(2*(1+self.nu))

    @property
    def lam(self):
        return self.E*self.nu/((1+self.nu)*(1-2*self.nu))

    def characteristic_length(self):
        return self.Gc*self.E/(self.ft**2)

    def moderation_parameter(self):
        lch = self.characteristic_length()
        return self.l0/(2*lch + self.l0)


class NoDamage(DamageModelBase):
    name = "no_damage"
    def coef_bary(self, state: DamageStateView, bcs, index=None):
        return 1.0
    def update_after_elastic(self, discr, state: DamageStateView, case):
        return


def main():
    mat = SimpleMaterial(E=10.0, nu=0.3, ft=0.01, Gc=1.0, l0=0.02)
    

    #case = SquareTensionCase(nx=16, ny=16, _model=mat)

    case = SquareTensionCase(with_fracture=True, refine=0, _model=mat)

    mesh = case.make_mesh()


    discr = HuZhangDiscretization(case, p=3, use_relaxation=True).build()
    damage = NoDamage()

    driver = HuZhangLocalDamageStaggeredDriver(
        case=case, discr=discr, damage=damage,
        tol=1e-12, maxit=2, debug=True
    )

    loads = [0.0, 1e-3]
    hist = driver.run(loads)
    #debug_isNedge_on_crack(mesh, discr.isNedge, crack_pred=lambda bc: SquareTensionCase.crack_pred(bc, tol=case.tol))

    
    print("\n=== step summary ===")
    # for s in hist:
    #     print(s)

    # 若要测局部损伤，取消注释（你工程里要有 LocalNodeDamage）
    from fracturex.damage.local_node_damage import LocalNodeDamage
    damage2 = LocalNodeDamage(ft=mat.ft, Hd=mat.moderation_parameter(),
                             criterion="rankine", lam=mat.lam, mu=mat.mu)
    print("LocalNodeDamage(ft,Hd) =", float(damage2.ft), float(damage2.Hd))

    driver2 = HuZhangLocalDamageStaggeredDriver(case=case, discr=discr, damage=damage2, tol=1e-8, maxit=30)
    loads2 = np.linspace(0.0, 0.05, 11).tolist()  # 最高 0.05 => sigma~0.5
    hist2 = driver2.run(loads2)
    print("\n=== local damage summary ===")
    # for s in hist2:
    #     print(s)
    print("max d =", float(np.max(discr.state.d[:])))

    case = SquareTensionCase(
        with_fracture=True,
        refine=4,
        debug_mesh=True,
        _model=mat
    )

    p_sigma = 3
    use_relaxation = True

    # damage (local node)
    ft = 0.01
    Gc = 1.0
    l0 = 0.02

    loads = bm.linspace(0.0, 0.05, 21)  # 0~0.05, 20 steps

    driver_options = dict(maxit=10, tol=1e-10, debug=True)



if __name__ == "__main__":
    main()
