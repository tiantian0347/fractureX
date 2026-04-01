# fracturex/tests/smoke_run_square_tension.py
from __future__ import annotations
import matplotlib.pyplot as plt

import numpy as np
from dataclasses import dataclass

from fracturex.cases.square_tension import SquareTensionCase
from fracturex.cases.base import debug_isNedge_on_crack
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_damage_staggered import HuZhangLocalDamageStaggeredDriver
from fracturex.damage.base import DamageModelBase, DamageStateView
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.damage.local_node_damage import LocalNodeDamage

from fealpy.backend import backend_manager as bm

@dataclass
class SimpleMaterial:
    E: float = 210 # Young's modulus in GPa as KN/mm²
    nu: float = 0.3 # Poisson's ratio
    Gc: float = 2.7e-3 # fracture energy in KN/mm
    l0: float = 0.07 # length scale parameter in mm. 0.015 for phase field

    @property
    def mu(self):
        return self.E/(2*(1+self.nu))

    @property
    def lam(self):
        return self.E*self.nu/((1+self.nu)*(1-2*self.nu))
    
    @property
    def sigma_c(self):
        return 9/16*np.sqrt(self.E*self.Gc/(6*self.l0))
    
    @property
    def ft(self):
        ft = np.sqrt(self.E*self.Gc/(self.l0))
        return ft
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
    mat = SimpleMaterial()
    print(f"Material sigma_c: {mat.sigma_c:.6f} KN/mm²")
    print(f"Material ft: {mat.ft:.6f} KN/mm²")
    print(f"Material characteristic length: {mat.characteristic_length():.6f} mm")
    print(f"Material moderation parameter Hd: {mat.moderation_parameter():.6f}")


    damage = LocalNodeDamage(ft=mat.ft, Hd=mat.moderation_parameter(),
                        criterion="rankine", lam=mat.lam, mu=mat.mu)
    
    #damage = NoDamage()

    case = SquareTensionCase(with_fracture=True, refine=5, _model=damage)
    #case = SquareTensionCase(with_fracture=False, refine=3, _model=damage)

    discr = HuZhangDiscretization(case, p=3, use_relaxation=True).build()

    # 构造装配器
    assembler = HuZhangElasticAssembler(discr, case, damage=damage)
    

    # 创建驱动程序
    driver = HuZhangLocalDamageStaggeredDriver(
        case=case, 
        discr=discr, 
        damage=damage, 
        assembler=assembler, 
        tol=1e-8, 
        maxit=50, 
        debug=True
    )
    #loads = bm.linspace(0.0, 6e-3, 11)
    loads = bm.concatenate((bm.linspace(0, 5e-3, 51, dtype=bm.float64), bm.linspace(5e-3, 6.1e-3, 1101, dtype=bm.float64)[1:]))
    
    loads_x = bm.linspace(0, 2.2e-2, 2201, dtype=bm.float64)
    # 运行仿真
    result = driver.run(loads)

    # 结果输出到VTK
    driver._safeve_vtkfile("simulation_output.vtu", cell_mode="mean")

    # 可视化损伤演化
    damages = [step_info.meta["max_d"] for step_info in result]

    plt.plot(loads, damages, label="Damage evolution")
    plt.xlabel("Load (N)")
    plt.ylabel("Damage")
    plt.title("Damage vs Load")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
