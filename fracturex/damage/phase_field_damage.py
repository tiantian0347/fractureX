# fracturex/damage/phase_field_damage.py
#TODO: implement phase-field damage model
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any

from fealpy.backend import backend_manager as bm
from .base import DamageModelBase, DamageStateView
from .phase_fracture_material import PhaseFractureMaterial  # 假设你文件里有这个类

@dataclass
class PhaseFieldDamage(DamageModelBase):
    name: str = "phase_field"
    eps_g: float = 1e-15
    material: Optional[Any] = None   # 你的分解材料对象（可注入）

    def on_build(self, discr, state: DamageStateView, case):
        if state.H is None:
            raise RuntimeError("PhaseFieldDamage requires state.H")
        state.H[:] = 0.0
        if self.material is None:
            # 用户可在外部传入不同分解模型
            self.material = PhaseFractureMaterial()

    def coef_bary(self, state: DamageStateView, bcs, index=None):
        d = state.d(bcs, index=index) if callable(state.d) else state.d
        return (1.0 - d)**2 + self.eps_g

    def update_after_elastic(self, discr, state: DamageStateView, case):
        if state.H is None:
            raise RuntimeError("PhaseFieldDamage requires state.H")
        mesh = discr.mesh
        node = mesh.entity("node")

        # 例：从 u 得到应变/从 sigma 得到应力，交给 material 产生 psi_plus
        # 这里先给一个最小版本：若你 material 接受 sigma，就用 sigma
        sig = state.sigma(node) if callable(state.sigma) else state.sigma
        lam = float(getattr(case.model(), "lam", getattr(case.model(), "lambda0", 0.0)))
        mu  = float(getattr(case.model(), "mu",  getattr(case.model(), "lambda1", 0.0)))

        psi_plus = self.material.tensile_energy_density(sig, lam=lam, mu=mu)
        state.H[:] = bm.maximum(state.H[:], psi_plus)

    def solve_damage(self, discr, state: DamageStateView, case, load: float):
        # TODO: 未来在这里装配并求解 d 方程
        return state.d
