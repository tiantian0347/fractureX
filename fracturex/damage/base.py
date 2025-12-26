# fracturex/damage/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Callable

import numpy as np
from fealpy.backend import backend_manager as bm


class DecompositionMaterial(Protocol):
    """
    用于相场分解材料（你现在的 phase_fracture_material.py）对外的最小接口约定。
    你可以不严格继承，只要实现同名方法即可。

    典型职责：
    - 给定应变/应力，返回“张拉驱动能量密度”或“等效历史量”用于更新 H
    """
    def tensile_energy_density(self, strain_or_stress: Any, *, lam: float, mu: float) -> Any:
        ...


@dataclass
class DamageStateView:
    """
    Driver/Assembler 不应该直接知道 state 的具体类，
    这里用一个“视图”来约定必须有的字段。

    约定：
    - d: 标量函数（通常 P1 节点），可索引/可写
    - H: 历史场（相场用），可写
    - r_hist: 局部损伤用历史变量，可写
    - sigma/u: 供 damage 更新时评估使用（可选）
    """
    d: Any
    H: Optional[Any] = None
    r_hist: Optional[Any] = None
    sigma: Optional[Any] = None
    u: Optional[Any] = None



class DamageModelBase:
    """
    DamageModel 的统一接口。

    当前（局部损伤 staggered）必需：
    - coef_bary: 返回退化系数 g(d) 在单元积分点处的值（给 HuZhangStressIntegrator 用）
    - update_after_elastic: 已知 (sigma,u) 更新 d/r_hist/H

    未来（phase-field/全耦合）可扩展：
    - assemble_damage_equation / newton_linearize 等
    """
    name: str = "damage_base"

    def on_build(self, discr, state: DamageStateView, case) -> None:
        """离散建好后初始化（可选）。"""
        return

    def coef_bary(self, state: DamageStateView, bcs, index=None):
        """
        返回 g(d) 在 (cell, quad) 上的值。
        允许返回标量 1.0（常数），也允许返回 shape=(NC,NQ) 或 (1,NQ) 等可 broadcast 形式。
        """
        return 1.0

    def update_after_elastic(self, discr, state: DamageStateView, case) -> None:
        """staggered: 用当前 sigma/u 更新 d、r_hist（以及 H）。"""
        return


__all__ = ["DecompositionMaterial", "DamageStateView", "DamageModelBase"]