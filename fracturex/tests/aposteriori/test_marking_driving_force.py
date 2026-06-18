"""M-DF 标记单元测试：无量纲驱动力 𝒟_τ 计算 + 阈值/尺寸下限标记逻辑。

对应 THEORY_marking_strategy.md §2(3)(4)、§6：
  𝒟_τ = (2 l0/G_c)·max_q H_{τ,q}，  临界 𝒟_c=1/3。
  M-DF 掩码：𝒟_τ ≥ θ_D=β/3  且  h_τ=√(2·area_τ) > l0/c_h。
机器精度对账（合成 H 场，不依赖求解器）。后端 numpy。
"""
from __future__ import annotations

import numpy as np
from fealpy.backend import backend_manager as bm

from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.adaptivity.adaptive_staggered import (
    driving_force_per_cell, mark_driving_force,
)


class _Mat:
    E, nu, Gc, l0, ft = 210.0, 0.3, 2.7e-3, 0.015, 3.0
    @property
    def mu(self): return self.E / (2.0 * (1.0 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _build(nx=4):
    mat = _Mat()
    case = SquareTensionPreCrackCase(_model=mat, nx=nx, ny=nx,
                                     crack_y=0.5, crack_length=0.5)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case=case, p=3, damage_p=1,
                                  use_relaxation=True).build(mesh=mesh)

    class _Dmg:  # 轻量替身：只需 Gc/l0
        Gc, l0 = mat.Gc, mat.l0
    return discr, _Dmg(), mat


def test_driving_force_formula():
    """𝒟_τ = (2 l0/G_c)·max_q H_{τ,q}，机器精度对账 + H=None 返回 0。"""
    bm.set_backend("numpy")
    discr, dmg, mat = _build(nx=4)
    NC = discr.mesh.number_of_cells()

    # H=None（未解）⇒ 全 0
    discr.state.H = None
    D0 = driving_force_per_cell(discr, dmg)
    assert D0.shape == (NC,) and float(bm.max(bm.abs(D0))) == 0.0

    # 合成 (NC,NQ) H 场：每元 NQ 点取已知值，逐元峰可控
    NQ = 6
    rng = np.random.default_rng(0)
    Hnp = rng.uniform(0.0, 5.0, size=(NC, NQ))
    discr.state.H = bm.asarray(Hnp)
    D = driving_force_per_cell(discr, dmg)
    expect = (2.0 * mat.l0 / mat.Gc) * Hnp.max(axis=1)
    err = float(bm.max(bm.abs(D - bm.asarray(expect))))
    assert err < 1e-13, f"𝒟 公式误差 {err:.2e}"
    print(f"[M-DF] 𝒟_τ 公式机器精度 OK  max err={err:.2e}  NC={NC}")


def test_driving_force_critical_value():
    """𝒟_c=1/3 标定：构造使某元历史峰恰对应 𝒟=1/3，验阈值判别。"""
    bm.set_backend("numpy")
    discr, dmg, mat = _build(nx=4)
    NC = discr.mesh.number_of_cells()
    # 令 H 使 𝒟=1/3 ⇒ H = (1/3)·G_c/(2 l0)
    H_crit = (1.0 / 3.0) * mat.Gc / (2.0 * mat.l0)
    Hnp = np.full((NC, 3), 0.0)
    Hnp[0, :] = H_crit              # 0 号元恰临界
    Hnp[1, :] = 0.99 * H_crit       # 1 号元略低于临界
    discr.state.H = bm.asarray(Hnp)
    D = driving_force_per_cell(discr, dmg)
    assert abs(float(D[0]) - 1.0 / 3.0) < 1e-12, f"𝒟[0]={float(D[0])}"
    assert float(D[1]) < 1.0 / 3.0
    print(f"[M-DF] 𝒟_c=1/3 标定 OK  𝒟[0]={float(D[0]):.6f} 𝒟[1]={float(D[1]):.6f}")


def test_mark_threshold_and_size_floor():
    """M-DF 掩码 = (𝒟≥β/3) ∧ (area>(l0/c_h)²/2)，与显式条件逐元对账。"""
    bm.set_backend("numpy")
    discr, dmg, mat = _build(nx=8)
    NC = discr.mesh.number_of_cells()
    cm = np.asarray(discr.mesh.entity_measure("cell"))

    beta, c_h, l0 = 0.6, 2.0, mat.l0
    theta_D = beta / 3.0
    area_floor = (l0 / c_h) ** 2 / 2.0

    rng = np.random.default_rng(1)
    Dnp = rng.uniform(0.0, 1.0, size=NC)          # 合成 𝒟
    marked = mark_driving_force(discr, bm.asarray(Dnp), l0=l0, beta=beta, c_h=c_h)
    expect = (Dnp >= theta_D) & (cm > area_floor)
    n_disagree = int(np.sum(np.asarray(marked) != expect))
    assert n_disagree == 0, f"{n_disagree} 元掩码不一致"
    print(f"[M-DF] 标记掩码逐元对账 OK  marked={int(np.sum(np.asarray(marked)))}/{NC} "
          f"(θ_D={theta_D:.3f}, area_floor={area_floor:.2e})")

    # 尺寸下限：把所有 𝒟 设大，仅 area>floor 的被标记
    big = bm.asarray(np.full(NC, 1.0))
    m2 = np.asarray(mark_driving_force(discr, big, l0=l0, beta=beta, c_h=c_h))
    assert np.all(m2 == (cm > area_floor)), "尺寸下限未生效"
    print(f"[M-DF] 尺寸下限 h≤l0/{c_h:.0f} 生效 OK  可加密 {int(np.sum(cm>area_floor))}/{NC}")


if __name__ == "__main__":
    test_driving_force_formula()
    test_driving_force_critical_value()
    test_mark_threshold_and_size_floor()
    print("\n[M-DF marking unit tests] ALL PASS")
