"""M3 full (最小集成): 真实 model1 staggered 演化中提取 (σ_h,d,u)，验 η_T 标记真实裂纹。

与 M3a（冻结解析带）的区别：本测试跑**真实相场 staggered 求解器**（HuZhangPhaseFieldStaggeredDriver），
通过 driver 的 adapt_hook 在每个载荷步拿到**真实演化**的：
  discr.sigma : Hu–Zhang 平衡应力 σ_h（生产退化混合解，正是估计子要的对象）
  discr.d     : 真实相场 d（演化裂纹）
  discr.u     : 混合位移（DG）
喂给 equilibrated_indicator 算 η_T，验证标记集中在真实裂纹带 y≈0.5。

这坐实：估计子能直接消费生产相场求解器的输出（σ_h 无需额外求解），
且在真实演化裂纹上正确定位——M3 的核心可行性。

注：driver 的 u 是 DG 混合位移（非 H¹ 连续），用于 η_T 的指示子计算（定位）足够；
严格可靠性界需连续 v（primal 重解），那是 M3 full 完整版的事。
model1 = 方形 y 拉伸 + 相场预裂纹（y=0.5, x∈[0,0.5]），无几何缺口（避开半边网格/Hu–Zhang 角点问题）。
运行较重（真实 staggered），默认 nx=16 + 少数载荷步。
"""
from __future__ import annotations

import os
import numpy as np

from fealpy.backend import backend_manager as bm
from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver

from fracturex.adaptivity.equilibrated_estimator import equilibrated_indicator


class _Mat:
    E, nu, Gc, l0, ft = 210.0, 0.3, 2.7e-3, 0.015, 3.0
    @property
    def mu(self): return self.E / (2.0 * (1.0 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _eta_localization(discr, lam, mu, k_res=1e-6, q=8):
    """从真实 discr 状态算 η_T 并返回 top-20% 单元落在裂纹带 y≈0.5 的占比。"""
    mesh = discr.mesh
    qf = mesh.quadrature_formula(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = mesh.entity_measure('cell')
    pts = mesh.bc_to_point(bcs)

    grad_uh = discr.state.u.grad_value(bcs)             # (NC,NQ,2,2) DG 位移梯度
    sigmah_qp = discr.state.sigma(bcs)                  # (NC,NQ,3) 真实 σ_h
    d_qp = discr.state.d(bcs)                           # (NC,NQ) 真实相场
    if d_qp.ndim == 3:
        d_qp = d_qp[..., 0]

    ind = equilibrated_indicator(mesh, grad_uh, sigmah_qp, d_qp,
                                 lam=lam, mu=mu, k_res=k_res,
                                 weights=ws, cellmeasure=cm)
    etaT = ind["eta_T"]
    cen = mesh.bc_to_point(bm.array([[1/3, 1/3, 1/3]]))[:, 0, :]
    thr = float(bm.sort(etaT)[int(0.8 * len(etaT))])
    hi = cen[etaT >= thr]
    in_band = float(bm.sum(bm.abs(hi[:, 1] - 0.5) < 0.15)) / max(len(hi), 1)
    return ind["eta"], in_band


def test_m3_real_staggered_eta_localizes():
    """真实 model1 staggered：η_T 在真实裂纹带 y≈0.5 集中。"""
    bm.set_backend("numpy")
    mat = _Mat()
    nx = int(os.environ.get("FRACTUREX_NX", "16"))
    case = SquareTensionPreCrackCase(_model=mat, nx=nx, ny=nx,
                                     crack_y=0.5, crack_length=0.5)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case=case, p=3, damage_p=2,
                                  use_relaxation=True).build(mesh=mesh)
    damage = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                   split="hybrid", eps_g=1e-6)
    el_asm = HuZhangElasticAssembler(discr, case, damage,
                                     formulation="standard", assembly_parallel=False)
    ph_asm = PhaseFieldAssembler(discr, case, damage, assembly_parallel=False)

    records = []

    def adapt_hook(s, load, d_, dm_):
        if s < 1:   # 跳过 step0（载荷 0）
            return
        eta, in_band = _eta_localization(discr, mat.lam, mat.mu)
        md = float(bm.max(discr.state.d[:]))
        records.append((s, load, eta, in_band, md))
        print(f"[M3] step={s} load={load:.2e} η={eta:.3e} "
              f"裂纹带占比={100*in_band:.0f}% max_d={md:.3f}")

    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case, discr=discr, damage=damage,
        elastic_assembler=el_asm, phase_assembler=ph_asm,
        tol=1e-4, maxit=200, d_relaxation=1.0,
        elastic_solver=HuZhangPhaseFieldStaggeredDriver._default_spsolve,
        compute_linear_residual=False, debug=False, timing=False,
        save_vtu_per_step=False,
    )
    driver.adapt_hook = adapt_hook

    # 少数载荷步：跨过起裂（预裂纹已存在，几步即见 η 标记）
    loads = np.linspace(0.0, 5e-3, 6, dtype=float).tolist()
    driver.run(loads)

    assert records, "无记录（adapt_hook 未触发）"
    # 预裂纹在 y=0.5 ⇒ 各步 η_T 应集中该带
    band_fracs = [r[3] for r in records]
    avg = sum(band_fracs) / len(band_fracs)
    print(f"[M3] 裂纹带占比均值={100*avg:.0f}% over {len(records)} 步")
    assert avg > 0.7, f"η_T 未集中真实裂纹带（均值 {100*avg:.0f}%<70%）"
    print("[M3] 真实 staggered σ_h/d → η_T 标记集中真实裂纹带  OK")


if __name__ == "__main__":
    test_m3_real_staggered_eta_localizes()
    print("\n[M3 full minimal] DONE")
