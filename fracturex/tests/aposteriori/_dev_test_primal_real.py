"""开发自测：solve_primal_real 在真实算例接受态网格上跑通 + 喂 eta_from_state 出严格 Θ。

非正式 pytest——用于 Phase 3 立稳：建 discr → 解一载荷步 → 连续 primal 重解 → η_τ。
环境 py312 + PYTHONPATH。运行: python fracturex/tests/aposteriori/_dev_test_primal_real.py
"""
from __future__ import annotations

from fealpy.backend import backend_manager as bm

from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.adaptivity.adaptive_staggered import make_assemblers, eta_from_state
from fracturex.adaptivity.primal_resolve_real import solve_primal_real


class _Mat:
    E, nu, Gc, l0, ft = 210.0, 0.3, 2.7e-3, 0.015, 3.0
    @property
    def mu(self): return self.E / (2.0 * (1.0 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def main():
    bm.set_backend("numpy")
    mat = _Mat()
    k_res = 1e-6
    nx = 24
    load = 5.0e-3   # 接近峰值的载荷，d 已有发育
    case = SquareTensionPreCrackCase(_model=mat, nx=nx, ny=nx,
                                     crack_y=0.5, crack_length=0.5)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case=case, p=3, damage_p=1,
                                  use_relaxation=True).build(mesh=mesh)
    damage = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                   split="hybrid", eps_g=k_res)
    el_asm, ph_asm = make_assemblers(discr, case, damage)
    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case, discr=discr, damage=damage,
        elastic_assembler=el_asm, phase_assembler=ph_asm,
        tol=1e-4, maxit=200, d_relaxation=1.0,
        compute_linear_residual=False, debug=False, timing=False,
        save_vtu_per_step=False, stagger_print_interval=0,
    )
    driver.initialize()
    # 走几个载荷步把 d 发育起来（粗略，仅自测）
    for s, ld in enumerate([1e-3, 2e-3, 3e-3, 4e-3, 5e-3]):
        info = driver.solve_one_step(step=s, load=ld)
    print(f"[dev] staggered solved: load={ld} max_d={float(info.max_d):.4f} "
          f"R={abs(float(info.meta.get('R',0))):.4e} nc={discr.mesh.number_of_cells()}")

    # --- 连续 primal 重解（真实 BC + 离散 d 场）---
    out = solve_primal_real(discr, case, lam=mat.lam, mu=mat.mu, load=ld, k_res=k_res)
    uh = out["uh"]
    print(f"[dev] primal_real solved: u_dofs={uh.shape} "
          f"||u||_inf={float(bm.max(bm.abs(uh[:]))):.4e} "
          f"g_cell∈[{float(bm.min(out['g_cell'])):.2e},{float(bm.max(out['g_cell'])):.3f}]")

    # 边界正确性：y=1 边的 u_y 应≈load，y=0 边 u≈0
    space = out["space"]
    ip = space.mesh.interpolation_points(p=space.p)
    import numpy as np
    # 分量优先布局 (GD,npoints)：reshape(2,npoint) → [x行, y行]
    npoint = ip.shape[0]
    uflat = bm.to_numpy(uh[:]).reshape(2, npoint)
    y = bm.to_numpy(ip[:, 1])
    top = np.abs(y - 1.0) < 1e-9
    bot = np.abs(y - 0.0) < 1e-9
    print(f"[dev] BC check: y=1 u_y mean={uflat[1, top].mean():.4e} (target {ld:.4e}); "
          f"y=0 |u| max={np.abs(uflat[:, bot]).max():.2e} (target 0)")

    # --- 严格 η_τ：用连续 H¹ primal u_h 的梯度替换 DG-u 喂 η（Prager–Synge 需协调 primal）---
    est_dg = eta_from_state(discr, lam=mat.lam, mu=mat.mu, k_res=k_res)
    est_strict = eta_from_state(discr, lam=mat.lam, mu=mat.mu, k_res=k_res,
                                u_override=uh)
    print(f"[dev] eta(DG-u,非严格)     = {float(est_dg['eta']):.4e}")
    print(f"[dev] eta(连续 primal,严格) = {float(est_strict['eta']):.4e}  "
          f"← 保证型界(reconstruction-free, 常数=1)")
    # 注：真实算例无解析解 ⇒ 报 η 作认证界即可；效率 Θ=η/err 须真解，已在 T6(MMS)验 Θ≈1。
    print("[dev] OK: solve_primal_real + 严格 η_τ 跑通。")


if __name__ == "__main__":
    main()
