"""T6: 退化系数下 effectivity 的 k_res 门槛（THEORY §5 第3层，DESIGN T6）。
   + T3: 可靠性 η ≥ 真误差（常数=1）。

go/no-go 核心命题：Θ=η/‖ε(u_h)-ε(u)‖_{C_d} 跟随**局部对比度 κ**（裂纹带光滑 ⇒ O(1)），
**不**随全局 k_res→0 发散（THEORY §5 朴素界 (8) 含 k_res^{-1}，但局部界 (9) 与之解耦）。

设计：退化 MMS（解析 u + 裂纹带 d(x) + 一致源 f），同一网格上
  u_h ← 标准 FEM 退化解（solve_primal_degraded）
  σ_h ← Hu–Zhang 混合退化解（solve_degraded_huzhang，平衡）
扫 k_res ∈ {1e-3,1e-5,1e-7}，量 η / 真误差 / Θ。

注：f≠0（含 ∇g），故有 osc(f) 余项（一并报）；primal 用逐元常数 g（κ_T=O(1) 前提）。
诚实标注见 RESULTS。运行 p=3（Hu–Zhang 空间要求）。
"""
from __future__ import annotations

import numpy as np
from sympy import symbols, sin, pi

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.adaptivity.degraded_mms import DegradedElasticMMS, crack_band_d
from fracturex.adaptivity.degraded_huzhang_solve import solve_degraded_huzhang
from fracturex.adaptivity.primal_elastic_solve import solve_primal_degraded
from fracturex.adaptivity.equilibrated_estimator import (
    equilibrated_indicator, energy_error_Cd, effectivity_index,
)

LAM, MU = 121.15, 80.77


def _all_dirichlet(bc):
    tol = 1e-12
    return ((bm.abs(bc[:, 0] - 0) < tol) | (bm.abs(bc[:, 0] - 1) < tol)
            | (bm.abs(bc[:, 1] - 0) < tol) | (bm.abs(bc[:, 1] - 1) < tol))


def run_one(N, p, k_res, *, width=0.15, dmax=0.9, q=12):
    """单次 (N,k_res) 实验，返回 η/真误差/Θ/osc/局部对比度 κ。"""
    x, y = symbols('x y')
    u = [sin(pi * x) * sin(pi * y), 0.5 * sin(pi * x) * sin(pi * y)]
    d = crack_band_d(x, y, x0=0.5, width=width, dmax=dmax)
    pde = DegradedElasticMMS(u, d, lam=LAM, mu=MU, k_res=k_res)

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)

    # 两个解
    prim = solve_primal_degraded(mesh, p, pde, lam=LAM, mu=MU, k_res=k_res,
                                 dirichlet=pde.displacement)
    hz = solve_degraded_huzhang(mesh, p, pde, lam=LAM, mu=MU, k_res=k_res,
                                isD_bd=_all_dirichlet)

    # 同一组 qp 上求值
    qf = mesh.quadrature_formula(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = mesh.entity_measure('cell')
    pts = mesh.bc_to_point(bcs)

    grad_uh = prim["uh"].grad_value(bcs)            # (NC,NQ,2,2)
    sigmah_qp = hz["sigmah"](bcs)                   # (NC,NQ,3)
    d_qp = pde.damage(pts)                          # (NC,NQ)
    grad_u_exact = pde.grad_u(pts)                  # (NC,NQ,2,2)

    ind = equilibrated_indicator(mesh, grad_uh, sigmah_qp, d_qp,
                                 lam=LAM, mu=MU, k_res=k_res,
                                 weights=ws, cellmeasure=cm)
    eta = ind["eta"]
    err = energy_error_Cd(grad_uh, grad_u_exact, d_qp,
                          lam=LAM, mu=MU, k_res=k_res,
                          weights=ws, cellmeasure=cm)
    theta = effectivity_index(eta, err)

    # 局部对比度 κ = max_qp g / min_qp g（全域）
    g = (1.0 - d_qp) ** 2 + k_res
    kappa = float(bm.max(g) / bm.min(g))

    # Θ-1 的真实来源：‖σ_h-σ‖_A / err（精确表达式 Θ²=1+ratio²，THEORY (7)）
    # 比 Θ≈1.000 灵敏——直接暴露有效性是否随 k_res 退化
    from fracturex.adaptivity.equilibrated_estimator import (
        compliance_apply_voigt, voigt_inner)
    sig_exact = pde.stress(pts)
    rdiff = sigmah_qp - sig_exact
    Ardiff = compliance_apply_voigt(rdiff, LAM, MU)
    sig_diff = float(bm.sqrt(bm.sum(
        cm * bm.einsum('q,cq->c', ws, voigt_inner(rdiff, Ardiff) / g))))
    ratio = sig_diff / max(err, 1e-30)
    return {"eta": eta, "err": err, "theta": theta, "kappa": kappa,
            "sig_diff": sig_diff, "ratio": ratio, "N": N, "k_res": k_res}


def test_reliability_T3():
    """T3: η ≥ 真误差（可靠性常数=1，THEORY (4)）。允许 osc 小裕度。"""
    bm.set_backend("numpy")
    fails = []
    for k_res in (1e-3, 1e-5):
        for N in (12, 16):
            r = run_one(N, 3, k_res)
            ratio = r["eta"] / max(r["err"], 1e-30)
            print(f"[T3] N={N} k_res={k_res:.0e}  η={r['eta']:.3e} "
                  f"err={r['err']:.3e}  η/err={ratio:.3f}")
            if ratio < 0.98:  # 容 2% 数值/osc 裕度
                fails.append((N, k_res, ratio))
    assert not fails, f"reliability η<err: {fails}"
    print("[T3] 可靠性 η≥真误差 (常数≈1)  OK")


def test_kres_threshold_T6():
    """T6 门槛: 有效性不随 k_res→0 发散。

    用 ratio=‖σ_h-σ‖_A/err 作灵敏指标（Θ²=1+ratio²，THEORY (7)）——
    比 Θ≈1.000 四舍五入更能暴露退化。朴素界 (8) 预测 ratio∝k_res^{-1/2}，
    局部界 (9) 预测 ratio 随局部对比度 κ 饱和、与 k_res 解耦。
    """
    bm.set_backend("numpy")
    N = 16
    rows = []
    for k_res in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        r = run_one(N, 3, k_res)
        rows.append(r)
        print(f"[T6] k_res={k_res:.0e}  ratio=‖σh-σ‖/err={r['ratio']:.4f}  "
              f"κ={r['kappa']:.1f}  Θ={r['theta']:.6f}")
    ratios = [r["ratio"] for r in rows]
    # 门槛：ratio 跨 5 个量级 k_res 应饱和（增长有界），不随 k_res^{-1/2} 爆
    growth = ratios[-1] / max(ratios[0], 1e-30)
    naive = (1e-2 / 1e-7) ** 0.5      # 朴素界 (8) 预测增长 ~316×
    print(f"[T6] ratio 增长 (k_res 1e-2→1e-7) = {growth:.2f}×；"
          f"朴素界 (8) 预测 ~{naive:.0f}×")
    assert growth < 5.0, (
        f"有效性随 k_res→0 发散 (ratio 增长 {growth:.1f}×) ⇒ 触发门槛退路")
    print(f"[T6] ratio 饱和 (增长 {growth:.2f}× << 朴素界 {naive:.0f}×) "
          f"⇒ 跟随局部 κ、与全局 k_res 解耦  GO ✅")


if __name__ == "__main__":
    test_reliability_T3()
    print()
    test_kres_threshold_T6()
    print("\n[T3+T6] DONE")
