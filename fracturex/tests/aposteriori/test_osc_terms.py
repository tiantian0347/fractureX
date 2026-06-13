"""T4: 数据振荡 osc(f) 随网格高阶衰减（THEORY §4 (6)，DESIGN T4）。

当 σ_h ∉ Σ_f（P_h f ≠ f，即 f 不落在 Hu–Zhang 位移空间），可靠性界 (6) 多一项
  osc(f) = (Σ_T (h_T/π)² ‖g^{-1/2}(f-P_h f)‖²_{0,T})^{1/2}
P_h f = f 到 DG 位移空间（p-1，与 Hu–Zhang 配对）的逐元 L2 投影。

验证：osc(f) 随 h 以高阶（≈p+2）衰减 ⇒ 是高阶小量、不污染主估计子 η，且界仍是上界。
退化 MMS 的 f≠0（含 ∇g），天然适合测 osc。
判据：osc(f) 收敛率 ≥ 3（DG p=2 时理论 O(h^4)）。
"""
from __future__ import annotations

import numpy as np
from sympy import symbols, sin, pi

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

from fracturex.adaptivity.degraded_mms import DegradedElasticMMS, crack_band_d

LAM, MU = 121.15, 80.77


def osc_f(mesh, pde, k_res, pdeg, q=14):
    """数据振荡 osc(f)（THEORY §4 (6) 的内部残余项）。

    输入:
      mesh,pde,k_res; pdeg: P_h f 投影的 DG 次数; q 积分阶
    输出: 标量 osc(f)
    """
    sp = LagrangeFESpace(mesh, p=pdeg, ctype='D')
    qf = mesh.quadrature_formula(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = mesh.entity_measure('cell')
    pts = mesh.bc_to_point(bcs)
    NC = mesh.number_of_cells()
    f = pde.source(pts)
    dq = pde.damage(pts)
    g = (1.0 - dq) ** 2 + k_res
    phi = sp.basis(bcs)
    if phi.shape[0] == 1:
        phi = bm.broadcast_to(phi, (NC,) + phi.shape[1:])
    M = bm.einsum('q,c,cqi,cqj->cij', ws, cm, phi, phi)      # 逐元质量阵
    osc2 = bm.zeros(NC)
    h = cm ** 0.5
    for comp in range(2):
        b = bm.einsum('q,c,cqi,cq->ci', ws, cm, phi, f[..., comp])
        coef = bm.linalg.solve(M, b[..., None])[..., 0]      # L2 投影系数
        Phf = bm.einsum('ci,cqi->cq', coef, phi)
        res = f[..., comp] - Phf
        osc2 = osc2 + (h / np.pi) ** 2 * bm.einsum('q,c,cq->c', ws, cm, res * res / g)
    return float(bm.sqrt(bm.sum(osc2)))


def test_osc_f_high_order_decay():
    """osc(f) 以高阶（≈p+2）衰减 ⇒ 高阶小量，不破坏可靠性。"""
    bm.set_backend("numpy")
    x, y = symbols('x y')
    u = [sin(pi * x) * sin(pi * y), 0.5 * sin(pi * x) * sin(pi * y)]
    d = crack_band_d(x, y, x0=0.5, width=0.15, dmax=0.9)
    k_res = 1e-3
    pde = DegradedElasticMMS(u, d, lam=LAM, mu=MU, k_res=k_res)

    oscs = []
    for N in (8, 16, 32, 64):
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
        o = osc_f(mesh, pde, k_res, pdeg=2)
        oscs.append(o)
    rates = [np.log2(oscs[i] / oscs[i + 1]) for i in range(len(oscs) - 1)]
    for N, o, r in zip((16, 32, 64), oscs[1:], rates):
        print(f"[T4] N={N} osc(f)={o:.4e} rate={r:.2f}")
    # 渐近率（最细两档）≥3，确认高阶衰减
    assert rates[-1] >= 3.0, f"osc(f) 衰减率 {rates[-1]:.2f} < 3（应高阶）"
    # osc 在较细网格已远小于典型 η(~1)：oscs=[N8,N16,N32,N64]，取 N=32
    assert oscs[2] < 0.05, f"osc(f) N=32={oscs[2]:.3e} 未足够小"
    print(f"[T4] osc(f) 高阶衰减（率≈{rates[-1]:.1f}），高阶小量不污染 η  OK")


if __name__ == "__main__":
    test_osc_f_high_order_decay()
    print("\n[T4] DONE")
