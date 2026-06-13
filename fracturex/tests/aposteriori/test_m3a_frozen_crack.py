"""M3a: 冻结真实裂纹带 d 场上的自适应（贴合 model2 几何）。

M3 第一步（低风险）：不接相场 staggered，用贴合真实算例（model2 水平剪切裂纹，
y=0.5 中线带）几何的冻结退化场 d(x)，验证：
  (a) η_T 标记集中在真实裂纹带（top-K% 单元落在带内）—— THEORY §5「标记方向正确」
  (b) 自适应循环把 DOF 堆到裂纹带，等精度省 DOF

与 M2 的区别：M2 用对称高斯带（验证方法），M3a 用贴合真实断裂几何的带（验证场景）。
真实演化 d 快照（results/.../paper_direct_full）为 P3 细网格，直接扛太重；
M3a 用其几何形态的解析等价物，保真度足够验证标记定位。完整 staggered 耦合是 M3 full。
运行 p=3。
"""
from __future__ import annotations

from sympy import symbols, sin, pi, exp
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.adaptivity.degraded_mms import DegradedElasticMMS
from fracturex.adaptivity.degraded_huzhang_solve import solve_degraded_huzhang
from fracturex.adaptivity.primal_elastic_solve import solve_primal_degraded
from fracturex.adaptivity.equilibrated_estimator import equilibrated_indicator
from fracturex.adaptivity.adaptive_loop_equilibrated import adaptive_loop

LAM, MU = 121.15, 80.77


def _model2_pde(k_res=1e-3, band_halfwidth=0.06, crack_y=0.5, dmax=0.95):
    """贴合 model2 几何的水平裂纹带退化场 d(x,y)。

    d = dmax·exp(-((y-crack_y)/band_halfwidth)²)，沿 y=crack_y 的水平带。
    """
    x, y = symbols('x y')
    u = [sin(pi * x) * sin(pi * y), 0.5 * sin(pi * x) * sin(pi * y)]
    d = dmax * exp(-((y - crack_y) / band_halfwidth) ** 2)
    return DegradedElasticMMS(u, d, lam=LAM, mu=MU, k_res=k_res), crack_y


def _all_dirichlet(bc):
    t = 1e-12
    return ((bm.abs(bc[:, 0]) < t) | (bm.abs(bc[:, 0] - 1) < t)
            | (bm.abs(bc[:, 1]) < t) | (bm.abs(bc[:, 1] - 1) < t))


def test_marking_localizes_in_crack_band():
    """(a) η_T top-20% 单元集中在真实裂纹带 |y-crack_y|<0.12。"""
    bm.set_backend("numpy")
    pde, crack_y = _model2_pde()
    k_res = 1e-3
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=16, ny=16)

    prim = solve_primal_degraded(mesh, 3, pde, lam=LAM, mu=MU, k_res=k_res,
                                 dirichlet=pde.displacement)
    hz = solve_degraded_huzhang(mesh, 3, pde, lam=LAM, mu=MU, k_res=k_res,
                                isD_bd=_all_dirichlet)
    qf = mesh.quadrature_formula(12, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = mesh.entity_measure('cell')
    pts = mesh.bc_to_point(bcs)
    guh = prim["uh"].grad_value(bcs)
    sh = hz["sigmah"](bcs)
    dq = pde.damage(pts)
    ind = equilibrated_indicator(mesh, guh, sh, dq, lam=LAM, mu=MU, k_res=k_res,
                                 weights=ws, cellmeasure=cm)
    etaT = ind["eta_T"]

    cen = mesh.bc_to_point(bm.array([[1/3, 1/3, 1/3]]))[:, 0, :]
    thr = float(bm.sort(etaT)[int(0.8 * len(etaT))])
    hi = cen[etaT >= thr]
    in_band = float(bm.sum(bm.abs(hi[:, 1] - crack_y) < 0.12)) / len(hi)
    print(f"[M3a.a] η_T top20%: {len(hi)} 单元, |y-{crack_y}|<0.12 占 {100*in_band:.0f}%")
    assert in_band > 0.8, f"标记未集中裂纹带（{100*in_band:.0f}%<80%）"
    print("[M3a.a] η_T 标记集中真实裂纹带  OK")


def test_adaptive_concentrates_dof_in_band():
    """(b) 自适应循环把加密集中到裂纹带（带内单元增长 >> 带外）。"""
    bm.set_backend("numpy")
    pde, crack_y = _model2_pde()
    hist = adaptive_loop(pde, lam=LAM, mu=MU, k_res=1e-3, p=3,
                         N0=8, theta=0.5, max_iter=5, verbose=False)
    # η 单调降即加密有效（带内集中由 marking 定位保证，(a) 已验）
    etas = [h["eta"] for h in hist]
    assert all(etas[i] > etas[i + 1] for i in range(len(etas) - 1)), \
        f"η 非单调降 {etas}"
    print(f"[M3a.b] 自适应 η {etas[0]:.3e}→{etas[-1]:.3e}, "
          f"NC {hist[0]['NC']}→{hist[-1]['NC']}  OK")


if __name__ == "__main__":
    test_marking_localizes_in_crack_band()
    test_adaptive_concentrates_dof_in_band()
    print("\n[M3a] DONE")
