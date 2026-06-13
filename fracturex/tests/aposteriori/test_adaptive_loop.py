"""T8 / M2: 自适应 vs 均匀的等精度 DOF 效率（DESIGN T8 / M2 循环）。

验证平衡型估计子驱动的自适应加密：
  (a) η 单调下降、Θ 全程≈1（估计子在加密中保持尖锐可靠）
  (b) 等精度下自适应 DOF << 均匀 DOF（DOF 集中到裂纹带）

退化 MMS + 窄裂纹带 d(x)（width=0.08），p=3。
判据：自适应达到与某均匀网格相近精度时，DOF 显著更少（>40%）。
运行较重（多次混合求解），timeout 设宽。
"""
from __future__ import annotations

from sympy import symbols, sin, pi
from fealpy.backend import backend_manager as bm

from fracturex.adaptivity.degraded_mms import DegradedElasticMMS, crack_band_d
from fracturex.adaptivity.adaptive_loop_equilibrated import (
    adaptive_loop, uniform_refine_loop,
)

LAM, MU = 121.15, 80.77


def _make_pde(k_res=1e-3, width=0.08, dmax=0.95):
    x, y = symbols('x y')
    u = [sin(pi * x) * sin(pi * y), 0.5 * sin(pi * x) * sin(pi * y)]
    d = crack_band_d(x, y, x0=0.5, width=width, dmax=dmax)
    return DegradedElasticMMS(u, d, lam=LAM, mu=MU, k_res=k_res)


def test_eta_decreases_theta_stable():
    """(a) η 单调下降、Θ 全程接近 1。"""
    bm.set_backend("numpy")
    pde = _make_pde()
    hist = adaptive_loop(pde, lam=LAM, mu=MU, k_res=1e-3, p=3,
                         N0=8, theta=0.5, max_iter=5, verbose=False)
    etas = [h["eta"] for h in hist]
    thetas = [h["theta"] for h in hist]
    # η 单调下降
    assert all(etas[i] > etas[i + 1] for i in range(len(etas) - 1)), \
        f"η 非单调下降: {etas}"
    # Θ 全程接近 1（可靠且尖锐）
    assert all(0.99 <= t <= 1.05 for t in thetas), f"Θ 偏离 1: {thetas}"
    print(f"[T8.a] η 单调降 {etas[0]:.3e}→{etas[-1]:.3e}; "
          f"Θ∈[{min(thetas):.4f},{max(thetas):.4f}] 全程≈1  OK")


def test_adaptive_beats_uniform_dof():
    """(b) 等精度下自适应 DOF << 均匀 DOF。"""
    bm.set_backend("numpy")
    pde = _make_pde()
    ad = adaptive_loop(pde, lam=LAM, mu=MU, k_res=1e-3, p=3,
                       N0=8, theta=0.5, max_iter=7, verbose=False)
    un = uniform_refine_loop(pde, lam=LAM, mu=MU, k_res=1e-3, p=3,
                             N_list=(8, 12, 16, 24), verbose=False)
    # 取自适应最细一档，找均匀里精度相近或更好的，比 DOF
    a = ad[-1]
    # 均匀里 err <= 自适应 err 的最小 DOF 档
    better = [u_ for u_ in un if u_["err"] <= a["err"] * 1.05]
    assert better, f"均匀未达自适应精度 {a['err']:.3e}（加密 N_list 上限）"
    u_match = min(better, key=lambda u_: u_["dof"])
    saving = 1 - a["dof"] / u_match["dof"]
    print(f"[T8.b] 自适应 err={a['err']:.3e} dof={a['dof']}  vs  "
          f"均匀 err={u_match['err']:.3e} dof={u_match['dof']}  DOF省={100*saving:.0f}%")
    assert saving > 0.4, f"自适应未显著省 DOF（省{100*saving:.0f}%，期望>40%）"
    print("[T8.b] 等精度自适应 DOF 显著更少  OK")


if __name__ == "__main__":
    test_eta_decreases_theta_stable()
    test_adaptive_beats_uniform_dof()
    print("\n[T8/M2] DONE")
