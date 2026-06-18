# fracturex/tests/aposteriori/test_amor_dual_closed_form.py
"""T5: Amor 分裂对偶势闭式验证（THEORY (14) / Theorem 2，DESIGN T5）。

验证 amor_dual_energy 的闭式 ψ*(τ,d) 正确：
  (a) 对偶闭式 vs brute-force Legendre sup_ε(τ:ε-ψ(ε))  机器精度
  (b) Fenchel–Young 不等式 ψ(ε)+ψ*(τ)-τ:ε ≥ 0 逐点（任意 ε,τ）
  (c) 取等：τ=∂_ε ψ(ε) ⇒ gap=0（majorant 间隙为零的点）
扫 g（=退化）∈{1,0.37,1e-3} 覆盖退化区。

依赖 scipy.optimize（仅测试，第三方边界，允许 numpy）。
运行：PYTHONPATH=tian/fealpy:tian/fracturex python .../test_amor_dual_closed_form.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from fealpy.backend import backend_manager as bm
from fracturex.adaptivity.equilibrated_estimator import (
    amor_energy, amor_stress, amor_dual_energy, _bulk_modulus_2d,
)

LAM, MU = 121.15, 80.77
K = _bulk_modulus_2d(LAM, MU)


def _psi_np(e2x2, g):
    """numpy 标量版 ψ（独立实现，交叉校验 amor_energy）。"""
    tr = e2x2[0, 0] + e2x2[1, 1]
    dev = e2x2 - 0.5 * tr * np.eye(2)
    return g * (0.5 * K * max(tr, 0) ** 2 + MU * np.sum(dev * dev)) \
        + 0.5 * K * min(tr, 0) ** 2


def _to_voigt(e2x2):
    return bm.array([e2x2[0, 0], e2x2[0, 1], e2x2[1, 1]])


def _bruteforce_dual(t2x2, g):
    """ψ*(τ)=sup_ε(τ:ε-ψ(ε)) via 多起点 Nelder-Mead。"""
    def negobj(v):
        e = np.array([[v[0], v[2]], [v[2], v[1]]])
        return -(np.sum(t2x2 * e) - _psi_np(e, g))
    best = -1e18
    for x0 in [np.zeros(3), np.ones(3), -np.ones(3),
               np.array([1., -1., .5]), np.array([-2., 3., -1.])]:
        r = minimize(negobj, x0, method='Nelder-Mead',
                     options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 200000})
        best = max(best, -r.fun)
    return best


def _d_from_g(g):
    """由 g 反解 d（k_res 设极小使 g≈(1-d)²）。这里直接给 d，k_res 单列。"""
    return 1.0 - np.sqrt(max(g, 0.0))


def test_closed_form_vs_legendre():
    """(a) 闭式 ψ* = brute-force Legendre。"""
    bm.set_backend("numpy")
    taus = [np.array([[10., 2.], [2., -5.]]),
            np.array([[-8., 1.], [1., -3.]]),
            np.array([[3., -4.], [-4., 7.]]),
            np.array([[-6., 2.], [2., -6.]])]
    k_res = 1e-12  # 使 g≈(1-d)²，与 brute 的 g 对齐
    maxrel = 0.0
    for g in (1.0, 0.37, 1e-3):
        d = _d_from_g(g)
        for t in taus:
            cf = float(amor_dual_energy(_to_voigt(t), bm.array(d),
                                        lam=LAM, mu=MU, k_res=k_res))
            bf = _bruteforce_dual(t, g)
            rel = abs(cf - bf) / max(abs(bf), 1e-6)
            maxrel = max(maxrel, rel)
    assert maxrel < 1e-9, f"closed form vs Legendre max rel={maxrel:.2e}"
    print(f"[T5.a] ψ* 闭式 vs brute-force Legendre  max rel={maxrel:.2e}  OK")


def test_fenchel_young_nonneg():
    """(b) ψ(ε)+ψ*(τ)-τ:ε ≥ 0 逐点（任意 ε,τ）。"""
    bm.set_backend("numpy")
    k_res = 1e-6
    rng = [(np.array([[0.3, -0.1], [-0.1, 0.2]]), np.array([[5., 1.], [1., -2.]])),
           (np.array([[-0.4, .05], [.05, -0.6]]), np.array([[-3., 2.], [2., 4.]])),
           (np.array([[0.5, 0.3], [0.3, -0.5]]), np.array([[0., 0.], [0., 0.]]))]
    for g in (1.0, 0.37, 1e-3):
        d = _d_from_g(g)
        for e, t in rng:
            psi = float(amor_energy(_to_voigt(e), bm.array(d), lam=LAM, mu=MU, k_res=k_res))
            dual = float(amor_dual_energy(_to_voigt(t), bm.array(d), lam=LAM, mu=MU, k_res=k_res))
            pairing = float(np.sum(t * e))
            gap = psi + dual - pairing
            assert gap >= -1e-9, f"Fenchel-Young violated: gap={gap:.3e}"
    print("[T5.b] Fenchel–Young ψ+ψ*-τ:ε ≥ 0 逐点（扫 g）  OK")


def test_fenchel_young_equality_at_stress():
    """(c) τ=∂_ε ψ(ε) ⇒ gap=0（majorant 间隙为零）。"""
    bm.set_backend("numpy")
    k_res = 1e-6
    es = [np.array([[0.3, -0.1], [-0.1, 0.2]]),
          np.array([[-0.4, .05], [.05, -0.6]]),
          np.array([[0.5, 0.3], [0.3, -0.5]])]
    for g in (1.0, 0.37, 1e-3):
        d = bm.array(_d_from_g(g))
        for e in es:
            ev = _to_voigt(e)
            psi = float(amor_energy(ev, d, lam=LAM, mu=MU, k_res=k_res))
            t_v = amor_stress(ev, d, lam=LAM, mu=MU, k_res=k_res)
            dual = float(amor_dual_energy(t_v, d, lam=LAM, mu=MU, k_res=k_res))
            # τ:ε，Voigt off-diag 双倍
            pairing = float(t_v[0] * ev[0] + 2 * t_v[1] * ev[1] + t_v[2] * ev[2])
            gap = psi + dual - pairing
            assert abs(gap) < 1e-7, f"equality gap={gap:.3e} at g={g}"
    print("[T5.c] τ=∂_ε ψ ⇒ Fenchel–Young 取等 (gap≈0)  OK")


if __name__ == "__main__":
    test_closed_form_vs_legendre()
    test_fenchel_young_nonneg()
    test_fenchel_young_equality_at_stress()
    print("\n[T5] ALL PASS")
