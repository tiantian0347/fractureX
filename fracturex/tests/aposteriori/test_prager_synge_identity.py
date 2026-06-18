# fracturex/tests/aposteriori/test_prager_synge_identity.py
"""T1a: Prager–Synge 恒等式的核心正交性（THEORY Thm 1 / (7)，DESIGN T1）。

超圆恒等式 ‖C ε(v)-σ‖²_A + ‖τ-σ‖²_A = ‖C ε(v)-τ‖²_A 成立 ⟺ 交叉项
  ∫ (C ε(v)-σ):A:(σ-τ) = ∫ ε(w):(σ-τ) = 0,   w=v-u∈V_0
对任意 w（∂Ω 上为零）与任意自平衡 δ:=σ-τ（div δ=0、纯 Dirichlet ⇒ Γ_N=∅）成立。

本测试**解析构造** w 与 δ，在真实三角网格上用高阶求积验：
  (a) 交叉项 ∫ ε(w):δ ≈ 0  （机器/求积精度）
  (b) 完整超圆恒等式两端 rel < 1e-10
不解 PDE —— 直接坐实定理的代数/分析心脏。

自平衡 δ 由 Airy 应力函数 φ 生成（自动 div δ=0、逐点对称）：
  δ_xx = φ_yy,  δ_yy = φ_xx,  δ_xy = -φ_xy
取 φ = sin(πx) sin(πy)；w = (sin πx · sin πy, 0) 在 ∂[0,1]² 上为零。
设 d≡0（C_d=C），纯 Dirichlet（Γ_N=∅，τ∈Σ_f 仅需 div τ=-f）。

运行：
  PYTHONPATH=tian/fealpy:tian/fracturex python .../test_prager_synge_identity.py
"""
from __future__ import annotations

import math
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.adaptivity.equilibrated_estimator import (
    voigt_inner, compliance_apply_voigt, stress_voigt_from_strain,
)

LAM, MU = 121.15, 80.77
PI = math.pi


def _fields_at(pts):
    """解析场在物理点 pts:(...,2) 上求值。

    返回:
      eps_w   : (...,3) ε(w) Voigt (xx,xy,yy)
      delta   : (...,3) 自平衡应力 δ Voigt (xx,xy,yy), div δ=0
    """
    x = pts[..., 0]
    y = pts[..., 1]
    sx, cx = bm.sin(PI * x), bm.cos(PI * x)
    sy, cy = bm.sin(PI * y), bm.cos(PI * y)

    # w = (sin πx sin πy, 0)  ⇒ ∇w
    # w0_x = π cx sy, w0_y = π sx cy ; w1 = 0
    eps_xx = PI * cx * sy
    eps_yy = 0.0 * x
    eps_xy = 0.5 * (PI * sx * cy)          # 0.5(∂y w0 + ∂x w1)
    eps_w = bm.stack([eps_xx, eps_xy, eps_yy], axis=-1)

    # Airy φ = sin πx sin πy ⇒ φ_xx=φ_yy=-π² sx sy, φ_xy = π² cx cy
    phi_xx = -(PI ** 2) * sx * sy
    phi_yy = -(PI ** 2) * sx * sy
    phi_xy = (PI ** 2) * cx * cy
    d_xx = phi_yy
    d_yy = phi_xx
    d_xy = -phi_xy
    delta = bm.stack([d_xx, d_xy, d_yy], axis=-1)
    return eps_w, delta


def _integrate(mesh, q, fn):
    """∫_Ω fn(pts) dx via 高阶求积；fn 返回 (NC,NQ) 标量场。"""
    qf = mesh.quadrature_formula(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = mesh.entity_measure('cell')
    pts = mesh.bc_to_point(bcs)               # (NC,NQ,2)
    val = fn(pts)                             # (NC,NQ)
    return float(bm.sum(cm * bm.einsum('q,cq->c', ws, val)))


def test_cross_term_orthogonality():
    """(a) ∫ ε(w):δ = 0（w|∂Ω=0, div δ=0）。"""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=8, ny=8)

    def cross(pts):
        eps_w, delta = _fields_at(pts)
        return voigt_inner(eps_w, delta)      # ε(w):δ

    val = _integrate(mesh, q=10, fn=cross)
    assert abs(val) < 1e-10, f"cross term ∫ε(w):δ = {val:.3e} (should ≈0)"
    print(f"[T1a.a] ∫ ε(w):δ = {val:.3e}  (<1e-10)  OK")


def test_full_hypercircle_identity():
    """(b) ‖Cε(w)‖²_A + ‖δ‖²_A = ‖Cε(w)-δ‖²_A（A=C⁻¹, d≡0）。

    取 v=u+w ⇒ Cε(v)-σ=Cε(w)；τ=σ-δ ⇒ τ-σ=-δ。范数 ‖·‖_A=∫(·):C⁻¹(·)。
    """
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=8, ny=8)

    def make(fn_pick):
        def inner(pts):
            eps_w, delta = _fields_at(pts)
            Cew = stress_voigt_from_strain(eps_w, LAM, MU)   # C ε(w) (=a)
            b = -delta                                       # τ-σ = -δ
            a = Cew
            return fn_pick(a, b)
        return inner

    A = compliance_apply_voigt
    lhs1 = _integrate(mesh, 10, make(lambda a, b: voigt_inner(a, A(a, LAM, MU))))
    lhs2 = _integrate(mesh, 10, make(lambda a, b: voigt_inner(b, A(b, LAM, MU))))
    rhs = _integrate(mesh, 10, make(lambda a, b: voigt_inner(a - b, A(a - b, LAM, MU))))
    rel = abs((lhs1 + lhs2) - rhs) / max(abs(rhs), 1e-30)
    assert rel < 1e-10, f"hypercircle identity rel={rel:.3e}"
    print(f"[T1a.b] ‖a‖²+‖b‖²={lhs1+lhs2:.6e}  ‖a-b‖²={rhs:.6e}  rel={rel:.2e}  OK")


if __name__ == "__main__":
    test_cross_term_orthogonality()
    test_full_hypercircle_identity()
    print("\n[T1a] ALL PASS")
