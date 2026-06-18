# fracturex/adaptivity/equilibrated_estimator.py
"""平衡型（equilibrated）a posteriori 误差指示子。

理论见 docs/adaptive/THEORY_equilibrated_aposteriori.md：
  η_T^2 = ∫_T g^{-1} (C_d ε(u_h) - σ_h) : C^{-1} (C_d ε(u_h) - σ_h)
其中
  g(d)        = (1-d)^2 + k_res        退化函数（严格 > 0）
  C_d ε(u_h)  = g · (λ tr(ε) I + 2μ ε) 标准 FEM 退化应力（平面应变 Lamé）
  σ_h                                  Hu–Zhang 平衡应力（H(div;S) 协调、逐点对称）
  C^{-1}                               未退化柔度

可靠性：当 σ_h ∈ Σ_f（f=0、t_N=0 基准下精确）时 η ≥ ‖ε(u_h)-ε(u)‖_{C_d}，
可靠性常数严格 = 1（THEORY (4)）。

记号约定（与 fealpy HuZhangFESpace2d 一致）：
  2D 对称应力以 Voigt 三分量 (xx, xy, yy) 存储；qp 上张量场形如 (NC, NQ, 3)。
  位移梯度 grad_value 形如 (NC, NQ, 2, 2)。
计算一律走 backend_manager（bm），numpy 仅限文件 I/O / 第三方边界。
"""
from __future__ import annotations

from typing import Callable, Optional, Union

from fealpy.backend import backend_manager as bm


# ----------------------------------------------------------------------------
# 张量 ↔ Voigt 辅助（2D 对称，分量序 (xx, xy, yy)）
# ----------------------------------------------------------------------------
def strain_to_voigt(grad_u):
    """对称应变 ε(u)=sym(∇u) → Voigt (xx, xy, yy)。

    输入:
      grad_u : (..., 2, 2)  位移梯度 ∇u（最后两维为张量）
    输出:
      (..., 3)  应变 Voigt 分量 (ε_xx, ε_xy, ε_yy)，ε_xy 为真分量（非 2·ε_xy）
    """
    e_xx = grad_u[..., 0, 0]
    e_yy = grad_u[..., 1, 1]
    e_xy = 0.5 * (grad_u[..., 0, 1] + grad_u[..., 1, 0])
    return bm.stack([e_xx, e_xy, e_yy], axis=-1)


def stress_voigt_from_strain(eps_voigt, lam: float, mu: float):
    """未退化各向同性应力 C·ε（平面应变），Voigt 序 (xx, xy, yy)。

    平面应变: σ_xx = λ(ε_xx+ε_yy) + 2μ ε_xx
              σ_yy = λ(ε_xx+ε_yy) + 2μ ε_yy
              σ_xy = 2μ ε_xy
    输入:
      eps_voigt : (..., 3) 应变 (ε_xx, ε_xy, ε_yy)
      lam, mu   : Lamé 参数
    输出:
      (..., 3) 应力 (σ_xx, σ_xy, σ_yy)
    """
    e_xx = eps_voigt[..., 0]
    e_xy = eps_voigt[..., 1]
    e_yy = eps_voigt[..., 2]
    tr = e_xx + e_yy
    s_xx = lam * tr + 2.0 * mu * e_xx
    s_yy = lam * tr + 2.0 * mu * e_yy
    s_xy = 2.0 * mu * e_xy
    return bm.stack([s_xx, s_xy, s_yy], axis=-1)


def compliance_apply_voigt(sig_voigt, lam: float, mu: float):
    """应用未退化柔度 C^{-1}·σ → 应变，Voigt 序 (xx, xy, yy)。

    ε = (1/(2μ)) (σ - (λ/(2(λ+μ))) tr(σ) I)（平面应变，与 recover_strain.py 同款公式）。
    输入:
      sig_voigt : (..., 3) 应力 (σ_xx, σ_xy, σ_yy)
      lam, mu   : Lamé 参数
    输出:
      (..., 3) 应变 (ε_xx, ε_xy, ε_yy)
    """
    s_xx = sig_voigt[..., 0]
    s_xy = sig_voigt[..., 1]
    s_yy = sig_voigt[..., 2]
    tr = s_xx + s_yy
    coef = lam / (2.0 * (lam + mu))
    e_xx = (s_xx - coef * tr) / (2.0 * mu)
    e_yy = (s_yy - coef * tr) / (2.0 * mu)
    e_xy = s_xy / (2.0 * mu)
    return bm.stack([e_xx, e_xy, e_yy], axis=-1)


def voigt_inner(a_voigt, b_voigt):
    """对称张量 Voigt 内积 a:b（off-diagonal 计双倍），返回 (...,)。

    a:b = a_xx b_xx + 2 a_xy b_xy + a_yy b_yy（对称张量的真内积）。
    输入:
      a_voigt, b_voigt : (..., 3) Voigt (xx, xy, yy)
    输出:
      (...,) 标量场 a:b
    """
    return (a_voigt[..., 0] * b_voigt[..., 0]
            + 2.0 * a_voigt[..., 1] * b_voigt[..., 1]
            + a_voigt[..., 2] * b_voigt[..., 2])


def degradation(d_qp, k_res: float):
    """退化函数 g(d) = (1-d)^2 + k_res（严格 > 0）。

    输入:
      d_qp  : (...,) qp 上相场/损伤值，∈[0,1]
      k_res : 残余刚度，必须 > 0
    输出:
      (...,) g(d)
    """
    if k_res <= 0:
        raise ValueError(f"k_res must be > 0 (g must not degenerate to 0); got {k_res}")
    one_minus_d = 1.0 - d_qp
    return one_minus_d * one_minus_d + k_res


# ----------------------------------------------------------------------------
# 核心：逐元平衡型误差指示子
# ----------------------------------------------------------------------------
def equilibrated_indicator(
    mesh,
    grad_uh_qp,
    sigmah_qp,
    d_qp,
    *,
    lam: float,
    mu: float,
    k_res: float,
    weights,
    cellmeasure,
):
    """逐元平衡型误差指示子 η_T（THEORY (5)）。

    所有场已在同一组求积点 (bcs) 上求值后传入（解耦 FE 空间细节，便于测试）。

    输入:
      mesh        : 三角网格（仅用于一致性校验，可为 None）
      grad_uh_qp  : (NC, NQ, 2, 2) 标准 FEM 位移梯度 ∇u_h
      sigmah_qp   : (NC, NQ, 3)    Hu–Zhang 平衡应力 σ_h，Voigt (xx,xy,yy)
      d_qp        : (NC, NQ)       相场/损伤（冻结系数）
      lam, mu     : Lamé 参数
      k_res       : 残余刚度 (>0)
      weights     : (NQ,)          求积权重 ws
      cellmeasure : (NC,)          单元面积 |T|
    输出 dict:
      eta_T : (NC,)  逐元指示子 η_T = sqrt(∫_T g^{-1} r:C^{-1}r)
      eta   : 标量   全局 η = sqrt(sum_T η_T^2)
    数学:
      r = C_d ε(u_h) - σ_h = g·C ε(u_h) - σ_h
      η_T^2 = ∫_T g^{-1} r : C^{-1} r
    """
    g = degradation(d_qp, k_res)                       # (NC, NQ)
    eps_voigt = strain_to_voigt(grad_uh_qp)            # (NC, NQ, 3)
    Ceps = stress_voigt_from_strain(eps_voigt, lam, mu)  # C ε(u_h)
    Cd_eps = g[..., None] * Ceps                       # C_d ε(u_h) = g C ε
    r = Cd_eps - sigmah_qp                             # 残差应力 (NC, NQ, 3)
    Cinv_r = compliance_apply_voigt(r, lam, mu)        # C^{-1} r
    integrand = voigt_inner(r, Cinv_r) / g             # g^{-1} r:C^{-1}r  (NC, NQ)

    # ∫_T (·) = |T| · Σ_q w_q (·)_q
    eta_T2 = cellmeasure * bm.einsum('q,cq->c', weights, integrand)
    # 数值噪声可能给极小负值，clip 到 0 再开方
    eta_T2 = bm.where(eta_T2 < 0, bm.zeros_like(eta_T2), eta_T2)
    eta_T = bm.sqrt(eta_T2)
    eta = float(bm.sqrt(bm.sum(eta_T2)))
    return {"eta_T": eta_T, "eta": eta}


# ----------------------------------------------------------------------------
# effectivity 工具（仅 M0 验证用，不进生产标记路径）
# ----------------------------------------------------------------------------
def energy_error_Cd(grad_uh_qp, grad_u_qp, d_qp, *, lam, mu, k_res, weights, cellmeasure):
    """真能量误差 ‖ε(u_h)-ε(u)‖_{C_d}（需解析/细网格参照解 u）。

    输入:
      grad_uh_qp : (NC,NQ,2,2) 数值解梯度
      grad_u_qp  : (NC,NQ,2,2) 参照解梯度
      其余同 equilibrated_indicator
    输出:
      标量 sqrt(∫ g (ε_h-ε):C:(ε_h-ε))
    """
    g = degradation(d_qp, k_res)
    de = strain_to_voigt(grad_uh_qp) - strain_to_voigt(grad_u_qp)
    Cde = stress_voigt_from_strain(de, lam, mu)        # C (ε_h-ε)
    integrand = g * voigt_inner(de, Cde)
    val = bm.sum(cellmeasure * bm.einsum('q,cq->c', weights, integrand))
    return float(bm.sqrt(bm.where(val < 0, bm.zeros_like(val), val)))


def effectivity_index(eta: float, energy_err: float, tol: float = 1e-300) -> float:
    """effectivity index Θ = η / ‖ε(u_h)-ε(u)‖_{C_d}（THEORY §5）。

    可靠性要求 Θ ≥ 1；Θ → 1 表示估计子尖锐。energy_err≈0 时返回 inf。
    """
    if energy_err <= tol:
        return float("inf")
    return eta / energy_err


# ----------------------------------------------------------------------------
# Amor 拉压分裂势与闭式对偶（THEORY §6.3, Theorem 2）
# ----------------------------------------------------------------------------
def _bulk_modulus_2d(lam: float, mu: float) -> float:
    """2D 平面应变体积模量 K = λ + μ。"""
    return lam + mu


def amor_energy(eps_voigt, d_qp, *, lam: float, mu: float, k_res: float):
    """Amor 分裂退化能量密度 ψ(ε,d)（THEORY §6.3）。

    ψ = g(½K⟨trε⟩₊² + μ|dev ε|²) + ½K⟨trε⟩₋²,  g=(1-d)²+k_res, K=λ+μ。
    输入:
      eps_voigt : (...,3) 应变 Voigt (xx,xy,yy)
      d_qp      : (...,)  相场
      lam,mu    : Lamé；k_res: 残余刚度 (>0)
    输出:
      (...,) 能量密度 ψ
    """
    g = degradation(d_qp, k_res)
    K = _bulk_modulus_2d(lam, mu)
    e_xx, e_xy, e_yy = eps_voigt[..., 0], eps_voigt[..., 1], eps_voigt[..., 2]
    tr = e_xx + e_yy
    tr_p = 0.5 * (tr + bm.abs(tr))
    tr_m = 0.5 * (tr - bm.abs(tr))
    # dev ε = ε - ½ tr I ; |dev|² = dev:dev (off-diag 双倍)
    dxx = e_xx - 0.5 * tr
    dyy = e_yy - 0.5 * tr
    dev2 = dxx * dxx + 2.0 * e_xy * e_xy + dyy * dyy
    return g * (0.5 * K * tr_p ** 2 + mu * dev2) + 0.5 * K * tr_m ** 2


def amor_stress(eps_voigt, d_qp, *, lam: float, mu: float, k_res: float):
    """Amor 分裂应力 σ=∂_ε ψ，Voigt (xx,xy,yy)（THEORY §6.1）。

    σ = (gK⟨trε⟩₊ + K⟨trε⟩₋) I + 2gμ dev ε。
    输入/输出同 amor_energy；输出 (...,3) 应力 Voigt。
    """
    g = degradation(d_qp, k_res)
    K = _bulk_modulus_2d(lam, mu)
    e_xx, e_xy, e_yy = eps_voigt[..., 0], eps_voigt[..., 1], eps_voigt[..., 2]
    tr = e_xx + e_yy
    tr_p = 0.5 * (tr + bm.abs(tr))
    tr_m = 0.5 * (tr - bm.abs(tr))
    p_iso = g * K * tr_p + K * tr_m          # 各向同性应力系数
    dxx = e_xx - 0.5 * tr
    dyy = e_yy - 0.5 * tr
    s_xx = p_iso + 2.0 * g * mu * dxx
    s_yy = p_iso + 2.0 * g * mu * dyy
    s_xy = 2.0 * g * mu * e_xy
    return bm.stack([s_xx, s_xy, s_yy], axis=-1)


def amor_dual_energy(sig_voigt, d_qp, *, lam: float, mu: float, k_res: float):
    """Amor 分裂对偶势 ψ*(τ,d) 闭式（THEORY (14)，T5 验机器精度）。

    ψ* = (1/(8K))[⟨p⟩₊²/g + ⟨p⟩₋²] + |dev τ|²/(4gμ),  p=tr τ, K=λ+μ。
    注意迹通道因子 1/(8K)（对偶变量是 ½ tr τ，Legendre 带出 1/4）。
    输入:
      sig_voigt : (...,3) 应力 Voigt (xx,xy,yy)
      d_qp      : (...,)  相场
    输出:
      (...,) 对偶能量密度 ψ*
    """
    g = degradation(d_qp, k_res)
    K = _bulk_modulus_2d(lam, mu)
    s_xx, s_xy, s_yy = sig_voigt[..., 0], sig_voigt[..., 1], sig_voigt[..., 2]
    p = s_xx + s_yy
    p_p = 0.5 * (p + bm.abs(p))
    p_m = 0.5 * (p - bm.abs(p))
    dxx = s_xx - 0.5 * p
    dyy = s_yy - 0.5 * p
    dev2 = dxx * dxx + 2.0 * s_xy * s_xy + dyy * dyy
    return (1.0 / (8.0 * K)) * (p_p ** 2 / g + p_m ** 2) + dev2 / (4.0 * g * mu)
