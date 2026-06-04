# fracturex/damage/phasefield_damage.py
"""Hu-Zhang 框架下的混合（hybrid）相场损伤模型。

本模型把损伤 ``d`` 作为 ``space_d`` 上的有限元函数，历史场 ``H`` 作为积分点场
``(NC, NQ)`` 存于 ``state.H``；弹性退化用各向同性 ``g(d)``，历史场驱动力按谱分解
（spectral）/混合（hybrid）/各向同性（isotropic）三种能量分裂之一计算。

裂纹面密度（AT1/AT2 等）与退化函数分别委托给 ``CrackSurfaceDensityFunction`` 与
``EnergyDegradationFunction``；本类只负责装配端需要的系数评估与历史场更新。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from fealpy.backend import backend_manager as bm

from fracturex.damage.base import DamageModelBase
from fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction
from fracturex.phasefield.crack_surface_density_function import CrackSurfaceDensityFunction


def _material_lame_from_model(model) -> Tuple[float, float]:
    """从材料模型解析 Lamé 参数 ``(lam, mu)``。

    Args:
        model: 提供 ``(lam,mu)`` 或 ``(lambda0,lambda1)`` 或 ``(E,nu)`` 的对象或 dict。
    Returns:
        ``(lam, mu)`` 浮点元组。
    Raises:
        AttributeError: 无法从 model 推断出参数。
    """
    if hasattr(model, "lam") and hasattr(model, "mu"):
        return float(model.lam), float(model.mu)
    if hasattr(model, "lambda0") and hasattr(model, "lambda1"):
        return float(model.lambda0), float(model.lambda1)
    if hasattr(model, "E") and hasattr(model, "nu"):
        E = float(model.E)
        nu = float(model.nu)
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lam, mu
    if isinstance(model, dict):
        if "lam" in model and "mu" in model:
            return float(model["lam"]), float(model["mu"])
        if "lambda0" in model and "lambda1" in model:
            return float(model["lambda0"]), float(model["lambda1"])
        if "E" in model and "nu" in model:
            E = float(model["E"])
            nu = float(model["nu"])
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return lam, mu
    raise AttributeError("Cannot infer (lam,mu).")


def _model_get(model, name: str, default=None):
    """从对象属性或 dict 键读取 ``name``，缺失则返回 ``default``。"""
    if hasattr(model, name):
        return getattr(model, name)
    if isinstance(model, dict) and name in model:
        return model[name]
    return default


@dataclass
class PhaseFieldDamageModel(DamageModelBase):
    """
    Hybrid phase-field model for Hu-Zhang framework.

    Final design:
      - d : FE function in discr.space_d
      - H : quadrature-history field, shape (NC, NQ), stored in state.H
      - elasticity degradation: isotropic g(d)
      - history update: spectral / hybrid on quadrature points
    """
    density_type: str = "AT2"
    degradation_type: str = "quadratic"
    split: str = "hybrid"         # "hybrid", "spectral", "isotropic"
    eps_g: float = 1e-10
    clamp_max: float = 0.999999
    debug: bool = False

    lam: Optional[float] = None
    mu: Optional[float] = None
    Gc: Optional[float] = None
    l0: Optional[float] = None

    _gfun: Optional[Any] = None
    _hfun: Optional[Any] = None

    def on_build(self, discr, state, case):
        """初始化材料/断裂参数与退化、裂纹密度函数；裁剪初始 ``d`` 并清空历史场 ``H``。

        Args:
            discr: 离散化对象。
            state: 当前状态（其 ``d`` 被就地裁剪，``H`` 置 None）。
            case: 提供 ``model()``（含 ``Gc``、``l0`` 及 Lamé 参数）。
        """
        model = case.model()

        self.lam, self.mu = _material_lame_from_model(model)

        Gc = _model_get(model, "Gc", None)
        l0 = _model_get(model, "l0", None)
        if Gc is None:
            raise AttributeError("PhaseFieldDamageModel requires model.Gc.")
        if l0 is None:
            raise AttributeError("PhaseFieldDamageModel requires model.l0.")

        self.Gc = float(Gc)
        self.l0 = float(l0)

        self._gfun = EnergyDegradationFunction(self.degradation_type)
        self._hfun = CrackSurfaceDensityFunction(self.density_type)

        if state.d is not None:
            state.d[:] = bm.clip(state.d[:], 0.0, self.clamp_max)

        # NEW: H is quadrature-history, not FE function
        state.H = None

        if self.debug:
            print(
                f"[PhaseFieldDamageModel] on_build: "
                f"Gc={self.Gc}, l0={self.l0}, lam={self.lam}, mu={self.mu}, "
                f"density={self.density_type}, degradation={self.degradation_type}, split={self.split}"
            )

    def on_mesh_changed(self, old_discr, new_discr, old_state, new_state, case):
        """网格变更（自适应加密）后刷新材料/断裂参数并重置历史场 ``new_state.H=None``。

        Args:
            old_discr, new_discr: 变更前后的离散化对象。
            old_state, new_state: 变更前后的状态（重置 ``new_state.H``）。
            case: 提供 ``model()``。
        """
        model = case.model()
        self.lam, self.mu = _material_lame_from_model(model)

        Gc = _model_get(model, "Gc", None)
        l0 = _model_get(model, "l0", None)
        if Gc is None or l0 is None:
            raise AttributeError("PhaseFieldDamageModel requires model.Gc and model.l0 after mesh change.")
        self.Gc = float(Gc)
        self.l0 = float(l0)

        # safest choice for quadrature-history after mesh change:
        new_state.H = None

        if self.debug:
            print(
                f"[PhaseFieldDamageModel] on_mesh_changed: "
                f"Gc={self.Gc}, l0={self.l0}, lam={self.lam}, mu={self.mu}"
            )

    # ------------------------------------------------------------
    # elasticity degradation
    # ------------------------------------------------------------
    def coef_bary(self, state, bcs, index=None):
        """在重心坐标 ``bcs`` 处求弹性退化系数 ``g(d)``（带下限 ``eps_g`` 防零）。

        Args:
            state: 状态视图（提供 ``d``）。
            bcs: 重心坐标。
            index: 可选 cell 索引。
        Returns:
            退化系数数组 ``(NC, NQ)``（或与 bcs 评估匹配的形状）。
        """
        dval = state.d(bcs, index=index)
        dval = bm.clip(dval, 0.0, self.clamp_max)

        gd = self._gfun.degradation_function(dval)
        gd = bm.maximum(gd, self.eps_g)

        if self.debug:
            print(
                "[PhaseFieldDamageModel.coef_bary] "
                f"g min/max = {float(bm.min(gd)):.3e} / {float(bm.max(gd)):.3e}"
            )
        return gd

    # ------------------------------------------------------------
    # crack density / degradation helpers
    # ------------------------------------------------------------
    def crack_density(self, d):
        """裂纹面密度 ``c(d)``（委托裂纹密度函数）。输入损伤 ``d``，返回密度值。"""
        return self._hfun.density_function(d)

    def crack_density_grad(self, d):
        """裂纹面密度一阶导 ``c'(d)``。输入 ``d``，返回梯度。"""
        return self._hfun.grad_density_function(d)

    def crack_density_hess(self, d):
        """裂纹面密度二阶导，返回 ``(c'(d), c''(d))`` 或实现约定的元组。输入 ``d``。"""
        return self._hfun.grad_grad_density_function(d)

    def degradation(self, d):
        """能量退化函数 ``g(d)``（委托退化函数）。输入 ``d``，返回 ``g``。"""
        return self._gfun.degradation_function(d)

    def degradation_grad(self, d):
        """退化函数一阶导 ``g'(d)``。输入 ``d``，返回梯度。"""
        return self._gfun.grad_degradation_function(d)

    def degradation_hess(self, d):
        """退化函数二阶导 ``g''(d)``。输入 ``d``，返回 Hessian。"""
        return self._gfun.grad_grad_degradation_function(d)

    # ------------------------------------------------------------
    # quadrature-history update
    # ------------------------------------------------------------
    def update_history_on_quadrature(self, discr, state, case, bcs, index=None):
        """
        Update H on the given quadrature points.

        Parameters
        ----------
        bcs : quadrature barycentric points
        index : cell indices
        """
        grad_u = state.u.grad_value(bcs, index=index)               # (NC,NQ,GD,GD)
        strain = 0.5 * (grad_u + bm.swapaxes(grad_u, -2, -1))

        if self.split in ("hybrid", "spectral"):
            phip = self._spectral_positive_energy_density(strain)   # (NC,NQ)
        elif self.split == "isotropic":
            phip = self._isotropic_energy_density(strain)
            phip = bm.maximum(phip, 0.0)
        else:
            raise ValueError(f"Unknown split type: {self.split}")

        phip = bm.maximum(phip, 0.0)

        if state.H is None:
            state.H = phip.copy()
        else:
            if state.H.shape != phip.shape:
                raise ValueError(
                    f"Quadrature-history shape mismatch: state.H has {state.H.shape}, "
                    f"but current phip has {phip.shape}. "
                    f"Make sure assembler uses the same quadrature rule when updating/reading H."
                )
            state.H = bm.maximum(state.H, phip)

        if self.debug:
            print(
                "[PhaseFieldDamageModel.update_history_on_quadrature] "
                f"H min/max = {float(bm.min(state.H)):.3e} / {float(bm.max(state.H)):.3e}"
            )
        return state.H

    # backward-compatible alias
    def update_after_elastic(self, discr, state, case):
        """已弃用的旧接口：积分点历史场版本不再使用，调用即抛 ``RuntimeError``。

        请改用 :meth:`update_history_on_quadrature`（由 PhaseFieldAssembler 传入 bcs/index）。
        """
        raise RuntimeError(
            "Quadrature-history version does not use update_after_elastic(discr,state,case) "
            "directly. Call update_history_on_quadrature(..., bcs, index) from PhaseFieldAssembler."
        )

    # ------------------------------------------------------------
    # energy density
    # ------------------------------------------------------------
    def _isotropic_energy_density(self, strain):
        """各向同性（不分裂）弹性能密度 ``ψ = ½λ tr(ε)² + μ ε:ε``。

        Args:
            strain: 应变张量 ``(NC, NQ, GD, GD)``。
        Returns:
            能量密度 ``(NC, NQ)``。
        """
        lam = float(self.lam)
        mu = float(self.mu)
        tr = bm.einsum("...ii", strain)
        e2 = bm.einsum("...ij,...ij->...", strain, strain)
        psi = 0.5 * lam * tr**2 + mu * e2
        return psi

    def _spectral_positive_energy_density(self, strain):
        """谱分解正能量密度 ``ψ⁺ = ½λ⟨tr ε⟩₊² + μ tr(ε₊²)``（Miehe 分裂）。

        Args:
            strain: 应变张量 ``(NC, NQ, GD, GD)``。
        Returns:
            正能量密度 ``(NC, NQ)``。
        """
        lam = float(self.lam)
        mu = float(self.mu)

        eps_p, _ = self._strain_pm_eig_decomposition(strain)
        tr = bm.einsum("...ii", strain)
        tr_p, _ = self._macaulay_operation(tr)
        epp2 = bm.einsum("...ii", eps_p @ eps_p)

        psi_plus = 0.5 * lam * tr_p**2 + mu * epp2
        return psi_plus

    def _macaulay_operation(self, alpha):
        """Macaulay 括号：返回 ``(⟨α⟩₊, ⟨α⟩₋)`` 即正/负部。输入标量或数组 ``alpha``。"""
        val = bm.abs(alpha)
        p = 0.5 * (alpha + val)
        m = 0.5 * (alpha - val)
        return p, m

    def _strain_pm_eig_decomposition(self, s):
        """应变的正/负部谱分解 ``(ε₊, ε₋)``。

        2D 走闭式 :meth:`_strain_pm_2x2`，其余维度走对称特征分解（``eigh``）回退。

        Args:
            s: 对称应变张量 ``(..., GD, GD)``。
        Returns:
            ``(sp, sm)``：与 ``s`` 同形状的正、负部张量。
        """
        GD = int(s.shape[-1])
        if GD == 2:
            return self._strain_pm_2x2(s)

        # General fallback (e.g. 3D Hu-Zhang): symmetric eigendecomposition.
        w, v = bm.linalg.eigh(s)
        p, m = self._macaulay_operation(w)

        sp = bm.zeros_like(s)
        sm = bm.zeros_like(s)
        for i in range(GD):
            ni = v[..., i]

            nip = p[..., i, None] * ni
            sp = sp + nip[..., None] * ni[..., None, :]

            nim = m[..., i, None] * ni
            sm = sm + nim[..., None] * ni[..., None, :]

        return sp, sm

    def _strain_pm_2x2(self, s):
        """Closed-form ±-split of a symmetric 2×2 strain via spectral projectors.

        For ``s = λ1 P1 + λ2 P2`` with ``P1 + P2 = I`` and ``P1 = (s - λ2 I)/(λ1 - λ2)``,
        the positive/negative parts are ``s± = ⟨λ1⟩± P1 + ⟨λ2⟩± P2``. ``s±`` is
        basis-independent, so this matches the ``eigh`` reconstruction (and is better
        conditioned near eigenvalue coalescence), while avoiding ``eigh`` and the
        per-component reconstruction loop. ``λ1,2 = mean ± R``, ``R = √(diff² + c²)``.
        """
        a = s[..., 0, 0]
        b = s[..., 1, 1]
        c = 0.5 * (s[..., 0, 1] + s[..., 1, 0])   # symmetrize defensively
        mean = 0.5 * (a + b)
        diff = 0.5 * (a - b)
        R = bm.sqrt(diff * diff + c * c)
        l1 = mean + R
        l2 = mean - R

        p1 = bm.maximum(l1, 0.0)
        p2 = bm.maximum(l2, 0.0)
        m1 = bm.minimum(l1, 0.0)
        m2 = bm.minimum(l2, 0.0)

        twoR = 2.0 * R
        nondeg = twoR > 0.0
        # 1/(λ1 - λ2) = 1/(2R); the inner guard only avoids a 1/0 warning — the
        # exact-degenerate entries (R = 0) are overwritten by the `deg` branch below.
        inv = bm.where(nondeg, 1.0 / bm.where(nondeg, twoR, 1.0), 0.0)

        # P1 = (s - λ2 I)/(2R); P2 = I - P1
        P1_00 = (a - l2) * inv
        P1_11 = (b - l2) * inv
        P1_01 = c * inv
        P2_00 = 1.0 - P1_00
        P2_11 = 1.0 - P1_11
        P2_01 = -P1_01

        sp_00 = p1 * P1_00 + p2 * P2_00
        sp_11 = p1 * P1_11 + p2 * P2_11
        sp_01 = p1 * P1_01 + p2 * P2_01
        sm_00 = m1 * P1_00 + m2 * P2_00
        sm_11 = m1 * P1_11 + m2 * P2_11
        sm_01 = m1 * P1_01 + m2 * P2_01

        # Degenerate λ1 ≈ λ2 (R = 0 ⇒ s = mean·I): s± = ⟨mean⟩± I.
        deg = ~nondeg
        mp = bm.maximum(mean, 0.0)
        mm = bm.minimum(mean, 0.0)
        zero = bm.zeros_like(sp_01)
        sp_00 = bm.where(deg, mp, sp_00)
        sp_11 = bm.where(deg, mp, sp_11)
        sp_01 = bm.where(deg, zero, sp_01)
        sm_00 = bm.where(deg, mm, sm_00)
        sm_11 = bm.where(deg, mm, sm_11)
        sm_01 = bm.where(deg, zero, sm_01)

        sp = bm.stack([bm.stack([sp_00, sp_01], axis=-1),
                       bm.stack([sp_01, sp_11], axis=-1)], axis=-2)
        sm = bm.stack([bm.stack([sm_00, sm_01], axis=-1),
                       bm.stack([sm_01, sm_11], axis=-1)], axis=-2)
        return sp, sm

    # ------------------------------------------------------------
    # params
    # ------------------------------------------------------------
    def regularization_constant(self):
        """裂纹密度归一化常数 ``c_w``（由 ``crack_density(0)`` 得到）。返回浮点。"""
        _, c_d = self.crack_density(0.0)
        return float(c_d)

    def length_scale(self):
        """相场长度尺度 ``l0``，返回浮点。"""
        return float(self.l0)

    def fracture_toughness(self):
        """断裂韧度 ``Gc``，返回浮点。"""
        return float(self.Gc)