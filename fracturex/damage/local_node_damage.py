# fracturex/damage/local_node_damage.py
"""局部节点损伤模型（P1 节点损伤，非相场）。

基于等效应力准则（Rankine 最大主应力 / von Mises）的不可逆局部损伤演化：
``r_hist ← max(r_hist, r)``、``d = 1 - (ft/r)·exp(-2 Hd (r-ft)/ft)``，``d`` 单调不减。
弹性侧用退化系数 ``g(d)=(1-d)²`` 耦合到 Hu-Zhang 应力本构。

模块内自由函数为 Voigt↔矩阵转换与等效应力度量等工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Callable

import numpy as np
from fealpy.backend import backend_manager as bm


# -------- optional base class import --------
try:
    from fracturex.damage.base import DamageModelBase, DamageStateView

except Exception:  # pragma: no cover
    class DamageModelBase:  # minimal fallback
        """当 ``fracturex.damage.base`` 不可导入时的最小占位基类（仅声明接口方法）。"""
        def on_build(self, discr, state, case): ...
        def on_mesh_changed(self, old_discr, new_discr, old_state, new_state, case): ...
        def coef_bary(self, state, bcs, index=None): ...
        def update_after_elastic(self, discr, state, case): ...


def _material_lame_from_model(model) -> Tuple[float, float]:
    """
    Infer Lamé (lam, mu) from model.
    Supports:
      - model.lam, model.mu
      - model.lambda0, model.lambda1
      - model.E, model.nu
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
    raise AttributeError("Cannot infer (lam,mu): provide model.lam/mu or model.E/nu or model.lambda0/lambda1.")


def _call0_or_1arg(func: Callable, arg):
    """Support moderation_parameter() or moderation_parameter(mesh)."""
    try:
        return func()
    except TypeError:
        return func(arg)


def _voigt_to_mat(sig):
    """
    Convert Voigt to matrix.
    2D: [..., 3] -> [..., 2,2] with (xx, xy, yy)
    3D: [..., 6] -> [..., 3,3] with (xx, yy, zz, yz, xz, xy)  (common)
    NOTE: if your 3D ordering differs, adjust here.
    """
    if sig.shape[-1] == 3:
        sxx, sxy, syy = sig[..., 0], sig[..., 1], sig[..., 2]
        M = bm.zeros(sig.shape[:-1] + (2, 2), dtype=sig.dtype)
        M[..., 0, 0] = sxx
        M[..., 0, 1] = sxy
        M[..., 1, 0] = sxy
        M[..., 1, 1] = syy
        return M
    if sig.shape[-1] == 6:
        sxx, syy, szz, syz, sxz, sxy = (sig[..., 0], sig[..., 1], sig[..., 2],
                                        sig[..., 3], sig[..., 4], sig[..., 5])
        M = bm.zeros(sig.shape[:-1] + (3, 3), dtype=sig.dtype)
        M[..., 0, 0] = sxx
        M[..., 1, 1] = syy
        M[..., 2, 2] = szz
        M[..., 1, 2] = syz
        M[..., 2, 1] = syz
        M[..., 0, 2] = sxz
        M[..., 2, 0] = sxz
        M[..., 0, 1] = sxy
        M[..., 1, 0] = sxy
        return M
    raise ValueError(f"Unsupported Voigt length {sig.shape[-1]} (expect 3 or 6).")


def _rankine(sig_voigt):
    """Max principal stress from Voigt (2D/3D)."""
    M = _voigt_to_mat(sig_voigt)
    # eigvalsh supports batch for numpy; for bm backend it's typically numpy-compatible
    ev = bm.linalg.eigvalsh(M)
    return bm.max(ev, axis=-1)


def _von_mises(sig_voigt):
    """Von Mises from Voigt: 2D (plane stress/strain style) or 3D."""
    if sig_voigt.shape[-1] == 3:
        sxx, sxy, syy = sig_voigt[..., 0], sig_voigt[..., 1], sig_voigt[..., 2]
        return bm.sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)
    if sig_voigt.shape[-1] == 6:
        sxx, syy, szz, syz, sxz, sxy = (sig_voigt[..., 0], sig_voigt[..., 1], sig_voigt[..., 2],
                                        sig_voigt[..., 3], sig_voigt[..., 4], sig_voigt[..., 5])
        # J2 formula
        s1 = (sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2
        s2 = 6.0*(sxy**2 + syz**2 + sxz**2)
        return bm.sqrt(0.5*s1 + s2)
    raise ValueError(f"Unsupported Voigt length {sig_voigt.shape[-1]} for von Mises.")


@dataclass
class LocalNodeDamage(DamageModelBase):
    """
    Local nodal damage (P1 on nodes).

    Damage law (your current one):
        rr = max(r_hist, ft)
        d = 1 - exp( -2*Hd*(rr-ft)/ft )

    Notes:
    - r_hist is irreversible: r_hist <- max(r_hist, r)
    - d is irreversible: d <- max(d, d_new)
    - criterion can be "rankine" or "von_mises"
    - Hd comes from model.moderation_parameter() (your current signature)
    """
    criterion: str = "rankine"
    eps_g: float = 1e-6
    clamp_max: float = 0.999999
    tensile_only: bool = True
    debug: bool = False

    # filled in on_build
    ft: Optional[float] = None
    Hd: Optional[float] = None
    lam: Optional[float] = None
    mu: Optional[float] = None

    def on_build(self, discr, state, case):
        """Initialize local damage parameters and state fields.

        Inputs:
            discr: Discretization object containing mesh/spaces.
            state: Damage state view (`d`, `r_hist`, `u`, `sigma`).
            case: Case object that provides material model.
        Output:
            None. Updates `self.ft/self.Hd` and initializes state history values.
        """
        model = case.model()
        self.lam, self.mu = _material_lame_from_model(model)

        if self.ft is None:
            self.ft = float(getattr(model, "ft"))
        if self.Hd is None:
            if not hasattr(model, "moderation_parameter"):
                raise AttributeError("LocalNodeDamage requires model.moderation_parameter().")
            self.Hd = float(_call0_or_1arg(model.moderation_parameter, discr.mesh))

        # initialize histories if present
        if hasattr(state, "r_hist") and state.r_hist is not None:
            state.r_hist[:] = 0.0
        # ensure damage starts at 0
        if hasattr(state, "d") and state.d is not None:
            state.d[:] = bm.maximum(state.d[:], 0.0)

        if self.debug:
            print(f"[LocalNodeDamage] on_build: ft={self.ft}, Hd={self.Hd}, lam={self.lam}, mu={self.mu}")

    def on_mesh_changed(self, old_discr, new_discr, old_state, new_state, case):
        """
        If your Hd/ft depend on mesh later, you already have the right hook.
        Currently moderation_parameter() is mesh-independent, but recomputing is cheap and safe.
        """
        model = case.model()
        self.lam, self.mu = _material_lame_from_model(model)
        self.ft = float(getattr(model, "ft"))
        self.Hd = float(_call0_or_1arg(model.moderation_parameter, new_discr.mesh))

        if self.debug:
            print(f"[LocalNodeDamage] on_mesh_changed: ft={self.ft}, Hd={self.Hd}")

    # --- degradation function g(d) used in sigma-bilinear form ---
    def coef_bary(self, state, bcs, index=None):
        """
        g(d) on (cell, quad) from nodal damage field.
        Assumes state.d is a FEALPy Function that supports barycentric evaluation.
        """
        dval = state.d(bcs, index=index)  # <- 正确方式

        # 保证数值范围
        dval = bm.clip(dval, 0.0, self.clamp_max)

        # 退化函数（推荐二次）
        if getattr(self, "degradation", "quadratic") == "linear":
            g = 1.0 - dval
        else:
            g = (1.0 - dval) ** 2

        if getattr(self, "debug", False):
            print("[coef_bary] g min/max:", float(bm.min(g)), float(bm.max(g)))
        return g + self.eps_g
    
    def characteristic_length(self):
        """Return characteristic length `lch = Gc*E/ft^2`.

        Input:
            self: Damage model with `Gc`, `E`, `ft`.
        Output:
            Characteristic length scalar.
        """
        return self.Gc*self.E/(self.ft**2)

    def moderation_parameter(self):
        """Return moderation parameter from local damage model constants.

        Input:
            self: Damage model with `l0` and `characteristic_length`.
        Output:
            Scalar moderation parameter in damage evolution law.
        """
        lch = self.characteristic_length()
        return self.l0/(2*lch + self.l0)
    

    def update_after_elastic(self, discr, state, case):
        """Update nodal history and damage after elastic solve.

        Inputs:
            discr: Discretization containing mesh/space information.
            state: State with current `u`, `d`, `r_hist`.
            case: Case object (reserved for extensions).
        Output:
            Updated nodal damage field `state.d`.
        """
        mesh = discr.mesh
        NC = mesh.number_of_cells()
        qf = mesh.quadrature_formula(discr.p + 4, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()  # (NQ,3), (NQ,)

        index = bm.arange(NC)

        # -------------------------------------------------
        # (1) 从位移梯度计算应变 eps(u)
        # grad_u: (NC,NQ,GD,GD), grad_u[..., i, j] = ∂u_i/∂x_j
        # -------------------------------------------------
        grad_u = state.u.grad_value(bcs, index=index)

        exx = grad_u[..., 0, 0]
        eyy = grad_u[..., 1, 1]
        exy = 0.5 * (grad_u[..., 0, 1] + grad_u[..., 1, 0])  # tensor shear strain

        # -------------------------------------------------
        # (2) 计算“完好材料应力” bar_sigma = C:eps(u)
        # Voigt = [xx, xy, yy], with sigma_xy = 2 mu eps_xy
        # -------------------------------------------------
        lam, mu = float(self.lam), float(self.mu)
        tr = exx + eyy

        sxx = 2.0 * mu * exx + lam * tr
        syy = 2.0 * mu * eyy + lam * tr
        sxy = 2.0 * mu * exy

        # -------------------------------------------------
        # (3) Rankine 等效应力 r_q (NC,NQ)
        # -------------------------------------------------
        s1 = 0.5 * (sxx + syy) + bm.sqrt((0.5 * (sxx - syy))**2 + sxy**2)
        r_q = bm.maximum(s1, 0.0) if self.tensile_only else s1

        # -------------------------------------------------
        # (4) (cell,quad) -> cell 平均 -> node（面积加权散射）
        # -------------------------------------------------
        cell = mesh.entity("cell")          # (NC,3)
        area = mesh.entity_measure("cell")  # (NC,)

        # 用 quadrature 做单元平均（你原来除 bm.sum(ws) 的写法也可以）
        r_cell = bm.einsum('q,cq->c', ws, r_q) / bm.sum(ws)  # (NC,)

        NN = mesh.number_of_nodes()
        r_node = bm.zeros((NN,), dtype=r_cell.dtype)
        w_node = bm.zeros((NN,), dtype=r_cell.dtype)

        w = area / 3.0
        for lv in range(3):
            nid = cell[:, lv]
            bm.add.at(r_node, nid, r_cell * w)
            bm.add.at(w_node, nid, w)

        r_node = r_node / w_node

        # -------------------------------------------------
        # (5) history + damage update（你文档那条）
        # -------------------------------------------------
        ft = float(self.ft)
        Hd = float(self.Hd)

        r = bm.maximum(r_node, ft)
        state.r_hist[:] = bm.maximum(state.r_hist[:], r)

        rh = state.r_hist[:]
        dnew = 1.0 - (ft / rh) * bm.exp(-2.0 * Hd * (rh - ft) / ft)
        dnew = bm.clip(dnew, 0.0, self.clamp_max)
        state.d[:] = bm.maximum(state.d[:], dnew)

        if self.debug:
            print("[LocalNodeDamage-u] r_node min/max:",
                float(bm.min(r_node)), float(bm.max(r_node)),
                "d min/max:", float(bm.min(state.d[:])), float(bm.max(state.d[:])))

        return state.d

    '''
    # --- main update: after solving (sigma,u) ---
    def update_after_elastic(self, discr, state, case):
        mesh = discr.mesh
        NC = mesh.number_of_cells()
        qf = mesh.quadrature_formula(discr.p+4, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()  # (NQ,3), (NQ,)

        # 1) 在(cell,quad)评估退化应力 sigma
        sig = state.sigma(bcs, index=bm.arange(NC))        # (NC,NQ,3)

        # 2) 在(cell,quad)评估 d，并构造 g(d)
        dval = state.d(bcs, index=bm.arange(NC))           # (NC,NQ)
        dval = bm.clip(dval, 0.0, self.clamp_max)

        # 退化函数（推荐二次）
        if getattr(self, "degradation", "quadratic") == "linear":
            g = 1.0 - dval + self.eps_g
        else:
            g = (1.0 - dval) ** 2 + self.eps_g

        # 3) 恢复“完好材料应力” bar_sigma = sigma/g
        sig_eff = sig / g[..., None]                       # (NC,NQ,3)

        # 4) 计算等效应力（Rankine 举例；Mises也行）
        sxx, sxy, syy = sig_eff[...,0], sig_eff[...,1], sig_eff[...,2]
        s1 = 0.5*(sxx+syy) + bm.sqrt((0.5*(sxx-syy))**2 + sxy**2)  # max principal stress (NC,NQ)
        if self.tensile_only:
            r_q = bm.maximum(s1, 0.0)
        else:
            r_q = s1



        # 5) 把 r_q 投影到节点（面积加权的 cell-average 再 scatter 到3个顶点）
        cell = mesh.entity("cell")               # (NC,3)
        area = mesh.entity_measure("cell")       # (NC,)
        r_cell = bm.einsum('q,cq->c', ws, r_q) / bm.sum(ws)   # (NC,) 先做单元平均

        NN = mesh.number_of_nodes()
        r_node = bm.zeros((NN,), dtype=r_cell.dtype)
        w_node = bm.zeros((NN,), dtype=r_cell.dtype)
        w = area/3.0
        for lv in range(3):
            nid = cell[:, lv]
            bm.add.at(r_node, nid, r_cell*w)
            bm.add.at(w_node, nid, w)
        r_node = r_node / w_node


        # 6) history + damage update（与你文档一致）
        r = bm.maximum(r_node, self.ft)               # r_node 是等效应力（未退化）
        state.r_hist[:] = bm.maximum(state.r_hist[:], r)

        rh = state.r_hist[:]
        dnew = 1.0 - (self.ft / rh) * bm.exp(-2.0 * self.Hd * (rh - self.ft) / self.ft)
        dnew = bm.clip(dnew, 0.0, self.clamp_max)
        state.d[:] = bm.maximum(state.d[:], dnew)  # 不可逆

        return state.d
    '''


    # -----------------------
    # internal helpers
    # -----------------------
    def _eval_sigma_at_nodes(self, sigma_obj, mesh):
        """
        在全局节点处评估 sigma（Voigt: [xx, xy, yy]），做法：
        - 在每个单元的三个顶点 barycentric 点上评估 sigma
        - 对共享节点做简单平均

        返回: (NN, 3)
        """
        cell = mesh.entity("cell")            # (NC, 3)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        # 三角形三个顶点的 barycentric 坐标
        bcs = bm.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]], dtype=bm.float64)  # (3,3)

        # HuZhang Function 的 __call__ 期望的是 barycentric bcs
        # 期望返回形状 (NC, 3, 3) : 对每个 cell、每个顶点、Voigt 3分量
        vals = sigma_obj(bcs, index=bm.arange(NC))  # (NC, 3, 3) 或 (NC, 3, ?)

        # 兼容一下：如果返回 (NC,3) 说明少了“顶点维”，那就当作常值复制
        if vals.ndim == 2:
            vals = vals[:, None, :].repeat(3, axis=1)

        out = bm.zeros((NN, vals.shape[-1]), dtype=vals.dtype)
        cnt = bm.zeros((NN,), dtype=vals.dtype)

        # 三个局部顶点累加到全局节点
        for lv in range(3):
            nid = cell[:, lv]                # (NC,)
            bm.add.at(out, nid, vals[:, lv, :])
            bm.add.at(cnt, nid, 1.0)

        out = out / cnt[:, None]
        return out


    def _equivalent_measure(self, sig_voigt):
        """Dispatch equivalent stress measure by configured criterion.

        Input:
            sig_voigt: Stress in Voigt form.
        Output:
            Equivalent scalar stress array (Rankine or von Mises).
        """
        c = (self.criterion or "rankine").lower()
        if c in ("rankine", "max_principal_stress", "s1"):
            return _rankine(sig_voigt)
        if c in ("von_mises", "vmises", "mises"):
            return _von_mises(sig_voigt)
        raise ValueError(f"Unknown LocalNodeDamage.criterion={self.criterion}")


