# fracturex/damage/local_node_damage.py

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
    eps_g: float = 1e-15
    clamp_max: float = 0.999999
    tensile_only: bool = True
    debug: bool = False

    # filled in on_build
    ft: Optional[float] = None
    Hd: Optional[float] = None
    lam: Optional[float] = None
    mu: Optional[float] = None

    def on_build(self, discr, state, case):
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

    # --- main update: after solving (sigma,u) ---
    def update_after_elastic(self, discr, state, case):
        mesh = discr.mesh

        # evaluate sigma at nodes
        sig = self._eval_sigma_at_nodes(state.sigma, mesh)


        r = self._equivalent_measure(sig)
        if self.tensile_only:
            r = bm.maximum(r, 0.0)

        # update history
        if state.r_hist is None:
            raise RuntimeError("State must provide r_hist for LocalNodeDamage.")
        state.r_hist[:] = bm.maximum(state.r_hist[:], r)

        ft = float(self.ft)
        Hd = float(self.Hd)

        x = bm.maximum(state.r_hist[:] - ft, 0.0)
        dnew = 1.0 - bm.exp(-2.0 * Hd * x / ft)

        dnew = bm.clip(dnew, 0.0, self.clamp_max)

        # irreversibility
        state.d[:] = bm.maximum(state.d[:], dnew)

        if self.debug:
            print(
                f"[LocalNodeDamage] max r={float(bm.max(r)):.4e}, "
                f"max r_hist={float(bm.max(state.r_hist[:])):.4e}, "
                f"min/max d=({float(bm.min(state.d[:])):.4e}, {float(bm.max(state.d[:])):.4e})"
            )
        return state.d

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
        c = (self.criterion or "rankine").lower()
        if c in ("rankine", "max_principal_stress", "s1"):
            return _rankine(sig_voigt)
        if c in ("von_mises", "vmises", "mises"):
            return _von_mises(sig_voigt)
        raise ValueError(f"Unknown LocalNodeDamage.criterion={self.criterion}")
