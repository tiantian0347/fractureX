# fracturex/assemblers/huzhang_elastic_assembler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any

import numpy as np
from scipy.sparse import bmat

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian, barycentric

from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import VectorSourceIntegrator

from fracturex.cases.base import CaseBase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.base import DamageModelBase, DamageStateView
from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition,
    HuzhangStressBoundaryCondition,
)


@dataclass
class ElasticSystem:
    A: Any
    F: Any
    decode: Callable[[Any], Tuple[Any, Any]]  # X -> (sigma_fun, u_fun)
    meta: dict


class HuZhangElasticAssembler:
    """
    装配 HuZhang 混合线弹性系统：
      [ M(d)  B ]
      [ B^T   0 ]
    并统一处理 corner relaxation: M2=TM^T M TM, B2=TM^T B

    同时支持 piecewise 位移边界贡献到 σ 方程 RHS：
      F_sigma += TM^T * r_dirichlet
    """

    def __init__(self, discr: HuZhangDiscretization, case: CaseBase, damage: DamageModelBase, *, q: Optional[int] = None):
        self.discr = discr
        self.case = case
        self.damage = damage
        self.q = q  # 若 None，integrator 内部用默认

    def assemble(self, load: float) -> ElasticSystem:
        discr = self.discr
        case = self.case
        damage = self.damage
        mesh = discr.mesh

        space0 = discr.space_sigma   # σ
        space1 = discr.space_u       # u
        state = discr.state

        assert mesh is not None and space0 is not None and space1 is not None and state is not None

        gdof0 = space0.number_of_global_dofs()
        gdof1 = space1.number_of_global_dofs()

        # ---- 1) M(d) ----
        # 退化系数：在单元积分点上评估 g(d)
        @barycentric
        def coef_d(bcs, index=None):
            # DamageStateView 用于统一接口
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
            return damage.coef_bary(view, bcs, index=index)

        lam, mu = self._lame(case.model())

        bformM = BilinearForm(space0)
        bformM.add_integrator(HuZhangStressIntegrator(coef=coef_d, lambda0=lam, lambda1=mu))
        M = bformM.assembly().to_scipy().tocsr()

        # ---- 2) B ----
        bformB = BilinearForm((space1, space0))
        bformB.add_integrator(HuZhangMixIntegrator())
        B = bformB.assembly().to_scipy().tocsr()

        # ---- 3) TM transform ----
        TM = space0.TM.to_scipy().tocsr()
        M2 = TM.T @ M @ TM
        B2 = TM.T @ B

        A = bmat([[M2, B2],
                  [B2.T, None]], format="csr")

        # ---- 4) RHS: body force on u eqn ----
        lform = LinearForm(space1)

        @cartesian
        def f_body(x, index=None):
            return case.body_force(x)

        lform.add_integrator(VectorSourceIntegrator(source=f_body))
        b = lform.assembly()  # (gdof1,)
        b = np.asarray(b, dtype=float).reshape(-1)

        # ---- 5) RHS: Dirichlet displacement contributes to sigma equation ----
        HBC = HuzhangBoundaryCondition(space=space0, q=self.q)
        pieces = case.dirichlet_pieces(load)

        # 组装 piecewise
        piecewise = [(pc.threshold, pc.value, pc.direction) for pc in pieces]
        r_dir = HBC.displacement_boundary_condition(piecewise=piecewise)

        F = np.zeros(A.shape[0], dtype=float)
        F[:gdof0] = (TM.T @ r_dir).reshape(-1)   # 注意：sigma unknown 是 tilde 变量
        F[gdof0:] = -b

        # ---- 6) optional: essential stress/traction boundary on sigma (Neumann edges but essential on sigma dof) ----
        # 如果 case.neumann_data 返回 (gd, threshold, coord)，就做消元
        nd = case.neumann_data(load)
        if nd is not None:
            gd, thr, coord = nd
            HSBC = HuzhangStressBoundaryCondition(space=space0, q=self.q)

            # set_essential_bc returns (uh_sigma, isbddof_sigma) in tilde-space or physical?
            # 约定：HSBC.set_essential_bc 作用在 space0 的“当前自由度编号”（即与 TM 前一致）
            uh_sig, isBd = HSBC.set_essential_bc(gd, threshold=thr, coord=coord)

            # 消元到全系统（只作用在 sigma 块）
            A, F = self.apply_sigma_essential_to_system(A, F, uh_sig, isBd, gdof0)

        # decode: map solution X -> (sigma,u) functions
        def decode(X):
            X = np.asarray(X).reshape(-1)
            sig_tilde = X[:gdof0]
            u_vec = X[gdof0:]

            sigma = space0.function()
            sigma[:] = (TM @ sig_tilde).reshape(-1)

            u = space1.function()
            u[:] = u_vec
            return sigma, u

        meta = dict(gdof_sigma=int(gdof0), gdof_u=int(gdof1))
        return ElasticSystem(A=A, F=F, decode=decode, meta=meta)

    @staticmethod
    def apply_sigma_essential_to_system(A, F, uh_sigma, isBd_sigma, gdof_sigma: int):
        """
        把 σ 的本质边界值（uh_sigma, isBd_sigma）扩展到全系统并消元。
        这段就是你之前脚本里那段通用逻辑的封装版。
        """
        from scipy.sparse import spdiags

        total = A.shape[0]
        uh_global = np.zeros(total, dtype=float)
        isbd_global = np.zeros(total, dtype=bool)

        uh_global[:gdof_sigma] = np.asarray(uh_sigma).reshape(-1)
        isbd_global[:gdof_sigma] = np.asarray(isBd_sigma).reshape(-1).astype(bool)

        # F = F - A u_known
        F = F - A @ uh_global
        # enforce
        F[isbd_global] = uh_global[isbd_global]

        # A modification
        bdIdx = np.zeros(total, dtype=int)
        bdIdx[isbd_global] = 1
        Tbd = spdiags(bdIdx, 0, total, total)
        T = spdiags(1 - bdIdx, 0, total, total)
        A = T @ A @ T + Tbd

        return A, F

    @staticmethod
    def _lame(model):
        # 兼容你工程里 lambda0/lambda1 或 lam/mu
        if hasattr(model, "lam") and hasattr(model, "mu"):
            return float(model.lam), float(model.mu)
        if hasattr(model, "lambda0") and hasattr(model, "lambda1"):
            return float(model.lambda0), float(model.lambda1)
        if hasattr(model, "E") and hasattr(model, "nu"):
            E = float(model.E); nu = float(model.nu)
            mu = E/(2*(1+nu))
            lam = E*nu/((1+nu)*(1-2*nu))
            return lam, mu
        raise AttributeError("model must provide (lam,mu) or (lambda0,lambda1) or (E,nu)")
