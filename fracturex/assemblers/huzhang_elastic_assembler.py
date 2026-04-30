# fracturex/assemblers/huzhang_elastic_assembler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
import inspect

import numpy as np
from scipy.sparse import bmat, coo_matrix

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


class _ScaledSigmaView:
    """Callable sigma view: sigma = g(d) * sigma_tilde."""

    def __init__(self, sigma_tilde_fun, damage, state):
        self.sigma_tilde_fun = sigma_tilde_fun
        self.damage = damage
        self.state = state

    def __call__(self, bcs, index=None):
        sig_tilde = self.sigma_tilde_fun(bcs, index=index)
        view = DamageStateView(
            d=self.state.d,
            sigma=self.state.sigma,
            u=self.state.u,
            r_hist=self.state.r_hist,
            H=self.state.H,
        )
        gd = self.damage.coef_bary(view, bcs, index=index)
        sig_tilde = bm.asarray(sig_tilde)
        gd = bm.asarray(gd)

        # Some FE backends may ignore `index` and return shape (1, NQ, ...).
        # Broadcast explicitly to avoid out-of-bound indexing in reaction postprocess.
        if index is not None:
            ncell = int(bm.asarray(index).reshape(-1).shape[0])
            if sig_tilde.ndim >= 1 and int(sig_tilde.shape[0]) == 1 and ncell > 1:
                reps = [ncell] + [1] * (sig_tilde.ndim - 1)
                sig_tilde = bm.tile(sig_tilde, reps)
            if gd.ndim >= 1 and int(gd.shape[0]) == 1 and ncell > 1:
                reps = [ncell] + [1] * (gd.ndim - 1)
                gd = bm.tile(gd, reps)
        return sig_tilde * gd[..., None]


class HuZhangElasticAssembler:
    """
    装配 HuZhang 混合线弹性系统：
      [ M(d)  B ]
      [ B^T   0 ]
    并统一处理 corner relaxation: M2=TM^T M TM, B2=TM^T B

    同时支持 piecewise 位移边界贡献到 σ 方程 RHS：
      F_sigma += TM^T * r_dirichlet
    """

    def __init__(
        self,
        discr: HuZhangDiscretization,
        case: CaseBase,
        damage: DamageModelBase,
        *,
        q: Optional[int] = None,
        formulation: str = "standard",
    ):
        """Create elasticity block assembler for HuZhang mixed system.

        Inputs:
            discr: Discretization object with mesh/spaces/state.
            case: Case object with material/load/boundary data.
            damage: Damage model used to evaluate degradation `g(d)`.
            q: Optional quadrature order.
            formulation: `"standard"` or `"effective_stress"`.
        Output:
            None. Initializes cache fields; call `assemble(load)` to build system.
        """
        self.discr = discr
        self.case = case
        self.damage = damage
        self.q = q  # 若 None，integrator 内部用默认
        self.formulation = str(formulation).lower()
        self._const_cache = None
        self._const_cache_key = None
        # Per load-step cache (staggered inner iters share the same `load`):
        self._load_step_value: Optional[float] = None
        self._load_step_piecewise: Optional[List[Tuple[Any, Any, Any]]] = None
        self._r_dir_standard: Optional[Any] = None
        self._neumann_data_raw: Any = None
        self._neumann_uh_sig: Optional[Any] = None
        self._neumann_is_bd: Optional[Any] = None
        self._sigma_essential_mask_cache = {}

    def begin_load_step(self, load: float) -> None:
        """
        Precompute load-dependent data that do not change during staggered iterations
        at fixed `load` (e.g. Dirichlet shape for standard σ–u, Neumann/essential-sigma sparsity).

        Safe to skip: `assemble` falls back to the previous on-the-fly path.
        """
        load = float(load)
        self._load_step_value = load
        self._r_dir_standard = None
        self._neumann_data_raw = None
        self._neumann_uh_sig = None
        self._neumann_is_bd = None

        discr = self.discr
        case = self.case
        space0 = discr.space_sigma

        self._prepare_constant_blocks(load)
        pieces = case.dirichlet_pieces(load)
        self._load_step_piecewise = [(pc.threshold, pc.value, pc.direction) for pc in pieces]

        if self.formulation != "effective_stress" and space0 is not None and self._load_step_piecewise is not None:
            HBC = HuzhangBoundaryCondition(space=space0, q=self.q)
            self._r_dir_standard = HBC.displacement_boundary_condition(piecewise=self._load_step_piecewise)

        nd = case.neumann_data(load)
        self._neumann_data_raw = nd
        if nd is not None and space0 is not None:
            HSBC = HuzhangStressBoundaryCondition(space=space0, q=self.q)
            if isinstance(nd, (list, tuple)) and len(nd) > 0 and isinstance(nd[0], (list, tuple)) and len(nd[0]) in (3, 4):
                uh_sig, is_bd = HSBC.set_essential_bc(gd=None, piecewise=nd)
            else:
                thr, gd, coord = nd
                uh_sig, is_bd = HSBC.set_essential_bc(gd, threshold=thr, coord=coord)
            self._neumann_uh_sig = uh_sig
            self._neumann_is_bd = is_bd

    def assemble(self, load: float) -> ElasticSystem:
        """Assemble one elastic subproblem at given load and current damage.

        Input:
            load: Current scalar load value.
        Output:
            `ElasticSystem(A, F, decode, meta)` where `decode(X)` maps solution
            vector to FE functions `(sigma_tilde, u, sigma_physical_or_none)`.
        """
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

        lam, mu = self._lame(case.model())
        n = mesh.geo_dimension()   # GD 
        c0 = 1.0/(2.0*mu)
        c1 = lam/(2.0*mu*(2.0*mu + n*lam))

        const = self._prepare_constant_blocks(load)
        TM = const["TM"]
        b_vec = const["b_vec"]
        if self.formulation == "effective_stress":
            M2 = const["M2_const"]
            B2 = self._assemble_coupled_B2(TM, state)
            if damage.debug:
                dmax = float(bm.max(bm.asarray(state.d[:])))
                if dmax > 1e-10:
                    diff = B2 - const["B2_const"]
                    rel = float(np.linalg.norm(diff.data) / max(np.linalg.norm(const["B2_const"].data), 1e-30))
                    print(f"[HuZhangElasticAssembler] effective_stress: ||B(d)-B0||/||B0||={rel:.3e}, dmax={dmax:.3e}")
        else:
            # ---- 1) M(d) ----
            # 退化系数：在单元积分点上评估 g(d)
            @barycentric
            def coef_d(bcs, index=None):
                # DamageStateView 用于统一接口
                view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
                return 1.0 / damage.coef_bary(view, bcs, index=index)

            bformM = BilinearForm(space0)
            bformM.add_integrator(HuZhangStressIntegrator(coef=coef_d, lambda0=c0, lambda1=c1))
            M = bformM.assembly().to_scipy().tocsr()
            M2 = TM.T @ M @ TM
            B2 = const["B2_const"]

        A = bmat([[M2, B2],
                  [B2.T, None]], format="csr")

        # ---- 4) RHS: body force on u eqn ----
        b = b_vec

        # ---- 5) RHS: Dirichlet displacement contributes to sigma equation ----
        HBC = HuzhangBoundaryCondition(space=space0, q=self.q)
        if self._load_step_value is not None and float(load) == float(self._load_step_value) and self._load_step_piecewise is not None:
            piecewise = self._load_step_piecewise
        else:
            pieces = case.dirichlet_pieces(load)
            piecewise = [(pc.threshold, pc.value, pc.direction) for pc in pieces]

        if self.formulation == "effective_stress":
            @barycentric
            def coef_g_face(bcs, index=None):
                view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
                return damage.coef_bary(view, bcs, index=index)

            r_dir = HBC.displacement_boundary_condition(piecewise=piecewise, coef=coef_g_face)
        else:
            if self._r_dir_standard is not None and self._load_step_value is not None and float(load) == float(self._load_step_value):
                r_dir = self._r_dir_standard
            else:
                r_dir = HBC.displacement_boundary_condition(piecewise=piecewise)

        F = np.zeros(A.shape[0], dtype=float)
        F[:gdof0] = (TM.T @ r_dir).reshape(-1)   # sigma unknown is transformed variable
        F[gdof0:] = -b

        # ---- 6) optional: essential stress/traction boundary on sigma (Neumann edges but essential on sigma dof) ----
        # 如果 case.neumann_data 返回 (gd, threshold, coord)，就做消元
        if self._load_step_value is not None and float(load) == float(self._load_step_value) and self._neumann_data_raw is not None:
            nd = self._neumann_data_raw
        else:
            nd = case.neumann_data(load)
        if nd is not None:
            if self._neumann_uh_sig is not None and self._neumann_is_bd is not None and self._load_step_value is not None and float(load) == float(self._load_step_value):
                uh_sig, isBd = self._neumann_uh_sig, self._neumann_is_bd
            else:
                HSBC = HuzhangStressBoundaryCondition(space=space0, q=self.q)
                if isinstance(nd, (list, tuple)) and len(nd) > 0 and isinstance(nd[0], (list, tuple)) and len(nd[0]) in (3, 4):
                    uh_sig, isBd = HSBC.set_essential_bc(gd=None, piecewise=nd)
                else:
                    thr, gd, coord = nd
                    uh_sig, isBd = HSBC.set_essential_bc(gd, threshold=thr, coord=coord)

            A, F = self.apply_sigma_essential_to_system(A, F, uh_sig, isBd, gdof0)


        # decode: map solution X -> (sigma,u) functions
        def decode(X):
            if isinstance(X, (tuple, list)):
                X = X[0]
            X = bm.asarray(X).reshape(-1)
            sig_tilde = X[:gdof0]
            u_vec = X[gdof0:]

            sigma_tilde = space0.function()
            sigma_tilde[:] = (TM @ sig_tilde).reshape(-1)

            u = space1.function()
            u[:] = u_vec
            if self.formulation == "effective_stress":
                sigma_physical = _ScaledSigmaView(sigma_tilde, damage, state)
                return sigma_tilde, u, sigma_physical
            return sigma_tilde, u, None
        
       


        meta = dict(
            gdof_sigma=int(gdof0),
            gdof_u=int(gdof1),
            formulation=self.formulation,
        )
        return ElasticSystem(A=A, F=F, decode=decode, meta=meta)

    def _prepare_constant_blocks(self, load: float):
        """
        Cache d-independent components:
        - TM, B2
        - body-force vector on displacement equation
        """
        discr = self.discr
        case = self.case
        mesh = discr.mesh
        space0 = discr.space_sigma
        space1 = discr.space_u

        assert mesh is not None and space0 is not None and space1 is not None

        key = (
            id(mesh),
            int(space0.number_of_global_dofs()),
            int(space1.number_of_global_dofs()),
        )
        if self._const_cache is not None and self._const_cache_key == key:
            return self._const_cache

        bformB = BilinearForm((space1, space0))
        bformB.add_integrator(HuZhangMixIntegrator())
        B = bformB.assembly().to_scipy().tocsr()
        TM = space0.TM.to_scipy().tocsr()
        B2 = TM.T @ B

        lam, mu = self._lame(case.model())
        n = mesh.geo_dimension()
        c0 = 1.0/(2.0*mu)
        c1 = lam/(2.0*mu*(2.0*mu + n*lam))
        bformM_const = BilinearForm(space0)
        bformM_const.add_integrator(HuZhangStressIntegrator(coef=1.0, lambda0=c0, lambda1=c1))
        M_const = bformM_const.assembly().to_scipy().tocsr()
        M2_const = TM.T @ M_const @ TM

        lform = LinearForm(space1)

        @cartesian
        def f_body(x, index=None):
            return case.body_force(x)

        lform.add_integrator(VectorSourceIntegrator(source=f_body))
        b_vec = lform.assembly()
        b_vec = np.asarray(b_vec, dtype=float).reshape(-1)

        self._const_cache_key = key
        self._const_cache = {
            "TM": TM,
            "B2_const": B2,
            "M2_const": M2_const,
            "b_vec": b_vec,
        }
        return self._const_cache

    def _assemble_coupled_B2(self, TM, state):
        """Assemble damage-coupled mixed block `B2` for effective-stress form.

        Inputs:
            TM: Stress-space relaxation transform matrix.
            state: Current solution state containing `d`.
        Output:
            Coupled sparse block matrix `B2`.
        """
        @barycentric
        def coef_g(bcs, index=None):
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
            return self.damage.coef_bary(view, bcs, index=index)

        bformB = BilinearForm((self.discr.space_u, self.discr.space_sigma))
        # Different fealpy versions expose coefficient in different ways.
        # Force binding the coefficient after construction to avoid silent no-op.
        try:
            params = inspect.signature(HuZhangMixIntegrator.__init__).parameters
            if "coef" in params:
                mix_int = HuZhangMixIntegrator(coef=coef_g)
            else:
                mix_int = HuZhangMixIntegrator()
        except Exception:
            mix_int = HuZhangMixIntegrator()

        setattr(mix_int, "coef", coef_g)
        setattr(mix_int, "_coef", coef_g)
        if hasattr(mix_int, "set_coef"):
            try:
                mix_int.set_coef(coef_g)
            except Exception:
                pass
        bformB.add_integrator(mix_int)
        B = bformB.assembly().to_scipy().tocsr()
        B_main = TM.T @ B

        # div(g Ψ) = g div Ψ + Ψ ∇g (row divergence). HuZhangMixIntegrator only
        # assembles ∫ g (div Ψ)·φ; add the missing ∫ φ·(Ψ ∇g) for spatially varying g(d_h).
        B_corr = self._assemble_B2_gradg_chainrule(TM, state)
        if B_corr is not None:
            if getattr(self.damage, "debug", False):
                n_main = float(np.linalg.norm(B_main.data))
                n_corr = float(np.linalg.norm(B_corr.data))
                print(
                    "[HuZhangElasticAssembler] effective_stress B2: "
                    f"||TM^T B_gdiv||={n_main:.3e}, ||TM^T B_gradg||={n_corr:.3e}"
                )
            return B_main + B_corr
        return B_main

    def _assemble_B2_gradg_chainrule(self, TM, state):
        """
        B_corr[i,j] = ∫_Ω φ_j · (Ψ_i ∇g) dx with g = g(d_h), same dof layout as
        BilinearForm((space_u, space_sigma)) + B2 = TM.T @ B.
        """
        damage = self.damage
        gprime_fn = getattr(damage, "degradation_grad", None)
        if gprime_fn is None:
            return None

        space_sigma = self.discr.space_sigma
        space_u = self.discr.space_u
        mesh = self.discr.mesh
        p = int(space_sigma.p)
        qord = int(self.q) if self.q is not None else p + 3

        qf = mesh.quadrature_formula(qord, "cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        ws = bm.asarray(ws)
        cm = bm.asarray(mesh.entity_measure("cell"))

        psi = bm.asarray(space_sigma.basis(bcs))  # (NC, NQ, ls, 3) Voigt xx,xy,yy
        phi_u = bm.asarray(space_u.basis(bcs))  # (NC, NQ, lu, 2)

        d_vals = bm.asarray(state.d(bcs))
        grad_d = bm.asarray(self.discr.space_d.grad_value(state.d, bcs))

        gp = bm.asarray(gprime_fn(d_vals))
        # grad_d: (NC, NQ, GD); psi: (NC, NQ, ls, 3) — broadcast ∇g along stress dof axis
        gx = (gp * grad_d[..., 0])[..., None]
        gy = (gp * grad_d[..., 1])[..., None]

        sxx, sxy, syy = psi[..., 0], psi[..., 1], psi[..., 2]
        wx = sxx * gx + sxy * gy
        wy = sxy * gx + syy * gy
        wvec = bm.stack([wx, wy], axis=-1)  # (NC, NQ, ls, 2)

        K = bm.einsum("q, c, cqmd, cqld -> clm", ws, cm, phi_u, wvec)

        cell2dof_s = bm.asarray(space_sigma.cell_to_dof())
        cell2dof_u = bm.asarray(space_u.cell_to_dof())
        NC, ls, lu = (int(K.shape[0]), int(K.shape[1]), int(K.shape[2]))

        I = bm.broadcast_to(cell2dof_s[:, :, None], (NC, ls, lu))
        J = bm.broadcast_to(cell2dof_u[:, None, :], (NC, ls, lu))

        gdof_s = space_sigma.number_of_global_dofs()
        gdof_u = space_u.number_of_global_dofs()
        Bc = coo_matrix(
            (
                np.asarray(K.reshape(-1), dtype=float),
                (np.asarray(I.reshape(-1), dtype=np.int64), np.asarray(J.reshape(-1), dtype=np.int64)),
            ),
            shape=(gdof_s, gdof_u),
        ).tocsr()
        return TM.T @ Bc

    def apply_sigma_essential_to_system(self, A, F, uh_sigma, isBd_sigma, gdof_sigma: int):
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
        isbd_key = np.asarray(isbd_global, dtype=np.bool_).tobytes()
        cache_key = (int(total), isbd_key)
        cached = self._sigma_essential_mask_cache.get(cache_key)
        if cached is None:
            bdIdx = np.zeros(total, dtype=int)
            bdIdx[isbd_global] = 1
            Tbd = spdiags(bdIdx, 0, total, total)
            T = spdiags(1 - bdIdx, 0, total, total)
            self._sigma_essential_mask_cache[cache_key] = (T, Tbd)
        else:
            T, Tbd = cached
        A = T @ A @ T + Tbd

        return A, F

    @staticmethod
    def _lame(model):
        """Resolve Lamé parameters from material model.

        Input:
            model: Object with `(lam, mu)` or `(lambda0, lambda1)` or `(E, nu)`.
        Output:
            Tuple `(lam, mu)` as floats.
        """
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
