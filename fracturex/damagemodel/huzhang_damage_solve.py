import numpy as np
from dataclasses import dataclass

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator

from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from fracturex.damagemodel.huzhang_boundary_condition import (
    HuzhangBoundaryCondition,
    HuzhangStressBoundaryCondition,
    build_isNedge_from_isD,
)
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d


# ---------------------------
# DamageModel 接口
# ---------------------------

class DamageModelBase:
    def setup(self, mesh, p, *, device=None):
        raise NotImplementedError

    def coef_bary(self, bcs, index=None):
        """给 HuZhangStressIntegrator 的 coef(bcs,index)"""
        raise NotImplementedError

    def update_after_elastic(self, sigmah, uh):
        """局部损伤更新：d <- update( sigmah, uh )"""
        raise NotImplementedError

    # --- 未来 phase-field / monolithic Newton 预留 ---
    def solve_damage(self, sigmah, uh):
        """phase-field: 解 d 方程，更新 self.dh"""
        raise NotImplementedError

    def residual(self, sigmah, uh):
        raise NotImplementedError

    def jacobian(self, sigmah, uh):
        raise NotImplementedError


class LocalNodeDamage(DamageModelBase):
    def __init__(self, *, ft, Hd, eps=1e-12, gtype="inv",
                 criterion="rankine", lam=None, mu=None):
        self.ft = float(ft)
        self.Hd = float(Hd)
        self.eps = float(eps)
        self.gtype = gtype

        self.criterion = str(criterion).lower()
        self.lam = lam
        self.mu = mu

        self.spaced = None
        self.dh = None
        self.r_hist = None

    # ---------- 等效量：2D ----------
    def _vmises_2d(self, sig):
        sxx = sig[..., 0]
        sxy = sig[..., 1]
        syy = sig[..., 2]
        return bm.sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

    def _principal_stress_max_2d(self, sig):
        sxx = sig[..., 0]
        sxy = sig[..., 1]
        syy = sig[..., 2]
        tr = 0.5*(sxx + syy)
        rad = bm.sqrt((0.5*(sxx - syy))**2 + sxy**2)
        s1 = tr + rad
        s2 = tr - rad
        # Rankine：只对拉主应力驱动（压缩不损伤）
        return bm.maximum(s1, s2)

    def _principal_strain_max_2d_from_stress(self, sig):
        """
        用 Hooke 反推应变，再取最大主应变。
        注意：这里你需要给 lam, mu（或 E,nu）。
        """
        if self.lam is None or self.mu is None:
            raise ValueError("criterion='pstrain' needs lam and mu (or implement strain recovery).")

        lam = self.lam
        mu = self.mu

        # 2D 里要区分 plane stress / plane strain。
        # 你现在 HuZhang 多用于平面应变（常见），这里先给 plane strain 的反算：
        #
        # sigma = 2mu*eps + lam*tr(eps)*I
        # => eps = 1/(2mu) sigma - lam/(2mu*(2mu+2lam)) tr(sigma) I
        #
        sxx = sig[..., 0]
        sxy = sig[..., 1]
        syy = sig[..., 2]
        trS = sxx + syy

        coef = lam / (2.0*mu*(2.0*mu + 2.0*lam))
        exx = (1.0/(2.0*mu))*sxx - coef*trS
        eyy = (1.0/(2.0*mu))*syy - coef*trS
        exy = (1.0/(2.0*mu))*sxy  # engineering shear? 这里按张量剪切

        # 主应变
        tr = 0.5*(exx + eyy)
        rad = bm.sqrt((0.5*(exx - eyy))**2 + exy**2)
        e1 = tr + rad
        e2 = tr - rad
        return bm.maximum(e1, e2)

    def equiv_measure(self, sig):
        c = self.criterion
        if c in ("rankine", "max_principal_stress", "s1"):
            return self._principal_stress_max_2d(sig)
        elif c in ("vmises", "vonmises", "mises"):
            return self._vmises_2d(sig)
        elif c in ("pstrain", "max_principal_strain", "e1"):
            return self._principal_strain_max_2d_from_stress(sig)
        else:
            raise ValueError(f"Unknown criterion={self.criterion}. Use rankine/vmises/pstrain.")

    # ---------- damage update ----------
    def update_after_elastic(self, sigmah, uh):
        mesh = self.spaced.mesh
        node = mesh.entity("node")      # (NN,2)

        sig_node = sigmah(node)         # 期望 (NN,3)
        r = self.equiv_measure(sig_node)

        # 只允许拉驱动（对 rankine 本来就是，vmises/pstrain 可加截断）
        if self.criterion in ("vmises", "vonmises", "mises", "pstrain"):
            r = bm.maximum(r, 0.0)

        # 不可逆历史
        if self.r_hist is None:
            self.r_hist = r.copy()
        else:
            self.r_hist = bm.maximum(self.r_hist, r)

        ft = self.ft
        rr = bm.maximum(self.r_hist, ft)

        dnew = 1.0 - bm.exp(-2.0*self.Hd*(rr-ft)/ft)
        dnew = bm.clip(dnew, 0.0, 0.999999)

        self.dh[:] = bm.maximum(self.dh[:], dnew)
        return self.dh



# ---------------------------
# 通用 fracture solver
# ---------------------------

class HuZhangFractureSolver2D:
    def __init__(self, case, *, p=4, q=None, use_relaxation=True, debug=False):
        self.case = case
        self.p = int(p)
        self.q = int(q) if q is not None else self.p + 3
        self.use_relaxation = bool(use_relaxation)
        self.debug = bool(debug)

        self.mesh = None
        self.space0 = None  # sigma
        self.space1 = None  # u

        self.sigmah = None
        self.uh = None

        self.damage = None  # DamageModelBase 实例

    def setup(self, N, damage_model: DamageModelBase):
        mesh = self.case.make_mesh(N)
        self.mesh = mesh

        # ΓN from ΓD
        isNedge = build_isNedge_from_isD(mesh, self.case.isD_bd)

        self.space0 = HuZhangFESpace2d(mesh, p=self.p,
                                       use_relaxation=self.use_relaxation,
                                       isNedge=isNedge)
        lag = LagrangeFESpace(mesh, p=self.p-1, ctype='D')
        self.space1 = TensorFunctionSpace(lag, shape=(-1, 2))

        self.sigmah = self.space0.function()
        self.uh = self.space1.function()

        # damage
        self.damage = damage_model
        self.damage.setup(mesh, self.p)

        return mesh

    def solve_elastic_once(self, load_value):
        mesh = self.mesh
        gdof0 = self.space0.number_of_global_dofs()
        gdof1 = self.space1.number_of_global_dofs()

        # ---- assemble M(d), B ----
        bform1 = BilinearForm(self.space0)
        bform1.add_integrator(
            HuZhangStressIntegrator(
                coef=self.damage.coef_bary,
                lambda0=self.case.lambda0(),  # 你 case 里可以返回 lambda0/lambda1 或 (lam,mu)
                lambda1=self.case.lambda1(),
            )
        )

        bform2 = BilinearForm((self.space1, self.space0))
        bform2.add_integrator(HuZhangMixIntegrator())

        M = bform1.assembly().to_scipy().tocsr()
        B = bform2.assembly().to_scipy().tocsr()

        TMsp = None
        if self.use_relaxation:
            TMsp = self.space0.TM.to_scipy().tocsr()
            M2 = TMsp.T @ M @ TMsp
            B2 = TMsp.T @ B
            A = bmat([[M2, B2],
                      [B2.T, None]], format="csr")
        else:
            A = bmat([[M, B],
                      [B.T, None]], format="csr")

        # ---- RHS ----
        HBC = HuzhangBoundaryCondition(space=self.space0, q=self.q)
        a = HBC.displacement_boundary_condition(
            piecewise=self.case.dirichlet_pieces(load_value)
        )

        lform = LinearForm(self.space1)
        lform.add_integrator(VectorSourceIntegrator(source=self.case.body_force))
        b = lform.assembly()

        F = bm.zeros(A.shape[0], dtype=mesh.ftype)
        if TMsp is not None:
            F[:gdof0] = TMsp.T @ a
        else:
            F[:gdof0] = a
        F[gdof0:] = -b

        # ---- optional essential stress BC on ΓN ----
        stress_info = self.case.stress_bc()
        if stress_info is not None:
            gd, thr, coord = stress_info
            HSBC = HuzhangStressBoundaryCondition(self.space0, q=self.q, debug=self.debug)
            A, F = HSBC.apply_essential_bc_to_system(
                A, bm.to_numpy(F), gd=gd, threshold=thr, coord=coord,
                sigma_offset=0, sigma_gdof=gdof0
            )

        # ---- solve ----
        X = spsolve(A, bm.to_numpy(F))
        sig_rel = X[:gdof0]
        uval = X[gdof0:]

        if TMsp is not None:
            sig_phys = TMsp @ sig_rel
        else:
            sig_phys = sig_rel

        self.sigmah[:] = bm.tensor(sig_phys)
        self.uh[:] = bm.tensor(uval)

    def run_staggered(self, N, damage_model, *, max_inner=30, tol=1e-6):
        self.setup(N, damage_model)

        loads = self.case.load_steps()
        for k, lv in enumerate(loads):
            if self.debug:
                print(f"\n=== step {k}/{len(loads)-1} load={lv} ===")

            # staggered inner iteration
            for it in range(max_inner):
                sigma_old = self.sigmah.copy()
                u_old = self.uh.copy()
                d_old = self.damage.dh.copy()

                self.solve_elastic_once(lv)
                self.damage.update_after_elastic(self.sigmah, self.uh)

                # 收敛判据
                ds = bm.linalg.norm(self.sigmah - sigma_old) / (bm.linalg.norm(self.sigmah) + 1e-15)
                du = bm.linalg.norm(self.uh - u_old) / (bm.linalg.norm(self.uh) + 1e-15)
                dd = bm.linalg.norm(self.damage.dh - d_old) / (bm.linalg.norm(self.damage.dh) + 1e-15)

                if self.debug:
                    print(f"  it={it} ds={float(ds):.2e} du={float(du):.2e} dd={float(dd):.2e}")

                if max(ds, du, dd) < tol:
                    break

        return self.sigmah, self.uh, self.damage.dh
