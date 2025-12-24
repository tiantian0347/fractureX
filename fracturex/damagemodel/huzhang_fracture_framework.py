
"""
HuZhang fracture framework (staggered now, extensible to adaptivity + fast solvers + monolithic Newton later).

Key ideas:
- Case: provides mesh + BC definitions + material parameters.
- Discretization: rebuild spaces/dofs/TM when mesh/p changes.
- State: sigma/u/d (all as space.function), plus history (irreversible).
- Assembler: assembles blocks, with cache + invalidation.
- LinearSolver: pluggable (direct/Krylov/block precond).
- Adaptivity: stub interface (estimate/mark/refine/transfer).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, Tuple

import numpy as np
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fealpy.fem import BilinearForm
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator

from fracturex.damagemodel.huzhang_boundary_condition import (
    HuzhangBoundaryCondition,
    HuzhangStressBoundaryCondition,
    build_isNedge_from_isD,  # you already have this helper
)


# ---------------------------
# Config
# ---------------------------

@dataclass
class SolverConfig:
    p: int = 3
    q: Optional[int] = None
    use_relaxation: bool = True

    # damage (local)
    criterion: str = "rankine"   # rankine / vmises / pstrain
    Hd: float = 1.0
    ft: float = 1.0
    eps_g: float = 1e-15

    # staggered
    max_inner: int = 50
    tol_d: float = 1e-6

    # linear solver
    linear: str = "direct"       # direct / krylov (placeholder)
    krylov_tol: float = 1e-10
    krylov_maxit: int = 300

    # adaptivity hooks (stub)
    adapt_every: int = 0         # 0 means off
    adapt_max_levels: int = 0


# ---------------------------
# Case interface (you can wrap your existing model)
# ---------------------------

class CaseBase:
    def material(self) -> Any:
        """Return an object with lam, mu, ft ..."""
        raise NotImplementedError

    def build_mesh(self, N: int) -> TriangleMesh:
        raise NotImplementedError

    # displacement BC pieces for sigma-equation RHS (HuZhangBoundaryCondition)
    def dirichlet_pieces(self, load: float):
        """
        Return piecewise list: [(threshold, value, direction), ...]
        threshold: callable(bc)->bool mask or index array
        value: callable(points)->(GD,) or scalar
        direction: 'x'/'y'/None
        """
        raise NotImplementedError

    # Neumann/stress essential BC on sigma dofs (HuZhangStressBoundaryCondition)
    def neumann_selector(self) -> Callable:
        """
        Return isNedge selector. Either:
        - callable(bc_edge)->bool mask on boundary edges
        - (NE,) bool mask
        """
        return None

    # if user prefers to specify Dirichlet boundary instead of Neumann selector
    def isD_bd(self) -> Optional[Callable]:
        """
        callable(bc_edge)->bool mask for Dirichlet edges.
        If provided and neumann_selector is None, we build isNedge = boundary & ~isD.
        """
        return None


# ---------------------------
# Discretization (rebuild when mesh or p changes)
# ---------------------------

class Discretization2D:
    def __init__(self, cfg: SolverConfig, case: CaseBase):
        self.cfg = cfg
        self.case = case
        self.mesh: TriangleMesh = None

        self.hspace: HuZhangFESpace2d = None   # sigma space
        self.lspace: LagrangeFESpace = None
        self.tspace: TensorFunctionSpace = None
        self.dspace: LagrangeFESpace = None

        self.isNedge = None  # (NE,) bool

        self._key = None

    def build(self, mesh: TriangleMesh):
        self.mesh = mesh
        p = self.cfg.p
        q = self.cfg.q if self.cfg.q is not None else p + 3

        # ---- build isNedge (NE,) ----
        isNedge = self.case.neumann_selector()
        isD_bd = self.case.isD_bd()

        if isNedge is None and isD_bd is not None:
            isNedge = build_isNedge_from_isD(mesh, isD_bd)

        self.isNedge = isNedge

        # ---- spaces ----
        self.hspace = HuZhangFESpace2d(mesh, p=p, use_relaxation=self.cfg.use_relaxation, isNedge=isNedge, isD_bd=isD_bd)
        self.lspace = LagrangeFESpace(mesh, p=p)
        self.tspace = TensorFunctionSpace(self.lspace, shape=(-1, mesh.geo_dimension()))

        # local damage at nodes: P1 C0 is typical
        self.dspace = LagrangeFESpace(mesh, p=1, ctype='C')

        self.q = q
        self._key = (id(mesh), p, self.cfg.use_relaxation, q)
        return self

    @property
    def key(self):
        return self._key


# ---------------------------
# State (all as space.function)
# ---------------------------

class State:
    def __init__(self):
        self.sigma = None
        self.u = None
        self.d = None
        self.r_hist = None  # irreversible history (node-based)

    def allocate(self, discr: Discretization2D):
        self.sigma = discr.hspace.function()
        self.u = discr.tspace.function()
        self.d = discr.dspace.function()
        self.d[:] = 0.0
        self.r_hist = bm.zeros((discr.mesh.number_of_nodes(),), dtype=discr.mesh.ftype)
        return self

    def transfer_from(self, old: "State", old_discr: Discretization2D, new_discr: Discretization2D):
        """
        Minimal but robust transfer:
        - d and r_hist: interpolate to new nodes + enforce irreversibility via max
        - sigma/u: interpolate as initial guess
        """
        new_mesh = new_discr.mesh
        new_node = new_mesh.entity('node')

        # sigma/u evaluated at nodes
        try:
            self.sigma[:] = old.sigma(new_node)
        except Exception:
            pass
        try:
            self.u[:] = old.u(new_node)
        except Exception:
            pass

        # d at nodes: old.d is P1, so can eval on new_node
        d_new = old.d(new_node)
        self.d[:] = bm.maximum(self.d[:], d_new)

        # r_hist: stored as array on old nodes; interpolate by nearest (cheap) as placeholder
        # (Better: L2 projection or barycentric interpolation; leave hook here)
        # simple nearest-neighbor:
        old_node = old_discr.mesh.entity('node')
        # compute nearest indices (O(N^2) naive) - replace in future with KDTree
        # here do a safe fallback: if same mesh, copy; else reset to 0
        if old_node.shape[0] == new_node.shape[0] and bm.max(bm.abs(old_node - new_node)) < 1e-14:
            self.r_hist[:] = old.r_hist.copy()
        else:
            self.r_hist[:] = 0.0

        return self


# ---------------------------
# Damage model (local, node-based)
# ---------------------------

class LocalNodeDamage:
    def __init__(self, *, ft: float, Hd: float, criterion="rankine", lam=None, mu=None):
        self.ft = float(ft)
        self.Hd = float(Hd)
        self.criterion = str(criterion).lower()
        self.lam = lam
        self.mu = mu

    def _vmises_2d(self, sig):
        sxx = sig[..., 0]; sxy = sig[..., 1]; syy = sig[..., 2]
        return bm.sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

    def _rankine_2d(self, sig):
        sxx = sig[..., 0]; sxy = sig[..., 1]; syy = sig[..., 2]
        tr = 0.5*(sxx + syy)
        rad = bm.sqrt((0.5*(sxx - syy))**2 + sxy**2)
        s1 = tr + rad
        s2 = tr - rad
        return bm.maximum(s1, s2)

    def equiv(self, sig):
        if self.criterion in ("rankine", "max_principal_stress", "s1"):
            return self._rankine_2d(sig)
        if self.criterion in ("vmises", "vonmises", "mises"):
            return self._vmises_2d(sig)
        raise ValueError(f"Unknown criterion={self.criterion}")

    def update(self, discr: Discretization2D, state: State):
        mesh = discr.mesh
        node = mesh.entity('node')
        sig_node = state.sigma(node)         # (NN,3)
        r = self.equiv(sig_node)
        r = bm.maximum(r, 0.0)

        # irreversible history
        state.r_hist[:] = bm.maximum(state.r_hist, r)

        rr = bm.maximum(state.r_hist, self.ft)
        dnew = 1.0 - bm.exp(-2.0*self.Hd*(rr-self.ft)/self.ft)
        dnew = bm.clip(dnew, 0.0, 0.999999)

        # enforce irreversibility on d itself
        state.d[:] = bm.maximum(state.d[:], dnew)
        return state.d


# ---------------------------
# Assembler with cache
# ---------------------------

class HuZhangAssembler:
    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg
        self._cache_key = None
        self._B = None   # cached B (depends on mesh/p only)

    def invalidate(self):
        self._cache_key = None
        self._B = None

    def assemble_system(self, discr: Discretization2D, state: State, case: CaseBase, load: float):
        mesh = discr.mesh
        hspace = discr.hspace
        tspace = discr.tspace
        gdof0 = hspace.number_of_global_dofs()

        p = self.cfg.p
        q = discr.q

        # ---- M(d) ----
        @bm.barycentric
        def coef_func(bcs, index=None):
            # use d(bcs) like your current code
            return 1.0 - state.d(bcs) + self.cfg.eps_g

        bform1 = BilinearForm(hspace)
        bform1.add_integrator(HuZhangStressIntegrator(coef=coef_func, lambda0=case.material().lam, lambda1=case.material().mu))

        # ---- B (cached) ----
        if self._cache_key != discr.key:
            bform2 = BilinearForm((tspace, hspace))
            bform2.add_integrator(HuZhangMixIntegrator())
            B = bform2.assembly().to_scipy().tocsr()
            self._B = B
            self._cache_key = discr.key

        M = bform1.assembly().to_scipy().tocsr()
        B = self._B

        # ---- relaxation transform ----
        if self.cfg.use_relaxation:
            TM = hspace.TM.to_scipy().tocsr()
            M2 = TM.T @ M @ TM
            B2 = TM.T @ B
            A = bmat([[M2,  B2],
                      [B2.T, None]], format="csr")
        else:
            A = bmat([[M, B],
                      [B.T, None]], format="csr")

        # ---- RHS from displacement BC (sigma equation) ----
        HBC = HuzhangBoundaryCondition(space=hspace, q=q)
        pieces = case.dirichlet_pieces(load)
        a = HBC.displacement_boundary_condition(piecewise=pieces)

        F = bm.zeros(A.shape[0], dtype=A.dtype)
        if self.cfg.use_relaxation:
            TM = hspace.TM.to_scipy().tocsr()
            F[:gdof0] = TM.T @ a
        else:
            F[:gdof0] = a

        # ---- essential stress BC (Neumann) on sigma dofs (optional) ----
        # if you need: use HuzhangStressBoundaryCondition to modify A,F
        # (left as hook; implemented via helper below)
        return A, F, gdof0

    @staticmethod
    def apply_essential_sigma_bc_to_block(A, F, gdof0, uh_sigma, isbddof_sigma):
        """
        Encapsulate your block "eliminate known dofs" logic on sigma block.
        """
        uh_global = bm.zeros(A.shape[0], dtype=A.dtype)
        uh_global[:gdof0] = uh_sigma

        isbddof_global = bm.zeros(A.shape[0], dtype=bool)
        isbddof_global[:gdof0] = isbddof_sigma

        F = F - A @ uh_global
        F[isbddof_global] = uh_global[isbddof_global]

        total = A.shape[0]
        bdIdx = bm.zeros(total, dtype=bm.int32)
        bdIdx[isbddof_global] = 1

        Tbd = spdiags(bdIdx, 0, total, total)
        T = spdiags(1 - bdIdx, 0, total, total)
        A = T @ A @ T + Tbd
        return A, F


# ---------------------------
# Linear solver (pluggable)
# ---------------------------

class LinearSolver:
    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg

    def solve(self, A, F, x0=None):
        # placeholder: direct
        return spsolve(A, F)


# ---------------------------
# Adaptivity (stub)
# ---------------------------

class AdaptivityBase:
    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg

    def enabled(self):
        return self.cfg.adapt_every > 0 and self.cfg.adapt_max_levels > 0

    def maybe_adapt(self, step: int, discr: Discretization2D, state: State, case: CaseBase):
        """
        Return (mesh_new, did_adapt).
        You can plug in:
        - indicator based on |grad d|, crack front, stress recovery estimator, etc.
        - mark & refine
        - then state.transfer_from(...)
        """
        return discr.mesh, False


# ---------------------------
# Driver: staggered now, extensible later
# ---------------------------

class HuZhangFractureDriver2D:
    def __init__(self, case: CaseBase, cfg: SolverConfig, damage: Optional[LocalNodeDamage] = None,
                 adapt: Optional[AdaptivityBase] = None):
        self.case = case
        self.cfg = cfg
        self.damage = damage
        self.adapt = adapt if adapt is not None else AdaptivityBase(cfg)

        self.discr = Discretization2D(cfg, case)
        self.state = State()
        self.assembler = HuZhangAssembler(cfg)
        self.linsolver = LinearSolver(cfg)

    def setup(self, N: int):
        mesh = self.case.build_mesh(N)
        self.discr.build(mesh)
        self.state.allocate(self.discr)

        if self.damage is None:
            mat = self.case.material()
            self.damage = LocalNodeDamage(
                ft=getattr(mat, "ft", self.cfg.ft),
                Hd=self.cfg.Hd,
                criterion=self.cfg.criterion,
                lam=mat.lam, mu=mat.mu
            )
        return self

    def solve_one_elastic(self, load: float):
        A, F, gdof0 = self.assembler.assemble_system(self.discr, self.state, self.case, load)

        # optional: essential stress BC (sigma) if case provides it
        # HSBC = HuzhangStressBoundaryCondition(space=self.discr.hspace, q=self.discr.q)
        # uh_sigma, isbddof_sigma = HSBC.set_essential_bc(self.case.stress_bc(), threshold=self.discr.isNedge, coord="auto")
        # A, F = self.assembler.apply_essential_sigma_bc_to_block(A, F, gdof0, uh_sigma, isbddof_sigma)

        X = self.linsolver.solve(A, F)
        if self.cfg.use_relaxation:
            TM = self.discr.hspace.TM.to_scipy().tocsr()
            self.state.sigma[:] = TM @ X[:gdof0]
        else:
            self.state.sigma[:] = X[:gdof0]
        self.state.u[:] = X[gdof0:]
        return self.state.sigma, self.state.u

    def run_staggered(self, loads):
        """
        loads: iterable of load values (e.g. displacement levels)
        """
        for istep, load in enumerate(loads):
            # ---- adaptivity hook (pre-step) ----
            if self.adapt.enabled() and (istep % self.cfg.adapt_every == 0) and istep > 0:
                mesh_new, did = self.adapt.maybe_adapt(istep, self.discr, self.state, self.case)
                if did:
                    old_discr, old_state = self.discr, self.state
                    self.discr = Discretization2D(self.cfg, self.case).build(mesh_new)
                    self.state = State().allocate(self.discr).transfer_from(old_state, old_discr, self.discr)
                    self.assembler.invalidate()

            # ---- inner staggered loop ----
            for it in range(self.cfg.max_inner):
                d_old = self.state.d.copy()

                self.solve_one_elastic(load)
                self.damage.update(self.discr, self.state)

                nd = bm.linalg.norm(self.state.d - d_old)
                if float(nd) < self.cfg.tol_d:
                    break

            yield istep, load, self.state


