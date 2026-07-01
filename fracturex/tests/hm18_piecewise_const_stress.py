"""[HM18] §5.1-style piecewise constant stress test.

Domain Ω = [0,1]^2, cut into two halves at y = 0.5:
  σ = σ_lower  in {y < 0.5}
  σ = σ_upper  in {y > 0.5}

Chosen so that:
  - div σ = 0 (piecewise constant → satisfies momentum);
  - across y=0.5: σ · n = σ · (0, 1) must match → σ_xy and σ_yy are equal
    between top and bottom (H(div) requirement);
  - σ_xx can differ → traction on the vertical boundaries x=0 and x=1 is
    DISCONTINUOUS at y=0.5 (this is the corner traction inconsistency
    the wrapper is supposed to resolve).

Boundaries:
  - Bottom (y=0):        ΓD (impose u from analytic integration)
  - Left  (x=0):         ΓN (σ · (-1, 0) = -σ_xx * (1, 0))
  - Right (x=1):         ΓN (σ · (+1, 0) = +σ_xx * (1, 0))
  - Top   (y=1):         ΓN (σ · (0, 1))

Corners at (0, 0.5) and (1, 0.5) are NN corners where two ΓN edges meet with
different σ · n values (top/bottom split). base mode cannot express this
inconsistency exactly; wrapper should reduce error toward machine precision.

Since σ is piecewise CONSTANT, the P_3 stress space can express it exactly →
expected error = O(machine ε) once corner treatment is right.
"""
from __future__ import annotations
import sys
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import VectorSourceIntegrator, BilinearForm, LinearForm
from fealpy.decorator import cartesian, barycentric
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat, spdiags
import scipy.sparse as sp

from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition, HuzhangStressBoundaryCondition, build_isNedge_from_isD,
)
from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax
from fracturex.assemblers.huzhang_unc_assembler import assemble_M_unc, assemble_B_unc


# 分片常应力：跨 y=0.5 跳跃，只让 σ_xx 跳（保 H(div)）
SIGMA_LOWER = np.array([1.0, 0.5, 2.0])   # (σ_xx, σ_xy, σ_yy)
SIGMA_UPPER = np.array([3.0, 0.5, 2.0])   # 只有 σ_xx 与下半不同


def sigma_exact_fn(pts):
    """pts shape (..., 2) → σ shape (..., 3) Voigt."""
    pts = np.asarray(pts)
    y = pts[..., 1]
    lo = SIGMA_LOWER.reshape((1,) * (pts.ndim - 1) + (3,))
    up = SIGMA_UPPER.reshape((1,) * (pts.ndim - 1) + (3,))
    lo = np.broadcast_to(lo, pts.shape[:-1] + (3,))
    up = np.broadcast_to(up, pts.shape[:-1] + (3,))
    mask = (y < 0.5)[..., None]
    return np.where(mask, lo, up)


# 位移：ε = A σ = (1/(2μ)) (σ - λ/(2μ+nλ)·trσ·I)
# 用测试脚本里的 λ0=4, λ1=1 → 常数柔度算子；ε 也分片常
# ε_xx = (1/λ0)·(σ_xx - λ1/(λ0-2λ1)·(σ_xx+σ_yy))
# 简单起见：解析 u = ε · x（分片线性），保证 y=0.5 处连续（若 ε 跨界面法向分量连续）。
# ε_yy 上下一样（σ_yy, σ_xx 里 λ 系数），ε_xx 上下不一样（σ_xx 不同），ε_xy 一样。
# 所以 u_x = ε_xx x + ε_xy y 在跨界面 y=0.5 处：上下 ε_xy 相同，ε_xx 不同 → u_x 分片线性 in x，
# 在 y=0.5 上仍然 x 连续 ✓。u_y = ε_xy x + ε_yy y → 上下 ε_yy 相同 → 全域连续。
# u|_{y=0} = ε_xx x（下半）, u|_{y=1} = ε_xx(上) x + (ε_xy x + ε_yy)。

def build_u_exact(lambda0, lambda1):
    """Return (u_of_pts, eps_of_pts) analytic functions given elasticity constants."""
    c0 = 1.0 / lambda0
    c1 = lambda1 / (lambda0 - 2 * lambda1)

    def eps_from_sigma(sig_voigt):
        # sig_voigt shape (..., 3) → ε shape (..., 3)
        sig = np.asarray(sig_voigt)
        tr = sig[..., 0] + sig[..., 2]
        eps = np.empty_like(sig)
        eps[..., 0] = c0 * (sig[..., 0] - c1 * tr)
        eps[..., 1] = c0 * sig[..., 1]
        eps[..., 2] = c0 * (sig[..., 2] - c1 * tr)
        return eps

    def u_exact(pts):
        pts = np.asarray(pts)
        x, y = pts[..., 0], pts[..., 1]
        sig = sigma_exact_fn(pts)      # (..., 3)
        eps = eps_from_sigma(sig)
        # 需要 u 连续，且 y=0.5 处上下匹配
        # 用 u_x = ε_xx(x, y)·x + ε_xy·y
        # 上下 ε_xy 相同（σ_xy 相同）；ε_xx 上下不同 → 在 y=0.5 处 u_x(x, 0.5^-) = ε_xx(lower)·x + ε_xy·0.5
        # u_x(x, 0.5^+) = ε_xx(upper)·x + ε_xy·0.5 → 不连续！
        # 需要修正：在 y>0.5 加上一个 y-依赖的常数 (ε_xx(lower)-ε_xx(upper))·x·(y-0.5) 到 u_x？
        # 但那会破坏 ε_xy = symmetric grad = (∂u_x/∂y + ∂u_y/∂x)/2 = ε_xy 分片常，不再匹配。
        # 简单方案：让 σ 只在 σ_yy 上跳（法向分量不能跳，破坏 H(div)），或让 σ_xy 跳（引起
        # ε_xy 跳，破坏 u 光滑性）。
        # 实际上分片常 σ 要严格连续 u 只可能 ε 全域相同 → σ 全域相同 (常张量)。
        # → 本测试放弃"连续 u"，用位移在 ΓD 上的分片线性投影作为 essential BC。
        u = np.empty_like(pts)
        u[..., 0] = eps[..., 0] * x + eps[..., 1] * y
        u[..., 1] = eps[..., 1] * x + eps[..., 2] * y
        return u

    return u_exact, eps_from_sigma


def isD_bottom(bc):
    return bm.abs(bc[:, 1] - 0.0) < 1e-12


def solve(N: int, p: int, *, mode: str, lambda0: float, lambda1: float,
          skip_nn_corner_nodes: bool = True):
    """mode: 'base' or 'relax'."""
    mesh = TriangleMesh.from_box([0., 1., 0., 1.], nx=N, ny=N)
    isNedge = build_isNedge_from_isD(mesh, isD_bottom)
    base_space = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isNedge)
    space_u_sc = LagrangeFESpace(mesh, p=p - 1, ctype='D')
    space_u = TensorFunctionSpace(space_u_sc, shape=(-1, 2))
    u_exact, _ = build_u_exact(lambda0, lambda1)

    @barycentric
    def coef(bcs, index=None):
        return bm.ones((1,) + bcs.shape[:-1], dtype=bm.float64)
    bform1 = BilinearForm(base_space)
    bform1.add_integrator(HuZhangStressIntegrator(coef=coef, lambda0=lambda0, lambda1=lambda1))
    bform2 = BilinearForm((space_u, base_space))
    bform2.add_integrator(HuZhangMixIntegrator())

    L = LinearForm(space_u)
    @cartesian
    def src(x, index=None):
        # div σ = 0 → source f = 0
        return bm.zeros(x.shape, dtype=x.dtype)
    L.add_integrator(VectorSourceIntegrator(source=src))
    b = L.assembly()

    HBC = HuzhangBoundaryCondition(space=base_space)

    def u_D(pts):
        return bm.tensor(u_exact(np.asarray(pts)), dtype=base_space.ftype)
    a = HBC.displacement_boundary_condition(value=u_D, threshold=isD_bottom)

    HSBC = HuzhangStressBoundaryCondition(space=base_space)

    def sig_gd(pts):
        return bm.tensor(sigma_exact_fn(np.asarray(pts)), dtype=base_space.ftype)
    uh_stress_base, isbd_stress_base = HSBC.set_essential_bc_v2(
        sig_gd, threshold=isNedge, coord='auto',
        skip_nn_corner_nodes=skip_nn_corner_nodes,
    )

    if mode == 'base':
        M = bform1.assembly().to_scipy().tocsr()
        B = bform2.assembly().to_scipy().tocsr()
        A = bmat([[M, B], [B.T, None]], format='csr')
        gdof_sig = base_space.number_of_global_dofs()
        F = np.zeros(A.shape[0])
        F[:gdof_sig] = np.asarray(a)
        F[gdof_sig:] = -np.asarray(b)
        uh_g = np.zeros(A.shape[0])
        isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof_sig] = np.asarray(uh_stress_base)
        isbd_g[:gdof_sig] = np.asarray(isbd_stress_base)
        ncp = 0
        relax = None
    elif mode == 'relax':
        relax = HuZhangCornerRelax(mesh, p=p, isNedge=isNedge, base_space=base_space)
        ncp = relax.diag()['n_corners']
        M_unc = assemble_M_unc(relax, lambda0=lambda0, lambda1=lambda1, coef=coef)
        B_unc = assemble_B_unc(relax, space_u)
        C = relax.C_constraint
        nC = C.shape[0]
        gdof_u = space_u.number_of_global_dofs()
        zero_uC = sp.csr_matrix((gdof_u, nC))
        zero_Cu = sp.csr_matrix((nC, gdof_u))
        zero_CC = sp.csr_matrix((nC, nC))
        A = bmat([[M_unc, B_unc, C.T],
                  [B_unc.T, None, zero_uC],
                  [C, zero_Cu, zero_CC]], format='csr')
        gdof_sig = relax.gdof_unc
        # 右端：a 从 base 投到 unc（非角点恒等，角点节点 base 3 → 所有 cells）
        node2dof = np.asarray(base_space.dof.node_to_internal_dof())
        corner_base = set()
        for c in relax.corners:
            corner_base.update(node2dof[c.nid].tolist())
        a_unc = np.zeros(relax.gdof_unc)
        a_np = np.asarray(a)
        for d in range(relax.gdof_base):
            if d in corner_base:
                continue
            a_unc[d] = a_np[d]
        for c in relax.corners:
            base3 = node2dof[c.nid]
            for comp in range(3):
                bd = int(base3[comp])
                for k in range(len(c.cells)):
                    a_unc[int(c.unc_dofs[k, comp])] = a_np[bd]
        F = np.zeros(A.shape[0])
        F[:gdof_sig] = a_unc
        F[gdof_sig:gdof_sig + gdof_u] = -np.asarray(b)
        # essential bc 在 unc 上（fan-end lock）
        uh_unc, isbd_unc = relax.lift_base_bc_to_unc(
            np.asarray(uh_stress_base), np.asarray(isbd_stress_base))
        uh_g = np.zeros(A.shape[0])
        isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof_sig] = uh_unc
        isbd_g[:gdof_sig] = isbd_unc
    else:
        raise ValueError(mode)

    F = F - A @ uh_g
    F[isbd_g] = uh_g[isbd_g]
    n = A.shape[0]
    bdIdx = isbd_g.astype(int)
    A = spdiags(1 - bdIdx, 0, n, n) @ A @ spdiags(1 - bdIdx, 0, n, n) + spdiags(bdIdx, 0, n, n)

    X = spsolve(A, F)
    sig_sol = X[:gdof_sig]

    if mode == 'base':
        sigmah = base_space.function()
        sigmah[:] = sig_sol
        e = float(mesh.error(sigmah, sigma_exact_fn, q=2 * p + 6))
        return e, ncp
    else:
        # 用 wrapper 的 cell-local σ 度量误差
        e = relax.l2_error_unc(sig_sol, sigma_exact_fn, is_unc=True, q=2 * p + 6)
        return e, ncp


def main(p: int = 3, maxit: int = 4):
    lambda0, lambda1 = 4.0, 1.0
    print(f"[HM18-piecewise-const] p={p}")
    print(f"σ_lower = {SIGMA_LOWER.tolist()}, σ_upper = {SIGMA_UPPER.tolist()}")
    print(f"σ_xx jump at y=0.5: {SIGMA_UPPER[0] - SIGMA_LOWER[0]}")

    for mode in ('base', 'relax'):
        print(f"\n=== mode={mode} (skip_nn_corner_nodes=True) ===")
        print(f"{'N':>5} {'NCP':>5} {'|σ-σh|':>14} {'rate':>6}")
        prev = None
        for i in range(maxit):
            N = 2 ** (i + 1)
            e, ncp = solve(N, p, mode=mode, lambda0=lambda0, lambda1=lambda1,
                           skip_nn_corner_nodes=True)
            rate = '-' if prev is None else f"{np.log2(prev / e):.2f}"
            print(f"{N:>5} {ncp:>5} {e:>14.4e} {rate:>6}")
            prev = e


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    maxit = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(p, maxit)
