"""[HM18]-style test: TRUE traction inconsistency at a geometric NN corner.

Ω = [0,1]². Set:
  - ΓD = top edge (y=1) + left edge (x=0), impose u = 0
  - ΓN = bottom edge (y=0) + right edge (x=1)
    - bottom: σn = g_bot(x)  (n = (0, -1))
    - right : σn = g_right(y) (n = (+1, 0))
  Corner (1, 0) is a geometric NN corner (m=2, two ΓN edges meet at 90°).

We DELIBERATELY choose g_bot and g_right that DO NOT come from a single stress
tensor at (1, 0):
  g_bot(x)  = (0, -σ_yy_bot)  → says σ_yy(1,0) = SIGMA_BOT_YY
  g_right(y) = (σ_xx_right, 0) → says σ_xx(1,0) = SIGMA_RIGHT_XX

These are compatible in the σ tensor (they only constrain σ_yy and σ_xx
individually), so there's no true inconsistency yet. To create inconsistency,
we also let each edge specify σ_xy at the corner independently:

  g_bot(x)  = (SIGMA_XY_BOT, -σ_yy_bot)  → σ_xy(1,0)|_bot = SIGMA_XY_BOT
  g_right(y) = (σ_xx_right, SIGMA_XY_RIGHT) → σ_xy(1,0)|_right = SIGMA_XY_RIGHT

If SIGMA_XY_BOT ≠ SIGMA_XY_RIGHT, no single σ(1,0) can satisfy both →
this is a [HM18] §4 traction inconsistency scenario at a geometric NN corner.

Because there's no consistent σ satisfying the two conflicting BCs at (1,0),
there's no "exact" solution to compare against. Instead we look at:
  - base mode: node trace DOF at (1,0) gets clobbered by whichever edge writes
    last → biased σ_h near (1,0), O(1) local error.
  - wrapper: each side keeps its own cell-local σ satisfying its own BC,
    smaller near-corner error.

We use a fixed reference σ solution (from an over-refined base mesh?) as an
alternative — but that's circular. Better: measure the *jump* of σ_h · n at
the corner between the two ΓN cells adjacent to (1,0). base should produce a
zero jump (single node DOF), wrapper should produce a jump matching the
prescribed g_bot vs g_right mismatch.
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


# 真正的角点 traction 不相容：两条 ΓN 边给出的 σ_xy(1,0) 不一致
SIGMA_XY_BOT = 1.0     # 底边给的 σ_xy(1,0)
SIGMA_XY_RIGHT = 2.0   # 右边给的 σ_xy(1,0)（≠ SIGMA_XY_BOT）
SIGMA_YY_BOT = 1.0     # 底边给的 σ_yy
SIGMA_XX_RIGHT = 3.0   # 右边给的 σ_xx


def sigma_g(pts):
    """Voigt stress at ΓN points (right edge x=1 and top edge y=1).

    At corner (1,1) two edges give conflicting σ_xy:
      right (x=1, n=(1,0)):  σ_xy = SIGMA_XY_RIGHT
      top   (y=1, n=(0,1)):  σ_xy = SIGMA_XY_TOP
    """
    pts = np.asarray(pts)
    x, y = pts[..., 0], pts[..., 1]
    on_right = np.abs(x - 1.0) < 1e-10
    on_top = np.abs(y - 1.0) < 1e-10
    sig = np.zeros(pts.shape[:-1] + (3,))
    # 默认按右边
    sig[..., 0] = SIGMA_XX_RIGHT
    sig[..., 1] = SIGMA_XY_RIGHT     # right 视角
    sig[..., 2] = 0.0
    # 顶边覆盖
    sig[on_top, 0] = 0.0
    sig[on_top, 1] = SIGMA_XY_BOT     # 复用 SIGMA_XY_BOT 作为"top 视角"的 σ_xy
    sig[on_top, 2] = SIGMA_YY_BOT     # 顶边给的 σ_yy
    return sig


def isD_bot_left(bc):
    """ΓD = bottom (y=0) + left (x=0), so (1,1) is NN corner."""
    tol = 1e-10
    on_bot = bm.abs(bc[:, 1] - 0.0) < tol
    on_left = bm.abs(bc[:, 0] - 0.0) < tol
    return on_bot | on_left


def u_D(pts):
    """u = 0 on ΓD."""
    p = np.asarray(pts)
    return np.zeros(p.shape)


def solve_corner_incompat(N: int, p: int, *, mode: str, lambda0: float, lambda1: float,
                            skip_nn: bool = True):
    mesh = TriangleMesh.from_box([0., 1., 0., 1.], nx=N, ny=N)
    isNedge = build_isNedge_from_isD(mesh, isD_bot_left)
    base_space = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isNedge)
    space_u_sc = LagrangeFESpace(mesh, p=p - 1, ctype='D')
    space_u = TensorFunctionSpace(space_u_sc, shape=(-1, 2))

    @barycentric
    def coef(bcs, index=None):
        return bm.ones((1,) + bcs.shape[:-1], dtype=bm.float64)
    bform1 = BilinearForm(base_space)
    bform1.add_integrator(HuZhangStressIntegrator(coef=coef, lambda0=lambda0, lambda1=lambda1))
    bform2 = BilinearForm((space_u, base_space))
    bform2.add_integrator(HuZhangMixIntegrator())

    @cartesian
    def src(x, index=None):
        return bm.zeros(x.shape, dtype=x.dtype)
    L = LinearForm(space_u); L.add_integrator(VectorSourceIntegrator(source=src))
    b = L.assembly()

    HBC = HuzhangBoundaryCondition(space=base_space)
    def uD_bm(pts):
        return bm.tensor(u_D(np.asarray(pts)), dtype=base_space.ftype)
    a = HBC.displacement_boundary_condition(value=uD_bm, threshold=isD_bot_left)

    HSBC = HuzhangStressBoundaryCondition(space=base_space)
    def sig_bm(pts):
        return bm.tensor(sigma_g(np.asarray(pts)), dtype=base_space.ftype)
    uh_stress_base, isbd_stress_base = HSBC.set_essential_bc_v2(
        sig_bm, threshold=isNedge, coord='auto', skip_nn_corner_nodes=skip_nn)

    if mode == 'base':
        M = bform1.assembly().to_scipy().tocsr()
        B = bform2.assembly().to_scipy().tocsr()
        A = bmat([[M, B], [B.T, None]], format='csr')
        gdof_sig = base_space.number_of_global_dofs()
        F = np.zeros(A.shape[0])
        F[:gdof_sig] = np.asarray(a); F[gdof_sig:] = -np.asarray(b)
        uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof_sig] = np.asarray(uh_stress_base)
        isbd_g[:gdof_sig] = np.asarray(isbd_stress_base)
        ncp = 0; relax = None
    elif mode == 'relax':
        relax = HuZhangCornerRelax(mesh, p=p, isNedge=isNedge, base_space=base_space)
        ncp = relax.diag()['n_corners']
        M_unc = assemble_M_unc(relax, lambda0=lambda0, lambda1=lambda1, coef=coef)
        B_unc = assemble_B_unc(relax, space_u)
        C = relax.C_constraint
        nC = C.shape[0]; gdof_u = space_u.number_of_global_dofs()
        A = bmat([[M_unc, B_unc, C.T],
                  [B_unc.T, None, sp.csr_matrix((gdof_u, nC))],
                  [C, sp.csr_matrix((nC, gdof_u)), sp.csr_matrix((nC, nC))]], format='csr')
        gdof_sig = relax.gdof_unc
        # a → a_unc
        node2dof = np.asarray(base_space.dof.node_to_internal_dof())
        corner_base = set()
        for c in relax.corners:
            corner_base.update(node2dof[c.nid].tolist())
        a_np = np.asarray(a); a_unc = np.zeros(relax.gdof_unc)
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
        F[:gdof_sig] = a_unc; F[gdof_sig:gdof_sig + gdof_u] = -np.asarray(b)
        # essential BC 分两部分：
        # (1) 非角点 base DOFs：从 fealpy 的 uh_stress_base 直接复制；
        # (2) 角点 fan-end cells：wrapper 沿每条 ΓN 边的近端点评估 sigma_g，独立 lock。
        uh_base_np = np.asarray(uh_stress_base)
        isbd_base_np = np.asarray(isbd_stress_base)
        uh_unc = np.zeros(relax.gdof_unc)
        isbd_unc = np.zeros(relax.gdof_unc, dtype=bool)
        n2d_local = np.asarray(base_space.dof.node_to_internal_dof())
        corner_base_local = set()
        for c in relax.corners:
            corner_base_local.update(n2d_local[c.nid].tolist())
        for d in range(relax.gdof_base):
            if d in corner_base_local:
                continue
            if isbd_base_np[d]:
                uh_unc[d] = uh_base_np[d]
                isbd_unc[d] = True
        # 角点 fan-end lock (only σ_xx, σ_xy at cartesian nsframe)
        uh_c, isbd_c = relax.apply_corner_essential_bc_unc(lambda pt: sigma_g(np.asarray(pt)))
        for d in range(relax.gdof_unc):
            if isbd_c[d]:
                uh_unc[d] = uh_c[d]
                isbd_unc[d] = True
        uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof_sig] = uh_unc; isbd_g[:gdof_sig] = isbd_unc
    else:
        raise ValueError(mode)

    F = F - A @ uh_g; F[isbd_g] = uh_g[isbd_g]
    n = A.shape[0]; bdIdx = isbd_g.astype(int)
    A = spdiags(1 - bdIdx, 0, n, n) @ A @ spdiags(1 - bdIdx, 0, n, n) + spdiags(bdIdx, 0, n, n)
    X = spsolve(A, F)

    # 评估：σ_h 在 (1, 0) 附近两条边上是否分别 ≈ prescribed values
    node = np.asarray(mesh.entity('node'))
    cell = np.asarray(mesh.entity('cell'))
    corner_xy = np.array([1.0, 1.0])
    nid = int(np.argmin(np.linalg.norm(node - corner_xy, axis=1)))
    inc_cells = np.where(np.any(cell == nid, axis=1))[0]
    # 从 σ_h 读该 nid 处每个 cell 上的 σ 值
    if mode == 'base':
        sigmah = base_space.function(); sigmah[:] = X[:gdof_sig]
        vals = []
        for ic in inc_cells:
            loc = int(np.where(cell[ic] == nid)[0][0])
            bcs = np.zeros((1, 3)); bcs[0, loc] = 1.0
            sig_here = np.asarray(sigmah(bm.tensor(bcs)))[int(ic), 0]
            vals.append(sig_here)
    else:
        sig_unc = X[:gdof_sig]
        # 找哪个 corner 对应 (1,0)
        for c in relax.corners:
            if c.nid == nid:
                vals = [sig_unc[c.unc_dofs[k]] for k in range(len(c.cells))]
                break
        else:
            vals = []
    print(f'  σ_h at (1,0) in each incident cell:')
    for v in vals:
        print(f'    {np.asarray(v).round(4).tolist()}')
    # BC: bottom cell wants σ_xy = 1.0, right cell wants σ_xy = 2.0
    print(f'  prescribed: bottom→σ_xy={SIGMA_XY_BOT}, right→σ_xy={SIGMA_XY_RIGHT}')
    return ncp


def main(p: int = 3):
    lambda0, lambda1 = 4.0, 1.0
    for mode in ('base', 'relax'):
        print(f'\n=== mode={mode} ===')
        for N in [4, 8, 16]:
            print(f'N={N}:')
            ncp = solve_corner_incompat(N, p, mode=mode, lambda0=lambda0, lambda1=lambda1,
                                        skip_nn=False)  # skip=False 让 corner 强制 BC
            print(f'  NCP={ncp}')


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(p)
