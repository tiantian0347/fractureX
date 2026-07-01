"""L 形凹角自适应循环：oracle-based marking，对比 base 与 wrapper 在自适应下的常数改善。

设置:
    Ω = [-1,1]^2 \\ [0,1]^2      (L 形，凹角在 (0,0)，interior angle 3π/2)
    ΓD = 内部 L-arm 上的凹角两条边：y=0 (x∈[0,1]) 与 x=0 (y∈[0,1])
    ΓN = 其余边界，σn 从解析 σ 计算
    解析解: u = (sin(π(x+1)/2)·sin(π(y+1)/2))^2 双分量 (quartic，边界 σn 非零)

指示子:
    per-cell L2 error against σ_exact (oracle)——直接量真误差，取代 estimator，
    让自适应循环用"完美"的标记，纯粹对比 base vs wrapper 在同一网格序列上的结果。

标记:
    Dörfler bulk marking (fealpy Mesh.mark, method='L2', θ=0.3)

加密:
    Bisect (fealpy TriangleMesh.bisect)

比较 base 模式与 wrapper (KKT + A 方案) 在 5 轮自适应下的 (DOF, |σ-σh|_L2) 序列。
"""
from __future__ import annotations
import sys
import numpy as np
from sympy import symbols, sin, pi as sym_pi
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

sys.path.insert(0, '/Users/tian00/repository/fractureX/fracturex/tests')
from linear_elastic_pde import LinearElasticPDE


LAMBDA0 = 4.0
LAMBDA1 = 1.0


def make_lshape_init(N: int):
    """L 形 = [-1,1]² \\ [0,1]²，凹角在 (0,0)。"""
    if N % 2 != 0:
        N += 1
    mesh = TriangleMesh.from_box([-1., 1., -1., 1.], nx=N, ny=N)
    cell = mesh.entity('cell')
    node = mesh.entity('node')
    bary = node[cell].mean(axis=1)
    keep = ~((bary[:, 0] > 0) & (bary[:, 1] > 0))
    new_cell = cell[keep]
    used = bm.unique(new_cell.reshape(-1))
    remap = -bm.ones(node.shape[0], dtype=new_cell.dtype)
    remap[used] = bm.arange(used.shape[0], dtype=new_cell.dtype)
    new_node = node[used]
    new_cell = remap[new_cell]
    return TriangleMesh(new_node, new_cell)


def isD_reentrant(bc):
    """凹角两条 arm 边界: y=0 (x∈[0,1]) 与 x=0 (y∈[0,1]) 作为 ΓN"""
    tol = 1e-9
    x = bc[:, 0]; y = bc[:, 1]
    on_xeq0 = (bm.abs(x) < tol) & (y > -tol) & (y < 1 + tol)
    on_yeq0 = (bm.abs(y) < tol) & (x > -tol) & (x < 1 + tol)
    return ~(on_xeq0 | on_yeq0)


def solve_on_mesh(mesh, pde, p, *, mode: str):
    """单网格上 solve 一次。返回 (per-cell L2 error, total error, ncp, dof)."""
    lambda0 = getattr(pde, 'lambda0', LAMBDA0)
    lambda1 = getattr(pde, 'lambda1', LAMBDA1)
    isNedge = build_isNedge_from_isD(mesh, isD_reentrant)
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
        return pde.source(x)
    L = LinearForm(space_u); L.add_integrator(VectorSourceIntegrator(source=src))
    b = L.assembly()

    HBC = HuzhangBoundaryCondition(space=base_space)
    a = HBC.displacement_boundary_condition(value=pde.displacement, threshold=isD_reentrant)

    HSBC = HuzhangStressBoundaryCondition(space=base_space)
    uh_stress_base, isbd_stress_base = HSBC.set_essential_bc_v2(
        pde.stress, threshold=isNedge, coord='auto', skip_nn_corner_nodes=True)

    if mode == 'base':
        M = bform1.assembly().to_scipy().tocsr()
        B = bform2.assembly().to_scipy().tocsr()
        A = bmat([[M, B], [B.T, None]], format='csr')
        gdof_sig = base_space.number_of_global_dofs()
        F = np.zeros(A.shape[0]); F[:gdof_sig] = np.asarray(a); F[gdof_sig:] = -np.asarray(b)
        uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof_sig] = np.asarray(uh_stress_base); isbd_g[:gdof_sig] = np.asarray(isbd_stress_base)
        ncp = 0; relax = None
        total_dof = A.shape[0]
    elif mode == 'relax':
        relax = HuZhangCornerRelax(mesh, p=p, isNedge=isNedge, base_space=base_space)
        ncp = relax.diag()['n_corners']
        M_unc = assemble_M_unc(relax, lambda0=lambda0, lambda1=lambda1, coef=coef)
        B_unc = assemble_B_unc(relax, space_u)
        C = relax.C_constraint; nC = C.shape[0]; gdof_u = space_u.number_of_global_dofs()
        A = bmat([[M_unc, B_unc, C.T],
                  [B_unc.T, None, sp.csr_matrix((gdof_u, nC))],
                  [C, sp.csr_matrix((nC, gdof_u)), sp.csr_matrix((nC, nC))]], format='csr')
        gdof_sig = relax.gdof_unc
        n2d = np.asarray(base_space.dof.node_to_internal_dof())
        corner_base = set()
        for c in relax.corners:
            corner_base.update(n2d[c.nid].tolist())
        a_np = np.asarray(a); a_unc = np.zeros(relax.gdof_unc)
        for d in range(relax.gdof_base):
            if d in corner_base:
                continue
            a_unc[d] = a_np[d]
        for c in relax.corners:
            base3 = n2d[c.nid]
            for comp in range(3):
                bd = int(base3[comp])
                for k in range(len(c.cells)):
                    a_unc[int(c.unc_dofs[k, comp])] = a_np[bd]
        F = np.zeros(A.shape[0]); F[:gdof_sig] = a_unc; F[gdof_sig:gdof_sig + gdof_u] = -np.asarray(b)
        # A 方案：不锁角点 essential
        uh_base_np = np.asarray(uh_stress_base); isbd_base_np = np.asarray(isbd_stress_base)
        uh_unc = np.zeros(relax.gdof_unc); isbd_unc = np.zeros(relax.gdof_unc, dtype=bool)
        for d in range(relax.gdof_base):
            if d in corner_base:
                continue
            if isbd_base_np[d]:
                uh_unc[d] = uh_base_np[d]; isbd_unc[d] = True
        uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof_sig] = uh_unc; isbd_g[:gdof_sig] = isbd_unc
        total_dof = A.shape[0]
    else:
        raise ValueError(mode)

    F = F - A @ uh_g; F[isbd_g] = uh_g[isbd_g]
    bdIdx = isbd_g.astype(int)
    A = spdiags(1 - bdIdx, 0, total_dof, total_dof) @ A @ spdiags(1 - bdIdx, 0, total_dof, total_dof) \
        + spdiags(bdIdx, 0, total_dof, total_dof)
    X = spsolve(A, F)

    if mode == 'base':
        sigmah = base_space.function(); sigmah[:] = X[:gdof_sig]
        eta_per_cell = np.asarray(mesh.error(sigmah, pde.stress, q=2 * p + 6, celltype=True))
    else:
        # KKT 求解出的是 σ_unc；直接调 unc-aware 版本平均回 base
        sig_unc = X[:gdof_sig]
        # 手写：非角点直接取；角点 base 3 id ← 平均各 cell 副本
        n2d_ = np.asarray(base_space.dof.node_to_internal_dof())
        corner_base_ = set()
        for c in relax.corners:
            corner_base_.update(n2d_[c.nid].tolist())
        sig_avg = np.zeros(relax.gdof_base)
        for d in range(relax.gdof_base):
            if d in corner_base_:
                continue
            sig_avg[d] = sig_unc[d]
        for c in relax.corners:
            base3 = n2d_[c.nid]; m = len(c.cells)
            for comp in range(3):
                vals = [float(sig_unc[int(c.unc_dofs[k, comp])]) for k in range(m)]
                sig_avg[int(base3[comp])] = float(np.mean(vals))
        sigmah = base_space.function(); sigmah[:] = sig_avg
        eta_per_cell = np.asarray(mesh.error(sigmah, pde.stress, q=2 * p + 6, celltype=True))

    total_err = float(np.sqrt(np.sum(eta_per_cell ** 2)))
    return eta_per_cell, total_err, ncp, gdof_sig


def adaptive_loop(pde, p, mode, N0, n_iter, theta=0.3):
    print(f"\n=== L-shape adaptive | mode={mode} | θ={theta} ===")
    print(f"{'iter':>4} {'DOFσ':>7} {'NCP':>4} {'NC':>6} {'|σ-σh|':>12}")
    mesh = make_lshape_init(N0)
    for it in range(n_iter):
        eta, e, ncp, dof = solve_on_mesh(mesh, pde, p, mode=mode)
        NC = mesh.number_of_cells()
        print(f"{it:>4} {dof:>7} {ncp:>4} {NC:>6} {e:>12.4e}")
        if it == n_iter - 1:
            break
        isMarked = np.asarray(TriangleMesh.mark(bm.tensor(eta), theta, method='L2'))
        mesh.bisect(isMarked, options={'disp': False})


def main(p=3, n_iter=5, theta=0.3):
    """使用 Williams 奇异解（真 σ 奇异）驱动自适应."""
    from hm18_williams_singular import (
        displacement_exact as w_disp, stress_exact as w_stress,
        LAMBDA0 as W_L0, LAMBDA1 as W_L1, make_rotated_lshape_mesh, isD_around_reentrant,
    )
    # 复用 Williams 的 pde 接口
    class WilliamsPDE:
        lambda0 = W_L0; lambda1 = W_L1
        @staticmethod
        @cartesian
        def source(x, index=None):
            return bm.zeros(x.shape, dtype=x.dtype)
        @staticmethod
        def stress(pts):
            return bm.tensor(w_stress(np.asarray(pts)), dtype=bm.float64)
        @staticmethod
        def displacement(pts):
            return bm.tensor(w_disp(np.asarray(pts)), dtype=bm.float64)
    pde = WilliamsPDE
    # 用 rotated L-shape + Williams 的 ΓD
    global make_lshape_init, isD_reentrant, LAMBDA0, LAMBDA1
    make_lshape_init_backup = make_lshape_init
    make_lshape_init = make_rotated_lshape_mesh
    isD_reentrant_backup = isD_reentrant
    isD_reentrant = isD_around_reentrant
    LAMBDA0 = W_L0
    LAMBDA1 = W_L1
    for mode in ('base', 'relax'):
        adaptive_loop(pde, p, mode, N0=4, n_iter=n_iter, theta=theta)


if __name__ == '__main__':
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    main(p, n_iter, theta)
