"""L 形线弹性 — 用 fracturex 自家的 HuZhangCornerRelax wrapper 求解。

不动 fealpy，在 base 空间装配 M, B 之后，用 Q = P^T @ TM 把矩阵压到 relaxed 空间：

    M2 = Q^T M_base Q
    B2 = Q^T B_base

边界条件：
- 位移 Dirichlet（在线性形式右端）：保持原路径，再 Q^T 投影
- 应力本质 Neumann（在 stress DOF 上）：本测试用全 Dirichlet 与 mixed (NN 凹角) 两种
  - 全 Dirichlet：wrapper 检测 0 个 NN 角点，等价 base
  - mixed：凹角两条边为 ΓN，应力本质条件 ``HSBC.set_essential_bc`` 给出 isbddof_stress 与
    uh_stress（在 base 空间）；本脚本把它直接投影到 rel：rel 空间的"角点节点 3 个原 DOF"
    继承 base 上的值（cell-0 副本），其余角点 unc DOF（cells 1..m-1 副本）保持自由
    —— 由 TM 的零空间约束（法向连续）一致地通过 relaxed DOF 表达。

注意：本测试不在 NN 角点上单独施加角点 traction 值；fealpy 的 HSBC 不会专门处理角点节点的
两边 traction 不相容，松弛的全部价值就在于此。

跑：
    PYTHONPATH=... python lshape_corner_relax_solve.py 3 4
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
from scipy.sparse.linalg import spsolve as scipy_spsolve
from scipy.sparse import bmat, spdiags
import scipy.sparse as sp

from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition, HuzhangStressBoundaryCondition, build_isNedge_from_isD,
)
from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax
from linear_elastic_pde import LinearElasticPDE


def make_lshape_mesh(N):
    if N % 2 != 0:
        N += 1
    mesh = TriangleMesh.from_box([-1.0, 1.0, -1.0, 1.0], nx=N, ny=N)
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


def isD_full(bc):
    return bm.ones(bc.shape[0], dtype=bool)


def isD_reentrant_NN(bc):
    tol = 1e-9
    x = bc[:, 0]; y = bc[:, 1]
    on_xeq0 = (bm.abs(x) < tol) & (y > -tol) & (y < 1 + tol)
    on_yeq0 = (bm.abs(y) < tol) & (x > -tol) & (x < 1 + tol)
    return ~(on_xeq0 | on_yeq0)


def solve_with_wrapper(pde, N, p, *, isD_bd, mode: str):
    """mode: 'base' (no relaxation) or 'relax' (use wrapper)."""
    mesh = make_lshape_mesh(N)
    isNedge = build_isNedge_from_isD(mesh, isD_bd)

    base_space = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isNedge)
    space_u_scalar = LagrangeFESpace(mesh, p=p - 1, ctype='D')
    space1 = TensorFunctionSpace(space_u_scalar, shape=(-1, 2))

    lambda0 = pde.lambda0
    lambda1 = pde.lambda1
    gdof0 = base_space.number_of_global_dofs()

    @barycentric
    def coef(bcs, index=None):
        return bm.ones((1,) + bcs.shape[:-1], dtype=bm.float64)

    bform1 = BilinearForm(base_space)
    bform1.add_integrator(HuZhangStressIntegrator(coef=coef, lambda0=lambda0, lambda1=lambda1))
    bform2 = BilinearForm((space1, base_space))
    bform2.add_integrator(HuZhangMixIntegrator())

    lform1 = LinearForm(space1)
    @cartesian
    def src(x, index=None):
        return pde.source(x)
    lform1.add_integrator(VectorSourceIntegrator(source=src))
    b = lform1.assembly()

    HBC = HuzhangBoundaryCondition(space=base_space)
    a = HBC.displacement_boundary_condition(value=pde.displacement, threshold=isD_bd)

    HSBC = HuzhangStressBoundaryCondition(space=base_space)
    uh_stress_base, isbddof_stress_base = HSBC.set_essential_bc_v2(
        pde.stress, threshold=isNedge, coord='auto',
        skip_nn_corner_nodes=True,
    )

    if mode == 'base':
        M = bform1.assembly().to_scipy().tocsr()
        B = bform2.assembly().to_scipy().tocsr()
        relax = None
        A = bmat([[M, B], [B.T, None]], format='csr')
        gdof_sig = base_space.number_of_global_dofs()
        F = bm.zeros(A.shape[0], dtype=A.dtype)
        F[:gdof_sig] = a
        F[gdof_sig:] = -b
        uh_global = bm.zeros(A.shape[0], dtype=A.dtype)
        isbddof_global = bm.zeros(A.shape[0], dtype=bool)
        uh_global[:gdof_sig] = uh_stress_base
        isbddof_global[:gdof_sig] = isbddof_stress_base
        ncp = 0
    elif mode == 'relax':
        from fracturex.assemblers.huzhang_unc_assembler import assemble_M_unc, assemble_B_unc
        relax = HuZhangCornerRelax(mesh, p=p, isNedge=isNedge, base_space=base_space)
        ncp = relax.diag()['n_corners']
        M_unc = assemble_M_unc(relax, lambda0=lambda0, lambda1=lambda1, coef=coef)
        B_unc = assemble_B_unc(relax, space1)
        # KKT 形式：[M_unc  B_unc  C^T] [σ_unc]   [r_a_unc]
        #          [B_unc^T  0    0 ] [u   ] = [-b      ]
        #          [C       0    0 ] [λ   ]   [0       ]
        C = relax.C_constraint
        nC = C.shape[0]
        gdof_u_local = space1.number_of_global_dofs()
        zero_uC = sp.csr_matrix((gdof_u_local, nC))           # (gdof_u, nC)
        zero_Cu = sp.csr_matrix((nC, gdof_u_local))           # (nC, gdof_u)
        zero_CC = sp.csr_matrix((nC, nC))
        A = bmat([[M_unc, B_unc, C.T],
                  [B_unc.T, None, zero_uC],
                  [C, zero_Cu, zero_CC]], format='csr')
        gdof_sig = relax.gdof_unc
        # 右端：a 在 base 上，需要 P (gdof_base × gdof_unc) 的转置投到 unc：a_unc = P^T a？
        # 不对：a 是 base 的右端项 (∫ τn · u_D ds)，τ 是 base 张量基；σ_unc 中每个 cell-local
        # 副本对应 base 张量基（同形状）。每个 cell 上 ∫(τn·u_D) 局部计算用的就是 base 张量基；
        # 把 base 全局右端 a 散到 unc 上：对非角点 base id 复制，对角点 base 3 id 把值复制到
        # 所有 cell-local 副本。这与 P^T 等价（P 是 unc→base 累加，所以 P^T 是 base→unc 复制）。
        # 用 wrapper 现成的 P^T 投影：a 在 base 上，a_unc = P^T @ a。
        # P = Q @ inv(TM) 这条路不通；直接用 Q.T 的话 Q = P @ TM → Q.T = TM.T @ P.T，不是 P.T。
        # 简单：自己写 P_T
        node2dof = np.asarray(base_space.dof.node_to_internal_dof())
        corner_base_ids = set()
        for c in relax.corners:
            corner_base_ids.update(node2dof[c.nid].tolist())
        a_unc = np.zeros(relax.gdof_unc)
        for d in range(relax.gdof_base):
            if d in corner_base_ids:
                continue
            a_unc[d] = float(a[d])
        for c in relax.corners:
            base3 = node2dof[c.nid]
            for comp in range(3):
                bd = int(base3[comp])
                for k in range(len(c.cells)):
                    a_unc[int(c.unc_dofs[k, comp])] = float(a[bd])
        F = bm.zeros(A.shape[0], dtype=A.dtype)
        F[:gdof_sig] = a_unc
        F[gdof_sig:gdof_sig + space1.number_of_global_dofs()] = -b
        # λ 段右端 = 0 (法向连续约束)
        # essential bc：在 unc 上施加。非角点 base id 直接锁；角点 base 3 id 复制到所有 cell 副本。
        uh_unc, isbd_unc = relax.lift_base_bc_to_unc(uh_stress_base, isbddof_stress_base)
        uh_global = bm.zeros(A.shape[0], dtype=A.dtype)
        isbddof_global = bm.zeros(A.shape[0], dtype=bool)
        uh_global[:gdof_sig] = uh_unc
        isbddof_global[:gdof_sig] = isbd_unc
    else:
        raise ValueError(mode)

    F = F - A @ uh_global
    F[isbddof_global] = uh_global[isbddof_global]
    total_dof = A.shape[0]
    bdIdx = bm.zeros(total_dof, dtype=bm.int32)
    bdIdx[isbddof_global] = 1
    Tbd = spdiags(bdIdx, 0, total_dof, total_dof)
    T_ = spdiags(1 - bdIdx, 0, total_dof, total_dof)
    A = T_ @ A @ T_ + Tbd

    X = scipy_spsolve(A, F)
    sigma_sol = X[:gdof_sig]
    uval = X[gdof_sig:gdof_sig + space1.number_of_global_dofs()]

    uh = space1.function(); uh[:] = uval
    if mode == 'base':
        sigmah = base_space.function(); sigmah[:] = sigma_sol
        return sigmah, uh, mesh, ncp, None
    else:
        sigmah_avg = base_space.function()
        sigmah_avg[:] = relax.lift_stress_base_averaged_from_unc(sigma_sol) \
            if hasattr(relax, 'lift_stress_base_averaged_from_unc') \
            else _avg_from_unc(relax, sigma_sol)
        return sigmah_avg, uh, mesh, ncp, (relax, sigma_sol, 'unc')


def _avg_from_unc(relax, sigma_unc):
    """Average corner cell-local copies back to base, identity elsewhere."""
    node2dof = np.asarray(relax.base_space.dof.node_to_internal_dof())
    sigma_base = np.zeros(relax.gdof_base)
    corner_base_ids = set()
    for c in relax.corners:
        corner_base_ids.update(node2dof[c.nid].tolist())
    for d in range(relax.gdof_base):
        if d in corner_base_ids:
            continue
        sigma_base[d] = sigma_unc[d]
    for c in relax.corners:
        base3 = node2dof[c.nid]
        m = len(c.cells)
        for comp in range(3):
            vals = [float(sigma_unc[int(c.unc_dofs[k, comp])]) for k in range(m)]
            sigma_base[int(base3[comp])] = float(np.mean(vals))
    return sigma_base


def run(label, pde, p, maxit, isD_bd, mode):
    print(f"\n=== {label} | mode={mode} ===")
    print(f"{'N':>5} {'NCP':>5} {'|σ-σh|':>14} {'rate':>6} {'|u-uh|':>14} {'rate':>6}")
    es_prev = None; eu_prev = None
    for i in range(maxit):
        N = 2 ** (i + 1)
        sigmah, uh, mesh, ncp, extra = solve_with_wrapper(pde, N, p, isD_bd=isD_bd, mode=mode)
        # 用 mesh.error 评估 averaged σ（一致度量）
        e_sigma = float(mesh.error(sigmah, pde.stress, q=2*p+6))
        e_u = float(mesh.error(uh, pde.displacement, q=2*p+6))
        rs = '-' if es_prev is None else f"{np.log2(es_prev / e_sigma):.2f}"
        ru = '-' if eu_prev is None else f"{np.log2(eu_prev / e_u):.2f}"
        print(f"{N:>5} {ncp:>5} {e_sigma:>14.4e} {rs:>6} {e_u:>14.4e} {ru:>6}")
        es_prev, eu_prev = e_sigma, e_u


if __name__ == "__main__":
    lambda0 = 4.0
    lambda1 = 1.0
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    maxit = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    x, y = symbols('x y')
    # 用 (sin(pi*(x+1)/2)*sin(pi*(y+1)/2))^2 让 σn 在 L 形整个外边界上 = 0
    # （函数在 x=±1 或 y=±1 处为 0 且导数为 0），凹角两条边 (x=0, y=0) 上仍然有非零 σn
    u0 = (sin(sym_pi*(x+1)/2) * sin(sym_pi*(y+1)/2))**2
    u1 = (sin(sym_pi*(x+1)/2) * sin(sym_pi*(y+1)/2))**2
    pde = LinearElasticPDE([u0, u1], lambda0, lambda1)

    run("L-shape full Dirichlet baseline",    pde, p, maxit, isD_full, mode='base')
    run("L-shape full Dirichlet (wrapper, NCP=0 expected)", pde, p, maxit, isD_full, mode='relax')
    run("L-shape NN at re-entrant (base, no relax)", pde, p, maxit, isD_reentrant_NN, mode='base')
    run("L-shape NN at re-entrant (wrapper)",        pde, p, maxit, isD_reentrant_NN, mode='relax')
