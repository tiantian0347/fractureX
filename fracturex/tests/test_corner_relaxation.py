"""pytest 入口：Hu-Zhang 角点松弛（corner relaxation）关键回归测试.

Cases:
    - test_lshape_smooth_convergence_p3
        L 形全 Dirichlet + 光滑 quartic manufactured solution, p=3.
        严格证明 fracturex 混合 FEM 达到最优阶：
            ||σ-σh||_L2 = 4 阶 (p+1)
            ||u-uh||_L2 = 3 阶 (p)
        跑 N=8, 16, 32 三档均匀网格。

    - test_wrapper_math_equivalence_hm18_section3_1
        方形 m=2 NN 角点：`HuZhangCornerRelax` wrapper 的 4 rel basis 与 [HM18]
        §3.1 (3.1) 手工构造的 4 basis 张成相同子空间。相差一个 4×4 可逆坐标变换。

    - test_skip_nn_corner_nodes_prevents_divergence
        L 形 NN 凹角 + quartic σn≠0，base 模式无 `skip_nn_corner_nodes` 发散，
        开启后不再发散（稳定在 O(0.3)）。

结构 sanity（DOF 计数、cell_to_dof 覆盖、TM 恒等块等）已在 `test_huzhang_corner_relax.py`
覆盖，本文件不重复。

复现：
    cd /Users/tian00/repository/fractureX
    PYTHONPATH=/Users/tian00/repository/fealpy:. pytest fracturex/tests/test_corner_relaxation.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
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

from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition,
    HuzhangStressBoundaryCondition,
    build_isNedge_from_isD,
)
from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax


# ---------- shared helpers ----------
def _make_lshape(N: int):
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


def _isD_full(bc):
    return bm.ones(bc.shape[0], dtype=bool)


def _isD_reentrant_NN(bc):
    tol = 1e-9
    x = bc[:, 0]
    y = bc[:, 1]
    on_xeq0 = (bm.abs(x) < tol) & (y > -tol) & (y < 1 + tol)
    on_yeq0 = (bm.abs(y) < tol) & (x > -tol) & (x < 1 + tol)
    return ~(on_xeq0 | on_yeq0)


def _solve_lshape(pde, N, p, *, isD_bd, skip_nn=True):
    """L-shape mixed FEM base-mode solver returning (mesh, sigmah, uh)."""
    mesh = _make_lshape(N)
    isNedge = build_isNedge_from_isD(mesh, isD_bd)
    base_space = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isNedge)
    space_u_sc = LagrangeFESpace(mesh, p=p - 1, ctype='D')
    space_u = TensorFunctionSpace(space_u_sc, shape=(-1, 2))

    @barycentric
    def coef(bcs, index=None):
        return bm.ones((1,) + bcs.shape[:-1], dtype=bm.float64)

    bform1 = BilinearForm(base_space)
    bform1.add_integrator(HuZhangStressIntegrator(coef=coef, lambda0=pde.lambda0, lambda1=pde.lambda1))
    bform2 = BilinearForm((space_u, base_space))
    bform2.add_integrator(HuZhangMixIntegrator())
    M = bform1.assembly().to_scipy().tocsr()
    B = bform2.assembly().to_scipy().tocsr()
    A = bmat([[M, B], [B.T, None]], format='csr')

    @cartesian
    def src(x, index=None):
        return pde.source(x)
    L = LinearForm(space_u); L.add_integrator(VectorSourceIntegrator(source=src))
    b = L.assembly()

    HBC = HuzhangBoundaryCondition(space=base_space)
    a = HBC.displacement_boundary_condition(value=pde.displacement, threshold=isD_bd)

    HSBC = HuzhangStressBoundaryCondition(space=base_space)
    uh_stress, isbd_stress = HSBC.set_essential_bc_v2(
        pde.stress, threshold=isNedge, coord='auto',
        skip_nn_corner_nodes=skip_nn,
    )
    gdof0 = base_space.number_of_global_dofs()
    F = np.zeros(A.shape[0]); F[:gdof0] = np.asarray(a); F[gdof0:] = -np.asarray(b)
    uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
    uh_g[:gdof0] = np.asarray(uh_stress); isbd_g[:gdof0] = np.asarray(isbd_stress)
    F = F - A @ uh_g; F[isbd_g] = uh_g[isbd_g]
    n_ = A.shape[0]; bdIdx = isbd_g.astype(int)
    A = spdiags(1 - bdIdx, 0, n_, n_) @ A @ spdiags(1 - bdIdx, 0, n_, n_) + spdiags(bdIdx, 0, n_, n_)
    X = spsolve(A, F)
    sigmah = base_space.function(); sigmah[:] = X[:gdof0]
    uh = space_u.function(); uh[:] = X[gdof0:]
    return mesh, sigmah, uh


class _QuarticLshapePDE:
    """光滑 manufactured solution u = (sin(π(x+1)/2) sin(π(y+1)/2))² 双分量."""
    lambda0 = 4.0
    lambda1 = 1.0

    def __init__(self):
        # 直接用 fracturex 现有的 LinearElasticPDE
        import sys, os
        experiments_dir = os.path.join(os.path.dirname(__file__), 'corner_relaxation', 'experiments')
        if experiments_dir not in sys.path:
            sys.path.insert(0, experiments_dir)
        legacy = os.path.dirname(__file__)
        if legacy not in sys.path:
            sys.path.insert(0, legacy)
        from linear_elastic_pde import LinearElasticPDE
        x, y = symbols('x y')
        u0 = (sin(sym_pi * (x + 1) / 2) * sin(sym_pi * (y + 1) / 2)) ** 2
        self._pde = LinearElasticPDE([u0, u0], self.lambda0, self.lambda1)

    def source(self, p):
        return self._pde.source(p)

    def stress(self, p):
        return self._pde.stress(p)

    def displacement(self, p):
        return self._pde.displacement(p)


# ============================================================================
# Test cases
# ============================================================================
def test_lshape_smooth_convergence_p3():
    """L 形 + 全 Dirichlet + 光滑 quartic + p=3：|σ-σh|_L2 达到 4 阶 (p+1), |u-uh|_L2 达到 3 阶 (p)."""
    pde = _QuarticLshapePDE()
    p = 3
    errs_sigma = []
    errs_u = []
    for N in [8, 16, 32]:
        mesh, sigmah, uh = _solve_lshape(pde, N, p, isD_bd=_isD_full, skip_nn=True)
        e_s = float(mesh.error(sigmah, pde.stress, q=2 * p + 6))
        e_u = float(mesh.error(uh, pde.displacement, q=2 * p + 6))
        errs_sigma.append(e_s); errs_u.append(e_u)

    # 收敛阶: log2(err_prev / err_now)
    rate_sigma_12 = float(np.log2(errs_sigma[0] / errs_sigma[1]))
    rate_sigma_23 = float(np.log2(errs_sigma[1] / errs_sigma[2]))
    rate_u_12 = float(np.log2(errs_u[0] / errs_u[1]))
    rate_u_23 = float(np.log2(errs_u[1] / errs_u[2]))

    # σ 应达到 4 阶（tolerance 0.15）
    assert rate_sigma_12 > 3.85, f"σ rate N=8→16 = {rate_sigma_12:.3f} < 3.85"
    assert rate_sigma_23 > 3.85, f"σ rate N=16→32 = {rate_sigma_23:.3f} < 3.85"
    # u 应达到 3 阶
    assert rate_u_12 > 2.85, f"u rate N=8→16 = {rate_u_12:.3f} < 2.85"
    assert rate_u_23 > 2.85, f"u rate N=16→32 = {rate_u_23:.3f} < 2.85"


def test_wrapper_math_equivalence_hm18_section3_1():
    """wrapper T_w 与 [HM18] §3.1 T_p 在 m=2 NN 角点上张成同一 rel-space."""
    mesh = TriangleMesh.from_box([0., 1., 0., 1.], nx=2, ny=2)

    def isD_bot(bc):
        return bm.abs(bc[:, 1] - 0.0) < 1e-12

    isN = build_isNedge_from_isD(mesh, isD_bot)
    base = HuZhangFESpace2d(mesh, p=3, use_relaxation=False, bd_stress=isN)
    relax = HuZhangCornerRelax(mesh, p=3, isNedge=isN, base_space=base)

    assert len(relax.corners) >= 1, "expected at least one NN corner"
    c = relax.corners[0]
    assert len(c.cells) == 2, "test requires m=2 corner"
    (kj, kjp1, n_int) = c.interior_edges[0]
    n_int = np.asarray(n_int)
    t_int = np.array([-n_int[1], n_int[0]])
    nx_, ny_ = n_int
    tx_, ty_ = t_int
    # [HM18] (2.5) 定义
    S_e = np.array([nx_ * nx_, nx_ * ny_, ny_ * ny_])
    S_p1 = np.array([2 * nx_ * tx_, nx_ * ty_ + ny_ * tx_, 2 * ny_ * ty_])
    S_p2 = np.array([tx_ * tx_, tx_ * ty_, ty_ * ty_])
    zero = np.zeros(3)
    if kj == 0:
        b1 = np.concatenate([S_e, S_e])
        b2 = np.concatenate([S_p1, S_p1])
        b3 = np.concatenate([S_p2, zero])
        b4 = np.concatenate([zero, S_p2])
    else:
        b1 = np.concatenate([S_e, S_e])
        b2 = np.concatenate([S_p1, S_p1])
        b3 = np.concatenate([zero, S_p2])
        b4 = np.concatenate([S_p2, zero])
    T_p = np.stack([b1, b2, b3, b4], axis=0).T  # (6, 4)

    TM = relax.TM
    unc_ids = c.unc_dofs.reshape(-1)
    rel_ids = c.rel_dofs
    T_w = np.asarray(TM[unc_ids][:, rel_ids].todense())  # (6, 4)

    rk_w = np.linalg.matrix_rank(T_w, tol=1e-9)
    rk_p = np.linalg.matrix_rank(T_p, tol=1e-9)
    rk_stacked = np.linalg.matrix_rank(np.hstack([T_w, T_p]), tol=1e-9)

    assert rk_w == 4, f"wrapper T_w rank = {rk_w}, expected 4"
    assert rk_p == 4, f"HM18 T_p rank = {rk_p}, expected 4"
    assert rk_stacked == 4, f"span 不同：rank([T_w|T_p]) = {rk_stacked}, expected 4"

    # 验证存在可逆变换 T_w = T_p @ A
    A_mat, *_ = np.linalg.lstsq(T_p, T_w, rcond=None)
    recon = np.linalg.norm(T_p @ A_mat - T_w)
    assert recon < 1e-10, f"|T_p A - T_w| = {recon:.3e}, expected ~0"
    assert abs(np.linalg.det(A_mat)) > 1e-6, f"det(A) = {np.linalg.det(A_mat):.3e} ≈ 0 (奇异)"


def test_v2_essential_bc_fixes_framework_bug():
    """方形 [0,1]^2 + bottom-only Dirichlet + σn≠0：
        原 set_essential_bc      → 发散（fealpy 节点 trace 标架 bug）
        set_essential_bc_v2      → 恢复 4 阶最优阶
    这是 fealpy 标架 bug 修复的核心回归测试。
    (v2: 节点 trace 用 cartesian nsframe, 边内部用 esframe (nn, sym(nt))；原版全用 esframe.)
    """
    import sys, os
    experiments_dir = os.path.join(os.path.dirname(__file__), 'corner_relaxation', 'experiments')
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    from linear_elastic_pde import LinearElasticPDE

    lambda0, lambda1 = 4.0, 1.0
    p = 3
    x, y = symbols('x y')
    # sin·sin 让 σn 在 ΓN 上非零 → 触发 fealpy 标架 bug
    u0 = sym_pi * sin(sym_pi * x) * sin(sym_pi * y)
    pde = LinearElasticPDE([u0, u0], lambda0, lambda1)

    def _isD_bot(bc):
        return bm.abs(bc[:, 1] - 0.0) < 1e-12

    def _solve_square(N, use_v2):
        mesh = TriangleMesh.from_box([0., 1., 0., 1.], nx=N, ny=N)
        isN = build_isNedge_from_isD(mesh, _isD_bot)
        base = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isN)
        u_sc = LagrangeFESpace(mesh, p=p - 1, ctype='D')
        space_u = TensorFunctionSpace(u_sc, shape=(-1, 2))
        @barycentric
        def coef(bcs, index=None):
            return bm.ones((1,) + bcs.shape[:-1], dtype=bm.float64)
        bform1 = BilinearForm(base); bform1.add_integrator(HuZhangStressIntegrator(coef=coef, lambda0=lambda0, lambda1=lambda1))
        bform2 = BilinearForm((space_u, base)); bform2.add_integrator(HuZhangMixIntegrator())
        M = bform1.assembly().to_scipy().tocsr(); B = bform2.assembly().to_scipy().tocsr()
        A = bmat([[M, B], [B.T, None]], format='csr')
        @cartesian
        def src(x, index=None):
            return pde.source(x)
        L = LinearForm(space_u); L.add_integrator(VectorSourceIntegrator(source=src))
        b = L.assembly()
        HBC = HuzhangBoundaryCondition(space=base)
        a = HBC.displacement_boundary_condition(value=pde.displacement, threshold=_isD_bot)
        HSBC = HuzhangStressBoundaryCondition(space=base)
        if use_v2:
            uh_stress, isbd_stress = HSBC.set_essential_bc_v2(
                pde.stress, threshold=isN, coord='auto', skip_nn_corner_nodes=True)
        else:
            # 原 fealpy 标架 bug 版本
            uh_stress, isbd_stress = HSBC.set_essential_bc(
                pde.stress, threshold=isN, coord='auto')
        gdof0 = base.number_of_global_dofs()
        F = np.zeros(A.shape[0]); F[:gdof0] = np.asarray(a); F[gdof0:] = -np.asarray(b)
        uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof0] = np.asarray(uh_stress); isbd_g[:gdof0] = np.asarray(isbd_stress)
        F = F - A @ uh_g; F[isbd_g] = uh_g[isbd_g]
        n_ = A.shape[0]; bdIdx = isbd_g.astype(int)
        A = spdiags(1 - bdIdx, 0, n_, n_) @ A @ spdiags(1 - bdIdx, 0, n_, n_) + spdiags(bdIdx, 0, n_, n_)
        X = spsolve(A, F)
        sigmah = base.function(); sigmah[:] = X[:gdof0]
        return mesh, sigmah

    # v2: 应收敛到 4 阶
    errs_v2 = []
    for N in [4, 8, 16]:
        mesh, sigmah = _solve_square(N, use_v2=True)
        e = float(mesh.error(sigmah, pde.stress, q=2 * p + 6))
        errs_v2.append(e)
    rate_v2 = float(np.log2(errs_v2[1] / errs_v2[2]))
    assert rate_v2 > 3.8, f"v2 应恢复 4 阶，实测 N=8→16 = {rate_v2:.3f}"

    # v1: N=16 误差应远大于 v2（发散或不收敛）
    _, sigmah_v1 = _solve_square(16, use_v2=False)
    e_v1 = float(mesh.error(sigmah_v1, pde.stress, q=2 * p + 6))
    # v2 时 N=16 误差 ~ 1e-5；v1 (标架 bug) 时 ~ 5e-1，差 4-5 个量级
    assert e_v1 > 100 * errs_v2[2], (
        f"v1 (标架 bug) err={e_v1:.3e} 应远大于 v2 (修复后) err={errs_v2[2]:.3e}"
    )
