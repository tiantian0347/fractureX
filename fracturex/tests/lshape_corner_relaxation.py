"""L 形域线弹性：验证 Hu-Zhang 角点松弛对 m=3 凹角是否生效 / 精度多少。

设计：
- 域：L = [-1,1]^2 \ ([0,1] x [0,1])，凹角在原点 (0,0)。
- 网格：先 from_box([-1,1]^2)，再删除中心在右上 quadrant 的单元。
- 解析解：取整体光滑 u（多项式），σ = c0 (c1 tr(eps) I + eps)，误差仅来自离散。
- 边界条件：
    - "DN 全 Dirichlet" 基线：全边界 ΓD ⇒ 没有 NN 角点 ⇒ 松弛与否结果应一致（也用于排除其他差异）。
    - "NN at re-entrant" 测试：凹角两条邻边（x=0,0<=y<=1 与 y=0,0<=x<=1）设为 ΓN，
      其余边界 ΓD。原点变成 NN 角点 ⇒ 触发松弛。
- 对 use_relaxation=False/True 各跑 maxit 次细化，打印 |σ-σh|, |u-uh| 与收敛阶。
- 同时打印 NCP（NN 角点数）以确认角点确实被识别。

[HM18] §5.2 预期：粗网格上误差差异最大；松弛 ON 在常数上明显下降。
"""
from __future__ import annotations
import sys
import numpy as np
from sympy import symbols, sin, cos, pi as sym_pi

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm, LinearForm
from fealpy.decorator import cartesian, barycentric
from scipy.sparse.linalg import spsolve as scipy_spsolve
from scipy.sparse import bmat, spdiags

from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition, HuzhangStressBoundaryCondition, build_isNedge_from_isD,
)
from linear_elastic_pde import LinearElasticPDE


def make_lshape_mesh(N: int):
    """Build an L-shape mesh via box minus top-right quadrant.

    Domain: [-1, 1]^2  \\  [0, 1] x [0, 1].  Re-entrant corner at (0, 0).
    N: subdivisions per unit length (must be even so x=0 / y=0 lie on edges).
    """
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
    """Mark ΓD = all boundary EXCEPT the two edges meeting at the re-entrant corner.

    Re-entrant corner is (0,0). Its two boundary edges are:
      - x=0, 0<=y<=1
      - y=0, 0<=x<=1
    Those two segments become ΓN; everything else ΓD.
    """
    tol = 1e-9
    x = bc[:, 0]
    y = bc[:, 1]
    on_xeq0 = (bm.abs(x) < tol) & (y > -tol) & (y < 1 + tol)
    on_yeq0 = (bm.abs(y) < tol) & (x > -tol) & (x < 1 + tol)
    isN = on_xeq0 | on_yeq0
    return ~isN


def solve(pde, N, p, *, use_relaxation: bool, isD_bd):
    mesh = make_lshape_mesh(N)
    isNedge = build_isNedge_from_isD(mesh, isD_bd)

    space0 = HuZhangFESpace2d(
        mesh, p=p,
        use_relaxation=use_relaxation,
        bd_stress=isNedge,
        debug=False,
    )
    NCP = int(space0.NCP)

    space_u_scalar = LagrangeFESpace(mesh, p=p - 1, ctype='D')
    space1 = TensorFunctionSpace(space_u_scalar, shape=(-1, 2))

    lambda0 = pde.lambda0
    lambda1 = pde.lambda1

    gdof0 = space0.number_of_global_dofs()

    @barycentric
    def coef(bcs, index=None):
        return bm.ones((1,) + bcs.shape[:-1], dtype=bm.float64)

    bform1 = BilinearForm(space0)
    bform1.add_integrator(HuZhangStressIntegrator(coef=coef, lambda0=lambda0, lambda1=lambda1))
    bform2 = BilinearForm((space1, space0))
    bform2.add_integrator(HuZhangMixIntegrator())

    M = bform1.assembly().to_scipy().tocsr()
    B = bform2.assembly().to_scipy().tocsr()
    TM = space0.TM.to_scipy().tocsr()

    M2 = TM.T @ M @ TM
    B2 = TM.T @ B
    A = bmat([[M2, B2], [B2.T, None]], format="csr")

    lform1 = LinearForm(space1)

    @cartesian
    def src(x, index=None):
        return pde.source(x)
    lform1.add_integrator(VectorSourceIntegrator(source=src))
    b = lform1.assembly()

    HBC = HuzhangBoundaryCondition(space=space0)
    a = HBC.displacement_boundary_condition(value=pde.displacement, threshold=isD_bd)

    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[:gdof0] = TM.T @ a
    F[gdof0:] = -b

    HSBC = HuzhangStressBoundaryCondition(space=space0)
    uh_stress, isbddof_stress = HSBC.set_essential_bc(pde.stress, threshold=isNedge, coord="auto")

    uh_global = bm.zeros(A.shape[0], dtype=A.dtype)
    uh_global[:gdof0] = uh_stress
    isbddof_global = bm.zeros(A.shape[0], dtype=bool)
    isbddof_global[:gdof0] = isbddof_stress

    F = F - A @ uh_global
    F[isbddof_global] = uh_global[isbddof_global]

    total_dof = A.shape[0]
    bdIdx = bm.zeros(total_dof, dtype=bm.int32)
    bdIdx[isbddof_global] = 1
    Tbd = spdiags(bdIdx, 0, total_dof, total_dof)
    T_ = spdiags(1 - bdIdx, 0, total_dof, total_dof)
    A = T_ @ A @ T_ + Tbd

    X = scipy_spsolve(A, F)
    sigmaval = TM @ X[:gdof0]
    uval = X[gdof0:]

    sigmah = space0.function(); sigmah[:] = sigmaval
    uh = space1.function();    uh[:] = uval
    return sigmah, uh, mesh, NCP


def run_case(label: str, pde, p: int, maxit: int, isD_bd, use_relaxation: bool):
    print(f"\n=== {label} | use_relaxation={use_relaxation} ===")
    print(f"{'N':>5} {'NCP':>5} {'|σ-σh|':>14} {'rate':>6} {'|u-uh|':>14} {'rate':>6}")
    es_prev = None; eu_prev = None
    for i in range(maxit):
        N = 2 ** (i + 1)
        sigmah, uh, mesh, NCP = solve(pde, N, p, use_relaxation=use_relaxation, isD_bd=isD_bd)
        e_sigma = float(mesh.error(sigmah, pde.stress))
        e_u = float(mesh.error(uh, pde.displacement))
        rs = '-' if es_prev is None else f"{np.log2(es_prev / e_sigma):.2f}"
        ru = '-' if eu_prev is None else f"{np.log2(eu_prev / e_u):.2f}"
        print(f"{N:>5} {NCP:>5} {e_sigma:>14.4e} {rs:>6} {e_u:>14.4e} {ru:>6}")
        es_prev, eu_prev = e_sigma, e_u


if __name__ == "__main__":
    lambda0 = 4.0
    lambda1 = 1.0
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    maxit = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    x, y = symbols('x y')
    u0 = sym_pi * sin(sym_pi * x) * sin(sym_pi * y)
    u1 = sym_pi * sin(sym_pi * x) * sin(sym_pi * y)
    pde = LinearElasticPDE([u0, u1], lambda0, lambda1)

    run_case("L-shape full Dirichlet (baseline)", pde, p, maxit, isD_full, use_relaxation=False)
    run_case("L-shape full Dirichlet (relax ON, no NN corner expected)", pde, p, maxit, isD_full, use_relaxation=True)
    run_case("L-shape NN at re-entrant (relax OFF)", pde, p, maxit, isD_reentrant_NN, use_relaxation=False)
    run_case("L-shape NN at re-entrant (relax ON)",  pde, p, maxit, isD_reentrant_NN, use_relaxation=True)
