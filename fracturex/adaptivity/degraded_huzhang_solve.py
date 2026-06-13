"""退化 Hu–Zhang 混合求解器（T6 门槛专用）：出平衡应力 σ_h ∈ Σ_f。

理论见 docs/adaptive/THEORY_equilibrated_aposteriori.md。
混合 Hellinger–Reissner 格式，柔度块注入退化系数 g(d)^{-1}：
  弱形式  ∫ τ : A(d) σ  +  ∫ div τ · u  =  <边界>
  其中 A(d) = g(d)^{-1} C^{-1}，C^{-1} 由 (lambda0, lambda1) 编码：
    ∫ τ:C^{-1}τ = lambda0 ∫ τ:τ - lambda1 ∫ (tr τ)²,  平面应变
    lambda0 = 1/(2μ),  lambda1 = λ/(2μ(2λ+2μ))
  HuZhangStressIntegrator 的标量 coef 逐 qp 乘进柔度块 ⇒ 设 coef=g(d)^{-1} 即注入退化。

基于 tests/linear_elastic_with_huzhang.py::solve 的可用装配/边界处理，
参数化为退化 + 任意 Dirichlet 边界。线性求解走 scipy（第三方边界，允许 numpy）。
不修改 fealpy。
"""
from __future__ import annotations

from typing import Callable

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import BilinearForm, LinearForm, VectorSourceIntegrator
from fealpy.decorator import cartesian, barycentric

from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition, HuzhangStressBoundaryCondition,
    build_isNedge_from_isD,
)

from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve as scipy_spsolve


def compliance_lame_coeffs(lam: float, mu: float):
    """柔度弱形式系数 (lambda0, lambda1)，平面应变。

    ∫ τ:C^{-1}τ = lambda0 ∫ τ:τ - lambda1 ∫ (tr τ)²。
    输入: lam,mu Lamé；输出: (lambda0, lambda1)。
    """
    lambda0 = 1.0 / (2.0 * mu)
    lambda1 = lam / (2.0 * mu * (2.0 * lam + 2.0 * mu))
    return lambda0, lambda1


def solve_degraded_huzhang(mesh, p: int, pde, *, lam: float, mu: float, k_res: float,
                           isD_bd: Callable):
    """解退化 Hu–Zhang 混合系统，返回平衡应力 σ_h 与混合位移 u_h（DG）。

    输入:
      mesh   : TriangleMesh
      p      : Hu–Zhang 应力次数（位移用 p-1 DG）
      pde    : 提供 displacement(p)->(...,2), stress(p)->(...,3), source(p)->(...,2),
               damage(p)->(...,)（DegradedElasticMMS 实例）
      lam,mu : Lamé；k_res: 残余刚度 (>0)
      isD_bd : 边重心 bc:(NEb,2)->bool，标记 Dirichlet（位移）边
    输出 dict:
      sigmah : Hu–Zhang 应力 FE function（平衡，Voigt (xx,xy,yy)）
      uh     : 混合位移 FE function（DG）
      space_sigma, space_u
    """
    lambda0, lambda1 = compliance_lame_coeffs(lam, mu)
    isNedge = build_isNedge_from_isD(mesh, isD_bd)

    space0 = HuZhangFESpace2d(mesh, p=p, use_relaxation=True, bd_stress=isNedge)
    space_d = LagrangeFESpace(mesh, p=p - 1, ctype='D')
    space1 = TensorFunctionSpace(space_d, shape=(-1, 2))

    gdof0 = space0.number_of_global_dofs()
    uh = space1.function()

    # 退化柔度系数 coef = g(d)^{-1}，逐 qp 求值（barycentric）
    @barycentric
    def coef_ginv(bcs, index=None):
        # bcs:(NQ,3) -> 物理点 (NC,NQ,2)
        pts = mesh.bc_to_point(bcs)
        d_qp = pde.damage(pts)                      # (NC,NQ)
        g = (1.0 - d_qp) ** 2 + k_res
        return 1.0 / g                              # (NC,NQ)

    bform1 = BilinearForm(space0)
    bform1.add_integrator(HuZhangStressIntegrator(
        coef=coef_ginv, lambda0=lambda0, lambda1=lambda1))
    bform2 = BilinearForm((space1, space0))
    bform2.add_integrator(HuZhangMixIntegrator())

    M = bform1.assembly().to_scipy().tocsr()
    B = bform2.assembly().to_scipy().tocsr()
    TM = space0.TM.to_scipy().tocsr()

    M2 = TM.T @ M @ TM
    B2 = TM.T @ B
    A = bmat([[M2, B2], [B2.T, None]], format="csr")

    lform = LinearForm(space1)

    @cartesian
    def source(x, index=None):
        return pde.source(x)
    lform.add_integrator(VectorSourceIntegrator(source=source))
    b = lform.assembly()

    HBC = HuzhangBoundaryCondition(space=space0)
    a = HBC.displacement_boundary_condition(value=pde.displacement, threshold=isD_bd)

    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[:gdof0] = TM.T @ a
    F[gdof0:] = -b

    HSBC = HuzhangStressBoundaryCondition(space=space0)
    uh_stress, isbddof_stress = HSBC.set_essential_bc(
        pde.stress, threshold=isNedge, coord="auto")

    uh_global = bm.zeros(A.shape[0], dtype=A.dtype)
    uh_global[:gdof0] = uh_stress
    isbddof_global = bm.zeros(A.shape[0], dtype=bool)
    isbddof_global[:gdof0] = isbddof_stress

    F = F - A @ uh_global
    F[isbddof_global] = uh_global[isbddof_global]

    total = A.shape[0]
    bdIdx = bm.zeros(total, dtype=bm.int32)
    bdIdx[isbddof_global] = 1
    Tbd = spdiags(bm.to_numpy(bdIdx), 0, total, total)
    T = spdiags(1 - bm.to_numpy(bdIdx), 0, total, total)
    A = T @ A @ T + Tbd

    X = scipy_spsolve(A, bm.to_numpy(F))
    X = bm.array(X)

    sigmaval = TM @ X[:gdof0]
    sigmah = space0.function()
    sigmah[:] = sigmaval
    uh[:] = X[gdof0:]
    return {"sigmah": sigmah, "uh": uh,
            "space_sigma": space0, "space_u": space1}
