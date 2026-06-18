# fracturex/adaptivity/primal_elastic_solve.py
"""标准连续 Lagrange 弹性求解器（超圆里的运动学容许位移 u_h）。

THEORY/DESIGN：u_h ∈ V_g（H¹-协调、连续），与 Hu–Zhang 平衡应力 σ_h 一起喂超圆
指示子 η_T。本模块只负责「给定退化系数 g(d) 与边界，解出连续位移 u_h」。

约定:
  - 平面应变，Lamé (λ,μ)；退化 g(d)=(1-d)²+k_res 通过系数函数注入刚度。
  - 计算走 backend_manager (bm)；线性求解器走 scipy（第三方边界，允许 numpy）。
  - 不修改 fealpy：用 fealpy 的 LinearElasticIntegrator/LagrangeFESpace 组合。

当前版本（M0）：常系数 g≡const（或 g=1）路径，MMS 验收敛阶；
变 g(d) 注入留待 T6（degraded effectivity）扩展。
"""
from __future__ import annotations

from typing import Callable, Optional

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem import (
    BilinearForm, LinearForm, LinearElasticIntegrator,
    VectorSourceIntegrator,
)
from fealpy.fem import DirichletBC
from fealpy.decorator import cartesian


def make_material(lam: float, mu: float, g_const: float = 1.0):
    """退化常系数平面应变材料 g·C。

    输入:
      lam, mu : Lamé 参数
      g_const : 退化常数 (g=(1-d)²+k_res 的常数情形；默认 1=无退化)
    输出:
      LinearElasticMaterial，弹性张量为 g·C（平面应变）
    """
    return LinearElasticMaterial(
        name="degraded",
        lame_lambda=g_const * lam,
        shear_modulus=g_const * mu,
        hypo="plane_strain",
    )


def solve_primal(mesh, p: int, lam: float, mu: float,
                 source: Callable, dirichlet: Callable,
                 g_const: float = 1.0, q: Optional[int] = None):
    """解标准连续 Lagrange 弹性位移 u_h（全 Dirichlet）。

    输入:
      mesh      : TriangleMesh
      p         : 位移多项式次数
      lam, mu   : Lamé
      source    : @cartesian f(pts)->(...,2) 体力（= -div(gC ε(u_exact))）
      dirichlet : @cartesian uD(pts)->(...,2) 边界位移
      g_const   : 退化常数
      q         : 求积阶（默认 p+3）
    输出 dict:
      uh    : 位移 FE function（TensorFunctionSpace）
      space : 向量位移空间
    """
    q = (p + 3) if q is None else q
    scalar_space = LagrangeFESpace(mesh, p=p)
    space = TensorFunctionSpace(scalar_space, shape=(-1, 2))  # 向量位移

    material = make_material(lam, mu, g_const)
    bform = BilinearForm(space)
    bform.add_integrator(LinearElasticIntegrator(material, q=q))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_integrator(VectorSourceIntegrator(source=source, q=q))
    b = lform.assembly()

    uh = space.function()
    bc = DirichletBC(space, gd=dirichlet)
    A, b = bc.apply(A, b)

    from fealpy.solver import spsolve
    uh[:] = spsolve(A, b, solver="scipy")
    return {"uh": uh, "space": space}


class DegradedElasticMaterial(LinearElasticMaterial):
    """空间变化退化材料：elastic_matrix 返回逐元 g(d_T)·C。

    d 在单元重心求值 ⇒ 逐元分片常数 g（primal 标准 FEM 足够；裂纹带光滑 ⇒ κ_T=O(1)）。
    """

    def __init__(self, lam, mu, g_cell):
        """
        输入:
          lam,mu : Lamé
          g_cell : (NC,) 逐元退化值 g(d_T)
        """
        super().__init__(name="degraded_var", lame_lambda=lam,
                         shear_modulus=mu, hypo="plane_strain")
        self._g_cell = g_cell

    def elastic_matrix(self, bcs=None):
        """返回 (NC,1,3,3) 逐元退化弹性矩阵 g(d_T)·D。"""
        D0 = self.D                                  # (3,3) 未退化
        g = self._g_cell                             # (NC,)
        return g[:, None, None, None] * D0[None, None, ...]


def solve_primal_degraded(mesh, p: int, pde, *, lam: float, mu: float, k_res: float,
                          dirichlet, q=None):
    """解空间变化退化标准 FEM 位移 u_h（全 Dirichlet）。

    输入:
      mesh      : TriangleMesh
      p         : 位移次数
      pde       : 提供 damage(p)->(...,)（DegradedElasticMMS）
      lam,mu    : Lamé；k_res: 残余刚度
      dirichlet : @cartesian uD(pts)->(...,2)
      q         : 求积阶（默认 p+3）
    输出 dict: uh, space
    数学: 刚度 = g(d_T)·C 逐元；g 在单元重心求值。
    """
    q = (p + 3) if q is None else q
    scalar_space = LagrangeFESpace(mesh, p=p)
    space = TensorFunctionSpace(scalar_space, shape=(-1, 2))

    # 单元重心处 d → g_cell
    bc_center = bm.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]], dtype=bm.float64)
    centroid = mesh.bc_to_point(bc_center)[:, 0, :]      # (NC,2)
    d_cell = pde.damage(centroid)                        # (NC,)
    g_cell = (1.0 - d_cell) ** 2 + k_res

    material = DegradedElasticMaterial(lam, mu, g_cell)
    bform = BilinearForm(space)
    bform.add_integrator(LinearElasticIntegrator(material, q=q))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_integrator(VectorSourceIntegrator(source=pde.source, q=q))
    b = lform.assembly()

    uh = space.function()
    bc = DirichletBC(space, gd=dirichlet)
    A, b = bc.apply(A, b)

    from fealpy.solver import spsolve
    uh[:] = spsolve(A, b, solver="scipy")
    return {"uh": uh, "space": space}
