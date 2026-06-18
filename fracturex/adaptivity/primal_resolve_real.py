"""非-MMS 连续标准 FEM primal 重解（真实算例）—— 严格 Θ 认证的位移侧。

THEORY/DECISION：η_τ 是 Prager–Synge **双边**估计子（[[THEORY_equilibrated]] (7)），其
位移侧残差 r=ℂ_d ε(u_h^cont)−σ_h 须用**连续 H¹-协调** u_h^cont。M3 PC v2 用 driver 的
DG 混合位移 u（`eta_from_state` 的 grad_uh 通道），= RESULTS v2 诚实标注 #1 的根：那不是严格
可靠性界。本模块在**接受态网格**上，用**离散 d 场** g(d_T) + **真实算例 Dirichlet 载荷**
（非 MMS 解析 uD）解标准位移 FEM，产出 u_h^cont 喂回 `eta_from_state` ⇒ 严格 Θ。

与 `primal_elastic_solve.solve_primal_degraded`（MMS-only：吃 pde.damage/pde.source、全 Dirichlet）
的区别：
  - 退化系数从 **discr.state.d**（生产相场，逐元重心取 g(d_T)），非 MMS 解析 damage()。
  - 边界用 **case.dirichlet_pieces(load)**（分量式：y=0 固定 x,y；y=1 仅 u_y=load；侧边自由），
    经 VectorDirichletBC 施加，非 MMS 全 Dirichlet uD。
  - 体力 f=0（断裂无源），非 MMS 的 source=-div(gCε(u))。

约定：计算走 bm；线性解走 scipy（第三方边界允许 numpy）。不改 fealpy。
"""
from __future__ import annotations

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearElasticIntegrator

from fracturex.adaptivity.primal_elastic_solve import DegradedElasticMaterial
from fracturex.phasefield.vector_Dirichlet_bc import VectorDirichletBC


def _g_cell_from_state(discr, *, k_res: float):
    """从离散相场 state.d 取逐元退化 g(d_T)=(1-d_T)²+k_res（单元重心求值）。

    输入: discr 已解的 HuZhangDiscretization；k_res 残余刚度。
    输出: (NC,) g_cell。damage_p≥1 时 d 是节点场，bc_to_point 在重心插值。
    """
    mesh = discr.mesh
    bc_center = bm.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]], dtype=bm.float64)
    d_func = discr.state.d
    d_cell = d_func(bc_center)                       # (NC,1[,1]) 重心相场
    d_cell = bm.asarray(d_cell).reshape(-1)
    g_cell = (1.0 - d_cell) ** 2 + k_res
    return g_cell


def solve_primal_real(discr, case, *, lam: float, mu: float, load: float,
                      k_res: float = 1e-6, p: Optional[int] = None,
                      q: Optional[int] = None):
    """在接受态网格上解真实算例的连续标准 FEM 退化位移 u_h^cont。

    输入:
      discr : 已解的 HuZhangDiscretization（取网格 + 离散 d 场算 g_cell）
      case  : 算例（提供 dirichlet_pieces(load) 的分量式位移边界）
      lam,mu: Lamé（平面应变）
      load  : 当前载荷（= y=1 边的 u_y）
      k_res : 残余刚度 g=(1-d)²+k_res
      p     : 位移次数（默认取 discr 的位移次数；Hu–Zhang p=3 配 u 为 P2 ⇒ 默认 2）
      q     : 求积阶（默认 p+3）
    输出 dict:
      uh    : 连续位移 FE function（TensorFunctionSpace, H¹-协调）
      space : 向量位移空间
      g_cell: (NC,) 逐元退化（诊断/复用）
    数学: 刚度 A=∫ g(d_T) C ε:ε，体力 f=0；分量式 Dirichlet 经 VectorDirichletBC 对称消元。
    """
    mesh = discr.mesh
    # 位移次数：Hu–Zhang p=3 的生产配置位移为 P2；显式传 p 可覆盖。
    if p is None:
        p = int(getattr(discr, "p_u", getattr(discr, "u_degree", 2)))
    q = (p + 3) if q is None else q

    scalar_space = LagrangeFESpace(mesh, p=p)
    # 分量优先布局 (GD,-1)：与生产位移空间（huzhang_discretization 用 shape=(GD,-1)）及
    # VectorDirichletBC 的 (GD,npoints).ravel() 一致；用 (-1,2) 会让分量式 BC 落到错 dof。
    space = TensorFunctionSpace(scalar_space, shape=(2, -1))

    g_cell = _g_cell_from_state(discr, k_res=k_res)
    material = DegradedElasticMaterial(lam, mu, g_cell)

    bform = BilinearForm(space)
    bform.add_integrator(LinearElasticIntegrator(material, q=q))
    A = bform.assembly()

    # 体力 f=0（断裂无源）：RHS 初始全零，仅靠 Dirichlet 驱动。
    ndof = space.number_of_global_dofs()
    f = bm.zeros(ndof, dtype=bm.float64)

    uh = space.function()
    # 分量式 Dirichlet（真实算例边界）：y=0 固定 (x,y)，y=1 仅 u_y=load，侧边自由 Neumann。
    # VectorDirichletBC 用标量 gd 填该方向边界 dof；本算例 piece value 为常数（fix=0/load=const），
    # 在单点求值取对应分量即得 gd（非常数边界须扩展为按插值点求值，本算例不需要）。
    _axis = {"x": 0, "y": 1, "z": 2}
    probe = bm.array([[0.5, 0.5]], dtype=bm.float64)     # 任意点（常数场与位置无关）
    for piece in case.dirichlet_pieces(load):
        vec = bm.asarray(piece.value(probe)).reshape(-1)  # (GD,)
        comp = _axis[piece.direction] if piece.direction is not None else None
        gd = float(vec[comp]) if comp is not None else float(vec[0])
        if comp is None and not bool(bm.all(vec == vec[0])):
            raise NotImplementedError(
                "direction=None 仅支持各分量同值常数边界（本算例 fix=0 满足）。")
        bc = VectorDirichletBC(space, gd, piece.threshold, direction=piece.direction)
        A, f = bc.apply(A, f)

    from fealpy.solver import spsolve
    uh[:] = spsolve(A, f, solver="scipy")
    return {"uh": uh, "space": space, "g_cell": g_cell}
