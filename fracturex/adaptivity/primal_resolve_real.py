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


def apply_dirichlet_pieces_lifted(A, f, space, case, load):
    """对分量式 Dirichlet 片做**带非齐次提升**的对称消元，解 u（非增量）。

    背景（2026-06-21 bug 修复，见 RESULTS §Θ<1 根因诊断）：`VectorDirichletBC.apply`
    的契约是「只做齐次对称消元」——它**不**把 −A[:,bd]·g_D 加进内部 RHS。对非零载荷
    g_D≠0（如 y=1 上 u_y=load）直接 `apply` 会丢弃载荷耦合，使解把位移跳变全挤到载荷边
    一排单元 ⇒ ε 出 O(1/h) 边界层尖峰 ⇒ 能量假发散。正确做法照 `main_solve.solve_displacement`：
      ① 把所有 piece 的 g_D 写进 u_D（`apply_value`）；
      ② 提升 f ← f − A·u_D；
      ③ 各 piece 用 **g_D=0** 齐次消元（f 在边界 dof 置 0）；
      ④ 解增量 du，u = u_D + du（边界 dof du=0 ⇒ 保留 u_D 的 g_D）。

    输入:
      A     : 已装配刚度（SparseTensor；本函数内被对称消元修改并返回）。
      f     : 右端（体力项；断裂无源时全 0）。
      space : 向量位移空间（分量优先 (GD,-1)，与 VectorDirichletBC 一致）。
      case  : 提供 dirichlet_pieces(load) 的分量式位移边界。
      load  : 当前载荷（= y=1 边的 u_y）。
    输出:
      uh    : 解出的连续位移 FE function（含正确非齐次边界值）。
      A, f  : 消元后的系统（诊断/复用）。
    限制: piece value 为常数（在单点求值取分量得 g_D）；direction=None 仅支持各分量同值。
    """
    _axis = {"x": 0, "y": 1, "z": 2}
    probe = bm.array([[0.5, 0.5]], dtype=bm.float64)     # 常数场与位置无关
    pieces = list(case.dirichlet_pieces(load))
    specs = []                                            # [(gd, threshold, direction)]
    for piece in pieces:
        vec = bm.asarray(piece.value(probe)).reshape(-1)  # (GD,)
        comp = _axis[piece.direction] if piece.direction is not None else None
        gd = float(vec[comp]) if comp is not None else float(vec[0])
        if comp is None and not bool(bm.all(vec == vec[0])):
            raise NotImplementedError(
                "direction=None 仅支持各分量同值常数边界（本算例 fix=0 满足）。")
        specs.append((gd, piece.threshold, piece.direction))

    # ① 写非齐次边界值进 u_D
    uh = space.function()
    for gd, thr, direction in specs:
        bc = VectorDirichletBC(space, gd, thr, direction=direction)
        uh, _ = bc.apply_value(uh)
    # ② 提升：f ← f − A·u_D
    f = f - A @ uh[:]
    # ③ 各 piece 齐次消元（g_D=0）
    for _, thr, direction in specs:
        bc0 = VectorDirichletBC(space, 0, thr, direction=direction)
        A, f = bc0.apply(A, f)
    # ④ 解增量并叠加（边界 dof du=0，保留 u_D 的 g_D）
    from fealpy.solver import spsolve
    du = spsolve(A, f, solver="scipy")
    uh[:] = uh[:] + du
    return uh, A, f



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

    # 分量式 Dirichlet（真实算例边界）：y=0 固定 (x,y)，y=1 仅 u_y=load，侧边自由 Neumann。
    # **带非齐次提升**的对称消元（2026-06-21 修复：直接 apply 漏 −A·u_D 致载荷边能量假发散）。
    uh, A, f = apply_dirichlet_pieces_lifted(A, f, space, case, load)
    return {"uh": uh, "space": space, "g_cell": g_cell}
