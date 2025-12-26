# fracturex/discretization/huzhang_discretization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD

from fracturex.cases.base import CaseBase


@dataclass
class HuZhangState:
    """
    统一管理离散未知量/内部变量（都用 space.function() 创建）。
    - sigma: HuZhang 空间函数（Voigt）
    - u    : 位移空间函数 (TensorFunctionSpace)
    - d    : 节点损伤（P1 标量）
    - r_hist: 局部损伤历史量（P1 标量）
    - H    : 相场历史驱动量（P1 标量，暂时可不用）
    """
    sigma: Any
    u: Any
    d: Any
    r_hist: Any
    H: Optional[Any] = None

    def as_view(self) -> Dict[str, Any]:
        return dict(sigma=self.sigma, u=self.u, d=self.d, r_hist=self.r_hist, H=self.H)


class HuZhangDiscretization:
    """
    只管“网格 + 空间 + state”，不管装配/求解/迭代。

    设计目标：
    - 当前：staggered (sigma,u) -> update d
    - 未来：phase-field 要解 d 方程 / 全耦合牛顿 / 自适应重建
    """

    def __init__(
        self,
        case: CaseBase,
        *,
        p: int,
        use_relaxation: bool = True,
        damage_p: int = 1,                 # d/r_hist/H 的空间阶次（局部损伤一般 P1）
        u_space_order: Optional[int] = None,  # 位移空间阶次默认 p-1（与你之前一致）
    ):
        self.case = case
        self.p = int(p)
        self.use_relaxation = bool(use_relaxation)
        self.damage_p = int(damage_p)
        self.u_space_order = int(u_space_order) if u_space_order is not None else int(p - 1)

        # built objects
        self.mesh: Optional[TriangleMesh] = None
        self.space_sigma: Optional[HuZhangFESpace2d] = None
        self.space_u: Optional[TensorFunctionSpace] = None
        self.space_d: Optional[LagrangeFESpace] = None

        self.state: Optional[HuZhangState] = None

    # -----------------------------
    # build / rebuild
    # -----------------------------
    def build(self, *, nx: Optional[int] = None, ny: Optional[int] = None, mesh=None):
        """
        构建 mesh + spaces + state
        - mesh 可以外部传入（自适应/读入网格等）
        - 否则用 case.make_mesh(nx,ny)
        """
        if mesh is None:
            self.mesh = self.case.make_mesh(nx=nx, ny=ny)
        else:
            self.mesh = mesh

        GD = self.mesh.geo_dimension()
        if GD != 2:
            raise ValueError("HuZhangDiscretization currently expects 2D mesh (GD=2).")

        # 1) sigma space: HuZhang, 关键：传入 isD_bd 用于构造 isNedge + corner/TM

        isNedge = build_isNedge_from_isD(self.mesh, self.case.isD_bd)

        self.space_sigma = HuZhangFESpace2d(
            self.mesh,
            p=self.p,
            use_relaxation=self.use_relaxation,
            bd_stress=isNedge,
        )

        # 2) displacement space: (p-1) Lagrange -> TensorFunctionSpace shape=(-1,2)
        if self.u_space_order < 1:
            raise ValueError(f"u_space_order must be >= 1, got {self.u_space_order}")
        u_scalar = LagrangeFESpace(self.mesh, p=self.u_space_order, ctype="D")
        self.space_u = TensorFunctionSpace(u_scalar, shape=(-1, GD))

        # 3) damage space: nodal P1 (continuous) by default
        self.space_d = LagrangeFESpace(self.mesh, p=self.damage_p, ctype="C")

        # 4) functions from spaces
        sigma = self.space_sigma.function()          # sigmah
        u = self.space_u.function()                  # uh
        d = self.space_d.function()                  # damage
        r_hist = self.space_d.function()             # history for local damage
        H = self.space_d.function()                  # history for phase-field (reserved)

        # init values
        sigma[:] = 0.0
        u[:] = 0.0
        d[:] = 0.0
        r_hist[:] = 0.0
        H[:] = 0.0

        self.state = HuZhangState(sigma=sigma, u=u, d=d, r_hist=r_hist, H=H)
        return self

    def rebuild_on_new_mesh(self, new_mesh, *, transfer: Optional[callable] = None):
        """
        自适应/换网格后重建 spaces/state。
        transfer(old_discr, new_discr, old_state, new_state) 可选：
        - 用于把 d / r_hist / H 从旧网格转移到新网格（并做 max 保不可逆）
        """
        old_discr = self.snapshot()
        old_state = self.state

        self.build(mesh=new_mesh)

        if transfer is not None and old_state is not None:
            transfer(old_discr, self, old_state, self.state)

        return self

    # -----------------------------
    # helpers
    # -----------------------------
    def snapshot(self):
        """
        给 transfer / adaptivity 用的轻量快照（不复制大数组）。
        """
        return dict(
            mesh=self.mesh,
            p=self.p,
            use_relaxation=self.use_relaxation,
            u_space_order=self.u_space_order,
            damage_p=self.damage_p,
            space_sigma=self.space_sigma,
            space_u=self.space_u,
            space_d=self.space_d,
        )

    @property
    def gdof_sigma(self) -> int:
        return int(self.space_sigma.number_of_global_dofs())

    @property
    def gdof_u(self) -> int:
        return int(self.space_u.number_of_global_dofs())

    @property
    def TM(self):
        """
        给 assembler 用：corner relaxation 变换矩阵（可能是 fealpy 的稀疏类型）
        """
        return self.space_sigma.TM

    def check(self):
        """
        一些基本一致性检查（调试用）
        """
        assert self.mesh is not None
        assert self.space_sigma is not None
        assert self.space_u is not None
        assert self.space_d is not None
        assert self.state is not None
        # sigma/u/d 都应该可写
        _ = float(bm.sum(self.state.d[:]) * 0.0)
        return True
