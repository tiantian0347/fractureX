# fracturex/cases/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Any, Protocol

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


# ---- type aliases ----
Threshold = Callable[[TensorLike], TensorLike]  # points -> bool mask (N,)
VectorValue = Callable[[TensorLike], TensorLike]  # points -> (N, GD)


class ModelProtocol(Protocol):
    """
    材料模型最小协议（driver/assembler 不关心你的实现细节，只要这些属性/方法存在即可）。
    """
    ft: float
    def moderation_parameter(self) -> float: ...
    # 可选：lam/mu 或 lambda0/lambda1 或 E/nu
    lam: float
    mu: float


@dataclass
class DirichletPiece:
    """
    一段 Dirichlet: 在 threshold 选中的边界上施加 value（向量），direction 可选。
    - value 建议返回 (N, GD) 的位移向量（即使只给 uy，也返回 [0, uy]）
    - direction 若使用你现有 HBC 实现，也可以保留（例如只固定 y 分量）
    """
    threshold: Threshold
    value: VectorValue
    direction: Optional[str] = None


class CaseBase:
    """
    断裂/损伤算例基类：只管“物理场景”，不管“怎么解”。
    """

    name: str = "case"

    # --- mesh / model ---
    def make_mesh(self, **kwargs):
        raise NotImplementedError

    def model(self) -> ModelProtocol:
        raise NotImplementedError

    # --- boundary classification: D/N split (用于构造 isNedge 与 corner relaxation) ---
    def isD_bd(self, points: TensorLike) -> TensorLike:
        """
        输入 points 通常是边界边重心 (NEb, GD)。
        返回 True 表示 Dirichlet 边界。
        """
        raise NotImplementedError

    # --- boundary values ---
    def dirichlet_pieces(self, load: float) -> List[DirichletPiece]:
        """
        返回多段 Dirichlet（用于 y=0 固定、y=1 加载等 piecewise 情况）
        """
        raise NotImplementedError

    def neumann_data(self, load: float):
        """
        可选：返回应力/牵引边界数据
        建议约定返回:
           (gd, threshold, coord)
        其中 coord in {"voigt","nt","auto"} 等
        """
        return None

    # --- body force (可选) ---
    def body_force(self, points: TensorLike) -> TensorLike:
        """
        默认无体力。需要的话覆盖它，返回 (N, GD)。
        """
        GD = points.shape[-1]
        return bm.zeros(points.shape[:-1] + (GD,), dtype=points.dtype)
    
    def traction(self, points, load: float):
        # 默认无 Neumann
        return None

    def isNedge(self, points):
        # 默认无（或全部边界 Neumann），看你项目约定
        return None
    
    def load_boundary_threshold(self) -> Callable[[TensorLike], TensorLike]:
        """
        返回一个 callable(points)->bool，用来选“加载位移所在边界”（边重心输入）
        默认：用 dirichlet_pieces 里最后一段当加载边界（你也可以改成更明确的规则）
        """
        def _thr(points: TensorLike):
            pieces = self.dirichlet_pieces(load=0.0)
            if len(pieces) == 0:
                raise RuntimeError("dirichlet_pieces is empty; cannot infer load boundary.")
            return pieces[-1].threshold(points)
        return _thr
    


def debug_isNedge_on_crack(mesh, isNedge, crack_edge_ids=None, crack_pred=None, tol=1e-9):
    """
    打印：
      - 全边界边数量 NEb
      - 边界边重心 bc 的 xmin/xmax, ymin/ymax
      - crack edges 数量 + crack bc min/max
      - crack edges 是否被 isNedge 选中
      - 每条 crack edge 的端点与法向（可选）
    crack_edge_ids: 直接给裂纹边的全局 eid 数组
    crack_pred:     函数 crack_pred(bc)->bool，基于边界边重心筛裂纹边
    """
    edge = mesh.entity('edge')
    node = mesh.entity('node')

    isBd = mesh.boundary_edge_flag()
    bdedge = bm.where(isBd)[0]
    NEb = int(bdedge.shape[0])

    bc = mesh.entity_barycenter('edge', index=bdedge)  # (NEb, 2)
    xmin, xmax = float(bm.min(bc[:, 0])), float(bm.max(bc[:, 0]))
    ymin, ymax = float(bm.min(bc[:, 1])), float(bm.max(bc[:, 1]))
    print(f"[isNedge debug] NEb={NEb}, bc xmin/xmax=({xmin:g},{xmax:g}), ymin/ymax=({ymin:g},{ymax:g})")

    # --- 找 crack edges ---
    if crack_edge_ids is None:
        if crack_pred is None:
            # 默认给一个“方形裂纹”的识别：y≈0.5 且 x<0.5（按你这个网格）
            crack_pred = lambda b: (bm.abs(b[:, 1] - 0.5) < tol) & (b[:, 0] < 0.5 + tol) & (b[:, 0] > 0.0 + tol)
        flag = bm.asarray(crack_pred(bc)).astype(bm.bool)
        crack_edge_ids = bdedge[flag]

    crack_edge_ids = bm.asarray(crack_edge_ids, dtype=bm.int32)
    ncr = int(crack_edge_ids.shape[0])
    print(f"[isNedge debug] crack edges = {ncr}")

    if ncr == 0:
        return

    # crack bc min/max
    cbc = mesh.entity_barycenter('edge', index=crack_edge_ids)
    cxmin, cxmax = float(bm.min(cbc[:, 0])), float(bm.max(cbc[:, 0]))
    cymin, cymax = float(bm.min(cbc[:, 1])), float(bm.max(cbc[:, 1]))
    print(f"[isNedge debug] crack bc xmin/xmax=({cxmin:g},{cxmax:g}), ymin/ymax=({cymin:g},{cymax:g})")

    # crack 是否落在 isNedge
    sel = bm.asarray(isNedge, dtype=bm.bool)[crack_edge_ids]
    print(f"[isNedge debug] isNedge on crack: {sel.tolist()}")

    # 可选：逐条打印端点与法向
    nvec = mesh.edge_unit_normal(index=crack_edge_ids)
    for i, eid in enumerate(crack_edge_ids):
        p0 = node[int(edge[int(eid), 0])]
        p1 = node[int(edge[int(eid), 1])]
        nv = nvec[i]
        print(f"  eid={int(eid)} isBd={bool(isBd[int(eid)])} isN={bool(isNedge[int(eid)])} "
              f"p0={[float(p0[0]), float(p0[1])]} p1={[float(p1[0]), float(p1[1])]} "
              f"normal={[float(nv[0]), float(nv[1])]}")

