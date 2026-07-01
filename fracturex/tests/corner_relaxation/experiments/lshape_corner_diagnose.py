"""诊断脚本：L 形凹角处 NN 角点为何被判 dangling。

只构网格 + 创建 HuZhangFESpace2d（不绕过过滤），把以下数据全部打印出来：

  1. mesh 基本量：node, edge, cell, boundary edges
  2. 凹角节点 (0,0) 的 incident boundary edges & 是否 ΓN
  3. corner_all：全部候选角点（含 type）
  4. _filter_active_corners_by_support 之前的 corner & corner2dof（含 newdof）
  5. 同一 newdof 是否出现在 cell_to_dof 中、为什么没出现
  6. edge_to_dof（重点：corner 关联的两条入边在端点位置上的 dof 是不是被 newdof 接管了）
"""
from __future__ import annotations
import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d, HuZhangFEDof2d
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


def make_lshape_mesh(N: int):
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


def isD_reentrant_NN(bc):
    tol = 1e-9
    x = bc[:, 0]; y = bc[:, 1]
    on_xeq0 = (bm.abs(x) < tol) & (y > -tol) & (y < 1 + tol)
    on_yeq0 = (bm.abs(y) < tol) & (x > -tol) & (x < 1 + tol)
    return ~(on_xeq0 | on_yeq0)


def main(N: int = 2, p: int = 3):
    mesh = make_lshape_mesh(N)
    node = np.asarray(mesh.entity('node'))
    edge = np.asarray(mesh.entity('edge'))
    cell = np.asarray(mesh.entity('cell'))
    NN, NE, NC = mesh.number_of_nodes(), mesh.number_of_edges(), mesh.number_of_cells()
    print(f"\n[mesh] N={N}, NN={NN}, NE={NE}, NC={NC}")

    # 找凹角节点 id
    corner_xy = np.array([0.0, 0.0])
    diffs = np.linalg.norm(node - corner_xy, axis=1)
    nid_corner = int(np.argmin(diffs))
    print(f"[corner] re-entrant node id = {nid_corner}, coord = {node[nid_corner]}")

    # boundary edges
    isBd = np.asarray(mesh.boundary_edge_flag())
    bdedge_ids = np.where(isBd)[0]
    inc_bd = bdedge_ids[np.any(edge[bdedge_ids] == nid_corner, axis=1)]
    print(f"[corner] incident boundary edges (eid): {inc_bd.tolist()}")
    for eid in inc_bd:
        a, b = edge[eid]
        print(f"   eid={eid}: node {a}{node[a]} - node {b}{node[b]}")

    # isNedge 判定
    isNedge = np.asarray(build_isNedge_from_isD(mesh, isD_reentrant_NN))
    print(f"[corner] inc edges  N={[bool(isNedge[e]) for e in inc_bd]}")

    # incident cells at corner
    inc_cells = np.where(np.any(cell == nid_corner, axis=1))[0]
    print(f"[corner] incident cells: {inc_cells.tolist()} (m = {len(inc_cells)})")

    # 创建 space（关掉 debug 过滤打印，自己来打印）
    space = HuZhangFESpace2d(mesh, p=p, use_relaxation=True, bd_stress=isNedge, debug=False)
    corner_all = space.corner_all
    print("\n[corner_all] (before filter)")
    print(f"  idx     = {np.asarray(corner_all['idx']).tolist()}")
    print(f"  type    = {np.asarray(corner_all['type']).tolist()}")
    print(f"  to_edge = {np.asarray(corner_all['to_edge']).tolist()}")

    # 重新做一次"未过滤"corner_to_dof 让我们看 newdof
    raw = {k: np.asarray(v) for k, v in corner_all.items()}
    ctype = raw['type'].astype(np.int32)
    toE = raw['to_edge']
    valid = (toE[:, 0] >= 0) & (toE[:, 2] >= 0) & (toE[:, 0] != toE[:, 2]) \
        & (toE[:, 1] >= 0) & (toE[:, 1] <= 1) & (toE[:, 3] >= 0) & (toE[:, 3] <= 1)
    active = (ctype == 2) & valid
    print(f"[corner_all] active mask = {active.tolist()}  (要 type==2 且 valid)")

    corner_unfiltered = {k: v[active] for k, v in raw.items()}
    NCP_unf = int(corner_unfiltered['idx'].shape[0])
    print(f"[corner_unfiltered] NCP = {NCP_unf}")

    tmp_dof = HuZhangFEDof2d(mesh, p, corner_unfiltered, use_relaxation=True)
    c2d_corner = np.asarray(tmp_dof.corner_to_dof())   # (NCP, 4)
    newdof = c2d_corner[:, -1]
    print(f"[corner_unfiltered] corner2dof =\n{c2d_corner}")
    print(f"[corner_unfiltered] newdof = {newdof.tolist()}")

    c2d = np.asarray(tmp_dof.cell_to_dof())
    print(f"[cell_to_dof] shape = {c2d.shape}, max = {c2d.max()}, unique count = {len(np.unique(c2d))}")
    in_c2d = [bool(d in c2d) for d in newdof]
    print(f"[cell_to_dof] newdof∈cell_to_dof? {in_c2d}")

    # 看看 edge_to_dof（这是 newdof 进入 cell_to_dof 的唯一路径）
    e2d = np.asarray(tmp_dof.edge_to_dof())
    print(f"[edge_to_dof] shape = {e2d.shape}")
    # 找 corner 两条入边
    eid0 = int(corner_unfiltered['to_edge'][0, 0])
    loc0 = int(corner_unfiltered['to_edge'][0, 1])
    eid1 = int(corner_unfiltered['to_edge'][0, 2])
    loc1 = int(corner_unfiltered['to_edge'][0, 3])
    print(f"[corner] to_edge: eid0={eid0} loc0={loc0}, eid1={eid1} loc1={loc1}")
    print(f"   edge_to_dof[eid0={eid0}] = {e2d[eid0].tolist()}")
    print(f"   edge_to_dof[eid1={eid1}] = {e2d[eid1].tolist()}")
    print(f"   newdof = {newdof.tolist()}")
    # 这里如果 e2d[eidk] 包含 newdof[0]，那就说明被接管了；否则就是 bug 所在

    # 把每条入边对应的两个相邻单元也找出来
    cell_of_edge = []
    edge2cell = np.asarray(mesh.edge_to_cell()) if hasattr(mesh, 'edge_to_cell') else None
    if edge2cell is not None:
        print(f"[edge_to_cell] {edge2cell[[eid0, eid1]].tolist()}")

    # 最关键：把过滤后 NCP 打出来
    print(f"\n[filter result] post-filter NCP = {int(space.NCP)}")


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(N, p)
