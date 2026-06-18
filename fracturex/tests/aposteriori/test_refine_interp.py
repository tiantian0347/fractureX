"""T7-interp: 自适应加密后四种场的数据转移/插值正确性（H, d, u, σ）。

加密（bisect）后，子网格上的场必须由父网格的场转移得到。四种场布局不同，
fealpy 内置 mesh.bisect(options['data']) 的转移分支各异，本测试逐一验证：

  H (历史场)  : 逐元分片常数 (NC,)        → 子单元继承父值（DG0）
  d (相场)    : 连续 Lagrange p1 (NN,)    → 新节点边中点平均
  u (位移)    : 向量 Lagrange (2*NN,)     → ⚠ 一维布局，fealpy 节点分支不匹配
  σ (Hu–Zhang): H(div) 协调 (gdof 特殊)   → ⚠ 非节点非标准 ldof，内置分支必坏

判据：转移后场在子网格 qp 上的值，与解析场（线性场应精确、光滑场应小误差）比对。
线性场转移误差应 ~机器精度（bisect 的插值对 P1 线性场精确）。

目的：暴露 u/σ 需要专门转移（不能直接喂 fealpy bisect），为 M2 自适应循环定接口。
运行 p=3（Hu–Zhang 要求）。
"""
from __future__ import annotations

import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace


def _linear_scalar(p):
    """线性标量场 f=1+2x+3y（bisect 对线性场应精确转移）。"""
    return 1.0 + 2.0 * p[..., 0] + 3.0 * p[..., 1]


def test_H_piecewise_constant():
    """H: 分片常数 (NC,) → 子继承父。线性场用单元重心值，加密后子单元重心≠父，
    故分片常数本质不精确；这里验「子单元确实继承了父单元值」(转移机制对)。"""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    NC0 = mesh.number_of_cells()
    bc_c = bm.array([[1/3, 1/3, 1/3]])
    centroid = mesh.bc_to_point(bc_c)[:, 0, :]
    H = _linear_scalar(centroid)                       # (NC,) 父单元重心值

    isMarked = bm.zeros(NC0, dtype=bm.bool); isMarked = bm.set_at(isMarked, slice(0, NC0, 2), True)
    data = {'H': H.copy()}
    opt = mesh.bisect_options(data=data, disp=False)
    mesh.bisect(isMarked, options=opt)
    Hn = opt['data']['H']
    NC1 = mesh.number_of_cells()
    assert Hn.shape == (NC1,), f"H shape {Hn.shape} != ({NC1},)"
    assert NC1 > NC0, "no refinement happened"
    # 继承性：所有子单元值都来自父集合（分片常数不产生新值）
    parent_vals = set(np.round(np.asarray(H), 10).tolist())
    child_vals = set(np.round(np.asarray(Hn), 10).tolist())
    assert child_vals <= parent_vals, "H 产生了非继承的新值（转移机制错）"
    print(f"[interp.H] 分片常数继承 OK  NC {NC0}->{NC1}, 值集 {len(child_vals)}<={len(parent_vals)}")


def test_d_nodal_linear_exact():
    """d: 连续 Lagrange p1 (NN,) → 线性场转移应机器精度精确。"""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    NN0 = mesh.number_of_nodes(); NC0 = mesh.number_of_cells()
    node = mesh.entity('node')
    d = _linear_scalar(node)                           # (NN,) 节点值

    isMarked = bm.zeros(NC0, dtype=bm.bool); isMarked = bm.set_at(isMarked, slice(0, NC0, 2), True)
    data = {'d': d.copy()}
    opt = mesh.bisect_options(data=data, disp=False)
    mesh.bisect(isMarked, options=opt)
    dn = opt['data']['d'].reshape(-1)
    node_new = mesh.entity('node')
    assert dn.shape[0] == node_new.shape[0], f"d gdof {dn.shape} != NN {node_new.shape[0]}"
    exact = _linear_scalar(node_new)
    err = float(bm.max(bm.abs(dn - exact)))
    assert err < 1e-12, f"d 线性场转移误差 {err:.2e} (应机器精度)"
    print(f"[interp.d] 连续P1线性场转移精确  max err={err:.2e}  OK")


def _run_all():
    test_H_piecewise_constant()
    test_d_nodal_linear_exact()
    test_u_vector_via_IM()
    test_sigma_huzhang_no_nodal_transfer()
    print("\n[refine interp H,d,u,σ] ALL DONE")


def test_u_vector_via_IM():
    """u: 向量 Lagrange (2*NN,) → 直接喂 fealpy data dict 会崩溃；
    正确做法用 bisect 的 IM 节点插值矩阵按 (NN,2) 转移。线性场应精确。"""
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    NC0 = mesh.number_of_cells(); NN0 = mesh.number_of_nodes()

    def uvec(p):
        return bm.stack([1 + 2 * p[..., 0] + 3 * p[..., 1],
                         4 - p[..., 0] + 5 * p[..., 1]], axis=-1)
    u_nodal = uvec(mesh.entity('node'))                # (NN0,2)

    isMarked = bm.zeros(NC0, dtype=bm.bool); isMarked = bm.set_at(isMarked, slice(0, NC0, 2), True)
    opt = mesh.bisect_options(disp=False); opt['IM'] = True
    mesh.bisect(isMarked, options=opt)
    IM = opt['IM']                                     # (NNn, NN0)
    u_new = bm.array(IM @ u_nodal)                     # (NNn,2)
    exact = uvec(mesh.entity('node'))
    err = float(bm.max(bm.abs(u_new - exact)))
    assert err < 1e-12, f"u via IM 线性转移误差 {err:.2e}"
    print(f"[interp.u] 向量场经 IM 节点矩阵转移精确  max err={err:.2e}  OK")


def test_sigma_huzhang_no_nodal_transfer():
    """σ: Hu–Zhang H(div) → 加密后自由度数变、无节点对应，不能用 IM/节点平均。
    结论：σ 不转移，自适应每步【重解】混合系统得到。本测试坐实「gdof 变化、
    无直接转移」这一事实，锁定 M2 架构：演化态只转移 H/d，u/σ 每步现解。"""
    bm.set_backend("numpy")
    from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    NC0 = mesh.number_of_cells()
    gdof0 = HuZhangFESpace2d(mesh, p=3).number_of_global_dofs()
    isMarked = bm.zeros(NC0, dtype=bm.bool); isMarked = bm.set_at(isMarked, slice(0, NC0, 2), True)
    opt = mesh.bisect_options(disp=False)
    mesh.bisect(isMarked, options=opt)
    gdof1 = HuZhangFESpace2d(mesh, p=3).number_of_global_dofs()
    assert gdof1 != gdof0, "σ gdof 未变？"
    assert gdof1 > gdof0
    print(f"[interp.σ] Hu–Zhang gdof {gdof0}->{gdof1}，无节点转移 ⇒ M2 每步重解 σ  OK")


# 扩展 __main__
if __name__ == "__main__":
    _run_all()
