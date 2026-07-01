"""验证 wrapper 的 TM 块与 [HM18] §4.2 (4.2) 显式公式给出的 basis 等价。

Setup:
    Square [0,1]^2, N=2 mesh, bottom = ΓD, other 3 sides = ΓN.
    NN corner at (1,1) with m=2 cells.

Test:
    1. 取 wrapper 在该角点的 unc-DOF 6 行 × rel-DOF 4 列 TM 块 T_w。
    2. 按 [HM18] (4.2) 手工构造 4 个 basis τ_1..τ_4 每 basis 在 (K+, K-) 上是 3+3=6
       个 cartesian 分量，得 4×6 矩阵 T_paper（行=basis，列=unc DOF）。
       T_paper.T 是 (6, 4) unc × rel。
    3. 检验：两者 column-span 相等（即互为可逆变换）。

若等价：证明 wrapper 数学正确等同 [HM18] §4.2；若不等价：暴露 bug。
"""
from __future__ import annotations
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD
from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax


def build_hm18_basis_local(t_e, n_e, K_plus_is_first: bool = True):
    """构造 [HM18] §3.1 (3.1) 的 4 个 basis (cartesian Voigt).

    §3.1 (3.1) 说明在内部边中点 x_c 处的 4 basis 是：
      共享部分（两侧连续，φ_{x_c} 是常规节点 Lagrange）：
        b1 = φ_{x_c} · n n^T           = φ · S_e         (Voigt: (nx², nx·ny, ny²))
        b2 = φ_{x_c} · (n t^T + t n^T) = φ · S_perp_1    (Voigt: (2nx·tx, nx·ty+ny·tx, 2ny·ty))
      独立部分（切向纯切，两侧独立）：
        b3 = φ_{x_c}^+ · t t^T         [K+ only]         (Voigt: (tx², tx·ty, ty²))
        b4 = φ_{x_c}^- · t t^T         [K- only]         (Voigt 同 b3 但作用在 K-)

    在 unc-DOF 坐标下（每个 basis 用 (K+σ_xx, K+σ_xy, K+σ_yy, K-σ_xx, K-σ_xy, K-σ_yy)）：
      b1 = (S_e, S_e)              两侧同一 S_e
      b2 = (S_perp_1, S_perp_1)    两侧同一 S_perp_1
      b3 = (S_perp_2, 0)           只在 K+
      b4 = (0, S_perp_2)           只在 K-
    """
    tx, ty = t_e
    nx, ny = n_e
    S_e = np.array([nx * nx, nx * ny, ny * ny])
    S_p1 = np.array([2 * nx * tx, nx * ty + ny * tx, 2 * ny * ty])
    S_p2 = np.array([tx * tx, tx * ty, ty * ty])

    zero = np.zeros(3)
    if K_plus_is_first:
        b1 = np.concatenate([S_e, S_e])
        b2 = np.concatenate([S_p1, S_p1])
        b3 = np.concatenate([S_p2, zero])
        b4 = np.concatenate([zero, S_p2])
    else:
        b1 = np.concatenate([S_e, S_e])
        b2 = np.concatenate([S_p1, S_p1])
        b3 = np.concatenate([zero, S_p2])
        b4 = np.concatenate([S_p2, zero])
    return np.stack([b1, b2, b3, b4], axis=0)   # (4, 6)


def main():
    # 方形网格
    mesh = TriangleMesh.from_box([0., 1., 0., 1.], nx=2, ny=2)
    def isD_bot(bc):
        return bm.abs(bc[:, 1] - 0.0) < 1e-12
    isN = build_isNedge_from_isD(mesh, isD_bot)
    base = HuZhangFESpace2d(mesh, p=3, use_relaxation=False, bd_stress=isN)
    relax = HuZhangCornerRelax(mesh, p=3, isNedge=isN, base_space=base, verbose=True)
    print()
    node = np.asarray(mesh.entity('node'))
    for c in relax.corners:
        print(f'--- corner nid={c.nid} coord={node[c.nid]} m={len(c.cells)} ---')
        assert len(c.cells) == 2, "本验证只对 m=2 角点"
        # 内部边法向：c.interior_edges[0][2]
        (kj, kjp1, n_int) = c.interior_edges[0]
        # 切向：与 n 正交
        n_int = np.asarray(n_int)
        t_int = np.array([-n_int[1], n_int[0]])  # rotated 90 CCW
        print(f'  interior edge: cells {c.cells[kj]} → {c.cells[kjp1]}, n = {n_int.round(3).tolist()}, t = {t_int.round(3).tolist()}')

        # 3. 从 wrapper 的 TM 中取该角点的 6×4 块
        TM = relax.TM
        unc_ids = c.unc_dofs.reshape(-1)                # (6,)
        rel_ids = c.rel_dofs                             # (4,)
        assert len(rel_ids) == 4, f'expected m+2=4 rel DOFs, got {len(rel_ids)}'
        TM_block = np.asarray(TM[unc_ids][:, rel_ids].todense())   # (6, 4)
        print(f'\n  wrapper T_w (unc_6 × rel_4):')
        print(TM_block.round(4))

        # 4. 手工构造 [HM18] (4.2) 的 4×6 basis
        T_paper = build_hm18_basis_local(t_int, n_int, K_plus_is_first=(kj == 0))
        # 转置为 (6, 4) 与 T_w 对比
        T_p = T_paper.T
        print(f'\n  paper T_p (unc_6 × rel_4):')
        print(T_p.round(4))

        # 5. 秩、span 比较
        rk_w = np.linalg.matrix_rank(TM_block, tol=1e-9)
        rk_p = np.linalg.matrix_rank(T_p, tol=1e-9)
        rk_stacked = np.linalg.matrix_rank(np.hstack([TM_block, T_p]), tol=1e-9)
        print(f'\n  rank(T_w) = {rk_w}, rank(T_p) = {rk_p}, rank([T_w | T_p]) = {rk_stacked}')
        print(f'  → equivalent span iff rk_w == rk_p == rk_stacked')
        if rk_w == rk_p == rk_stacked:
            # 具体求变换矩阵 T_w = T_p · A (4×4)
            A, resid, _, _ = np.linalg.lstsq(T_p, TM_block, rcond=None)
            recon_err = np.linalg.norm(T_p @ A - TM_block)
            print(f'  T_w ≈ T_p · A, |T_p A - T_w| = {recon_err:.3e}, det(A) = {np.linalg.det(A):.3e}')
        else:
            print(f'  两者 span 不同！wrapper 可能有 bug 或与文献约定不同。')


if __name__ == '__main__':
    main()
