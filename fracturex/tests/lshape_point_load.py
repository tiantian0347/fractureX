"""L-shape 工程算例: 上短边固定 + 凹角内水平短边中点集中面荷载.

Setup:
    Ω = [-1, 1]^2 \\ [0, 1]^2, 凹角在 (0, 0).
    ΓD (u=0): 上短边 y=1, x ∈ [-1, 0]
    ΓN (σn = g): 其余外边界
      - 载荷段: y=0, x ∈ [0.5-ε, 0.5+ε], σn = (0, -F/(2ε))
      - 其余段: σn = 0 (traction-free)
    材料: E=210 GPa, ν=0.3

Since no analytic solution, we verify by:
  (a) no NaN / matrix non-singular;
  (b) u_y at (0.5, 0) < 0 (loaded direction);
  (c) 上短边 u ≡ 0；
  (d) σ_yy 在载荷段有明显负值（压缩）；
  (e) 凹角 (0,0) 附近应力集中显现；
  (f) 网格加密时位移场收敛（Richardson 型比较，取粗/细两组自解对照）。

用 base + skip_nn_corner_nodes=True (生产模式).
"""
from __future__ import annotations
import sys
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import VectorSourceIntegrator, BilinearForm, LinearForm
from fealpy.decorator import cartesian, barycentric
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat, spdiags

from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition, HuzhangStressBoundaryCondition, build_isNedge_from_isD,
)


# 材料（无量纲化：E=1, ν=0.3，工程结果由用户按 E_real 缩放）
E = 1.0
NU = 0.3
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
LAMBDA0 = 2 * MU
LAMBDA1 = MU * LAMBDA / (2 * LAMBDA + MU)

# 载荷（无量纲）
F_TOTAL = 1.0
LOAD_HALF_WIDTH = 0.25   # 载荷分布半宽 → 覆盖 x∈[0.25, 0.75]
LOAD_CENTER = 0.5
G_MAG = F_TOTAL / (2 * LOAD_HALF_WIDTH)


def make_lshape_mesh(N):
    """L = [-1, 1]^2 \\ [0, 1]^2, 凹角在 (0, 0). N 需能整除到让 x=0/y=0 和 x=0.5, 0.45, 0.55 落网格线上"""
    if N % 4 != 0:
        N += (4 - N % 4)
    mesh = TriangleMesh.from_box([-1., 1., -1., 1.], nx=N, ny=N)
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


def isD_top_short(bc):
    """ΓD = 上短边 y=1, x∈[-1, 0] + 左短边上部 x=-1, y∈[0, 1] (共同锚定去除 rigid body 模式)."""
    tol = 1e-9
    x = bc[:, 0]; y = bc[:, 1]
    on_top = (bm.abs(y - 1.0) < tol) & (x < 0 + tol)
    on_left_upper = (bm.abs(x - (-1.0)) < tol) & (y > 0 - tol)
    return on_top | on_left_upper


def sigma_gd(pts):
    """σ · n = g on ΓN.

    Load segment: y=0, x∈[0.5-ε, 0.5+ε], n=(0, -1) (外法向指向下)
      σ · n = (σ_xy · 0 + σ_xx · 0, σ_yy · (-1) ...) — 我们要提供直接 σ_gd tensor
      使 σ·(-1)*y_hat = (0, -F/(2ε))  →  σ_gd 的 (0, -1)·n = (0, -G_MAG)
      → σ_xy = 0, σ_yy · (-1) = -G_MAG → σ_yy = G_MAG (即向下压)
      令 σ_xx = 0, σ_xy = 0, σ_yy = G_MAG on load segment.
    其余 ΓN: σ = 0 (traction-free)
    """
    pts = np.asarray(pts)
    x = pts[..., 0]; y = pts[..., 1]
    on_load = (np.abs(y) < 1e-9) & (x > LOAD_CENTER - LOAD_HALF_WIDTH - 1e-9) \
              & (x < LOAD_CENTER + LOAD_HALF_WIDTH + 1e-9)
    sig = np.zeros(pts.shape[:-1] + (3,))
    sig[..., 2] = np.where(on_load, G_MAG, 0.0)
    return sig


def zero_u_D(pts):
    return bm.zeros(np.asarray(pts).shape, dtype=bm.float64)


def solve(N: int, p: int = 3):
    mesh = make_lshape_mesh(N)
    isN = build_isNedge_from_isD(mesh, isD_top_short)
    base = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isN)
    u_sc = LagrangeFESpace(mesh, p=p-1, ctype='D')
    space_u = TensorFunctionSpace(u_sc, shape=(-1, 2))

    @barycentric
    def coef(bcs, index=None):
        return bm.ones((1,) + bcs.shape[:-1], dtype=bm.float64)
    bform1 = BilinearForm(base)
    bform1.add_integrator(HuZhangStressIntegrator(coef=coef, lambda0=LAMBDA0, lambda1=LAMBDA1))
    bform2 = BilinearForm((space_u, base))
    bform2.add_integrator(HuZhangMixIntegrator())

    M = bform1.assembly().to_scipy().tocsr()
    B = bform2.assembly().to_scipy().tocsr()
    A = bmat([[M, B], [B.T, None]], format='csr')

    @cartesian
    def src(x, index=None):
        return bm.zeros(x.shape, dtype=x.dtype)
    L = LinearForm(space_u); L.add_integrator(VectorSourceIntegrator(source=src))
    b = L.assembly()

    HBC = HuzhangBoundaryCondition(space=base)
    a = HBC.displacement_boundary_condition(value=zero_u_D, threshold=isD_top_short)

    HSBC = HuzhangStressBoundaryCondition(space=base)
    def sig_bm(pts):
        return bm.tensor(sigma_gd(np.asarray(pts)), dtype=base.ftype)
    uh_sig, isbd_sig = HSBC.set_essential_bc_v2(
        sig_bm, threshold=isN, coord='auto',
        skip_nn_corner_nodes=True,
    )

    gdof0 = base.number_of_global_dofs()
    F = np.zeros(A.shape[0])
    F[:gdof0] = np.asarray(a); F[gdof0:] = -np.asarray(b)
    uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
    uh_g[:gdof0] = np.asarray(uh_sig); isbd_g[:gdof0] = np.asarray(isbd_sig)
    F = F - A @ uh_g; F[isbd_g] = uh_g[isbd_g]
    n_ = A.shape[0]; bdIdx = isbd_g.astype(int)
    A = spdiags(1 - bdIdx, 0, n_, n_) @ A @ spdiags(1 - bdIdx, 0, n_, n_) + spdiags(bdIdx, 0, n_, n_)

    X = spsolve(A, F)
    if np.any(np.isnan(X)):
        print(f'N={N}: NaN detected in solution')
        return None
    sigmah = base.function(); sigmah[:] = X[:gdof0]
    uh = space_u.function(); uh[:] = X[gdof0:]
    return mesh, sigmah, uh


def probe(mesh, sigmah, uh, points_desc):
    """Evaluate σ_h, u_h at listed points."""
    node = np.asarray(mesh.entity('node'))
    cell = np.asarray(mesh.entity('cell'))
    for name, target in points_desc:
        # 找最近节点及包含它的 cell
        d = np.linalg.norm(node - np.array(target), axis=1)
        nid = int(np.argmin(d))
        inc = np.where(np.any(cell == nid, axis=1))[0]
        if len(inc) == 0:
            print(f'  {name} @ {target}: no cell found near nid {nid}'); continue
        ic = int(inc[0])
        loc = int(np.where(cell[ic] == nid)[0][0])
        bcs = np.zeros((1, 3)); bcs[0, loc] = 1.0
        sig = np.asarray(sigmah(bm.tensor(bcs)))[ic, 0]
        u = np.asarray(uh(bm.tensor(bcs)))[ic, 0]
        print(f'  {name} @ {node[nid].tolist()} (nid={nid}, cell={ic}):')
        print(f'    u_h = {u.round(6).tolist()}')
        print(f'    σ_h (Voigt xx,xy,yy) = {sig.round(2).tolist()}')


def eval_at_points(mesh, sigmah, uh, points):
    """Evaluate σ_h, u_h at physical points (returns arrays)."""
    node = np.asarray(mesh.entity('node'))
    cell = np.asarray(mesh.entity('cell'))
    sig_vals = np.zeros((len(points), 3))
    u_vals = np.zeros((len(points), 2))
    for i, target in enumerate(points):
        d = np.linalg.norm(node - np.array(target), axis=1)
        nid = int(np.argmin(d))
        inc = np.where(np.any(cell == nid, axis=1))[0]
        if len(inc) == 0:
            continue
        ic = int(inc[0])
        loc = int(np.where(cell[ic] == nid)[0][0])
        bcs = np.zeros((1, 3)); bcs[0, loc] = 1.0
        sig_vals[i] = np.asarray(sigmah(bm.tensor(bcs)))[ic, 0]
        u_vals[i] = np.asarray(uh(bm.tensor(bcs)))[ic, 0]
    return sig_vals, u_vals


def cauchy_convergence(N_list, N_ref, p=3):
    """Compute pointwise Cauchy convergence: (σ_h(N), u_h(N)) vs reference on N_ref."""
    print(f"\n=== Cauchy 收敛阶（reference: N={N_ref}, p={p}）===")
    probe_pts = [
        (-0.5, 0.5),       # 上短边下方（固定区域内部）
        (0.5, -0.5),        # 载荷正下方
        (0.5, 0.0),         # 载荷正上（边界）
        (0.0, -0.5),        # 凹角下方
        (0.5, -1.0),        # 底边点
        (1.0, -0.5),        # 右边中点
    ]
    # reference
    print(f'... solving reference N={N_ref}')
    mesh_r, sig_r, uh_r = solve(N_ref, p=p)
    sig_ref, u_ref = eval_at_points(mesh_r, sig_r, uh_r, probe_pts)
    print(f'{"N":>5} {"h":>8} {"|Δσ|_l2":>12} {"σ rate":>8} {"|Δu|_l2":>12} {"u rate":>8}')
    prev_es, prev_eu = None, None
    for N in N_list:
        mesh, sigh, uh = solve(N, p=p)
        sig_h, u_h = eval_at_points(mesh, sigh, uh, probe_pts)
        e_sig = float(np.linalg.norm(sig_h - sig_ref))
        e_u = float(np.linalg.norm(u_h - u_ref))
        rs = '-' if prev_es is None else f'{np.log2(prev_es / e_sig):.2f}'
        ru = '-' if prev_eu is None else f'{np.log2(prev_eu / e_u):.2f}'
        h = 2.0 / N
        print(f'{N:>5} {h:>8.4f} {e_sig:>12.4e} {rs:>8} {e_u:>12.4e} {ru:>8}')
        prev_es, prev_eu = e_sig, e_u


def main():
    print(f"=== L-shape 短边固定 + 平行短边点力测试 ===")
    print(f"E = {E:.4f}, ν = {NU}, F_total = {F_TOTAL:.4f} (无量纲)")
    print(f"载荷段: y=0, x∈[{LOAD_CENTER-LOAD_HALF_WIDTH}, {LOAD_CENTER+LOAD_HALF_WIDTH}]")
    print(f"G_MAG (线荷载) = {G_MAG:.4f}")

    # 快速物理性检查：N=32 点值
    print(f'\n--- 物理性检查 N=32 ---')
    mesh, sigh, uh = solve(32, p=3)
    probe(mesh, sigh, uh, [
        ('load center', (0.5, 0.0)),
        ('re-entrant', (0.0, 0.0)),
        ('fixed top-left', (-1.0, 1.0)),
        ('right-mid', (1.0, -0.5)),
    ])

    # Cauchy 收敛阶
    cauchy_convergence(N_list=[8, 16, 32], N_ref=64, p=3)


if __name__ == '__main__':
    main()
