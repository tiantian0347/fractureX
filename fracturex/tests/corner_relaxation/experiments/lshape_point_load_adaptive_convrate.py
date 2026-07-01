"""L 形点力算例：自适应加密的真 L² Cauchy 收敛阶（DOF vs 真误差）.

方法:
    1. 先算一个非常细的均匀 (reference) 网格解 σ_h_ref.
    2. 对自适应加密序列的每个 σ_h_iter，在**稠密固定 probe 网格**（Ω 内 200x200）
       上做 pointwise L² 差:
          err_iter = sqrt(∫_Ω |σ_h_iter - σ_h_ref|² dx) ≈
                      sqrt(mean_over_probe |σ_h_iter(x_probe) - σ_h_ref(x_probe)|²) · area(Ω)
    3. plot log(err) vs log(DOF), 线性最小二乘拟合斜率.

若 err ~ DOF^(-2), 则自适应达到 p=3 最优阶.
"""
from __future__ import annotations
import sys
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.decorator import barycentric

sys.path.insert(0, '/Users/tian00/repository/fractureX/fracturex/tests/corner_relaxation/experiments')
sys.path.insert(0, '/Users/tian00/repository/fractureX/fracturex/tests')
from lshape_point_load import (
    LAMBDA0, LAMBDA1, F_TOTAL, LOAD_HALF_WIDTH, LOAD_CENTER, G_MAG,
    make_lshape_mesh, isD_top_short, sigma_gd, zero_u_D,
)
from lshape_point_load_adaptive import solve_and_estimate, cell_fluctuation_indicator, constitutive_residual_indicator


# ================ 稠密 probe 网格（在 L 形 Ω 内均匀采样） ==================
def build_probe_points(nx=200, ny=200):
    """L 形 Ω = [-1,1]² \\ [0,1]²: 生成 Ω 内 probe 点."""
    xs = np.linspace(-1, 1, nx, endpoint=True)
    ys = np.linspace(-1, 1, ny, endpoint=True)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    # 剔除 [0,1]² 内部
    in_lshape = ~((pts[:, 0] > 0) & (pts[:, 1] > 0))
    return pts[in_lshape]


def eval_sig_on_probes(mesh, sigmah, probe_pts):
    """在 probe_pts 上评估 σ_h."""
    node = np.asarray(mesh.entity('node'))
    cell = np.asarray(mesh.entity('cell'))
    NC = cell.shape[0]
    v0 = node[cell[:, 0]]; v1 = node[cell[:, 1]]; v2 = node[cell[:, 2]]
    e01 = v1 - v0; e02 = v2 - v0
    det = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]
    # 对每个 probe 点，找包含它的 cell 并算重心坐标
    P = probe_pts
    N = len(P)
    sig_vals = np.zeros((N, 3))
    unfound = 0
    for i in range(N):
        p = P[i]
        rp = p - v0
        # 重心 lam1, lam2 for each cell
        lam1 = (rp[:, 0] * e02[:, 1] - rp[:, 1] * e02[:, 0]) / det
        lam2 = (e01[:, 0] * rp[:, 1] - e01[:, 1] * rp[:, 0]) / det
        lam0 = 1 - lam1 - lam2
        tol = 1e-9
        inside = (lam0 > -tol) & (lam1 > -tol) & (lam2 > -tol)
        idx = np.where(inside)[0]
        if len(idx) == 0:
            unfound += 1
            continue
        ic = int(idx[0])
        bcs = np.array([[lam0[ic], lam1[ic], lam2[ic]]])
        # clip to sum=1 & ≥ 0
        bcs = np.maximum(bcs, 0)
        bcs = bcs / bcs.sum(axis=1, keepdims=True)
        sig_vals[i] = np.asarray(sigmah(bm.tensor(bcs)))[ic, 0]
    return sig_vals, unfound


def solve_uniform(N: int, p: int = 3):
    """均匀网格解，用于 reference."""
    from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
    from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
    from fealpy.fem import VectorSourceIntegrator, BilinearForm, LinearForm
    from fealpy.decorator import cartesian, barycentric as bary
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import bmat, spdiags
    from fracturex.boundarycondition.huzhang_boundary_condition import (
        HuzhangBoundaryCondition, HuzhangStressBoundaryCondition, build_isNedge_from_isD,
    )
    mesh = make_lshape_mesh(N)
    isN = build_isNedge_from_isD(mesh, isD_top_short)
    base = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isN)
    u_sc = LagrangeFESpace(mesh, p=p-1, ctype='D')
    space_u = TensorFunctionSpace(u_sc, shape=(-1, 2))
    @bary
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
    uh_sig, isbd_sig = HSBC.set_essential_bc_v2(sig_bm, threshold=isN, coord='auto', skip_nn_corner_nodes=True)
    gdof0 = base.number_of_global_dofs()
    F = np.zeros(A.shape[0]); F[:gdof0] = np.asarray(a); F[gdof0:] = -np.asarray(b)
    uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
    uh_g[:gdof0] = np.asarray(uh_sig); isbd_g[:gdof0] = np.asarray(isbd_sig)
    F = F - A @ uh_g; F[isbd_g] = uh_g[isbd_g]
    n_ = A.shape[0]; bdIdx = isbd_g.astype(int)
    A = spdiags(1 - bdIdx, 0, n_, n_) @ A @ spdiags(1 - bdIdx, 0, n_, n_) + spdiags(bdIdx, 0, n_, n_)
    X = spsolve(A, F)
    sigmah = base.function(); sigmah[:] = X[:gdof0]
    return mesh, sigmah, gdof0


def adaptive_run_with_ref_error(n_iter, N0, theta, ref_N=96, p=3):
    print(f'\n=== Adaptive 真 L² Cauchy 误差 | N0={N0}, θ={theta}, ref_N={ref_N} ===')

    # 1) 计算 reference solution on very fine uniform mesh
    print(f'... solving reference N_uniform = {ref_N}')
    ref_mesh, ref_sig, ref_dof = solve_uniform(ref_N, p=p)
    print(f'    reference DOF = {ref_dof}')

    # 2) 生成 probe 点
    probes = build_probe_points(nx=150, ny=150)
    print(f'    probe points = {len(probes)}')

    # 3) reference σ 在 probes 上的值
    sig_ref, u_ref = eval_sig_on_probes(ref_mesh, ref_sig, probes)
    print(f'    (unfound probe pts on ref: {u_ref})')
    if u_ref > 0.01 * len(probes):
        print(f'    warning: 大量 probe 点未找到 cell')

    # 4) 跑自适应加密循环，每 iter 算 |σ_iter - σ_ref| L²
    mesh = make_lshape_mesh(N0)
    print(f'{"iter":>4} {"DOF":>6} {"NC":>5} {"|σ-σ_ref|":>12} {"per-step":>9} {"cum LSQ":>9}')
    err_list = []
    dof_list = []
    for it in range(n_iter):
        res = solve_and_estimate(mesh, p=p)
        if res is None:
            break
        mesh, sigmah, uh, eta, total_eta, gdof0, eta_fluc, eta_cst = res
        sig_it, u_it = eval_sig_on_probes(mesh, sigmah, probes)
        diff = sig_it - sig_ref
        # weighted Voigt L² (proper Frobenius)
        w = np.array([1.0, 2.0, 1.0])
        d2 = (diff * diff * w).sum(axis=-1)
        # 面积权：L 形面积 = 3, mean·area ≈ integral
        err = np.sqrt(np.mean(d2) * 3.0)
        err_list.append(err); dof_list.append(gdof0)
        # per-step rate: (err_i / err_{i-1}) vs (dof_i / dof_{i-1})
        if it >= 1:
            rate = -np.log(err_list[-1] / err_list[-2]) / np.log(dof_list[-1] / dof_list[-2] + 1e-30)
        else:
            rate = None
        # cumulative LSQ rate
        if it >= 2:
            p_fit = np.polyfit(np.log(dof_list), np.log(err_list), 1)
            cum_lsq = p_fit[0]
        else:
            cum_lsq = None
        r_str = f'{rate:.3f}' if rate is not None else '-'
        c_str = f'{cum_lsq:.3f}' if cum_lsq is not None else '-'
        print(f'{it:>4} {gdof0:>6} {mesh.number_of_cells():>5} {err:>12.4e} {r_str:>9} {c_str:>9}')
        if it == n_iter - 1:
            break
        isMarked = np.asarray(TriangleMesh.mark(bm.tensor(eta), theta, method='L2'))
        mesh.bisect(isMarked, options={'disp': False})

    # 最终 LSQ 拟合
    if len(dof_list) >= 3:
        p_fit, cov = np.polyfit(np.log(dof_list), np.log(err_list), 1, cov=True)
        print(f'\n最终 log-log 最小二乘拟合: err ~ DOF^({p_fit[0]:.3f} ± {np.sqrt(cov[0,0]):.3f})')


if __name__ == '__main__':
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    N0 = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    ref_N = int(sys.argv[4]) if len(sys.argv) > 4 else 96
    adaptive_run_with_ref_error(n_iter, N0, theta, ref_N=ref_N)
