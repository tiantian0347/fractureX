"""L 形点力算例 + 基于 div σ_h + f 残差的自适应加密循环.

Estimator (simplest residual-type, valid for mixed FEM):
    η_T² = h_T² · ‖div σ_h - (-f)‖_{L²(T)}²  = h_T² · ‖div σ_h + f‖_{L²(T)}²
    (physically: momentum balance residual per cell)

对我们这个算例 f ≡ 0（无 body force）→ η_T = h_T · ‖div σ_h‖_{L²(T)}.
mixed FEM 强制 div σ_h + f = 0 在 P_{p-1} 上，但 f=0 时非零表示离散近似误差（点力
段的 ΓN essential BC 导致 σ_h 内部 div 有量级）。

Adaptive loop:
    for iter in range(n_iter):
        solve → η_T → Dörfler mark → bisect
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
    E, NU, LAMBDA, MU, LAMBDA0, LAMBDA1,
    F_TOTAL, LOAD_HALF_WIDTH, LOAD_CENTER, G_MAG,
    make_lshape_mesh, isD_top_short, sigma_gd, zero_u_D, solve, eval_at_points,
)


def constitutive_residual_indicator(mesh, sigmah, uh, p, *, lambda0, lambda1, q=None):
    """η_T^{cst} = ‖A σ_h − ε(u_h)‖_{L²(T)}.

    Mixed FEM enforces (A σ_h, τ) + (div τ, u_h) = ... weakly. The constitutive
    residual A σ_h − ε(u_h) equals zero exactly only if u_h has enough smoothness
    (needs C¹). Since u_h ∈ P_{p-1} DG, ε(u_h) is P_{p-2} per cell (jumping),
    while A σ_h is smoother — their difference measures constitutive error and is
    a natural mixed-FEM a posteriori estimator (as in [HM18] (2.1) sans the
    edge-jump term).

    A σ = c0 (σ − c1·trσ·I) with c0=1/λ0, c1=λ1/(λ0−2λ1).
    ε(u) Voigt: (∂u_x/∂x, (∂u_x/∂y+∂u_y/∂x)/2, ∂u_y/∂y).
    """
    from fealpy.backend import backend_manager as bm
    space_sig = sigmah.space
    space_u = uh.space
    if q is None:
        q = 2 * p + 2
    qf = mesh.quadrature_formula(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    # σ_h at qp
    sig = np.asarray(sigmah(bcs))       # (NC, NQ, 3) Voigt
    tr = sig[..., 0] + sig[..., 2]
    c0 = 1.0 / lambda0
    c1 = lambda1 / (lambda0 - 2 * lambda1)
    Asig = np.empty_like(sig)
    Asig[..., 0] = c0 * (sig[..., 0] - c1 * tr)
    Asig[..., 1] = c0 * sig[..., 1]
    Asig[..., 2] = c0 * (sig[..., 2] - c1 * tr)

    # ε(u_h) via grad_basis of scalar Lagrange space
    # uh is TensorFunctionSpace shape (-1, 2) over scalar LagrangeFESpace P_{p-1} DG
    # grad_basis returns (NC, NQ, ldof, 2) for scalar space
    sc_space = space_u.scalar_space if hasattr(space_u, 'scalar_space') else None
    if sc_space is None:
        # 尝试从 TensorFunctionSpace 拿
        sc_space = getattr(space_u, 'space', None)
    if sc_space is None:
        raise RuntimeError("cannot access scalar base for uh's TensorFunctionSpace")
    grad_phi = np.asarray(sc_space.grad_basis(bcs))   # (NC, NQ, ldof_sc, 2)
    c2d_sc = np.asarray(sc_space.cell_to_dof())         # (NC, ldof_sc)
    gdof_sc = int(sc_space.number_of_global_dofs())
    uh_np = np.asarray(uh)                              # (2*gdof_sc,) or similar
    # TensorFunctionSpace shape=(-1, 2) 存储：可能是 (gdof_sc, 2) 或 flat (2*gdof_sc)
    if uh_np.ndim == 1:
        # 惯例 fealpy: [d0_dofs, d1_dofs]? 还是交错？
        # 检查 uh.shape
        uh_arr = uh_np.reshape(-1, 2) if uh_np.size == 2 * gdof_sc else uh_np
        u0_coeff = uh_arr[:, 0] if uh_arr.ndim == 2 else uh_np[:gdof_sc]
        u1_coeff = uh_arr[:, 1] if uh_arr.ndim == 2 else uh_np[gdof_sc:]
    else:
        u0_coeff = uh_np[:, 0]; u1_coeff = uh_np[:, 1]
    # (NC, NQ, 2) — ∇u0 at qp
    grad_u0 = np.einsum('cl, cqld -> cqd', u0_coeff[c2d_sc], grad_phi)
    grad_u1 = np.einsum('cl, cqld -> cqd', u1_coeff[c2d_sc], grad_phi)
    eps_xx = grad_u0[..., 0]
    eps_yy = grad_u1[..., 1]
    eps_xy = 0.5 * (grad_u0[..., 1] + grad_u1[..., 0])
    eps = np.stack([eps_xx, eps_xy, eps_yy], axis=-1)

    diff = Asig - eps                                    # (NC, NQ, 3) Voigt
    w = np.array([1.0, 2.0, 1.0])
    d2 = (diff * diff * w).sum(axis=-1)
    cm = np.asarray(mesh.entity_measure('cell'))
    val = np.einsum('q, cq -> c', np.asarray(ws), d2) * cm
    return np.sqrt(val)


def cell_fluctuation_indicator(mesh, sigmah, p, q=None):
    """η_T = ‖σ_h(x) - σ_h(centroid(T))‖_{L²(T)} — cell-wise L² fluctuation.

    Rationale: σ_h ∈ HuZhang(p) restricted to T is a polynomial. The deviation
    from its centroid value measures how much σ_h varies within T. In smooth
    regions this scales like h^p; near singularities/stress concentrations it
    stays O(1) → naturally marks regions to refine.

    This is a cheap, reliable ZZ-type indicator that works even without an
    analytic reference solution.
    """
    from fealpy.backend import backend_manager as bm
    space = sigmah.space
    if q is None:
        q = 2 * p + 2
    qf = mesh.quadrature_formula(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    # σ_h at all quadrature points, shape (NC, NQ, 3)
    sig_qp = np.asarray(sigmah(bcs))
    # σ_h at cell centroid (bcs=(1/3,1/3,1/3))
    bc_center = bm.tensor([[1./3, 1./3, 1./3]])
    sig_ct = np.asarray(sigmah(bc_center))    # (NC, 1, 3)
    diff = sig_qp - sig_ct                     # (NC, NQ, 3)
    # Voigt Frobenius weight
    w = np.array([1.0, 2.0, 1.0])
    d2 = (diff * diff * w).sum(axis=-1)        # (NC, NQ)
    cm = np.asarray(mesh.entity_measure('cell'))
    val = np.einsum('q, cq -> c', np.asarray(ws), d2) * cm
    return np.sqrt(val)


def solve_and_estimate(N_or_mesh, p=3):
    """Solve on given mesh (or N -> build mesh), return (mesh, sigmah, uh, eta_per_cell, total_eta)."""
    if isinstance(N_or_mesh, int):
        mesh = make_lshape_mesh(N_or_mesh)
    else:
        mesh = N_or_mesh
    # 直接调用 solve 里的一段（solve 内嵌了 make_lshape_mesh(N)，需绕过）
    # 用局部 solver：复制 solve 逻辑但接受 mesh
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
    uh_sig, isbd_sig = HSBC.set_essential_bc_v2(
        sig_bm, threshold=isN, coord='auto', skip_nn_corner_nodes=True)

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
        return None
    sigmah = base.function(); sigmah[:] = X[:gdof0]
    uh = space_u.function(); uh[:] = X[gdof0:]

    eta_fluc = cell_fluctuation_indicator(mesh, sigmah, p=p)
    eta_cst = constitutive_residual_indicator(mesh, sigmah, uh, p=p,
                                              lambda0=LAMBDA0, lambda1=LAMBDA1)
    # 归一化各自最大值后相加，让两个 estimator 平等贡献
    eta_fluc_norm = eta_fluc / (float(np.max(eta_fluc)) + 1e-30)
    eta_cst_norm = eta_cst / (float(np.max(eta_cst)) + 1e-30)
    eta = eta_fluc_norm + eta_cst_norm
    total_eta = float(np.sqrt(np.sum(eta_fluc**2 + eta_cst**2)))
    return mesh, sigmah, uh, eta, total_eta, gdof0, eta_fluc, eta_cst


def adaptive_loop(n_iter, N0=8, p=3, theta=0.5):
    print(f'\n=== Adaptive loop | N0={N0}, p={p}, θ={theta}, n_iter={n_iter} ===')
    print(f'{"iter":>4} {"DOFσ":>6} {"NC":>6} {"η":>12} {"σ diff":>12} {"u diff":>12}')
    mesh = make_lshape_mesh(N0)
    # 保存 σ_h, u_h 序列，做 Cauchy 收敛（每两轮）
    probe_pts = [
        (-0.5, 0.5),
        (0.5, -0.5),
        (0.5, 0.0),
        (0.0, -0.5),
        (0.5, -1.0),
        (1.0, -0.5),
    ]
    sig_prev = None; u_prev = None
    for it in range(n_iter):
        res = solve_and_estimate(mesh, p=p)
        if res is None:
            print(f'  iter {it}: NaN'); break
        mesh, sigmah, uh, eta, total_eta, gdof0, eta_fluc, eta_cst = res
        NC = mesh.number_of_cells()
        sig_now, u_now = eval_at_points(mesh, sigmah, uh, probe_pts)
        if sig_prev is None:
            ds = du = '-'
        else:
            ds = f'{float(np.linalg.norm(sig_now - sig_prev)):.4e}'
            du = f'{float(np.linalg.norm(u_now - u_prev)):.4e}'
        print(f'{it:>4} {gdof0:>6} {NC:>6} {total_eta:>12.4e} {ds:>12} {du:>12}')
        # 诊断: fluctuation vs constitutive residual 谁主导
        cell_ = np.asarray(mesh.entity('cell'))
        node_ = np.asarray(mesh.entity('node'))
        arg_fluc = int(np.argmax(eta_fluc)); arg_cst = int(np.argmax(eta_cst))
        b_fluc = node_[cell_[arg_fluc]].mean(axis=0)
        b_cst = node_[cell_[arg_cst]].mean(axis=0)
        print(f'      max eta_fluc={eta_fluc[arg_fluc]:.3e} @ {b_fluc.round(3).tolist()}   '
              f'max eta_cst={eta_cst[arg_cst]:.3e} @ {b_cst.round(3).tolist()}')
        sig_prev, u_prev = sig_now, u_now
        if it == n_iter - 1:
            break
        isMarked = np.asarray(TriangleMesh.mark(bm.tensor(eta), theta, method='L2'))
        # 打印被 mark cells 的重心（诊断）
        cell = np.asarray(mesh.entity('cell'))
        node = np.asarray(mesh.entity('node'))
        marked_bary = node[cell[np.where(isMarked)[0]]].mean(axis=1)
        if len(marked_bary) > 0:
            print(f'      marked {int(isMarked.sum())} cells, bary avg = {marked_bary.mean(axis=0).round(3).tolist()}, |bary max_x|={np.max(np.abs(marked_bary[:,0])):.2f}')
        mesh.bisect(isMarked, options={'disp': False})


if __name__ == '__main__':
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    N0 = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    adaptive_loop(n_iter=n_iter, N0=N0, p=3, theta=theta)
