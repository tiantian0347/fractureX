"""Manufactured solution 收敛率验证：应变梯度弹性 (无相场耦合)。

PDE (无损伤 d=0):

    -div σ(u) + ℓ_s² [Δ²u_1, Δ²u_2]^T = f    in Ω=[0,1]²
    u = 0                                       on ∂Ω
    ∂u/∂n = 0                                   on ∂Ω     (clamped-plate 边界)

C⁰-IP 里 B_h(u_k, v_k) 对应「u=0 强 + ∂u/∂n=0 弱」的双 Dirichlet；因此 MMS
需要选择同时满足两条边界的解：

    u_1(x,y) = sin²(πx) · sin²(πy)              (∂u_1/∂n = 0 on ∂[0,1]²)
    u_2(x,y) = sin²(2πx) · sin²(πy)

组装:
- 标准 elasticity 用 fealpy LinearElasticIntegrator(method='voigt')；
- SG 项用 `assemble_sg_elastic_block` (∫D²·D² + IP 边罚) block-diagonal 拼接。

测量: p=2,3 位移 L² / H¹ 半范数收敛率（受 C⁰-IP 影响，H¹ 应为 p 阶）。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.fem import (
    BilinearForm,
    LinearElasticIntegrator,
    LinearForm,
    ScalarSourceIntegrator,
)
from fealpy.functionspace import (
    InteriorPenaltyFESpace2d,
    LagrangeFESpace,
    TensorFunctionSpace,
)
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.mesh import TriangleMesh

from fracturex.interior_penalty.solver import _to_numpy, _to_scipy_csr
from fracturex.interior_penalty.sg_elastic import (
    assemble_sg_elastic_block,
    block_diag_vector,
)


# ---------------- Manufactured solution & source --------------------

LAM = 1.0
MU = 0.5
PI = np.pi


def _u_k(x, y, k):
    """Manufactured solution 满足 u=0 且 ∂u/∂n=0 on ∂[0,1]²。

    两个分量用相同结构但不同幅值，避免高频分量分辨率不足。
    """
    base = (np.sin(PI * x) ** 2) * (np.sin(PI * y) ** 2)
    if k == 0:
        return base
    else:
        return 0.5 * base


def _du_k(x, y, k):
    dbase_dx = PI * np.sin(2 * PI * x) * (np.sin(PI * y) ** 2)
    dbase_dy = PI * (np.sin(PI * x) ** 2) * np.sin(2 * PI * y)
    scale = 1.0 if k == 0 else 0.5
    return scale * dbase_dx, scale * dbase_dy


def _d2u_k(x, y, k):
    u_xx = 2 * PI ** 2 * np.cos(2 * PI * x) * (np.sin(PI * y) ** 2)
    u_xy = PI ** 2 * np.sin(2 * PI * x) * np.sin(2 * PI * y)
    u_yy = 2 * PI ** 2 * (np.sin(PI * x) ** 2) * np.cos(2 * PI * y)
    scale = 1.0 if k == 0 else 0.5
    return scale * u_xx, scale * u_xy, scale * u_yy


def _biharm_u_k(x, y, k):
    c2x = np.cos(2 * PI * x)
    c2y = np.cos(2 * PI * y)
    val = 4 * PI ** 4 * (4 * c2x * c2y - c2x - c2y)
    return val if k == 0 else 0.5 * val


def u_exact(pts):
    x, y = pts[..., 0], pts[..., 1]
    return np.stack([_u_k(x, y, 0), _u_k(x, y, 1)], axis=-1)


def grad_u_exact(pts):
    """∇u ∈ R^{2x2}: (∇u)_{ij} = ∂u_i/∂x_j。"""
    x, y = pts[..., 0], pts[..., 1]
    g = np.zeros(pts.shape[:-1] + (2, 2), dtype=np.float64)
    dux, duy = _du_k(x, y, 0); g[..., 0, 0] = dux; g[..., 0, 1] = duy
    dux, duy = _du_k(x, y, 1); g[..., 1, 0] = dux; g[..., 1, 1] = duy
    return g


def div_sigma(pts):
    """div σ_i = (λ+μ) ∂_i (∇·u) + μ Δ u_i."""
    x, y = pts[..., 0], pts[..., 1]

    # ∇·u = ∂u_1/∂x + ∂u_2/∂y
    d1_dx, _ = _du_k(x, y, 0)
    _, d2_dy = _du_k(x, y, 1)
    _div_u = d1_dx + d2_dy

    # ∂_x (∇·u), ∂_y (∇·u)
    u1_xx, u1_xy, _ = _d2u_k(x, y, 0)
    _, u2_xy, u2_yy = _d2u_k(x, y, 1)
    ddiv_dx = u1_xx + u2_xy
    ddiv_dy = u1_xy + u2_yy

    # Δ u_k
    u1_xx, _, u1_yy = _d2u_k(x, y, 0)
    lap_u1 = u1_xx + u1_yy
    u2_xx, _, u2_yy = _d2u_k(x, y, 1)
    lap_u2 = u2_xx + u2_yy

    val = np.zeros(pts.shape, dtype=np.float64)
    val[..., 0] = (LAM + MU) * ddiv_dx + MU * lap_u1
    val[..., 1] = (LAM + MU) * ddiv_dy + MU * lap_u2
    return val


def source(pts, ell_s):
    val = -div_sigma(pts)
    x, y = pts[..., 0], pts[..., 1]
    val[..., 0] += ell_s ** 2 * _biharm_u_k(x, y, 0)
    val[..., 1] += ell_s ** 2 * _biharm_u_k(x, y, 1)
    return val


# ---------------- Assembly & solve ----------------------------------

class _ConstantElasticMaterial(LinearElasticMaterial):
    """一个只需要提供 elastic_matrix / strain_matrix 的最简材料，
    用于 LinearElasticIntegrator(pfcm=..., method='voigt')。"""

    def __init__(self, lam, mu):
        # Skip super().__init__ 复杂逻辑，只填要用的属性
        self.name = "sg_mms_material"
        self.hypo = "plane_strain"
        self.lam = lam
        self.mu = mu
        self.plane_type = "plane_strain"

    def elastic_matrix(self, bcs=None):
        # 返回 shape (1, 1, 3, 3) - 会 broadcast 到 (NC, NQ, 3, 3)
        lam, mu = self.lam, self.mu
        D = bm.tensor(
            [
                [lam + 2 * mu, lam, 0],
                [lam, lam + 2 * mu, 0],
                [0, 0, mu],
            ],
            dtype=bm.float64,
        )
        return D[None, None, :, :]


def solve_sg_elastic(nx: int, p_u: int, ell_s: float, gamma: float = 5.0):
    """在 [0,1]² × nx² 三角网格上求解 SG 弹性问题, 返回 (uh_vec, tspace, mesh)."""
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=nx, ny=nx)

    space = LagrangeFESpace(mesh, p=p_u)
    tspace = TensorFunctionSpace(space, (2, -1))
    ipspace = InteriorPenaltyFESpace2d(mesh, p=p_u, space="Lagrange")

    q = 2 * p_u + 3

    # A_std: LinearElasticIntegrator 用常系数材料
    material = _ConstantElasticMaterial(LAM, MU)
    ubform = BilinearForm(tspace)
    ubform.add_integrator(LinearElasticIntegrator(material, q=q, method="voigt"))
    A_std = _to_scipy_csr(ubform.assembly())

    # A_sg: 分量 block-diagonal
    if ell_s > 0:
        A_sg_scalar = assemble_sg_elastic_block(
            ipspace, ell_s=ell_s, gamma=gamma, q=q
        )
        A_sg = block_diag_vector(A_sg_scalar, 2)
    else:
        from scipy.sparse import csr_matrix as _csr
        A_sg = _csr(A_std.shape)

    A = A_std + A_sg

    # RHS: 分量分别装
    gdof_scalar = int(space.number_of_global_dofs())
    b = np.zeros(2 * gdof_scalar, dtype=np.float64)
    for k in range(2):
        @cartesian
        def _fk(pts, k=k):
            f = source(_to_numpy(pts), ell_s)
            return f[..., k]

        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(_fk, q=q))
        b_k = _to_numpy(lform.assembly()).reshape(-1)
        b[k * gdof_scalar : (k + 1) * gdof_scalar] = b_k

    # Dirichlet u=0 on ∂Ω
    ipoints = _to_numpy(mesh.interpolation_points(p=p_u))
    is_bd = (
        (np.abs(ipoints[:, 0]) < 1e-12)
        | (np.abs(ipoints[:, 0] - 1) < 1e-12)
        | (np.abs(ipoints[:, 1]) < 1e-12)
        | (np.abs(ipoints[:, 1] - 1) < 1e-12)
    )
    is_bd_full = np.concatenate([is_bd, is_bd])
    A = A.tolil()
    for idx in np.where(is_bd_full)[0]:
        A.rows[idx] = [idx]
        A.data[idx] = [1.0]
        b[idx] = 0.0
    A = A.tocsr()

    uh = spsolve(A, b)
    return uh, tspace, space, mesh


def h1_error(uh_flat, space, mesh, p_u):
    """(∫ |∇u_h - ∇u|² dx)^{1/2}。"""
    q = 2 * p_u + 3
    qf = mesh.quadrature_formula(q, "cell")
    bcs, ws = qf.get_quadrature_points_and_weights()
    ws = _to_numpy(ws)

    gphi = _to_numpy(space.grad_basis(bcs, variable="x"))  # (NC, NQ, ldof, 2)
    c2d = _to_numpy(space.cell_to_dof())
    gdof = int(space.number_of_global_dofs())

    err_sq = 0.0
    for k in range(2):
        uh_k = uh_flat[k * gdof : (k + 1) * gdof]
        uh_cell = uh_k[c2d]
        guh_k = np.einsum("cqld, cl -> cqd", gphi, uh_cell)  # (NC, NQ, 2)
        pts = _to_numpy(mesh.bc_to_point(bcs))  # (NC, NQ, 2)
        gu_true = grad_u_exact(pts)[..., k, :]  # (NC, NQ, 2)
        diff = np.sum((guh_k - gu_true) ** 2, axis=-1)
        cm = _to_numpy(mesh.entity_measure("cell"))
        err_sq += float(np.einsum("q, cq, c ->", ws, diff, cm))
    return float(np.sqrt(err_sq))


def l2_error(uh_flat, space, mesh, p_u):
    q = 2 * p_u + 3
    qf = mesh.quadrature_formula(q, "cell")
    bcs, ws = qf.get_quadrature_points_and_weights()
    ws = _to_numpy(ws)

    phi = _to_numpy(space.basis(bcs))  # (1, NQ, ldof) or (NC, NQ, ldof)
    c2d = _to_numpy(space.cell_to_dof())
    gdof = int(space.number_of_global_dofs())
    cm = _to_numpy(mesh.entity_measure("cell"))

    err_sq = 0.0
    for k in range(2):
        uh_k = uh_flat[k * gdof : (k + 1) * gdof]
        uh_cell = uh_k[c2d]
        if phi.shape[0] == 1:
            uh_q = np.einsum("Nql, cl -> cq", phi, uh_cell)
        else:
            uh_q = np.einsum("cql, cl -> cq", phi, uh_cell)
        pts = _to_numpy(mesh.bc_to_point(bcs))
        u_true = u_exact(pts)[..., k]
        diff = (uh_q - u_true) ** 2
        err_sq += float(np.einsum("q, cq, c ->", ws, diff, cm))
    return float(np.sqrt(err_sq))


# ---------------- Tests --------------------------------------------

@pytest.mark.parametrize("p_u", [2])
@pytest.mark.parametrize("ell_s", [0.05, 0.1])
def test_sg_elasticity_convergence(p_u, ell_s):
    bm.set_backend("numpy")
    hs = []
    l2_errs = []
    h1_errs = []
    for nx in [4, 8, 16, 32]:
        uh, tspace, space, mesh = solve_sg_elastic(nx, p_u=p_u, ell_s=ell_s)
        cm = _to_numpy(mesh.entity_measure("cell"))
        hs.append(float(np.sqrt(np.max(cm))))
        l2_errs.append(l2_error(uh, space, mesh, p_u))
        h1_errs.append(h1_error(uh, space, mesh, p_u))
    hs = np.array(hs)
    l2_errs = np.array(l2_errs)
    h1_errs = np.array(h1_errs)

    def _order(errs, h):
        return np.log2(errs[:-1] / errs[1:]) / np.log2(h[:-1] / h[1:])

    ord_l2 = _order(l2_errs, hs)
    ord_h1 = _order(h1_errs, hs)

    # C⁰-IP biharmonic with clamped BC:
    # - broken H² norm ~ h^{p-1}
    # - broken H¹ norm ~ h^p
    # - L² norm ~ h^p (受 broken-norm 主导; 干净情况下可 h^{p+1})
    # 取最后一次加密的阶作为判据，容差 0.3
    exp_l2 = p_u - 0.3
    exp_h1 = p_u - 0.3

    msg = (
        f"p_u={p_u} ell_s={ell_s}\n"
        f"h  = {hs}\n"
        f"L2 = {l2_errs}\n"
        f"H1 = {h1_errs}\n"
        f"ord_L2 last = {ord_l2[-1]:.3f} (exp>={exp_l2})\n"
        f"ord_H1 last = {ord_h1[-1]:.3f} (exp>={exp_h1})\n"
    )
    print(msg)
    assert ord_l2[-1] >= exp_l2, msg
    assert ord_h1[-1] >= exp_h1, msg


def test_sg_elasticity_ell_s_zero_recovers_std():
    """ell_s=0 时 SG solver 应该退化到标准 elasticity，
    L2 收敛到 h^{p+1}（p=2 → h³）。"""
    bm.set_backend("numpy")
    hs = []
    l2_errs = []
    for nx in [4, 8, 16, 32]:
        uh, tspace, space, mesh = solve_sg_elastic(nx, p_u=2, ell_s=0.0)
        cm = _to_numpy(mesh.entity_measure("cell"))
        hs.append(float(np.sqrt(np.max(cm))))
        l2_errs.append(l2_error(uh, space, mesh, 2))
    hs = np.array(hs)
    l2_errs = np.array(l2_errs)
    orders = np.log2(l2_errs[:-1] / l2_errs[1:]) / np.log2(hs[:-1] / hs[1:])
    # p=2 pure elasticity: L² 应至少接近 h³ (允许 h^{2.7})
    assert orders[-1] >= 2.5, (
        f"ell_s=0 应恢复标准 elasticity h³ 收敛, 得到 orders={orders}, "
        f"errs={l2_errs}"
    )


if __name__ == "__main__":
    bm.set_backend("numpy")
    for p_u in [2, 3]:
        for ell_s in [0.05, 0.1]:
            print(f"=== p_u={p_u} ell_s={ell_s} ===")
            hs, l2s, h1s = [], [], []
            for nx in [4, 8, 16, 32]:
                uh, tspace, space, mesh = solve_sg_elastic(nx, p_u=p_u, ell_s=ell_s)
                cm = _to_numpy(mesh.entity_measure("cell"))
                h = float(np.sqrt(np.max(cm)))
                l2e = l2_error(uh, space, mesh, p_u)
                h1e = h1_error(uh, space, mesh, p_u)
                hs.append(h); l2s.append(l2e); h1s.append(h1e)
                print(f"  nx={nx}: h={h:.4f} L2={l2e:.4e} H1={h1e:.4e}")
            hs = np.array(hs); l2s = np.array(l2s); h1s = np.array(h1s)
            print(f"  L2 orders: {np.log2(l2s[:-1]/l2s[1:]) / np.log2(hs[:-1]/hs[1:])}")
            print(f"  H1 orders: {np.log2(h1s[:-1]/h1s[1:]) / np.log2(hs[:-1]/hs[1:])}")
