"""[HM18] §5.2 Williams singular solution on rotated L-shape.

Domain: rotated L-shape Ω = [-2, 2]^2 \\ {x >= 0, y <= 0}
Re-entrant corner at origin (0, 0), interior angle 2ω = 3π/2, so ω = 3π/4.

Exact solution (polar coords at origin):
    u_r(r, φ) = r^α / (2μ) [−(α+1) cos((α+1)φ) + (C_2 − α − 1) C_1 cos((α−1)φ)]
    u_φ(r, φ) =  r^α / (2μ) [(α+1) sin((α+1)φ) + (C_2 + α − 1) C_1 sin((α−1)φ)]

with:
    α = 0.544483736782  (positive root of  α sin(2ω) + sin(2αω) = 0)
    C_1 = -cos((α+1)ω) / cos((α−1)ω)
    C_2 = 2(λ + 2μ) / (λ + μ)
    E = 1e5,  ν = 0.499

σ = ε(u) via elasticity constitutive relation → singular at origin, σ ~ r^(α−1).

Boundary:
    Γ_D (Dirichlet u): the two ΓD edges around the singular corner
      (in the paper's convention these are the two straight edges emanating
       from origin along the "inside" of the L; here we use bottom horizontal
       (y=0, x<=0) and left vertical (x=0, y>=0))
    Γ_N (Neumann σn): remaining outer boundaries, values from analytic σ.

Because σ ~ r^(α−1) is singular at origin, the base mode is expected to yield
sub-optimal convergence rate ~α ≈ 0.54. Wrapper's local relaxation at the
re-entrant NN corner may help constant but not the rate.
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
import scipy.sparse as sp

from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition, HuzhangStressBoundaryCondition, build_isNedge_from_isD,
)
from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax
from fracturex.assemblers.huzhang_unc_assembler import assemble_M_unc, assemble_B_unc


# ============ Material constants ==============
E = 1e5
NU = 0.499
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
# fracturex 惯用 lambda0, lambda1: A σ = (1/λ0)(σ - λ1/(λ0-2λ1) trσ I)
# 对照标准: A σ = 1/(2μ)(σ - λ/(2(λ+μ)) trσ I) 用 plane strain 2D
# → λ0 = 2μ, λ1 = λ / (2(λ+μ)) · (λ0 - 2λ1) ... 直接推:
# ε_ij = 1/(2μ)(σ_ij - λ/(2(λ+μ)) trσ δ_ij)  → 用 c0=1/λ0, c1=λ1/(λ0 - 2λ1):
# c0 = 1/(2μ) → λ0 = 2μ
# c1 = λ/(2(λ+μ)) → λ1/(λ0 - 2λ1) = λ/(2(λ+μ))
# 令 λ0=2μ, 则 λ0-2λ1 = 2μ - 2λ1，c1 = λ1/(2μ-2λ1) = λ/(2(λ+μ))
# → λ1 = μλ/(λ+μ+λ) = μλ/(λ+2μ) ? 我算下:
# λ1(2μ+λ) - λ1·λ = ... 反解：λ1(2(λ+μ)) = λ(2μ - 2λ1) → λ1(2λ+2μ+2λ) = 2μλ
# → λ1 = μλ/(2λ+2μ+2λ)? 直接数值代入更稳
LAMBDA0 = 2 * MU
# 反推 λ1: c1 = λ1/(2μ-2λ1) = λ/(2(λ+μ)) → 2(λ+μ)·λ1 = λ·(2μ-2λ1) → λ1·(2λ+2μ+2λ) = 2μλ
LAMBDA1 = MU * LAMBDA / (2 * LAMBDA + MU)

# ============ Williams singular solution ==============
OMEGA = 3 * np.pi / 4       # 半开角，L 形凹角内部角 = 2ω = 3π/2

def _williams_alpha():
    """Positive root of α sin(2ω) + sin(2αω) = 0 in (0, 1)."""
    from scipy.optimize import brentq
    return brentq(lambda a: a * np.sin(2 * OMEGA) + np.sin(2 * a * OMEGA), 0.1, 0.99)

ALPHA = 0.544483736782  # or _williams_alpha()

C1 = -np.cos((ALPHA + 1) * OMEGA) / np.cos((ALPHA - 1) * OMEGA)
C2 = 2 * (LAMBDA + 2 * MU) / (LAMBDA + MU)


def displacement_exact(pts):
    """u(x, y) as (…, 2) array."""
    pts = np.asarray(pts)
    x = pts[..., 0]; y = pts[..., 1]
    r = np.sqrt(x * x + y * y) + 1e-30
    # 论文用 φ ∈ (0, 2ω)；旋转 L 形 (Fig 5.2) 缺角在第四象限 x>0,y<0，
    # 保留区域从底左沿逆时针到右上再到顶左，角度覆盖 φ ∈ (0, 3π/2)（若 φ 从 +x 逆时针）
    # 但 §5.2 用的是 rotated L：Fig 5.2 显示切除的是第四象限；开角从 +x 轴顺时针到 -y 轴外部
    # 采用 φ = atan2(y, x) mod 2π，取 φ ∈ [0, 2ω] = [0, 3π/2]
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    # 凹角在原点，两条 Γ_D 边（φ=0 与 φ=2ω=3π/2）
    factor = r ** ALPHA / (2 * MU)
    u_r = factor * (-(ALPHA + 1) * np.cos((ALPHA + 1) * phi)
                    + (C2 - ALPHA - 1) * C1 * np.cos((ALPHA - 1) * phi))
    u_phi = factor * ((ALPHA + 1) * np.sin((ALPHA + 1) * phi)
                      + (C2 + ALPHA - 1) * C1 * np.sin((ALPHA - 1) * phi))
    # 转 cartesian
    ux = u_r * np.cos(phi) - u_phi * np.sin(phi)
    uy = u_r * np.sin(phi) + u_phi * np.cos(phi)
    out = np.zeros(pts.shape)
    out[..., 0] = ux; out[..., 1] = uy
    return out


def stress_exact(pts):
    """σ Voigt (…, 3) computed via finite diff of u_exact (avoid symbolic grind)."""
    pts = np.asarray(pts)
    h = 1e-5
    def U(x, y):
        p = np.stack([x, y], axis=-1)
        return displacement_exact(p)
    x = pts[..., 0]; y = pts[..., 1]
    u_px = U(x + h, y); u_mx = U(x - h, y)
    u_py = U(x, y + h); u_my = U(x, y - h)
    dudx = (u_px - u_mx) / (2 * h)
    dudy = (u_py - u_my) / (2 * h)
    exx = dudx[..., 0]
    eyy = dudy[..., 1]
    exy = 0.5 * (dudx[..., 1] + dudy[..., 0])
    tr = exx + eyy
    sxx = LAMBDA * tr + 2 * MU * exx
    syy = LAMBDA * tr + 2 * MU * eyy
    sxy = 2 * MU * exy
    out = np.zeros(pts.shape[:-1] + (3,))
    out[..., 0] = sxx; out[..., 1] = sxy; out[..., 2] = syy
    return out


# ============ Mesh: rotated L-shape ==============
def make_rotated_lshape_mesh(N):
    """Ω = [-2, 2]^2 \\ (x>=0 且 y<=0). 凹角在原点。"""
    if N % 4 != 0:
        N += (4 - N % 4)
    mesh = TriangleMesh.from_box([-2., 2., -2., 2.], nx=N, ny=N)
    cell = mesh.entity('cell')
    node = mesh.entity('node')
    bary = node[cell].mean(axis=1)
    # 保留：不在"第四象限内部"
    keep = ~((bary[:, 0] > 0) & (bary[:, 1] < 0))
    new_cell = cell[keep]
    used = bm.unique(new_cell.reshape(-1))
    remap = -bm.ones(node.shape[0], dtype=new_cell.dtype)
    remap[used] = bm.arange(used.shape[0], dtype=new_cell.dtype)
    new_node = node[used]
    new_cell = remap[new_cell]
    return TriangleMesh(new_node, new_cell)


def isD_around_reentrant(bc):
    """ΓD = 凹角出发的两条边界边：y=0 (x∈[0, 2]) 和 x=0 (y∈[-2, 0])."""
    tol = 1e-9
    x = bc[:, 0]; y = bc[:, 1]
    on_yeq0 = (bm.abs(y - 0.0) < tol) & (x > -tol) & (x < 2 + tol)
    on_xeq0 = (bm.abs(x - 0.0) < tol) & (y > -2 - tol) & (y < tol)
    return on_yeq0 | on_xeq0


def solve(N: int, p: int, *, mode: str, skip_nn_corner_nodes: bool = True):
    mesh = make_rotated_lshape_mesh(N)
    isNedge = build_isNedge_from_isD(mesh, isD_around_reentrant)
    base_space = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isNedge)
    space_u_sc = LagrangeFESpace(mesh, p=p - 1, ctype='D')
    space_u = TensorFunctionSpace(space_u_sc, shape=(-1, 2))

    @barycentric
    def coef(bcs, index=None):
        return bm.ones((1,) + bcs.shape[:-1], dtype=bm.float64)
    bform1 = BilinearForm(base_space)
    bform1.add_integrator(HuZhangStressIntegrator(coef=coef, lambda0=LAMBDA0, lambda1=LAMBDA1))
    bform2 = BilinearForm((space_u, base_space))
    bform2.add_integrator(HuZhangMixIntegrator())

    @cartesian
    def src(x, index=None):
        return bm.zeros(x.shape, dtype=x.dtype)   # div σ = 0 for Williams exact soln
    L = LinearForm(space_u); L.add_integrator(VectorSourceIntegrator(source=src))
    b = L.assembly()

    HBC = HuzhangBoundaryCondition(space=base_space)
    def uD_bm(pts):
        return bm.tensor(displacement_exact(np.asarray(pts)), dtype=base_space.ftype)
    a = HBC.displacement_boundary_condition(value=uD_bm, threshold=isD_around_reentrant)

    HSBC = HuzhangStressBoundaryCondition(space=base_space)
    def sig_bm(pts):
        return bm.tensor(stress_exact(np.asarray(pts)), dtype=base_space.ftype)
    uh_stress_base, isbd_stress_base = HSBC.set_essential_bc_v2(
        sig_bm, threshold=isNedge, coord='auto', skip_nn_corner_nodes=skip_nn_corner_nodes)

    if mode == 'base':
        M = bform1.assembly().to_scipy().tocsr()
        B = bform2.assembly().to_scipy().tocsr()
        A = bmat([[M, B], [B.T, None]], format='csr')
        gdof_sig = base_space.number_of_global_dofs()
        F = np.zeros(A.shape[0])
        F[:gdof_sig] = np.asarray(a); F[gdof_sig:] = -np.asarray(b)
        uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof_sig] = np.asarray(uh_stress_base)
        isbd_g[:gdof_sig] = np.asarray(isbd_stress_base)
        ncp = 0; relax = None
    elif mode == 'relax':
        relax = HuZhangCornerRelax(mesh, p=p, isNedge=isNedge, base_space=base_space)
        ncp = relax.diag()['n_corners']
        M_unc = assemble_M_unc(relax, lambda0=LAMBDA0, lambda1=LAMBDA1, coef=coef)
        B_unc = assemble_B_unc(relax, space_u)
        C = relax.C_constraint
        nC = C.shape[0]; gdof_u = space_u.number_of_global_dofs()
        A = bmat([[M_unc, B_unc, C.T],
                  [B_unc.T, None, sp.csr_matrix((gdof_u, nC))],
                  [C, sp.csr_matrix((nC, gdof_u)), sp.csr_matrix((nC, nC))]], format='csr')
        gdof_sig = relax.gdof_unc
        # 右端: a from base → a_unc (只处理非角点段)
        n2d = np.asarray(base_space.dof.node_to_internal_dof())
        corner_base = set()
        for c in relax.corners:
            corner_base.update(n2d[c.nid].tolist())
        a_np = np.asarray(a); a_unc = np.zeros(relax.gdof_unc)
        for d in range(relax.gdof_base):
            if d in corner_base:
                continue
            a_unc[d] = a_np[d]
        for c in relax.corners:
            base3 = n2d[c.nid]
            for comp in range(3):
                bd = int(base3[comp])
                for k in range(len(c.cells)):
                    a_unc[int(c.unc_dofs[k, comp])] = a_np[bd]
        F = np.zeros(A.shape[0])
        F[:gdof_sig] = a_unc; F[gdof_sig:gdof_sig + gdof_u] = -np.asarray(b)
        # essential BC：只复制非角点 base id 上 fealpy 已经写好的 essential。
        # 角点上的 σ 完全交给"边内部 trace + C 约束 + 变分方程"决定（A 方案）。
        uh_base_np = np.asarray(uh_stress_base)
        isbd_base_np = np.asarray(isbd_stress_base)
        uh_unc = np.zeros(relax.gdof_unc); isbd_unc = np.zeros(relax.gdof_unc, dtype=bool)
        for d in range(relax.gdof_base):
            if d in corner_base:
                continue
            if isbd_base_np[d]:
                uh_unc[d] = uh_base_np[d]; isbd_unc[d] = True
        uh_g = np.zeros(A.shape[0]); isbd_g = np.zeros(A.shape[0], dtype=bool)
        uh_g[:gdof_sig] = uh_unc; isbd_g[:gdof_sig] = isbd_unc
    else:
        raise ValueError(mode)

    F = F - A @ uh_g; F[isbd_g] = uh_g[isbd_g]
    n = A.shape[0]; bdIdx = isbd_g.astype(int)
    A = spdiags(1 - bdIdx, 0, n, n) @ A @ spdiags(1 - bdIdx, 0, n, n) + spdiags(bdIdx, 0, n, n)
    X = spsolve(A, F)

    if mode == 'base':
        sigmah = base_space.function(); sigmah[:] = X[:gdof_sig]
        e_s = float(mesh.error(sigmah, stress_exact, q=2 * p + 6))
    else:
        e_s = relax.l2_error_unc(X[:gdof_sig], stress_exact, is_unc=True, q=2 * p + 6)
    return e_s, ncp


def main(p: int = 3, maxit: int = 4):
    print(f"[HM18-Williams] rotated L-shape, α={ALPHA:.6f}, p={p}")
    for mode in ('base', 'relax'):
        print(f"\n=== mode={mode} ===")
        print(f"{'N':>5} {'NCP':>5} {'|σ-σh|':>14} {'rate':>6}")
        prev = None
        for i in range(maxit):
            N = 2 ** (i + 2)   # 4, 8, 16, 32
            e, ncp = solve(N, p, mode=mode)
            rate = '-' if prev is None else f"{np.log2(prev / e):.2f}"
            print(f"{N:>5} {ncp:>5} {e:>14.4e} {rate:>6}")
            prev = e


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    maxit = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(p, maxit)
