"""
双调和方程的内罚 (Bernstein/Lagrange) 有限元求解器。

组装 A = A_biharmonic + A_penalty, 右端 f = ScalarSourceIntegrator,
用节点插值 + Dirichlet 边界条件把边界自由度替换成解析值，最后 spsolve。

所有稀疏矩阵都在 numpy/scipy 层面处理 (fealpy 组装出来就是 scipy.sparse)，
tensor 场值出入口再用 bm.to_numpy 兜一下，保证 numpy/pytorch/jax 都能跑。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import InteriorPenaltyFESpace2d
from fealpy.fem import (
    BilinearForm,
    LinearForm,
    ScalarBiharmonicIntegrator,
    ScalarSourceIntegrator,
)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        return bm.to_numpy(x)
    except Exception:
        return np.asarray(x)


def _to_scipy_csr(A):
    """把 fealpy 输出的矩阵（scipy csr / fealpy CSRTensor）统一成 scipy csr。"""
    if isinstance(A, csr_matrix):
        return A
    if hasattr(A, "indices") and hasattr(A, "indptr"):
        values_attr = getattr(A, "values", None)
        if callable(values_attr):
            data = _to_numpy(values_attr())
        elif values_attr is not None:
            data = _to_numpy(values_attr)
        else:
            data = _to_numpy(A.data)
        indices = _to_numpy(A.indices)
        indptr = _to_numpy(A.indptr)
        return csr_matrix((data, indices, indptr), shape=tuple(A.shape))
    if hasattr(A, "row") and hasattr(A, "col"):
        data = _to_numpy(getattr(A, "data", getattr(A, "values", None)))
        row = _to_numpy(A.row)
        col = _to_numpy(A.col)
        return csr_matrix((data, (row, col)), shape=tuple(A.shape)).tocsr()
    if hasattr(A, "toarray"):
        return csr_matrix(_to_numpy(A.toarray()))
    raise TypeError(f"unknown sparse matrix type: {type(A)}")


def _assemble_interior_penalty(space, gamma, q, edge_coef_inner=None, edge_coef_bd=None):
    """
    C^0 IPDG 内罚项，用于双调和方程。

    fealpy v3 里 `ScalarInteriorPenaltyIntegrator` 的 consistency 项符号取正，
    导致方法非一致（对齐 Brenner-Sung/SIP 应为负）。这里按老代码 (fealpy_old_2)
    的约定组装 —— 内部 / 边界边都用 `-grad_normal_jump_basis`：

        a_P(u,v) = -∫_e {∂²u/∂n²}[∂v/∂n]
                   -∫_e {∂²v/∂n²}[∂u/∂n]
                   +γ ∫_e [∂u/∂n][∂v/∂n]

    Parameters
    ----------
    edge_coef_inner
        Optional (NIE,) 数组，作为内部边的 cell-average scalar 权系数
        (整个 P1+P2+P2T 一起乘)。默认 None → 权系数 = 1。
    edge_coef_bd
        Optional (NBE,) 数组，作为边界边的权系数。
    """
    mesh = space.mesh
    isBdEdge = mesh.boundary_edge_flag()
    isInnerEdge = ~isBdEdge
    em = mesh.entity_measure("edge")
    qf = mesh.quadrature_formula(q, "edge")
    bcs, ws = qf.get_quadrature_points_and_weights()

    # 内部边
    gnjphi = -space.grad_normal_jump_basis(bcs)
    gn2jphi = space.grad_grad_normal_jump_basis(bcs)
    P1 = bm.einsum("q, qfi, qfj->fij", ws, gnjphi, gnjphi) * gamma
    P2 = bm.einsum("q, qfi, qfj, f->fij", ws, gnjphi, gn2jphi, em[isInnerEdge])
    P2T = bm.permute_dims(P2, axes=(0, 2, 1))
    Pi = P1 + P2 + P2T
    if edge_coef_inner is not None:
        w = np.asarray(edge_coef_inner, dtype=np.float64).reshape(-1)
        Pi = Pi * w[:, None, None]

    ie2cd = space.dof.iedge2celldof
    be2cd = space.dof.bedge2celldof
    gdof = space.number_of_global_dofs()
    I = bm.broadcast_to(ie2cd[:, :, None], Pi.shape)
    J = bm.broadcast_to(ie2cd[:, None, :], Pi.shape)
    Mat = csr_matrix(
        (_to_numpy(Pi).ravel(), (_to_numpy(I).ravel(), _to_numpy(J).ravel())),
        shape=(gdof, gdof),
    )

    # 边界边
    bgnjphi = -space.boundary_edge_grad_normal_jump_basis(bcs)
    bggnjphi = space.boundary_edge_grad_grad_normal_jump_basis(bcs)
    Pb1 = bm.einsum("q, qfi, qfj->fij", ws, bgnjphi, bgnjphi) * gamma
    Pb2 = bm.einsum("q, qfi, qfj, f->fij", ws, bgnjphi, bggnjphi, em[isBdEdge])
    Pb2T = bm.permute_dims(Pb2, axes=(0, 2, 1))
    Pb = Pb1 + Pb2 + Pb2T
    if edge_coef_bd is not None:
        w = np.asarray(edge_coef_bd, dtype=np.float64).reshape(-1)
        Pb = Pb * w[:, None, None]

    Ib = bm.broadcast_to(be2cd[:, :, None], Pb.shape)
    Jb = bm.broadcast_to(be2cd[:, None, :], Pb.shape)
    Mat = Mat + csr_matrix(
        (_to_numpy(Pb).ravel(), (_to_numpy(Ib).ravel(), _to_numpy(Jb).ravel())),
        shape=(gdof, gdof),
    )
    return Mat


def _apply_dirichlet(A, f, uh, isDDof):
    A = _to_scipy_csr(A)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    uh = np.asarray(uh, dtype=np.float64).reshape(-1)
    isDDof = np.asarray(isDDof).reshape(-1).astype(bool)

    f = f - A @ uh
    bdIdx = np.zeros(A.shape[0], dtype=np.int64)
    bdIdx[isDDof] = 1
    D0 = spdiags(1 - bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0 @ A @ D0 + D1
    f[isDDof] = uh[isDDof]
    return A, f


def _boundary_dof_mask(pts_xy, box):
    x0, x1, y0, y1 = box
    eps = 1e-12
    x = pts_xy[..., 0]
    y = pts_xy[..., 1]
    return (x < x0 + eps) | (x > x1 - eps) | (y < y0 + eps) | (y > y1 - eps)


def solve_biharmonic(pde, nx, ny, p, box=(0.0, 1.0, 0.0, 1.0),
                    gamma=5.0, q=None, space_type="Lagrange"):
    """
    在 [x0,x1] x [y0,y1] 上用 p 次内罚有限元求解双调和方程。

    返回 (uh_numpy, mesh, space, ipoints)
    """
    x0, x1, y0, y1 = box
    mesh = TriangleMesh.from_box([x0, x1, y0, y1], nx=nx, ny=ny)
    space = InteriorPenaltyFESpace2d(mesh, p=p, space=space_type)

    q = 2 * p + 3 if q is None else q

    bform = BilinearForm(space)
    bform.add_integrator(ScalarBiharmonicIntegrator(q=q))
    A0 = bform.assembly()

    P = _assemble_interior_penalty(space, gamma=gamma, q=q)
    A = _to_scipy_csr(A0) + P

    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source, q=q))
    b = _to_numpy(lform.assembly()).reshape(-1)

    ipoints_t = mesh.interpolation_points(p=p)
    ipoints = _to_numpy(ipoints_t)

    exact = pde.solution(ipoints)
    isBd = _boundary_dof_mask(ipoints, box)
    gd = np.zeros_like(exact, dtype=np.float64)
    gd[isBd] = exact[isBd]

    A, f = _apply_dirichlet(A, b, gd, isBd)

    uh_arr = spsolve(A, f)
    return uh_arr, mesh, space, ipoints


def l2_error_from_dof(uh_arr, mesh, space, pde, q=None):
    """离散 L2 误差：∫(uh - u)^2 dx, 在单元上高斯积分。"""
    p = space.p
    q = 2 * p + 3 if q is None else q
    qf = mesh.quadrature_formula(q, "cell")
    bcs, ws = qf.get_quadrature_points_and_weights()
    ws = _to_numpy(ws)

    phi = _to_numpy(space.basis(bcs))  # (1 or NC, NQ, ldof)
    c2d = _to_numpy(space.cell_to_dof())  # (NC, ldof)
    uh_cell = uh_arr[c2d]  # (NC, ldof)
    if phi.shape[0] == 1:
        uh_q = np.einsum("Nql, cl -> cq", phi, uh_cell)  # (NC, NQ)
    else:
        uh_q = np.einsum("cql, cl -> cq", phi, uh_cell)

    pts = _to_numpy(mesh.bc_to_point(bcs))  # (NC, NQ, 2)
    u_q = pde.solution(pts)  # (NC, NQ)

    cm = _to_numpy(mesh.entity_measure("cell"))
    diff = (uh_q - u_q) ** 2
    val = np.einsum("q, cq, c ->", ws, diff, cm)
    return float(np.sqrt(val))


def h1_semi_error_from_dof(uh_arr, mesh, space, pde, q=None):
    """H1 半范数误差：sqrt(∫|∇uh - ∇u|^2 dx)。"""
    p = space.p
    q = 2 * p + 3 if q is None else q
    qf = mesh.quadrature_formula(q, "cell")
    bcs, ws = qf.get_quadrature_points_and_weights()
    ws = _to_numpy(ws)

    gphi = _to_numpy(space.grad_basis(bcs, variable="x"))  # (NC, NQ, ldof, 2)
    c2d = _to_numpy(space.cell_to_dof())
    uh_cell = uh_arr[c2d]
    guh = np.einsum("cqld, cl -> cqd", gphi, uh_cell)  # (NC, NQ, 2)

    pts = _to_numpy(mesh.bc_to_point(bcs))  # (NC, NQ, 2)
    gu = pde.gradient(pts)  # (NC, NQ, 2)

    cm = _to_numpy(mesh.entity_measure("cell"))
    diff = np.sum((guh - gu) ** 2, axis=-1)  # (NC, NQ)
    val = np.einsum("q, cq, c ->", ws, diff, cm)
    return float(np.sqrt(val))


def h2_semi_error_from_dof(uh_arr, mesh, space, pde, q=None):
    """H2 半范数误差：sqrt(∫|Hess uh - Hess u|^2 dx)。"""
    p = space.p
    q = 2 * p + 3 if q is None else q
    qf = mesh.quadrature_formula(q, "cell")
    bcs, ws = qf.get_quadrature_points_and_weights()
    ws = _to_numpy(ws)

    hphi = _to_numpy(space.hess_basis(bcs, variable="x"))  # (NC, NQ, ldof, 2, 2)
    c2d = _to_numpy(space.cell_to_dof())
    uh_cell = uh_arr[c2d]
    huh = np.einsum("cqlij, cl -> cqij", hphi, uh_cell)  # (NC, NQ, 2, 2)

    pts = _to_numpy(mesh.bc_to_point(bcs))  # (NC, NQ, 2)
    hu = pde.hessian(pts)  # (NC, NQ, 2, 2)

    cm = _to_numpy(mesh.entity_measure("cell"))
    diff = np.sum((huh - hu) ** 2, axis=(-1, -2))  # (NC, NQ)
    val = np.einsum("q, cq, c ->", ws, diff, cm)
    return float(np.sqrt(val))


def convergence_study(pde, p, maxit=4, nx0=4, box=(0.0, 1.0, 0.0, 1.0),
                     gamma=5.0, space_type="Lagrange"):
    """在一系列加密网格上跑求解，返回 (h, errL2, errH1, errH2)。"""
    h = np.zeros(maxit)
    errL2 = np.zeros(maxit)
    errH1 = np.zeros(maxit)
    errH2 = np.zeros(maxit)
    nx = nx0
    for i in range(maxit):
        uh_arr, mesh, space, _ = solve_biharmonic(
            pde, nx=nx, ny=nx, p=p, box=box,
            gamma=gamma, space_type=space_type,
        )
        cm = _to_numpy(mesh.entity_measure("cell"))
        h[i] = float(np.sqrt(np.max(cm)))
        errL2[i] = l2_error_from_dof(uh_arr, mesh, space, pde)
        errH1[i] = h1_semi_error_from_dof(uh_arr, mesh, space, pde)
        errH2[i] = h2_semi_error_from_dof(uh_arr, mesh, space, pde)
        nx *= 2
    return h, errL2, errH1, errH2


def compute_orders(errors, h):
    orders = np.zeros_like(errors)
    for i in range(1, len(errors)):
        if errors[i - 1] > 0 and errors[i] > 0:
            orders[i] = np.log2(errors[i - 1] / errors[i]) / np.log2(h[i - 1] / h[i])
    return orders
