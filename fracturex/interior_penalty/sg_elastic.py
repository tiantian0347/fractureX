"""Strain-gradient elasticity C0-IP 装配（配 IPFEMPhaseFieldSGSolver 用）。

对齐 `ipfem_paper.tex` §Discretization 里 §"Extension to strain-gradient elasticity" 加的双线性形式：

    B_h^u(w, z) = Σ_k B_h(w_k, z_k)

其中 `B_h` 是标量 4 阶 C0-IP 双线性形式（∫ D²·D² + 边罚项，参见
`ScalarBiharmonicIntegrator` 与 `_assemble_interior_penalty`）。总位移块 LHS 里
在标准 elasticity 项之上再加 `g(d) * ℓ_s² * B_h^u(u, v)`。

由于 fealpy TensorFunctionSpace((GD, -1)) 用 dof_priority=True 的布局
(component 0 用标量 dof 0..gdof-1，component 1 用 gdof..2*gdof-1)，
我们只需组标量矩阵 A_s，然后 block-diagonal 拼接到 2*gdof 的向量 dof 上。

可选 cell-wise 权重（Ali 2024 谱分解版）：把 α_cell = g(d)·χ_+(ε)
作为分片常数系数馈给 biharmonic + IP，用于拉伸区选择性退化。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, block_diag as scipy_block_diag

from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric
from fealpy.fem import BilinearForm, ScalarBiharmonicIntegrator
from fealpy.functionspace import InteriorPenaltyFESpace2d

from .solver import _to_numpy, _to_scipy_csr, _assemble_interior_penalty


def _edge_coef_from_cell(mesh, cell_coef: np.ndarray):
    """把 (NC,) 的 cell 权系数下探到 (NIE,) 内部边 + (NBE,) 边界边上。

    内部边 → 取相邻两 cell 的均值；边界边 → 取所属唯一 cell 的值。
    """
    e2c = _to_numpy(mesh.edge_to_cell())  # (NE, 4): [c0, c1, e0_local, e1_local]
    isBdEdge = _to_numpy(mesh.boundary_edge_flag())
    cell_coef = np.asarray(cell_coef, dtype=np.float64).reshape(-1)

    inner_mask = ~isBdEdge
    left = e2c[inner_mask, 0]
    right = e2c[inner_mask, 1]
    edge_coef_inner = 0.5 * (cell_coef[left] + cell_coef[right])

    bd_cell = e2c[isBdEdge, 0]
    edge_coef_bd = cell_coef[bd_cell]

    return edge_coef_inner, edge_coef_bd


def assemble_sg_elastic_block(
    ipspace_u: InteriorPenaltyFESpace2d,
    *,
    ell_s: float,
    gamma: float = 5.0,
    q: Optional[int] = None,
    scalar_coef: Optional[np.ndarray] = None,
    cell_coef: Optional[np.ndarray] = None,
) -> csr_matrix:
    """
    组装标量 ℓ_s² · B_h(w, z) 矩阵（一次调用给一个 component 用）。

    Parameters
    ----------
    ipspace_u
        位移向量场底下的标量内罚空间 (p_u >= 2)。
    ell_s
        应变梯度长度尺度；`ell_s == 0` 时直接返回零矩阵。
    gamma
        C0-IP 罚参。
    q
        积分阶（默认 2*p+3）。
    scalar_coef
        (NC,) 数组。空间平均近似（早期版本兼容，未来会移除）。若和 cell_coef 同时给，cell_coef 优先。
    cell_coef
        (NC,) 数组。作 piecewise-constant cell 权系数：
        biharmonic 项用 `ScalarBiharmonicIntegrator(coef=α_cell)`, IP 边罚用
        edge-average of α_cell 作 (NIE,) / (NBE,) 权系数。
        与 tex §"Tensile-only strain-gradient degradation" 一致。
    """
    gdof = int(ipspace_u.number_of_global_dofs())
    if ell_s == 0.0:
        return csr_matrix((gdof, gdof))

    p = ipspace_u.p
    if p < 2:
        raise ValueError(
            f"strain-gradient 需要 p_u >= 2，但 ipspace_u.p = {p}."
        )
    q = 2 * p + 3 if q is None else q

    mesh = ipspace_u.mesh

    if cell_coef is not None:
        cell_coef = np.asarray(cell_coef, dtype=np.float64).reshape(-1)
        NC = mesh.number_of_cells()
        assert cell_coef.shape[0] == NC, (
            f"cell_coef 长度 {cell_coef.shape[0]} != NC {NC}"
        )

        @barycentric
        def _bh_coef(bc, index):
            # (NC,) -> (NC, NQ)
            NQ = bc.shape[0]
            arr = bm.tensor(cell_coef, dtype=bm.float64)
            if index is None or isinstance(index, slice):
                pass
            return arr[..., None] * bm.ones((cell_coef.shape[0], NQ), dtype=bm.float64)

        bform = BilinearForm(ipspace_u)
        bform.add_integrator(ScalarBiharmonicIntegrator(coef=_bh_coef, q=q))
        A_bh = _to_scipy_csr(bform.assembly())

        edge_ci, edge_cb = _edge_coef_from_cell(mesh, cell_coef)
        A_ip = _assemble_interior_penalty(
            ipspace_u, gamma=gamma, q=q,
            edge_coef_inner=edge_ci, edge_coef_bd=edge_cb,
        )
        A = A_bh + A_ip
    else:
        bform = BilinearForm(ipspace_u)
        bform.add_integrator(ScalarBiharmonicIntegrator(q=q))
        A_bh = _to_scipy_csr(bform.assembly())
        A_ip = _assemble_interior_penalty(ipspace_u, gamma=gamma, q=q)
        A = A_bh + A_ip

        if scalar_coef is not None:
            avg = float(np.mean(np.asarray(scalar_coef, dtype=np.float64)))
            A = A * avg

    return (ell_s ** 2) * A


def block_diag_vector(A_scalar: csr_matrix, gd: int) -> csr_matrix:
    """把标量 A_scalar 扩成 gd*gdof × gd*gdof 的 block-diagonal 向量矩阵。"""
    return scipy_block_diag([A_scalar] * gd).tocsr()
