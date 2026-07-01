"""Unconstrained-space assembler for Hu-Zhang corner relaxation.

The standard fealpy ``BilinearForm.assembly()`` scatters cell-local matrices
through ``space.cell_to_dof()``, which uses the ``base_space`` global DOFs.
For the corner-relaxation wrapper to truly inject extra DOFs at NN corners,
the same cell-local matrices must instead be scattered through
``HuZhangCornerRelax.cell_to_dof_unc`` into the **unc** space (size
``gdof_unc``). This module reproduces only the scatter step, reusing fealpy
integrators for the cell-local matrices themselves.

Pipeline
========

  M_unc = scatter(integrator.assembly(base_space), c2d_unc, gdof_unc)
  M2    = TM.T @ M_unc @ TM    # final relaxed matrix, shape (gdof_rel, gdof_rel)

Same for ``B_unc`` (mixed div block) and ``F_unc`` if needed.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from fealpy.backend import backend_manager as bm
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator

from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax


def _scatter_local_to_global(local: np.ndarray, c2d_row: np.ndarray,
                             c2d_col: np.ndarray, n_row: int, n_col: int) -> sp.csr_matrix:
    """Scatter cell-local matrices (NC, vldof, uldof) using c2d_row, c2d_col.

    Args:
        local: (NC, vldof, uldof) cell-local matrices.
        c2d_row: (NC, vldof) row global ids.
        c2d_col: (NC, uldof) col global ids.
        n_row: total rows (=gdof_v).
        n_col: total cols (=gdof_u).
    """
    NC, vldof, uldof = local.shape
    rows = np.broadcast_to(c2d_row[:, :, None], (NC, vldof, uldof)).reshape(-1)
    cols = np.broadcast_to(c2d_col[:, None, :], (NC, vldof, uldof)).reshape(-1)
    vals = local.reshape(-1)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_row, n_col)).tocsr()


def assemble_M_unc(relax: HuZhangCornerRelax, *, lambda0: float, lambda1: float,
                   coef=None) -> sp.csr_matrix:
    """Assemble the stress mass matrix in the unc space."""
    base = relax.base_space
    integ = HuZhangStressIntegrator(coef=coef, lambda0=lambda0, lambda1=lambda1)
    A_local = np.asarray(integ.assembly(base))     # (NC, ldof, ldof)
    c2d_unc = relax.cell_to_dof_unc                # (NC, ldof)
    return _scatter_local_to_global(A_local, c2d_unc, c2d_unc,
                                    relax.gdof_unc, relax.gdof_unc)


def assemble_B_unc(relax: HuZhangCornerRelax, space_u) -> sp.csr_matrix:
    """Assemble the mixed div block; B[σ_dof, u_dof] = (div τ, v).

    fealpy ``HuZhangMixIntegrator.assembly((space_u, base_space))`` returns a
    cell-local matrix of shape ``(NC, σ_ldof, u_ldof)`` (σ as rows). We rescatter
    the σ side via ``c2d_unc`` and keep u-side identical.
    """
    base = relax.base_space
    integ = HuZhangMixIntegrator()
    B_local = np.asarray(integ.assembly((space_u, base)))   # (NC, σ_ldof, u_ldof)
    c2d_u = np.asarray(space_u.cell_to_dof())              # (NC, u_ldof)
    c2d_unc = relax.cell_to_dof_unc                          # (NC, σ_ldof)
    gdof_u = int(space_u.number_of_global_dofs())
    return _scatter_local_to_global(B_local, c2d_unc, c2d_u, relax.gdof_unc, gdof_u)


def project_to_rel(relax: HuZhangCornerRelax, M_unc: sp.csr_matrix,
                   B_unc: sp.csr_matrix) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Apply TM to compress (M_unc, B_unc) to the rel space.

    Returns:
        M2 = TM^T @ M_unc @ TM     shape (gdof_rel, gdof_rel)
        B2 = TM^T @ B_unc           shape (gdof_rel, gdof_u)
    """
    TM = relax.TM
    M2 = (TM.T @ M_unc @ TM).tocsr()
    B2 = (TM.T @ B_unc).tocsr()
    return M2, B2
