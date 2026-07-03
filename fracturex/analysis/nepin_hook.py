"""Bridge between the fracturex phase-field assembler and the NEPIN kernel.

Given a ``PhaseFieldSystem`` produced by
``fracturex.assemblers.phasefield_assembler.PhaseFieldAssembler.assemble``,
this module builds the ``residual`` and ``jacobian`` callbacks that
``fracturex.analysis.nonlinear_elimination.NEPINEliminator`` requires.

Contract
--------
The assembler returns ``A dd = rhs - A d_old`` (residual/increment form),
so ``sys_d.F = rhs - A @ d_old`` and ``sys_d.A`` is the (constant, for the
frozen elastic + history state) phase-field stiffness matrix. NEPIN wants
the full residual :math:`R(d) = A d - rhs` evaluated on a *candidate*
damage vector; we therefore freeze ``rhs = sys_d.F + sys_d.A @ d_old``
once and reuse it inside the callbacks.

For frozen ``u`` and ``H``, the phase-field residual is affine in ``d``,
so the sub-Jacobian ``A[S][:, S]`` is independent of ``d`` and a single
LU factorization inside the kernel suffices.

Multi-backend
-------------
Compute is expressed in ``bm``; the sub-matrix extraction goes through
``scipy.sparse`` at the boundary because fealpy ``CSRTensor``'s
fancy-index semantics are not part of the multi-backend surface. See
``docs/architecture/multibackend_convention.md``.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np  # boundary-only per multi-backend convention §3
from fealpy.backend import backend_manager as bm

from fracturex.utilfuc.linear_solvers import as_scipy_csr


def build_nepin_callbacks(
    sys_d: Any,
    d_old: Any,
) -> Tuple[Callable[[Any, Any], Any], Callable[[Any, Any, Any], Any]]:
    """Return ``(residual, jacobian)`` callbacks bound to ``sys_d``.

    Parameters
    ----------
    sys_d : PhaseFieldSystem
        Result of ``phase_assembler.assemble(load)``. Only ``sys_d.A`` and
        ``sys_d.F`` are read; the object is not mutated.
    d_old : Any
        The damage iterate that produced ``sys_d`` (bm array or numpy
        array of shape ``(n_d,)``).

    Returns
    -------
    residual : callable ``(d, x_frozen) -> R``
        ``R = A @ d - rhs`` as a bm array of shape ``(n_d,)``.
        ``x_frozen`` is accepted and ignored (the freezing is baked into
        the closure).
    jacobian : callable ``(d, x_frozen, subset_dofs) -> J_SS``
        Dense ``bm`` array of shape ``(|S|, |S|)`` -- the sub-block
        ``A[S][:, S]``.
    """
    A_csr = as_scipy_csr(sys_d.A)
    d_old_np = np.asarray(bm.to_numpy(d_old), dtype=np.float64).reshape(-1)
    F_np = np.asarray(bm.to_numpy(sys_d.F), dtype=np.float64).reshape(-1)
    rhs_np = F_np + (A_csr @ d_old_np)

    def residual(d: Any, _x_frozen: Any) -> Any:  # noqa: ARG001
        d_np = np.asarray(bm.to_numpy(d), dtype=np.float64).reshape(-1)
        R_np = A_csr @ d_np - rhs_np
        return bm.asarray(R_np, dtype=bm.float64)

    def jacobian(_d: Any, _x_frozen: Any, subset_dofs: Any) -> Any:  # noqa: ARG001
        idx = np.asarray(bm.to_numpy(subset_dofs), dtype=np.int64).reshape(-1)
        J_SS_np = A_csr[idx][:, idx].toarray().astype(np.float64)
        return bm.asarray(J_SS_np, dtype=bm.float64)

    return residual, jacobian


__all__ = ["build_nepin_callbacks"]
