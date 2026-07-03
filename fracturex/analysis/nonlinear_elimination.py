"""Damage-only NEPIN nonlinear-elimination preconditioner.

Given a phase-field iterate ``d`` with a strongly-nonlinear region
``Omega_s = {d > d_c}`` (typically the fully-damaged crack band),
NEPIN performs a local Newton solve on the damage sub-problem
restricted to ``Omega_s`` and returns the corrected damage. The outer
staggered Newton then sees a residual whose affine-covariant Lipschitz
constant is contracted by roughly ``L_{S^c} / L_S`` (the ratio of the
elastic bulk's Lipschitz constant to the fully-damaged region's).

Theory
------
See ``docs/preconditioner/THEORY_nonlinear_elimination.md``.
Primary reference: Cai and Keyes, *Nonlinearly preconditioned inexact
Newton algorithms*, SIAM J. Sci. Comput. 24 (2002) 183-200.

Design
------
See ``docs/preconditioner/DESIGN_nepin_driver.md``.

Multi-backend
-------------
Compute uses ``fealpy.backend.backend_manager`` (``bm``); numpy and
scipy live at the linear-solver / I/O boundary only. See
``docs/architecture/multibackend_convention.md``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np  # boundary-only per multi-backend convention §3
from fealpy.backend import backend_manager as bm


# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NEPINConfig:
    """Configuration for one NEPIN elimination call.

    Parameters
    ----------
    d_c : float
        Threshold defining Omega_s = {d > d_c}. Default 0.82 matches the
        paper_aux localization hard-wall reported in
        ``docs/preconditioner/PIPELINE_STATUS.md``.
    max_local_iter : int
        Cap on inner Newton iterations for the local damage solve.
        Cai-Keyes 2002 report convergence in 1-3 iters; 5 is a safe cap.
    local_tol : float
        Inner stopping criterion:
        ``||F_S(y_final)|| <= local_tol * ||F_S(0)||``.
        1e-2 is Cai-Keyes' recommendation (deliberately loose; the outer
        Newton absorbs the inexactness).
    damping : float
        Newton step damping in ``(0, 1]``; 1.0 means full step. On any
        iteration that would increase ``||F_S||``, we halve the effective
        damping down to a floor of 1e-4 before giving up.
    """
    d_c: float = 0.82
    max_local_iter: int = 5
    local_tol: float = 1e-2
    damping: float = 1.0


@dataclass(frozen=True)
class NEPINResult:
    """Report of one NEPIN elimination call.

    Fields
    ------
    d_corrected : Any
        ``bm`` array of shape ``(n_d,)`` -- the corrected damage after
        the local Newton, extended by zero on ``S^c``.
    subset_size : int
        ``|S|`` -- number of eliminated dofs.
    local_iters : int
        Actual inner Newton iterations used.
    local_res_reduction : float
        ``||F_S(y_final)|| / ||F_S(0)||``. 1.0 if ``|S|=0``.
    converged : bool
        True iff the reduction met ``local_tol``.
    wall_time : float
        Seconds spent inside :meth:`NEPINEliminator.eliminate`.
    """
    d_corrected: Any
    subset_size: int
    local_iters: int
    local_res_reduction: float
    converged: bool
    wall_time: float


# ---------------------------------------------------------------------------
# Subset identification
# ---------------------------------------------------------------------------

def identify_subset(d_full: Any, *, d_c: float) -> Any:
    """Return a ``bm`` bool array of shape ``(n_d,)`` marking dofs where
    ``d > d_c``.

    This is the strict node-wise (mesh-free) indicator. The
    element-adjacency ("include_interface") indicator that grows the
    subset by one layer of neighbouring elements requires mesh
    connectivity and is applied by the caller (spike script) before
    passing the resulting mask to :meth:`NEPINEliminator.eliminate`
    via its ``subset_mask`` argument.
    """
    d_arr = bm.asarray(d_full, dtype=bm.float64)
    threshold = bm.asarray(float(d_c), dtype=bm.float64)
    return d_arr > threshold


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class NEPINEliminator:
    """Damage-only NEPIN nonlinear-elimination preconditioner.

    Callbacks (both are user-supplied, backend-neutral)
    ---------------------------------------------------
    residual : callable ``F(d_full, x_frozen) -> R_d``
        Given a full damage vector ``d_full`` (bm array shape ``(n_d,)``)
        and an opaque ``x_frozen`` bag (containing anything the residual
        needs that is not perturbed by the local solve -- typically u,
        history H, load, mesh), returns the full damage residual as a
        bm array of shape ``(n_d,)``.
    jacobian : callable ``J(d_full, x_frozen, subset_dofs) -> J_SS``
        Given the additional integer index array ``subset_dofs`` of shape
        ``(|S|,)``, returns the dense sub-Jacobian ``J_SS`` of shape
        ``(|S|, |S|)`` as a bm array. For ``|S| > few * 1e4``, replace
        with a sparse variant at the caller (out of scope here).

    The two callbacks decouple this module from the fracturex assembly:
    the caller (a driver hook or the spike script) is responsible for
    building them by binding a ``phase_assembler``, ``u``, ``H`` etc.
    """

    def __init__(
        self,
        residual: Callable[[Any, Any], Any],
        jacobian: Callable[[Any, Any, Any], Any],
        config: Optional[NEPINConfig] = None,
    ):
        self._F = residual
        self._J = jacobian
        self.config = config or NEPINConfig()

    # -- public API --------------------------------------------------------

    def eliminate(
        self,
        d_full: Any,
        x_frozen: Any,
        *,
        subset_mask: Optional[Any] = None,
    ) -> NEPINResult:
        """Perform one local Newton on Omega_s and return the corrected
        damage as a ``NEPINResult``.

        Parameters
        ----------
        d_full : Any
            bm array of shape ``(n_d,)`` -- current damage iterate.
        x_frozen : Any
            Opaque bag passed unchanged to the residual/jacobian
            callbacks.
        subset_mask : Optional[Any]
            bm bool array of shape ``(n_d,)``. If ``None``, uses
            :func:`identify_subset` on ``d_full`` with ``config.d_c``.

        Returns
        -------
        NEPINResult
        """
        t0 = time.perf_counter()
        d_full = bm.asarray(d_full, dtype=bm.float64)
        n_d = int(d_full.shape[0])

        if subset_mask is None:
            subset_mask = identify_subset(d_full, d_c=self.config.d_c)
        else:
            subset_mask = bm.asarray(subset_mask)

        # Convert mask -> integer index array.
        subset_dofs = self._mask_to_indices(subset_mask, n_d)
        subset_size = int(subset_dofs.shape[0])

        if subset_size == 0:
            # Nothing to eliminate; return unmodified iterate.
            return NEPINResult(
                d_corrected=d_full,
                subset_size=0,
                local_iters=0,
                local_res_reduction=1.0,
                converged=True,
                wall_time=time.perf_counter() - t0,
            )

        # F0_full = F(d_full, x_frozen); we need F0 = F0_full[S].
        F_full = bm.asarray(self._F(d_full, x_frozen), dtype=bm.float64)
        F0 = F_full[subset_dofs]  # bm read via fancy int index
        norm_F0 = float(bm.linalg.norm(F0))

        if norm_F0 == 0.0:
            return NEPINResult(
                d_corrected=d_full,
                subset_size=subset_size,
                local_iters=0,
                local_res_reduction=1.0,
                converged=True,
                wall_time=time.perf_counter() - t0,
            )

        # Local Newton loop on d[S].
        d_curr = bm.asarray(d_full, dtype=bm.float64)  # will be updated
        F_curr = F_full
        F_S_curr = F0
        norm_curr = norm_F0
        cfg = self.config
        target = cfg.local_tol * norm_F0

        local_iters = 0
        converged = False
        for m in range(cfg.max_local_iter):
            local_iters = m + 1

            J_SS = self._J(d_curr, x_frozen, subset_dofs)
            step = self._local_solve(F_S_curr, J_SS)  # step = -J_SS^{-1} F_S

            # Damped Newton: try full step first, then halve on backtrack.
            damping = cfg.damping
            accepted = False
            while damping >= 1e-4:
                d_trial = self._scatter_add(d_curr, subset_dofs, damping * step)
                F_trial_full = bm.asarray(
                    self._F(d_trial, x_frozen), dtype=bm.float64
                )
                F_S_trial = F_trial_full[subset_dofs]
                norm_trial = float(bm.linalg.norm(F_S_trial))
                if norm_trial < norm_curr:
                    accepted = True
                    d_curr = d_trial
                    F_curr = F_trial_full
                    F_S_curr = F_S_trial
                    norm_curr = norm_trial
                    break
                damping *= 0.5

            if not accepted:
                # Backtrack failed -- report honest non-convergence.
                break

            if norm_curr <= target:
                converged = True
                break

        return NEPINResult(
            d_corrected=d_curr,
            subset_size=subset_size,
            local_iters=local_iters,
            local_res_reduction=(norm_curr / norm_F0),
            converged=converged,
            wall_time=time.perf_counter() - t0,
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _mask_to_indices(mask: Any, n: int) -> Any:
        """Convert a bm bool mask of length n to an integer index array.

        Uses ``bm.arange(n)`` + fancy read; avoids ``np.nonzero`` at the
        boundary so we stay backend-neutral.
        """
        idx_all = bm.arange(n)
        return idx_all[mask]

    @staticmethod
    def _scatter_add(
        base: Any, indices: Any, values: Any,
    ) -> Any:
        """Return a copy of ``base`` with ``values`` added at ``indices``.

        Uses ``bm.index_add`` per multi-backend convention §2 (avoids
        ``np.add.at``). ``base`` is assumed to be a bm array; the
        returned tensor is the same shape.
        """
        return bm.index_add(base, indices, values)

    @staticmethod
    def _local_solve(F_S: Any, J_SS: Any) -> Any:
        """Solve ``J_SS @ step = -F_S`` via dense LU at the numpy/scipy
        boundary. Returns ``step`` as a bm array.

        For sizes up to ~ 1e4 this is well below the global GMRES cost;
        larger subsets need a sparse variant (out of scope for the
        first-cut damage-only NEPIN).
        """
        try:
            from scipy.linalg import lu_factor, lu_solve
        except Exception as exc:  # pragma: no cover -- scipy always present in prod
            raise RuntimeError(
                "NEPIN local solver requires scipy.linalg.lu_factor; "
                "no scipy in this environment."
            ) from exc

        F_np = bm.to_numpy(F_S).astype(np.float64)
        J_np = bm.to_numpy(J_SS).astype(np.float64)
        lu, piv = lu_factor(J_np, check_finite=False)
        step_np = lu_solve((lu, piv), -F_np, check_finite=False)
        return bm.asarray(step_np, dtype=bm.float64)


__all__ = [
    "NEPINConfig",
    "NEPINResult",
    "NEPINEliminator",
    "identify_subset",
]
