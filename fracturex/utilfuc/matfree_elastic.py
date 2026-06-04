"""Matrix-free Hu-Zhang elastic saddle-point operator.

Applies the action of the `standard`-formulation elastic system

    A = [[ M2,  B2 ],
         [ B2', 0  ]],   M2 = TM' M(d) TM,   B2 = TM' B

WITHOUT ever materializing the large stress mass block ``M2`` (or the
``(ldof,ldof)`` element kernel ``Phi``). Only ``M2``'s action is matrix-free;
the small, d-independent pieces ``B2`` and ``TM`` stay assembled (scipy CSR),
because the aux-space preconditioner still needs ``B2`` and the Schur block
explicitly. The damage coefficient ``coef_d = 1/g(d)`` is frozen for the
lifetime of one operator (one staggered elastic solve), matching the assembled
path where the assembler rebuilds ``A`` every staggered iteration.

Memory win: the dominant p=3 stress block is replaced by the much smaller basis
arrays ``phi (NC,NQ,ldof,nsym)`` + ``trphi (NC,NQ,ldof)`` (~order of magnitude
below ``M2``/``Phi``). On CPU the per-matvec element contraction makes the solve
slower; the speed payoff comes on GPU (future ④). The element math is the same
contraction ``HuZhangStressIntegrator`` performs, just contracted against the
input vector instead of assembled.

GPU-port seam: the element einsums below are written against numpy for CPU
development; porting to GPU replaces the numpy kernel arrays + scipy TM/B2
matvecs with ``bm``/``torch.sparse`` equivalents (the contraction structure is
unchanged).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator


class MatrixFreeElasticOperator(LinearOperator):
    """LinearOperator for the Hu-Zhang elastic saddle system (standard form).

    Drop-in for the assembled scipy ``A`` on the ITERATIVE paths only
    (``solve_huzhang_block_gmres_fast``/``_auxspace``): GMRES needs only the
    matvec, and the solver pulls the divergence block ``B`` and the (approximate)
    stress-diagonal from this object instead of slicing/``.diagonal()`` a sparse
    matrix.

    Constructor inputs:
        gdof_sigma: int, size of the (transformed) stress block M2.
        gdof_u: int, size of the displacement block.
        TM, TMT: scipy CSR, corner-relaxation transform and its transpose
            (shape (gdof_sigma, gdof_sigma); d-independent).
        B2: scipy CSR, coupling block, shape (gdof_sigma, gdof_u). The solver's
            divergence block is ``B_div = B2.T`` (shape (gdof_u, gdof_sigma)).
        kernel: dict with numpy arrays
            phi   (NC, NQ, ldof, nsym)  stress basis at quad points,
            trphi (NC, NQ, ldof)        basis trace,
            W     (NC, NQ)              cell-measure * quad-weight,
            num   (nsym,)              symmetry multiplicities,
            cell2dof (NC, ldof) int64  local->global stress dof map,
            coef  (NC, NQ)             frozen 1/g(d) at quad points,
            c0, c1: float compliance constants.
        isbd_global: optional bool array (len gdof_sigma+gdof_u) marking essential
            sigma dofs; when given the operator emulates ``T A T + Tbd`` so GMRES
            solves the same masked system as the assembled path.

    Exposes ``.diagonal()``, ``.gdof_sigma``, ``.B_div`` / ``.B_div_T`` and
    ``.diag_inv_sigma()`` so the solver seam can read what it needs.
    """

    def __init__(self, *, gdof_sigma, gdof_u, TM, TMT, B2, kernel,
                 isbd_global: Optional[np.ndarray] = None,
                 recompute: bool = False, chunk: int = 8192):
        self.gdof_sigma = int(gdof_sigma)
        self.gdof_u = int(gdof_u)
        n = self.gdof_sigma + self.gdof_u
        super().__init__(dtype=np.float64, shape=(n, n))

        self._TM = TM
        self._TMT = TMT
        self._B2 = B2                      # (gdof_sigma, gdof_u)
        self.B_div = (B2.T).tocsr()        # (gdof_u, gdof_sigma) == A_[m:, :m]
        self.B_div_T = B2.tocsr()          # (gdof_sigma, gdof_u) == B_div.T

        # frozen element kernel (numpy). Always small: W/coef (NC,NQ), cell2dof (NC,ldof).
        self._W = np.ascontiguousarray(kernel["W"], dtype=np.float64)
        self._num = np.ascontiguousarray(kernel["num"], dtype=np.float64)
        self._coef = np.ascontiguousarray(kernel["coef"], dtype=np.float64)
        self._cell2dof = np.ascontiguousarray(kernel["cell2dof"], dtype=np.int64)
        self._c0 = float(kernel["c0"])
        self._c1 = float(kernel["c1"])
        self._cell2dof_flat = self._cell2dof.reshape(-1)
        self._NC = int(self._cell2dof.shape[0])

        # Two memory modes for the basis arrays phi/trphi (NC,NQ,ldof,*):
        #   recompute=False (cache): store phi/trphi once. Avoids the (ldof,ldof) Phi
        #     kernel + M2 of the assembled path, but phi itself is sizable at p=3.
        #   recompute=True (chunked): store nothing; recompute basis per cell-chunk
        #     inside each matvec via space0.basis(bcs, index=chunk). The phi array only
        #     exists transiently for `chunk` cells -> peak memory bounded well below M2.
        #     Slower on CPU (basis recomputed every Krylov iter); the speed returns on GPU.
        self._recompute = bool(recompute)
        self._chunk = int(chunk)
        if self._recompute:
            self._space0 = kernel["space0"]
            self._bcs = kernel["bcs"]
            self._phi = None
            self._trphi = None
        else:
            self._space0 = None
            self._bcs = None
            self._phi = np.ascontiguousarray(kernel["phi"], dtype=np.float64)
            self._trphi = np.ascontiguousarray(kernel["trphi"], dtype=np.float64)

        # essential-sigma mask (model0 path): emulate T A T + Tbd
        if isbd_global is not None:
            self._isbd = np.asarray(isbd_global, dtype=bool).reshape(-1)
            assert self._isbd.shape[0] == n
        else:
            self._isbd = None

        # approximate diag(M2) for the preconditioner D_inv (diag-only; affects
        # niter, never the solution). diag(M) computed exactly element-wise, then
        # mapped through the elementwise-squared transform: diag(M2) ~= (TM o TM)' diag(M).
        diagM = self._element_diag_M()                       # (gdof_sigma,)
        TM2 = TM.copy()
        TM2.data = TM2.data ** 2
        diagM2 = np.asarray(TM2.T @ diagM, dtype=np.float64)  # (gdof_sigma,)
        self._diag_unmasked = np.concatenate(
            [diagM2, np.zeros(self.gdof_u, dtype=np.float64)]
        )
        self._diag = self._diag_unmasked.copy()
        if self._isbd is not None:
            self._diag[self._isbd] = 1.0                      # Tbd rows -> 1

    def set_essential_mask(self, isbd_global: np.ndarray) -> "MatrixFreeElasticOperator":
        """Attach a sigma-essential mask after construction (emulates T A T + Tbd).

        Used by the assembler when ``case.neumann_data`` triggers essential-sigma
        elimination. Updates the masked diagonal (essential rows -> 1). Returns self.
        """
        self._isbd = np.asarray(isbd_global, dtype=bool).reshape(-1)
        assert self._isbd.shape[0] == self.shape[0]
        self._diag = self._diag_unmasked.copy()
        self._diag[self._isbd] = 1.0
        return self

    # ---- per-chunk basis provider (cache: slice stored; recompute: re-evaluate) ----
    def _phi_trphi(self, c0: int, c1: int):
        """Return (phi, trphi) for cells [c0,c1): phi (nc,NQ,ldof,nsym), trphi (nc,NQ,ldof).

        In recompute mode the basis is evaluated only for this chunk, so it exists
        transiently and never for all cells at once (bounded peak memory).
        """
        if not self._recompute:
            return self._phi[c0:c1], self._trphi[c0:c1]
        from fealpy.backend import backend_manager as bm
        idx = bm.asarray(np.arange(c0, c1, dtype=np.int64))
        phi = np.ascontiguousarray(bm.to_numpy(self._space0.basis(self._bcs, index=idx)),
                                   dtype=np.float64)
        trphi = phi[..., 0] + phi[..., -1]   # 2D trace (xx + yy)
        return phi, trphi

    def _chunks(self):
        cs = self._chunk if self._recompute else self._NC
        cs = max(1, int(cs))
        for c0 in range(0, self._NC, cs):
            yield c0, min(c0 + cs, self._NC)

    # ---- element-free M(d) action (no Phi, no M2) ----
    def _M_action(self, xt: np.ndarray) -> np.ndarray:
        """Apply untransformed M(d) to a stress-space vector ``xt`` (len gdof_sigma).

        Per cell: y_l = sum_q W coef ( c0 sum_d num_d phi_lд (phi_:д . x_l)
                                       - c1 trphi_l (trphi_: . x_l) ), scattered over
        cell2dof. Processed in cell-chunks so recompute mode bounds peak memory.
        """
        y = np.zeros(self.gdof_sigma, dtype=np.float64)
        for a, b in self._chunks():
            phi, trphi = self._phi_trphi(a, b)
            c2d = self._cell2dof[a:b]
            x_local = xt[c2d]                                         # (nc, ldof)
            s_d = np.einsum("cqmd,cm->cqd", phi, x_local)             # project on components
            t = np.einsum("cqm,cm->cq", trphi, x_local)              # trace projection
            wcoef = self._W[a:b] * self._coef[a:b]                    # (nc, NQ)
            term1 = self._c0 * np.einsum("cq,d,cqld,cqd->cl", wcoef, self._num, phi, s_d)
            term2 = self._c1 * np.einsum("cq,cql,cq->cl", wcoef, trphi, t)
            y_local = term1 - term2                                   # (nc, ldof)
            np.add.at(y, c2d.reshape(-1), y_local.reshape(-1))
        return y

    def _element_diag_M(self) -> np.ndarray:
        """Exact diag(M) element-wise: Ke[c,l,l] scattered over cell2dof (chunked)."""
        diag = np.zeros(self.gdof_sigma, dtype=np.float64)
        for a, b in self._chunks():
            phi, trphi = self._phi_trphi(a, b)
            wcoef = self._W[a:b] * self._coef[a:b]                   # (nc, NQ)
            d1 = self._c0 * np.einsum("cq,d,cqld->cl", wcoef, self._num, phi ** 2)
            d2 = self._c1 * np.einsum("cq,cql->cl", wcoef, trphi ** 2)
            d_local = d1 - d2                                        # (nc, ldof)
            np.add.at(diag, self._cell2dof[a:b].reshape(-1), d_local.reshape(-1))
        return diag

    # ---- unmasked saddle action ----
    def _raw_matvec(self, z: np.ndarray) -> np.ndarray:
        m = self.gdof_sigma
        x_sigma = z[:m]
        x_u = z[m:]
        yt = self._M_action(self._TM @ x_sigma)            # M (TM x_sigma)
        y_sigma = (self._TMT @ yt) + (self._B2 @ x_u)       # TM' M TM x + B2 x_u
        y_u = self.B_div @ x_sigma                          # B2' x_sigma
        return np.concatenate([y_sigma, y_u])

    def _matvec(self, z):
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if self._isbd is None:
            return self._raw_matvec(z)
        # masked: T (A (T z)) + Tbd z  ==  zero essential in/out, then copy z on essential
        zt = z.copy()
        zt[self._isbd] = 0.0
        y = self._raw_matvec(zt)
        y[self._isbd] = z[self._isbd]
        return y

    # ---- solver seam helpers ----
    def diagonal(self):
        """Full diagonal (approx on sigma block, masked essential->1, zeros on u)."""
        return self._diag

    def diag_inv_sigma(self, floor: float = 1e-30) -> np.ndarray:
        """1/diag of the stress block (for preconditioner D_inv); floored."""
        d = self._diag[: self.gdof_sigma]
        d = np.where(np.abs(d) > floor, d, 1.0)
        return 1.0 / d

    def apply_to(self, v: np.ndarray) -> np.ndarray:
        """Public masked matvec."""
        return self._matvec(v)

    def apply_unmasked(self, v: np.ndarray) -> np.ndarray:
        """Public UNMASKED saddle action ``A @ v`` (raw bmat, before T A T + Tbd).

        The assembler's essential-BC step computes ``F = F - A @ uh_global`` with
        the UNMASKED A (matching `apply_sigma_essential_to_system`), so use this.
        """
        return self._raw_matvec(np.asarray(v, dtype=np.float64).reshape(-1))
