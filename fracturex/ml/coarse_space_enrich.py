"""SPD-safe Galerkin enrichment of the auxiliary P1 coarse space (D13 L2).

The geometric P1 coarse correction (``e += PI_s M_c^{-1} PI_s^T r`` per displacement
component) cannot represent the sharp interface jump modes that appear when a crack
localizes; this caps the two-level condition number (plan §4.0 obstruction). This
module adds a SECOND, additive coarse correction on a small learned/synthetic set of
interface enrichment modes ``Phi`` via a Galerkin (congruence) projection that is
**SPD-safe for any Phi** — see :class:`EnrichmentOperator`.

Math (single displacement component, Schur block ``S_b`` SPD, residual ``r``):

    W      := PI_s @ Phi              # (sgdof, k), enrichment modes prolonged to the
                                      #   displacement component space
    S_Phi  := W^T S_b W              # (k, k), Galerkin coarse matrix; >= 0 by
                                      #   congruence, SPD when W has full column rank
    e      += W @ pinv(S_Phi) @ W^T @ r

``pinv`` is a regularized symmetric pseudo-inverse (``pinvh`` + eps), so correlated /
rank-deficient columns never break well-definedness. Adding this positive-semidefinite
correction on top of the geometric P1 one keeps the block preconditioner nonsingular
and leaves the right-preconditioned GMRES solution set unchanged (plan command 4
generalized): correctness is independent of Phi, the worst case is "Phi useless".

Backend policy (docs/architecture/multibackend_convention.md): the solver passes a
scipy CSR ``PI_s`` and a numpy/scipy Schur matvec (the solver boundary is numpy/scipy,
exempt). Feature-side inputs are bm tensors; we convert to numpy at this boundary.
This module imports NO solver and NO torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np  # solver-side boundary is numpy/scipy (exempt); see module docstring.
import scipy.linalg as sla
from scipy.sparse import csr_matrix

from fealpy.backend import backend_manager as bm


def _to_numpy_2d(arr) -> np.ndarray:
    """Coerce a bm/numpy array to a contiguous float64 numpy 2D array (N, k)."""
    a = np.asarray(bm.to_numpy(arr) if not isinstance(arr, np.ndarray) else arr,
                   dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return np.ascontiguousarray(a)


@dataclass
class EnrichmentOperator:
    """Per-component DEFLATION enrichment, precomputed once per setup.

    Why deflation (not additive Galerkin): a separate additive correction
    ``e += W S_Phi^+ W^T r`` alongside the geometric coarse term double-counts
    overlapping modes and does NOT monotonically reduce the two-level condition number
    (verified spectrally; see D13_IMPL §2.2 / §6.2 -- additive made kappa WORSE). The
    deflation/projection form removes the targeted (interface) subspace from the base
    preconditioner and solves it exactly, which DOES reduce kappa and -- crucially --
    kills the high-contrast dependence (kappa stays bounded as contrast 1e3 -> 1e5),
    i.e. plan command 6. It wraps ANY base Schur solve (GS / coarse V-cycle).

    For deflation subspace ``W`` (prolonged modes) and Schur block ``S_b``:

        Q  = W (W^T S_b W)^+ W^T            (S_b-orthogonal coarse solve onto span W)
        M_def^{-1} r = Q r + (I - Q S_b) base^{-1} (I - S_b Q) r

    ``M_def^{-1}`` is SPD when ``base^{-1}`` is SPD and ``W`` has full rank; the
    pseudo-inverse + ridge keeps it well-defined for rank-deficient ``W``. Right
    preconditioning => the GMRES solution is unchanged (plan command 4).

    Attributes:
        W: prolonged enrichment basis ``(sgdof, k)`` (numpy), shared by all components.
        SW_per_comp: list (len gdim) of ``S_b^{(i)} W`` ``(sgdof, k)`` (for ``I - S_b Q``).
        WtSWi_per_comp: list (len gdim) of ``(k, k)`` regularized ``(W^T S_b W)^+``.
        gdim, sgdof, k: dimensions.
    """

    W: np.ndarray
    SW_per_comp: list
    WtSWi_per_comp: list
    gdim: int
    sgdof: int
    k: int

    @classmethod
    def from_modes(
        cls,
        phi_modes,
        PI_s: csr_matrix,
        schur_block_apply: Callable[[np.ndarray, int], np.ndarray],
        *,
        gdim: int,
        sgdof: int,
        reg: float = 1e-12,
    ) -> Optional["EnrichmentOperator"]:
        """Precompute the per-component deflation operators from coarse-space modes.

        Args:
            phi_modes: ``(NN, k)`` enrichment columns in the P1 coarse space (bm/numpy).
                ``NN == PI_s.shape[1]``. ``k == 0`` (or all-zero) -> returns None (no-op).
            PI_s: scipy CSR ``(sgdof, NN)`` P1 prolongation (one component).
            schur_block_apply: ``(v, comp) -> S_b^{(comp)} @ v`` Schur block matvec.
            gdim: number of displacement components.
            sgdof: per-component dof count (== PI_s.shape[0]).
            reg: relative ridge added before the symmetric pseudo-inverse.
        Returns:
            an :class:`EnrichmentOperator`, or ``None`` when there is nothing to add.
        """
        Phi = _to_numpy_2d(phi_modes)
        k = int(Phi.shape[1])
        if k == 0 or not np.any(Phi):
            return None
        nrows = Phi.shape[0]
        if nrows == PI_s.shape[1]:
            # coarse-space modes (NN): prolong into the displacement component space.
            W = np.ascontiguousarray((PI_s @ Phi))  # (sgdof, k)
        elif nrows == PI_s.shape[0]:
            # fine-space modes (sgdof): use directly -- these are the genuine out-of-V_H
            # GenEO worst modes (top_k_worst_modes); prolonging would be wrong.
            W = np.ascontiguousarray(Phi)
        else:
            raise ValueError(
                f"phi_modes has {nrows} rows but expects either NN={PI_s.shape[1]} "
                f"(coarse, prolonged) or sgdof={PI_s.shape[0]} (fine, direct)."
            )

        SW_list, inv_list = [], []
        for comp in range(gdim):
            SW = np.empty_like(W)
            for j in range(k):
                SW[:, j] = np.asarray(
                    schur_block_apply(np.ascontiguousarray(W[:, j]), comp)
                ).reshape(-1)
            S_phi = W.T @ SW  # (k, k) == W^T S_b W, symmetric PSD
            S_phi = 0.5 * (S_phi + S_phi.T)
            scale = float(np.trace(S_phi)) / max(k, 1)
            ridge = reg * (scale if scale > 0 else 1.0)
            S_phi = S_phi + ridge * np.eye(k)
            SW_list.append(SW)
            inv_list.append(sla.pinvh(S_phi))
        return cls(W=W, SW_per_comp=SW_list, WtSWi_per_comp=inv_list,
                   gdim=gdim, sgdof=sgdof, k=k)

    def apply_deflated(self, r_block, base_solve, comp: int) -> np.ndarray:
        """Deflated Schur solve for one component: ``M_def^{-1} r``.

        ``base_solve(rhs) -> e`` is the existing (geometric) Schur preconditioner action
        for this component (GS sweeps + coarse V-cycle). Deflation wraps it:

            coarse = Q r ;  e = coarse + (I - Q S_b) base_solve((I - S_b Q) r)

        with ``Q = W (W^T S_b W)^+ W^T`` and ``S_b Q`` available via the cached ``SW``.

        Args:
            r_block: ``(sgdof,)`` residual for component ``comp``.
            base_solve: ``callable(rhs) -> e`` the base Schur preconditioner action.
            comp: displacement component index.
        Returns:
            ``(sgdof,)`` deflated correction (numpy).
        """
        r = np.asarray(r_block, dtype=np.float64).reshape(-1)
        W = self.W
        SW = self.SW_per_comp[comp]
        WtSWi = self.WtSWi_per_comp[comp]
        # Q r = W (W^T S_b W)^+ (W^T r)
        coarse = W @ (WtSWi @ (W.T @ r))
        # (I - S_b Q) r = r - SW (W^T S_b W)^+ (W^T r)  [reuses coarse coeff]
        rhs = r - SW @ (WtSWi @ (W.T @ r))
        y = np.asarray(base_solve(rhs), dtype=np.float64).reshape(-1)
        # (I - Q S_b) y = y - W (W^T S_b W)^+ (W^T S_b y) = y - W (W^T S_b W)^+ (SW^T y)
        y = y - W @ (WtSWi @ (SW.T @ y))
        return coarse + y

    def apply_deflated_full(self, r_full, base_solve_full, sgdof: int) -> np.ndarray:
        """Block-wise deflation around a FULL stacked-displacement Schur solve.

        The solver's ``pre_of_S`` acts on the full displacement vector (all components
        stacked, ``nu = gdim*sgdof``); the geometric coarse correction it contains is
        already block-diagonal per component. This wraps it with the same block-diagonal
        deflation: pre-project the rhs, run the base solve once, post-project, add the
        coarse-space part.

            rhs[i]   = r[i] - SW_i (W^T S_b W)^+ (W^T r[i])      # (I - S_b Q) r, block i
            y        = base_solve_full(rhs)                       # one full Schur solve
            y[i]     = y[i] - W (W^T S_b W)^+ (SW_i^T y[i])       # (I - Q S_b) y, block i
            return  Q r + y                                       # coarse + deflated base

        Args:
            r_full: ``(nu,)`` full stacked residual.
            base_solve_full: ``callable(rhs_full) -> e_full`` the base Schur solver.
            sgdof: per-component dof count.
        Returns:
            ``(nu,)`` deflated correction (numpy).
        """
        r = np.asarray(r_full, dtype=np.float64).reshape(-1)
        W = self.W
        rhs = r.copy()
        coarse = np.zeros_like(r)
        for i in range(self.gdim):
            i0, i1 = i * sgdof, (i + 1) * sgdof
            ri = r[i0:i1]
            c = self.WtSWi_per_comp[i] @ (W.T @ ri)   # (k,)
            coarse[i0:i1] = W @ c
            rhs[i0:i1] = ri - self.SW_per_comp[i] @ c
        y = np.asarray(base_solve_full(rhs), dtype=np.float64).reshape(-1)
        for i in range(self.gdim):
            i0, i1 = i * sgdof, (i + 1) * sgdof
            yi = y[i0:i1]
            y[i0:i1] = yi - W @ (self.WtSWi_per_comp[i] @ (self.SW_per_comp[i].T @ yi))
        return coarse + y


def build_jump_template_modes(
    features,
    *,
    grad_threshold: float = 0.1,
    feature_index_gradd: int = 1,
    feature_index_d: int = 0,
):
    """Synthetic interface jump modes (no learning) for L2-alpha mechanism tests.

    Builds a single coarse-space column that is supported on the localized crack band
    (nodes with large dimensionless damage gradient ``gradd_l0``) and signed by which
    side of the interface a node sits on (via ``d`` above/below the band median). This
    is the fixed template later modulated by a learned per-node amplitude (L2-beta);
    here it lets us exercise the Galerkin seam with a physically-motivated Phi.

    Args:
        features: ``(NN, n_feat)`` per-node feature block (bm/numpy), columns per
            ``coarse_features.FEATURE_NAMES``.
        grad_threshold: nodes with ``gradd_l0 > grad_threshold`` form the interface band.
        feature_index_gradd, feature_index_d: column indices of the gradient / damage
            features.
    Returns:
        ``(NN, 1)`` numpy float64 template column (zero off the band). All-zero if no
        node exceeds the threshold (caller treats as no-op).
    """
    phi = _to_numpy_2d(features)
    gradd = phi[:, feature_index_gradd]
    d = phi[:, feature_index_d]
    band = gradd > grad_threshold
    col = np.zeros(phi.shape[0], dtype=np.float64)
    if not np.any(band):
        return col.reshape(-1, 1)
    # sign by side of the interface; amplitude by gradient strength (normalized).
    d_mid = float(np.median(d[band]))
    sign = np.where(d >= d_mid, 1.0, -1.0)
    gmax = float(np.max(gradd[band]))
    amp = np.where(band, gradd / (gmax if gmax > 0 else 1.0), 0.0)
    col = sign * amp
    return col.reshape(-1, 1)


def scale_modes(template, amplitude) -> np.ndarray:
    """Per-node amplitude modulation of a fixed template: ``Phi = template * amp``.

    Args:
        template: ``(NN, k)`` fixed template columns (bm/numpy).
        amplitude: ``(NN,)`` or ``(NN, 1)`` per-node scalar (bm/numpy; e.g. learned).
    Returns:
        ``(NN, k)`` numpy modes.
    """
    T = _to_numpy_2d(template)
    a = np.asarray(bm.to_numpy(amplitude) if not isinstance(amplitude, np.ndarray)
                   else amplitude, dtype=np.float64).reshape(-1, 1)
    return T * a
