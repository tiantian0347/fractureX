# fracturex/damage/phasefield_damage.py

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Any, Optional, Tuple

import numpy as np
from fealpy.backend import backend_manager as bm

from fracturex.damage.base import DamageModelBase
from fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction
from fracturex.phasefield.crack_surface_density_function import CrackSurfaceDensityFunction


def _history_update_nproc_default() -> int:
    return max(1, int(os.cpu_count() or 1))


def _phase_history_chunk_worker(args):
    damage, state, bcs, idx_chunk = args
    idx_chunk = bm.asarray(idx_chunk)
    grad_u = state.u.grad_value(bcs, index=idx_chunk)
    strain = 0.5 * (grad_u + bm.swapaxes(grad_u, -2, -1))
    if damage.split in ("hybrid", "spectral"):
        phip = damage._spectral_positive_energy_density(strain)
    elif damage.split == "isotropic":
        phip = damage._isotropic_energy_density(strain)
        phip = bm.maximum(phip, 0.0)
    else:
        raise ValueError(f"Unknown split type: {damage.split}")
    return bm.maximum(phip, 0.0)


def _material_lame_from_model(model) -> Tuple[float, float]:
    if hasattr(model, "lam") and hasattr(model, "mu"):
        return float(model.lam), float(model.mu)
    if hasattr(model, "lambda0") and hasattr(model, "lambda1"):
        return float(model.lambda0), float(model.lambda1)
    if hasattr(model, "E") and hasattr(model, "nu"):
        E = float(model.E)
        nu = float(model.nu)
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lam, mu
    if isinstance(model, dict):
        if "lam" in model and "mu" in model:
            return float(model["lam"]), float(model["mu"])
        if "lambda0" in model and "lambda1" in model:
            return float(model["lambda0"]), float(model["lambda1"])
        if "E" in model and "nu" in model:
            E = float(model["E"])
            nu = float(model["nu"])
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return lam, mu
    raise AttributeError("Cannot infer (lam,mu).")


def _model_get(model, name: str, default=None):
    if hasattr(model, name):
        return getattr(model, name)
    if isinstance(model, dict) and name in model:
        return model[name]
    return default


@dataclass
class PhaseFieldDamageModel(DamageModelBase):
    """
    Hybrid phase-field model for Hu-Zhang framework.

    Final design:
      - d : FE function in discr.space_d
      - H : quadrature-history field, shape (NC, NQ), stored in state.H
      - elasticity degradation: isotropic g(d)
      - history update: spectral / hybrid on quadrature points
    """
    density_type: str = "AT2"
    degradation_type: str = "quadratic"
    split: str = "hybrid"         # "hybrid", "spectral", "isotropic"
    eps_g: float = 1e-10
    clamp_max: float = 0.999999
    debug: bool = False

    lam: Optional[float] = None
    mu: Optional[float] = None
    Gc: Optional[float] = None
    l0: Optional[float] = None

    _gfun: Optional[Any] = None
    _hfun: Optional[Any] = None

    def on_build(self, discr, state, case):
        model = case.model()

        self.lam, self.mu = _material_lame_from_model(model)

        Gc = _model_get(model, "Gc", None)
        l0 = _model_get(model, "l0", None)
        if Gc is None:
            raise AttributeError("PhaseFieldDamageModel requires model.Gc.")
        if l0 is None:
            raise AttributeError("PhaseFieldDamageModel requires model.l0.")

        self.Gc = float(Gc)
        self.l0 = float(l0)

        self._gfun = EnergyDegradationFunction(self.degradation_type)
        self._hfun = CrackSurfaceDensityFunction(self.density_type)

        if state.d is not None:
            state.d[:] = bm.clip(state.d[:], 0.0, self.clamp_max)

        # NEW: H is quadrature-history, not FE function
        state.H = None

        if self.debug:
            print(
                f"[PhaseFieldDamageModel] on_build: "
                f"Gc={self.Gc}, l0={self.l0}, lam={self.lam}, mu={self.mu}, "
                f"density={self.density_type}, degradation={self.degradation_type}, split={self.split}"
            )

    def on_mesh_changed(self, old_discr, new_discr, old_state, new_state, case):
        model = case.model()
        self.lam, self.mu = _material_lame_from_model(model)

        Gc = _model_get(model, "Gc", None)
        l0 = _model_get(model, "l0", None)
        if Gc is None or l0 is None:
            raise AttributeError("PhaseFieldDamageModel requires model.Gc and model.l0 after mesh change.")
        self.Gc = float(Gc)
        self.l0 = float(l0)

        # safest choice for quadrature-history after mesh change:
        new_state.H = None

        if self.debug:
            print(
                f"[PhaseFieldDamageModel] on_mesh_changed: "
                f"Gc={self.Gc}, l0={self.l0}, lam={self.lam}, mu={self.mu}"
            )

    # ------------------------------------------------------------
    # elasticity degradation
    # ------------------------------------------------------------
    def coef_bary(self, state, bcs, index=None):
        dval = state.d(bcs, index=index)
        dval = bm.clip(dval, 0.0, self.clamp_max)

        gd = self._gfun.degradation_function(dval)
        gd = bm.maximum(gd, self.eps_g)

        if self.debug:
            print(
                "[PhaseFieldDamageModel.coef_bary] "
                f"g min/max = {float(bm.min(gd)):.3e} / {float(bm.max(gd)):.3e}"
            )
        return gd

    # ------------------------------------------------------------
    # crack density / degradation helpers
    # ------------------------------------------------------------
    def crack_density(self, d):
        return self._hfun.density_function(d)

    def crack_density_grad(self, d):
        return self._hfun.grad_density_function(d)

    def crack_density_hess(self, d):
        return self._hfun.grad_grad_density_function(d)

    def degradation(self, d):
        return self._gfun.degradation_function(d)

    def degradation_grad(self, d):
        return self._gfun.grad_degradation_function(d)

    def degradation_hess(self, d):
        return self._gfun.grad_grad_degradation_function(d)

    # ------------------------------------------------------------
    # quadrature-history update
    # ------------------------------------------------------------
    def update_history_on_quadrature(
        self,
        discr,
        state,
        case,
        bcs,
        index=None,
        *,
        parallel: bool = False,
        nproc: Optional[int] = None,
    ):
        """
        Update H on the given quadrature points.

        With ``parallel=True`` (typically from ``PhaseFieldAssembler`` when
        ``assembly_parallel`` is on), cells are processed in batches via threads.
        Set env ``FRACTUREX_HISTORY_UPDATE_PARALLEL=0`` (or ``false``/``off``) to force
        the serial path even if ``parallel=True``. Optional one-line wall-clock timing
        from the caller: ``FRACTUREX_PROFILE_HISTORY_UPDATE=1``.

        Large-array performance may still benefit from tuning BLAS/OpenMP threads;
        compare against batched parallel using profiling on representative meshes.

        Parameters
        ----------
        bcs : quadrature barycentric points
        index : cell indices
        parallel : optional batched cell parallel update
        nproc : optional worker cap (defaults to CPU count)
        """
        mesh = discr.mesh
        if mesh is None:
            raise RuntimeError("discr.mesh is None in update_history_on_quadrature.")

        idx_all = bm.arange(mesh.number_of_cells()) if index is None else bm.asarray(index).reshape(-1)
        n_cell = int(idx_all.shape[0])

        env_serial = str(os.getenv("FRACTUREX_HISTORY_UPDATE_PARALLEL", "")).strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        parallel_eff = bool(parallel) and (not env_serial) and n_cell >= 2

        phip = None
        if parallel_eff:
            nw = int(nproc) if nproc is not None else _history_update_nproc_default()
            nproc_use = min(max(1, nw), n_cell)
            edges = np.linspace(0, n_cell, nproc_use + 1, dtype=int)
            tasks = []
            for k in range(nproc_use):
                i0, i1 = int(edges[k]), int(edges[k + 1])
                if i1 > i0:
                    tasks.append((self, state, bcs, idx_all[i0:i1]))
            workers = min(int(nw), len(tasks))
            try:
                with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                    parts = list(pool.map(_phase_history_chunk_worker, tasks))
                phip = bm.concatenate(parts, axis=0)
            except Exception:
                phip = None

        if phip is None:
            grad_u = state.u.grad_value(bcs, index=index)
            strain = 0.5 * (grad_u + bm.swapaxes(grad_u, -2, -1))

            if self.split in ("hybrid", "spectral"):
                phip = self._spectral_positive_energy_density(strain)
            elif self.split == "isotropic":
                phip = self._isotropic_energy_density(strain)
                phip = bm.maximum(phip, 0.0)
            else:
                raise ValueError(f"Unknown split type: {self.split}")

            phip = bm.maximum(phip, 0.0)

        if state.H is None:
            state.H = phip.copy()
        else:
            if state.H.shape != phip.shape:
                raise ValueError(
                    f"Quadrature-history shape mismatch: state.H has {state.H.shape}, "
                    f"but current phip has {phip.shape}. "
                    f"Make sure assembler uses the same quadrature rule when updating/reading H."
                )
            state.H = bm.maximum(state.H, phip)

        if self.debug:
            print(
                "[PhaseFieldDamageModel.update_history_on_quadrature] "
                f"H min/max = {float(bm.min(state.H)):.3e} / {float(bm.max(state.H)):.3e}"
            )
        return state.H

    # backward-compatible alias
    def update_after_elastic(self, discr, state, case):
        raise RuntimeError(
            "Quadrature-history version does not use update_after_elastic(discr,state,case) "
            "directly. Call update_history_on_quadrature(..., bcs, index) from PhaseFieldAssembler."
        )

    # ------------------------------------------------------------
    # energy density
    # ------------------------------------------------------------
    def _isotropic_energy_density(self, strain):
        lam = float(self.lam)
        mu = float(self.mu)
        tr = bm.einsum("...ii", strain)
        e2 = bm.einsum("...ij,...ij->...", strain, strain)
        psi = 0.5 * lam * tr**2 + mu * e2
        return psi

    def _spectral_positive_energy_density(self, strain):
        lam = float(self.lam)
        mu = float(self.mu)

        eps_p, _ = self._strain_pm_eig_decomposition(strain)
        tr = bm.einsum("...ii", strain)
        tr_p, _ = self._macaulay_operation(tr)
        epp2 = bm.einsum("...ii", eps_p @ eps_p)

        psi_plus = 0.5 * lam * tr_p**2 + mu * epp2
        return psi_plus

    def _macaulay_operation(self, alpha):
        val = bm.abs(alpha)
        p = 0.5 * (alpha + val)
        m = 0.5 * (alpha - val)
        return p, m

    def _strain_pm_eig_decomposition(self, s):
        w, v = bm.linalg.eigh(s)
        p, m = self._macaulay_operation(w)

        sp = bm.zeros_like(s)
        sm = bm.zeros_like(s)

        GD = s.shape[-1]
        for i in range(GD):
            ni = v[..., i]

            nip = p[..., i, None] * ni
            sp = sp + nip[..., None] * ni[..., None, :]

            nim = m[..., i, None] * ni
            sm = sm + nim[..., None] * ni[..., None, :]

        return sp, sm

    # ------------------------------------------------------------
    # params
    # ------------------------------------------------------------
    def regularization_constant(self):
        _, c_d = self.crack_density(0.0)
        return float(c_d)

    def length_scale(self):
        return float(self.l0)

    def fracture_toughness(self):
        return float(self.Gc)