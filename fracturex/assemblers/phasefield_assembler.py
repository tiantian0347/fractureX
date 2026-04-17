# fracturex/assemblers/phasefield_assembler.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict

from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric
from fealpy.fem import (
    BilinearForm,
    LinearForm,
    ScalarDiffusionIntegrator,
    ScalarMassIntegrator,
    ScalarSourceIntegrator,
    DirichletBC,
)


@dataclass
class PhaseFieldSystem:
    A: Any
    F: Any
    decode: Callable[[Any], Any]   # dd -> d_new
    meta: Dict[str, Any]


class PhaseFieldAssembler:
    """
    Quadrature-history version of phase-field assembler.

    Final structure:
      1) build current quadrature rule
      2) update H on the SAME quadrature points
      3) assemble phase-field system using state.H directly as (NC,NQ)
    """

    def __init__(self, discr, case, damage, *, q: Optional[int] = None, debug: bool = False):
        self.discr = discr
        self.case = case
        self.damage = damage
        self.q = q
        self.debug = bool(debug)
        self._phase_initial_damage_applied = False
        self._quad_cache = None
        self._quad_cache_key = None
        self._g0_const_coef = None

    def assemble(self, load: float) -> PhaseFieldSystem:
        discr = self.discr
        case = self.case
        damage = self.damage
        state = discr.state

        if state is None:
            raise RuntimeError("Discretization state is None. Call discr.build(...) first.")

        # One-time initialization: mark pre-crack by setting d directly (e.g. d=1 on notch line)
        self._apply_phase_initial_damage_once(load)

        space = discr.space_d
        if space is None:
            raise RuntimeError("discr.space_d is None.")

        d_old = state.d
        gdof = space.number_of_global_dofs()

        Gc = float(damage.fracture_toughness())
        l0 = float(damage.length_scale())

        # IMPORTANT: phase-field quadrature is tied to damage space, not Hu-Zhang p
        q = discr.damage_p + 3 if self.q is None else self.q

        bcs, index = self._get_quadrature_data(q)

        # ----------------------------------------------------------
        # update quadrature-history H on exactly the same quadrature
        # ----------------------------------------------------------
        damage.update_history_on_quadrature(
            discr=discr,
            state=state,
            case=case,
            bcs=bcs,
            index=index,
        )

        if state.H is None:
            raise RuntimeError("state.H is None after update_history_on_quadrature.")

        # ----------------------------------------------------------
        # coefficient functions
        # ----------------------------------------------------------
        @barycentric
        def diff_coef(bc, index=None):
            _, c_d = damage.crack_density_hess(d_old(bc, index=index))
            return Gc * l0 * 2.0 / c_d

        @barycentric
        def mass_coef1(bc, index=None):
            gg_hd, c_d = damage.crack_density_hess(d_old(bc, index=index))
            return gg_hd * Gc / (l0 * c_d)

        @barycentric
        def mass_coef2(bc, index=None):
            Hq = state.H if index is None else state.H[index]   # (NC,NQ)
            Hq = bm.maximum(Hq, 0.0)

            gg_gd = damage.degradation_hess(d_old(bc, index=index))
            return gg_gd * Hq

        @barycentric
        def source_coef(bc, index=None):
            Hq = state.H if index is None else state.H[index]   # (NC,NQ)
            Hq = bm.maximum(Hq, 0.0)

            if hasattr(damage._gfun, "grad_degradation_function_constant_coef"):
                if self._g0_const_coef is None:
                    self._g0_const_coef = float(damage._gfun.grad_degradation_function_constant_coef())
                return -1.0 * self._g0_const_coef * Hq

            z = bm.zeros_like(Hq)
            g0p = damage.degradation_grad(z)
            return -g0p * Hq

        # ----------------------------------------------------------
        # assemble A
        # ----------------------------------------------------------
        bform_const = BilinearForm(space)
        bform_const.add_integrator(
            ScalarDiffusionIntegrator(coef=diff_coef, q=q),
            ScalarMassIntegrator(coef=mass_coef1, q=q),
        )
        A_const = bform_const.assembly()

        bform_hist = BilinearForm(space)
        bform_hist.add_integrator(
            ScalarMassIntegrator(coef=mass_coef2, q=q),
        )
        A_hist = bform_hist.assembly()
        A = A_const + A_hist

        # ----------------------------------------------------------
        # assemble rhs
        # ----------------------------------------------------------
        lform = LinearForm(space)
        lform.add_integrator(
            ScalarSourceIntegrator(source=source_coef, q=q)
        )
        rhs = lform.assembly()

        # residual / increment form: A dd = rhs - A d_old
        F = rhs - A @ d_old[:]

        # ----------------------------------------------------------
        # phase-field Dirichlet BC
        # ----------------------------------------------------------
        A, F = self._apply_phase_dirichlet_bc(A, F, load)

        if self.debug:
            hmin = float(bm.min(state.H)) if state.H is not None else 0.0
            hmax = float(bm.max(state.H)) if state.H is not None else 0.0
            dmin = float(bm.min(bm.asarray(d_old[:]))) if d_old is not None else 0.0
            dmax = float(bm.max(bm.asarray(d_old[:]))) if d_old is not None else 0.0
            print(
                "[PhaseFieldAssembler] "
                f"gdof={gdof}, q={q}, Gc={Gc:.3e}, l0={l0:.3e}, "
                f"d min/max={dmin:.3e}/{dmax:.3e}, "
                f"H min/max={hmin:.3e}/{hmax:.3e}"
            )

        def decode(dd):
            d_new = space.function()
            d_new[:] = d_old[:] + bm.asarray(dd).reshape(-1)
            d_new[:] = bm.clip(d_new[:], 0.0, 1.0)
            return d_new

        meta = dict(
            type="phasefield",
            model=getattr(damage, "density_type", "unknown"),
            degradation=getattr(damage, "degradation_type", "unknown"),
            load=float(load),
            gdof=int(gdof),
            q=int(q),
            Gc=float(Gc),
            l0=float(l0),
        )

        return PhaseFieldSystem(A=A, F=F, decode=decode, meta=meta)

    def _get_quadrature_data(self, q: int):
        """Cache cell quadrature points/indices for fixed mesh and order."""
        mesh = self.discr.mesh
        NC = mesh.number_of_cells()
        key = (id(mesh), int(q), int(NC))
        if self._quad_cache is not None and self._quad_cache_key == key:
            return self._quad_cache["bcs"], self._quad_cache["index"]

        qf = mesh.quadrature_formula(q, "cell")
        bcs, _ = qf.get_quadrature_points_and_weights()
        index = bm.arange(NC)
        self._quad_cache_key = key
        self._quad_cache = {"bcs": bcs, "index": index}
        return bcs, index

    def _apply_phase_initial_damage_once(self, load: float):
        if self._phase_initial_damage_applied:
            return

        case = self.case
        discr = self.discr
        state = discr.state
        space = discr.space_d

        if state is None or space is None:
            return

        if not hasattr(case, "phasefield_initial_damage_data"):
            self._phase_initial_damage_applied = True
            return

        bcdata = case.phasefield_initial_damage_data(load)
        if bcdata is None:
            self._phase_initial_damage_applied = True
            return

        if isinstance(bcdata, dict):
            bcdata = [bcdata]

        ip = space.interpolation_points()
        darr = bm.asarray(state.d[:]).copy()

        if self.debug:
            print(f"[PhaseFieldAssembler._apply_phase_initial_damage_once] bcdata={bcdata}")

        for item in bcdata:
            if "bcdof" not in item or "value" not in item:
                raise ValueError(
                    "phasefield_initial_damage_data(load) must return dict(s) with keys {'bcdof','value'}."
                )

            thr = item["bcdof"]
            val = item["value"]

            if callable(thr):
                mask = bm.asarray(thr(ip)).astype(bm.bool)
                idx = bm.where(mask)[0]
            else:
                arr = bm.asarray(thr)
                if getattr(arr, "dtype", None) == bm.bool:
                    idx = bm.where(arr)[0]
                else:
                    idx = arr

            if self.debug:
                print(f"  Found {len(idx)} DOFs on pre-crack, setting to value={val}")
                if len(idx) > 0:
                    print(f"    DOF indices (first 10): {idx[:10]}")

            if callable(val):
                v = val(ip[idx])
            else:
                v = val

            darr = bm.set_at(darr, idx, v)

            if self.debug and len(idx) > 0:
                print(f"    After set_at: darr[idx] min={bm.min(darr[idx]):.6e}, max={bm.max(darr[idx]):.6e}")

        state.d[:] = bm.clip(darr, 0.0, 1.0)
        self._phase_initial_damage_applied = True
        
        if self.debug:
            print(f"[PhaseFieldAssembler._apply_phase_initial_damage_once] COMPLETE: state.d min={bm.min(state.d[:]):.6e}, max={bm.max(state.d[:]):.6e}")

    def _apply_phase_dirichlet_bc(self, A, F, load: float):
        case = self.case
        space = self.discr.space_d

        if not hasattr(case, "phasefield_dirichlet_data"):
            return A, F

        bcdata = case.phasefield_dirichlet_data(load)
        if bcdata is None:
            return A, F

        if isinstance(bcdata, dict):
            bcdata = [bcdata]

        for item in bcdata:
            if "bcdof" not in item or "value" not in item:
                raise ValueError(
                    "phasefield_dirichlet_data(load) must return dict(s) with keys {'bcdof','value'}."
                )
            bc = DirichletBC(space, gd=item["value"], threshold=item["bcdof"])
            A, F = bc.apply(A, F)

        return A, F