# fracturex/drivers/huzhang_phasefield_staggered.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List

import os
import time
import numpy as np
from scipy.sparse.linalg import spsolve, lgmres

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.utils import timer

from fracturex.cases.base import CaseBase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.base import DamageStateView, DamageModelBase
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.postprocess.reaction import reaction_from_sigma
from fracturex.utilfuc.vtk_lagrange_writer import (
    sample_fields_for_lagrange_triangle,
    write_lagrange_triangle_vtu,
)


@dataclass
class StepInfo:
    step: int
    load: float
    iters: int
    converged: bool
    err_u: float
    err_d: float
    max_d: float
    meta: Dict[str, Any]


class HuZhangPhaseFieldStaggeredDriver:
    def __init__(
        self,
        *,
        case: CaseBase,
        discr: HuZhangDiscretization,
        damage: DamageModelBase,
        elastic_assembler: Optional[HuZhangElasticAssembler] = None,
        phase_assembler: Optional[PhaseFieldAssembler] = None,
        tol: float = 1e-5,
        maxit: int = 1000,
        compute_linear_residual: bool = False,
        elastic_solver: Optional[Callable[[Any, Any], Any]] = None,
        phase_solver: Optional[Callable[[Any, Any], Any]] = None,
        adapt_hook: Optional[Callable[[int, float, HuZhangDiscretization, DamageModelBase], None]] = None,
        cell_mode: str = "mean",
        debug: bool = False,
        timing: bool = False,
        recorder: Optional[Any] = None,
    ):
        """Initialize staggered HuZhang + phase-field solver driver.

        Inputs:
            case/discr/damage: Core case, discretization and damage model objects.
            elastic_assembler/phase_assembler: Optional custom assemblers.
            tol/maxit: Nonlinear staggered convergence tolerance and max iterations.
            compute_linear_residual: Whether to compute `||Ax-b||/||b||` diagnostics.
            elastic_solver/phase_solver: Linear solver callbacks `(A, b) -> x` or `(x, info)`.
            adapt_hook: Optional callback after each load step.
            cell_mode/debug/timing/recorder: Output and diagnostics switches.
        Output:
            None. Stores configuration; real initialization happens in `initialize`.
        """
        self.case = case
        self.discr = discr
        self.damage = damage
        self.recorder = recorder

        self.elastic_assembler = (
            elastic_assembler
            if elastic_assembler is not None
            else HuZhangElasticAssembler(discr, case, damage)
        )
        self.phase_assembler = (
            phase_assembler
            if phase_assembler is not None
            else PhaseFieldAssembler(discr, case, damage)
        )

        self.tol = float(tol)
        self.maxit = int(maxit)
        self.debug = bool(debug)
        self.timing = bool(timing)
        self.compute_linear_residual = bool(compute_linear_residual)
        self.cell_mode = cell_mode

        self.elastic_solver = elastic_solver if elastic_solver is not None else self._default_spsolve
        self.phase_solver = phase_solver if phase_solver is not None else self._default_spsolve

        self.adapt_hook = adapt_hook
        self._initialized = False
        self._sigma_physical_eval = None
        self._tmr = timer() if self.timing else None
        if self._tmr is not None:
            next(self._tmr)

    def _timer_mark(self, tag: Optional[str]):
        if self._tmr is None:
            return
        try:
            self._tmr.send(tag)
        except Exception:
            pass

    @staticmethod
    def _relative_residual(A, x, b) -> float:
        try:
            Ax = A @ np.asarray(x).reshape(-1)
            b_ = np.asarray(b).reshape(-1)
            num = np.linalg.norm(Ax - b_)
            den = max(np.linalg.norm(b_), 1e-30)
            return float(num / den)
        except Exception:
            return float("nan")

    @staticmethod
    def _solver_name(solver: Any) -> str:
        name = getattr(solver, "__name__", None)
        if name:
            return str(name)
        return solver.__class__.__name__

    @staticmethod
    def _solve_with_diagnostics(solver: Callable[[Any, Any], Any], A, b):
        """
        Normalize linear solver outputs:
        - x only
        - (x, info-like) where info-like may carry krylov diagnostics
        """
        out = solver(A, b)
        info = None
        if isinstance(out, tuple) and len(out) >= 1:
            x = np.asarray(out[0], dtype=float).reshape(-1)
            info = out[1] if len(out) > 1 else None
        else:
            x = np.asarray(out, dtype=float).reshape(-1)

        diag = dict(
            solver=HuZhangPhaseFieldStaggeredDriver._solver_name(solver),
            niter=1,            # direct solve equivalent
            converged=True,     # direct solve equivalent
            residual_norm=float("nan"),
        )
        if info is not None:
            if hasattr(info, "solver"):
                diag["solver"] = str(getattr(info, "solver"))
            if hasattr(info, "niter"):
                try:
                    diag["niter"] = int(getattr(info, "niter"))
                except Exception:
                    pass
            if hasattr(info, "converged"):
                try:
                    diag["converged"] = bool(getattr(info, "converged"))
                except Exception:
                    pass
            if hasattr(info, "residual_norm"):
                try:
                    diag["residual_norm"] = float(getattr(info, "residual_norm"))
                except Exception:
                    pass
        return x, diag

    def initialize(self):
        """Initialize damage model with current discretization state.

        Inputs:
            None (uses `self.discr`, `self.damage`, `self.case`).
        Output:
            None. Calls `damage.on_build(...)` once and sets initialized flag.
        """
        if self._initialized:
            return

        state = self.discr.state
        if state is None:
            raise RuntimeError("Discretization must be built before driver.initialize().")

        view = DamageStateView(
            d=state.d,
            sigma=state.sigma,
            u=state.u,
            r_hist=state.r_hist,
            H=state.H,
        )
        self.damage.on_build(self.discr, view, self.case)

        self._initialized = True

    def run(self, loads: List[float]) -> List[StepInfo]:
        """Run the full staggered solve over all load steps.

        Input:
            loads: List of prescribed load values per step.
        Output:
            List of `StepInfo`, one item per load step.
        """
        self._timer_mark("run_start")
        self.initialize()
        out: List[StepInfo] = []

        if self.recorder is not None:
            m = self.case.model()
            meta = dict(
                case=self.case.name,
                p=int(self.discr.p),
                use_relaxation=bool(self.discr.use_relaxation),
                elastic_formulation=str(getattr(self.elastic_assembler, "formulation", "standard")),
                history_source=str(getattr(self.damage, "history_source", "from_u")),
                mesh=dict(
                    NN=int(self.discr.mesh.number_of_nodes()),
                    NE=int(self.discr.mesh.number_of_edges()),
                    NC=int(self.discr.mesh.number_of_cells()),
                ),
                material={
                    k: float(getattr(m, k))
                    for k in ["lam", "mu", "E", "nu", "Gc", "l0"]
                    if hasattr(m, k)
                },
                timing_enabled=bool(self.timing),
                compute_linear_residual=bool(self.compute_linear_residual),
            )
            self.recorder.write_meta(meta)

        for s, load in enumerate(loads):
            info = self.solve_one_step(step=s, load=float(load))
            out.append(info)

            if self.adapt_hook is not None:
                self.adapt_hook(s, float(load), self.discr, self.damage)
                self.elastic_assembler.discr = self.discr
                self.phase_assembler.discr = self.discr
                # future-proof for adaptive remesh: initial-damage mask should be
                # allowed to re-apply on the rebuilt discretization if needed.
                if hasattr(self.phase_assembler, "_phase_initial_damage_applied"):
                    self.phase_assembler._phase_initial_damage_applied = False

        self._timer_mark("run_end")
        return out

    def solve_one_step(self, *, step: int, load: float) -> StepInfo:
        """Solve one load step using staggered elastic/phase iterations.

        Inputs:
            step: Load step index.
            load: Current scalar load/displacement value.
        Output:
            `StepInfo` including convergence stats and postprocess metadata.
        """
        discr = self.discr
        state = discr.state
        if state is None:
            raise RuntimeError("Discretization must be built before solving.")

        converged = False
        err_u = bm.inf
        err_d = bm.inf

        e0_u = None
        e0_d = None
        t_step0 = time.perf_counter()
        t_init_damage = 0.0
        t_elastic_assemble = 0.0
        t_elastic_solve = 0.0
        t_phase_assemble = 0.0
        t_phase_solve = 0.0
        r_lin_e = float("nan")
        r_lin_d = float("nan")
        lin_solver_e = self._solver_name(self.elastic_solver)
        lin_solver_d = self._solver_name(self.phase_solver)
        lin_niter_e = 1
        lin_niter_d = 1
        lin_converged_e = True
        lin_converged_d = True
        lin_cb_res_e = float("nan")
        lin_cb_res_d = float("nan")
        need_iter_stats = bool(self.debug or (self.recorder is not None and hasattr(self.recorder, "append_iteration")))

        # Apply one-time initial damage (e.g. pre-crack d=1) BEFORE
        # capturing d_old in staggered iterations, so irreversibility
        # uses the correct baseline.
        if hasattr(self.phase_assembler, "_apply_phase_initial_damage_once"):
            t0 = time.perf_counter()
            self._timer_mark("init_damage_start")
            self.phase_assembler._apply_phase_initial_damage_once(load)
            self._timer_mark("init_damage_end")
            t_init_damage += time.perf_counter() - t0

        # Debug/validation switch: disable per-load-step cache optimization to
        # compare against the legacy path when investigating symmetry issues.
        disable_step_cache = os.environ.get("FRACTUREX_DISABLE_LOADSTEP_CACHE", "0") == "1"
        if (not disable_step_cache) and hasattr(self.elastic_assembler, "begin_load_step"):
            self.elastic_assembler.begin_load_step(load)
        if (not disable_step_cache) and hasattr(self.phase_assembler, "begin_load_step"):
            self.phase_assembler.begin_load_step(load)

        for k in range(self.maxit):
            t_iter0 = time.perf_counter()
            u_old = bm.asarray(state.u[:]).copy()
            d_old = bm.asarray(state.d[:]).copy()
            t_elastic_assemble_iter = 0.0
            t_elastic_solve_iter = 0.0
            t_phase_assemble_iter = 0.0
            t_phase_solve_iter = 0.0

            # ----------------------------------------------------------
            # (1) elasticity solve with current d
            # ----------------------------------------------------------
            t0 = time.perf_counter()
            self._timer_mark("elastic_assemble_start")
            sys_e = self.elastic_assembler.assemble(load)
            self._timer_mark("elastic_assemble_end")
            t_elastic_assemble_iter = float(time.perf_counter() - t0)
            t_elastic_assemble += t_elastic_assemble_iter

            t0 = time.perf_counter()
            self._timer_mark("elastic_solve_start")
            Xe, lin_e = self._solve_with_diagnostics(self.elastic_solver, sys_e.A, sys_e.F)
            self._timer_mark("elastic_solve_end")
            t_elastic_solve_iter = float(time.perf_counter() - t0)
            t_elastic_solve += t_elastic_solve_iter
            lin_solver_e = str(lin_e["solver"])
            lin_niter_e = int(lin_e["niter"])
            lin_converged_e = bool(lin_e["converged"])
            lin_cb_res_e = float(lin_e["residual_norm"])
            if self.compute_linear_residual:
                r_lin_e = self._relative_residual(sys_e.A, Xe, sys_e.F)

            sigma, u, sigma_physical = sys_e.decode(Xe)
            state.sigma[:] = sigma[:]
            state.u[:] = u[:]
            self._sigma_physical_eval = sigma_physical

            # ----------------------------------------------------------
            # (2) phase-field assembly will update H internally on its own quadrature
            # ----------------------------------------------------------
            t0 = time.perf_counter()
            self._timer_mark("phase_assemble_start")
            sys_d = self.phase_assembler.assemble(load)
            self._timer_mark("phase_assemble_end")
            t_phase_assemble_iter = float(time.perf_counter() - t0)
            t_phase_assemble += t_phase_assemble_iter

            t0 = time.perf_counter()
            self._timer_mark("phase_solve_start")
            dd, lin_d = self._solve_with_diagnostics(self.phase_solver, sys_d.A, sys_d.F)
            self._timer_mark("phase_solve_end")
            t_phase_solve_iter = float(time.perf_counter() - t0)
            t_phase_solve += t_phase_solve_iter
            lin_solver_d = str(lin_d["solver"])
            lin_niter_d = int(lin_d["niter"])
            lin_converged_d = bool(lin_d["converged"])
            lin_cb_res_d = float(lin_d["residual_norm"])
            if self.compute_linear_residual:
                r_lin_d = self._relative_residual(sys_d.A, dd, sys_d.F)

            d_trial = sys_d.decode(dd)

            # irreversibility
            state.d[:] = bm.maximum(d_old, d_trial[:])
            state.d[:] = bm.clip(state.d[:], 0.0, 1.0)

            # ----------------------------------------------------------
            # (3) staggered convergence check by iterate increments
            # ----------------------------------------------------------
            du_abs = float(bm.linalg.norm(bm.asarray(state.u[:]) - u_old))
            dd_abs = float(bm.linalg.norm(bm.asarray(state.d[:]) - d_old))

            # Follow phasefield/main_solve convergence style:
            # normalize by the first iteration increment in each load step.
            # Add a tiny, state-scaled lower bound to avoid pathological
            # denominator collapse when the first increment is numerically tiny.
            if e0_u is None:
                u_scale = float(bm.linalg.norm(bm.asarray(state.u[:])))
                e0_u = max(du_abs, 1e-14 * u_scale, 1e-30)
            if e0_d is None:
                d_scale = float(bm.linalg.norm(bm.asarray(state.d[:])))
                e0_d = max(dd_abs, 1e-14 * d_scale, 1e-30)

            err_u = du_abs / e0_u
            err_d = dd_abs / e0_d
            error = max(err_u, err_d)
            max_d_iter = float(bm.max(bm.asarray(state.d[:])))
            gdof_sigma = int(discr.gdof_sigma)
            gdof_u = int(discr.gdof_u)
            gdof_d = int(discr.space_d.number_of_global_dofs())

            if need_iter_stats:
                iter_row = dict(
                    step=int(step),
                    load=float(load),
                    iter=int(k + 1),
                    gdof_sigma=gdof_sigma,
                    gdof_u=gdof_u,
                    gdof_d=gdof_d,
                    du_abs=float(du_abs),
                    dd_abs=float(dd_abs),
                    err_u=float(err_u),
                    err_d=float(err_d),
                    max_d=max_d_iter,
                    linear_solver_elastic=str(lin_solver_e),
                    linear_solver_phase=str(lin_solver_d),
                    linear_niter_elastic=int(lin_niter_e),
                    linear_niter_phase=int(lin_niter_d),
                    linear_converged_elastic=bool(lin_converged_e),
                    linear_converged_phase=bool(lin_converged_d),
                    linear_cb_res_elastic=float(lin_cb_res_e),
                    linear_cb_res_phase=float(lin_cb_res_d),
                    linear_res_elastic=float(r_lin_e),
                    linear_res_phase=float(r_lin_d),
                    t_elastic_assemble_iter_s=float(t_elastic_assemble_iter),
                    t_elastic_solve_iter_s=float(t_elastic_solve_iter),
                    t_phase_assemble_iter_s=float(t_phase_assemble_iter),
                    t_phase_solve_iter_s=float(t_phase_solve_iter),
                    t_elastic_iter_total_s=float(t_elastic_assemble_iter + t_elastic_solve_iter),
                    t_phase_iter_total_s=float(t_phase_assemble_iter + t_phase_solve_iter),
                    t_iter_s=float(time.perf_counter() - t_iter0),
                )
                if self.recorder is not None and hasattr(self.recorder, "append_iteration"):
                    self.recorder.append_iteration(iter_row)

            # Always print nonlinear iteration progress for realtime monitoring.
            print(
                f"[step {step} load {load:.4e}] iter {k + 1}: "
                f"err_u={err_u:.3e}, err_d={err_d:.3e}, error={error:.3e}, "
                f"max_d={max_d_iter:.3e}"
            )

            if error < self.tol:
                converged = True
                iters = k + 1
                break
        else:
            iters = self.maxit

        q = self.discr.damage_p + 3
        if getattr(self.case, "output_enabled", False):
            self._save_vtkfile(
                f"results/{self.case.name}_step_{step:03d}.vtu",
                cell_mode=self.cell_mode,
                q=q,
                sigma_eval=self._sigma_physical_eval,
            )

        lp = self.case.load_dirichlet_piece(load)
        dir_load = (lp.direction or "y")
        if dir_load not in ("x", "y", "z"):
            dir_load = "y"

        sigma_react = self._sigma_physical_eval if self._sigma_physical_eval is not None else state.sigma
        R = reaction_from_sigma(
            self.discr.mesh,
            sigma_react,
            lp.threshold,
            direction=dir_load,
            q=q,
            sign=-1,
        )

        step_walltime = float(time.perf_counter() - t_step0)

        meta = dict(
            gdof_sigma=int(discr.gdof_sigma),
            gdof_u=int(discr.gdof_u),
            gdof_d=int(discr.space_d.number_of_global_dofs()),
            elastic_formulation=str(getattr(self.elastic_assembler, "formulation", "standard")),
            history_source=str(getattr(self.damage, "history_source", "from_u")),
            max_d=float(bm.max(bm.asarray(state.d[:]))),
            max_H=float(bm.max(state.H)) if state.H is not None else 0.0,
            R=float(R),
            residual_force=float(R),
            R_dir=dir_load,
            load=float(load),
            load_dir=dir_load,
            linear_solver_elastic=str(lin_solver_e),
            linear_solver_phase=str(lin_solver_d),
            linear_niter_elastic=int(lin_niter_e),
            linear_niter_phase=int(lin_niter_d),
            linear_converged_elastic=bool(lin_converged_e),
            linear_converged_phase=bool(lin_converged_d),
            linear_cb_res_elastic=float(lin_cb_res_e),
            linear_cb_res_phase=float(lin_cb_res_d),
            linear_res_elastic=float(r_lin_e),
            linear_res_phase=float(r_lin_d),
            t_step_s=step_walltime,
            t_init_damage_s=float(t_init_damage),
            t_elastic_assemble_s=float(t_elastic_assemble),
            t_elastic_solve_s=float(t_elastic_solve),
            t_phase_assemble_s=float(t_phase_assemble),
            t_phase_solve_s=float(t_phase_solve),
        )
        meta[f"reaction_{dir_load}"] = float(R)
        meta[f"disp_{dir_load}"] = float(load)

        info = StepInfo(
            step=step,
            load=float(load),
            iters=int(iters),
            converged=converged,
            err_u=float(err_u),
            err_d=float(err_d),
            max_d=float(bm.max(bm.asarray(state.d[:]))),
            meta=meta,
        )

        if self.recorder is not None:
            row = dict(info.meta)
            row.update(
                step=info.step,
                load=info.load,
                iters=info.iters,
                converged=info.converged,
                err_u=info.err_u,
                err_d=info.err_d,
                max_d=info.max_d,
                residual_force=float(R),
                linear_res_elastic=float(r_lin_e),
                linear_res_phase=float(r_lin_d),
                t_step_s=step_walltime,
                t_init_damage_s=float(t_init_damage),
                t_elastic_assemble_s=float(t_elastic_assemble),
                t_elastic_solve_s=float(t_elastic_solve),
                t_phase_assemble_s=float(t_phase_assemble),
                t_phase_solve_s=float(t_phase_solve),
            )
            self.recorder.write_step(row)
            self.recorder.save_checkpoint(step, discr, state)

        return info

    @staticmethod
    def _default_spsolve(A, F):
        """Default direct linear solver wrapper.

        Inputs:
            A: Sparse matrix in SciPy/compatible format.
            F: Right-hand-side vector.
        Output:
            Solution vector from `scipy.sparse.linalg.spsolve`.
        """
        if hasattr(A, "to_scipy"):
            A_ = A.to_scipy().tocsr()
        else:
            A_ = A.tocsr() if hasattr(A, "tocsr") else A

        F_ = np.asarray(F, dtype=float).reshape(-1)
        return spsolve(A_, F_)

    @staticmethod
    def _default_lgmres(
        A,
        F,
        *,
        atol: float = 1e-20,
        rtol: float = 1e-10,
        maxiter: int = 2000,
        check_rtol: float = 1e-8,
        fallback_to_spsolve: bool = True,
    ):
        """Default iterative solver wrapper with stability fallback.

        Inputs:
            A/F: Linear system matrix and rhs.
            atol/rtol/maxiter: LGMRES stopping controls.
            check_rtol: Extra posterior residual threshold.
            fallback_to_spsolve: Whether to fallback to direct solve on instability.
        Output:
            Solution vector; may come from LGMRES or fallback direct solve.
        """
        if hasattr(A, "to_scipy"):
            A_ = A.to_scipy().tocsr()
        else:
            A_ = A.tocsr() if hasattr(A, "tocsr") else A

        F_ = np.asarray(F, dtype=float).reshape(-1)
        x, info = lgmres(A_, F_, atol=atol, rtol=rtol, maxiter=maxiter)

        x = np.asarray(x, dtype=float).reshape(-1)
        bnorm = max(float(np.linalg.norm(F_)), 1e-30)
        rrel = float(np.linalg.norm(A_ @ x - F_) / bnorm)
        bad = (info != 0) or (not np.isfinite(x).all()) or (not np.isfinite(rrel)) or (rrel > check_rtol)

        if bad:
            print(
                "[HuZhangPhaseFieldStaggeredDriver] "
                f"lgmres unstable/non-converged: info={info}, relres={rrel:.3e}; "
                f"fallback_to_spsolve={fallback_to_spsolve}"
            )
            if fallback_to_spsolve:
                return spsolve(A_, F_)
        return x


    def _u_to_nodal(self, u_fun, mesh):
        """
        Recover vector displacement to mesh nodal values by element-vertex averaging.

        Parameters
        ----------
        u_fun : FE function on discr.space_u
        mesh  : TriangleMesh

        Returns
        -------
        out : (NN, GD)
            Nodal displacement values for vtk nodedata.
        """
        cell = mesh.entity("cell")              # (NC, 3) for triangles
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()

        if cell.shape[1] != 3:
            raise NotImplementedError("_u_to_nodal currently only supports triangular meshes.")

        bcs = bm.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]], dtype=bm.float64)

        vals = u_fun(bcs, index=bm.arange(NC))
        # expected shape: (NC, 3, GD)
        if vals.ndim != 3 or vals.shape[1] != 3:
            raise ValueError(f"Unexpected u_fun evaluation shape: {vals.shape}")

        out = bm.zeros((NN, GD), dtype=vals.dtype)
        cnt = bm.zeros((NN,), dtype=vals.dtype)

        for lv in range(3):
            nid = cell[:, lv]
            bm.add.at(out, nid, vals[:, lv, :])
            bm.add.at(cnt, nid, 1.0)

        out = out / cnt[:, None]
        return out


    def _u_to_cell_barycenter(self, u_fun, mesh):
        """
        Evaluate displacement at cell barycenters.

        Returns
        -------
        val : (NC, GD)
        """
        NC = mesh.number_of_cells()
        cell = mesh.entity("cell")

        if cell.shape[1] != 3:
            raise NotImplementedError("_u_to_cell_barycenter currently only supports triangular meshes.")

        bc = bm.array([[1/3, 1/3, 1/3]], dtype=bm.float64)
        val = u_fun(bc, index=bm.arange(NC))

        # expected shape: (NC, 1, GD)
        if val.ndim == 3:
            val = val[:, 0, :]
        return val


    def _sigma_to_cell_barycenter(self, sigma_fun, mesh):
        """
        Evaluate sigma (Voigt form) at cell barycenters.

        Returns
        -------
        sig : (NC, 3) in 2D, corresponding to [sxx, sxy, syy]
        """
        NC = mesh.number_of_cells()
        cell = mesh.entity("cell")

        if cell.shape[1] != 3:
            raise NotImplementedError("_sigma_to_cell_barycenter currently only supports triangular meshes.")

        bc = bm.array([[1/3, 1/3, 1/3]], dtype=bm.float64)
        sig = sigma_fun(bc, index=bm.arange(NC))

        # expected shape: (NC, 1, 3) or (NC, 3)
        if sig.ndim == 3:
            sig = sig[:, 0, :]
        return sig


    def _sigma_to_nodal(self, sigma_fun, mesh):
        """
        Recover sigma (Voigt) to nodal values by element-vertex averaging.

        Returns
        -------
        out : (NN, 3) in 2D, corresponding to [sxx, sxy, syy]
        """
        cell = mesh.entity("cell")              # (NC, 3)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        if cell.shape[1] != 3:
            raise NotImplementedError("_sigma_to_nodal currently only supports triangular meshes.")

        bcs = bm.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]], dtype=bm.float64)

        vals = sigma_fun(bcs, index=bm.arange(NC))
        # expected shape: (NC, 3, 3) in 2D:
        #   3 local vertices, 3 Voigt components
        if vals.ndim == 2:
            vals = vals[:, None, :].repeat(3, axis=1)
        if vals.ndim != 3 or vals.shape[1] != 3:
            raise ValueError(f"Unexpected sigma_fun evaluation shape: {vals.shape}")

        out = bm.zeros((NN, vals.shape[-1]), dtype=vals.dtype)
        cnt = bm.zeros((NN,), dtype=vals.dtype)

        for lv in range(3):
            nid = cell[:, lv]
            bm.add.at(out, nid, vals[:, lv, :])
            bm.add.at(cnt, nid, 1.0)

        out = out / cnt[:, None]
        return out


    def _voigt2d_principal_and_vm(self, sig):
        """
        Compute principal stress s1 and von Mises stress from 2D Voigt stress.

        Parameters
        ----------
        sig : (N, 3)
            [sxx, sxy, syy]

        Returns
        -------
        s1  : (N,)
        svm : (N,)
        """
        sxx = sig[:, 0]
        sxy = sig[:, 1]
        syy = sig[:, 2]

        s1 = 0.5 * (sxx + syy) + bm.sqrt((0.5 * (sxx - syy))**2 + sxy**2)
        svm = bm.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
        return s1, svm


    def _save_vtkfile(self, fname: str, *, cell_mode: str = "mean", q: Optional[int] = None, sigma_eval=None):
        """
        Save phase-field / displacement / stress results to vtk.

        Current output strategy
        -----------------------
        - damage:
            nodedata if current damage field is already nodal-compatible
        - displacement:
            * nodal recovered displacement -> nodedata["uh"], ["ux"], ["uy"]
            * cell-barycenter displacement -> celldata["uh_cell"], ["u_norm_cell"]
        - stress:
            * cell-barycenter stress -> celldata["sigma_cell"], ["sxx_cell"], ["sxy_cell"], ["syy_cell"]
            * nodal recovered stress -> nodedata["sigma_node"], ["sxx"], ["sxy"], ["syy"]
            * both output principal stress and von Mises
        - H:
            quadrature-history -> celldata["H_cell"] by simple quadrature average
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        mesh = self.discr.mesh
        state = self.discr.state
        sigma_fun = sigma_eval if sigma_eval is not None else state.sigma

        # --------------------------------------------------
        # damage
        # --------------------------------------------------
        try:
            # NOTE:
            # if damage_p > 1 in the future, this may also need a nodal recovery
            mesh.nodedata["damage"] = bm.asarray(state.d[:])
        except Exception as e:
            print(f"[vtk] damage output failed: {e}")

        # --------------------------------------------------
        # displacement: nodal recovery
        # --------------------------------------------------
        try:
            u_node = self._u_to_nodal(state.u, mesh)   # (NN, GD)
            mesh.nodedata["uh"] = u_node
            mesh.nodedata["ux"] = u_node[:, 0]
            if u_node.shape[1] > 1:
                mesh.nodedata["uy"] = u_node[:, 1]
            if u_node.shape[1] > 2:
                mesh.nodedata["uz"] = u_node[:, 2]
        except Exception as e:
            print(f"[vtk] displacement nodal recovery failed: {e}")

        # displacement: cell barycenter
        try:
            u_cell = self._u_to_cell_barycenter(state.u, mesh)   # (NC, GD)
            mesh.celldata["uh_cell"] = u_cell
            mesh.celldata["u_norm_cell"] = bm.sqrt(bm.sum(u_cell**2, axis=-1))
        except Exception as e:
            print(f"[vtk] displacement cell output failed: {e}")

        # --------------------------------------------------
        # stress: cell barycenter
        # --------------------------------------------------
        try:
            sig_cell = self._sigma_to_cell_barycenter(sigma_fun, mesh)  # (NC, 3)
            mesh.celldata["sigma_cell"] = sig_cell
            mesh.celldata["sxx_cell"] = sig_cell[:, 0]
            mesh.celldata["sxy_cell"] = sig_cell[:, 1]
            mesh.celldata["syy_cell"] = sig_cell[:, 2]

            s1_cell, svm_cell = self._voigt2d_principal_and_vm(sig_cell)
            mesh.celldata["s1_cell"] = s1_cell
            mesh.celldata["svm_cell"] = svm_cell
        except Exception as e:
            print(f"[vtk] stress cell output failed: {e}")

        # --------------------------------------------------
        # stress: nodal recovery
        # --------------------------------------------------
        try:
            sig_node = self._sigma_to_nodal(sigma_fun, mesh)   # (NN, 3)
            mesh.nodedata["sigma_node"] = sig_node
            mesh.nodedata["sigxx"] = sig_node[:, 0]
            mesh.nodedata["sigxy"] = sig_node[:, 1]
            mesh.nodedata["sig2"] = sig_node[:, 2]

            s1_node, svm_node = self._voigt2d_principal_and_vm(sig_node)
            mesh.nodedata["s1"] = s1_node
            mesh.nodedata["svm"] = svm_node
        except Exception as e:
            print(f"[vtk] stress nodal recovery failed: {e}")

        # --------------------------------------------------
        # quadrature-history H -> cell average
        # --------------------------------------------------
        if state.H is not None:
            try:
                if cell_mode == "max":
                    H_cell = bm.max(state.H, axis=1)
                else:
                    H_cell = bm.mean(state.H, axis=1)
                mesh.celldata["H_cell"] = H_cell
            except Exception as e:
                print(f"[vtk] H_cell output failed: {e}")

        mesh.to_vtk(fname=fname)

    @staticmethod
    def _reference_triangle_subdivision(level: int):
        """
        Build reference-triangle sampling points and sub-triangle connectivity.

        Parameters
        ----------
        level : int
            Subdivision level per edge. level=1 means no subdivision.

        Returns
        -------
        bcs : (Np, 3)
            Barycentric coordinates on the reference triangle.
        subcell : (Nsub, 3)
            Connectivity of reference sub-triangles using local point indices.
        """
        level = int(max(level, 1))
        idx_map = {}
        bcs = []
        cursor = 0
        for i in range(level + 1):
            for j in range(level + 1 - i):
                k = level - i - j
                bcs.append([i / level, j / level, k / level])
                idx_map[(i, j)] = cursor
                cursor += 1
        bcs = np.asarray(bcs, dtype=float)

        subcell = []
        for i in range(level):
            for j in range(level - i):
                v0 = idx_map[(i, j)]
                v1 = idx_map[(i + 1, j)]
                v2 = idx_map[(i, j + 1)]
                subcell.append([v0, v1, v2])
                if j < level - i - 1:
                    v3 = idx_map[(i + 1, j + 1)]
                    subcell.append([v1, v3, v2])
        subcell = np.asarray(subcell, dtype=np.int64)
        return bcs, subcell

    @staticmethod
    def _eval_on_cells(fun, bcs, NC: int):
        """Evaluate FE function on all cells at given barycentric points."""
        val = fun(bm.asarray(bcs), index=bm.arange(NC))
        val = np.asarray(val)
        if val.ndim == 3 and val.shape[1] == 1:
            val = val[:, 0, ...]
        return val

    def save_vtkfile_highorder(self, fname: str, *, vis_order: int = 4, sigma_eval=None):
        """
        Export high-order visualization by intra-cell resampling + sub-triangulation.

        This writes a linear VTU mesh built from sampled points inside each original
        triangle, so ParaView can show smooth high-order variation without requiring
        high-order VTK cell support.

        Parameters
        ----------
        fname : str
            Output VTU path.
        vis_order : int, optional
            Subdivision level per original cell edge. Higher value gives richer detail.
        sigma_eval : optional
            Optional stress evaluator (defaults to current state stress function).
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        mesh = self.discr.mesh
        state = self.discr.state
        if state is None:
            raise RuntimeError("Discretization state is empty; cannot export VTK.")

        cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
        node = np.asarray(mesh.entity("node"), dtype=float)
        if cell.ndim != 2 or cell.shape[1] != 3:
            raise NotImplementedError("save_vtkfile_highorder currently supports 2D triangles only.")

        NC = int(mesh.number_of_cells())
        GD = int(mesh.geo_dimension())

        bcs, subcell_ref = self._reference_triangle_subdivision(vis_order)
        Np = int(bcs.shape[0])
        Nsub = int(subcell_ref.shape[0])

        v0 = node[cell[:, 0], :]
        v1 = node[cell[:, 1], :]
        v2 = node[cell[:, 2], :]
        p0 = bcs[:, 0][None, :, None]
        p1 = bcs[:, 1][None, :, None]
        p2 = bcs[:, 2][None, :, None]
        xyz = p0 * v0[:, None, :] + p1 * v1[:, None, :] + p2 * v2[:, None, :]
        node_vis = xyz.reshape(NC * Np, GD)

        offset = (np.arange(NC, dtype=np.int64)[:, None] * Np)
        cell_vis = (subcell_ref[None, :, :] + offset[:, None, :]).reshape(NC * Nsub, 3)

        vis_mesh = TriangleMesh(bm.asarray(node_vis), bm.asarray(cell_vis))

        sigma_fun = sigma_eval if sigma_eval is not None else state.sigma

        # Evaluate fields on each sampled point in each original cell.
        try:
            d_samp = self._eval_on_cells(state.d, bcs, NC).reshape(-1)
            vis_mesh.nodedata["damage"] = bm.asarray(d_samp)
        except Exception as e:
            print(f"[vtk-highorder] damage sampling failed: {e}")

        try:
            u_samp = self._eval_on_cells(state.u, bcs, NC)
            if u_samp.ndim == 2:
                u_samp = u_samp[:, :, None]
            u_flat = u_samp.reshape(-1, u_samp.shape[-1])
            vis_mesh.nodedata["uh"] = bm.asarray(u_flat)
            vis_mesh.nodedata["ux"] = bm.asarray(u_flat[:, 0])
            if u_flat.shape[1] > 1:
                vis_mesh.nodedata["uy"] = bm.asarray(u_flat[:, 1])
            if u_flat.shape[1] > 2:
                vis_mesh.nodedata["uz"] = bm.asarray(u_flat[:, 2])
            vis_mesh.nodedata["u_norm"] = bm.asarray(np.linalg.norm(u_flat, axis=1))
        except Exception as e:
            print(f"[vtk-highorder] displacement sampling failed: {e}")

        try:
            sig_samp = self._eval_on_cells(sigma_fun, bcs, NC)
            if sig_samp.ndim == 2:
                sig_samp = sig_samp[:, :, None]
            sig_flat = sig_samp.reshape(-1, sig_samp.shape[-1])
            vis_mesh.nodedata["sigma_node"] = bm.asarray(sig_flat)
            if sig_flat.shape[1] >= 3:
                vis_mesh.nodedata["sigxx"] = bm.asarray(sig_flat[:, 0])
                vis_mesh.nodedata["sigxy"] = bm.asarray(sig_flat[:, 1])
                vis_mesh.nodedata["sig2"] = bm.asarray(sig_flat[:, 2])
                s1, svm = self._voigt2d_principal_and_vm(bm.asarray(sig_flat[:, :3]))
                vis_mesh.nodedata["s1"] = s1
                vis_mesh.nodedata["svm"] = svm
        except Exception as e:
            print(f"[vtk-highorder] stress sampling failed: {e}")

        # Keep quadrature-history output as cell quantity on original mesh if available.
        if state.H is not None:
            try:
                H_cell = np.mean(np.asarray(state.H), axis=1).reshape(-1)
                vis_mesh.celldata["H_cell_parent"] = bm.asarray(np.repeat(H_cell, Nsub))
            except Exception as e:
                print(f"[vtk-highorder] H_cell projection failed: {e}")

        vis_mesh.to_vtk(fname=fname)

    def save_vtkfile_lagrange(
        self,
        fname: str,
        *,
        order: int = 3,
        sigma_eval=None,
        fallback_linear: bool = True,
        fallback_vis_order: Optional[int] = None,
        always_write_fallback: bool = False,
    ):
        """
        Export VTU using native VTK_LAGRANGE_TRIANGLE high-order cells.

        Parameters
        ----------
        fname : str
            Output VTU path.
        order : int, optional
            Lagrange order of output cells (>=1).
        sigma_eval : optional
            Optional stress evaluator; defaults to current state stress function.
        fallback_linear : bool, optional
            If True, write a linear sub-triangulated fallback VTU when Lagrange
            export fails.
        fallback_vis_order : int, optional
            Subdivision order used by fallback writer. Defaults to max(2, order+1).
        always_write_fallback : bool, optional
            If True, always write fallback VTU in addition to Lagrange VTU.
        """
        state = self.discr.state
        if state is None:
            raise RuntimeError("Discretization state is empty; cannot export VTK.")

        sigma_fun = sigma_eval if sigma_eval is not None else state.sigma
        sampled = sample_fields_for_lagrange_triangle(
            mesh=self.discr.mesh,
            order=order,
            field_specs=(
                ("damage", state.d),
                ("uh", state.u),
                ("sigma_node", sigma_fun),
            ),
        )

        # Derived fields for convenient ParaView coloring.
        if sampled["uh"].ndim == 2 and sampled["uh"].shape[1] >= 2:
            u = sampled["uh"]
            sampled["ux"] = u[:, 0]
            sampled["uy"] = u[:, 1]
            sampled["u_norm"] = np.linalg.norm(u[:, :2], axis=1)
            if u.shape[1] > 2:
                sampled["uz"] = u[:, 2]
        if sampled["sigma_node"].ndim == 2 and sampled["sigma_node"].shape[1] >= 3:
            sig = sampled["sigma_node"][:, :3]
            sampled["sigxx"] = sig[:, 0]
            sampled["sigxy"] = sig[:, 1]
            sampled["sig2"] = sig[:, 2]
            s1, svm = self._voigt2d_principal_and_vm(bm.asarray(sig))
            sampled["s1"] = np.asarray(s1)
            sampled["svm"] = np.asarray(svm)

        lag_ok = False
        lag_err = None
        try:
            write_lagrange_triangle_vtu(
                fname=fname,
                mesh=self.discr.mesh,
                order=order,
                point_data=sampled,
            )
            lag_ok = True
        except Exception as e:
            lag_err = e
            print(f"[vtk-lagrange] export failed: {e}")

        need_fallback = bool(always_write_fallback or (fallback_linear and (not lag_ok)))
        if need_fallback:
            vis_order = int(max(1, fallback_vis_order if fallback_vis_order is not None else max(2, order + 1)))
            root, ext = os.path.splitext(fname)
            ext = ext or ".vtu"
            fallback_name = f"{root}.fallback_linear_o{vis_order}{ext}"
            self.save_vtkfile_highorder(
                fallback_name,
                vis_order=vis_order,
                sigma_eval=sigma_eval,
            )
            if lag_ok:
                print(f"[vtk-lagrange] fallback also written: {fallback_name}")
            else:
                print(f"[vtk-lagrange] fallback written due to lagrange failure: {fallback_name}")
                if lag_err is not None:
                    print(f"[vtk-lagrange] original error: {lag_err}")