# fracturex/drivers/huzhang_phasefield_staggered.py
"""Hu-Zhang 混合元 + 相场损伤的交错（staggered）准静态求解驱动。

按载荷步推进，每步在弹性子问题（Hu-Zhang 应力/位移）与相场子问题之间交错迭代至收敛，
负责：装配器/线性求解器调度、收敛判据、损伤不可逆与欠松弛、反力后处理、记录器与多档
VTK 输出（普通/高阶重采样/原生 Lagrange）。

线性求解器通过 ``linear_solver(method)`` 选择（spsolve/pardiso/mumps/lgmres）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List

import functools
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse.linalg import spsolve, lgmres

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.utils import timer

from fracturex.cases.base import CaseBase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.base import DamageStateView, DamageModelBase
from fracturex.drivers.anderson_acceleration import AndersonAccelerator
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.postprocess.reaction import reaction_from_sigma
from fracturex.postprocess.run_paths import resolve_vtk_step_path
from fracturex.utilfuc.sparse_direct_backends import solve_direct_mumps, solve_direct_pardiso
from fracturex.utilfuc.vtk_lagrange_writer import (
    sample_fields_for_lagrange_triangle,
    write_lagrange_triangle_vtu,
)


@dataclass
class StepInfo:
    """单个载荷步的求解结果摘要。

    Attributes:
        step: 载荷步序号。
        load: 该步规定载荷/位移值。
        iters: 交错迭代次数。
        converged: 是否收敛。
        err_u, err_d: 位移、损伤的相对收敛误差。
        max_d: 该步结束时的最大损伤值。
        meta: 详细元信息（反力、各阶段耗时、线性求解诊断等）。
    """

    step: int
    load: float
    iters: int
    converged: bool
    err_u: float
    err_d: float
    max_d: float
    meta: Dict[str, Any]


class HuZhangPhaseFieldStaggeredDriver:
    """Hu-Zhang 混合元 + 相场损伤的交错求解驱动器。

    组合弹性装配器、相场装配器与损伤模型，按载荷序列逐步交错求解并输出 :class:`StepInfo`。
    """

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
        output_dir: Optional[str] = None,
        save_vtu_per_step: bool = False,
        stagger_print_interval: Optional[int] = None,
        d_relaxation: Optional[float] = None,
    ):
        """Initialize staggered HuZhang + phase-field solver driver.

        Inputs:
            case/discr/damage: Core case, discretization and damage model objects.
            elastic_assembler/phase_assembler: Optional custom assemblers.
            tol/maxit: Nonlinear staggered convergence tolerance and max iterations.
            compute_linear_residual: Whether to compute `||Ax-b||/||b||` diagnostics.
            elastic_solver/phase_solver: Linear solver callbacks `(A, b) -> x` or `(x, info)`.
                Use :meth:`linear_solver` with ``'pardiso'`` (``pypardiso``) or ``'mumps'``
                (``python-mumps``) for direct solves, same backends as ``MainSolve``.
            adapt_hook: Optional callback after each load step.
            cell_mode/debug/timing/recorder: Output and diagnostics switches.
            output_dir: Directory for per-step VTK (``<dir>/vtk/step_XXX.vtu``).
                Defaults to ``recorder.outdir`` when omitted.
            save_vtu_per_step: Write VTK after each load step when an output dir exists.
                Legacy: also enabled when ``case.output_enabled`` is True.
            stagger_print_interval: Nonlinear staggered progress print stride.
                ``None`` reads env ``FRACTUREX_STAGGER_PRINT_INTERVAL`` (default ``1``).
                ``<= 0``: suppress periodic prints; still prints when ``debug`` is True,
                when the step converges, or on the last iteration if ``maxit`` is reached.
                ``N >= 1``: print every ``N`` iterations and always on converge / last iter.
            d_relaxation: Under-relaxation for the damage field in ``(0, 1]``.
                Update uses ``max(d_old, ω·d_trial + (1-ω)·d_old)`` then clip to ``[0,1]``.
                ``ω=1`` recovers the original map; ``ω<1`` dampens oscillations when ``d→1``.
                ``None`` reads env ``FRACTUREX_D_RELAXATION`` (default ``1.0``).
        Output:
            None. Stores configuration; real initialization happens in `initialize`.
        """
        self.case = case
        self.discr = discr
        self.damage = damage
        self.recorder = recorder
        self.output_dir = str(output_dir) if output_dir else None
        self.save_vtu_per_step = bool(save_vtu_per_step)

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
        # Unified default for phase-field subproblem: no-preconditioner LGMRES.
        self.phase_solver = phase_solver if phase_solver is not None else self._default_lgmres

        self.adapt_hook = adapt_hook
        self._stagger_print_interval = self._resolve_stagger_print_interval(stagger_print_interval)
        self._d_relaxation = self._resolve_d_relaxation(d_relaxation)
        # Optional Storvik-style Anderson acceleration of the staggered
        # fixed point (FRACTUREX_ANDERSON_DEPTH>0 enables; 0 = legacy path).
        try:
            self._anderson_depth = int(os.environ.get("FRACTUREX_ANDERSON_DEPTH", "0"))
        except (TypeError, ValueError):
            self._anderson_depth = 0
        try:
            self._anderson_beta = float(os.environ.get("FRACTUREX_ANDERSON_BETA", "1.0"))
        except (TypeError, ValueError):
            self._anderson_beta = 1.0
        try:
            self._anderson_omega = float(os.environ.get("FRACTUREX_ANDERSON_OMEGA", "1.0"))
        except (TypeError, ValueError):
            self._anderson_omega = 1.0
        # Safeguard 调参旋钮(默认即 AndersonAccelerator 的默认值):信赖域系数、
        # blowup restart 倍数、停滞 restart 步数。仅 DEPTH>0 时才用到。
        try:
            self._anderson_tr_factor = float(os.environ.get("FRACTUREX_ANDERSON_TR_FACTOR", "20.0"))
        except (TypeError, ValueError):
            self._anderson_tr_factor = 20.0
        try:
            self._anderson_blowup = float(os.environ.get("FRACTUREX_ANDERSON_BLOWUP", "2.0"))
        except (TypeError, ValueError):
            self._anderson_blowup = 2.0
        try:
            self._anderson_patience = int(os.environ.get("FRACTUREX_ANDERSON_PATIENCE", "3"))
        except (TypeError, ValueError):
            self._anderson_patience = 3
        self._anderson = None
        self._initialized = False
        self._sigma_physical_eval = None
        self._tmr = timer() if self.timing else None
        if self._tmr is not None:
            next(self._tmr)

    @staticmethod
    def _resolve_stagger_print_interval(explicit: Optional[int]) -> int:
        """解析交错进度打印间隔：显式值优先，否则读环境变量，缺省 1。返回整数。"""
        if explicit is not None:
            return int(explicit)
        raw = os.environ.get("FRACTUREX_STAGGER_PRINT_INTERVAL", "1")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _resolve_d_relaxation(explicit: Optional[float]) -> float:
        """解析损伤欠松弛因子 ω∈(0,1]：显式优先，否则读环境变量，缺省 1.0。返回浮点。"""
        if explicit is not None:
            omega = float(explicit)
        else:
            raw = os.environ.get("FRACTUREX_D_RELAXATION", "1.0")
            try:
                omega = float(raw)
            except (TypeError, ValueError):
                omega = 1.0
        if omega <= 0.0:
            raise ValueError(f"d_relaxation must be in (0, 1], got {omega}")
        return min(omega, 1.0)

    def _timer_mark(self, tag: Optional[str]):
        """向计时器发送阶段标记 ``tag``（计时关闭或失败时安全跳过）。"""
        if self._tmr is None:
            return
        try:
            self._tmr.send(tag)
        except Exception:
            pass

    @staticmethod
    def _relative_residual(A, x, b) -> float:
        """计算相对残差 ``||Ax-b|| / ||b||``；异常时返回 ``nan``。"""
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
        """返回线性求解器可调用对象的名字（``__name__`` 或类名），用于日志/记录。"""
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
            if hasattr(self.recorder, "save_mesh"):
                # New in 2026-05: persist mesh + space orders so dataset_export
                # can rebuild the discretization without a case instance.
                # Older recorders without the method silently skip.
                try:
                    self.recorder.save_mesh(self.discr)
                except Exception as exc:
                    if self.debug:
                        print(f"[driver] save_mesh failed: {exc}")

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

        # Reset Anderson window per load step (history is intra-step only).
        if self._anderson_depth > 0:
            self._anderson = AndersonAccelerator(
                depth=self._anderson_depth,
                beta=self._anderson_beta,
                omega=self._anderson_omega,
                restart_patience=self._anderson_patience,
                blowup_factor=self._anderson_blowup,
                tr_factor=self._anderson_tr_factor,
            )
        else:
            self._anderson = None

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
        gdof_sigma = int(discr.gdof_sigma)
        gdof_u = int(discr.gdof_u)
        gdof_d = int(discr.space_d.number_of_global_dofs())
        u_old_buf = None
        d_old_buf = None

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
            u_curr = bm.asarray(state.u[:])
            d_curr = bm.asarray(state.d[:])
            if u_old_buf is None:
                u_old_buf = u_curr.copy()
            else:
                u_old_buf[...] = u_curr
            if d_old_buf is None:
                d_old_buf = d_curr.copy()
            else:
                d_old_buf[...] = d_curr
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

            d_trial = bm.asarray(sys_d.decode(dd))

            # Plain (un-accelerated) fixed-point image G(d_old): irreversibility
            # + optional under-relaxation, then clip. This is what the staggered
            # convergence residual is measured against (see d_plain below), so an
            # Anderson step can never produce *false* convergence.
            omega = self._d_relaxation
            if omega >= 1.0 - 1e-15:
                d_blend_plain = d_trial
            else:
                d_blend_plain = omega * d_trial + (1.0 - omega) * d_old_buf
            d_plain = bm.clip(bm.maximum(d_old_buf, d_blend_plain), 0.0, 1.0)

            if self._anderson is not None:
                # Anderson accelerates the iterate (x=d_old -> G(x)=d_trial); the
                # projection (irreversibility + clip) is reapplied to the
                # accelerated combination and carried forward as the next iterate.
                d_acc = self._anderson.step(np.asarray(d_old_buf), np.asarray(d_trial))
                d_acc = bm.asarray(d_acc).reshape(bm.asarray(d_trial).shape)
                state.d[:] = bm.clip(bm.maximum(d_old_buf, d_acc), 0.0, 1.0)
            else:
                state.d[:] = d_plain

            # ----------------------------------------------------------
            # (3) staggered convergence check by iterate increments
            # ----------------------------------------------------------
            u_curr = bm.asarray(state.u[:])
            d_curr = bm.asarray(state.d[:])
            du_abs = float(bm.linalg.norm(u_curr - u_old_buf))
            # True staggered fixed-point residual on d uses the UN-accelerated
            # projected image d_plain, so convergence reflects ||G(d)-d|| rather
            # than the (possibly small) accelerated increment.
            dd_abs = float(bm.linalg.norm(d_plain - d_old_buf))

            # Follow phasefield/main_solve convergence style:
            # normalize by the first iteration increment in each load step.
            # Add a tiny, state-scaled lower bound to avoid pathological
            # denominator collapse when the first increment is numerically tiny.
            if e0_u is None:
                u_scale = float(bm.linalg.norm(u_curr))
                e0_u = max(du_abs, 1e-14 * u_scale, 1e-30)
            if e0_d is None:
                d_scale = float(bm.linalg.norm(d_curr))
                e0_d = max(dd_abs, 1e-14 * d_scale, 1e-30)

            err_u = du_abs / e0_u
            err_d = dd_abs / e0_d
            error = max(err_u, err_d)
            max_d_iter = float(bm.max(d_curr))

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
                    d_relaxation=float(self._d_relaxation),
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

            iter_num = k + 1
            last_iter = k == self.maxit - 1
            converged_now = error < self.tol
            if self.debug:
                should_print_progress = True
            elif self._stagger_print_interval <= 0:
                should_print_progress = converged_now or last_iter
            else:
                should_print_progress = (
                    (iter_num % self._stagger_print_interval == 0) or converged_now or last_iter
                )
            if should_print_progress:
                print(
                    f"[step {step} load {load:.4e}] iter {iter_num}: "
                    f"du_abs={du_abs:.3e}, dd_abs={dd_abs:.3e}, "
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
        want_vtu = self.save_vtu_per_step or getattr(self.case, "output_enabled", False)
        if want_vtu:
            vtk_fname = resolve_vtk_step_path(
                step=step,
                output_dir=self.output_dir,
                recorder=self.recorder,
            )
            if vtk_fname:
                os.makedirs(os.path.dirname(vtk_fname), exist_ok=True)
                self._save_vtkfile(
                    vtk_fname,
                    cell_mode=self.cell_mode,
                    q=q,
                    sigma_eval=self._sigma_physical_eval,
                )
            elif getattr(self.case, "output_enabled", False):
                print(
                    "[HuZhangPhaseFieldStaggeredDriver] output_enabled=True but no "
                    "output_dir/recorder.outdir; skip VTK (set output_dir or RunRecorder)."
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
        d_final = bm.asarray(state.d[:])

        meta = dict(
            gdof_sigma=gdof_sigma,
            gdof_u=gdof_u,
            gdof_d=gdof_d,
            elastic_formulation=str(getattr(self.elastic_assembler, "formulation", "standard")),
            history_source=str(getattr(self.damage, "history_source", "from_u")),
            max_d=float(bm.max(d_final)),
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
            d_relaxation=float(self._d_relaxation),
            stagger_tol=float(self.tol),
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
            max_d=float(bm.max(d_final)),
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
            if hasattr(self.recorder, "dump_quadrature_fields"):
                try:
                    self.recorder.dump_quadrature_fields(step, discr, state)
                except Exception as exc:
                    if self.debug:
                        print(f"[driver] dump_quadrature_fields failed: {exc}")

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

    @classmethod
    def linear_solver(cls, method: str, **kwargs) -> Callable[[Any, Any], Any]:
        """Build a ``(A, F) -> x`` callback for ``elastic_solver`` / ``phase_solver``.

        ``method`` is one of ``spsolve`` / ``direct``, ``pardiso``, ``mumps``, ``lgmres``.
        Extra keyword arguments apply only to ``lgmres`` (see :meth:`_default_lgmres`).
        """
        m = (method or "").strip().lower()
        if m in ("spsolve", "direct", "superlu"):
            return cls._default_spsolve
        if m == "pardiso":
            return solve_direct_pardiso
        if m == "mumps":
            return solve_direct_mumps
        if m == "lgmres":
            return functools.partial(cls._default_lgmres, **kwargs)
        raise ValueError(
            f"Unknown linear solver method {method!r}; "
            "expected one of: spsolve, direct, pardiso, mumps, lgmres."
        )

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

    @staticmethod
    def _vtk_highorder_parallel_enabled() -> bool:
        """是否启用高阶 VTK 重采样的并行（环境变量 ``FRACTUREX_VTK_HIGHORDER_PARALLEL``）。"""
        return str(os.getenv("FRACTUREX_VTK_HIGHORDER_PARALLEL", "1")).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def _eval_on_cells_parallel_default(self, fun, bcs, NC: int):
        """Chunked FE evaluation over cells; **on by default** (disable via ``FRACTUREX_VTK_HIGHORDER_PARALLEL=0``).

        Intended for visualization-only paths; wall time here is **not** part of the
        core staggered-solve benchmark used in papers or solver-efficiency tables.
        """
        if (
            not HuZhangPhaseFieldStaggeredDriver._vtk_highorder_parallel_enabled()
            or NC < 48
        ):
            return HuZhangPhaseFieldStaggeredDriver._eval_on_cells(fun, bcs, NC)
        max_w = max(1, int(os.cpu_count() or 1))
        nproc = min(max_w, max(2, NC // 24))
        edges = np.linspace(0, NC, nproc + 1, dtype=int)
        tasks = []
        for k in range(nproc):
            i0, i1 = int(edges[k]), int(edges[k + 1])
            if i1 > i0:
                tasks.append((fun, bcs, i0, i1))

        def _job(args):
            fn, bcs_loc, i0, i1 = args
            chunk = fn(bm.asarray(bcs_loc), index=bm.arange(i0, i1))
            chunk = np.asarray(chunk)
            if chunk.ndim == 3 and chunk.shape[1] == 1:
                chunk = chunk[:, 0, ...]
            return chunk

        workers = min(max_w, len(tasks))
        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                parts = list(pool.map(_job, tasks))
            return np.concatenate(parts, axis=0)
        except Exception:
            return HuZhangPhaseFieldStaggeredDriver._eval_on_cells(fun, bcs, NC)

    def save_vtkfile_highorder(self, fname: str, *, vis_order: int = 4, sigma_eval=None):
        """
        Export high-order visualization by intra-cell resampling + sub-triangulation.

        This writes a linear VTU mesh built from sampled points inside each original
        triangle, so ParaView can show smooth high-order variation without requiring
        high-order VTK cell support.

        Field sampling over cells uses **parallel chunks by default** (see
        ``FRACTUREX_VTK_HIGHORDER_PARALLEL``); that cost is visualization-only and
        should **not** be mixed into core solve/assembly timing for papers.

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
            d_samp = self._eval_on_cells_parallel_default(state.d, bcs, NC).reshape(-1)
            vis_mesh.nodedata["damage"] = bm.asarray(d_samp)
        except Exception as e:
            print(f"[vtk-highorder] damage sampling failed: {e}")

        try:
            u_samp = self._eval_on_cells_parallel_default(state.u, bcs, NC)
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
            sig_samp = self._eval_on_cells_parallel_default(sigma_fun, bcs, NC)
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