# fracturex/drivers/huzhang_damage_staggered.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List

import numpy as np
import os
import time
from scipy.sparse.linalg import spsolve, lgmres
from fealpy.utils import timer

from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.damage.base import DamageStateView, DamageModelBase
from fracturex.cases.base import CaseBase
from fracturex.postprocess.reaction import reaction_from_sigma

from fealpy.backend import backend_manager as bm



@dataclass
class StepInfo:
    step: int
    load: float
    iters: int
    converged: bool
    max_dd: float
    max_d: float
    meta: Dict[str, Any]


class HuZhangLocalDamageStaggeredDriver:
    """
    staggered 求解：
      给定 load(step):
        (1) 用当前 d 装配并解 (sigma,u)
        (2) 用 (sigma,u) 更新 r_hist,d
        (3) 循环直到 max(d^{k+1}-d^k) < tol

    预留：
    - adapt_hook: 每个 load step 后可做自适应（重建 discr 并 transfer state）
    - solver_hook: 替换线性求解器/预条件
    """

    def __init__(
        self,
        *,
        case: CaseBase,
        discr: HuZhangDiscretization,
        damage: DamageModelBase,
        assembler: Optional[HuZhangElasticAssembler] = None,
        tol: float = 1e-4,
        maxit: int = 30,
        linear_solver: Optional[Callable[[Any, Any], bm.ndarray]] = None,
        adapt_hook: Optional[Callable[[int, float, HuZhangDiscretization, DamageModelBase], None]] = None,
        cell_mode: str = "mean",
        debug: bool = False,
        timing: bool = False,
        recorder: Optional[Any] = None,
    ):
        self.case = case
        self.discr = discr
        self.damage = damage

        self.recorder = recorder

        self.assembler = assembler if assembler is not None else HuZhangElasticAssembler(discr, case, damage)
        self.tol = float(tol)
        self.maxit = int(maxit)
        self.debug = bool(debug)
        self.timing = bool(timing)
        self.cell_mode = cell_mode

        self.linear_solver = linear_solver if linear_solver is not None else self._default_spsolve
        self.adapt_hook = adapt_hook
        self._tmr = timer()
        next(self._tmr)

        # init damage model (once after build)
        self._initialized = False

    def _timer_mark(self, tag: Optional[str]):
        if not self.timing:
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

    def initialize(self):
        """call once after discr.build()"""
        if self._initialized:
            return
        state = self.discr.state
        if state is None:
            raise RuntimeError("Discretization must be built before driver.initialize().")
        view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
        self.damage.on_build(self.discr, view, self.case)
        self._initialized = True

    # --------------------------
    # main loop
    # --------------------------
    def run(self, loads: List[float]) -> List[StepInfo]:
        self.initialize()
        out: List[StepInfo] = []

        if self.recorder is not None:
            m = self.case.model()
            meta = dict(
                case=self.case.name,
                p=int(self.discr.p),
                use_relaxation=bool(self.discr.use_relaxation),
                mesh=dict(
                    NN=int(self.discr.mesh.number_of_nodes()),
                    NE=int(self.discr.mesh.number_of_edges()),
                    NC=int(self.discr.mesh.number_of_cells()),
                ),
                # 材料常数放这里即可（ft 不需要 step 记录）
                material={k: float(getattr(m, k)) for k in ["lam","mu","E","nu","Gc","ft"] if hasattr(m, k)},
                timing_enabled=bool(self.timing),
            )
            self.recorder.write_meta(meta)


        for s, load in enumerate(loads):
            info = self.solve_one_step(step=s, load=float(load))
            out.append(info)

            # optional adaptivity hook (after converged step)
            if self.adapt_hook is not None:
                self.adapt_hook(s, float(load), self.discr, self.damage)

        return out

    def solve_one_step(self, *, step: int, load: float) -> StepInfo:
        discr = self.discr
        state = discr.state
        if state is None:
            raise RuntimeError("Discretization must be built before solving.")

        converged = False
        max_dd = bm.inf
        t_step0 = time.perf_counter()
        t_elastic_assemble = 0.0
        t_elastic_solve = 0.0
        t_damage_update = 0.0
        r_lin_e = float("nan")

        # staggered iteration
        for k in range(self.maxit):
            t_iter0 = time.perf_counter()
            d_old = bm.asarray(state.d[:]).copy()

            # (1) assemble and solve elastic system with current d
            t0 = time.perf_counter()
            self._timer_mark("elastic_assemble_start")
            sys = self.assembler.assemble(load)
            self._timer_mark("elastic_assemble_end")
            t_elastic_assemble += time.perf_counter() - t0

            t0 = time.perf_counter()
            self._timer_mark("elastic_solve_start")
            X = self.linear_solver(sys.A, sys.F)
            self._timer_mark("elastic_solve_end")
            t_elastic_solve += time.perf_counter() - t0
            r_lin_e = self._relative_residual(sys.A, X, sys.F)

            sigma, u = sys.decode(X)

            state.sigma[:] = sigma[:]
            state.u[:] = u[:]


            # (2) update damage (and history) using (sigma,u)
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
            t0 = time.perf_counter()
            self._timer_mark("damage_update_start")
            self.damage.update_after_elastic(discr, view, self.case)
            self._timer_mark("damage_update_end")
            t_damage_update += time.perf_counter() - t0

            d_new = bm.asarray(state.d[:])
            max_dd = float(bm.max(bm.abs(d_new - d_old)))

            if self.recorder is not None and hasattr(self.recorder, "append_iteration"):
                self.recorder.append_iteration(dict(
                    step=int(step),
                    load=float(load),
                    iter=int(k + 1),
                    max_dd=float(max_dd),
                    max_d=float(bm.max(d_new)),
                    linear_res_elastic=float(r_lin_e),
                    t_iter_s=float(time.perf_counter() - t_iter0),
                ))

            if self.debug:
                print(f"[step {step} load {load:.4e}] iter {k}: max_dd={max_dd:.3e}, max_d={float(bm.max(d_new)):.3e}")

            if max_dd < self.tol:
                converged = True
                iters = k + 1
                break
        else:
            iters = self.maxit

        q = self.discr.p + 3
       
        # ---- output vtu (可选) ----
        if self.case.output_enabled:
            self._save_vtkfile(
                f"results/{self.case.name}_step_{step:03d}.vtu",
                cell_mode=self.cell_mode, q=q
            )

        # ---- reaction on load boundary (只算一次) ----
        lp = self.case.load_dirichlet_piece(load)
        dir_load = (lp.direction or "y")
        if dir_load not in ("x", "y", "z"):
            dir_load = "y"  # 默认 y 方向

        R = reaction_from_sigma(
            self.discr.mesh,
            state.sigma,
            lp.threshold,
            direction=dir_load,
            q=q,           
            sign=-1,     
        )

        step_walltime = float(time.perf_counter() - t_step0)

        # ---- meta（方向通用字段）----
        meta = dict(
            gdof_sigma=int(discr.gdof_sigma),
            gdof_u=int(discr.gdof_u),
            max_d=float(bm.max(bm.asarray(state.d[:]))),
            R=float(R),
            residual_force=float(R),
            R_dir=dir_load,
            load=float(load),
            load_dir=dir_load,
            linear_res_elastic=float(r_lin_e),
            t_step_s=step_walltime,
            t_elastic_assemble_s=float(t_elastic_assemble),
            t_elastic_solve_s=float(t_elastic_solve),
            t_damage_update_s=float(t_damage_update),
        )

        # 
        meta[f"reaction_{dir_load}"] = float(R)
        meta[f"disp_{dir_load}"] = float(load)

        info = StepInfo(
            step=step,
            load=float(load),
            iters=int(iters),
            converged=converged,
            max_dd=float(max_dd),
            max_d=float(bm.max(bm.asarray(state.d[:]))),
            meta=meta,
        )

        # ---- history / checkpoint（也只写一次）----
        if self.recorder is not None:
            row = dict(
                step=int(step),
                load=float(load),
                iters=int(iters),
                converged=bool(converged),
                max_dd=float(max_dd),
                max_d=float(bm.max(bm.asarray(state.d[:]))),
                R=float(R),
                residual_force=float(R),
                R_dir=dir_load,
                linear_res_elastic=float(r_lin_e),
                t_step_s=step_walltime,
                t_elastic_assemble_s=float(t_elastic_assemble),
                t_elastic_solve_s=float(t_elastic_solve),
                t_damage_update_s=float(t_damage_update),
                **{f"reaction_{dir_load}": float(R), f"disp_{dir_load}": float(load)},
            )
            self.recorder.append_history(row)
            self.recorder.save_checkpoint(step, discr, state)

        return info

    
    # --------------------------
    # output
    # --------------------------
    def _save_vtkfile(self, fname: str, *, cell_mode="mean", q=None):
        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
        discr = self.discr
        state = discr.state
        mesh = discr.mesh
        q = q if q is not None else discr.p + 3

        node = mesh.entity("node")          # (NN,2)
        NN = mesh.number_of_nodes()
        GD = mesh.geo_dimension()

        # ---- nodal: damage ----
        mesh.nodedata["damage"] = state.d

        # ---- nodal: displacement ----
        #mesh.nodedata["displacement"] = state.u.reshape(GD, -1).T  # (NN,GD)
        #mesh.nodedata["displacement"] = state.u.reshape(GD, -1)  # (NN,GD)
        uc = self.u_cell_bary(state.u, mesh)
        un = self.u_to_nodal(state.u, mesh)

        mesh.nodedata["displacement"] = un  # (NN,GD)
        mesh.celldata["displacement_cell"] = uc  # (NC,GD)


        # ---- cell data ----
        self.attach_cell_sigma_damage(
            mesh=mesh,
            sigma_fun=state.sigma,
            d_fun=None,                 # cell damage 可选（下面函数里也支持）
            cell_mode=cell_mode,
            q=q,
            prefix=""
        )

        mesh.to_vtk(fname=fname)


    def u_cell_bary(self, u_fun, mesh):
        """
        计算单元重心处的位移值，用于后处理。
        u_fun: Callable[[barycentric coords, index], values]
        mesh: TriangleMesh

        Returns:
            vals: (NC, GD) array of displacement at cell barycenters
        """
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        # 三角形重心 barycentric
        bcs = bm.array([[1/3, 1/3, 1/3]], dtype=bm.float64)  # (1,3)
        vals = u_fun(bcs, index=bm.arange(NC))               # (NC,1,GD) 或 (NC,GD)
        if vals.ndim == 3:
            vals = vals[:, 0, :]
        return vals  # (NC,GD)
    
    def u_to_nodal(self, u_fun, mesh):
        """
        将单元上的函数值 u_fun 转换为节点值（简单平均）。
        u_fun: Callable[[barycentric coords, index], values]
        mesh: TriangleMesh
        Returns:
            vals: (NN, GD) array of displacement at nodes
        """
        cell = mesh.entity("cell")          # (NC,3)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()

        bcs = bm.array([[1,0,0],[0,1,0],[0,0,1]], dtype=bm.float64)  # 三顶点 barycentric

        vals = u_fun(bcs, index=bm.arange(NC))   # 期望 (NC,3,GD)
        if vals.ndim == 2:                       # 兼容某些实现返回 (NC,GD)
            vals = vals[:, None, :].repeat(3, axis=1)

        out = bm.zeros((NN, GD), dtype=vals.dtype)
        cnt = bm.zeros((NN,), dtype=vals.dtype)

        for lv in range(3):
            nid = cell[:, lv]
            bm.add.at(out, nid, vals[:, lv, :])
            bm.add.at(cnt, nid, 1.0)

        out = out / cnt[:, None]
        return out  # (NN,GD)


    @staticmethod
    def attach_cell_sigma_damage(*, mesh, sigma_fun, d_fun=None,
                                cell_mode: str = "mean", q: int | None = None,
                                prefix: str = ""):
        NC = mesh.number_of_cells()

        if cell_mode == "barycenter":
            bcs = bm.asarray([[1/3, 1/3, 1/3]], dtype=mesh.ftype)  # (1,3)
            ws = None
            wsum = None
        elif cell_mode == "mean":
            if q is None:
                q = 6
            qf = mesh.quadrature_formula(q, etype="cell")
            bcs, ws = qf.get_quadrature_points_and_weights()       # (NQ,3),(NQ,)
            wsum = bm.sum(ws)
        else:
            raise ValueError("cell_mode must be 'mean' or 'barycenter'")

        # sigma_fun(bcs) 期望返回 (NC,NQ,3) 或 (NC,1,3)
        sig = bm.asarray(sigma_fun(bcs))
        if cell_mode == "mean":
            sig_mean = bm.einsum("q,cqk->ck", ws, sig[..., :3]) / wsum   # (NC,3)
        else:
            sig_mean = sig[:, 0, :3]                                     # (NC,3)

        sxx = sig_mean[:, 0]
        sxy = sig_mean[:, 1]
        syy = sig_mean[:, 2]

        # 1) voigt
        mesh.celldata[f"{prefix}sigma_voigt"] = sig_mean  # (NC,3) OK

        # 2) 拆分量（最稳）
        mesh.celldata[f"{prefix}sigma_xx"] = sxx
        mesh.celldata[f"{prefix}sigma_xy"] = sxy
        mesh.celldata[f"{prefix}sigma_yy"] = syy

        # 3) von Mises (2D)
        svm = bm.sqrt(bm.maximum(sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy, 0.0))
        mesh.celldata[f"{prefix}sigma_vm"] = svm

        # 4) 如果你一定想写“tensor”，写成 (NC,9) 而不是 (NC,3,3)
        ten9 = bm.zeros((NC, 9), dtype=sig_mean.dtype)
        ten9[:, 0] = sxx   # (0,0)
        ten9[:, 1] = sxy   # (0,1)
        ten9[:, 3] = sxy   # (1,0)
        ten9[:, 4] = syy   # (1,1)
        mesh.celldata[f"{prefix}sigma_9"] = ten9  # (NC,9) OK

        # ---- damage cell mean (可选) ----
        if d_fun is not None:
            dval = bm.asarray(d_fun(bcs))
            if cell_mode == "mean":
                d_mean = bm.einsum("q,cq->c", ws, dval) / wsum
            else:
                d_mean = dval[:, 0]
            mesh.celldata[f"{prefix}damage"] = d_mean


    # --------------------------
    # solver
    # --------------------------
    @staticmethod
    def _default_spsolve(A, F) -> bm.ndarray:
        return spsolve(A, F)
    
    @staticmethod
    def _lgmres_solver(A, F) -> bm.ndarray:
        if not np.isfinite(F).all():
            raise RuntimeError("RHS has NaN/Inf")

        X, info = lgmres(A, F)
        if info != 0:
            X = spsolve(A, F)
            raise RuntimeError(f"lgmres did not converge, info={info}")
            
        return X

