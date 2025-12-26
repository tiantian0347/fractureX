# fracturex/drivers/huzhang_damage_staggered.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List

import numpy as np
from scipy.sparse.linalg import spsolve

from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.damage.base import DamageStateView, DamageModelBase
from fracturex.cases.base import CaseBase
from fracturex.postprocess.reaction import reaction_force_y_from_sigma



@dataclass
class StepInfo:
    step: int
    load: float
    iters: int
    converged: bool
    max_dd: float
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
        tol: float = 1e-8,
        maxit: int = 30,
        linear_solver: Optional[Callable[[Any, Any], np.ndarray]] = None,
        adapt_hook: Optional[Callable[[int, float, HuZhangDiscretization, DamageModelBase], None]] = None,
        cell_mode: str = "mean",
        debug: bool = False,
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
        self.cell_mode = cell_mode

        self.linear_solver = linear_solver if linear_solver is not None else self._default_spsolve
        self.adapt_hook = adapt_hook

        # init damage model (once after build)
        self._initialized = False

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
        max_dd = np.inf

        # staggered iteration
        for k in range(self.maxit):
            d_old = np.asarray(state.d[:]).copy()

            # (1) assemble and solve elastic system with current d
            sys = self.assembler.assemble(load)
            X = self.linear_solver(sys.A, sys.F)
            sigma, u = sys.decode(X)

            state.sigma[:] = sigma[:]
            state.u[:] = u[:]

            # (2) update damage (and history) using (sigma,u)
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
            self.damage.update_after_elastic(discr, view, self.case)

            d_new = np.asarray(state.d[:])
            max_dd = float(np.max(np.abs(d_new - d_old)))

            if self.debug:
                print(f"[step {step} load {load:.4e}] iter {k}: max_dd={max_dd:.3e}, max_d={float(np.max(d_new)):.3e}")

            if max_dd < self.tol:
                converged = True
                iters = k + 1
                break
        else:
            iters = self.maxit

        q = self.discr.p + 3
        # ---- reaction on load boundary (for paper comparison) ----
        thr_load = self.case.load_boundary_threshold()
        qR = int(2 * q)  # 稳妥起见用更高阶积分
        Ry = reaction_force_y_from_sigma(self.discr.mesh, state.sigma, thr_load, q=qR, sign=-1.0)


        if self.case.output_enabled:
            self._save_vtkfile(f"results/{self.case.name}_step_{step:03d}.vtu", cell_mode=self.cell_mode, q=q)



        meta = dict(
            gdof_sigma=int(discr.gdof_sigma),
            gdof_u=int(discr.gdof_u),
        )

        meta["max_d"] = float(np.max(np.asarray(state.d[:])))
        meta["reaction_y"] = float(Ry)
        meta["disp_y"] = float(load)

        info = StepInfo(
            step=step,
            load=float(load),
            iters=int(iters),
            converged=converged,
            max_dd=float(max_dd),
            meta=meta,
        )
        if self.recorder is not None:
            row = dict(
                step=int(step),
                load=float(load),
                iters=int(iters),
                converged=bool(converged),
                max_dd=float(max_dd),
                max_d=float(np.max(np.asarray(state.d[:]))),
                reaction_y=float(Ry),
                disp_y=float(load),
                # reaction / energy 后续你加字段就行
            )
            self.recorder.append_history(row)
            self.recorder.save_checkpoint(step, discr, state)


        return info
    
    # --------------------------
    # output
    # --------------------------
    def _save_vtkfile(self, fname: str, *, cell_mode="mean", q=None):
        discr = self.discr
        state = discr.state
        mesh = discr.mesh
        q = q if q is not None else discr.p + 3

        # ---- nodal data: 直接塞 function（fealpy writer 会采样/处理）----
        mesh.nodedata["damage"] = state.d
        mesh.nodedata["uh"] = state.u

        # ---- cell data: 我们自己做 cell mean / barycenter ----
        self.attach_cell_sigma_damage(
            mesh=mesh,
            sigma_fun=state.sigma,
            d_fun=state.d,
            cell_mode=cell_mode,
            q=q,
            prefix=""
        )

        mesh.to_vtk(fname=fname)


    @staticmethod
    def attach_cell_sigma_damage(*, mesh, sigma_fun, d_fun=None,
                                cell_mode: str = "mean", q: int | None = None,
                                prefix: str = ""):
        """
        cell_mode:
        - "barycenter": 用单元重心（barycentric (1/3,1/3,1/3)）
        - "mean": 用单元积分平均（cell quadrature in barycentric）
        sigma_fun 返回 Voigt [sxx,sxy,syy]（你的 HuZhang 里就是这样）
        """
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

        # ---------- sigma ----------
        sig = bm.asarray(sigma_fun(bcs))  # mean: (NC,NQ,3)  bary: (NC,1,3)
        if cell_mode == "mean":
            sig_mean = bm.einsum("q,cqk->ck", ws, sig[..., :3]) / wsum
        else:
            sig_mean = sig[:, 0, :3]

        mesh.celldata[f"{prefix}sigma_voigt"] = sig_mean

        # tensor 3x3（2D 嵌入）ParaView 可直接显示 tensor
        sxx, sxy, syy = sig_mean[:, 0], sig_mean[:, 1], sig_mean[:, 2]
        ten = bm.zeros((NC, 3, 3), dtype=sig_mean.dtype)
        ten[:, 0, 0] = sxx
        ten[:, 0, 1] = sxy
        ten[:, 1, 0] = sxy
        ten[:, 1, 1] = syy
        mesh.celldata[f"{prefix}sigma"] = ten

        # von Mises（2D 常用）
        svm = bm.sqrt(bm.maximum(sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy, 0.0))
        mesh.celldata[f"{prefix}sigma_vm"] = svm

        # ---------- damage (cell) ----------
        if d_fun is not None:
            dval = bm.asarray(d_fun(bcs))  # mean: (NC,NQ) bary: (NC,1)
            if cell_mode == "mean":
                d_mean = bm.einsum("q,cq->c", ws, dval) / wsum
            else:
                d_mean = dval[:, 0]
            mesh.celldata[f"{prefix}damage"] = d_mean


    # --------------------------
    # solver
    # --------------------------
    @staticmethod
    def _default_spsolve(A, F) -> np.ndarray:
        return spsolve(A, F)

