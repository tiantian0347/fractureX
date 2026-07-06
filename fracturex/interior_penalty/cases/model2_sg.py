"""model2 SG variant: 单位方形+顶部剪切+应变梯度弹性耦合。

跟 `Model2ShearCase` 相同的几何/BC/材料，只是 solver 换成
`IPFEMPhaseFieldSGSolver`。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fealpy.mesh import TriangleMesh

from ..ipfem_phasefield_sg_solver import IPFEMPhaseFieldSGSolver
from .model1_square_tension import _default_init_mesh
from .model2_notch_shear import _default_load_sequence


@dataclass
class Model2SGCase:
    """model2 剪切 + 应变梯度耦合。"""

    name: str = "ipfem_model2_sg"
    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 0.004
    ell_s: float = 0.0
    sg_split: bool = False
    refine: int = 4
    gamma: float = 5.0
    p_disp: int = 2
    p_phase: int = 2
    model_type: str = "HybridModel"
    csd_type: str = "AT2"
    ed_type: str = "quadratic"
    load_sequence: Optional[np.ndarray] = None

    def build_material(self) -> dict:
        mu = self.E / (2 * (1 + self.nu))
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        return dict(E=self.E, nu=self.nu, lam=lam, mu=mu, Gc=self.Gc, l0=self.l0)

    def build_mesh(self) -> TriangleMesh:
        return _default_init_mesh(self.refine)

    def build_solver(self, mesh=None) -> IPFEMPhaseFieldSGSolver:
        mesh = mesh if mesh is not None else self.build_mesh()
        solver = IPFEMPhaseFieldSGSolver(
            mesh,
            self.build_material(),
            ell_s=self.ell_s,
            p_disp=self.p_disp,
            p_phase=self.p_phase,
            gamma=self.gamma,
            model_type=self.model_type,
            ed_type=self.ed_type,
            csd_type=self.csd_type,
            sg_split=self.sg_split,
        )
        solver.attach_boundary(
            is_disp_boundary=self.is_disp_boundary,
            is_dirchlet_boundary=self.is_dirchlet_boundary,
        )
        return solver

    @staticmethod
    def is_disp_boundary(p) -> np.ndarray:
        p = np.asarray(p)
        isDNode = np.abs(p[..., 1] - 1) < 1e-12
        return np.c_[isDNode, np.zeros(p.shape[0], dtype=bool)]

    @staticmethod
    def is_dirchlet_boundary(p) -> np.ndarray:
        p = np.asarray(p)
        return (np.abs(p[..., 1]) < 1e-12) | (np.abs(p[..., 1] - 1) < 1e-12)

    def loads(self) -> np.ndarray:
        return _default_load_sequence() if self.load_sequence is None else self.load_sequence

    def run(
        self,
        *,
        max_steps: Optional[int] = None,
        maxit_per_step: int = 30,
        rtol: float = 1e-5,
        verbose: bool = False,
    ) -> dict:
        solver = self.build_solver()
        disp = self.loads()
        if max_steps is not None:
            disp = disp[: max_steps + 1]
        force = np.zeros_like(disp)
        stored = np.zeros_like(disp)
        dissipated = np.zeros_like(disp)
        for i in range(len(disp) - 1):
            if verbose:
                print(
                    f"[model2_sg ell_s={self.ell_s}] step {i+1}/{len(disp)-1} "
                    f"disp={disp[i+1]:.4e}"
                )
            solver.newton_raphson(
                disp[i + 1], maxit=maxit_per_step, rtol=rtol, verbose=False
            )
            force[i + 1] = solver.force
            stored[i + 1] = solver.stored_energy
            dissipated[i + 1] = solver.dissipated_energy
        return dict(
            disp=disp, force=force,
            stored_energy=stored, dissipated_energy=dissipated,
            solver=solver,
        )
