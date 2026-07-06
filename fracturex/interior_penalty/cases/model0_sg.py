"""model0 SG variant: 方形+内圆孔+顶部拉伸+应变梯度弹性耦合。

跟 `Model0CircularHoleCase` 相同的几何/BC/材料，只是 solver 换成
`IPFEMPhaseFieldSGSolver`。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fealpy.mesh import TriangleMesh

from ..ipfem_phasefield_sg_solver import IPFEMPhaseFieldSGSolver
from ._domain_hole import SquareWithCircleHoleDomain
from .model0_circular_hole import _default_load_sequence


@dataclass
class Model0SGCase:
    """model0 内圆孔 + 顶部拉伸 + 应变梯度耦合。"""

    name: str = "ipfem_model0_sg"
    E: float = 200.0
    nu: float = 0.2
    Gc: float = 1.0
    l0: float = 0.02
    ell_s: float = 0.0
    sg_split: bool = False
    hmin: float = 0.05
    distmesh_maxit: int = 100
    gamma: float = 5.0
    p_disp: int = 2  # SG 需要 p_disp >= 2
    p_phase: int = 2
    model_type: str = "HybridModel"
    csd_type: str = "AT2"
    ed_type: str = "quadratic"
    load_sequence: Optional[np.ndarray] = None
    hole_center: tuple = (0.5, 0.5)
    hole_radius: float = 0.2

    def build_material(self) -> dict:
        mu = self.E / (2 * (1 + self.nu))
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        return dict(E=self.E, nu=self.nu, lam=lam, mu=mu, Gc=self.Gc, l0=self.l0)

    def build_mesh(self) -> TriangleMesh:
        domain = SquareWithCircleHoleDomain(hmin=self.hmin)
        return TriangleMesh.from_domain_distmesh(domain, maxit=self.distmesh_maxit)

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
            is_boundary_phase=self.is_boundary_phase,
        )
        return solver

    def is_disp_boundary(self, p) -> np.ndarray:
        p = np.asarray(p)
        isDNode = np.abs(p[..., 1] - 1) < 1e-12
        return np.c_[np.zeros(p.shape[0], dtype=bool), isDNode]

    def is_dirchlet_boundary(self, p) -> np.ndarray:
        return self._on_hole_boundary(p)

    def is_boundary_phase(self, p) -> np.ndarray:
        return self._on_hole_boundary(p)

    def _on_hole_boundary(self, p) -> np.ndarray:
        p = np.asarray(p)
        cx, cy = self.hole_center
        r2 = (p[..., 0] - cx) ** 2 + (p[..., 1] - cy) ** 2
        return np.abs(r2 - self.hole_radius ** 2) < 0.001

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
                    f"[model0_sg ell_s={self.ell_s}] step {i+1}/{len(disp)-1} "
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
