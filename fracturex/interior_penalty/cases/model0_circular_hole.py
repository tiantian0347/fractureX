"""model0: 方形 + 内圆孔算例（顶部拉伸，内圆边界固定 + 相场 d=0）。

对齐 ttthesis/code/ip_hybrid_mix/test_model0.py。

网格：[0,1]² 减去 (0.5, 0.5, r=0.2) 圆，distmesh 生成。
材料：E=200, ν=0.2, Gc=1.0, l0=0.02。
BC：
  - is_disp_boundary   : y=1 上 y 方向规定位移
  - is_dirchlet_boundary : 内圆边界固定 u=0
  - is_boundary_phase  : 内圆边界 d=0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fealpy.mesh import TriangleMesh

from ..ipfem_phasefield_solver import IPFEMPhaseFieldSolver
from ._domain_hole import SquareWithCircleHoleDomain


def _default_load_sequence() -> np.ndarray:
    return np.concatenate(
        (
            np.linspace(0, 70e-3, 6),
            np.linspace(70e-3, 125e-3, 26)[1:],
        )
    )


@dataclass
class Model0CircularHoleCase:
    """model0: 内圆孔 + 顶部拉伸。"""

    name: str = "ipfem_model0_circular_hole"
    E: float = 200.0
    nu: float = 0.2
    Gc: float = 1.0
    l0: float = 0.02
    hmin: float = 0.05
    distmesh_maxit: int = 100
    gamma: float = 5.0
    p_disp: int = 1
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

    def build_solver(self, mesh=None) -> IPFEMPhaseFieldSolver:
        mesh = mesh if mesh is not None else self.build_mesh()
        solver = IPFEMPhaseFieldSolver(
            mesh,
            self.build_material(),
            p_disp=self.p_disp,
            p_phase=self.p_phase,
            gamma=self.gamma,
            model_type=self.model_type,
            ed_type=self.ed_type,
            csd_type=self.csd_type,
        )
        solver.attach_boundary(
            is_disp_boundary=self.is_disp_boundary,
            is_dirchlet_boundary=self.is_dirchlet_boundary,
            is_boundary_phase=self.is_boundary_phase,
        )
        return solver

    # BC 定义 --------------------------------------------------------
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

    # 载荷 -----------------------------------------------------------
    def loads(self) -> np.ndarray:
        return _default_load_sequence() if self.load_sequence is None else self.load_sequence

    # 主循环 ---------------------------------------------------------
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
                print(f"[model0] step {i+1}/{len(disp)-1} disp={disp[i+1]:.4e}")
            solver.newton_raphson(
                disp[i + 1], maxit=maxit_per_step, rtol=rtol, verbose=verbose
            )
            force[i + 1] = solver.force
            stored[i + 1] = solver.stored_energy
            dissipated[i + 1] = solver.dissipated_energy
        return dict(
            disp=disp, force=force,
            stored_energy=stored, dissipated_energy=dissipated,
            solver=solver,
        )
