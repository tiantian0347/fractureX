"""model3: L 形 SENP 试件 (Winkler benchmark)。

对齐 ttthesis/code/ip_hybrid_mix/test_model3_Lshape.py。

几何：[0, 500]² 去掉右下 (x>250, y<250) → L 形。
材料：Gc=8.9e-5, l0=1.18, μ=10.95, λ=6.16（各向异性小样本）
BC：
  - is_disp_boundary   : 单点 (470, 250) 上 y 方向规定位移（三段位移历史）
  - is_dirchlet_boundary : 下边界 y=0 全固定
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fealpy.mesh import TriangleMesh

from ..ipfem_phasefield_solver import IPFEMPhaseFieldSolver


def _default_load_sequence() -> np.ndarray:
    return np.concatenate(
        (
            np.linspace(0, 0.3, 301),
            np.linspace(0.3, -0.2, 501)[1:],
            np.linspace(-0.2, 1.0, 1201)[1:],
        )
    )


def _lshape_threshold(p):
    return (p[..., 0] > 250) & (p[..., 1] < 250)


@dataclass
class Model3LShapeCase:
    """model3: L 形试件 + 端点集中位移。"""

    name: str = "ipfem_model3_lshape"
    Gc: float = 8.9e-5
    l0: float = 1.18
    mu: float = 10.95
    lam: float = 6.16
    nx: int = 50
    ny: int = 50
    box: tuple = (0.0, 500.0, 0.0, 500.0)
    load_point: tuple = (470.0, 250.0)
    gamma: float = 5.0
    p_disp: int = 1
    p_phase: int = 2
    model_type: str = "HybridModel"
    csd_type: str = "AT2"
    ed_type: str = "quadratic"
    load_sequence: Optional[np.ndarray] = None

    def build_material(self) -> dict:
        return dict(lam=self.lam, mu=self.mu, Gc=self.Gc, l0=self.l0)

    def build_mesh(self) -> TriangleMesh:
        return TriangleMesh.from_box(
            box=list(self.box), nx=self.nx, ny=self.ny,
            threshold=_lshape_threshold,
        )

    def build_solver(self, mesh=None) -> IPFEMPhaseFieldSolver:
        mesh = mesh if mesh is not None else self.build_mesh()
        # 检查 load_point 是否是网格节点
        node = np.asarray(mesh.entity("node"))
        d2 = np.sum((node - np.array(self.load_point)) ** 2, axis=-1)
        if d2.min() > 1e-6:
            raise RuntimeError(
                f"load_point={self.load_point} 不在网格节点上（min dist2={d2.min():.3g}），"
                f"调整 (nx, ny) 让节点落在 load_point 上。"
            )
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
        )
        return solver

    def is_disp_boundary(self, p) -> np.ndarray:
        p = np.asarray(p)
        px, py = self.load_point
        isDNode = (np.abs(p[..., 1] - py) < 1e-8) & (np.abs(p[..., 0] - px) < 1e-5)
        return np.c_[np.zeros(p.shape[0], dtype=bool), isDNode]

    @staticmethod
    def is_dirchlet_boundary(p) -> np.ndarray:
        p = np.asarray(p)
        return np.abs(p[..., 1]) < 1e-12

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
                print(f"[model3] step {i+1}/{len(disp)-1} disp={disp[i+1]:.4e}")
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
