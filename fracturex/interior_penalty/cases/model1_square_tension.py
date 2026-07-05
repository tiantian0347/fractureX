"""model1: 单位方形 + 顶部拉伸算例。

对齐 ttthesis/code/ip_hybrid_mix/test_model1.py。

网格：手工 8 个三角形起始 (中间线 y=0.5 有一条重复节点作预裂纹缝), uniform_refine(n).
材料：E=210, nu=0.3, Gc=2.7e-3, l0=0.0133。
BC：y=0 全固定；y=1 上 y 方向规定位移。
载荷序列：np.linspace(0, 5e-3, 501) ∪ np.linspace(5e-3, 6e-3, 1001)[1:]。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from fealpy.mesh import TriangleMesh

from ..ipfem_phasefield_solver import IPFEMPhaseFieldSolver


def _default_init_mesh(refine: int = 4) -> TriangleMesh:
    """老 test_model1.py 里的 8 个初始三角，其中 (0, 0.5) 节点重复以承载
    预裂纹（duplicate node 使 y=0.5、x∈[0, 0.5) 边成为几何断口）。"""
    node = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],  # 重复节点：预裂纹
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cell = np.array(
        [
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5],
        ],
        dtype=np.int_,
    )
    mesh = TriangleMesh(node, cell)
    if refine > 0:
        mesh.uniform_refine(n=refine)
    return mesh


def _default_load_sequence() -> np.ndarray:
    return np.concatenate(
        (
            np.linspace(0, 5e-3, 501),
            np.linspace(5e-3, 6e-3, 1001)[1:],
        )
    )


@dataclass
class Model1SquareTensionCase:
    """model1: 顶部 y 方向规定位移拉伸。"""

    name: str = "ipfem_model1_square_tension"
    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 0.0133
    refine: int = 4
    gamma: float = 5.0
    p_disp: int = 1
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
        )
        return solver

    # BC 定义 --------------------------------------------------------
    @staticmethod
    def is_disp_boundary(p) -> np.ndarray:
        p = np.asarray(p)
        isDNode = np.abs(p[..., 1] - 1) < 1e-12
        return np.c_[np.zeros(p.shape[0], dtype=bool), isDNode]

    @staticmethod
    def is_dirchlet_boundary(p) -> np.ndarray:
        p = np.asarray(p)
        return np.abs(p[..., 1]) < 1e-12

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
        """跑载荷序列，返回 dict(force, stored_energy, dissipated_energy, disp)."""
        solver = self.build_solver()
        disp = self.loads()
        if max_steps is not None:
            disp = disp[: max_steps + 1]
        force = np.zeros_like(disp)
        stored = np.zeros_like(disp)
        dissipated = np.zeros_like(disp)
        for i in range(len(disp) - 1):
            if verbose:
                print(f"[model1] step {i+1}/{len(disp)-1} disp={disp[i+1]:.4e}")
            solver.newton_raphson(
                disp[i + 1], maxit=maxit_per_step, rtol=rtol, verbose=verbose
            )
            force[i + 1] = solver.force
            stored[i + 1] = solver.stored_energy
            dissipated[i + 1] = solver.dissipated_energy
        return dict(
            disp=disp,
            force=force,
            stored_energy=stored,
            dissipated_energy=dissipated,
            solver=solver,
        )
