"""SENT tension benchmark (Miehe 2010 / Ambati 2015 标准算例)。

对齐 T6a.pre.B5：为审稿人提供的通用 benchmark。与 model1 结构一致
(单位方 + 顶部拉伸 + 底部固定 + 预裂纹通过重复节点实现)，只是采用
Miehe 2010 的材料参数、荷载序列与网格加密。

参数（Miehe 2010, IJNME 83:1273-1311, §5.1 SENT tension）:
    E = 210 [kN/mm²] (~ 210 GPa)
    ν = 0.3
    G_c = 2.7 × 10⁻³ [kN/mm]
    ℓ_0 = 0.015 [mm] (预裂纹宽度)
    domain: 1 × 1 mm²
    load: displacement-controlled tension along y at top, 0 → 6e-3 mm

预裂纹几何:
    (0, 0.5) → (0.5, 0.5) 一段水平缝，通过初始 mesh 中重复 (0, 0.5) 节点
    实现（与老 test_model1.py 完全一致），保留几何断口。

不同于 `Model1SquareTensionCase` 的地方：
    - 材料 ℓ_0 = 0.015 (Miehe 标准), 老 model1 用 0.0133
    - 加密到 refine=6（默认）或更细，可显式覆盖
    - 载荷序列按 Miehe 论文用 5000-6000 步的细分（可覆盖）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fealpy.mesh import TriangleMesh

from ..ipfem_phasefield_solver import IPFEMPhaseFieldSolver
from .model1_square_tension import _default_init_mesh


def _sent_load_sequence_miehe() -> np.ndarray:
    """Miehe 2010 §5.1 SENT tension 用的两段线性载荷：
    0 → 5e-3 步长 1e-5，5e-3 → 6e-3 步长 1e-6。
    """
    return np.concatenate(
        (
            np.linspace(0, 5e-3, 501),
            np.linspace(5e-3, 6e-3, 1001)[1:],
        )
    )


@dataclass
class SentTensionMieheCase:
    """SENT tension benchmark (Miehe 2010)."""

    name: str = "ipfem_sent_tension_miehe"
    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 0.015
    refine: int = 6  # 服务器规模；本地冒烟建议 refine=2 或 3
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

    @staticmethod
    def is_disp_boundary(p) -> np.ndarray:
        p = np.asarray(p)
        isDNode = np.abs(p[..., 1] - 1) < 1e-12
        return np.c_[np.zeros(p.shape[0], dtype=bool), isDNode]

    @staticmethod
    def is_dirchlet_boundary(p) -> np.ndarray:
        p = np.asarray(p)
        return np.abs(p[..., 1]) < 1e-12

    def loads(self) -> np.ndarray:
        return _sent_load_sequence_miehe() if self.load_sequence is None else self.load_sequence

    def run(
        self,
        *,
        max_steps: Optional[int] = None,
        maxit_per_step: int = 30,
        rtol: float = 1e-5,
        verbose: bool = False,
        vtk_every: Optional[int] = None,
        vtk_prefix: Optional[str] = None,
    ) -> dict:
        """跑载荷序列。

        vtk_every: 每 N 步写一次 vtu 到 f"{vtk_prefix}_step{i:06d}.vtu"。
        默认 None → 不写。
        """
        solver = self.build_solver()
        mesh = solver.mesh
        disp = self.loads()
        if max_steps is not None:
            disp = disp[: max_steps + 1]
        force = np.zeros_like(disp)
        stored = np.zeros_like(disp)
        dissipated = np.zeros_like(disp)

        for i in range(len(disp) - 1):
            if verbose:
                print(
                    f"[SENT] step {i+1}/{len(disp)-1} disp={disp[i+1]:.4e}"
                )
            solver.newton_raphson(
                disp[i + 1], maxit=maxit_per_step, rtol=rtol, verbose=False
            )
            force[i + 1] = solver.force
            stored[i + 1] = solver.stored_energy
            dissipated[i + 1] = solver.dissipated_energy

            if vtk_every is not None and vtk_prefix is not None and (i + 1) % vtk_every == 0:
                try:
                    from fracturex.interior_penalty.solver import _to_numpy
                    mesh.nodedata["damage"] = _to_numpy(solver.d).reshape(-1)
                    fname = f"{vtk_prefix}_step{i+1:06d}.vtu"
                    mesh.to_vtk(fname=fname)
                    if verbose:
                        print(f"    wrote {fname}")
                except Exception as exc:
                    print(f"    vtu write failed: {exc}")

        return dict(
            disp=disp, force=force,
            stored_energy=stored, dissipated_energy=dissipated,
            solver=solver,
        )
