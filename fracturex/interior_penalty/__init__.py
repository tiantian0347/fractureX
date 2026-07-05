"""
fracturex.interior_penalty — 双调和 / 相场断裂里要用的内罚有限元求解封装。

导入本子包时：
- 自动尝试给 fealpy numpy 后端补 `simplex_hess_shape_function`（若缺失）；
- 暴露 PDE 类、双调和 solver、IP-FEM 相场断裂 solver。
"""
from . import _numpy_hess_patch as _hp
_hp.apply_patch()

from .biharmonic_pde import DoubleLaplacePDE, default_pde, sin_sq_pde
from .solver import (
    solve_biharmonic,
    l2_error_from_dof,
    h1_semi_error_from_dof,
    h2_semi_error_from_dof,
    convergence_study,
    compute_orders,
)
from .ipfem_phasefield_solver import IPFEMPhaseFieldSolver
from .ipfem_phasefield_sg_solver import IPFEMPhaseFieldSGSolver

__all__ = [
    "DoubleLaplacePDE",
    "default_pde",
    "sin_sq_pde",
    "solve_biharmonic",
    "l2_error_from_dof",
    "h1_semi_error_from_dof",
    "h2_semi_error_from_dof",
    "convergence_study",
    "compute_orders",
    "IPFEMPhaseFieldSolver",
    "IPFEMPhaseFieldSGSolver",
]
