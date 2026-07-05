"""IP-FEM 相场断裂 solver, 带 Aifantis strain-gradient 弹性耦合 (2D)。

对齐 `Tian/thesis/ip_fracture/ipfem_paper.tex` §"Extension to strain-gradient
elasticity"：位移块 LHS 在标准 elasticity 之上加 `g(d) · ℓ_s² · B_h^u(u, v)`，
其中 `B_h^u` 是分量各自的 C⁰-IP 4 阶双线性形式。

设计要点：
- 继承 `IPFEMPhaseFieldSolver`，唯一新增参数 `ell_s`（默认 0，等价于原求解器）；
- `p_disp >= 2` 自动强制（strain-gradient 需 `D²u` 存在）；
- 用 `TensorFunctionSpace(scalar_space, (GD, -1))`（dof_priority=True）的
  block-diagonal 布局，标量 SG 矩阵按对角拼接到位移向量系统上；
- 与 `_assemble_interior_penalty` 已修正符号的负号约定一致（fealpy v3
  上游 IP 集成器 consistency 项符号问题的绕行）；
- 只在装配 A_u 时加入，其余流程（H 场、相场组装、Newton、加密）不变。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fealpy.fem import BilinearForm, LinearElasticIntegrator
from fealpy.functionspace import InteriorPenaltyFESpace2d, LagrangeFESpace, TensorFunctionSpace

from .ipfem_phasefield_solver import (
    IPFEMPhaseFieldSolver,
    _apply_dirichlet_scalar,
    _zero_dof_dirichlet,
)
from .solver import _to_numpy, _to_scipy_csr
from .sg_elastic import assemble_sg_elastic_block, block_diag_vector


class IPFEMPhaseFieldSGSolver(IPFEMPhaseFieldSolver):
    """IP-FEM 相场断裂 solver + Aifantis 应变梯度弹性耦合。

    Parameters
    ----------
    mesh
        TriangleMesh (2D)。
    material_params
        字典，需含 ``{'Gc','l0'}`` 以及 ``{'lam','mu'}`` 或 ``{'E','nu'}``。
    ell_s
        应变梯度长度尺度 (Aifantis 单参数版)；默认 0 → 退化为 IPFEMPhaseFieldSolver。
    p_disp
        位移多项式次数，默认 2（strain-gradient 要求 p_disp >= 2）。
    p_phase
        相场多项式次数，默认 2。
    gamma
        C⁰-IP 罚参数，默认 5。
    model_type / ed_type / csd_type
        同 IPFEMPhaseFieldSolver。
    """

    def __init__(
        self,
        mesh,
        material_params: dict,
        *,
        ell_s: float = 0.0,
        p_disp: int = 2,
        p_phase: int = 2,
        gamma: float = 5.0,
        model_type: str = "HybridModel",
        ed_type: str = "quadratic",
        csd_type: str = "AT2",
        sg_split: bool = False,
        sg_eps_reg: float = 1e-10,
    ):
        """
        Parameters
        ----------
        sg_split
            True → 用 Ali 2024 谱分解版：SG 项按拉伸/压缩劈开，只对 g(d)·χ_+(ε)
            部分退化。False (默认) → 基础 Aifantis, 全 SG 项都乘 g(d)。
        sg_eps_reg
            tex \eqref{eq:tensile-frac} 里 ε_r 正则化常数, 防止 tr(ε)≈0 时分母
            接近零导致数值不稳定。
        """
        if ell_s > 0 and p_disp < 2:
            raise ValueError(
                f"strain-gradient 需要 p_disp >= 2，但传入 p_disp={p_disp}。"
                " 请显式设置 p_disp=2 或以上。"
            )

        super().__init__(
            mesh,
            material_params,
            p_disp=p_disp,
            p_phase=p_phase,
            gamma=gamma,
            model_type=model_type,
            ed_type=ed_type,
            csd_type=csd_type,
        )
        self.ell_s = float(ell_s)
        self.sg_split = bool(sg_split)
        self.sg_eps_reg = float(sg_eps_reg)

        # 位移侧还需要一个标量 IP 空间以组 B_h^u。base_space (self.space) 就是标量
        # p_disp 的 Lagrange 空间，直接包一层 InteriorPenaltyFESpace2d 即可。
        # 当 ell_s == 0 时不用组装，跳过。
        if self.ell_s > 0:
            self.ipspace_u = InteriorPenaltyFESpace2d(
                mesh, p=p_disp, space="Lagrange"
            )
        else:
            self.ipspace_u = None

    # ------------------------------------------------------------------
    # 覆盖父类的位移块装配：加 SG 项
    # ------------------------------------------------------------------
    def _assemble_disp_lhs(self):
        """A_u = 标准 elasticity block + α_cell · ell_s^2 · B_h^u block-diagonal。

        α_cell = g(d)         (基础 Aifantis, sg_split=False, 全域退化)
              = g(d) · χ_+(ε) (Ali 2024, sg_split=True, 只对拉伸区退化)
        """
        ubform = BilinearForm(self.tspace)
        ubform.add_integrator(
            LinearElasticIntegrator(self.pfcm, q=self.q, method="voigt")
        )
        A_u = _to_scipy_csr(ubform.assembly())

        if self.ell_s > 0 and self.ipspace_u is not None:
            alpha_cell = self._compute_sg_cell_weight()  # (NC,) or None (retreat to no-weight)
            A_sg_scalar = assemble_sg_elastic_block(
                self.ipspace_u,
                ell_s=self.ell_s,
                gamma=self.gamma,
                q=self.q,
                cell_coef=alpha_cell,
            )
            GD = self.mesh.geo_dimension()
            A_sg = block_diag_vector(A_sg_scalar, GD)
            A_u = A_u + A_sg
        return A_u

    def _compute_sg_cell_weight(self):
        """在每个单元重心处评估 α_cell = g(d) · [sg_split ? χ_+(ε) : 1]。"""
        try:
            bc_center = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float64)
            gd_val = _to_numpy(
                self.ed_func.degradation_function(self.d(bc_center))
            ).reshape(-1)  # (NC,)
        except Exception:
            return None

        if not self.sg_split:
            return gd_val

        # tensile fraction χ_+ from tr(ε(u_h)) at cell centroid
        try:
            strain = _to_numpy(self.pfcm.strain_value(bc_center))
            NC = gd_val.shape[0]
            # strain shape 常见为 (NC, 1, 2, 2)，reshape (NC, 2, 2) 取 trace
            strain_flat = strain.reshape(NC, 2, 2)
            tr_eps = strain_flat[:, 0, 0] + strain_flat[:, 1, 1]
        except Exception:
            return gd_val

        abs_tr = np.abs(tr_eps)
        tr_pos = 0.5 * (tr_eps + abs_tr)
        chi_plus = tr_pos / (abs_tr + self.sg_eps_reg)  # ∈ [0, 1]
        alpha = gd_val * chi_plus
        return alpha

    def newton_raphson(
        self,
        disp: float,
        *,
        maxit: int = 50,
        rtol: float = 1e-5,
        verbose: bool = False,
    ) -> int:
        """对齐父类逻辑, 位移 LHS 换成 `_assemble_disp_lhs()`。"""
        assert self._disp_dof_mask is not None and self._fix_disp_mask is not None, (
            "调用 newton_raphson 前请先 attach_boundary(...)"
        )

        mesh = self.mesh
        GD = mesh.geo_dimension()

        # -- 位移 Dirichlet 掩码：一律用 tspace 的插值点，兼容 p_disp>=2
        ipoints = _to_numpy(mesh.interpolation_points(p=self.p_disp))
        disp_mask = np.asarray(self._disp_dof_mask(ipoints), dtype=bool)
        fix_mask = np.asarray(self._fix_disp_mask(ipoints), dtype=bool)
        fix_dof = np.zeros((ipoints.shape[0], GD), dtype=bool)
        fix_dof[fix_mask, :] = True

        if disp_mask.shape != fix_dof.shape:
            raise RuntimeError(
                f"is_disp_boundary 返回形状 {disp_mask.shape} 与 (Nip, GD)="
                f"{fix_dof.shape} 不匹配。"
            )
        disp_dof_flat = disp_mask.reshape(-1)
        fix_dof_flat = fix_dof.reshape(-1)

        # -- 相场 Dirichlet
        if self._phase_dof_mask is not None:
            phase_ipoints = _to_numpy(mesh.interpolation_points(p=self.p_phase))
            phase_fix_mask = np.asarray(
                self._phase_dof_mask(phase_ipoints), dtype=bool
            )
        else:
            phase_fix_mask = None

        uh_arr = _to_numpy(self.uh).reshape(-1).astype(np.float64).copy()
        d_arr = _to_numpy(self.d).reshape(-1).astype(np.float64).copy()

        uh_arr[disp_dof_flat] = disp
        all_fix_u = fix_dof_flat | disp_dof_flat

        r0_u, r0_d = None, None
        iters = 0
        for k in range(maxit):
            iters = k + 1
            self._sync_uh(uh_arr)
            self._sync_d(d_arr)

            A_u = self._assemble_disp_lhs()
            R_u = -A_u @ uh_arr

            self.force = float(np.sum(-R_u[disp_dof_flat]))

            A_u_bc, R_u_bc = _zero_dof_dirichlet(A_u, R_u, all_fix_u)
            du = spsolve(A_u_bc, R_u_bc)
            uh_arr = uh_arr + du
            self._sync_uh(uh_arr)

            bc_center = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float64)
            self.H = self.pfcm.maximum_historical_field(bc_center)

            A_d = self._assemble_phase_lhs()
            b_d = self._assemble_phase_rhs()
            R_d = b_d - A_d @ d_arr

            if phase_fix_mask is not None:
                A_d_bc, R_d_bc = _zero_dof_dirichlet(A_d, R_d, phase_fix_mask)
            else:
                A_d_bc, R_d_bc = _to_scipy_csr(A_d), np.asarray(R_d).reshape(-1)

            dd = spsolve(A_d_bc, R_d_bc)
            d_arr = d_arr + dd
            d_arr = np.maximum(d_arr, _to_numpy(self.d).reshape(-1))
            self._sync_d(d_arr)

            norm_u = float(np.linalg.norm(R_u_bc))
            norm_d = float(np.linalg.norm(R_d_bc))
            if k == 0:
                r0_u = max(norm_u, 1e-30)
                r0_d = max(norm_d, 1e-30)
            err_u = norm_u / r0_u
            err_d = norm_d / r0_d
            err = max(err_u, err_d)
            if verbose:
                print(
                    f"  [SG NR k={k}] err_u={err_u:.3e} err_d={err_d:.3e} "
                    f"|force|={self.force:.4e}"
                )
            if err < rtol:
                break

        self._write_back_uh(uh_arr)
        self._write_back_d(d_arr)

        self.stored_energy = self._compute_stored_energy()
        self.dissipated_energy = self._compute_dissipated_energy()
        if self._adaptive is not None:
            self._refine_if_needed()
        return iters
