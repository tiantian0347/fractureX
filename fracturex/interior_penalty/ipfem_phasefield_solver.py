"""IP-FEM 相场断裂 solver (2D)。

参考：`ttthesis/code/ip_hybrid_mix/` + `fealpy/csm/ipfem_phase_field_crack_hybrid_mix_model.py`。

耦合形式（staggered Newton-Raphson，位移 → 相场）：

* 位移块  ``A_u = ∫ B^T · D(d) · B dx``，其中 ``D(d) = g(d) · D_0``；
  边界条件：``is_dirchlet_boundary`` 上 u=0；``is_disp_boundary`` 上 u=disp_target。
* 相场块  ``A_d = Gc·l0·∫∇d·∇v + (2H + Gc/(2l0))·∫d·v + (Gc·l0³/16)·(∫D²d:D²v + IP)``；
  右端  ``R_d = ∫ 2H·v dx - A_d @ d_prev``；
  可选 ``is_boundary_phase`` 边上 d=0。
* 历史场 ``H = max(H_prev, ψ⁺(strain))``，由 `HybridModel` 谱分解给出。

用 fealpy v3 API + 我们的 `_assemble_interior_penalty` 组装。所有稀疏矩阵最后落到
scipy.sparse (LHS + spsolve)，因此可跨 numpy / pytorch 后端。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve

from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric
from fealpy.fem import (
    BilinearForm,
    LinearForm,
    LinearElasticIntegrator,
    ScalarBiharmonicIntegrator,
    ScalarDiffusionIntegrator,
    ScalarMassIntegrator,
    ScalarSourceIntegrator,
)
from fealpy.functionspace import (
    InteriorPenaltyFESpace2d,
    LagrangeFESpace,
    TensorFunctionSpace,
)

from fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction
from fracturex.phasefield.crack_surface_density_function import CrackSurfaceDensityFunction
from fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory
from fracturex.adaptivity.adaptive_refinement import AdaptiveRefinement

from .solver import _to_numpy, _to_scipy_csr, _assemble_interior_penalty


def _apply_dirichlet_scalar(A, f, uh, isDDof):
    A = _to_scipy_csr(A)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    uh = np.asarray(uh, dtype=np.float64).reshape(-1)
    isDDof = np.asarray(isDDof).reshape(-1).astype(bool)

    f = f - A @ uh
    bdIdx = np.zeros(A.shape[0], dtype=np.int64)
    bdIdx[isDDof] = 1
    D0 = spdiags(1 - bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0 @ A @ D0 + D1
    f[isDDof] = uh[isDDof]
    return A, f


def _zero_dof_dirichlet(A, R, isDDof):
    """强制 du[isDDof] = 0（用于 R = -A @ u_with_target 已经把目标写入 u 的场景）。"""
    A = _to_scipy_csr(A)
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    isDDof = np.asarray(isDDof).reshape(-1).astype(bool)
    bdIdx = np.zeros(A.shape[0], dtype=np.int64)
    bdIdx[isDDof] = 1
    D0 = spdiags(1 - bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0 @ A @ D0 + D1
    R = R.copy()
    R[isDDof] = 0.0
    return A, R


class IPFEMPhaseFieldSolver:
    """IP-FEM 相场断裂（2D）solver。

    Parameters
    ----------
    mesh
        TriangleMesh，2D 三角网格。
    material_params
        字典，需含 ``{'Gc','l0'}`` 以及 ``{'lam','mu'}`` 或 ``{'E','nu'}``。
    p_disp
        位移场次数，默认 1。
    p_phase
        相场次数（同时用于 IP 空间），默认 2。
    gamma
        内罚参数，默认 5。
    model_type
        相场材料模型，默认 ``'HybridModel'``。
    ed_type
        能量退化函数类型，默认 ``'quadratic'``。
    csd_type
        裂纹表面密度类型，默认 ``'AT2'``。
    """

    def __init__(
        self,
        mesh,
        material_params: dict,
        *,
        p_disp: int = 1,
        p_phase: int = 2,
        gamma: float = 5.0,
        model_type: str = "HybridModel",
        ed_type: str = "quadratic",
        csd_type: str = "AT2",
    ):
        if mesh.geo_dimension() != 2:
            raise NotImplementedError(
                "IPFEMPhaseFieldSolver 目前只支持 2D (fealpy 的 InteriorPenaltyFESpace2d 限制)。"
            )
        self.mesh = mesh
        self.material_params = dict(material_params)
        self.Gc = material_params["Gc"]
        self.l0 = material_params["l0"]
        self.gamma = gamma
        self.p_disp = p_disp
        self.p_phase = p_phase
        self.model_type = model_type
        self.q = 2 * max(p_disp, p_phase) + 3

        self.ed_func = EnergyDegradationFunction(degradation_type=ed_type)
        self.csd_func = CrackSurfaceDensityFunction(density_type=csd_type)
        self.pfcm = PhaseFractureMaterialFactory.create(
            model_type, self.material_params, self.ed_func
        )

        # 位移场：向量 Lagrange
        self.space = LagrangeFESpace(mesh, p=p_disp)
        self.tspace = TensorFunctionSpace(self.space, (mesh.geo_dimension(), -1))
        # 相场：标量 Lagrange
        self.dspace = LagrangeFESpace(mesh, p=p_phase)
        # 内罚空间
        self.ipspace = InteriorPenaltyFESpace2d(mesh, p=p_phase, space="Lagrange")

        self.uh = self.tspace.function()
        self.d = self.dspace.function()
        self.H = None  # 由 pfcm.maximum_historical_field 首次更新

        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)

        # 自适应加密（默认关，可通过 enable_adaptive_refinement(...) 打开）
        self._adaptive: Optional[AdaptiveRefinement] = None

        # 记录当前载荷步的力/能量
        self.force = 0.0
        self.stored_energy = 0.0
        self.dissipated_energy = 0.0

        # 边界条件回调，由用户 attach_boundary
        self._disp_dof_mask: Optional[Callable[[Any], Any]] = None
        self._fix_disp_mask: Optional[Callable[[Any], Any]] = None
        self._phase_dof_mask: Optional[Callable[[Any], Any]] = None

    # ------------------------------------------------------------------ BC api
    def attach_boundary(
        self,
        *,
        is_disp_boundary: Callable[[Any], Any],
        is_dirchlet_boundary: Callable[[Any], Any],
        is_boundary_phase: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Parameters
        ----------
        is_disp_boundary(p) -> (npoints, GD) bool
            指定哪些自由度上加规定位移增量（各分量分开）。
        is_dirchlet_boundary(p) -> (npoints,) bool
            上面固定 u=0 的边界节点。
        is_boundary_phase(p) -> (npoints,) bool, 可选
            上面固定 d=0 的边界节点（如内圆边界）。
        """
        self._disp_dof_mask = is_disp_boundary
        self._fix_disp_mask = is_dirchlet_boundary
        self._phase_dof_mask = is_boundary_phase

    # -------------------------------------------------------- adaptive refine
    def enable_adaptive_refinement(
        self,
        *,
        marking_strategy: str = "recovery",
        refine_method: str = "bisect",
        theta: float = 0.2,
    ):
        """打开自适应加密，参数同 `fracturex.adaptivity.AdaptiveRefinement`。

        每完成一次 staggered 位移+相场求解后，用相场做恢复型误差估计标记单元并
        二分加密；加密后重建有限元空间并把 uh/d/H 插值到新网格。
        """
        self._adaptive = AdaptiveRefinement(
            marking_strategy=marking_strategy,
            refine_method=refine_method,
            theta=theta,
        )

    def _refine_if_needed(self):
        if self._adaptive is None:
            return False
        # bisect 只支持 p=1 位移/相场的数据插值，因此这里把 uh/d 打包成 cell-local。
        ucell2dof = self.tspace.cell_to_dof()
        dcell2dof = self.dspace.cell_to_dof()
        uh_np = _to_numpy(self.uh)
        d_np = _to_numpy(self.d)
        H_np = _to_numpy(self.H) if self.H is not None else np.zeros(
            self.mesh.number_of_cells(), dtype=np.float64
        )
        data = {
            "uh": uh_np[_to_numpy(ucell2dof)],
            "d": d_np[_to_numpy(dcell2dof)],
            "H": H_np.reshape(-1),
        }
        try:
            new_mesh, new_data = self._adaptive.perform_refinement(
                self.mesh, self.d, data, self.l0
            )
        except Exception as exc:
            print(f"[IPFEMPhaseFieldSolver] adaptive refine skipped: {exc}")
            return False
        if new_data is None:
            return False
        # 重建空间和场
        self.mesh = new_mesh
        self._rebuild_spaces_after_refine(new_data)
        return True

    def _rebuild_spaces_after_refine(self, new_data):
        mesh = self.mesh
        self.space = LagrangeFESpace(mesh, p=self.p_disp)
        self.tspace = TensorFunctionSpace(self.space, (mesh.geo_dimension(), -1))
        self.dspace = LagrangeFESpace(mesh, p=self.p_phase)
        self.ipspace = InteriorPenaltyFESpace2d(mesh, p=self.p_phase, space="Lagrange")

        self.uh = self.tspace.function()
        self.d = self.dspace.function()

        # 从 cell-local 展平回全局 dof
        ucell2dof = _to_numpy(self.tspace.cell_to_dof())
        dcell2dof = _to_numpy(self.dspace.cell_to_dof())
        new_uh = np.asarray(new_data["uh"]).reshape(-1)
        new_d = np.asarray(new_data["d"]).reshape(-1)
        uh_arr = _to_numpy(self.uh).reshape(-1).copy()
        d_arr = _to_numpy(self.d).reshape(-1).copy()
        uh_arr[ucell2dof.reshape(-1)] = new_uh
        d_arr[dcell2dof.reshape(-1)] = new_d
        self._sync_uh(uh_arr)
        self._sync_d(d_arr)

        H_new = np.asarray(new_data["H"]).reshape(-1)
        # bisect 通常把 H 复制到子单元，长度应等于 new NC
        if H_new.shape[0] != mesh.number_of_cells():
            NC = mesh.number_of_cells()
            H_padded = np.zeros(NC, dtype=np.float64)
            m = min(NC, H_new.shape[0])
            H_padded[:m] = H_new[:m]
            H_new = H_padded
        # pfcm.spectral_model 里 phip 是 (NC, NQ) 形状，maximum(H, phip) 需要 H 也是 2-d，
        # 否则 numpy 广播会变成 (NC, NC) —— 与老代码用一维 H 的语义相反。
        self.H = H_new.reshape(-1, 1)
        self.pfcm.update_historical_field(self.H)


    # ---------------------------------------------------------------- solver
    def newton_raphson(
        self,
        disp: float,
        *,
        maxit: int = 50,
        rtol: float = 1e-5,
        verbose: bool = False,
    ) -> int:
        """在给定 disp（位移增量目标值）下做一轮 staggered Newton-Raphson。

        返回实际迭代次数。
        """
        assert self._disp_dof_mask is not None and self._fix_disp_mask is not None, (
            "调用 newton_raphson 前请先 attach_boundary(...)"
        )

        mesh = self.mesh
        GD = mesh.geo_dimension()

        # ---- 位移 Dirichlet 掩码：一律用 tspace 的插值点，兼容 p_disp>=2
        ipoints = _to_numpy(mesh.interpolation_points(p=self.p_disp))
        disp_mask = np.asarray(self._disp_dof_mask(ipoints), dtype=bool)  # (Nip, GD)
        fix_mask = np.asarray(self._fix_disp_mask(ipoints), dtype=bool)  # (Nip,)
        fix_dof = np.zeros((ipoints.shape[0], GD), dtype=bool)
        fix_dof[fix_mask, :] = True

        uh_arr = _to_numpy(self.uh).reshape(-1).astype(np.float64).copy()

        if disp_mask.shape != fix_dof.shape:
            raise RuntimeError(
                f"is_disp_boundary 返回形状 {disp_mask.shape} 与 (Nip, GD)="
                f"{fix_dof.shape} 不匹配。"
            )
        # tspace 的 dof 排列：LagrangeFESpace 上按 (NN, GD)，flatten 后 idx = i*GD + d
        disp_dof_flat = disp_mask.reshape(-1)
        fix_dof_flat = fix_dof.reshape(-1)

        # 把规定位移写入 uh
        uh_arr[disp_dof_flat] = disp

        # ---- 相场 Dirichlet
        if self._phase_dof_mask is not None:
            phase_ipoints = _to_numpy(mesh.interpolation_points(p=self.p_phase))
            phase_fix_mask = np.asarray(
                self._phase_dof_mask(phase_ipoints), dtype=bool
            )
        else:
            phase_fix_mask = None

        d_arr = _to_numpy(self.d).reshape(-1).astype(np.float64).copy()

        # 先把 disp 目标值写入 uh_arr，再进 NR 循环（沿用老代码的做法：BC 用 du=0）
        uh_arr[disp_dof_flat] = disp
        all_fix_u = fix_dof_flat | disp_dof_flat

        r0_u, r0_d = None, None
        iters = 0
        for k in range(maxit):
            iters = k + 1
            self._sync_uh(uh_arr)
            self._sync_d(d_arr)

            # ---------- 位移块
            ubform = BilinearForm(self.tspace)
            ubform.add_integrator(
                LinearElasticIntegrator(self.pfcm, q=self.q, method="voigt")
            )
            A_u = _to_scipy_csr(ubform.assembly())
            R_u = -A_u @ uh_arr

            self.force = float(np.sum(-R_u[disp_dof_flat]))

            A_u_bc, R_u_bc = _zero_dof_dirichlet(A_u, R_u, all_fix_u)
            du = spsolve(A_u_bc, R_u_bc)
            uh_arr = uh_arr + du
            self._sync_uh(uh_arr)

            # ---------- 更新历史场 H
            bc_center = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float64)
            self.H = self.pfcm.maximum_historical_field(bc_center)

            # ---------- 相场块
            A_d = self._assemble_phase_lhs()
            b_d = self._assemble_phase_rhs()
            R_d = b_d - A_d @ d_arr

            if phase_fix_mask is not None:
                A_d_bc, R_d_bc = _zero_dof_dirichlet(A_d, R_d, phase_fix_mask)
            else:
                A_d_bc, R_d_bc = _to_scipy_csr(A_d), np.asarray(R_d).reshape(-1)

            dd = spsolve(A_d_bc, R_d_bc)
            d_arr = d_arr + dd

            # 相场不可逆
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
                    f"  [NR k={k}] err_u={err_u:.3e} err_d={err_d:.3e} "
                    f"|force|={self.force:.4e}"
                )
            if err < rtol:
                break

        # 结束一步：写回持久 uh/d
        self._write_back_uh(uh_arr)
        self._write_back_d(d_arr)

        # 能量记录
        self.stored_energy = self._compute_stored_energy()
        self.dissipated_energy = self._compute_dissipated_energy()

        # 载荷步结束后（NR 收敛后）做一次自适应加密。
        # 加密只影响下一载荷步的初始网格，不干扰当前 NR 的收敛判据。
        if self._adaptive is not None:
            self._refine_if_needed()

        return iters

    # -------------------------------------------------------------- helpers
    def _sync_uh(self, uh_arr):
        try:
            self.uh[:] = bm.tensor(uh_arr, dtype=bm.float64)
        except Exception:
            self.uh[:] = uh_arr
        self.pfcm.update_disp(self.uh)

    def _sync_d(self, d_arr):
        try:
            self.d[:] = bm.tensor(d_arr, dtype=bm.float64)
        except Exception:
            self.d[:] = d_arr
        self.pfcm.update_phase(self.d)

    def _write_back_uh(self, uh_arr):
        self._sync_uh(uh_arr)

    def _write_back_d(self, d_arr):
        self._sync_d(d_arr)

    def _assemble_phase_lhs(self):
        """相场块 LHS: A_d = Gc*l0*Diff + (2H + Gc/(2*l0))*Mass + Gc*l0^3/16*(Biharm + IP)."""
        Gc, l0 = self.Gc, self.l0

        @barycentric
        def diff_coef(bc, index):
            return Gc * l0

        H = self.H  # 由 self.pfcm.maximum_historical_field 更新

        @barycentric
        def mass_coef(bc, index):
            # H shape (NC,) or (NC, NQ) - broadcast
            if hasattr(H, "shape") and H.ndim == 1:
                Hval = H[..., None]
            else:
                Hval = H
            return 2 * Hval + Gc / (2 * l0)

        dbform = BilinearForm(self.dspace)
        dbform.add_integrator(ScalarDiffusionIntegrator(coef=diff_coef, q=self.q))
        dbform.add_integrator(ScalarMassIntegrator(coef=mass_coef, q=self.q))
        A_dif_mass = _to_scipy_csr(dbform.assembly())

        # Biharmonic + IP，用 ipspace 组，再映射到 dspace 全局 dof。
        # 由于 p_phase 相同，两个 space 的 cell_to_dof 是一致的，直接相加即可。
        bh_form = BilinearForm(self.ipspace)
        bh_form.add_integrator(ScalarBiharmonicIntegrator(q=self.q))
        A_bh = _to_scipy_csr(bh_form.assembly())
        A_ip = _assemble_interior_penalty(self.ipspace, gamma=self.gamma, q=self.q)

        A_biharm_ip = (self.Gc * self.l0 ** 3 / 16) * (A_bh + A_ip)
        return A_dif_mass + A_biharm_ip

    def _assemble_phase_rhs(self):
        """RHS = ∫ 2H · v dx。"""
        H = self.H

        @barycentric
        def source_coef(bc, index):
            if hasattr(H, "shape") and H.ndim == 1:
                return 2 * H[..., None]
            return 2 * H

        lform = LinearForm(self.dspace)
        lform.add_integrator(ScalarSourceIntegrator(source=source_coef, q=self.q))
        return _to_numpy(lform.assembly()).reshape(-1)

    def _compute_stored_energy(self) -> float:
        """∫ g(d) · ψ⁺ dx（近似取每单元重心）。"""
        H = self.H
        if H is None:
            return 0.0
        cm = _to_numpy(self.mesh.entity_measure("cell"))
        # 每单元重心处的退化因子
        bc = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float64)
        try:
            d_val = _to_numpy(self.d(bc))
        except Exception:
            d_val = _to_numpy(self.d)
        d_val = d_val.reshape(-1)
        g = (1 - d_val) ** 2 + 1e-10
        H_arr = _to_numpy(H).reshape(-1)
        # 长度对齐（H 可能是每单元一个值）
        if g.shape[0] != H_arr.shape[0]:
            m = min(g.shape[0], H_arr.shape[0])
            g = g[:m]
            H_arr = H_arr[:m]
            cm = cm[:m]
        return float(np.sum(g * H_arr * cm))

    def _compute_dissipated_energy(self) -> float:
        """∫ Gc/(2 l0) · (d² + l0²·|∇d|²) dx。"""
        Gc, l0 = self.Gc, self.l0
        mesh = self.mesh
        qf = mesh.quadrature_formula(self.q, "cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        cm = _to_numpy(mesh.entity_measure("cell"))
        try:
            d_q = _to_numpy(self.d(bcs))  # (NC, NQ) or (NQ, NC)
            gd_q = _to_numpy(self.d.grad_value(bcs))  # (NC, NQ, GD)
        except Exception:
            return 0.0
        ws_np = _to_numpy(ws)
        # 对齐 (NC, NQ)
        if d_q.ndim == 2 and d_q.shape[0] == ws_np.shape[0]:
            d_q = d_q.T
            gd_q = np.moveaxis(gd_q, 0, 1) if gd_q.ndim == 3 and gd_q.shape[0] == ws_np.shape[0] else gd_q
        grad_sq = np.sum(gd_q ** 2, axis=-1)  # (NC, NQ)
        val = Gc / (2 * l0) * (d_q ** 2 + l0 ** 2 * grad_sq)  # (NC, NQ)
        return float(np.einsum("q,cq,c->", ws_np, val, cm))
