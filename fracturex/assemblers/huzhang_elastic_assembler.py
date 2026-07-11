# fracturex/assemblers/huzhang_elastic_assembler.py
"""Hu-Zhang 混合线弹性系统的块装配器。

装配鞍点系统 ``[[M(d), B], [B^T, 0]]``（应力 σ + 位移 u），并统一处理角点松弛
（corner relaxation）变换 ``M2=TM^T M TM``、``B2=TM^T B`` 与位移 Dirichlet / 应力本质
边界条件。

两种损伤本构放置（``formulation``）：
  - ``standard``：退化 ``g(d)`` 作用在应力质量块 ``M`` 上（积分器系数取 ``1/g``）；
    ``B`` 与 ``d`` 无关，可缓存复用。
  - ``effective_stress``：``g(d)`` 作用在耦合块 ``B`` 上（``div(gΨ)=g divΨ+Ψ∇g``，
    需补 ``∇g`` 链式项），``M`` 与 ``d`` 无关。

模块内的 ``_assemble_*`` / ``_*_chunk`` 自由函数为线程并行装配的 per-cell-slice 内核
（受环境变量 ``FRACTUREX_ASSEMBLY_PARALLEL`` / ``FRACTUREX_ASSEMBLY_NPROC`` 控制），
``_build_square_csr_scatter`` 等用于缓存 d-无关几何内核以加速重复装配。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
import inspect
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import bmat, coo_matrix, hstack as sp_hstack

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian, barycentric

from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.utils import process_coef_func
from fealpy.functionspace.functional import symmetry_index
from fealpy.typing import _S

from fracturex.cases.base import CaseBase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.base import DamageModelBase, DamageStateView
from fracturex.boundarycondition.huzhang_boundary_condition import (
    HuzhangBoundaryCondition,
    HuzhangStressBoundaryCondition,
)


@dataclass
class ElasticSystem:
    """一次弹性子问题的装配结果。

    Attributes:
        A: 系统矩阵（CSR 鞍点矩阵 ``[[M2, B2], [B2^T, 0]]``）。
        F: 右端向量 ``(gdof_sigma + gdof_u,)``。
        decode: 把解向量 ``X`` 映射为 ``(sigma_tilde, u, sigma_physical_or_none)`` 的回调。
        meta: 元信息 dict（``gdof_sigma``、``gdof_u``、``formulation``）。
    """

    A: Any
    F: Any
    decode: Callable[[Any], Tuple[Any, Any]]  # X -> (sigma_fun, u_fun)
    meta: dict


class _ScaledSigmaView:
    """Callable sigma view: sigma = g(d) * sigma_tilde."""

    def __init__(self, sigma_tilde_fun, damage, state):
        """Args:
            sigma_tilde_fun: 有效（变换）应力的可调用场 ``sigma_tilde(bcs, index)``。
            damage: 损伤模型，提供 ``coef_bary`` 计算退化 ``g(d)``。
            state: 当前状态视图（``d``、``sigma`` 等）。
        """
        self.sigma_tilde_fun = sigma_tilde_fun
        self.damage = damage
        self.state = state

    def __call__(self, bcs, index=None):
        """在重心坐标 ``bcs`` 处求物理应力 ``σ = g(d) · sigma_tilde``。

        Args:
            bcs: 重心坐标。
            index: 可选 cell 索引；部分后端会忽略它，此处显式广播以防越界。
        Returns:
            物理应力数组，最后一维为 Voigt 分量。
        """
        sig_tilde = self.sigma_tilde_fun(bcs, index=index)
        view = DamageStateView(
            d=self.state.d,
            sigma=self.state.sigma,
            u=self.state.u,
            r_hist=self.state.r_hist,
            H=self.state.H,
        )
        gd = self.damage.coef_bary(view, bcs, index=index)
        sig_tilde = bm.asarray(sig_tilde)
        gd = bm.asarray(gd)

        # Some FE backends may ignore `index` and return shape (1, NQ, ...).
        # Broadcast explicitly to avoid out-of-bound indexing in reaction postprocess.
        if index is not None:
            ncell = int(bm.asarray(index).reshape(-1).shape[0])
            if sig_tilde.ndim >= 1 and int(sig_tilde.shape[0]) == 1 and ncell > 1:
                reps = [ncell] + [1] * (sig_tilde.ndim - 1)
                sig_tilde = bm.tile(sig_tilde, reps)
            if gd.ndim >= 1 and int(gd.shape[0]) == 1 and ncell > 1:
                reps = [ncell] + [1] * (gd.ndim - 1)
                gd = bm.tile(gd, reps)
        return sig_tilde * gd[..., None]


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔型环境变量 ``name``（``1/true/yes/on`` 为真），缺省返回 ``default``。"""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """读取正整数环境变量 ``name``（下限 1），缺省或非法时返回 ``default``。"""
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return max(1, val)


def _default_assembly_nproc() -> int:
    """Use all logical CPUs by default (e.g. 10 on a 10-core machine)."""
    return max(1, int(os.cpu_count() or 1))


def _assemble_huzhang_m_fealpy_cell_slice(args):
    """Assemble M(d) on a cell slice using FEALPy's HuZhangStressIntegrator (same kernel as BilinearForm)."""
    space0, coef_d, c0, c1, q, i0, i1 = args
    sl = slice(int(i0), int(i1))
    integ = HuZhangStressIntegrator(coef=coef_d, q=q, lambda0=float(c0), lambda1=float(c1), index=sl)
    K = integ.assembly(space0)
    K = bm.asarray(K)
    cell2dof = bm.asarray(space0.cell_to_dof())[sl]
    nc = int(K.shape[0])
    ldof = int(K.shape[1])
    I = bm.broadcast_to(cell2dof[:, :, None], (nc, ldof, ldof)).reshape(-1)
    J = bm.broadcast_to(cell2dof[:, None, :], (nc, ldof, ldof)).reshape(-1)
    V = K.reshape(-1)
    return (
        bm.to_numpy(I).astype(np.int64, copy=False),
        bm.to_numpy(J).astype(np.int64, copy=False),
        bm.to_numpy(V).astype(np.float64, copy=False),
    )


def _fill_huzhang_mix_cell_matrix(mix_int, space_u, space_sigma, cellmeasure, index_sl, CM_out: np.ndarray) -> None:
    """Call HuZhangMixIntegrator.assembly_cell_matrix with fealpy version-tolerant kwargs."""
    fn = mix_int.assembly_cell_matrix
    attempts = (
        lambda: fn(space_u, space_sigma, cellmeasure=cellmeasure, index=index_sl, out=CM_out),
        lambda: fn(space_u, space_sigma, index=index_sl, cellmeasure=cellmeasure, out=CM_out),
        lambda: fn((space_u, space_sigma), cellmeasure=cellmeasure, index=index_sl, out=CM_out),
    )
    last_err: Optional[BaseException] = None
    for attempt in attempts:
        try:
            attempt()
            return
        except TypeError as e:
            last_err = e
            continue
    raise TypeError(f"HuZhangMixIntegrator.assembly_cell_matrix unsupported signature: {last_err}")


def _assemble_huzhang_mix_coupled_chunk(args):
    """装配 effective-stress 耦合混合块 ``B`` 在一段 cell 切片上的 COO 贡献。

    Args:
        args: 元组 ``(mix_int, space_u, space_sigma, i0, i1)``，``[i0, i1)`` 为 cell 区间。
    Returns:
        ``(I, J, V)`` 三个 1D numpy 数组（行索引、列索引、值），用于汇总成全局稀疏阵。
    """
    mix_int, space_u, space_sigma, i0, i1 = args
    sl = slice(int(i0), int(i1))
    nc = int(i1) - int(i0)
    mesh = space_sigma.mesh
    cellmeasure = bm.to_numpy(mesh.entity_measure("cell", index=sl))
    trial_ldof = int(space_u.number_of_local_dofs())
    test_ldof = int(space_sigma.number_of_local_dofs())
    CM = np.zeros((nc, test_ldof, trial_ldof), dtype=np.float64)
    _fill_huzhang_mix_cell_matrix(mix_int, space_u, space_sigma, cellmeasure, sl, CM)
    if CM.shape[1] == trial_ldof and CM.shape[2] == test_ldof:
        CM = np.transpose(CM, (0, 2, 1))
    cell2dof_s = np.asarray(space_sigma.cell_to_dof()[sl], dtype=np.int64)
    cell2dof_u = np.asarray(space_u.cell_to_dof()[sl], dtype=np.int64)
    ls, lu = test_ldof, trial_ldof
    I = np.broadcast_to(cell2dof_s[:, :, None], (nc, ls, lu)).reshape(-1)
    J = np.broadcast_to(cell2dof_u[:, None, :], (nc, ls, lu)).reshape(-1)
    V = np.asarray(CM.reshape(-1), dtype=np.float64)
    return I, J, V


def _parallel_TMt_M_TM(
    TM: Any,
    M: Any,
    *,
    max_workers: int,
    min_dim_parallel: int = 192,
    tm_csc: Any = None,
    tmt: Any = None,
) -> Any:
    """Compute ``M2 = TM.T @ M @ TM`` with column-block parallel matmul (two stages).

    ``tm_csc``/``tmt`` are the cached CSC / transposed-CSR views of the constant
    ``TM`` (S3); when provided the per-call format conversions are skipped.
    Falls back to a single-thread triple product on small systems or on failure.
    """
    TM_ss = TM.tocsr() if hasattr(TM, "tocsr") else TM
    M_ss = M.tocsr() if hasattr(M, "tocsr") else M
    n = int(TM_ss.shape[1])
    if max_workers <= 1 or n < min_dim_parallel:
        return TM_ss.T @ M_ss @ TM_ss
    try:
        TM_csc = tm_csc if tm_csc is not None else TM_ss.tocsc()
        nproc = min(int(max_workers), max(2, n // 48))
        edges = np.linspace(0, n, nproc + 1, dtype=int)
        tasks_m = []
        for k in range(nproc):
            j0, j1 = int(edges[k]), int(edges[k + 1])
            if j1 > j0:
                tasks_m.append((M_ss, TM_csc, j0, j1))

        def _mul_m_tm(args):
            M_loc, TM_csc_loc, j0, j1 = args
            return M_loc @ TM_csc_loc[:, j0:j1]

        workers = min(int(max_workers), len(tasks_m))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            blocks = list(pool.map(_mul_m_tm, tasks_m))
        P = sp_hstack(blocks).tocsr()

        TMt = tmt if tmt is not None else TM_ss.transpose().tocsr()
        tasks_t = []
        for k in range(nproc):
            k0, k1 = int(edges[k]), int(edges[k + 1])
            if k1 > k0:
                tasks_t.append((TMt, P, k0, k1))

        def _mul_tm_p(args):
            TMt_loc, P_loc, k0, k1 = args
            return TMt_loc @ P_loc[:, k0:k1]

        workers2 = min(int(max_workers), len(tasks_t))
        with ThreadPoolExecutor(max_workers=max(1, workers2)) as pool:
            blocks2 = list(pool.map(_mul_tm_p, tasks_t))
        return sp_hstack(blocks2).tocsr()
    except Exception as exc:
        print(f"[HuZhangElasticAssembler] parallel TM.T@M@TM failed, fallback to serial: {exc}")
        return TM_ss.T @ M_ss @ TM_ss


def _b2_gradg_K_chunk(args):
    """装配 ``∇g`` 链式修正项 ``∫ φ_u·(Ψ ∇g)`` 在一段 cell 切片上的 COO 贡献。

    Args:
        args: 元组 ``(ws, cm_chunk, phi_u_chunk, wvec_chunk, cell2dof_s_chunk,
            cell2dof_u_chunk)``——积分权重、cell 测度、位移基、含 ∇g 的应力加权向量及两套
            cell-to-dof 映射（均已按 cell 切片）。
    Returns:
        ``(I, J, V)`` 三个 1D numpy 数组（应力行、位移列、值）。
    """
    ws, cm_chunk, phi_u_chunk, wvec_chunk, cell2dof_s_chunk, cell2dof_u_chunk = args
    ws = bm.asarray(ws)
    cm_chunk = bm.asarray(cm_chunk)
    phi_u_chunk = bm.asarray(phi_u_chunk)
    wvec_chunk = bm.asarray(wvec_chunk)
    K = bm.einsum("q, c, cqmd, cqld -> clm", ws, cm_chunk, phi_u_chunk, wvec_chunk)
    nc = int(K.shape[0])
    ls = int(K.shape[1])
    lu = int(K.shape[2])
    cell2dof_s_chunk = bm.asarray(cell2dof_s_chunk)
    cell2dof_u_chunk = bm.asarray(cell2dof_u_chunk)
    I = bm.broadcast_to(cell2dof_s_chunk[:, :, None], (nc, ls, lu)).reshape(-1)
    J = bm.broadcast_to(cell2dof_u_chunk[:, None, :], (nc, ls, lu)).reshape(-1)
    V = K.reshape(-1)
    return (
        bm.to_numpy(I).astype(np.int64, copy=False),
        bm.to_numpy(J).astype(np.int64, copy=False),
        bm.to_numpy(V).astype(np.float64, copy=False),
    )


def _build_square_csr_scatter(cell2dof_np: np.ndarray, gdof: int):
    """Precompute CSR structure + scatter index for element matrices with a single
    dof map (square local blocks).

    Returns ``(indptr, indices, inv, nnz)``. ``inv[e]`` is the position in
    ``csr.data`` for the e-th flattened local entry, ordered as
    ``(cell, local_row, local_col)``. Per assembly:
    ``data = bincount(inv, weights=K.reshape(-1), minlength=nnz)``.
    """
    nc, ldof = int(cell2dof_np.shape[0]), int(cell2dof_np.shape[1])
    c2d = cell2dof_np.astype(np.int64, copy=False)
    I = np.broadcast_to(c2d[:, :, None], (nc, ldof, ldof)).reshape(-1)
    J = np.broadcast_to(c2d[:, None, :], (nc, ldof, ldof)).reshape(-1)
    coo = coo_matrix((np.zeros(I.shape[0], dtype=np.float64), (I, J)), shape=(gdof, gdof))
    csr = coo.tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    indptr = csr.indptr.astype(np.int64, copy=False)
    indices = csr.indices.astype(np.int64, copy=False)
    nnz = int(csr.nnz)
    rows = np.repeat(np.arange(gdof, dtype=np.int64), np.diff(indptr))
    csr_keys = rows * np.int64(gdof) + indices            # strictly ascending
    keys = I * np.int64(gdof) + J
    inv = np.searchsorted(csr_keys, keys)
    return indptr.astype(np.int32, copy=False), indices.astype(np.int32, copy=False), inv, nnz


class HuZhangElasticAssembler:
    """
    装配 HuZhang 混合线弹性系统：
      [ M(d)  B ]
      [ B^T   0 ]
    并统一处理 corner relaxation: M2=TM^T M TM, B2=TM^T B

    同时支持 piecewise 位移边界贡献到 σ 方程 RHS：
      F_sigma += TM^T * r_dirichlet
    """

    def __init__(
        self,
        discr: HuZhangDiscretization,
        case: CaseBase,
        damage: DamageModelBase,
        *,
        q: Optional[int] = None,
        formulation: str = "standard",
        assembly_parallel: Optional[bool] = None,
        assembly_nproc: Optional[int] = None,
    ):
        """Create elasticity block assembler for HuZhang mixed system.

        Inputs:
            discr: Discretization object with mesh/spaces/state.
            case: Case object with material/load/boundary data.
            damage: Damage model used to evaluate degradation `g(d)`.
            q: Optional quadrature order.
            formulation: `"standard"` or `"effective_stress"`.
        Output:
            None. Initializes cache fields; call `assemble(load)` to build system.
        """
        self.discr = discr
        self.case = case
        self.damage = damage
        self.q = q  # 若 None，integrator 内部用默认
        self.formulation = str(formulation).lower()
        self.assembly_parallel = _env_flag("FRACTUREX_ASSEMBLY_PARALLEL", True) if assembly_parallel is None else bool(assembly_parallel)
        self.assembly_nproc = (
            _env_int("FRACTUREX_ASSEMBLY_NPROC", _default_assembly_nproc())
            if assembly_nproc is None
            else max(1, int(assembly_nproc))
        )
        self._const_cache = None
        self._const_cache_key = None
        # Value-only M(d) kernel cache (A1+A2): constant geometric kernel Phi[c,q,l,m],
        # weights W[c,q]=cm*ws, and CSR pattern/scatter; only coef=1/g(d) varies.
        self._m_kernel = None
        self._m_kernel_key = None
        self._m_kernel_cache_enabled = _env_flag("FRACTUREX_M_KERNEL_CACHE", True)
        # Matrix-free stress block: return a MatrixFreeElasticOperator instead of an
        # assembled M2/bmat (standard form only). Saves the dominant p=3 stress-block
        # memory; only the iterative (fast/aux) solver paths consume it.
        self._matfree = _env_flag("FRACTUREX_ELASTIC_MATFREE", False)
        # Recompute (chunked) mode: do not cache basis phi; re-evaluate per cell-chunk
        # inside each matvec so peak memory is bounded well below M2 (real memory win;
        # CPU-slower). Default cache mode keeps phi (still avoids M2 + the (ldof,ldof) Phi).
        self._matfree_recompute = _env_flag("FRACTUREX_ELASTIC_MATFREE_RECOMPUTE", False)
        self._matfree_chunk = _env_int("FRACTUREX_MATFREE_CHUNK", 8192)
        # Per load-step cache (staggered inner iters share the same `load`):
        self._load_step_value: Optional[float] = None
        self._load_step_piecewise: Optional[List[Tuple[Any, Any, Any]]] = None
        self._r_dir_standard: Optional[Any] = None
        self._neumann_data_raw: Any = None
        self._neumann_uh_sig: Optional[Any] = None
        self._neumann_is_bd: Optional[Any] = None
        self._sigma_essential_mask_cache = {}

    def begin_load_step(self, load: float) -> None:
        """
        Precompute load-dependent data that do not change during staggered iterations
        at fixed `load` (e.g. Dirichlet shape for standard σ–u, Neumann/essential-sigma sparsity).

        Safe to skip: `assemble` falls back to the previous on-the-fly path.
        """
        load = float(load)
        self._load_step_value = load
        self._r_dir_standard = None
        self._neumann_data_raw = None
        self._neumann_uh_sig = None
        self._neumann_is_bd = None

        discr = self.discr
        case = self.case
        space0 = discr.space_sigma

        self._prepare_constant_blocks(load)
        pieces = case.dirichlet_pieces(load)
        self._load_step_piecewise = [(pc.threshold, pc.value, pc.direction) for pc in pieces]

        if self.formulation != "effective_stress" and space0 is not None and self._load_step_piecewise is not None:
            HBC = HuzhangBoundaryCondition(space=space0, q=self.q)
            self._r_dir_standard = HBC.displacement_boundary_condition(piecewise=self._load_step_piecewise)

        nd = case.neumann_data(load)
        self._neumann_data_raw = nd
        if nd is not None and space0 is not None:
            HSBC = HuzhangStressBoundaryCondition(space=space0, q=self.q)
            if isinstance(nd, (list, tuple)) and len(nd) > 0 and isinstance(nd[0], (list, tuple)) and len(nd[0]) in (3, 4):
                uh_sig, is_bd = HSBC.set_essential_bc(gd=None, piecewise=nd)
            else:
                thr, gd, coord = nd
                uh_sig, is_bd = HSBC.set_essential_bc(gd, threshold=thr, coord=coord)
            self._neumann_uh_sig = uh_sig
            self._neumann_is_bd = is_bd

    def assemble(self, load: float) -> ElasticSystem:
        """Assemble one elastic subproblem at given load and current damage.

        Input:
            load: Current scalar load value.
        Output:
            `ElasticSystem(A, F, decode, meta)` where `decode(X)` maps solution
            vector to FE functions `(sigma_tilde, u, sigma_physical_or_none)`.
        """
        discr = self.discr
        case = self.case
        damage = self.damage
        mesh = discr.mesh

        space0 = discr.space_sigma   # σ
        space1 = discr.space_u       # u
        state = discr.state

        assert mesh is not None and space0 is not None and space1 is not None and state is not None

        gdof0 = space0.number_of_global_dofs()
        gdof1 = space1.number_of_global_dofs()

        lam, mu = self._lame(case.model())
        n = mesh.geo_dimension()   # GD 
        c0 = 1.0/(2.0*mu)
        c1 = lam/(2.0*mu*(2.0*mu + n*lam))

        const = self._prepare_constant_blocks(load)
        TM = const["TM"]
        b_vec = const["b_vec"]
        if self.formulation == "effective_stress":
            M2 = const["M2_const"]
            B2 = self._assemble_coupled_B2(TM, state)
            if damage.debug:
                dmax = float(bm.max(bm.asarray(state.d[:])))
                if dmax > 1e-10:
                    diff = B2 - const["B2_const"]
                    rel = float(np.linalg.norm(diff.data) / max(np.linalg.norm(const["B2_const"].data), 1e-30))
                    print(f"[HuZhangElasticAssembler] effective_stress: ||B(d)-B0||/||B0||={rel:.3e}, dmax={dmax:.3e}")
        else:
            B2 = const["B2_const"]
            if self._matfree:
                # Matrix-free: never form M2/bmat (the dominant p=3 memory). Build a
                # LinearOperator applying [[TM' M(d) TM, B2],[B2', 0]] element-wise.
                from fracturex.utilfuc.matfree_elastic import MatrixFreeElasticOperator

                kernel = self._build_matfree_kernel(space0, c0, c1, state)
                A = MatrixFreeElasticOperator(
                    gdof_sigma=gdof0, gdof_u=gdof1,
                    TM=TM, TMT=const["TMt"], B2=B2, kernel=kernel,
                    recompute=self._matfree_recompute, chunk=self._matfree_chunk,
                )
            else:
                # ---- 1) M(d) ----
                M = self._assemble_M_block(space0, state, c0, c1)
                if self.assembly_parallel:
                    try:
                        M2 = _parallel_TMt_M_TM(
                            TM, M, max_workers=int(self.assembly_nproc),
                            tm_csc=const.get("TM_csc"), tmt=const.get("TMt"),
                        )
                    except Exception as exc:
                        print(f"[HuZhangElasticAssembler] parallel TM.T@M@TM raised, fallback serial: {exc}")
                        M2 = TM.T @ M @ TM
                else:
                    M2 = TM.T @ M @ TM
                A = bmat([[M2, B2],
                          [B2.T, None]], format="csr")

        if self.formulation == "effective_stress":
            A = bmat([[M2, B2],
                      [B2.T, None]], format="csr")

        # ---- 4) RHS: body force on u eqn ----
        b = b_vec

        # ---- 5) RHS: Dirichlet displacement contributes to sigma equation ----
        # Build HuzhangBoundaryCondition lazily: on the cached standard path below
        # (`_r_dir_standard` reuse) it is never used, so don't construct it there.
        if self._load_step_value is not None and float(load) == float(self._load_step_value) and self._load_step_piecewise is not None:
            piecewise = self._load_step_piecewise
        else:
            pieces = case.dirichlet_pieces(load)
            piecewise = [(pc.threshold, pc.value, pc.direction) for pc in pieces]

        if self.formulation == "effective_stress":
            @barycentric
            def coef_g_face(bcs, index=None):
                view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
                return damage.coef_bary(view, bcs, index=index)

            HBC = HuzhangBoundaryCondition(space=space0, q=self.q)
            r_dir = HBC.displacement_boundary_condition(piecewise=piecewise, coef=coef_g_face)
        else:
            if self._r_dir_standard is not None and self._load_step_value is not None and float(load) == float(self._load_step_value):
                r_dir = self._r_dir_standard
            else:
                HBC = HuzhangBoundaryCondition(space=space0, q=self.q)
                r_dir = HBC.displacement_boundary_condition(piecewise=piecewise)

        F = np.zeros(A.shape[0], dtype=float)
        F[:gdof0] = (TM.T @ r_dir).reshape(-1)   # sigma unknown is transformed variable
        F[gdof0:] = -b

        # ---- 6) optional: essential stress/traction boundary on sigma (Neumann edges but essential on sigma dof) ----
        # 如果 case.neumann_data 返回 (gd, threshold, coord)，就做消元
        if self._load_step_value is not None and float(load) == float(self._load_step_value) and self._neumann_data_raw is not None:
            nd = self._neumann_data_raw
        else:
            nd = case.neumann_data(load)
        if nd is not None:
            if self._neumann_uh_sig is not None and self._neumann_is_bd is not None and self._load_step_value is not None and float(load) == float(self._load_step_value):
                uh_sig, isBd = self._neumann_uh_sig, self._neumann_is_bd
            else:
                HSBC = HuzhangStressBoundaryCondition(space=space0, q=self.q)
                if isinstance(nd, (list, tuple)) and len(nd) > 0 and isinstance(nd[0], (list, tuple)) and len(nd[0]) in (3, 4):
                    uh_sig, isBd = HSBC.set_essential_bc(gd=None, piecewise=nd)
                else:
                    thr, gd, coord = nd
                    uh_sig, isBd = HSBC.set_essential_bc(gd, threshold=thr, coord=coord)

            if self._matfree:
                # MF-aware essential elimination: emulate F = F - A@uh; F[isbd]=uh[isbd]
                # and A <- T A T + Tbd via the operator's mask (no explicit T@A@T).
                total = A.shape[0]
                uh_global = np.zeros(total, dtype=float)
                isbd_global = np.zeros(total, dtype=bool)
                uh_global[:gdof0] = np.asarray(uh_sig).reshape(-1)
                isbd_global[:gdof0] = np.asarray(isBd).reshape(-1).astype(bool)
                F = F - A.apply_unmasked(uh_global)   # unmasked A, matches assembled path
                F[isbd_global] = uh_global[isbd_global]
                A = A.set_essential_mask(isbd_global)
            else:
                A, F = self.apply_sigma_essential_to_system(A, F, uh_sig, isBd, gdof0)


        # decode: map solution X -> (sigma,u) functions
        def decode(X):
            if isinstance(X, (tuple, list)):
                X = X[0]
            X = bm.asarray(X).reshape(-1)
            sig_tilde = X[:gdof0]
            u_vec = X[gdof0:]

            sigma_tilde = space0.function()
            sigma_tilde[:] = (TM @ sig_tilde).reshape(-1)

            u = space1.function()
            u[:] = u_vec
            if self.formulation == "effective_stress":
                sigma_physical = _ScaledSigmaView(sigma_tilde, damage, state)
                return sigma_tilde, u, sigma_physical
            return sigma_tilde, u, None
        
       


        meta = dict(
            gdof_sigma=int(gdof0),
            gdof_u=int(gdof1),
            formulation=self.formulation,
        )
        return ElasticSystem(A=A, F=F, decode=decode, meta=meta)

    def _assemble_M_block(self, space0, state, c0: float, c1: float):
        """装配损伤加权应力质量块 ``M(d)``（standard 形式），按可用性择优分派。

        依次尝试：缓存几何内核 → 线程并行 → 串行，前者失败时回退后者。

        Args:
            space0: Hu-Zhang 应力空间。
            state: 当前状态（提供 ``d`` 等用于 ``coef=1/g(d)``）。
            c0, c1: 柔度本构系数 ``c0=1/(2μ)``、``c1=λ/(2μ(2μ+nλ))``。
        Returns:
            ``(gdof_sigma, gdof_sigma)`` 的 scipy CSR 矩阵。
        """
        if self._m_kernel_cache_enabled:
            try:
                return self._assemble_M_block_cached(space0, state, c0, c1)
            except Exception as exc:
                print(f"[HuZhangElasticAssembler] cached M(d) kernel failed, fallback: {exc}")
        if self.assembly_parallel:
            try:
                return self._assemble_M_block_parallel(space0, state, c0, c1)
            except Exception as exc:
                print(f"[HuZhangElasticAssembler] parallel M(d) assembly failed, fallback to serial: {exc}")
        return self._assemble_M_block_serial(space0, state, c0, c1)

    def _build_m_kernel(self, space0, c0: float, c1: float):
        """Cache the d-independent material for the M(d) block (A1+A2).

        Replicates ``HuZhangStressIntegrator``'s element contraction with the coef
        factored out: ``Phi[c,q,l,m] = c0 Σ_d φ_lд φ_mд num_d - c1 trφ_l trφ_m``,
        so that ``M_e = Σ_q W[c,q] coef[c,q] Phi[c,q,l,m]`` with ``W=cm⊗ws``.
        Also caches the CSR pattern + scatter map. Keyed by (mesh, gdof, q, c0, c1).
        """
        mesh = self.discr.mesh
        assert mesh is not None
        p = int(space0.p)
        q = int(self.q) if self.q is not None else p + 3
        gdof = int(space0.number_of_global_dofs())
        key = (id(mesh), gdof, q, float(c0), float(c1))
        if self._m_kernel is not None and self._m_kernel_key == key:
            return self._m_kernel

        TD = int(mesh.top_dimension())
        cm = bm.asarray(mesh.entity_measure("cell"))
        qf = mesh.quadrature_formula(q, "cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        ws = bm.asarray(ws)
        phi = bm.asarray(space0.basis(bcs))             # (NC, NQ, ldof, nsym)
        if TD == 2:
            trphi = phi[..., 0] + phi[..., -1]
        else:
            trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]
        _, num = symmetry_index(d=TD, r=2)
        num = bm.asarray(num)

        # Coef-free element kernel (same contraction the integrator performs).
        Phi = c0 * bm.einsum("cqld, cqmd, d -> cqlm", phi, phi, num)
        Phi = Phi - c1 * bm.einsum("cql, cqm -> cqlm", trphi, trphi)
        W = bm.einsum("c, q -> cq", cm, ws)             # (NC, NQ)

        cell2dof = np.asarray(bm.to_numpy(space0.cell_to_dof()), dtype=np.int64)
        indptr, indices, inv, nnz = _build_square_csr_scatter(cell2dof, gdof)

        self._m_kernel_key = key
        self._m_kernel = {
            "bcs": bcs,
            "W": W,
            "Phi": Phi,
            "indptr": indptr,
            "indices": indices,
            "inv": inv,
            "nnz": nnz,
            "gdof": gdof,
        }
        return self._m_kernel

    def _build_matfree_kernel(self, space0, c0: float, c1: float, state) -> dict:
        """Element pieces for the matrix-free M(d) action — NO ``(ldof,ldof)`` Phi.

        Returns numpy arrays for :class:`MatrixFreeElasticOperator`:
            phi   (NC, NQ, ldof, nsym),  trphi (NC, NQ, ldof),  W (NC, NQ),
            num   (nsym,),  cell2dof (NC, ldof) int64,
            coef  (NC, NQ)  frozen 1/g(d) at quad points,  c0, c1.
        Unlike ``_build_m_kernel`` this keeps only ``phi``/``trphi`` (≈10x smaller
        than Phi) so the operator never materializes the dominant stress block.
        """
        mesh = self.discr.mesh
        assert mesh is not None
        p = int(space0.p)
        q = int(self.q) if self.q is not None else p + 3
        TD = int(mesh.top_dimension())

        cm = bm.asarray(mesh.entity_measure("cell"))
        qf = mesh.quadrature_formula(q, "cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        ws = bm.asarray(ws)
        _, num = symmetry_index(d=TD, r=2)
        W = bm.einsum("c, q -> cq", cm, ws)                  # (NC, NQ)
        cell2dof = np.asarray(bm.to_numpy(space0.cell_to_dof()), dtype=np.int64)
        # Recompute mode never stores phi/trphi -> also skip computing the full
        # (NC,...) basis here, so the builder's transient peak stays bounded.
        if not self._matfree_recompute:
            phi = bm.asarray(space0.basis(bcs))             # (NC, NQ, ldof, nsym)
            if TD == 2:
                trphi = phi[..., 0] + phi[..., -1]
            else:
                trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]

        # Freeze coef = 1/g(d) at quad points (constant within one staggered solve),
        # evaluated exactly as the integrator/assembled path does.
        @barycentric
        def coef_d(bcs_, index=None):
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u,
                                   r_hist=state.r_hist, H=state.H)
            return 1.0 / self.damage.coef_bary(view, bcs_, index=index)

        coef = process_coef_func(coef_d, bcs=bcs, mesh=mesh, etype="cell", index=_S)

        kernel = {
            "W": np.asarray(bm.to_numpy(W), dtype=np.float64),
            "num": np.asarray(bm.to_numpy(num), dtype=np.float64),
            "coef": np.asarray(bm.to_numpy(coef), dtype=np.float64),
            "cell2dof": cell2dof,
            "c0": float(c0),
            "c1": float(c1),
        }
        if self._matfree_recompute:
            # chunked recompute: hand the space + quad points; do NOT keep phi/trphi.
            kernel["space0"] = space0
            kernel["bcs"] = bcs
        else:
            kernel["phi"] = np.asarray(bm.to_numpy(phi), dtype=np.float64)
            kernel["trphi"] = np.asarray(bm.to_numpy(trphi), dtype=np.float64)
        return kernel

    def _assemble_M_block_cached(self, space0, state, c0: float, c1: float):
        """用缓存的 d-无关几何内核装配 ``M(d)``：只重算系数 ``1/g(d)`` 并 bincount 散射。

        Args/Returns 同 :meth:`_assemble_M_block`；要求 ``_build_m_kernel`` 的缓存命中（否则重建）。
        """
        from scipy.sparse import csr_matrix

        K = self._build_m_kernel(space0, c0, c1)
        mesh = self.discr.mesh

        @barycentric
        def coef_d(bcs, index=None):
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
            return 1.0 / self.damage.coef_bary(view, bcs, index=index)

        # Evaluate coef exactly as the integrator does (process_coef_func + full index).
        val = process_coef_func(coef_d, bcs=K["bcs"], mesh=mesh, etype="cell", index=_S)
        val = bm.asarray(val)                            # (NC, NQ)

        Ke = bm.einsum("cq, cq, cqlm -> clm", K["W"], val, K["Phi"])
        Ke = np.asarray(bm.to_numpy(Ke), dtype=np.float64)
        data = np.bincount(K["inv"], weights=Ke.reshape(-1), minlength=K["nnz"])
        return csr_matrix((data, K["indices"], K["indptr"]), shape=(K["gdof"], K["gdof"]))

    def _assemble_M_block_serial(self, space0, state, c0: float, c1: float):
        """串行装配 ``M(d)``：直接走 FEALPy ``BilinearForm`` + ``HuZhangStressIntegrator``。

        Args/Returns 同 :meth:`_assemble_M_block`。
        """
        @barycentric
        def coef_d(bcs, index=None):
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
            return 1.0 / self.damage.coef_bary(view, bcs, index=index)

        bformM = BilinearForm(space0)
        bformM.add_integrator(HuZhangStressIntegrator(coef=coef_d, lambda0=c0, lambda1=c1))
        return bformM.assembly().to_scipy().tocsr()

    def _assemble_M_block_parallel(self, space0, state, c0: float, c1: float):
        """线程并行装配 ``M(d)``：按 cell 切片分块调用 FEALPy 内核再汇总成 COO/CSR。

        分块数为 ``min(assembly_nproc, NC)``；退化为单块时回退串行。
        Args/Returns 同 :meth:`_assemble_M_block`。
        """
        mesh = self.discr.mesh
        assert mesh is not None

        nc = int(mesh.number_of_cells())

        @barycentric
        def coef_d(bcs, index=None):
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
            return 1.0 / self.damage.coef_bary(view, bcs, index=index)

        nproc = min(int(self.assembly_nproc), nc)
        if nproc <= 1 or nc < 2:
            return self._assemble_M_block_serial(space0, state, c0, c1)

        edges = np.linspace(0, nc, nproc + 1, dtype=int)
        tasks = []
        for k in range(nproc):
            i0, i1 = int(edges[k]), int(edges[k + 1])
            if i1 <= i0:
                continue
            tasks.append((space0, coef_d, c0, c1, self.q, i0, i1))

        workers = min(int(self.assembly_nproc), len(tasks))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            parts = list(pool.map(_assemble_huzhang_m_fealpy_cell_slice, tasks))

        I = np.concatenate([p[0] for p in parts], axis=0)
        J = np.concatenate([p[1] for p in parts], axis=0)
        V = np.concatenate([p[2] for p in parts], axis=0)
        gdof = int(space0.number_of_global_dofs())
        return coo_matrix((V, (I, J)), shape=(gdof, gdof)).tocsr()

    def _prepare_constant_blocks(self, load: float):
        """
        Cache d-independent components:
        - TM, B2
        - body-force vector on displacement equation
        """
        discr = self.discr
        case = self.case
        mesh = discr.mesh
        space0 = discr.space_sigma
        space1 = discr.space_u

        assert mesh is not None and space0 is not None and space1 is not None

        key = (
            id(mesh),
            int(space0.number_of_global_dofs()),
            int(space1.number_of_global_dofs()),
        )
        if self._const_cache is not None and self._const_cache_key == key:
            return self._const_cache

        bformB = BilinearForm((space1, space0))
        bformB.add_integrator(HuZhangMixIntegrator())
        B = bformB.assembly().to_scipy().tocsr()
        TM = space0.TM.to_scipy().tocsr()
        B2 = TM.T @ B

        # M2_const is consumed ONLY by the effective_stress branch of assemble(); for the
        # standard formulation it is dead weight (and its assembly + triple product is a
        # large transient memory peak). Build it lazily only when needed.
        if self.formulation == "effective_stress":
            lam, mu = self._lame(case.model())
            n = mesh.geo_dimension()
            c0 = 1.0/(2.0*mu)
            c1 = lam/(2.0*mu*(2.0*mu + n*lam))
            bformM_const = BilinearForm(space0)
            bformM_const.add_integrator(HuZhangStressIntegrator(coef=1.0, lambda0=c0, lambda1=c1))
            M_const = bformM_const.assembly().to_scipy().tocsr()
            M2_const = TM.T @ M_const @ TM
        else:
            M2_const = None

        lform = LinearForm(space1)

        @cartesian
        def f_body(x, index=None):
            return case.body_force(x)

        lform.add_integrator(VectorSourceIntegrator(source=f_body))
        b_vec = lform.assembly()
        b_vec = np.asarray(b_vec, dtype=float).reshape(-1)

        self._const_cache_key = key
        self._const_cache = {
            "TM": TM,
            # Cached format views of the constant corner-relaxation transform (S3):
            # the parallel triple product reuses these instead of re-converting TM.
            "TM_csc": TM.tocsc(),
            "TMt": TM.transpose().tocsr(),
            "B2_const": B2,
            "M2_const": M2_const,
            "b_vec": b_vec,
        }
        return self._const_cache

    def _assemble_coupled_B2(self, TM, state):
        """Assemble damage-coupled mixed block `B2` for effective-stress form.

        Inputs:
            TM: Stress-space relaxation transform matrix.
            state: Current solution state containing `d`.
        Output:
            Coupled sparse block matrix `B2`.
        """
        @barycentric
        def coef_g(bcs, index=None):
            view = DamageStateView(d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H)
            return self.damage.coef_bary(view, bcs, index=index)

        # Different fealpy versions expose coefficient in different ways.
        # Force binding the coefficient after construction to avoid silent no-op.
        try:
            params = inspect.signature(HuZhangMixIntegrator.__init__).parameters
            if "coef" in params:
                mix_int = HuZhangMixIntegrator(coef=coef_g)
            else:
                mix_int = HuZhangMixIntegrator()
        except Exception:
            mix_int = HuZhangMixIntegrator()

        setattr(mix_int, "coef", coef_g)
        setattr(mix_int, "_coef", coef_g)
        if hasattr(mix_int, "set_coef"):
            try:
                mix_int.set_coef(coef_g)
            except Exception:
                pass

        def _assemble_B_gdiv_csr(mix_int_local):
            if self.assembly_parallel:
                try:
                    return self._assemble_B_mix_gdiv_parallel(mix_int_local)
                except Exception as exc:
                    print(f"[HuZhangElasticAssembler] parallel B(g·div) assembly failed, fallback to serial: {exc}")
                    bf = BilinearForm((self.discr.space_u, self.discr.space_sigma))
                    bf.add_integrator(mix_int_local)
                    return bf.assembly().to_scipy().tocsr()
            bf = BilinearForm((self.discr.space_u, self.discr.space_sigma))
            bf.add_integrator(mix_int_local)
            return bf.assembly().to_scipy().tocsr()

        def _b_main_branch():
            B_loc = _assemble_B_gdiv_csr(mix_int)
            return TM.T @ B_loc

        def _b_corr_branch():
            return self._assemble_B2_gradg_chainrule(TM, state)

        # Overlap g·div block assembly + TM transform with ∇g correction assembly.
        if self.assembly_parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_main = pool.submit(_b_main_branch)
                fut_corr = pool.submit(_b_corr_branch)
                B_main = fut_main.result()
                B_corr = fut_corr.result()
        else:
            B_main = _b_main_branch()
            B_corr = _b_corr_branch()

        # div(g Ψ) = g div Ψ + Ψ ∇g (row divergence). HuZhangMixIntegrator only
        # assembles ∫ g (div Ψ)·φ; add the missing ∫ φ·(Ψ ∇g) for spatially varying g(d_h).
        if B_corr is not None:
            if getattr(self.damage, "debug", False):
                n_main = float(np.linalg.norm(B_main.data))
                n_corr = float(np.linalg.norm(B_corr.data))
                print(
                    "[HuZhangElasticAssembler] effective_stress B2: "
                    f"||TM^T B_gdiv||={n_main:.3e}, ||TM^T B_gradg||={n_corr:.3e}"
                )
            return B_main + B_corr
        return B_main

    def _assemble_B_mix_gdiv_parallel(self, mix_int) -> Any:
        """Parallel assembly of mixed block B from HuZhangMixIntegrator (effective stress)."""
        space_sigma = self.discr.space_sigma
        space_u = self.discr.space_u
        mesh = self.discr.mesh
        assert mesh is not None
        nc = int(mesh.number_of_cells())
        gdof_s = int(space_sigma.number_of_global_dofs())
        gdof_u = int(space_u.number_of_global_dofs())
        nproc = min(int(self.assembly_nproc), nc)
        if nproc <= 1 or nc < 2:
            I, J, V = _assemble_huzhang_mix_coupled_chunk((mix_int, space_u, space_sigma, 0, nc))
        else:
            edges = np.linspace(0, nc, nproc + 1, dtype=int)
            tasks = []
            for k in range(nproc):
                i0, i1 = int(edges[k]), int(edges[k + 1])
                if i1 <= i0:
                    continue
                tasks.append((mix_int, space_u, space_sigma, i0, i1))
            workers = min(int(self.assembly_nproc), len(tasks))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                parts = list(pool.map(_assemble_huzhang_mix_coupled_chunk, tasks))
            I = np.concatenate([p[0] for p in parts], axis=0)
            J = np.concatenate([p[1] for p in parts], axis=0)
            V = np.concatenate([p[2] for p in parts], axis=0)
        B = coo_matrix((V, (I, J)), shape=(gdof_s, gdof_u)).tocsr()
        return B

    def _assemble_B2_gradg_chainrule(self, TM, state):
        """
        B_corr[i,j] = ∫_Ω φ_j · (Ψ_i ∇g) dx with g = g(d_h), same dof layout as
        BilinearForm((space_u, space_sigma)) + B2 = TM.T @ B.
        """
        damage = self.damage
        gprime_fn = getattr(damage, "degradation_grad", None)
        if gprime_fn is None:
            return None

        space_sigma = self.discr.space_sigma
        space_u = self.discr.space_u
        mesh = self.discr.mesh
        p = int(space_sigma.p)
        qord = int(self.q) if self.q is not None else p + 3

        qf = mesh.quadrature_formula(qord, "cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        ws = bm.asarray(ws)
        cm = bm.asarray(mesh.entity_measure("cell"))

        psi = bm.asarray(space_sigma.basis(bcs))  # (NC, NQ, ls, 3) Voigt xx,xy,yy
        phi_u = bm.asarray(space_u.basis(bcs))  # (NC, NQ, lu, 2)

        d_vals = bm.asarray(state.d(bcs))
        grad_d = bm.asarray(self.discr.space_d.grad_value(state.d, bcs))

        gp = bm.asarray(gprime_fn(d_vals))
        # grad_d: (NC, NQ, GD); psi: (NC, NQ, ls, 3) — broadcast ∇g along stress dof axis
        gx = (gp * grad_d[..., 0])[..., None]
        gy = (gp * grad_d[..., 1])[..., None]

        sxx, sxy, syy = psi[..., 0], psi[..., 1], psi[..., 2]
        wx = sxx * gx + sxy * gy
        wy = sxy * gx + syy * gy
        wvec = bm.stack([wx, wy], axis=-1)  # (NC, NQ, ls, 2)

        cell2dof_s = bm.asarray(space_sigma.cell_to_dof())
        cell2dof_u = bm.asarray(space_u.cell_to_dof())
        NC = int(mesh.number_of_cells())
        gdof_s = space_sigma.number_of_global_dofs()
        gdof_u = space_u.number_of_global_dofs()

        if self.assembly_parallel:
            try:
                nproc = min(int(self.assembly_nproc), NC)
                if nproc <= 1 or NC < 2:
                    I, J, V = _b2_gradg_K_chunk((ws, cm, phi_u, wvec, cell2dof_s, cell2dof_u))
                else:
                    edges = np.linspace(0, NC, nproc + 1, dtype=int)
                    tasks = []
                    for k in range(nproc):
                        i0, i1 = int(edges[k]), int(edges[k + 1])
                        if i1 <= i0:
                            continue
                        tasks.append(
                            (
                                ws,
                                cm[i0:i1],
                                phi_u[i0:i1],
                                wvec[i0:i1],
                                cell2dof_s[i0:i1],
                                cell2dof_u[i0:i1],
                            )
                        )
                    workers = min(int(self.assembly_nproc), len(tasks))
                    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                        parts = list(pool.map(_b2_gradg_K_chunk, tasks))
                    I = np.concatenate([p[0] for p in parts], axis=0)
                    J = np.concatenate([p[1] for p in parts], axis=0)
                    V = np.concatenate([p[2] for p in parts], axis=0)
                Bc = coo_matrix((V, (I, J)), shape=(gdof_s, gdof_u)).tocsr()
                return TM.T @ Bc
            except Exception as exc:
                print(f"[HuZhangElasticAssembler] parallel B2_gradg assembly failed, fallback to serial: {exc}")

        K = bm.einsum("q, c, cqmd, cqld -> clm", ws, cm, phi_u, wvec)
        NCk, ls, lu = (int(K.shape[0]), int(K.shape[1]), int(K.shape[2]))
        I = bm.broadcast_to(cell2dof_s[:, :, None], (NCk, ls, lu))
        J = bm.broadcast_to(cell2dof_u[:, None, :], (NCk, ls, lu))
        Bc = coo_matrix(
            (
                np.asarray(K.reshape(-1), dtype=float),
                (np.asarray(I.reshape(-1), dtype=np.int64), np.asarray(J.reshape(-1), dtype=np.int64)),
            ),
            shape=(gdof_s, gdof_u),
        ).tocsr()
        return TM.T @ Bc

    def apply_sigma_essential_to_system(self, A, F, uh_sigma, isBd_sigma, gdof_sigma: int):
        """
        把 σ 的本质边界值（uh_sigma, isBd_sigma）扩展到全系统并消元。
        这段就是你之前脚本里那段通用逻辑的封装版。
        """
        from scipy.sparse import spdiags
  
        total = A.shape[0]
        uh_global = np.zeros(total, dtype=float)
        isbd_global = np.zeros(total, dtype=bool)

        uh_global[:gdof_sigma] = np.asarray(uh_sigma).reshape(-1)
        isbd_global[:gdof_sigma] = np.asarray(isBd_sigma).reshape(-1).astype(bool)

        # F = F - A u_known
        F = F - A @ uh_global
        # enforce
        F[isbd_global] = uh_global[isbd_global]

        # A modification
        isbd_key = np.asarray(isbd_global, dtype=np.bool_).tobytes()
        cache_key = (int(total), isbd_key)
        cached = self._sigma_essential_mask_cache.get(cache_key)
        if cached is None:
            bdIdx = np.zeros(total, dtype=int)
            bdIdx[isbd_global] = 1
            Tbd = spdiags(bdIdx, 0, total, total)
            T = spdiags(1 - bdIdx, 0, total, total)
            self._sigma_essential_mask_cache[cache_key] = (T, Tbd)
        else:
            T, Tbd = cached
        A = T @ A @ T + Tbd

        return A, F

    @staticmethod
    def _lame(model):
        """Resolve Lamé parameters from material model.

        Input:
            model: Object with `(lam, mu)` or `(lambda0, lambda1)` or `(E, nu)`.
        Output:
            Tuple `(lam, mu)` as floats.
        """
        # 兼容你工程里 lambda0/lambda1 或 lam/mu
        if hasattr(model, "lam") and hasattr(model, "mu"):
            return float(model.lam), float(model.mu)
        if hasattr(model, "lambda0") and hasattr(model, "lambda1"):
            return float(model.lambda0), float(model.lambda1)
        if hasattr(model, "E") and hasattr(model, "nu"):
            E = float(model.E); nu = float(model.nu)
            mu = E/(2*(1+nu))
            lam = E*nu/((1+nu)*(1-2*nu))
            return lam, mu
        raise AttributeError("model must provide (lam,mu) or (lambda0,lambda1) or (E,nu)")
