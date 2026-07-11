# fracturex/assemblers/phasefield_assembler.py
"""相场（phase-field）损伤子问题的装配器（积分点历史场版本）。

每次装配在当前积分点上先更新历史场 ``H``，再组装相场线性增量系统 ``A dd = F``：
  - ``A = A_const + A_hist``：``A_const`` 为扩散 + 与 H 无关的质量项（AT1/AT2 下可缓存复用），
    ``A_hist`` 为含历史场 ``g''(d)·H`` 的质量项；
  - ``F = rhs - A d_old``（残差/增量形式），``decode(dd)`` 返回裁剪到 [0,1] 的新损伤场。

模块内自由函数为线程并行装配的 per-cell-slice 内核及显式积分回退实现（当 FEALPy 积分器
签名不兼容时使用 ``_explicit_scalar_*`` 直接 einsum 装配）；并行/进程数受环境变量
``FRACTUREX_ASSEMBLY_PARALLEL`` / ``FRACTUREX_ASSEMBLY_NPROC`` 控制，
``FRACTUREX_AHIST_KERNEL_CACHE`` 控制 d-无关质量内核缓存。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.sparse import coo_matrix

from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric
from fealpy.sparse.csr_tensor import CSRTensor
from fealpy.fem import (
    BilinearForm,
    LinearForm,
    ScalarDiffusionIntegrator,
    ScalarMassIntegrator,
    ScalarSourceIntegrator,
    DirichletBC,
)
from fealpy.utils import process_coef_func
from fealpy.typing import _S

from fracturex.assemblers.huzhang_elastic_assembler import _build_square_csr_scatter


@dataclass
class PhaseFieldSystem:
    """一次相场线性增量子问题的装配结果。

    Attributes:
        A: 系统矩阵（CSRTensor 或 scipy 稀疏阵）。
        F: 右端向量（残差形式 ``rhs - A d_old``）。
        decode: 把增量解 ``dd`` 映射为裁剪后新损伤场 ``d_new`` 的回调。
        meta: 元信息 dict（模型类型、退化类型、load、gdof、q、Gc、l0 等）。
    """

    A: Any
    F: Any
    decode: Callable[[Any], Any]   # dd -> d_new
    meta: Dict[str, Any]


def _default_assembly_nproc() -> int:
    """默认并行装配进程数：逻辑 CPU 数（下限 1）。"""
    return max(1, int(os.cpu_count() or 1))


def _bl_integrator_coef(integ: Any):
    """取双线性积分器的系数属性（``coef`` 或 ``c``），都没有则返回 None。"""
    return getattr(integ, "coef", getattr(integ, "c", None))


def _source_integrand(src_int: Any):
    """取源项积分器的被积函数属性（``source`` 或 ``f``），都没有则返回 None。"""
    return getattr(src_int, "source", getattr(src_int, "f", None))


def _space_ldof(space) -> int:
    """返回有限元空间的单元局部自由度数（兼容两种 API）。"""
    if hasattr(space, "number_of_local_dofs"):
        return int(space.number_of_local_dofs())
    return int(space.dof.number_of_local_dofs())


def _to_numpy_float_array(val: Any) -> np.ndarray:
    """把标量或后端张量统一转成 ``float64`` numpy 数组。"""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return np.array(float(val), dtype=np.float64)
    return np.asarray(bm.to_numpy(val), dtype=np.float64)


def _coef_at_quadrature(mesh, coef: Any, bcs, index_sl) -> Any:
    """Evaluate coefficient at quadrature points (FEALPy-style callable rules)."""
    if coef is None:
        return None
    if callable(coef):
        if hasattr(coef, "coordtype"):
            ct = getattr(coef, "coordtype")
            if ct == "cartesian":
                ps = mesh.bc_to_point(bcs, index=index_sl)
                val = coef(ps)
            elif ct == "barycentric":
                val = coef(bcs, index=index_sl)
            else:
                ps = mesh.bc_to_point(bcs, index=index_sl)
                val = coef(ps)
        else:
            ps = mesh.bc_to_point(bcs, index=index_sl)
            val = coef(ps)
    else:
        val = coef
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    return _to_numpy_float_array(val)


def _cell_quadrature(mesh, q: int):
    """Return cell quadrature points/weights without triggering FEALPy deprecation warnings."""
    if hasattr(mesh, "quadrature_formula"):
        qf = mesh.quadrature_formula(int(q), "cell")
    else:
        qf = mesh.integrator(int(q), "cell")
    return qf.get_quadrature_points_and_weights()


def _as_q_cell_values(values: Any, *, nc: int, nq: int, name: str) -> np.ndarray:
    """Normalize coefficient samples to FEALPy's legacy quadrature layout: (NQ, NC)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.shape == (nq, nc):
        return arr
    if arr.shape == (nc, nq):
        return arr.T
    if arr.shape == (1, nq):
        return np.broadcast_to(arr.T, (nq, nc))
    if arr.shape == (nq, 1):
        return np.broadcast_to(arr, (nq, nc))
    if arr.shape == (1, nc):
        return np.broadcast_to(arr, (nq, nc))
    if arr.shape == (nc, 1):
        return np.broadcast_to(arr.T, (nq, nc))
    raise ValueError(f"Unsupported {name} shape {arr.shape}, expected (NQ, NC) or (NC, NQ).")


def _phi_grad_nq_nc_ld(space, bcs, index_sl, nc: int) -> np.ndarray:
    """计算梯度基并整理为 ``(NQ, NC, ldof, GD)`` 布局（供显式扩散装配的 einsum 用）。"""
    phi_g = bm.asarray(space.grad_basis(bcs, index=index_sl))
    phi_c = np.asarray(bm.to_numpy(_as_cell_first(phi_g, nc)), dtype=np.float64)
    return np.transpose(phi_c, (1, 0, 2, 3))


def _phi_basis_nq_nc_ld(space, bcs, index_sl, nc: int) -> np.ndarray:
    """计算标量基并整理为 ``(NQ, NC, ldof)`` 布局（供显式质量/源项装配的 einsum 用）。"""
    phi_b = bm.asarray(space.basis(bcs, index=index_sl))
    phi_c = np.asarray(bm.to_numpy(_as_cell_first(phi_b, nc)), dtype=np.float64)
    return np.transpose(phi_c, (1, 0, 2))


def _explicit_scalar_diffusion_cell(space, coef: Any, q: int, index_sl, out_cm: np.ndarray) -> None:
    """显式装配标量扩散单元矩阵 ``∫ coef ∇φ·∇φ``（FEALPy 积分器不兼容时的回退）。

    支持 coef 为 None / 标量 / ``(NC,)`` / ``(NQ,NC)`` / 各向异性 ``(GD,GD)`` 等多种形状，
    结果**累加**到 ``out_cm``（``(NC, ldof, ldof)``，原地修改，无返回值）。
    """
    mesh = space.mesh
    cellmeasure = np.asarray(bm.to_numpy(mesh.entity_measure("cell", index=index_sl)), dtype=np.float64)
    nc = int(cellmeasure.shape[0])
    bcs, ws = _cell_quadrature(mesh, int(q))
    ws = np.asarray(bm.to_numpy(ws), dtype=np.float64)
    nq = int(ws.shape[0])
    phi_q = _phi_grad_nq_nc_ld(space, bcs, index_sl, nc)

    if coef is None:
        out_cm += np.einsum("q,qcid,qcjd,c->cij", ws, phi_q, phi_q, cellmeasure, optimize=True)
        return

    cvals = _coef_at_quadrature(mesh, coef, bcs, index_sl)
    if isinstance(cvals, float):
        out_cm += cvals * np.einsum("q,qcid,qcjd,c->cij", ws, phi_q, phi_q, cellmeasure, optimize=True)
        return

    cvals = np.asarray(cvals, dtype=np.float64)
    if cvals.shape == (nc,):
        out_cm += np.einsum("q,c,qcid,qcjd,c->cij", ws, cvals, phi_q, phi_q, cellmeasure, optimize=True)
    elif cvals.ndim == 2:
        c_qc = _as_q_cell_values(cvals, nc=nc, nq=nq, name="diffusion coef")
        out_cm += np.einsum("q,qc,qcid,qcjd,c->cij", ws, c_qc, phi_q, phi_q, cellmeasure, optimize=True)
    else:
        GD = int(phi_q.shape[-1])
        if cvals.shape == (GD, GD):
            out_cm += np.einsum("q,dn,qcin,qcjd,c->cij", ws, cvals, phi_q, phi_q, cellmeasure, optimize=True)
        elif cvals.ndim == 3 and cvals.shape[-2:] == (GD, GD):
            out_cm += np.einsum("q,cdn,qcin,qcjd,c->cij", ws, cvals, phi_q, phi_q, cellmeasure, optimize=True)
        elif cvals.ndim == 4:
            if cvals.shape[:2] == (nc, nq):
                cvals = np.transpose(cvals, (1, 0, 2, 3))
            out_cm += np.einsum("q,qcdn,qcin,qcjd,c->cij", ws, cvals, phi_q, phi_q, cellmeasure, optimize=True)
        else:
            raise ValueError(f"Unsupported diffusion coef shape {cvals.shape}")


def _explicit_scalar_mass_cell(space, coef: Any, q: int, index_sl, out_cm: np.ndarray) -> None:
    """显式装配标量质量单元矩阵 ``∫ coef φ φ``（FEALPy 积分器不兼容时的回退）。

    coef 支持 None / 标量 / ``(NC,)`` / ``(NQ,NC)``；结果**累加**到 ``out_cm``（原地，无返回）。
    """
    mesh = space.mesh
    cellmeasure = np.asarray(bm.to_numpy(mesh.entity_measure("cell", index=index_sl)), dtype=np.float64)
    nc = int(cellmeasure.shape[0])
    bcs, ws = _cell_quadrature(mesh, int(q))
    ws = np.asarray(bm.to_numpy(ws), dtype=np.float64)
    nq = int(ws.shape[0])
    phi_q = _phi_basis_nq_nc_ld(space, bcs, index_sl, nc)

    if coef is None:
        out_cm += np.einsum("q,qci,qcj,c->cij", ws, phi_q, phi_q, cellmeasure, optimize=True)
        return

    cvals = _coef_at_quadrature(mesh, coef, bcs, index_sl)
    if isinstance(cvals, float):
        out_cm += cvals * np.einsum("q,qci,qcj,c->cij", ws, phi_q, phi_q, cellmeasure, optimize=True)
        return

    cvals = np.asarray(cvals, dtype=np.float64)
    if cvals.shape == (nc,):
        out_cm += np.einsum("q,c,qci,qcj,c->cij", ws, cvals, phi_q, phi_q, cellmeasure, optimize=True)
    else:
        c_qc = _as_q_cell_values(cvals, nc=nc, nq=nq, name="mass coef")
        out_cm += np.einsum("q,qc,qci,qcj,c->cij", ws, c_qc, phi_q, phi_q, cellmeasure, optimize=True)


def _explicit_scalar_source_cell(space, f: Any, q: int, index_sl, out_bb: np.ndarray) -> None:
    """显式装配标量源项单元向量 ``∫ f φ``（FEALPy 积分器不兼容时的回退）。

    f 支持标量 / ``(NC,)`` / ``(NQ,NC)``；结果**累加**到 ``out_bb``（``(NC, ldof)``，原地，无返回）。
    """
    mesh = space.mesh
    cellmeasure = np.asarray(bm.to_numpy(mesh.entity_measure("cell", index=index_sl)), dtype=np.float64)
    nc = int(cellmeasure.shape[0])
    bcs, ws = _cell_quadrature(mesh, int(q))
    ws = np.asarray(bm.to_numpy(ws), dtype=np.float64)
    nq = int(ws.shape[0])
    phi_q = _phi_basis_nq_nc_ld(space, bcs, index_sl, nc)

    val = _coef_at_quadrature(mesh, f, bcs, index_sl)
    if isinstance(val, float):
        out_bb += val * np.einsum("q,qci,c->ci", ws, phi_q, cellmeasure, optimize=True)
        return

    val = np.asarray(val, dtype=np.float64)
    if val.shape == (nc,):
        out_bb += np.einsum("q,c,qci,c->ci", ws, val, phi_q, cellmeasure, optimize=True)
    else:
        val_qc = _as_q_cell_values(val, nc=nc, nq=nq, name="source value")
        out_bb += np.einsum("q,qc,qci,c->ci", ws, val_qc, phi_q, cellmeasure, optimize=True)


def _call_scalar_integrator_cell_matrix(integrator, space, index_sl, out_cm: np.ndarray, *, accumulate: bool) -> None:
    """调用 FEALPy 积分器的 ``assembly_cell_matrix``，容忍跨版本签名差异。

    Args:
        integrator: 标量双线性积分器。
        space: 有限元空间。
        index_sl: cell 切片。
        out_cm: 输出单元矩阵 ``(NC, ldof, ldof)``，原地写入/累加。
        accumulate: True 累加到 ``out_cm``，False 覆盖。
    Raises:
        AttributeError: 积分器无 ``assembly_cell_matrix``。
        TypeError: 所有签名尝试均失败。
    """
    fn = getattr(integrator, "assembly_cell_matrix", None)
    if fn is None:
        raise AttributeError("assembly_cell_matrix")
    mesh = space.mesh
    cm = np.asarray(bm.to_numpy(mesh.entity_measure("cell", index=index_sl)), dtype=np.float64)
    last_err: Optional[BaseException] = None
    for thunk in (
        lambda: fn(space, index=index_sl, out=out_cm, cellmeasure=cm),
        lambda: fn(space, index=index_sl, cellmeasure=cm, out=out_cm),
        lambda: fn(space, index=index_sl, out=out_cm),
    ):
        try:
            thunk()
            return
        except TypeError as e:
            last_err = e
    try:
        R = fn(space, index=index_sl)
        R = np.asarray(R, dtype=out_cm.dtype)
        if accumulate:
            out_cm += R
        else:
            out_cm[:] = R
    except Exception as e:
        raise TypeError(f"assembly_cell_matrix incompatible: {last_err!r}; return fallbacks failed: {e!r}") from e


def _call_scalar_integrator_cell_vector(integrator, space, index_sl, out_bb: np.ndarray) -> None:
    """调用 FEALPy 积分器的 ``assembly_cell_vector``，容忍跨版本签名差异。

    Args:
        integrator: 标量源项积分器。
        space: 有限元空间。
        index_sl: cell 切片。
        out_bb: 输出单元向量 ``(NC, ldof)``，原地累加。
    Raises:
        AttributeError: 积分器无 ``assembly_cell_vector``。
        TypeError: 所有签名尝试均失败。
    """
    fn = getattr(integrator, "assembly_cell_vector", None)
    if fn is None:
        raise AttributeError("assembly_cell_vector")
    mesh = space.mesh
    cm = np.asarray(bm.to_numpy(mesh.entity_measure("cell", index=index_sl)), dtype=np.float64)
    last_err: Optional[BaseException] = None
    for thunk in (
        lambda: fn(space, index=index_sl, out=out_bb, cellmeasure=cm),
        lambda: fn(space, index=index_sl, cellmeasure=cm, out=out_bb),
        lambda: fn(space, index=index_sl, out=out_bb),
    ):
        try:
            thunk()
            return
        except TypeError as e:
            last_err = e
    try:
        R = fn(space, index=index_sl)
        R = np.asarray(R, dtype=out_bb.dtype)
        out_bb += R
    except Exception as e:
        raise TypeError(f"assembly_cell_vector incompatible: {last_err!r}; return fallbacks failed: {e!r}") from e


def _assemble_phase_aconst_chunk(args):
    """Cell-slice assembly for diffusion + mass_coef1 (A_const block)."""
    space, diff_int, mass_int, i0, i1 = args
    sl = slice(int(i0), int(i1))
    nc = int(i1) - int(i0)
    ldof = int(_space_ldof(space))
    ftype = np.dtype(np.float64)
    if hasattr(space, "ftype"):
        ftype = np.dtype(space.ftype)
    CM = np.zeros((nc, ldof, ldof), dtype=ftype)
    try:
        _call_scalar_integrator_cell_matrix(diff_int, space, sl, CM, accumulate=False)
        _call_scalar_integrator_cell_matrix(mass_int, space, sl, CM, accumulate=True)
    except (AttributeError, TypeError, ValueError):
        CM.fill(0)
        qd = diff_int.q if getattr(diff_int, "q", None) is not None else space.p + 1
        qm = mass_int.q if getattr(mass_int, "q", None) is not None else space.p + 1
        _explicit_scalar_diffusion_cell(space, _bl_integrator_coef(diff_int), int(qd), sl, CM)
        _explicit_scalar_mass_cell(space, _bl_integrator_coef(mass_int), int(qm), sl, CM)

    cell2dof = np.asarray(space.cell_to_dof()[sl], dtype=np.int64)
    I = np.broadcast_to(cell2dof[:, :, None], (nc, ldof, ldof)).reshape(-1)
    J = np.broadcast_to(cell2dof[:, None, :], (nc, ldof, ldof)).reshape(-1)
    V = np.asarray(CM.reshape(-1), dtype=np.float64)
    return I, J, V


def _assemble_phase_rhs_chunk(args):
    """Cell-slice assembly for ScalarSourceIntegrator."""
    space, src_int, i0, i1 = args
    sl = slice(int(i0), int(i1))
    nc = int(i1) - int(i0)
    ldof = int(_space_ldof(space))
    ftype = np.dtype(np.float64)
    if hasattr(space, "ftype"):
        ftype = np.dtype(space.ftype)
    bb = np.zeros((nc, ldof), dtype=ftype)
    try:
        _call_scalar_integrator_cell_vector(src_int, space, sl, bb)
    except (AttributeError, TypeError, ValueError):
        bb.fill(0)
        qp = src_int.q if getattr(src_int, "q", None) is not None else space.p + 3
        _explicit_scalar_source_cell(space, _source_integrand(src_int), int(qp), sl, bb)

    c2d = np.asarray(space.cell_to_dof()[sl], dtype=np.int64)
    return np.asarray(bb, dtype=np.float64), c2d


def _assemble_scalar_mass_chunk(args):
    """装配含系数的标量质量块 ``∫ coef φ φ`` 在一段 cell 上的 COO 贡献。

    Args:
        args: 元组 ``(phi_chunk, coef_chunk, ws, cm_chunk, cell2dof_chunk)``——基函数、
            积分点系数、权重、cell 测度、cell-to-dof（均已按 cell 切片）。
    Returns:
        ``(I, J, V)`` 三个 1D numpy 数组。
    """
    phi_chunk, coef_chunk, ws, cm_chunk, cell2dof_chunk = args
    phi_chunk = bm.asarray(phi_chunk)
    coef_chunk = bm.asarray(coef_chunk)
    ws = bm.asarray(ws)
    cm_chunk = bm.asarray(cm_chunk)
    cell2dof_chunk = bm.asarray(cell2dof_chunk)
    local = bm.einsum("q,c,cq,cql,cqm->clm", ws, cm_chunk, coef_chunk, phi_chunk, phi_chunk)
    nc, ldof = int(local.shape[0]), int(local.shape[1])
    I = bm.broadcast_to(cell2dof_chunk[:, :, None], (nc, ldof, ldof)).reshape(-1)
    J = bm.broadcast_to(cell2dof_chunk[:, None, :], (nc, ldof, ldof)).reshape(-1)
    V = local.reshape(-1)
    return (
        bm.to_numpy(I).astype(np.int64, copy=False),
        bm.to_numpy(J).astype(np.int64, copy=False),
        bm.to_numpy(V).astype(np.float64, copy=False),
    )


def _as_cell_first(arr, nc: int):
    """Normalize FEALPy arrays to cell-first layout: (NC, ...).

    FEALPy may return one template slice as (1,NQ,...), implicitly broadcast to all NC cells.
    """
    arr = bm.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"Expected tensor with ndim>=2, got shape={arr.shape}, nc={nc}.")
    if int(arr.shape[0]) == nc:
        return arr
    if int(arr.shape[1]) == nc:
        return bm.moveaxis(arr, 1, 0)
    if int(arr.shape[0]) == 1 and nc > 1:
        return bm.broadcast_to(arr, (nc,) + tuple(int(s) for s in arr.shape[1:]))
    raise ValueError(f"Cannot infer cell axis for shape={arr.shape}, nc={nc}.")


class PhaseFieldAssembler:
    """
    Quadrature-history version of phase-field assembler.

    Final structure:
      1) build current quadrature rule
      2) update H on the SAME quadrature points
      3) assemble phase-field system using state.H directly as (NC,NQ)
    """

    def __init__(
        self,
        discr,
        case,
        damage,
        *,
        q: Optional[int] = None,
        debug: bool = False,
        assembly_parallel: Optional[bool] = None,
        assembly_nproc: Optional[int] = None,
    ):
        """Create phase-field assembler with quadrature-history formulation.

        Inputs:
            discr: Discretization containing damage space/state.
            case: Case object for BC and initial-crack data.
            damage: Damage model providing `H` update and constitutive functions.
            q: Optional quadrature order (defaults to `damage_p + 3`).
            debug: Verbose diagnostics switch.
        Output:
            None. Initializes caches and load-step state.
        """
        self.discr = discr
        self.case = case
        self.damage = damage
        self.q = q
        self.debug = bool(debug)
        env_parallel = str(os.getenv("FRACTUREX_ASSEMBLY_PARALLEL", "1")).strip().lower() in ("1", "true", "yes", "on")
        self.assembly_parallel = env_parallel if assembly_parallel is None else bool(assembly_parallel)
        env_nproc = os.getenv("FRACTUREX_ASSEMBLY_NPROC")
        try:
            env_nproc_val = int(env_nproc) if env_nproc is not None else _default_assembly_nproc()
        except (TypeError, ValueError):
            env_nproc_val = _default_assembly_nproc()
        self.assembly_nproc = max(1, env_nproc_val if assembly_nproc is None else int(assembly_nproc))
        self._phase_initial_damage_applied = False
        self._quad_cache = None
        self._quad_cache_key = None
        self._g0_const_coef = None
        self._aconst_cache = None
        self._aconst_cache_key = None
        # Value-only A_hist kernel cache (P1): constant mass kernel Phi[c,q,l,m],
        # weights W[c,q]=cm*ws, CSR pattern/scatter; only coef=g''(d)*H varies.
        self._ahist_kernel = None
        self._ahist_kernel_key = None
        self._ahist_kernel_cache_enabled = str(
            os.getenv("FRACTUREX_AHIST_KERNEL_CACHE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        # Per load-step: phase-field Dirichlet descriptor and quadrature pre-warm.
        self._load_step_value: Optional[float] = None
        self._pf_bcdata_cache: Any = None
        self._pf_bcdof_cache: Any = None

    def begin_load_step(self, load: float) -> None:
        """
        Cache `case.phasefield_dirichlet_data(load)` and pre-warm quadrature for the
        current damage order — unchanged during staggered iterations at fixed `load`.

        Safe to skip: `assemble` falls back to the previous on-the-fly path.
        """
        self._load_step_value = float(load)
        case = self.case
        if hasattr(case, "phasefield_dirichlet_data"):
            self._pf_bcdata_cache = case.phasefield_dirichlet_data(load)
        else:
            self._pf_bcdata_cache = None
        self._pf_bcdof_cache = self._resolve_phase_dirichlet_bc(self._pf_bcdata_cache)
        discr = self.discr
        if discr is None or discr.space_d is None or discr.mesh is None:
            return
        q = int(discr.damage_p + 3 if self.q is None else self.q)
        self._get_quadrature_data(q)

    def assemble(self, load: float) -> PhaseFieldSystem:
        """Assemble one phase-field linearized increment system.

        Input:
            load: Current scalar load value (for BC and one-time crack seeding logic).
        Output:
            `PhaseFieldSystem(A, F, decode, meta)`, where `decode(dd)` returns
            clipped updated damage field `d_new`.
        """
        discr = self.discr
        case = self.case
        damage = self.damage
        state = discr.state

        if state is None:
            raise RuntimeError("Discretization state is None. Call discr.build(...) first.")

        # One-time initialization: mark pre-crack by setting d directly (e.g. d=1 on notch line)
        self._apply_phase_initial_damage_once(load)

        space = discr.space_d
        if space is None:
            raise RuntimeError("discr.space_d is None.")

        d_old = state.d
        gdof = space.number_of_global_dofs()

        Gc = float(damage.fracture_toughness())
        l0 = float(damage.length_scale())

        # IMPORTANT: phase-field quadrature is tied to damage space, not Hu-Zhang p
        q = discr.damage_p + 3 if self.q is None else self.q

        bcs, index = self._get_quadrature_data(q)

        # ----------------------------------------------------------
        # update quadrature-history H on exactly the same quadrature
        # ----------------------------------------------------------
        _prof_hist = str(os.getenv("FRACTUREX_PROFILE_HISTORY_UPDATE", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if _prof_hist:
            t_hist0 = time.perf_counter()
        damage.update_history_on_quadrature(
            discr=discr,
            state=state,
            case=case,
            bcs=bcs,
            index=index,
        )
        if _prof_hist:
            dt_hist = time.perf_counter() - t_hist0
            nc = int(discr.mesh.number_of_cells())
            nq = int(state.H.shape[1]) if state.H is not None and state.H.ndim >= 2 else -1
            print(
                "[PhaseFieldAssembler] update_history_on_quadrature "
                f"wall_s={dt_hist:.4e} NC={nc} NQ={nq} split={getattr(damage, 'split', '')}"
            )

        if state.H is None:
            raise RuntimeError("state.H is None after update_history_on_quadrature.")

        # ----------------------------------------------------------
        # coefficient functions
        # ----------------------------------------------------------
        @barycentric
        def diff_coef(bc, index=None):
            _, c_d = damage.crack_density_hess(d_old(bc, index=index))
            return Gc * l0 * 2.0 / c_d

        @barycentric
        def mass_coef1(bc, index=None):
            gg_hd, c_d = damage.crack_density_hess(d_old(bc, index=index))
            return gg_hd * Gc / (l0 * c_d)

        @barycentric
        def mass_coef2(bc, index=None):
            Hq = state.H if index is None else state.H[index]   # (NC,NQ)
            Hq = bm.maximum(Hq, 0.0)

            gg_gd = damage.degradation_hess(d_old(bc, index=index))
            return gg_gd * Hq

        @barycentric
        def source_coef(bc, index=None):
            Hq = state.H if index is None else state.H[index]   # (NC,NQ)
            Hq = bm.maximum(Hq, 0.0)

            if hasattr(damage._gfun, "grad_degradation_function_constant_coef"):
                if self._g0_const_coef is None:
                    self._g0_const_coef = float(damage._gfun.grad_degradation_function_constant_coef())
                return -1.0 * self._g0_const_coef * Hq

            z = bm.zeros_like(Hq)
            g0p = damage.degradation_grad(z)
            return -g0p * Hq

        # ----------------------------------------------------------
        # assemble A
        # ----------------------------------------------------------
        # Terms independent of H can be cached and reused across solves.
        # For built-in AT1/AT2 crack-density models, h''(d), c_d are constants,
        # so A_const does not change over staggered iterations.
        aconst_key = (id(discr.mesh), int(gdof), int(q), float(Gc), float(l0), str(getattr(damage, "density_type", "unknown")))
        cacheable_aconst = str(getattr(damage, "density_type", "")).lower() in ("at1", "at2")
        if cacheable_aconst and self._aconst_cache is not None and self._aconst_cache_key == aconst_key:
            A_const = self._aconst_cache
        else:
            diff_int = ScalarDiffusionIntegrator(coef=diff_coef, q=q)
            mass_int = ScalarMassIntegrator(coef=mass_coef1, q=q)
            if self.assembly_parallel:
                try:
                    A_const = self._assemble_A_const_parallel(space, diff_int, mass_int)
                except Exception as exc:
                    print(f"[PhaseFieldAssembler] parallel A_const assembly failed, fallback to serial: {exc}")
                    bform_const = BilinearForm(space)
                    bform_const.add_integrator(
                        ScalarDiffusionIntegrator(coef=diff_coef, q=q),
                        ScalarMassIntegrator(coef=mass_coef1, q=q),
                    )
                    A_const = bform_const.assembly()
            else:
                bform_const = BilinearForm(space)
                bform_const.add_integrator(
                    ScalarDiffusionIntegrator(coef=diff_coef, q=q),
                    ScalarMassIntegrator(coef=mass_coef1, q=q),
                )
                A_const = bform_const.assembly()
            if cacheable_aconst:
                self._aconst_cache_key = aconst_key
                self._aconst_cache = A_const

        A_hist = self._assemble_A_hist(space, q, mass_coef2)
        A = A_const + A_hist

        # ----------------------------------------------------------
        # assemble rhs
        # ----------------------------------------------------------
        src_int = ScalarSourceIntegrator(source=source_coef, q=q)
        if self.assembly_parallel:
            try:
                rhs = self._assemble_rhs_parallel(space, src_int)
            except Exception as exc:
                print(f"[PhaseFieldAssembler] parallel RHS assembly failed, fallback to serial: {exc}")
                lform = LinearForm(space)
                lform.add_integrator(ScalarSourceIntegrator(source=source_coef, q=q))
                rhs = lform.assembly()
        else:
            lform = LinearForm(space)
            lform.add_integrator(src_int)
            rhs = lform.assembly()

        # residual / increment form: A dd = rhs - A d_old
        F = rhs - A @ d_old[:]

        # ----------------------------------------------------------
        # phase-field Dirichlet BC
        # ----------------------------------------------------------
        A, F = self._apply_phase_dirichlet_bc(A, F, load)

        if self.debug:
            hmin = float(bm.min(state.H)) if state.H is not None else 0.0
            hmax = float(bm.max(state.H)) if state.H is not None else 0.0
            dmin = float(bm.min(bm.asarray(d_old[:]))) if d_old is not None else 0.0
            dmax = float(bm.max(bm.asarray(d_old[:]))) if d_old is not None else 0.0
            print(
                "[PhaseFieldAssembler] "
                f"gdof={gdof}, q={q}, Gc={Gc:.3e}, l0={l0:.3e}, "
                f"d min/max={dmin:.3e}/{dmax:.3e}, "
                f"H min/max={hmin:.3e}/{hmax:.3e}"
            )

        def decode(dd):
            d_new = space.function()
            d_new[:] = d_old[:] + bm.asarray(dd).reshape(-1)
            d_new[:] = bm.clip(d_new[:], 0.0, 1.0)
            return d_new

        meta = dict(
            type="phasefield",
            model=getattr(damage, "density_type", "unknown"),
            degradation=getattr(damage, "degradation_type", "unknown"),
            load=float(load),
            gdof=int(gdof),
            q=int(q),
            Gc=float(Gc),
            l0=float(l0),
        )

        return PhaseFieldSystem(A=A, F=F, decode=decode, meta=meta)

    def _assemble_A_const_parallel(self, space, diff_int: ScalarDiffusionIntegrator, mass_int: ScalarMassIntegrator):
        """线程并行装配与历史场无关的 ``A_const = 扩散 + mass_coef1`` 块。

        Args:
            space: 损伤空间。
            diff_int, mass_int: 扩散、质量积分器（系数为 d 的函数但与 H 无关）。
        Returns:
            ``CSRTensor`` 形式的 ``(gdof, gdof)`` 矩阵。
        """
        mesh = self.discr.mesh
        nc = int(mesh.number_of_cells())
        gdof = int(space.number_of_global_dofs())
        nproc = min(int(self.assembly_nproc), nc)
        if nproc <= 1 or nc < 2:
            I, J, V = _assemble_phase_aconst_chunk((space, diff_int, mass_int, 0, nc))
        else:
            edges = np.linspace(0, nc, nproc + 1, dtype=int)
            tasks = []
            for k in range(nproc):
                i0, i1 = int(edges[k]), int(edges[k + 1])
                if i1 <= i0:
                    continue
                tasks.append((space, diff_int, mass_int, i0, i1))
            workers = min(int(self.assembly_nproc), len(tasks))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                parts = list(pool.map(_assemble_phase_aconst_chunk, tasks))
            I = np.concatenate([p[0] for p in parts], axis=0)
            J = np.concatenate([p[1] for p in parts], axis=0)
            V = np.concatenate([p[2] for p in parts], axis=0)
        A_sp = coo_matrix((V, (I, J)), shape=(gdof, gdof)).tocsr()
        return CSRTensor.from_scipy(A_sp)

    def _assemble_rhs_parallel(self, space, src_int: ScalarSourceIntegrator):
        """线程并行装配相场源项右端向量 ``∫ source_coef φ``。

        Args:
            space: 损伤空间。
            src_int: 源项积分器（系数含历史场 H）。
        Returns:
            ``(gdof,)`` 的 float64 numpy 向量。
        """
        mesh = self.discr.mesh
        nc = int(mesh.number_of_cells())
        gdof = int(space.number_of_global_dofs())
        rhs = np.zeros((gdof,), dtype=np.float64)
        nproc = min(int(self.assembly_nproc), nc)
        if nproc <= 1 or nc < 2:
            bb, c2d = _assemble_phase_rhs_chunk((space, src_int, 0, nc))
            np.add.at(rhs, c2d, bb)
        else:
            edges = np.linspace(0, nc, nproc + 1, dtype=int)
            tasks = []
            for k in range(nproc):
                i0, i1 = int(edges[k]), int(edges[k + 1])
                if i1 <= i0:
                    continue
                tasks.append((space, src_int, i0, i1))
            workers = min(int(self.assembly_nproc), len(tasks))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                parts = list(pool.map(_assemble_phase_rhs_chunk, tasks))
            for bb, c2d in parts:
                np.add.at(rhs, c2d, bb)
        return rhs

    def _assemble_A_hist(self, space, q: int, coef_callable):
        """装配含历史场的质量块 ``A_hist = ∫ (g''(d)·H) φ φ``，按可用性择优分派。

        依次尝试：缓存几何内核 → 线程并行 → 串行 ``BilinearForm``。

        Args:
            space: 损伤空间。
            q: 积分阶。
            coef_callable: 重心坐标系数闭包（``mass_coef2``，返回 ``g''(d)·H``）。
        Returns:
            ``CSRTensor``/稀疏矩阵形式的 ``(gdof, gdof)`` 矩阵。
        """
        if self._ahist_kernel_cache_enabled:
            try:
                return self._assemble_A_hist_cached(space, q, coef_callable)
            except Exception as exc:
                print(f"[PhaseFieldAssembler] cached A_hist kernel failed, fallback: {exc}")
        if self.assembly_parallel:
            try:
                return self._assemble_A_hist_parallel(space, q, coef_callable)
            except Exception as exc:
                print(f"[PhaseFieldAssembler] parallel A_hist assembly failed, fallback to serial: {exc}")
        bform_hist = BilinearForm(space)
        bform_hist.add_integrator(ScalarMassIntegrator(coef=coef_callable, q=q))
        return bform_hist.assembly()

    def _build_ahist_kernel(self, space, q: int):
        """Cache d-independent material for the A_hist mass block (P1).

        Mirrors FEALPy's ``bilinear_integral(phi, phi, ws, cm, coef)`` with coef
        factored out: ``Phi[c,q,l,m] = phi_l phi_m``, ``W[c,q] = cm ws``, so that
        ``A_e = Σ_q W coef Phi``. Also caches the CSR pattern + scatter map.
        """
        mesh = self.discr.mesh
        gdof = int(space.number_of_global_dofs())
        key = (id(mesh), gdof, int(q))
        if self._ahist_kernel is not None and self._ahist_kernel_key == key:
            return self._ahist_kernel

        qf = mesh.quadrature_formula(int(q), "cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        ws = bm.asarray(ws)
        cm = bm.asarray(mesh.entity_measure("cell"))
        nc = int(mesh.number_of_cells())
        phi = _as_cell_first(bm.asarray(space.basis(bcs)), nc)   # (NC, NQ, ldof)
        Phi = bm.einsum("cql, cqm -> cqlm", phi, phi)
        W = bm.einsum("c, q -> cq", cm, ws)

        cell2dof = np.asarray(bm.to_numpy(space.cell_to_dof()), dtype=np.int64)
        indptr, indices, inv, nnz = _build_square_csr_scatter(cell2dof, gdof)

        self._ahist_kernel_key = key
        self._ahist_kernel = {
            "bcs": bcs,
            "W": W,
            "Phi": Phi,
            "indptr": indptr,
            "indices": indices,
            "inv": inv,
            "nnz": nnz,
            "gdof": gdof,
        }
        return self._ahist_kernel

    def _assemble_A_hist_cached(self, space, q: int, coef_callable):
        """用缓存的 d-无关质量内核装配 ``A_hist``：只重算系数并 bincount 散射。

        Args/Returns 同 :meth:`_assemble_A_hist`。
        """
        K = self._build_ahist_kernel(space, q)
        mesh = self.discr.mesh
        val = process_coef_func(coef_callable, bcs=K["bcs"], mesh=mesh, etype="cell", index=_S)
        val = bm.asarray(val)                                    # (NC, NQ)
        Ke = bm.einsum("cq, cq, cqlm -> clm", K["W"], val, K["Phi"])
        Ke = np.asarray(bm.to_numpy(Ke), dtype=np.float64)
        data = np.bincount(K["inv"], weights=Ke.reshape(-1), minlength=K["nnz"])
        from scipy.sparse import csr_matrix
        A_sp = csr_matrix((data, K["indices"], K["indptr"]), shape=(K["gdof"], K["gdof"]))
        return CSRTensor.from_scipy(A_sp)

    def _assemble_A_hist_parallel(self, space, q: int, coef_callable):
        """线程并行装配含历史场的质量块 ``A_hist``（按 cell 切片分块）。

        Args/Returns 同 :meth:`_assemble_A_hist`。
        """
        mesh = self.discr.mesh
        qf = mesh.quadrature_formula(q, "cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        ws = bm.asarray(ws)
        cm = bm.asarray(mesh.entity_measure("cell"))
        cell2dof = bm.asarray(space.cell_to_dof())
        nc = int(cell2dof.shape[0])
        phi = _as_cell_first(bm.asarray(space.basis(bcs)), nc)
        coef = _as_cell_first(bm.asarray(coef_callable(bcs, index=bm.arange(nc))), nc)

        nproc = min(int(self.assembly_nproc), nc)
        if nproc <= 1 or nc < 2:
            I, J, V = _assemble_scalar_mass_chunk((phi, coef, ws, cm, cell2dof))
        else:
            edges = np.linspace(0, nc, nproc + 1, dtype=int)
            tasks = []
            for k in range(nproc):
                i0, i1 = int(edges[k]), int(edges[k + 1])
                if i1 <= i0:
                    continue
                tasks.append((phi[i0:i1], coef[i0:i1], ws, cm[i0:i1], cell2dof[i0:i1]))
            workers = min(int(self.assembly_nproc), len(tasks))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                parts = list(pool.map(_assemble_scalar_mass_chunk, tasks))
            I = np.concatenate([p[0] for p in parts], axis=0)
            J = np.concatenate([p[1] for p in parts], axis=0)
            V = np.concatenate([p[2] for p in parts], axis=0)

        gdof = int(space.number_of_global_dofs())
        A_sp = coo_matrix((V, (I, J)), shape=(gdof, gdof)).tocsr()
        return CSRTensor.from_scipy(A_sp)

    def _get_quadrature_data(self, q: int):
        """Cache cell quadrature points/indices for fixed mesh and order."""
        mesh = self.discr.mesh
        NC = mesh.number_of_cells()
        key = (id(mesh), int(q), int(NC))
        if self._quad_cache is not None and self._quad_cache_key == key:
            return self._quad_cache["bcs"], self._quad_cache["index"]

        qf = mesh.quadrature_formula(q, "cell")
        bcs, _ = qf.get_quadrature_points_and_weights()
        index = bm.arange(NC)
        self._quad_cache_key = key
        self._quad_cache = {"bcs": bcs, "index": index}
        return bcs, index

    def _apply_phase_initial_damage_once(self, load: float):
        """Apply one-time initial damage constraints (pre-crack).

        Input:
            load: Current load used to query case initial-damage descriptor.
        Output:
            None. Updates `state.d` in place once and sets internal applied flag.
        """
        if self._phase_initial_damage_applied:
            return

        case = self.case
        discr = self.discr
        state = discr.state
        space = discr.space_d

        if state is None or space is None:
            return

        if not hasattr(case, "phasefield_initial_damage_data"):
            self._phase_initial_damage_applied = True
            return

        bcdata = case.phasefield_initial_damage_data(load)
        if bcdata is None:
            self._phase_initial_damage_applied = True
            return

        if isinstance(bcdata, dict):
            bcdata = [bcdata]

        ip = space.interpolation_points()
        darr = bm.asarray(state.d[:]).copy()

        if self.debug:
            print(f"[PhaseFieldAssembler._apply_phase_initial_damage_once] bcdata={bcdata}")

        for item in bcdata:
            if "bcdof" not in item or "value" not in item:
                raise ValueError(
                    "phasefield_initial_damage_data(load) must return dict(s) with keys {'bcdof','value'}."
                )

            thr = item["bcdof"]
            val = item["value"]

            if callable(thr):
                mask = bm.asarray(thr(ip)).astype(bm.bool)
                idx = bm.where(mask)[0]
            else:
                arr = bm.asarray(thr)
                if getattr(arr, "dtype", None) == bm.bool:
                    idx = bm.where(arr)[0]
                else:
                    idx = arr

            if self.debug:
                print(f"  Found {len(idx)} DOFs on pre-crack, setting to value={val}")
                if len(idx) > 0:
                    print(f"    DOF indices (first 10): {idx[:10]}")

            if callable(val):
                v = val(ip[idx])
            else:
                v = val

            darr = bm.set_at(darr, idx, v)

            if self.debug and len(idx) > 0:
                print(f"    After set_at: darr[idx] min={bm.min(darr[idx]):.6e}, max={bm.max(darr[idx]):.6e}")

        state.d[:] = bm.clip(darr, 0.0, 1.0)
        self._phase_initial_damage_applied = True
        
        if self.debug:
            print(f"[PhaseFieldAssembler._apply_phase_initial_damage_once] COMPLETE: state.d min={bm.min(state.d[:]):.6e}, max={bm.max(state.d[:]):.6e}")

    def _apply_phase_dirichlet_bc(self, A, F, load: float):
        """Apply phase-field Dirichlet boundary conditions to linear system.

        Inputs:
            A: Phase-field system matrix.
            F: Phase-field rhs vector.
            load: Current load used to query case BC descriptors.
        Output:
            Tuple `(A_bc, F_bc)` after Dirichlet elimination.
        """
        case = self.case
        space = self.discr.space_d

        if not hasattr(case, "phasefield_dirichlet_data"):
            return A, F

        if self._load_step_value is not None and float(load) == float(self._load_step_value) and self._pf_bcdata_cache is not None:
            bcdata = self._pf_bcdata_cache
            bcdata_resolved = self._pf_bcdof_cache
        else:
            bcdata = case.phasefield_dirichlet_data(load)
            bcdata_resolved = self._resolve_phase_dirichlet_bc(bcdata)
        if bcdata is None:
            return A, F

        for item in bcdata_resolved:
            if "bcdof" not in item or "value" not in item:
                raise ValueError(
                    "phasefield_dirichlet_data(load) must return dict(s) with keys {'bcdof','value'}."
                )
            A, F = self._apply_phase_dirichlet_by_dof(A, F, bcdof=item["bcdof"], value=item["value"])

        return A, F

    def _apply_phase_dirichlet_by_dof(self, A, F, bcdof, value):
        """Apply Dirichlet elimination for explicit dof-index boundary set.

        This path avoids sparse triple-products like `T @ A @ T + Tbd`.
        We enforce BC by directly modifying matrix rows/columns in sparse format.
        """
        if hasattr(A, "to_scipy"):
            A = A.to_scipy().tocsr(copy=True)
        else:
            A = A.tocsr(copy=True)
        F = np.asarray(F, dtype=float).reshape(-1)

        idx = np.asarray(bcdof, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return A, F
        idx = np.unique(idx)

        gdof = int(A.shape[0])
        if np.any(idx < 0) or np.any(idx >= gdof):
            raise ValueError(f"Dirichlet dof index out of range [0, {gdof}): {idx}.")
        ip = bm.asarray(self.discr.space_d.interpolation_points())
        pts = ip[idx]
        if callable(value):
            vals = np.asarray(bm.asarray(value(pts)), dtype=float).reshape(-1)
        else:
            vals = np.asarray(value, dtype=float).reshape(-1)

        if vals.size == 1:
            vals = np.full(idx.shape[0], float(vals[0]), dtype=float)
        elif vals.size == gdof:
            vals = vals[idx]
        elif vals.size != idx.shape[0]:
            raise ValueError(f"Dirichlet value shape mismatch: got {vals.size}, expect 1, gdof({gdof}) or len(bcdof)({idx.shape[0]}).")

        # Residual correction from known Dirichlet values:
        # F <- F - A[:, idx] * vals
        # then overwrite boundary rhs with prescribed values.
        F = F - A[:, idx] @ vals
        F[idx] = vals

        # Zero constrained columns first (for all rows), using CSC for fast column access.
        A_csc = A.tocsc(copy=False)
        for j in idx:
            p0, p1 = A_csc.indptr[j], A_csc.indptr[j + 1]
            if p1 > p0:
                A_csc.data[p0:p1] = 0.0
        A = A_csc.tocsr(copy=False)

        # Zero constrained rows and set unit diagonal to enforce d = value on boundary dofs.
        for i in idx:
            p0, p1 = A.indptr[i], A.indptr[i + 1]
            if p1 > p0:
                A.data[p0:p1] = 0.0
            A[i, i] = 1.0
        A.eliminate_zeros()
        return A, F

    def _resolve_phase_dirichlet_bc(self, bcdata):
        """Resolve phase Dirichlet threshold descriptors to concrete DOF indices."""
        if bcdata is None:
            return None
        if isinstance(bcdata, dict):
            bc_items = [bcdata]
        else:
            bc_items = bcdata

        space = self.discr.space_d
        ip = space.interpolation_points()
        resolved = []
        for item in bc_items:
            if "bcdof" not in item or "value" not in item:
                raise ValueError(
                    "phasefield_dirichlet_data(load) must return dict(s) with keys {'bcdof','value'}."
                )
            thr = item["bcdof"]
            if callable(thr):
                mask = bm.asarray(thr(ip)).astype(bm.bool)
                bcdof = bm.where(mask)[0]
            else:
                arr = bm.asarray(thr)
                if getattr(arr, "dtype", None) == bm.bool:
                    bcdof = bm.where(arr)[0]
                else:
                    bcdof = arr
            resolved.append({"bcdof": bcdof, "value": item["value"]})
        return resolved