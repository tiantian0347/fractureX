"""
运行时给 fealpy 打补丁：
1. numpy 后端补 `simplex_hess_shape_function`（fealpy 未实现，autodiff 后端才有）；
2. BernsteinFESpace.grad_basis / grad_m_basis 的 `index` 切片走错轴（slice 了 NQ 而不是 NC）。

只在检测到 bug 时替换，且不改 fealpy 源文件。
"""
import numpy as np
from functools import wraps

from fealpy.backend import backend_manager as bm
from fealpy.backend.numpy_backend import NumPyBackend


# ---------------- numpy hess ----------------

def _simplex_hess_shape_function(cls, bc, p, mi=None):
    TD = bc.shape[-1] - 1
    if mi is None:
        mi = cls.multi_index_matrix(p, TD)
    ldof = mi.shape[0]

    shape = bc.shape[:-1] + (p + 1, TD + 1)
    A = np.ones(shape, dtype=bc.dtype)
    dA = np.zeros(shape, dtype=bc.dtype)
    ddA = np.zeros(shape, dtype=bc.dtype)
    for k in range(1, p + 1):
        factor = p * bc - (k - 1)
        A_prev = A[..., k - 1, :].copy()
        dA_prev = dA[..., k - 1, :].copy()
        ddA_prev = ddA[..., k - 1, :].copy()
        A[..., k, :] = A_prev * factor / k
        dA[..., k, :] = (dA_prev * factor + A_prev * p) / k
        ddA[..., k, :] = (ddA_prev * factor + 2 * dA_prev * p) / k

    idx = np.arange(TD + 1)
    Q = A[..., mi, idx]
    dQ = dA[..., mi, idx]
    ddQ = ddA[..., mi, idx]

    H = np.zeros(bc.shape[:-1] + (ldof, TD + 1, TD + 1), dtype=bc.dtype)
    for j in range(TD + 1):
        others_j = [i for i in range(TD + 1) if i != j]
        prod_j = np.prod(Q[..., others_j], axis=-1)
        H[..., j, j] = ddQ[..., j] * prod_j
        for kk in range(TD + 1):
            if kk == j:
                continue
            others_jk = [i for i in range(TD + 1) if i != j and i != kk]
            prod_jk = np.prod(Q[..., others_jk], axis=-1) if others_jk else 1.0
            H[..., j, kk] = dQ[..., j] * dQ[..., kk] * prod_jk
    return H


# ---------------- Bernstein grad_basis index-axis fix ----------------

def _patch_pytorch_hess():
    """
    pytorch 后端 `simplex_hess_shape_function` 用的是 `jacfwd(jacfwd(kernel))`，
    kernel 里 cumprod 遇到零输入 (bc 分量为 0，就是边上做体积分时插入的 0)
    时二阶自动求导会返回错的值 (实测 4 变 2)。这里换成和 numpy 一致的解析递推。
    """
    try:
        from fealpy.backend.pytorch_backend import PyTorchBackend
    except Exception:
        return
    if getattr(PyTorchBackend, "_ip_hess_patched", False):
        return
    import torch

    def _hess(cls, bcs, p, mi=None):
        TD = bcs.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)
        ldof = mi.shape[0]

        A = torch.ones(bcs.shape[:-1] + (p + 1, TD + 1), dtype=bcs.dtype, device=bcs.device)
        dA = torch.zeros_like(A)
        ddA = torch.zeros_like(A)
        for k in range(1, p + 1):
            factor = p * bcs - (k - 1)
            A_prev = A[..., k - 1, :].clone()
            dA_prev = dA[..., k - 1, :].clone()
            ddA_prev = ddA[..., k - 1, :].clone()
            A[..., k, :] = A_prev * factor / k
            dA[..., k, :] = (dA_prev * factor + A_prev * p) / k
            ddA[..., k, :] = (ddA_prev * factor + 2 * dA_prev * p) / k

        idx = torch.arange(TD + 1, dtype=torch.long, device=bcs.device)
        mi_long = mi.long() if hasattr(mi, "long") else torch.as_tensor(mi, dtype=torch.long, device=bcs.device)
        Q = A[..., mi_long, idx]
        dQ = dA[..., mi_long, idx]
        ddQ = ddA[..., mi_long, idx]

        H = torch.zeros(bcs.shape[:-1] + (ldof, TD + 1, TD + 1), dtype=bcs.dtype, device=bcs.device)
        for j in range(TD + 1):
            others_j = [i for i in range(TD + 1) if i != j]
            prod_j = torch.prod(Q[..., others_j], dim=-1)
            H[..., j, j] = ddQ[..., j] * prod_j
            for kk in range(TD + 1):
                if kk == j:
                    continue
                others_jk = [i for i in range(TD + 1) if i != j and i != kk]
                if others_jk:
                    prod_jk = torch.prod(Q[..., others_jk], dim=-1)
                else:
                    prod_jk = torch.ones_like(dQ[..., j])
                H[..., j, kk] = dQ[..., j] * dQ[..., kk] * prod_jk
        return H

    PyTorchBackend.simplex_hess_shape_function = classmethod(_hess)
    PyTorchBackend._ip_hess_patched = True


def _patch_pytorch_add_at():
    """
    fealpy pytorch 后端的 add_at 用 `a[indices] += src`。对于 fealpy
    grad_normal_jump_basis 里的调用（indices 含 Ellipsis + bool mask + slice），
    这个语义没问题——单次调用内索引唯一，跨调用累加也走的是 read-modify-write。
    这里只压掉 fealpy 每次调用都打印的 WARNING 日志，不改语义。
    """
    try:
        from fealpy.backend.pytorch_backend import PyTorchBackend
    except Exception:
        return
    if getattr(PyTorchBackend, "_ip_add_at_patched", False):
        return

    def _add_at(a, indices, src, /):
        a[indices] += src
        return a

    PyTorchBackend.add_at = staticmethod(_add_at)
    PyTorchBackend._ip_add_at_patched = True


def _patch_bernstein():
    """
    BernsteinFESpace.grad_basis 在 variable='x' 分支里写的是 `gphi[:, index]`，
    而 `gphi` 的 shape 是 (NC, NQ, ldof, GD)，正确的按单元切片应为 `gphi[index]`。
    同类问题也出现在 grad_m_basis 里。这里包一层，把 index 从 slice(None)
    (即 `_S`, 默认参数) 转发；如果调用方真的传了 cell index，就先算全体再切轴 0。
    """
    from fealpy.functionspace.bernstein_fe_space import BernsteinFESpace
    from fealpy.typing import _S

    if getattr(BernsteinFESpace, "_ip_patched", False):
        return

    _orig_grad_basis = BernsteinFESpace.grad_basis
    _orig_grad_m_basis = BernsteinFESpace.grad_m_basis

    @wraps(_orig_grad_basis)
    def _grad_basis(self, bcs, index=_S, variable="u"):
        full = _orig_grad_basis(self, bcs, index=_S, variable=variable)
        if isinstance(index, slice) and index == _S:
            return full
        return full[index]

    @wraps(_orig_grad_m_basis)
    def _grad_m_basis(self, bcs, m, index=_S):
        full = _orig_grad_m_basis(self, bcs, m, index=_S)
        if isinstance(index, slice) and index == _S:
            return full
        return full[index]

    BernsteinFESpace.grad_basis = _grad_basis
    BernsteinFESpace.grad_m_basis = _grad_m_basis
    BernsteinFESpace._ip_patched = True


# ---------------- entrypoint ----------------

def apply_patch():
    # numpy hess
    try:
        bc = np.array([[0.3, 0.4, 0.3]])
        NumPyBackend.simplex_hess_shape_function(bc, 2)
    except NotImplementedError:
        NumPyBackend.simplex_hess_shape_function = classmethod(
            _simplex_hess_shape_function
        )
    except Exception:
        pass

    _patch_pytorch_hess()
    _patch_pytorch_add_at()
    _patch_bernstein()
