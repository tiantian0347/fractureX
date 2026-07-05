"""
基于 sympy 的解析双调和 PDE，给收敛测试用。
solution / gradient / hessian / source 全是 lambdify 出来的，
输入 `p` 是 fealpy 的坐标点数组（形状 (..., 2)）。
"""
import numpy as np
import sympy as sp

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian


def _to_np(p):
    if isinstance(p, np.ndarray):
        return p
    try:
        return bm.to_numpy(p)
    except Exception:
        return np.asarray(p)


def _as_backend(arr, ref):
    """把 numpy 结果转成与 ref 相同后端的 tensor。"""
    if isinstance(ref, np.ndarray):
        return arr
    try:
        return bm.tensor(arr, dtype=bm.float64)
    except Exception:
        return arr


class DoubleLaplacePDE:
    def __init__(self, u):
        x, y = sp.symbols("x y")

        ux = sp.diff(u, x)
        uy = sp.diff(u, y)
        uxx = sp.diff(ux, x)
        uxy = sp.diff(ux, y)
        uyy = sp.diff(uy, y)

        Lu = uxx + uyy
        L2u = sp.diff(Lu, x, 2) + sp.diff(Lu, y, 2)

        self._u = sp.lambdify(("x", "y"), u, "numpy")
        self._ux = sp.lambdify(("x", "y"), ux, "numpy")
        self._uy = sp.lambdify(("x", "y"), uy, "numpy")
        self._uxx = sp.lambdify(("x", "y"), uxx, "numpy")
        self._uxy = sp.lambdify(("x", "y"), uxy, "numpy")
        self._uyy = sp.lambdify(("x", "y"), uyy, "numpy")
        self._L2u = sp.lambdify(("x", "y"), L2u, "numpy")

    def solution(self, p):
        p_np = _to_np(p)
        x, y = p_np[..., 0], p_np[..., 1]
        val = np.asarray(self._u(x, y), dtype=np.float64)
        return _as_backend(val, p)

    def gradient(self, p):
        p_np = _to_np(p)
        x, y = p_np[..., 0], p_np[..., 1]
        val = np.zeros(p_np.shape, dtype=np.float64)
        val[..., 0] = self._ux(x, y)
        val[..., 1] = self._uy(x, y)
        return _as_backend(val, p)

    def hessian(self, p):
        p_np = _to_np(p)
        x, y = p_np[..., 0], p_np[..., 1]
        val = np.zeros(p_np.shape[:-1] + (2, 2), dtype=np.float64)
        val[..., 0, 0] = self._uxx(x, y)
        val[..., 0, 1] = self._uxy(x, y)
        val[..., 1, 0] = self._uxy(x, y)
        val[..., 1, 1] = self._uyy(x, y)
        return _as_backend(val, p)

    def source(self, p):
        p_np = _to_np(p)
        x, y = p_np[..., 0], p_np[..., 1]
        val = np.asarray(self._L2u(x, y), dtype=np.float64)
        return _as_backend(val, p)
    source = cartesian(source)

    def dirichlet(self, p):
        return self.solution(p)
    dirichlet = cartesian(dirichlet)


def default_pde():
    """老论文用的解 u = (sin(pi*y)*sin(pi^2 x))^2, 单位方形域。"""
    x, y = sp.symbols("x y")
    u = (sp.sin(sp.pi * y) * sp.sin(sp.pi**2 * x)) ** 2
    return DoubleLaplacePDE(u)


def sin_sq_pde():
    """简单一点的 u = sin(pi*x)^2 * sin(pi*y)^2, 收敛快。"""
    x, y = sp.symbols("x y")
    u = (sp.sin(sp.pi * x) * sp.sin(sp.pi * y)) ** 2
    return DoubleLaplacePDE(u)
