# fracturex/utilfuc/sparse_direct_backends.py
"""Shared sparse direct linear solvers (PARDISO, MUMPS) for SciPy / FeALPy matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def _matrix_to_scipy_csr(A: Any):
    """把 FEALPy/SciPy 稀疏矩阵统一转为 ``scipy.sparse.csr_matrix``（无对应接口则原样返回）。"""
    if hasattr(A, "to_scipy"):
        return A.to_scipy().tocsr()
    if hasattr(A, "tocsr"):
        return A.tocsr()
    return A


def solve_direct_pardiso(A: Any, rhs: Any) -> np.ndarray:
    """Intel MKL PARDISO via ``pypardiso`` (CSR SciPy matrix).

    Parameters
    ----------
    A : sparse matrix or FeALPy operator with ``to_scipy``.
    rhs : 1d array-like right-hand side.

    Returns
    -------
    ndarray, shape (n,)
    """
    try:
        import pypardiso  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PARDISO requires the `pypardiso` package (Intel MKL PARDISO). "
            "Typical install: conda install -c conda-forge pypardiso"
        ) from exc
    Acsr = _matrix_to_scipy_csr(A)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    return np.asarray(pypardiso.spsolve(Acsr, b), dtype=np.float64).reshape(-1)


def solve_direct_mumps(A: Any, rhs: Any) -> np.ndarray:
    """Sequential MUMPS via ``python-mumps`` (``import mumps``).

    Parameters
    ----------
    A : sparse matrix or FeALPy operator with ``to_scipy``.
    rhs : 1d array-like right-hand side.

    Returns
    -------
    ndarray, shape (n,)
    """
    try:
        import mumps  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "MUMPS requires the `python-mumps` distribution (import name `mumps`). "
            "See https://pypi.org/project/python-mumps/ (conda-forge recommended)."
        ) from exc
    Acsc = _matrix_to_scipy_csr(A).tocsc()
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    ctx = mumps.Context()
    ctx.analyze(Acsc, ordering="pord")
    ctx.factor(Acsc)
    return np.asarray(ctx.solve(b), dtype=np.float64).reshape(-1)
