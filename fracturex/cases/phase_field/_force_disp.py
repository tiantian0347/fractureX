"""Shared force–displacement I/O for MainSolve Ambati FEM drivers.

Used by model4 / model5 / model6 standard-FEM scripts. Does not build meshes
or attach BCs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def merge_force_disp(prev_path: Path, disp: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Stack a continuation segment onto a previous ``force_disp`` table.

    Drops the first new row if it duplicates the last previous displacement.
    """
    old = np.loadtxt(prev_path, skiprows=1)
    if old.ndim == 1:
        old = old.reshape(1, -1)
    new = np.c_[disp, force]
    if len(old) and abs(abs(old[-1, 0]) - abs(new[0, 0])) < 1e-9:
        new = new[1:]
    return np.vstack([old, new])


def write_force_disp(path: Path, disp: np.ndarray, force: np.ndarray, merge_with: str | None = None) -> np.ndarray:
    """Write ``disp(mm)  reaction_force`` and return the saved table."""
    out = (
        merge_force_disp(Path(merge_with), disp, force)
        if merge_with
        else np.c_[disp, force]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, out, header="disp(mm)  reaction_force", comments="")
    return out
