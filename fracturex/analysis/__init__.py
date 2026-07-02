"""Affine-invariant Newton diagnostics.

See ``docs/preconditioner/THEORY_affine_invariant_newton.md`` and
``docs/preconditioner/DESIGN_affine_invariant_diagnostics.md``.
"""
from fracturex.analysis.affine_invariant import (
    AffineInvariantMonitor,
    AffineInvariantSummary,
    NewtonStepRecord,
    read_iterations_csv,
    write_iteration_detail_csv,
    write_summary_csv,
)

__all__ = [
    "AffineInvariantMonitor",
    "AffineInvariantSummary",
    "NewtonStepRecord",
    "read_iterations_csv",
    "write_iteration_detail_csv",
    "write_summary_csv",
]
