"""Analysis-layer diagnostics and preconditioners.

See ``docs/preconditioner/THEORY_affine_invariant_newton.md`` (Ch. 8),
``docs/preconditioner/THEORY_nonlinear_elimination.md`` (Ch. 10),
and their matched design docs.
"""
from fracturex.analysis.affine_invariant import (
    AffineInvariantMonitor,
    AffineInvariantSummary,
    NewtonStepRecord,
    read_iterations_csv,
    write_iteration_detail_csv,
    write_summary_csv,
)
from fracturex.analysis.nonlinear_elimination import (
    NEPINConfig,
    NEPINEliminator,
    NEPINResult,
    identify_subset,
)

__all__ = [
    "AffineInvariantMonitor",
    "AffineInvariantSummary",
    "NewtonStepRecord",
    "read_iterations_csv",
    "write_iteration_detail_csv",
    "write_summary_csv",
    "NEPINConfig",
    "NEPINEliminator",
    "NEPINResult",
    "identify_subset",
]
