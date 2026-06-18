"""Learned auxiliary-space coarse-space enrichment for the Hu-Zhang phase-field
block preconditioner (D13).

This package is intentionally decoupled from the solver and the operator-learning
``learn/`` line:

- modules here import **no** solver and **no** torch at module top level
  (``coarse_features`` is pure numpy + FEALPy mesh/function);
- the torch model and the solver-facing adapter live in separate modules so the
  GMRES hot path never touches torch (inference runs once per staggered setup).

See ``docs/preconditioner/D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md`` §13.1.
"""
