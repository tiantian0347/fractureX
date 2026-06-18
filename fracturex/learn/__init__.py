# fracturex/learn/__init__.py
"""Operator-learning surrogate package.

Decoupled from the FEALPy-backed solver side: this package only reads the
data protocol defined in docs/operator_learning/SURROGATE_DATA_SCHEMA.md. PyTorch is the
default training backend; JAX support is optional.

See docs/operator_learning/plan_operator_learning.md §M1 onward.
"""
from __future__ import annotations

__schema_version__ = "0.1"
