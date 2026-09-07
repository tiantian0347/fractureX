"""Unit tests for native-source audit step selection."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).parents[2] / "scripts" / "paper_operator_learning" / "audit_m3b_native_sources.py"
_SPEC = importlib.util.spec_from_file_location("audit_m3b_native_sources", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
select_converged_steps = _MODULE.select_converged_steps


def test_select_converged_steps_includes_available_endpoints():
    selected = select_converged_steps(
        list(range(8)),
        np.array([1, 0, 1, 1, 0, 1, 1, 1], dtype=bool),
        max_steps=4,
    )
    assert selected[0] == 0
    assert selected[-1] == 7
    assert len(selected) == 4
    assert all(step in {0, 2, 3, 5, 6, 7} for step in selected)


def test_select_converged_steps_returns_empty_when_none_are_valid():
    assert select_converged_steps([0, 1], np.zeros(2, dtype=bool), 4) == []
