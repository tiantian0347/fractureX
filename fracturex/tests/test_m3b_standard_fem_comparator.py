"""Tests for the standard-FEM comparator smoke script's result readers."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[2] / "scripts" / "paper_operator_learning" / "audit_m3b_standard_fem_comparator.py"
_SPEC = importlib.util.spec_from_file_location("audit_m3b_standard_fem_comparator", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_load_recorded_reactions_and_find_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "residual_force_vs_displacement.csv").write_text(
        "step,load,residual_force_abs\n21,0.1052,0.0375\n30,0.125,0.0255\n"
    )
    checkpoint = checkpoint_dir / "report_21_load_0.1052.npz"
    checkpoint.touch()

    rows = _MODULE.load_recorded_reactions(
        run_dir / "residual_force_vs_displacement.csv"
    )
    assert rows[21] == {"load": 0.1052, "reaction": 0.0375}
    assert _MODULE.find_report_checkpoint(run_dir, 21) == checkpoint
