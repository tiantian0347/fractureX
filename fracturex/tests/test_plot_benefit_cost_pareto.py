"""Test the paper benefit--cost figure's CSV contract and smoke output.

The tests use synthetic accepted records and do not run finite-element solves.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.paper_solver.plot_benefit_cost_pareto import (
    _read_records,
    plot_benefit_cost_sweep,
)


FIELDNAMES = (
    "all_acceptance_checks_passed",
    "theta",
    "region",
    "coupled_region_survival_factor",
    "total_residual_equivalent_evaluations",
    "total_wall_time_seconds",
)


def _write_sweep_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal verification-style CSV for one plotting test."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _matched_rows() -> list[dict[str, object]]:
    """Return two matched thresholds for both supported region families."""
    return [
        {
            "all_acceptance_checks_passed": True,
            "theta": theta,
            "region": region,
            "coupled_region_survival_factor": survival,
            "total_residual_equivalent_evaluations": work,
            "total_wall_time_seconds": time,
        }
        for theta, region, survival, work, time in (
            (0.5, "slow", 0.80, 205, 3.7),
            (0.6, "slow", 0.76, 191, 3.5),
            (0.5, "damage", 0.81, 185, 3.4),
            (0.6, "damage", 0.77, 185, 3.4),
        )
    ]


def test_benefit_cost_plot_writes_nonempty_pdf(tmp_path: Path) -> None:
    """Matched accepted records produce a nonempty vector figure."""
    input_path = tmp_path / "sweep.csv"
    output_path = tmp_path / "sweep.pdf"
    _write_sweep_csv(input_path, _matched_rows())

    plot_benefit_cost_sweep(input_path, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 1000


def test_benefit_cost_plot_rejects_unmatched_thresholds(tmp_path: Path) -> None:
    """Region comparisons require identical trace-threshold sets."""
    input_path = tmp_path / "unmatched.csv"
    rows = _matched_rows()
    rows.pop()
    _write_sweep_csv(input_path, rows)

    with pytest.raises(RuntimeError, match="matching thresholds"):
        _read_records(input_path)
