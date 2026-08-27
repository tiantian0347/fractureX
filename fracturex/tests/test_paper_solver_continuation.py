"""Test deterministic helpers used by the slow-mode paper drivers.

The tests cover continuation grids, result-path validation, CSV readers, and
the symmetry completion used by the resolved localization figure. Standard-FE
assembly and checkpoint integration remain in the paper verification scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.paper_solver.verify_slow_mode_fracturex import _intermediate_loads
from scripts.paper_solver.run_model0_fine_reference import (
    _read_resume_report_rows,
    _solve_load,
    _write_internal_state_csv,
    _write_report_csv,
    build_report_loads,
)
from scripts.paper_solver.scan_model0_internal_path_online_rate import (
    read_internal_manifests,
    select_load_interval,
)
from scripts.paper_solver.scan_model0_resolved_online_rate import (
    discover_checkpoints,
    parse_loads,
    require_results_directory,
)
from scripts.paper_solver.scan_model0_resolved_spectrum import parse_coverages
from scripts.paper_solver.plot_model0_resolved_online_rate import (
    read_online_scan,
    read_reaction_curve,
    require_results_figure,
)
from scripts.paper_solver.plot_resolved_localization_map import (
    _reflection_average_cell_field,
)


class _HistoryProbeMaterial:
    """Record history snapshots supplied by the continuation driver."""

    def __init__(self) -> None:
        self.snapshots: list[np.ndarray] = []

    def update_disp(self, displacement: np.ndarray) -> None:
        """Accept the displacement update used by state restoration."""

    def update_phase(self, damage: np.ndarray) -> None:
        """Accept the phase update used by the test driver."""

    def update_historical_field(self, history: np.ndarray) -> None:
        """Record the restored committed history."""
        self.snapshots.append(np.asarray(history, dtype=np.float64).copy())


class _HistoryProbeSolver:
    """Minimal two-sweep solver exposing history leakage."""

    def __init__(self) -> None:
        self.uh = np.zeros(1, dtype=np.float64)
        self.d = np.zeros(1, dtype=np.float64)
        self.H = np.array([2.0], dtype=np.float64)
        self.pfcm = _HistoryProbeMaterial()
        self.phase_entry_history: list[float] = []
        self.iteration = 0

    def solve_displacement(self) -> float:
        """Return a residual that decreases on the second sweep."""
        self.iteration += 1
        return 1.0 if self.iteration == 1 else 0.1

    def solve_phase_field(self) -> float:
        """Create a large trial-history peak and a decreasing phase update."""
        self.phase_entry_history.append(float(self.H[0]))
        self.H = np.array([10.0 + self.iteration], dtype=np.float64)
        self.d += 1.0 if self.iteration == 1 else 0.1
        return 1.0


def test_fine_path_history_is_transactional_within_load_step() -> None:
    """Intermediate trial history is reset to H_(n-1) before every sweep."""
    solver = _HistoryProbeSolver()

    converged, iterations = _solve_load(
        solver,
        load=0.1,
        maxit=3,
        tolerance=0.2,
        damage_relaxation=1.0,
        anderson_depth=0,
    )

    assert converged
    assert iterations == 2
    assert solver.phase_entry_history == pytest.approx([2.0, 2.0])
    assert solver.H.tolist() == pytest.approx([12.0])


def test_exact_decimal_step_is_not_split_by_roundoff() -> None:
    """An interval equal to the requested maximum has no interior point."""
    assert _intermediate_loads(0.105, 0.1075, 0.0025) == []


@pytest.mark.parametrize("start,target", [(0.0, 0.105), (0.105, 0.0)])
def test_subdivision_is_monotone_and_respects_maximum_step(
    start: float, target: float
) -> None:
    """Both loading directions use 42 equal increments for this interval."""
    interior = _intermediate_loads(start, target, 0.0025)
    full_path = np.asarray([start, *interior, target], dtype=np.float64)

    assert len(interior) == 41
    assert np.all(np.sign(target - start) * np.diff(full_path) > 0.0)
    assert np.max(np.abs(np.diff(full_path))) <= 0.0025 * (1.0 + 1.0e-13)


def test_invalid_continuation_step_is_rejected() -> None:
    """The numerical contract requires a finite positive maximum step."""
    with pytest.raises(ValueError, match="finite and positive"):
        _intermediate_loads(0.0, 0.1, 0.0)


def test_model0_report_loads_match_case_contract() -> None:
    """The fine reference preserves the original 31-point load schedule."""
    loads = build_report_loads()

    assert loads.shape == (31,)
    assert loads[0] == pytest.approx(0.0)
    assert loads[5] == pytest.approx(0.07)
    assert loads[-1] == pytest.approx(0.125)
    assert np.all(np.diff(loads) > 0.0)


def test_partial_model0_csv_can_be_extended_atomically(tmp_path) -> None:
    """A converged prefix remains a valid restart record after replacement."""
    csv_path = tmp_path / "residual_force_vs_displacement.csv"
    _write_report_csv(
        csv_path,
        [0.0, 0.014],
        [0.0, 5.9],
        [0, 4],
        "test_stage",
    )

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("step,load,residual_force_abs")
    assert rows[-1].split(",")[:4] == ["1", "0.014", "5.9", "4"]
    assert not csv_path.with_suffix(".csv.tmp").exists()


def test_named_report_checkpoint_truncates_a_longer_resume_csv(tmp_path) -> None:
    """An earlier report restart cannot inherit later report rows."""
    csv_path = tmp_path / "residual_force_vs_displacement.csv"
    _write_report_csv(
        csv_path,
        [0.0, 0.0788, 0.0810],
        [0.0, 20.0, 21.0],
        [0, 7, 9],
        "test_stage",
    )

    rows = _read_resume_report_rows(
        csv_path,
        tmp_path / "report_09_load_0.0788.npz",
    )

    assert [float(row["load"]) for row in rows] == pytest.approx([0.0, 0.0788])


def test_internal_state_manifests_stitch_at_one_restart_boundary(tmp_path) -> None:
    """Two physical-path stages retain the second stage's shared restart."""
    first_dir = tmp_path / "prefix"
    second_dir = tmp_path / "postpeak"
    first_checkpoints = [first_dir / f"state_{index}.npz" for index in range(2)]
    second_checkpoints = [second_dir / f"state_{index}.npz" for index in range(2)]
    for checkpoint in [*first_checkpoints, *second_checkpoints]:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()

    def row(
        index: int,
        load: float,
        checkpoint: Path,
        *,
        report_index: int,
    ) -> dict[str, object]:
        """Build one valid manifest row for this stitching test."""
        return {
            "accepted_index": index,
            "load": load,
            "previous_load": "" if index == 0 else load - 0.0011,
            "step_size": 0.0 if index == 0 else 0.0011,
            "report_index": report_index,
            "is_report": True,
            "staggered_iterations": index,
            "damage_relaxation": 0.5,
            "anderson_depth": 5,
            "algorithm_stage": "safeguarded_anderson",
            "checkpoint": str(checkpoint),
        }

    first_manifest = first_dir / "accepted_internal_states.csv"
    second_manifest = second_dir / "accepted_internal_states.csv"
    _write_internal_state_csv(
        first_manifest,
        [
            row(0, 0.0788, first_checkpoints[0], report_index=9),
            row(1, 0.0799, first_checkpoints[1], report_index=10),
        ],
    )
    _write_internal_state_csv(
        second_manifest,
        [
            row(0, 0.0799, second_checkpoints[0], report_index=10),
            row(1, 0.0810, second_checkpoints[1], report_index=11),
        ],
    )

    records = read_internal_manifests([first_manifest, second_manifest])
    selected = select_load_interval(records, start_load=0.0788, end_load=0.0810)

    assert [record.load for record in selected] == pytest.approx(
        [0.0788, 0.0799, 0.0810]
    )
    assert selected[1].path == first_checkpoints[1].resolve()


def test_resolved_scan_load_parser_enforces_monotone_targets() -> None:
    """Online-rate replay targets have a unique chronological ordering."""
    assert parse_loads("0.0810, 0.0832,0.0854") == pytest.approx(
        [0.0810, 0.0832, 0.0854]
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_loads("0.0832,0.0810")


def test_resolved_spectrum_coverages_are_unique_and_increasing() -> None:
    """Nested slow regions require ordered fractions in the unit interval."""
    assert parse_coverages("0.5, 0.6,0.7") == pytest.approx([0.5, 0.6, 0.7])
    with pytest.raises(ValueError, match="unique, increasing"):
        parse_coverages("0.7,0.6")
    with pytest.raises(ValueError, match="unique, increasing"):
        parse_coverages("0.5,1.1")


def test_resolved_scan_discovers_checkpoint_indices(tmp_path) -> None:
    """Prefix and post-peak checkpoint directories form one ordered path."""
    prefix = tmp_path / "prefix" / "checkpoints"
    postpeak = tmp_path / "postpeak" / "checkpoints"
    prefix.mkdir(parents=True)
    postpeak.mkdir(parents=True)
    (prefix / "report_09_load_0.0788.npz").touch()
    (postpeak / "report_10_load_0.0810.npz").touch()

    records = discover_checkpoints([prefix.parent, postpeak.parent])

    assert [record.index for record in records] == [9, 10]
    assert [record.load for record in records] == pytest.approx([0.0788, 0.0810])


def test_resolved_bridge_plot_merges_validated_csv_inputs(tmp_path) -> None:
    """The paper plot joins the physical path and split online-rate records."""
    reaction_csv = tmp_path / "reaction.csv"
    reaction_csv.write_text(
        "load,residual_force_abs\n0.0,0.0\n0.08,2.0\n0.10,0.2\n",
        encoding="utf-8",
    )
    first_scan = tmp_path / "scan_a.csv"
    first_scan.write_text(
        "load,rhohat_online\n0.08,0.8\n0.09,0.95\n",
        encoding="utf-8",
    )
    second_scan = tmp_path / "scan_b.csv"
    second_scan.write_text(
        "load,rhohat_online\n0.10,1.0\n",
        encoding="utf-8",
    )

    loads, reactions = read_reaction_curve(reaction_csv)
    scan_loads, rates = read_online_scan([first_scan, second_scan])

    assert loads.tolist() == pytest.approx([0.0, 0.08, 0.10])
    assert reactions.tolist() == pytest.approx([0.0, 2.0, 0.2])
    assert scan_loads.tolist() == pytest.approx([0.08, 0.09, 0.10])
    assert rates.tolist() == pytest.approx([0.8, 0.95, 1.0])


def test_paper_experiment_outputs_are_confined_to_results(tmp_path) -> None:
    """Simulation tables and canonical figures stay below their results root."""
    results_root = tmp_path / "results" / "phasefield_solver"
    valid_directory = results_root / "resolved_scan"
    valid_figure = valid_directory / "bridge.pdf"

    assert require_results_directory(
        valid_directory, results_root=results_root
    ) == valid_directory.resolve()
    assert require_results_figure(
        valid_figure, results_root=results_root
    ) == valid_figure.resolve()
    with pytest.raises(ValueError, match="must be stored below"):
        require_results_directory(tmp_path / "paper", results_root=results_root)
    with pytest.raises(ValueError, match="must be stored below"):
        require_results_figure(tmp_path / "paper.pdf", results_root=results_root)


def test_resolved_trace_reflection_average_restores_pair_symmetry() -> None:
    """A left--right cell pair receives the same completed trace value."""
    nodes = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.4, 1.0],
            [1.0, 0.0],
            [0.6, 0.0],
            [0.6, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    completed, defect = _reflection_average_cell_field(
        nodes, cells, np.array([2.0, 6.0], dtype=np.float64)
    )

    assert completed.tolist() == pytest.approx([4.0, 4.0])
    assert completed[0] == completed[1]
    assert defect == pytest.approx(np.sqrt(0.8))
