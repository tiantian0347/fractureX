#!/usr/bin/env python3
"""Plot the Model-0 benefit--cost threshold sweep from its flat CSV result.

The figure aligns the spectral benefit and solver work against the same trace
threshold. It visualizes existing verification data and does not run
finite-element solves.

Usage
-----
python scripts/paper_solver/plot_benefit_cost_pareto.py \
    --input results/phasefield_solver/model0_benefit_cost_pareto_h005/benefit_cost_pareto.csv \
    --output figures/model0_benefit_cost_pareto.pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REGION_STYLES = {
    "slow": {
        "color": "#1F6D8F",
        "marker": "o",
        "label": "Slow-subspace region",
    },
    "damage": {
        "color": "#D07A24",
        "marker": "s",
        "label": "Damage region",
    },
}


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read and validate accepted threshold-sweep records.

    Parameters
    ----------
    path : Path
        UTF-8 CSV produced by ``verify_slow_mode_fracturex.py``.

    Returns
    -------
    list[dict[str, Any]]
        Accepted records with finite ``theta``, survival factor, and equivalent
        work converted to ``float``. The input file is not modified.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    RuntimeError
        If accepted records are missing, non-finite, contain an unsupported
        region, or do not provide matched threshold sets.
    """
    if not path.is_file():
        raise FileNotFoundError(f"benefit--cost CSV does not exist: {path}")
    numeric_fields = (
        "theta",
        "coupled_region_survival_factor",
        "total_residual_equivalent_evaluations",
        "total_wall_time_seconds",
    )
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["all_acceptance_checks_passed"] != "True":
                continue
            for field in numeric_fields:
                row[field] = float(row[field])
            if row["region"] not in REGION_STYLES:
                raise RuntimeError(
                    f"unsupported accepted region {row['region']!r} in {path}"
                )
            if not np.isfinite([row[field] for field in numeric_fields]).all():
                raise RuntimeError(f"non-finite plotting value in {path}")
            records.append(row)
    if not records:
        raise RuntimeError("benefit--cost CSV contains no accepted records")
    theta_sets = {
        region: {
            float(record["theta"])
            for record in records
            if record["region"] == region
        }
        for region in REGION_STYLES
    }
    if any(not values for values in theta_sets.values()):
        raise RuntimeError("benefit--cost CSV must contain both region families")
    if theta_sets["slow"] != theta_sets["damage"]:
        raise RuntimeError("slow and damage regions must use matching thresholds")
    return records


def _plot_threshold_series(
    axis: Any,
    records: list[dict[str, Any]],
    y_field: str,
) -> None:
    """Plot one metric against the common dimensionless threshold.

    ``records`` must satisfy the contract established by ``_read_records``.
    The function mutates only the supplied Matplotlib axis.
    """
    for region, style in REGION_STYLES.items():
        selected = sorted(
            (record for record in records if record["region"] == region),
            key=lambda record: record["theta"],
        )
        x_values = np.asarray([record["theta"] for record in selected])
        y_values = np.asarray([record[y_field] for record in selected])
        axis.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=1.8,
            markersize=5.8,
            markerfacecolor="white",
            markeredgewidth=1.5,
            solid_capstyle="round",
            zorder=3,
        )
    thresholds = sorted({float(record["theta"]) for record in records})
    axis.set_xticks(thresholds)
    axis.set_xticklabels([f"{value:.1f}" for value in thresholds])
    axis.set_xlabel(r"Trace threshold $\theta$")
    axis.grid(axis="y", color="#DDE3E8", linewidth=0.65)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(0.9)


def _annotate_key_results(axes: Any, records: list[dict[str, Any]]) -> None:
    """Highlight the global minimum residual and minimum-work plateau."""
    residual_record = min(
        records,
        key=lambda record: record["coupled_region_survival_factor"],
    )
    axes[0].scatter(
        residual_record["theta"],
        residual_record["coupled_region_survival_factor"],
        s=30,
        color=REGION_STYLES["slow"]["color"],
        zorder=4,
    )
    axes[0].text(
        residual_record["theta"] - 0.025,
        residual_record["coupled_region_survival_factor"] + 0.055,
        r"lowest $\chi_\omega$",
        horizontalalignment="right",
        fontsize=8,
        color="#33414A",
    )

    minimum_work = min(
        record["total_residual_equivalent_evaluations"] for record in records
    )
    minimum_records = [
        record
        for record in records
        if np.isclose(
            record["total_residual_equivalent_evaluations"], minimum_work
        )
    ]
    minimum_thresholds = sorted(record["theta"] for record in minimum_records)
    axes[1].fill_between(
        [minimum_thresholds[0], minimum_thresholds[-1]],
        minimum_work - 0.65,
        minimum_work + 0.65,
        color=REGION_STYLES["damage"]["color"],
        alpha=0.12,
        linewidth=0.0,
        zorder=1,
    )
    axes[1].text(
        float(np.mean(minimum_thresholds)),
        minimum_work + 1.8,
        rf"minimum work: $N_{{\rm eq}}={minimum_work:.0f}$",
        horizontalalignment="center",
        fontsize=8,
        color="#33414A",
    )


def plot_benefit_cost_sweep(input_path: Path, output_path: Path) -> None:
    """Create the aligned two-panel benefit--cost figure.

    Parameters
    ----------
    input_path : Path
        Accepted Model--0 threshold records in CSV format.
    output_path : Path
        PDF or raster destination. Parent directories are created. Existing
        output is replaced; the source CSV is never modified.
    """
    records = _read_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.55))
    _plot_threshold_series(
        axes[0], records, "coupled_region_survival_factor"
    )
    _plot_threshold_series(
        axes[1], records, "total_residual_equivalent_evaluations"
    )
    axes[0].set_ylabel(r"Slow-mode survival factor $\chi_{\omega,W}(\mathcal{V}_2)$")
    axes[1].set_ylabel(r"Equivalent work $N_{\rm eq}$")
    axes[0].set_ylim(0.45, 0.84)
    axes[1].set_ylim(182.0, 208.0)
    axes[0].text(
        -0.13, 1.06, "(a)", transform=axes[0].transAxes, clip_on=False
    )
    axes[1].text(
        -0.13, 1.06, "(b)", transform=axes[1].transAxes, clip_on=False
    )
    _annotate_key_results(axes, records)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=2.0,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.22,
        top=0.78,
        wspace=0.30,
    )
    figure.savefig(output_path, dpi=240)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    """Parse input and output paths for deterministic figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate the requested benefit--cost figure."""
    args = _parse_args()
    plot_benefit_cost_sweep(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
