#!/usr/bin/env python3
"""Plot the online slowdown indicator along the resolved loading path.

Purpose
-------
Create the paper figure for the tangent-weighted online staggered slowdown
indicator on the length-scale-resolved loading path. The reaction-force response is
reported separately, so this figure contains one panel only.

Scope
-----
The script performs CSV validation and plotting only. It does not recompute
finite-element states or alter experiment outputs.

Usage
-----
python scripts/paper_solver/plot_model0_resolved_online_rate.py

The canonical PDF, PNG, and metadata are written below the repository
``results`` directory. A manuscript may keep a copied PDF as a typesetting
asset, while the results directory remains the source of record.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = REPOSITORY_ROOT / "results/phasefield_solver"
DEFAULT_REACTION_CSV = (
    RESULTS_ROOT
    / "model0_fine_curve_audit_unrelaxed_h0065"
    / "residual_force_vs_displacement.csv"
)
DEFAULT_SCAN_CSVS = (
    RESULTS_ROOT / "model0_resolved_online_rate_h0065" / "online_rate_scan.csv",
    RESULTS_ROOT
    / "model0_resolved_online_rate_h0065_load0103"
    / "online_rate_scan.csv",
)
DEFAULT_OUTPUT_PDF = (
    RESULTS_ROOT / "model0_resolved_online_rate_h0065" / "physical_slowdown_bridge.pdf"
)


def read_reaction_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a finite, strictly increasing reaction-force curve.

    Parameters
    ----------
    path : pathlib.Path
        CSV containing ``load`` and ``residual_force_abs`` columns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Load and nonnegative reaction vectors with identical shape.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    loads = np.asarray([float(row["load"]) for row in rows], dtype=np.float64)
    reactions = np.asarray(
        [float(row["residual_force_abs"]) for row in rows], dtype=np.float64
    )
    if (
        loads.size == 0
        or not np.isfinite(loads).all()
        or not np.isfinite(reactions).all()
        or np.any(np.diff(loads) <= 0.0)
        or np.any(reactions < 0.0)
    ):
        raise ValueError("reaction curve must be finite, ordered, and nonnegative")
    return loads, reactions


def require_results_figure(
    output_pdf: Path, *, results_root: Path = RESULTS_ROOT
) -> Path:
    """Resolve a PDF path and require it to lie below the results directory.

    Parameters
    ----------
    output_pdf : pathlib.Path
        Requested canonical figure path. Its suffix must be ``.pdf``.
    results_root : pathlib.Path
        Allowed results root; injectable for unit tests.

    Returns
    -------
    pathlib.Path
        Absolute validated PDF path.

    Raises
    ------
    ValueError
        If the suffix is not PDF or the path lies outside ``results_root``.
    """
    resolved_output = output_pdf.resolve()
    resolved_root = results_root.resolve()
    if resolved_output.suffix.lower() != ".pdf":
        raise ValueError("canonical figure output must use a .pdf suffix")
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"experiment figures must be stored below {resolved_root}"
        ) from error
    return resolved_output


def read_online_scan(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Merge finite online-rate records and return them in load order.

    Parameters
    ----------
    paths : iterable[pathlib.Path]
        CSV files containing ``load`` and ``rhohat_online`` columns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Unique ordered loads and their finite online slowdown indicators.

    Raises
    ------
    ValueError
        If no records exist or a load occurs more than once.
    """
    records: list[tuple[float, float]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                records.append((float(row["load"]), float(row["rhohat_online"])))
    if not records:
        raise ValueError("online scan contains no records")
    records.sort(key=lambda record: record[0])
    values = np.asarray(records, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("online scan loads must be finite and unique")
    return values[:, 0], values[:, 1]


def make_figure(
    reaction_csv: Path,
    scan_csvs: Iterable[Path],
    output_pdf: Path,
) -> Path:
    """Create the online-rate figure and save PDF plus PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_pdf = require_results_figure(output_pdf)
    # Validate the paired physical response even though this figure displays
    # only the online indicator; the response is plotted separately in Fig. 3.
    read_reaction_curve(reaction_csv)
    scan_paths = [path.resolve() for path in scan_csvs]
    scan_loads, rates = read_online_scan(scan_paths)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    text_color = "#202124"
    fast_color = "#277DA1"
    slow_color = "#B23A48"
    grid_color = "#D9DDE3"
    gray_zone_color = "#F4E7B2"
    figure, axis_rate = plt.subplots(figsize=(6.7, 3.8))

    axis_rate.axhspan(0.87, 0.89, color=gray_zone_color, alpha=0.8, zorder=0)
    axis_rate.axhline(0.89, color="#8A6D1D", linestyle="--", linewidth=1.0)
    axis_rate.axhline(1.0, color="#73777F", linestyle=":", linewidth=1.0)
    # The second replay starts with a fresh rolling window.  A large downward
    # jump at that boundary is therefore a diagnostic reset, not a physical
    # branch of the loading path; draw the two records as separate segments.
    reset_indices = np.flatnonzero(np.diff(rates) < -0.25) + 1
    segment_starts = np.concatenate(([0], reset_indices))
    segment_ends = np.concatenate((reset_indices, [rates.size]))
    for start, end in zip(segment_starts, segment_ends):
        # Retain the reset sample as a point, but do not draw an artificial
        # segment across the fresh rolling-window initialization.
        if end - start > 2 or not reset_indices.size:
            axis_rate.plot(
                scan_loads[start:end],
                rates[start:end],
                color=fast_color,
                linewidth=1.55,
                zorder=2,
            )
    slow_mask = rates >= 0.89
    reset_mask = np.zeros(rates.shape, dtype=bool)
    reset_mask[reset_indices] = True
    axis_rate.scatter(
        scan_loads[~slow_mask & ~reset_mask],
        rates[~slow_mask & ~reset_mask],
        color=fast_color,
        s=27,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=r"$\widehat\rho_{\rm on}$",
    )
    if reset_indices.size:
        axis_rate.scatter(
            scan_loads[reset_mask],
            rates[reset_mask],
            color="#8A8F98",
            s=27,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label="monitoring-window restart",
        )
    axis_rate.scatter(
        scan_loads[slow_mask],
        rates[slow_mask],
        color=slow_color,
        s=31,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    axis_rate.text(
        0.0801,
        0.895,
        r"activation threshold $0.89$",
        color="#7A601A",
        fontsize=8.0,
        va="bottom",
    )
    axis_rate.set_ylabel(r"online slowdown indicator $\widehat\rho_{\rm on}$")
    axis_rate.set_xlabel(r"prescribed displacement $\bar u$")
    axis_rate.set_ylim(0.47, 1.035)
    axis_rate.set_xlim(0.0, 0.125)
    axis_rate.legend(loc="lower left", frameon=False, fontsize=8.3, ncol=2)

    for axis in (axis_rate,):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8.5, colors=text_color)
        axis.yaxis.grid(True, color=grid_color, linewidth=0.6, alpha=0.75)
        axis.xaxis.grid(False)
        axis.set_axisbelow(True)

    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.96)
    figure.savefig(output_pdf, bbox_inches="tight")
    output_png = output_pdf.with_suffix(".png")
    figure.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "command": shlex.join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reaction_csv": str(reaction_csv.resolve()),
        "scan_csvs": [str(path) for path in scan_paths],
        "output_pdf": str(output_pdf.resolve()),
        "output_png": str(output_png.resolve()),
        "output_dir": str(output_pdf.parent.resolve()),
        "storage_contract": "canonical experiment artifacts are stored below results/",
        "gate": 0.89,
        "gray_zone": [0.87, 0.89],
        "online_window_resets": [int(index) for index in reset_indices],
    }
    output_pdf.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_pdf


def parse_args() -> argparse.Namespace:
    """Parse input CSVs and deterministic figure output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-csv", type=Path, default=DEFAULT_REACTION_CSV)
    parser.add_argument(
        "--scan-csv",
        type=Path,
        nargs="+",
        default=list(DEFAULT_SCAN_CSVS),
    )
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    return parser.parse_args()


def main() -> None:
    """Create the configured paper figure."""
    args = parse_args()
    path = make_figure(
        args.reaction_csv.resolve(),
        args.scan_csv,
        args.output_pdf.resolve(),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
