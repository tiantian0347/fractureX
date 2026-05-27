#!/usr/bin/env python3
# paper_make_loaddisp.py
#
# Build the load-displacement comparison figure for the paper's section 7.2
# (Consistency of aux-space and direct solvers, Claim C1). Each panel overlays
# the direct-solver curve and the aux-space-preconditioned-GMRES curve for one
# benchmark, with optional reference data (e.g. digitized Miehe 2010 for model-1).
#
# Inputs are residual_force_vs_displacement.csv files written by run_case.py via
# fracturex.postprocess.run_report.export_residual_force_displacement_curve.
#
# Layout
#   - Multi-panel horizontal: one panel per case
#   - Direct: solid line, Aux: dashed line, Reference: dotted line + markers
#   - Sanity check: refuses to plot a curve with max|F| below --min-force-threshold
#     (catches the 2026-05-27 model2 bogus-data trap; see memory
#     model2-paper-direct-bogus and phasefield-sanity-check)
#
# Outputs
#   <out_prefix>.png
#   <out_prefix>.pdf
#   <out_prefix>.npz   (per-case arrays of disp / force_direct / force_aux / ref)
#
# Usage
#   python paper_make_loaddisp.py \
#       --case model0 \
#           --direct results/.../paper_direct_h2/epsg_1e-06/residual_force_vs_displacement.csv \
#           --aux    results/.../paper_aux_h2/epsg_1e-06/residual_force_vs_displacement.csv \
#       --case model1 \
#           --direct results/.../square_tension_precrack/paper_direct/epsg_1e-06/residual_force_vs_displacement.csv \
#           --aux    results/.../square_tension_precrack/paper_aux/epsg_1e-06/residual_force_vs_displacement.csv \
#           --ref    references/miehe2010_model1.csv \
#       --out figures/loaddisp_all
#
# Case args are positional groups: each --case starts a new group; --direct /
# --aux / --ref / --label inside that group all attach to it.

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------- data model


@dataclass
class CaseInputs:
    name: str
    label: Optional[str] = None  # display label override; defaults to name
    direct_csv: Optional[Path] = None
    aux_csv: Optional[Path] = None
    ref_csv: Optional[Path] = None

    @property
    def display_label(self) -> str:
        return self.label or self.name


@dataclass
class CurveData:
    disp: np.ndarray
    force: np.ndarray
    source: Path


@dataclass
class CasePanel:
    inputs: CaseInputs
    direct: Optional[CurveData] = None
    aux: Optional[CurveData] = None
    ref: Optional[CurveData] = None
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- CLI


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", required=True, help="output prefix (no extension); writes .png/.pdf/.npz")
    p.add_argument("--case", action="append", default=[], help="start a new case group")
    p.add_argument("--label", action="append", default=[], help="display label for the current case (optional)")
    p.add_argument("--direct", action="append", default=[], help="direct-solver CSV for the current case")
    p.add_argument("--aux", action="append", default=[], help="aux-space CSV for the current case (optional)")
    p.add_argument("--ref", action="append", default=[], help="reference CSV for the current case (optional)")
    p.add_argument(
        "--min-force-threshold",
        type=float,
        default=1e-3,
        help="warn (and skip plotting) when max|F| of a curve falls below this. Default 1e-3.",
    )
    p.add_argument("--no-sanity", action="store_true", help="disable the min-force sanity check")
    p.add_argument("--figsize", default="auto", help="figsize, e.g. 12,4. Default: auto from number of cases.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--force-scale",
        type=float,
        default=1.0,
        help="scale factor applied to force values (e.g. 1e-3 to convert to kN).",
    )
    p.add_argument("--force-unit", default="", help="optional unit label for the y-axis, e.g. 'kN'.")
    args = p.parse_args(argv)
    return args


def assemble_cases(args: argparse.Namespace) -> List[CaseInputs]:
    """Group --direct / --aux / --ref / --label flags under their --case parent.

    The argparse 'append' lists are ordered, but we need to know which flag was
    attached to which --case group. We use a simple convention: every --case
    consumes the *next* available --direct / --aux / --ref / --label slot from
    each list in order. That matches positional usage like:
        --case A --direct a.csv --aux a_aux.csv --case B --direct b.csv
    Missing optional flags inside a group are simply None.
    """
    if not args.case:
        raise SystemExit("at least one --case is required.")

    # We need to walk sys.argv to learn the relative order of --case vs the
    # other group-scoped flags. argparse alone won't tell us.
    argv = sys.argv[1:]
    cases: List[CaseInputs] = []
    current: Optional[CaseInputs] = None
    direct_iter = iter(args.direct)
    aux_iter = iter(args.aux)
    ref_iter = iter(args.ref)
    label_iter = iter(args.label)

    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--case":
            name = argv[i + 1]
            current = CaseInputs(name=name)
            cases.append(current)
            i += 2
            continue
        if tok in ("--direct", "--aux", "--ref", "--label"):
            if current is None:
                raise SystemExit(f"{tok} appeared before any --case")
            val = argv[i + 1]
            if tok == "--direct":
                current.direct_csv = Path(val)
                next(direct_iter, None)
            elif tok == "--aux":
                current.aux_csv = Path(val)
                next(aux_iter, None)
            elif tok == "--ref":
                current.ref_csv = Path(val)
                next(ref_iter, None)
            elif tok == "--label":
                current.label = val
                next(label_iter, None)
            i += 2
            continue
        i += 1
    return cases


# ---------------------------------------------------------------- IO + sanity


def load_curve(path: Path) -> CurveData:
    """Read a (step, displacement, residual_force_abs) CSV emitted by
    fracturex.postprocess.run_report. Other 2-column or 3-column CSVs are
    accepted by treating the last two columns as (disp, force).
    """
    disp: List[float] = []
    force: List[float] = []
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        d_col, f_col = _infer_disp_force_columns(header, path)
        for row in rdr:
            if not row:
                continue
            try:
                d = float(row[d_col])
                v = float(row[f_col])
            except (ValueError, IndexError):
                continue
            disp.append(d)
            force.append(v)
    return CurveData(disp=np.asarray(disp), force=np.asarray(force), source=path)


def _infer_disp_force_columns(header: List[str], path: Path) -> Tuple[int, int]:
    norm = [h.strip().lower() for h in header]
    d_keys = ("displacement", "disp", "u", "load")
    f_keys = ("residual_force_abs", "residual_force", "reaction", "force", "f", "r")
    d_col = next((i for i, h in enumerate(norm) if h in d_keys), None)
    f_col = next((i for i, h in enumerate(norm) if h in f_keys), None)
    if d_col is None or f_col is None:
        if len(header) >= 3:
            return 1, 2
        if len(header) == 2:
            return 0, 1
        raise SystemExit(f"{path}: cannot infer (disp, force) columns from header {header}")
    return d_col, f_col


def sanity_check(curve: CurveData, threshold: float) -> Optional[str]:
    """Return a warning string if the curve looks degenerate; None if it's OK."""
    if curve.force.size == 0:
        return "empty force array"
    fmax = float(np.max(np.abs(curve.force)))
    if fmax < threshold:
        return (
            f"max|F|={fmax:.3e} < threshold={threshold:.3e}; "
            f"likely a degenerate solve (e.g. u=0). Source: {curve.source}"
        )
    return None


# ---------------------------------------------------------------- plotting


def build_panels(cases: List[CaseInputs], threshold: float, do_sanity: bool) -> List[CasePanel]:
    panels: List[CasePanel] = []
    for c in cases:
        panel = CasePanel(inputs=c)
        if c.direct_csv:
            panel.direct = load_curve(c.direct_csv)
            if do_sanity:
                msg = sanity_check(panel.direct, threshold)
                if msg:
                    panel.warnings.append(f"DIRECT: {msg}")
                    panel.direct = None
        if c.aux_csv:
            panel.aux = load_curve(c.aux_csv)
            if do_sanity:
                msg = sanity_check(panel.aux, threshold)
                if msg:
                    panel.warnings.append(f"AUX: {msg}")
                    panel.aux = None
        if c.ref_csv:
            panel.ref = load_curve(c.ref_csv)
        panels.append(panel)
    return panels


def plot_panels(
    panels: List[CasePanel],
    out_prefix: Path,
    *,
    figsize: Tuple[float, float],
    dpi: int,
    force_scale: float,
    force_unit: str,
) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes_flat = axes[0]

    for ax, panel in zip(axes_flat, panels):
        any_drawn = False
        if panel.direct is not None:
            ax.plot(
                panel.direct.disp,
                np.abs(panel.direct.force) * force_scale,
                "-",
                color="#1f77b4",
                lw=1.6,
                label="direct (Pardiso)",
            )
            any_drawn = True
        if panel.aux is not None:
            ax.plot(
                panel.aux.disp,
                np.abs(panel.aux.force) * force_scale,
                "--",
                color="#d62728",
                lw=1.6,
                label="aux-space GMRES",
            )
            any_drawn = True
        if panel.ref is not None:
            ax.plot(
                panel.ref.disp,
                np.abs(panel.ref.force) * force_scale,
                ":o",
                color="#2ca02c",
                lw=1.2,
                ms=4,
                label="reference",
            )
            any_drawn = True

        ax.set_title(panel.inputs.display_label)
        ax.set_xlabel("imposed displacement")
        unit_str = f" [{force_unit}]" if force_unit else ""
        ax.set_ylabel(f"|reaction force|{unit_str}")
        ax.grid(True, alpha=0.3)
        if any_drawn:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "no usable data\n(see log)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
            )

    fig.tight_layout()
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


def save_npz(panels: List[CasePanel], out_prefix: Path) -> None:
    payload = {}
    for panel in panels:
        key = panel.inputs.name
        if panel.direct is not None:
            payload[f"{key}__direct_disp"] = panel.direct.disp
            payload[f"{key}__direct_force"] = panel.direct.force
        if panel.aux is not None:
            payload[f"{key}__aux_disp"] = panel.aux.disp
            payload[f"{key}__aux_force"] = panel.aux.force
        if panel.ref is not None:
            payload[f"{key}__ref_disp"] = panel.ref.disp
            payload[f"{key}__ref_force"] = panel.ref.force
    out_npz = out_prefix.with_suffix(".npz")
    np.savez(out_npz, **payload)
    print(f"wrote {out_npz}")


# ---------------------------------------------------------------- main


def _resolve_figsize(spec: str, n_panels: int) -> Tuple[float, float]:
    if spec == "auto":
        return (4.5 * max(1, n_panels), 3.6)
    parts = spec.split(",")
    if len(parts) != 2:
        raise SystemExit(f"--figsize must be 'auto' or 'W,H'; got {spec!r}")
    return float(parts[0]), float(parts[1])


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cases = assemble_cases(args)
    panels = build_panels(cases, args.min_force_threshold, do_sanity=not args.no_sanity)

    # Report warnings up front so the human sees them.
    has_warn = False
    for panel in panels:
        if panel.warnings:
            has_warn = True
            print(f"[{panel.inputs.name}]")
            for w in panel.warnings:
                print(f"  WARNING: {w}")
    if has_warn:
        print()

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    figsize = _resolve_figsize(args.figsize, n_panels=len(panels))
    plot_panels(
        panels,
        out_prefix,
        figsize=figsize,
        dpi=args.dpi,
        force_scale=args.force_scale,
        force_unit=args.force_unit,
    )
    save_npz(panels, out_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
