#!/usr/bin/env python3
# paper_make_iter_vs_N.py
#
# Build the iteration-count-vs-stress-DOF figure for the paper's section 7.3
# (Mesh-independent GMRES iteration count, Claim C2). This is the paper's
# flagship plot: a curve per solver variant, with the auxiliary-space
# preconditioner expected to stay essentially flat across the five-tier mesh
# hierarchy of model-0.
#
# Inputs are iterations.csv files written by run_case.py via
# fracturex.postprocess.recorder. The script aggregates linear_niter_elastic
# (the inner GMRES iteration count of the elastic block solve) across all
# (load step, staggered iteration) entries of each run, then plots the
# per-run aggregate vs gdof_sigma (the size of the stress block, which scales
# with the mesh tier).
#
# Each --series defines one curve. Within a series, one --point per mesh tier
# attaches a (label, iterations.csv) pair. Example:
#
#   paper_make_iter_vs_N.py \
#       --series "aux-space GMRES" \
#           --point h1=results/.../paper_aux_h1/epsg_1e-06/iterations.csv \
#           --point h2=results/.../paper_aux_h2/epsg_1e-06/iterations.csv \
#           --point h3=results/.../paper_aux_h3/epsg_1e-06/iterations.csv \
#       --series "block-diag (direct A^{-1})" \
#           --point h1=results/.../paper_direct_block_h1/.../iterations.csv \
#       --out figures/iter_vs_N
#
# Aggregation defaults to the mean linear_niter_elastic; --reduce can switch
# to median / p90 / max. The shaded band shows the 10th-90th percentile range
# (set --no-band to disable).

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------- data model


@dataclass
class PointSpec:
    tag: str  # mesh tier label, e.g. "h1", "h3"
    path: Path


@dataclass
class SeriesSpec:
    label: str
    points: List[PointSpec] = field(default_factory=list)


@dataclass
class PointStats:
    tag: str
    source: Path
    n_rows: int
    gdof_sigma: int
    gdof_u: int
    gdof_d: int
    iter_mean: float
    iter_median: float
    iter_min: int
    iter_max: int
    iter_p10: float
    iter_p90: float
    iter_p99: float
    iter_std: float
    n_nonconverged: int  # rows with linear_converged_elastic != True


@dataclass
class SeriesResult:
    label: str
    points: List[PointStats]
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- CLI


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", required=True, help="output prefix (no extension); writes .png/.pdf/.npz/.json")
    p.add_argument("--series", action="append", default=[], help="start a new series with the given label")
    p.add_argument(
        "--point",
        action="append",
        default=[],
        help="add a point to the current series as TAG=path/to/iterations.csv",
    )
    p.add_argument(
        "--reduce",
        choices=("mean", "median", "p90", "max"),
        default="mean",
        help="aggregation function for the main curve. Default: mean.",
    )
    p.add_argument("--no-band", action="store_true", help="disable the P10-P90 shaded band")
    p.add_argument("--xscale", choices=("log", "linear"), default="log")
    p.add_argument("--yscale", choices=("log", "linear"), default="linear")
    p.add_argument("--ylim", default="auto", help="y-axis limits as 'low,high' or 'auto'.")
    p.add_argument("--ref-line", type=float, default=None, help="optional horizontal reference line, e.g. 1.0")
    p.add_argument("--figsize", default="6.5,4.2", help="figsize, e.g. 6.5,4.2")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--min-rows",
        type=int,
        default=5,
        help="skip a point if its iterations.csv has fewer than this many rows (default 5)",
    )
    p.add_argument(
        "--exclude-trivial",
        action="store_true",
        help=(
            "drop rows where linear_niter_elastic == 1 from the aggregation. Useful for the "
            "direct-A-on-saddle baseline where Pardiso solves the whole block and reports iter=1, "
            "which is meaningless for an iteration-count comparison."
        ),
    )
    return p.parse_args(argv)


def assemble_series(args: argparse.Namespace) -> List[SeriesSpec]:
    if not args.series:
        raise SystemExit("at least one --series is required.")

    argv = sys.argv[1:]
    series: List[SeriesSpec] = []
    current: Optional[SeriesSpec] = None

    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--series":
            label = argv[i + 1]
            current = SeriesSpec(label=label)
            series.append(current)
            i += 2
            continue
        if tok == "--point":
            if current is None:
                raise SystemExit("--point appeared before any --series")
            spec = argv[i + 1]
            if "=" not in spec:
                raise SystemExit(f"--point expects TAG=path; got {spec!r}")
            tag, path = spec.split("=", 1)
            current.points.append(PointSpec(tag=tag, path=Path(path)))
            i += 2
            continue
        i += 1
    return series


# ---------------------------------------------------------------- IO + reduction


def load_iter_stats(
    spec: PointSpec,
    *,
    min_rows: int,
    exclude_trivial: bool,
) -> Optional[PointStats]:
    if not spec.path.exists():
        return _stats_with_warning(spec, f"file not found: {spec.path}")

    rows = list(csv.DictReader(spec.path.open()))
    if len(rows) < min_rows:
        return _stats_with_warning(spec, f"only {len(rows)} rows (< min-rows={min_rows})")

    try:
        gdof_sigma = int(float(rows[0]["gdof_sigma"]))
        gdof_u = int(float(rows[0]["gdof_u"]))
        gdof_d = int(float(rows[0]["gdof_d"]))
    except (KeyError, ValueError) as e:
        return _stats_with_warning(spec, f"missing gdof columns: {e}")

    iters: List[int] = []
    n_nonconverged = 0
    for r in rows:
        try:
            v = int(float(r["linear_niter_elastic"]))
        except (KeyError, ValueError):
            continue
        if exclude_trivial and v <= 1:
            continue
        iters.append(v)
        conv = r.get("linear_converged_elastic", "").strip().lower()
        if conv not in ("true", "1", "yes"):
            n_nonconverged += 1

    if not iters:
        return _stats_with_warning(spec, "no usable linear_niter_elastic rows after filtering")

    a = np.asarray(iters, dtype=float)
    return PointStats(
        tag=spec.tag,
        source=spec.path,
        n_rows=len(iters),
        gdof_sigma=gdof_sigma,
        gdof_u=gdof_u,
        gdof_d=gdof_d,
        iter_mean=float(np.mean(a)),
        iter_median=float(np.median(a)),
        iter_min=int(np.min(a)),
        iter_max=int(np.max(a)),
        iter_p10=float(np.percentile(a, 10)),
        iter_p90=float(np.percentile(a, 90)),
        iter_p99=float(np.percentile(a, 99)),
        iter_std=float(np.std(a)),
        n_nonconverged=n_nonconverged,
    )


def _stats_with_warning(spec: PointSpec, msg: str) -> None:
    print(f"  WARNING ({spec.tag}): {msg}")
    return None


def build_results(series: List[SeriesSpec], *, min_rows: int, exclude_trivial: bool) -> List[SeriesResult]:
    out: List[SeriesResult] = []
    for s in series:
        print(f"[{s.label}]")
        stats: List[PointStats] = []
        for pt in s.points:
            r = load_iter_stats(pt, min_rows=min_rows, exclude_trivial=exclude_trivial)
            if r is None:
                continue
            print(
                f"  {pt.tag}: N_sigma={r.gdof_sigma:>10d}  "
                f"iter mean={r.iter_mean:6.2f}  median={r.iter_median:5.1f}  "
                f"P10-P90={r.iter_p10:.1f}-{r.iter_p90:.1f}  max={r.iter_max}  "
                f"(n_rows={r.n_rows}, nonconv={r.n_nonconverged})"
            )
            stats.append(r)
        stats.sort(key=lambda x: x.gdof_sigma)
        out.append(SeriesResult(label=s.label, points=stats))
    return out


# ---------------------------------------------------------------- plotting


_SERIES_STYLE = [
    {"color": "#c0392b", "marker": "o", "linestyle": "-"},
    {"color": "#1f3a93", "marker": "s", "linestyle": "--"},
    {"color": "#16a085", "marker": "^", "linestyle": "-."},
    {"color": "#8e44ad", "marker": "D", "linestyle": ":"},
]


def _paper_rc_params() -> dict:
    return {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 6.0,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
    }


def _reduce_curve(stats: List[PointStats], reduce: str) -> np.ndarray:
    table = {
        "mean": "iter_mean",
        "median": "iter_median",
        "p90": "iter_p90",
        "max": "iter_max",
    }
    attr = table[reduce]
    return np.asarray([float(getattr(s, attr)) for s in stats])


def plot_iter_vs_N(
    results: List[SeriesResult],
    out_prefix: Path,
    *,
    reduce: str,
    show_band: bool,
    xscale: str,
    yscale: str,
    ylim: Optional[Tuple[float, float]],
    ref_line: Optional[float],
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    with plt.rc_context(_paper_rc_params()):
        fig, ax = plt.subplots(figsize=figsize)

        any_drawn = False
        for idx, res in enumerate(results):
            if not res.points:
                continue
            style = _SERIES_STYLE[idx % len(_SERIES_STYLE)]
            N = np.asarray([s.gdof_sigma for s in res.points], dtype=float)
            y = _reduce_curve(res.points, reduce)
            ax.plot(
                N,
                y,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                label=res.label,
                mec="white",
                mew=0.6,
            )
            if show_band and len(res.points) > 0:
                lo = np.asarray([s.iter_p10 for s in res.points])
                hi = np.asarray([s.iter_p90 for s in res.points])
                ax.fill_between(N, lo, hi, color=style["color"], alpha=0.12, linewidth=0)
            any_drawn = True

        if ref_line is not None:
            ax.axhline(ref_line, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(r"stress DOF count $N_\sigma$")
        reduce_label = {
            "mean": r"mean GMRES iters / staggered step",
            "median": r"median GMRES iters / staggered step",
            "p90": r"P90 GMRES iters / staggered step",
            "max": r"max GMRES iters / staggered step",
        }[reduce]
        ax.set_ylabel(reduce_label)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.grid(True, which="major")
        ax.grid(True, which="minor", linewidth=0.3, alpha=0.25)
        ax.minorticks_on()

        if any_drawn:
            ax.legend(loc="best", frameon=True, framealpha=0.9)
        else:
            ax.text(
                0.5,
                0.5,
                "no usable data\n(see log)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
                style="italic",
            )

        fig.tight_layout()
        out_png = out_prefix.with_suffix(".png")
        out_pdf = out_prefix.with_suffix(".pdf")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


def save_artifacts(results: List[SeriesResult], out_prefix: Path) -> None:
    npz_payload: Dict[str, np.ndarray] = {}
    json_payload: Dict = {"series": []}
    for res in results:
        key = res.label.replace(" ", "_").replace("/", "_")
        N = np.asarray([s.gdof_sigma for s in res.points], dtype=int)
        mean = np.asarray([s.iter_mean for s in res.points])
        med = np.asarray([s.iter_median for s in res.points])
        p10 = np.asarray([s.iter_p10 for s in res.points])
        p90 = np.asarray([s.iter_p90 for s in res.points])
        vmax = np.asarray([s.iter_max for s in res.points])
        npz_payload[f"{key}__N_sigma"] = N
        npz_payload[f"{key}__iter_mean"] = mean
        npz_payload[f"{key}__iter_median"] = med
        npz_payload[f"{key}__iter_p10"] = p10
        npz_payload[f"{key}__iter_p90"] = p90
        npz_payload[f"{key}__iter_max"] = vmax
        json_payload["series"].append(
            {
                "label": res.label,
                "points": [
                    {
                        "tag": s.tag,
                        "source": str(s.source),
                        "N_sigma": s.gdof_sigma,
                        "iter_mean": s.iter_mean,
                        "iter_median": s.iter_median,
                        "iter_p10": s.iter_p10,
                        "iter_p90": s.iter_p90,
                        "iter_p99": s.iter_p99,
                        "iter_max": s.iter_max,
                        "iter_std": s.iter_std,
                        "n_rows": s.n_rows,
                        "n_nonconverged": s.n_nonconverged,
                    }
                    for s in res.points
                ],
            }
        )

    out_npz = out_prefix.with_suffix(".npz")
    out_json = out_prefix.with_suffix(".json")
    np.savez(out_npz, **npz_payload)
    out_json.write_text(json.dumps(json_payload, indent=2))
    print(f"wrote {out_npz}")
    print(f"wrote {out_json}")


# ---------------------------------------------------------------- main


def _parse_ylim(spec: str) -> Optional[Tuple[float, float]]:
    if spec == "auto":
        return None
    parts = spec.split(",")
    if len(parts) != 2:
        raise SystemExit(f"--ylim must be 'auto' or 'low,high'; got {spec!r}")
    return float(parts[0]), float(parts[1])


def _parse_figsize(spec: str) -> Tuple[float, float]:
    parts = spec.split(",")
    if len(parts) != 2:
        raise SystemExit(f"--figsize must be 'W,H'; got {spec!r}")
    return float(parts[0]), float(parts[1])


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    series = assemble_series(args)
    results = build_results(series, min_rows=args.min_rows, exclude_trivial=args.exclude_trivial)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    plot_iter_vs_N(
        results,
        out_prefix,
        reduce=args.reduce,
        show_band=not args.no_band,
        xscale=args.xscale,
        yscale=args.yscale,
        ylim=_parse_ylim(args.ylim),
        ref_line=args.ref_line,
        figsize=_parse_figsize(args.figsize),
        dpi=args.dpi,
    )
    save_artifacts(results, out_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
