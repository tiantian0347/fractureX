#!/usr/bin/env python3
# make_v5_stress_slice.py
#
# Paper section 7.6 (V5): stress profile along a horizontal slice through the
# crack tip, comparing the Hu-Zhang stress against a displacement-based Lagrange
# (post-processed) stress at a comparable DOF budget.
#
# This is the POST-PROCESSOR. It samples a chosen stress component along the
# line y=y0 (default the crack plane y=0.5) from one or more VTU files and
# overlays the profiles. It works directly on:
#   * Hu-Zhang exported VTU (has nodal stress fields: sigxx, sigxy, sig2, svm),
#   * Lagrange phase-field VTU produced by run_v5_lagrange.py with --export-stress
#     (which adds the degraded recovered stress as point_data).
#
# Sampling uses matplotlib's LinearTriInterpolator on the (triangulated) mesh,
# so it works on the unstructured / structured triangle meshes used here.
#
# Usage (Hu-Zhang side, available now):
#   python make_v5_stress_slice.py --field svm --y0 0.5 \
#       --label "Hu-Zhang p=3" \
#         results/phasefield/square_tension_precrack/paper_direct/epsg_1e-06/vtk/step_0048_load_4.800000e-03.vtu
#   add more "--label NAME file.vtu" groups to overlay Lagrange once available.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista as pv

OUT = Path("/home/gongshihua/tian/Frac_huzhang/figures")
OUT.mkdir(parents=True, exist_ok=True)

# accepted aliases -> the field name actually present in the VTU
FIELD_ALIASES = {
    "svm": ["svm", "sigma_vm", "von_mises"],
    "syy": ["sig2", "syy", "sigma_yy"],
    "sxx": ["sigxx", "sxx", "sigma_xx"],
    "sxy": ["sigxy", "sxy", "sigma_xy"],
    "damage": ["damage", "d"],
}


def resolve_field(mesh, field):
    for cand in FIELD_ALIASES.get(field, [field]):
        if cand in mesh.point_data:
            return cand
    raise KeyError(f"field '{field}' not in {list(mesh.point_data.keys())}")


def _recovered_nodal_stress(m, field, lam, mu, xi=1e-6):
    """Degraded recovered stress sigma = g(d)(2 mu eps + lam tr(eps) I) from the
    nodal displacement (uh) and damage in a VTU. P1 gradient per triangle ->
    cell-constant stress -> area-averaged to nodes (so LinearTriInterpolator can
    sample it). g(d) = (1-d)^2 + xi. For the Lagrange phase-field VTU; the
    component must be one of svm/syy/sxx/sxy."""
    pts = m.points
    cells = m.cells.reshape(-1, 4)[:, 1:]
    if "uh" in m.point_data:
        uh = np.asarray(m.point_data["uh"])[:, :2]
    else:
        uh = np.stack([np.asarray(m.point_data["ux"]), np.asarray(m.point_data["uy"])], axis=1)
    d = np.asarray(m.point_data[resolve_field(m, "damage")])
    x = pts[:, 0]; y = pts[:, 1]
    i, j, k = cells[:, 0], cells[:, 1], cells[:, 2]
    # P1 shape-function gradients per triangle
    x1, y1 = x[i], y[i]; x2, y2 = x[j], y[j]; x3, y3 = x[k], y[k]
    detT = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / detT[:, None]  # dphi/dx
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / detT[:, None]  # dphi/dy
    uxc = uh[cells, 0]; uyc = uh[cells, 1]  # (NC,3)
    exx = np.sum(uxc * b, axis=1)
    eyy = np.sum(uyc * c, axis=1)
    exy = 0.5 * (np.sum(uxc * c, axis=1) + np.sum(uyc * b, axis=1))
    tr = exx + eyy
    sxx = lam * tr + 2 * mu * exx
    syy = lam * tr + 2 * mu * eyy
    sxy = 2 * mu * exy
    gd = (1.0 - d[cells].mean(axis=1)) ** 2 + xi
    sxx *= gd; syy *= gd; sxy *= gd
    if field in ("svm", "von_mises"):
        comp = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    elif field in ("syy",):
        comp = syy
    elif field in ("sxx",):
        comp = sxx
    elif field in ("sxy",):
        comp = sxy
    else:
        raise ValueError(f"--recover-lame supports svm/syy/sxx/sxy, got {field}")
    # area-average cell values to nodes
    area = np.abs(detT) / 2.0
    nv = np.zeros(len(pts)); nw = np.zeros(len(pts))
    for t in range(3):
        np.add.at(nv, cells[:, t], comp * area)
        np.add.at(nw, cells[:, t], area)
    return nv / np.maximum(nw, 1e-30)


def sample_line(vtu_path, field, y0, n=400, recover_lame=None):
    m = pv.read(vtu_path)
    pts = m.points
    cells = m.cells.reshape(-1, 4)[:, 1:]
    try:
        fname = resolve_field(m, field)
        vals = np.asarray(m.point_data[fname])
    except KeyError:
        if recover_lame is None:
            raise
        lam, mu = recover_lame
        vals = _recovered_nodal_stress(m, field, lam, mu)
        fname = f"{field} (recovered g(d)C:eps)"
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells)
    interp = mtri.LinearTriInterpolator(tri, vals)
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    xs = np.linspace(xmin, xmax, n)
    ss = interp(xs, np.full_like(xs, y0))  # masked array outside the hull
    return xs, np.ma.filled(ss, np.nan), fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="svm", help="svm|syy|sxx|sxy|damage")
    ap.add_argument("--y0", type=float, default=0.5, help="slice line y=const (crack plane)")
    ap.add_argument("--out", default="v5_stress_slice", help="output prefix under figures/")
    ap.add_argument("--curve", nargs=2, action="append", metavar=("LABEL", "VTU"),
                    default=[], help='a labelled VTU to sample; repeatable')
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--recover-lame", nargs=2, type=float, metavar=("LAMBDA", "MU"),
                    default=None, help="for VTUs lacking a stress field (e.g. the "
                    "Lagrange phase-field output), recover sigma=g(d)(2 mu eps+lam tr eps I) "
                    "from uh+damage. SEN plane-strain: --recover-lame 121.15 80.77")
    ap.add_argument("vtus", nargs="*", help="positional VTUs (paired with --label)")
    args = ap.parse_args()

    curves = list(args.curve)
    for lab, f in zip(args.label, args.vtus):
        curves.append((lab, f))
    if not curves:
        ap.error("provide at least one '--curve LABEL VTU' or '--label L file.vtu'")

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    styles = ["-", "--", "-.", ":"]
    for i, (label, vtu) in enumerate(curves):
        xs, ss, fname = sample_line(vtu, args.field, args.y0, recover_lame=args.recover_lame)
        ax.plot(xs, ss, styles[i % len(styles)], lw=1.8, label=f"{label} [{fname}]")
    ax.axvline(0.5, color="grey", ls=":", lw=0.8)
    ax.text(0.5, ax.get_ylim()[1], " crack tip", fontsize=8, va="top", color="grey")
    ax.set_xlabel("x")
    ax.set_ylabel(rf"$\sigma$ component ({args.field}) on $y={args.y0}$")
    ax.set_title("V5: crack-tip stress slice")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{args.out}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{args.out}.pdf")


if __name__ == "__main__":
    main()
