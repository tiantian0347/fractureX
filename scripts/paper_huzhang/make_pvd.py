#!/usr/bin/env python3
"""make_pvd.py — build a ParaView .pvd time-series from a fracturex vtk/ dir.

The per-step VTUs are named ``step_<idx>_load_<val>.vtu`` (two embedded numbers),
which confuses ParaView's automatic series detection. This writes an explicit
``.pvd`` whose <DataSet timestep=...> entries map each VTU to a monotone time
(the prescribed load value by default, or the step index), sorted by step.

Usage:
  python make_pvd.py <vtk_dir> [--time load|step] [--out series.pvd]

Example:
  python make_pvd.py results/.../paper_direct/epsg_1e-06/vtk
  -> writes results/.../paper_direct/epsg_1e-06/vtk/series.pvd
Open series.pvd in ParaView; it loads as one time-series ready for animation.
"""
import argparse
import glob
import os
import re
import sys
from xml.sax.saxutils import escape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk_dir")
    ap.add_argument("--time", choices=["load", "step"], default="load",
                    help="timestep value source (default: load value)")
    ap.add_argument("--out", default=None, help="output .pvd path")
    args = ap.parse_args()

    vtk_dir = os.path.abspath(args.vtk_dir)
    if not os.path.isdir(vtk_dir):
        sys.exit(f"not a dir: {vtk_dir}")

    entries = []  # (step, load, basename)
    for f in glob.glob(os.path.join(vtk_dir, "step_*.vtu")):
        m = re.search(r"step_(\d+)_load_([0-9.eE+-]+)\.vtu$", os.path.basename(f))
        if not m:
            continue
        entries.append((int(m.group(1)), float(m.group(2)), os.path.basename(f)))
    if not entries:
        sys.exit(f"no step_*.vtu found in {vtk_dir}")
    entries.sort(key=lambda t: t[0])

    out = args.out or os.path.join(vtk_dir, "series.pvd")
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '  <Collection>']
    for step, load, name in entries:
        t = load if args.time == "load" else float(step)
        lines.append(f'    <DataSet timestep="{t:.9g}" group="" part="0" '
                     f'file="{escape(name)}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(entries)} frames, "
          f"step {entries[0][0]}..{entries[-1][0]}, "
          f"{args.time} {entries[0][1] if args.time=='load' else entries[0][0]:.6g}.."
          f"{entries[-1][1] if args.time=='load' else entries[-1][0]:.6g}")


if __name__ == "__main__":
    main()
