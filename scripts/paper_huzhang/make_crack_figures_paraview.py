#!/usr/bin/env pvpython
"""Render crack (damage) figures for model0/1/2 from phase-field VTU output.

Usage (in a ParaView install that ships pvpython):
    pvpython scripts/paper_huzhang/make_crack_figures_paraview.py
    pvpython scripts/paper_huzhang/make_crack_figures_paraview.py --models model0 model2
    pvpython scripts/paper_huzhang/make_crack_figures_paraview.py --warp 0.0   # no displacement warp

Input  : results/phasefield/<case>/<run>/epsg_1e-06/vtk/step_*.vtu
         (point field 'damage' in [0,1], displacement field 'uh').
Output : docs/figures/crack/<model>_step####_d<maxd>.png  (one PNG per selected step)

The PNG content is identical to make_crack_figures_vtk.py (same RUNS / STEPS / COLORMAP);
this file is the ParaView pvpython front-end, that one is the bare-VTK fallback used when
ParaView is not installed.
"""
import argparse
import os
import sys

from paraview.simple import (  # type: ignore
    ColorBy,
    GetActiveViewOrCreate,
    GetColorTransferFunction,
    GetScalarBar,
    Hide,
    SaveScreenshot,
    Show,
    Threshold,
    Warp,
    XMLUnstructuredGridReader,
    Delete,
)

# ---- shared config (keep in sync with make_crack_figures_vtk.py) -----------
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VTK_ROOT = os.path.join(REPO, "results", "phasefield")
OUT_DIR = os.path.join(REPO, "docs", "figures", "crack")

# run dir (relative to VTK_ROOT) holding the vtk/ subfolder, per model
RUNS = {
    "model0": "model0_circular_notch/paper_aux_h2/epsg_1e-06",
    "model1": "square_tension_precrack/paper_direct/epsg_1e-06",
    "model2": "model2_notch_x_stretch/paper_direct_full/epsg_1e-06",
}
# representative step ids (initiation -> propagation -> through-crack)
STEPS = {
    "model0": [0, 7, 15, 30],
    "model1": [0, 48, 56, 64],
    "model2": [51, 123, 159, 200],
}
FIELD = "damage"
WARP_DEFAULT = 0.0          # displacement magnification (0 = undeformed mesh)
CLIP_DEFAULT = 0.5          # crack band = d>=0.5 (isolates localized crack)
IMG_SIZE = [900, 820]
# blue (intact, d=0) -> red (cracked, d=1)
COLORMAP = [0.0, 0.23, 0.30, 0.75,
            0.5, 0.95, 0.95, 0.95,
            1.0, 0.70, 0.02, 0.15]


def _vtu_for_step(run_rel, step):
    vtk_dir = os.path.join(VTK_ROOT, run_rel, "vtk")
    if not os.path.isdir(vtk_dir):
        return None
    pref = "step_%04d_load_" % step
    for fn in sorted(os.listdir(vtk_dir)):
        if fn.startswith(pref) and fn.endswith(".vtu"):
            return os.path.join(vtk_dir, fn)
    return None


def render(models, warp_factor, variants):
    os.makedirs(OUT_DIR, exist_ok=True)
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = IMG_SIZE
    view.OrientationAxesVisibility = 0
    view.Background = [1.0, 1.0, 1.0]
    view.CameraParallelProjection = 1

    written = []
    for model in models:
        run_rel = RUNS[model]
        for step in STEPS[model]:
            vtu = _vtu_for_step(run_rel, step)
            if vtu is None:
                print("  [skip] %s step %d: no VTU" % (model, step))
                continue

            for suffix, clip in variants:
                reader = XMLUnstructuredGridReader(FileName=[vtu])
                reader.PointArrayStatus = [FIELD, "uh"]
                src = reader
                if warp_factor and warp_factor > 0:
                    w = Warp(Input=src)
                    w.Vectors = ["POINTS", "uh"]
                    w.ScaleFactor = warp_factor
                    src = w
                # keep only the crack band (d>=clip) over a clean background
                if clip and clip > 0:
                    th = Threshold(Input=src)
                    th.Scalars = ["POINTS", FIELD]
                    th.LowerThreshold = clip
                    th.UpperThreshold = 1.0e30
                    th.AllScalars = 0
                    src = th

                disp = Show(src, view)
                disp.Representation = "Surface"
                ColorBy(disp, ("POINTS", FIELD))
                lut = GetColorTransferFunction(FIELD)
                lut.RGBPoints = COLORMAP
                lut.RescaleTransferFunction(0.0, 1.0)

                sb = GetScalarBar(lut, view)
                sb.Title = "damage"
                sb.ComponentTitle = ""
                sb.TitleColor = [0, 0, 0]
                sb.LabelColor = [0, 0, 0]
                disp.SetScalarBarVisibility(view, True)

                view.ResetCamera()
                out = os.path.join(OUT_DIR,
                                   "%s_step%04d%s.png" % (model, step, suffix))
                SaveScreenshot(out, view, ImageResolution=IMG_SIZE,
                               TransparentBackground=0)
                written.append(out)
                print("  [ok] %s" % out)

                Hide(src, view)
                Delete(src)
                if src is not reader:
                    Delete(reader)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(RUNS),
                    choices=list(RUNS))
    ap.add_argument("--warp", type=float, default=WARP_DEFAULT,
                    help="displacement warp factor (0 = undeformed)")
    ap.add_argument("--clip", type=float, default=None,
                    help="damage threshold: keep only d>=clip (isolates crack "
                         "band). Omit for full field.")
    ap.add_argument("--both", action="store_true",
                    help="emit both full field (<model>_step####.png) and "
                         "clipped (<model>_step####_crack.png) versions")
    args = ap.parse_args()
    if args.both:
        variants = [("", 0.0), ("_crack", CLIP_DEFAULT)]
    elif args.clip is not None:
        variants = [("_crack", args.clip)]
    else:
        variants = [("", 0.0)]
    w = render(args.models, args.warp, variants)
    print("\nwrote %d PNG(s) to %s" % (len(w), OUT_DIR))


if __name__ == "__main__":
    main()
