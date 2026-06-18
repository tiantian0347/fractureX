#!/usr/bin/env python
"""Bare-VTK fallback for make_crack_figures_paraview.py (identical output).

Use when ParaView/pvpython is not installed. Renders the same damage figures
via the `vtk` python module (offscreen). Shares RUNS / STEPS / COLORMAP with
the pvpython script so the two produce matching PNGs.

    python scripts/paper_huzhang/make_crack_figures_vtk.py
    python scripts/paper_huzhang/make_crack_figures_vtk.py --models model0 --warp 0
"""
import argparse
import glob
import os

import vtk

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VTK_ROOT = os.path.join(REPO, "results", "phasefield")
OUT_DIR = os.path.join(REPO, "docs", "figures", "crack")

RUNS = {
    "model0": "model0_circular_notch/paper_aux_h2/epsg_1e-06",
    "model1": "square_tension_precrack/paper_direct/epsg_1e-06",
    "model2": "model2_notch_x_stretch/paper_direct_full/epsg_1e-06",
}
STEPS = {
    "model0": [0, 7, 15, 30],
    "model1": [0, 48, 56, 64],
    "model2": [51, 123, 159, 200],
}
FIELD = "damage"
WARP_DEFAULT = 0.0
CLIP_DEFAULT = 0.5          # crack band = d>=0.5 (isolates localized crack)
IMG_SIZE = (900, 820)
# (scalar, r, g, b) control points: blue intact -> white -> red cracked
COLORMAP = [(0.0, 0.23, 0.30, 0.75),
            (0.5, 0.95, 0.95, 0.95),
            (1.0, 0.70, 0.02, 0.15)]


def _vtu_for_step(run_rel, step):
    vtk_dir = os.path.join(VTK_ROOT, run_rel, "vtk")
    hits = glob.glob(os.path.join(vtk_dir, "step_%04d_load_*.vtu" % step))
    return sorted(hits)[0] if hits else None


def _lut():
    f = vtk.vtkColorTransferFunction()
    for s, r, g, b in COLORMAP:
        f.AddRGBPoint(s, r, g, b)
    f.SetColorSpaceToRGB()
    return f


def _max_damage(grid):
    a = grid.GetPointData().GetArray(FIELD)
    return max(a.GetValue(i) for i in range(a.GetNumberOfTuples()))


def render_one(vtu, lut, warp_factor, clip):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu)
    reader.Update()
    grid = reader.GetOutput()
    maxd = _max_damage(grid)

    src_port = reader.GetOutputPort()
    if warp_factor and warp_factor > 0:
        grid.GetPointData().SetActiveVectors("uh")
        warp = vtk.vtkWarpVector()
        warp.SetInputConnection(src_port)
        warp.SetScaleFactor(warp_factor)
        warp.Update()
        src_port = warp.GetOutputPort()

    # The intact bulk carries a diffuse AT2 damage field (d up to ~0.3) that is
    # physically real but clutters the figure. clip>0 keeps only cells whose
    # damage exceeds the threshold, so only the localized crack band (d->1) is
    # drawn over a clean (white) background; the bulk is shown undamaged.
    if clip and clip > 0:
        thr = vtk.vtkThreshold()
        thr.SetInputConnection(src_port)
        thr.SetInputArrayToProcess(0, 0, 0,
                                   vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, FIELD)
        # keep cells with damage in [clip, +inf); BETWEEN with a high upper is
        # used because THRESHOLD_UPPER's keep-direction is version-dependent.
        thr.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
        thr.SetLowerThreshold(clip)
        thr.SetUpperThreshold(1.0e30)
        # keep a cell if ANY vertex exceeds the threshold (default requires ALL,
        # which drops the whole crack band since its edge cells straddle d=clip)
        thr.SetAllScalars(0)
        thr.Update()
        src_port = thr.GetOutputPort()

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(src_port)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(FIELD)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(0.0, 1.0)
    mapper.InterpolateScalarsBeforeMappingOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.AddActor(actor)

    # When clipping, draw the specimen outline (incl. the inner hole) so the
    # geometry is still legible behind the isolated crack band.
    if clip and clip > 0:
        ge = vtk.vtkGeometryFilter()
        ge.SetInputConnection(reader.GetOutputPort())
        fe = vtk.vtkFeatureEdges()
        fe.SetInputConnection(ge.GetOutputPort())
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ManifoldEdgesOff()
        em = vtk.vtkPolyDataMapper()
        em.SetInputConnection(fe.GetOutputPort())
        em.ScalarVisibilityOff()
        ea = vtk.vtkActor()
        ea.SetMapper(em)
        ea.GetProperty().SetColor(0.2, 0.2, 0.2)
        ea.GetProperty().SetLineWidth(1.5)
        ren.AddActor(ea)

    sbar = vtk.vtkScalarBarActor()
    sbar.SetLookupTable(lut)
    sbar.SetTitle("damage")
    sbar.SetNumberOfLabels(5)
    sbar.GetTitleTextProperty().SetColor(0, 0, 0)
    sbar.GetLabelTextProperty().SetColor(0, 0, 0)
    sbar.SetMaximumWidthInPixels(80)
    ren.AddActor2D(sbar)

    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    ren.ResetCamera()

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(*IMG_SIZE)
    rw.AddRenderer(ren)
    rw.Render()
    return rw, maxd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--warp", type=float, default=WARP_DEFAULT)
    ap.add_argument("--clip", type=float, default=None,
                    help="damage threshold: keep only d>=clip (isolates crack "
                         "band over clean background). Omit for full field.")
    ap.add_argument("--both", action="store_true",
                    help="emit both full field (<model>_step####.png) and "
                         "clipped (<model>_step####_crack.png) versions")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    lut = _lut()
    # (suffix, clip-threshold) variants to render
    if args.both:
        variants = [("", 0.0), ("_crack", CLIP_DEFAULT)]
    elif args.clip is not None:
        variants = [("_crack", args.clip)]
    else:
        variants = [("", 0.0)]

    written = []
    for model in args.models:
        for step in STEPS[model]:
            vtu = _vtu_for_step(RUNS[model], step)
            if vtu is None:
                print("  [skip] %s step %d: no VTU" % (model, step))
                continue
            for suffix, clip in variants:
                rw, maxd = render_one(vtu, lut, args.warp, clip)
                w2i = vtk.vtkWindowToImageFilter()
                w2i.SetInput(rw)
                w2i.Update()
                out = os.path.join(OUT_DIR,
                                   "%s_step%04d%s.png" % (model, step, suffix))
                w = vtk.vtkPNGWriter()
                w.SetFileName(out)
                w.SetInputConnection(w2i.GetOutputPort())
                w.Write()
                written.append(out)
                print("  [ok] %s  (maxd=%.3f, clip=%.2f)" % (out, maxd, clip))
    print("\nwrote %d PNG(s) to %s" % (len(written), OUT_DIR))


if __name__ == "__main__":
    main()
