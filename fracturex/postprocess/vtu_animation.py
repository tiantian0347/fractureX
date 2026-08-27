"""Animate a sequence of 2D VTU nodal fields (phase-field damage).

Boundary: 2D snapshots only. Does not warp by displacement, and does not
read 3D cells. Mesh overlay is off by default; use ``--mesh`` or
``--mesh-every N`` when edges help.

Each frame may have a different mesh (adaptive remeshing is OK). Frames
are loaded one VTU at a time so a long run does not stay in RAM.

Library:
    from fracturex.postprocess.vtu_animation import (
        list_vtu_sequence,
        write_field_animation,
    )

CLI:
    python -m fracturex.postprocess.vtu_animation \\
        --vtu-dir path/to/vtu --out damage.gif
    python -m fracturex.postprocess.vtu_animation \\
        --vtu-dir path/to/vtu --out damage.gif --mesh
    python -m fracturex.postprocess.vtu_animation \\
        --vtu-dir path/to/vtu --out damage.gif --mesh-every 10 --stride 2
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from fracturex.postprocess.vtu_plot import (
    VtuMesh,
    draw_mesh_scalar,
    guess_scalar_field,
    read_vtu_mesh,
)

FrameSource = Union[VtuMesh, Path, str]


def _numeric_sort_key(path: Path) -> tuple:
    """Sort by the last integer in the stem, then by name."""
    nums = [int(x) for x in re.findall(r"\d+", path.stem)]
    return (nums[-1] if nums else -1, path.name)


def list_vtu_sequence(
    vtu_dir: str | Path,
    *,
    glob: str = "*.vtu",
    stride: int = 1,
    start: int = 0,
    stop: int | None = None,
) -> list[Path]:
    """List VTU files in a directory, sorted by the step index in the name.

    Understands both ``step_032.vtu`` and ``model5_std0000000012.vtu``.

    Args:
        vtu_dir: directory to search (not recursive unless ``glob`` contains ``**/``).
        glob: glob relative to ``vtu_dir`` (default ``*.vtu``).
        stride: keep every ``stride``-th file after sort (must be >= 1).
        start, stop: slice the sorted list, Python semantics.

    Returns:
        Sorted paths, possibly empty.

    Raises:
        FileNotFoundError: ``vtu_dir`` is not a directory.
        ValueError: ``stride < 1``.
    """
    root = Path(vtu_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    paths = sorted(root.glob(glob), key=_numeric_sort_key)
    paths = [p for p in paths if p.is_file() and p.suffix.lower() == ".vtu"]
    sliced = paths[start:stop:stride]
    return sliced


def _should_draw_mesh(index: int, n_frames: int, mesh_every: int) -> bool:
    """True if frame ``index`` (0-based) should overlay element edges.

    ``mesh_every <= 0`` never; ``1`` always; ``N>1`` every N-th frame and the last.
    """
    if mesh_every <= 0 or n_frames <= 0:
        return False
    if mesh_every == 1:
        return True
    return index % mesh_every == 0 or index == n_frames - 1


def _figsize_from_mesh(xy: np.ndarray, figsize: tuple[float, float] | None) -> tuple[float, float]:
    if figsize is not None:
        return figsize
    dx = float(xy[:, 0].max() - xy[:, 0].min())
    dy = float(xy[:, 1].max() - xy[:, 1].min())
    aspect = dx / max(dy, 1e-12)
    width = 6.4
    height = float(np.clip(width / aspect + 0.9, 2.6, 7.0))
    return (width, height)


def _title_for_frame(mesh: VtuMesh, field: str, template: str | None, index: int) -> str:
    stem = mesh.path.stem if mesh.path is not None else f"frame_{index:04d}"
    dmax = float(mesh.scalar(field).max())
    ctx = {
        "stem": stem,
        "nc": mesh.n_cells,
        "index": index,
        "field": field,
        "dmax": dmax,
    }
    if template:
        return template.format(**ctx)
    return rf"{stem}  NC={mesh.n_cells}  $d_\mathrm{{max}}$={dmax:.3f}"


def _frame_to_rgb(
    mesh: VtuMesh,
    field: str,
    *,
    show_mesh: bool,
    vmin: float | None,
    vmax: float | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    cmap: str,
    mesh_lw: float,
    figsize: tuple[float, float],
    dpi: int,
    title: str,
    colorbar_label: str,
) -> np.ndarray:
    """Render one frame to RGB uint8, shape ``(H, W, 3)``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    tpc = draw_mesh_scalar(
        ax,
        mesh,
        field,
        show_mesh=show_mesh,
        vmin=vmin,
        vmax=vmax,
        xlim=xlim,
        ylim=ylim,
        cmap=cmap,
        mesh_lw=mesh_lw,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    cbar = fig.colorbar(tpc, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.array(rgba[:, :, :3], copy=True)
    plt.close(fig)
    return rgb


def _write_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    from PIL import Image

    if not frames:
        raise ValueError("no frames to write")
    images = [Image.fromarray(fr) for fr in frames]
    duration_ms = max(1, int(round(1000.0 / float(fps))))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def _write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "MP4 output needs imageio (pip install imageio imageio-ffmpeg). "
            "Use --out something.gif otherwise."
        ) from exc
    imageio.mimsave(path, frames, fps=float(fps))


def write_field_animation(
    frames: Sequence[FrameSource],
    out: str | Path,
    *,
    field: str | None = None,
    mesh_every: int = 0,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    mesh_lw: float = 0.18,
    figsize: tuple[float, float] | None = None,
    dpi: int = 120,
    fps: float = 8.0,
    title_template: str | None = None,
    colorbar_label: str | None = None,
) -> Path:
    """Write a GIF or MP4 of a nodal scalar over a list of snapshots.

    Args:
        frames: ``VtuMesh`` objects or paths to ``.vtu`` files (loaded one by one).
        out: ``.gif`` or ``.mp4``.
        field: point-data name; ``None`` uses :func:`guess_scalar_field`.
        mesh_every: 0 = no mesh, 1 = mesh on every frame, N = every N-th + last.
        vmin, vmax: color limits (phase field usually 0 and 1).
        xlim, ylim: shared axis limits; default is the first frame's bbox.
        cmap, mesh_lw, figsize, dpi, fps: figure / movie settings.
        title_template: format string with ``stem``, ``nc``, ``index``,
            ``field``, ``dmax``. ``None`` uses a default title.
        colorbar_label: colorbar text; default is the field name.

    Returns:
        Resolved output path.

    Raises:
        ValueError: empty ``frames``, or unknown output suffix.
        ImportError: GIF needs Pillow; MP4 needs imageio.
    """
    if not frames:
        raise ValueError("frames is empty")
    out_path = Path(out)
    suffix = out_path.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError(f"out suffix must be .gif or .mp4, got {out_path.suffix!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(frames)
    rgb_frames: list[np.ndarray] = []
    field_name: str | None = None
    used_figsize = figsize
    used_xlim, used_ylim = xlim, ylim
    label = colorbar_label

    for i, src in enumerate(frames):
        mesh = src if isinstance(src, VtuMesh) else read_vtu_mesh(src)
        name = guess_scalar_field(mesh, field)
        if field_name is None:
            field_name = name
            if used_figsize is None:
                used_figsize = _figsize_from_mesh(mesh.xy, None)
            if used_xlim is None:
                used_xlim = (float(mesh.xy[:, 0].min()), float(mesh.xy[:, 0].max()))
            if used_ylim is None:
                used_ylim = (float(mesh.xy[:, 1].min()), float(mesh.xy[:, 1].max()))
            if label is None:
                label = name
        rgb_frames.append(
            _frame_to_rgb(
                mesh,
                name,
                show_mesh=_should_draw_mesh(i, n, mesh_every),
                vmin=vmin,
                vmax=vmax,
                xlim=used_xlim,
                ylim=used_ylim,
                cmap=cmap,
                mesh_lw=mesh_lw,
                figsize=used_figsize,
                dpi=dpi,
                title=_title_for_frame(mesh, name, title_template, i),
                colorbar_label=label or name,
            )
        )

    if suffix == ".gif":
        _write_gif(rgb_frames, out_path, fps)
    else:
        _write_mp4(rgb_frames, out_path, fps)
    return out_path.resolve()


def _parse_xy_lim(values: Sequence[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("xlim/ylim need exactly two numbers")
    return (float(values[0]), float(values[1]))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Animate a VTU sequence of a nodal phase-field / damage."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--vtu-dir", type=Path, help="directory of .vtu snapshots")
    src.add_argument("--vtu", nargs="+", type=Path, help="explicit .vtu file list")
    p.add_argument("--glob", default="*.vtu", help="glob under --vtu-dir")
    p.add_argument("--out", required=True, type=Path, help="output .gif or .mp4")
    p.add_argument(
        "--field",
        default=None,
        help="point-data name (default: guess damage/d/phase/...)",
    )
    p.add_argument(
        "--mesh",
        action="store_true",
        help="overlay element edges on every frame",
    )
    p.add_argument(
        "--mesh-every",
        type=int,
        default=0,
        metavar="N",
        help="overlay mesh every N frames and on the last frame (ignored if --mesh)",
    )
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start", type=int, default=0, help="first index after sort")
    p.add_argument("--stop", type=int, default=None, help="exclusive end index")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("X0", "X1"))
    p.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("Y0", "Y1"))
    p.add_argument("--cmap", default="YlOrRd")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--mesh-lw", type=float, default=0.18)
    p.add_argument("--title-template", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Returns process exit code."""
    args = _build_parser().parse_args(argv)
    if args.vtu_dir is not None:
        paths = list_vtu_sequence(
            args.vtu_dir,
            glob=args.glob,
            stride=args.stride,
            start=args.start,
            stop=args.stop,
        )
    else:
        paths = list(args.vtu)[args.start : args.stop : args.stride]
    if not paths:
        raise SystemExit("no .vtu files matched")
    mesh_every = 1 if args.mesh else int(args.mesh_every)
    out = write_field_animation(
        paths,
        args.out,
        field=args.field,
        mesh_every=mesh_every,
        vmin=args.vmin,
        vmax=args.vmax,
        xlim=_parse_xy_lim(args.xlim),
        ylim=_parse_xy_lim(args.ylim),
        cmap=args.cmap,
        mesh_lw=args.mesh_lw,
        dpi=args.dpi,
        fps=args.fps,
        title_template=args.title_template,
    )
    print(f"wrote {out}  ({len(paths)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
