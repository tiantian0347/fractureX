"""Animate nodal damage stored in ``.npz`` restart / checkpoint files.

Boundary: 2D triangles. Does not reconstruct a true load-step history from a
**single** MainSolve restart (that file only has the final ``d``). One file
is shown as a ramp ``α d_final`` (slides only). A directory of npz files
with matching ``d`` is a real sequence.

Mesh comes from ``node``+``cell`` in the npz when present; otherwise
``--case model5`` rebuilds the Ambati beam (same as the h=0.015 stills).

Library:
    from fracturex.postprocess.npz_animation import (
        load_nodal_damage,
        meshes_from_npz_paths,
        meshes_from_damage_ramp,
    )

CLI:
    python -m fracturex.postprocess.npz_animation \\
        --npz model5_std_state.npz --case model5 --mesh-size 0.015 \\
        --ramp-frames 32 --out d.gif
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from fracturex.postprocess.vtu_animation import write_field_animation
from fracturex.postprocess.vtu_plot import VtuMesh

_DAMAGE_KEYS = ("d", "damage", "phase", "phasefield")


def load_nodal_damage(path: str | Path, field: str | None = None) -> np.ndarray:
    """Read a 1-D nodal damage array from one npz.

    Args:
        path: ``.npz`` file.
        field: array name; ``None`` tries :data:`_DAMAGE_KEYS`.

    Returns:
        ``(n_nodes,)`` float64.

    Raises:
        FileNotFoundError: path missing.
        KeyError: no usable 1-D field.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    dat = np.load(path)
    names = (field,) if field else _DAMAGE_KEYS
    for name in names:
        if name is None or name not in dat.files:
            continue
        arr = np.asarray(dat[name], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim == 1:
            return arr
    raise KeyError(
        f"{path} has no 1-D damage field among {list(names)}; keys={dat.files}"
    )


def _numeric_sort_key(path: Path) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", path.stem)]
    return (nums[-1] if nums else -1, path.name)


def list_npz_sequence(
    npz_dir: str | Path,
    *,
    glob: str = "*.npz",
    stride: int = 1,
    start: int = 0,
    stop: int | None = None,
) -> list[Path]:
    """List ``.npz`` files sorted by the last integer in the stem."""
    root = Path(npz_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    paths = sorted(root.glob(glob), key=_numeric_sort_key)
    paths = [p for p in paths if p.is_file() and p.suffix.lower() == ".npz"]
    return paths[start:stop:stride]


def mesh_from_npz_geometry(path: str | Path, d: np.ndarray) -> VtuMesh | None:
    """Build a mesh if the npz stores ``node`` and ``cell``.

    Returns:
        ``VtuMesh`` or ``None`` if geometry keys are missing.

    Raises:
        ValueError: geometry present but ``d.size`` ≠ number of nodes.
    """
    dat = np.load(path)
    if "node" not in dat.files or "cell" not in dat.files:
        return None
    xy = np.asarray(dat["node"], dtype=np.float64)[:, :2]
    cell = np.asarray(dat["cell"], dtype=np.int64)
    if d.shape != (xy.shape[0],):
        raise ValueError(f"d.shape={d.shape} != NN={xy.shape[0]} in {path}")
    return VtuMesh(
        xy=xy,
        triangles=cell,
        n_cells=int(cell.shape[0]),
        point_data={"damage": d},
        path=Path(path),
    )


def rebuild_model5_mesh(d: np.ndarray, mesh_size: float, source: Path) -> VtuMesh:
    """Rebuild the Ambati TPB gmsh mesh and attach ``d``.

    Args:
        d: nodal damage, length must equal rebuilt ``NN``.
        mesh_size: gmsh target edge length (mm), e.g. ``0.015``.
        source: npz path stored on the returned mesh.

    Raises:
        ValueError: node count mismatch.
    """
    from fracturex.cases.phase_field.model5_three_point_bending import (
        Model5StandardFEM,
    )

    model = Model5StandardFEM(mesh_size=float(mesh_size), with_geometric_notch=True)
    mesh = model.build_mesh()
    xy = np.asarray(mesh.entity("node"), dtype=np.float64)[:, :2]
    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
    if d.shape != (xy.shape[0],):
        raise ValueError(
            f"rebuilt model5 NN={xy.shape[0]} but d.shape={d.shape}; "
            "check --mesh-size against the npz run"
        )
    return VtuMesh(
        xy=xy,
        triangles=cell,
        n_cells=int(cell.shape[0]),
        point_data={"damage": np.asarray(d, dtype=np.float64)},
        path=source,
    )


def attach_damage(mesh: VtuMesh, d: np.ndarray, source: Path | None = None) -> VtuMesh:
    """Return a new :class:`VtuMesh` sharing ``xy``/``triangles``, new ``d``.

    Raises:
        ValueError: ``d`` length ≠ ``n_nodes``.
    """
    d = np.asarray(d, dtype=np.float64)
    if d.shape != (mesh.xy.shape[0],):
        raise ValueError(f"d.shape={d.shape} != NN={mesh.xy.shape[0]}")
    return VtuMesh(
        xy=mesh.xy,
        triangles=mesh.triangles,
        n_cells=mesh.n_cells,
        point_data={"damage": d},
        path=source if source is not None else mesh.path,
    )


def meshes_from_damage_ramp(mesh: VtuMesh, n_frames: int) -> list[VtuMesh]:
    """Frames ``d_k = α_k d_final`` with ``α`` equally spaced in ``[0, 1]``.

    This is **not** a load-step evolution. Use only when a single final ``d``
    is available.

    Args:
        mesh: geometry plus the final damage field named ``damage``.
        n_frames: >= 2.

    Raises:
        ValueError: ``n_frames < 2``.
    """
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    d_final = mesh.scalar("damage")
    out: list[VtuMesh] = []
    for i in range(n_frames):
        alpha = i / (n_frames - 1)
        stem = mesh.path.stem if mesh.path is not None else "d"
        out.append(
            attach_damage(
                mesh,
                alpha * d_final,
                source=Path(f"{stem}_alpha_{i:03d}.npz"),
            )
        )
    return out


def resolve_base_mesh(
    first_npz: Path,
    d0: np.ndarray,
    *,
    case: str | None,
    mesh_size: float,
) -> VtuMesh:
    """Geometry for the first snapshot: npz ``node``/``cell`` or a named case."""
    geom = mesh_from_npz_geometry(first_npz, d0)
    if geom is not None:
        return geom
    if case == "model5":
        return rebuild_model5_mesh(d0, mesh_size, first_npz)
    raise SystemExit(
        f"{first_npz} has no node/cell; pass --case model5 --mesh-size ..."
    )


def meshes_from_npz_paths(
    paths: Sequence[Path],
    *,
    case: str | None = None,
    mesh_size: float = 0.015,
    field: str | None = None,
    ramp_frames: int = 0,
) -> list[VtuMesh]:
    """Turn npz path(s) into :class:`VtuMesh` frames.

    One file + ``ramp_frames>=2`` → α-ramp of that ``d``.
    Several files → one frame per file (same mesh, each file's ``d``).
    """
    if not paths:
        raise ValueError("paths is empty")
    d0 = load_nodal_damage(paths[0], field)
    base = resolve_base_mesh(paths[0], d0, case=case, mesh_size=mesh_size)
    if len(paths) == 1:
        if ramp_frames < 2:
            raise ValueError(
                "a single npz is not a time series of d; pass --npz-dir of "
                "step_*.npz from --save-damage-dir, or --ramp-frames N only "
                "for a non-physical α·d_final fade"
            )
        return meshes_from_damage_ramp(base, ramp_frames)
    frames = [base]
    for p in paths[1:]:
        frames.append(attach_damage(base, load_nodal_damage(p, field), source=p))
    return frames


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Animate damage d from npz restart/checkpoint files."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", nargs="+", type=Path, help="one or more .npz")
    src.add_argument("--npz-dir", type=Path, help="directory of .npz")
    p.add_argument("--glob", default="*.npz")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--field", default=None, help="array name (default: d/damage/phase)")
    p.add_argument(
        "--case",
        default=None,
        choices=("model5",),
        help="rebuild mesh when npz has no node/cell",
    )
    p.add_argument("--mesh-size", type=float, default=0.015)
    p.add_argument(
        "--ramp-frames",
        type=int,
        default=0,
        help="FORBIDDEN for papers: α·d_final fade if a single npz is given. "
             "Default 0 = refuse. Use a step_*.npz directory instead.",
    )
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--mesh-every", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--xlim", nargs=2, type=float, default=None)
    p.add_argument("--ylim", nargs=2, type=float, default=None)
    p.add_argument("--cmap", default="YlOrRd")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--figsize", nargs=2, type=float, default=None)
    p.add_argument("--title-template", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Returns process exit code."""
    args = _build_parser().parse_args(argv)
    if args.npz_dir is not None:
        paths = list_npz_sequence(
            args.npz_dir,
            glob=args.glob,
            stride=args.stride,
            start=args.start,
            stop=args.stop,
        )
    else:
        paths = list(args.npz)[args.start : args.stop : args.stride]
    if not paths:
        raise SystemExit("no .npz files matched")
    frames = meshes_from_npz_paths(
        paths,
        case=args.case,
        mesh_size=args.mesh_size,
        field=args.field,
        ramp_frames=args.ramp_frames,
    )
    mesh_every = 1 if args.mesh else int(args.mesh_every)
    figsize = tuple(args.figsize) if args.figsize is not None else None
    xlim = tuple(args.xlim) if args.xlim is not None else None
    ylim = tuple(args.ylim) if args.ylim is not None else None
    title = args.title_template
    if title is None and len(paths) == 1:
        title = r"{stem}  NC={nc}  $d_\mathrm{{max}}$={dmax:.3f}  (α·d_final)"
    out = write_field_animation(
        frames,
        args.out,
        field="damage",
        mesh_every=mesh_every,
        vmin=args.vmin,
        vmax=args.vmax,
        xlim=xlim,
        ylim=ylim,
        cmap=args.cmap,
        figsize=figsize,
        dpi=args.dpi,
        fps=args.fps,
        title_template=title,
        colorbar_label=r"damage $d$",
    )
    kind = "ramp" if len(paths) == 1 else "sequence"
    print(f"wrote {out}  ({len(frames)} frames, {kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
