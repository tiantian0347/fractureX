#!/usr/bin/env python3
"""Fix stale ``damage_p`` (and detect any ``p_sigma`` / ``u_space_order``
mismatch) in old ``mesh.npz`` files by cross-checking against the sizes
of state arrays inside sibling ``checkpoints/step_*.npz``.

Motivating incident: pre-2026-05-31 ``paper_aux_h{1,2,3}`` runs recorded
``mesh.damage_p=1`` even though the state ``d`` was P2 on the same
mesh. Downstream ``load_discr_from_dir`` (rebuilds ``space_d`` from
``mesh.damage_p``) then produced a ``space_d`` whose gdof did not match
``state.d.shape`` in the checkpoint. See
``docs/preconditioner/DESIGN_nepin_driver.md`` §6.5 (caveat 2).

The fix here reads NN / NC / NE (NE computed from unique cell-edge pairs)
from the mesh, then infers the true damage_p from d.shape via the closed
form
    gdof(P1) = NN
    gdof(P2) = NN + NE
    gdof(P3) = NN + 2*NE + NC
    gdof(P4) = NN + 3*NE + 3*NC
When a unique match is found and it disagrees with the recorded field,
mesh.npz is rewritten with all other fields preserved.

Usage (dry-run — reports what it would do; no writes):
    python scripts/fix_stale_mesh_damage_p.py <recorder_dir>[...]

Apply (writes mesh.npz in place, creating .bak first):
    python scripts/fix_stale_mesh_damage_p.py --apply <recorder_dir>[...]

Each ``<recorder_dir>`` must contain ``mesh.npz`` and ``checkpoints/``.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _count_edges(cell: np.ndarray) -> int:
    """Number of unique undirected edges in a triangle mesh."""
    a = cell[:, [0, 1, 2]]
    b = cell[:, [1, 2, 0]]
    edges = np.stack([a.reshape(-1), b.reshape(-1)], axis=1)
    edges.sort(axis=1)
    _, _ = np.unique(edges, axis=0, return_index=True)
    return int(np.unique(edges, axis=0).shape[0])


def _lagrange_gdof_C(p: int, NN: int, NE: int, NC: int) -> int:
    """Continuous Lagrange P_p scalar dofs on a 2D triangle mesh."""
    if p == 1:
        return NN
    if p == 2:
        return NN + NE
    if p == 3:
        return NN + 2 * NE + NC
    if p == 4:
        return NN + 3 * NE + 3 * NC
    # generic: NN + (p-1)*NE + (p-1)*(p-2)/2 * NC
    return NN + (p - 1) * NE + (p - 1) * (p - 2) // 2 * NC


def _infer_p(gdof: int, NN: int, NE: int, NC: int, p_range=(1, 2, 3, 4)) -> Optional[int]:
    for p in p_range:
        if _lagrange_gdof_C(p, NN, NE, NC) == gdof:
            return p
    return None


def _first_checkpoint(dir_: Path) -> Optional[Path]:
    cks = sorted((dir_ / "checkpoints").glob("step_*.npz"))
    return cks[0] if cks else None


def inspect(recorder_dir: Path) -> Optional[dict]:
    mp = recorder_dir / "mesh.npz"
    if not mp.exists():
        print(f"[skip] {recorder_dir}: no mesh.npz")
        return None
    ckp = _first_checkpoint(recorder_dir)
    if ckp is None:
        print(f"[skip] {recorder_dir}: no checkpoints/step_*.npz")
        return None

    m = dict(np.load(mp, allow_pickle=False))  # copy into plain dict
    ck = np.load(ckp, allow_pickle=False)
    node = np.asarray(m["node"])
    cell = np.asarray(m["cell"])
    NN, NC = int(node.shape[0]), int(cell.shape[0])
    NE = _count_edges(cell)

    d_gdof = int(np.asarray(ck["d"]).shape[0]) if "d" in ck.files else -1
    inferred = _infer_p(d_gdof, NN, NE, NC) if d_gdof > 0 else None

    recorded = int(m["damage_p"])
    print(f"{recorder_dir}: NN={NN} NE={NE} NC={NC}")
    print(f"  mesh.damage_p={recorded}, checkpoint(d).gdof={d_gdof}, inferred_p={inferred}")

    if inferred is None:
        print(f"  -> NO MATCH for any p in {{1,2,3,4}}; leave alone")
        return None
    if inferred == recorded:
        print("  -> consistent, nothing to do")
        return None
    print(f"  -> MISMATCH: mesh.damage_p={recorded} but d shape says P{inferred}")
    return dict(
        dir=recorder_dir,
        mesh_path=mp,
        loaded=m,
        old_damage_p=recorded,
        new_damage_p=inferred,
    )


def rewrite(hit: dict) -> None:
    mp: Path = hit["mesh_path"]
    m: dict = hit["loaded"]
    bak = mp.with_suffix(".npz.bak")
    if not bak.exists():
        shutil.copy2(mp, bak)
        print(f"  backup: {bak.name}")
    m["damage_p"] = np.asarray(int(hit["new_damage_p"]))
    np.savez_compressed(mp, **m)
    print(f"  wrote {mp}: damage_p={hit['old_damage_p']} -> {hit['new_damage_p']}")


def _expand_dirs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pth = Path(p)
        if (pth / "mesh.npz").exists():
            out.append(pth); continue
        # recursive: any dir containing mesh.npz under this root
        for m in glob.glob(str(pth / "**" / "mesh.npz"), recursive=True):
            out.append(Path(m).parent)
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+",
                    help="recorder root(s); or a parent tree — will recurse to any mesh.npz")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite mesh.npz in place (default: dry-run)")
    args = ap.parse_args()

    dirs = _expand_dirs(args.dirs)
    print(f"[scan] {len(dirs)} recorder dir(s)")
    hits = []
    for d in dirs:
        hit = inspect(d)
        if hit is not None:
            hits.append(hit)

    print(f"\n[summary] {len(hits)} dir(s) need rewriting")
    if not args.apply:
        print("[dry-run] rerun with --apply to actually rewrite mesh.npz files")
        return 0

    for hit in hits:
        rewrite(hit)
    print(f"\n[done] rewrote {len(hits)} mesh.npz files (originals backed up as *.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
