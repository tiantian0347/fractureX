"""Unit tests for fracturex.postprocess.npz_animation.

Uses a 1-triangle mesh (no gmsh, no vtk). GIF write needs Pillow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fracturex.postprocess.npz_animation import (
    attach_damage,
    list_npz_sequence,
    load_nodal_damage,
    meshes_from_damage_ramp,
    meshes_from_npz_paths,
)
from fracturex.postprocess.vtu_plot import VtuMesh


def _tri(d: np.ndarray) -> VtuMesh:
    return VtuMesh(
        xy=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        n_cells=1,
        point_data={"damage": np.asarray(d, dtype=np.float64)},
        path=Path("final.npz"),
    )


def test_load_nodal_damage_reads_d_key(tmp_path: Path) -> None:
    p = tmp_path / "s.npz"
    np.savez(p, d=np.array([0.0, 0.5, 1.0]), uh=np.zeros(6))
    np.testing.assert_allclose(load_nodal_damage(p), [0.0, 0.5, 1.0])


def test_list_npz_sequence_numeric_sort(tmp_path: Path) -> None:
    for name in ("step_10.npz", "step_2.npz", "mesh.npz"):
        np.savez(tmp_path / name, d=np.zeros(3))
    names = [p.name for p in list_npz_sequence(tmp_path)]
    assert names == ["mesh.npz", "step_2.npz", "step_10.npz"]


def test_damage_ramp_scales_final_field() -> None:
    frames = meshes_from_damage_ramp(_tri(np.array([0.0, 0.5, 1.0])), 5)
    assert len(frames) == 5
    np.testing.assert_allclose(frames[0].scalar("damage"), 0.0)
    np.testing.assert_allclose(frames[-1].scalar("damage"), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(frames[2].scalar("damage"), [0.0, 0.25, 0.5])


def test_single_npz_without_ramp_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "final.npz"
    np.savez(
        p,
        node=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cell=np.array([[0, 1, 2]], dtype=np.int64),
        d=np.array([0.2, 0.4, 1.0]),
    )
    with pytest.raises(ValueError, match="not a time series"):
        meshes_from_npz_paths([p], ramp_frames=0)


def test_meshes_from_npz_sequence_with_embedded_geometry(tmp_path: Path) -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cell = np.array([[0, 1, 2]], dtype=np.int64)
    paths = []
    for i, scale in enumerate((0.0, 0.5, 1.0)):
        p = tmp_path / f"step_{i:03d}.npz"
        np.savez(p, node=xy, cell=cell, d=np.array([0.0, 0.5, 1.0]) * scale)
        paths.append(p)
    frames = meshes_from_npz_paths(paths)
    assert len(frames) == 3
    np.testing.assert_allclose(frames[1].scalar("damage"), [0.0, 0.25, 0.5])


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("PIL") is None,
    reason="Pillow not installed",
)
def test_ramp_gif_from_npz_with_geometry(tmp_path: Path) -> None:
    from fracturex.postprocess.vtu_animation import write_field_animation

    p = tmp_path / "final.npz"
    np.savez(
        p,
        node=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cell=np.array([[0, 1, 2]], dtype=np.int64),
        d=np.array([0.2, 0.4, 1.0]),
    )
    frames = meshes_from_npz_paths([p], ramp_frames=4)
    out = tmp_path / "d.gif"
    write_field_animation(frames, out, field="damage", fps=4, dpi=40)
    assert out.stat().st_size > 0


def test_attach_damage_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="d.shape"):
        attach_damage(_tri(np.zeros(3)), np.zeros(4))
