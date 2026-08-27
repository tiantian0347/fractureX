"""Unit tests for fracturex.postprocess.vtu_animation.

Uses in-memory :class:`VtuMesh` frames (no vtk). GIF write needs Pillow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fracturex.postprocess.vtu_animation import (
    _should_draw_mesh,
    list_vtu_sequence,
    write_field_animation,
)
from fracturex.postprocess.vtu_plot import VtuMesh, guess_scalar_field


def _one_tri(damage: np.ndarray, stem: str | None = None) -> VtuMesh:
    mesh = VtuMesh(
        xy=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        n_cells=1,
        point_data={"damage": np.asarray(damage, dtype=np.float64)},
    )
    if stem:
        mesh.path = Path(f"{stem}.vtu")
    return mesh


def test_list_vtu_sequence_sorts_by_step_index(tmp_path: Path) -> None:
    for name in ("step_10.vtu", "step_2.vtu", "step_1.vtu", "notes.txt"):
        (tmp_path / name).write_text("x")
    names = [p.name for p in list_vtu_sequence(tmp_path)]
    assert names == ["step_1.vtu", "step_2.vtu", "step_10.vtu"]


def test_list_vtu_sequence_stride_and_slice(tmp_path: Path) -> None:
    for i in range(6):
        (tmp_path / f"step_{i:03d}.vtu").write_text("x")
    names = [p.name for p in list_vtu_sequence(tmp_path, stride=2, start=1, stop=5)]
    assert names == ["step_001.vtu", "step_003.vtu"]


def test_should_draw_mesh_every_n_includes_last() -> None:
    assert _should_draw_mesh(0, 5, 0) is False
    assert all(_should_draw_mesh(i, 4, 1) for i in range(4))
    assert _should_draw_mesh(0, 5, 2) is True
    assert _should_draw_mesh(1, 5, 2) is False
    assert _should_draw_mesh(4, 5, 2) is True


def test_guess_scalar_field_prefers_damage() -> None:
    mesh = VtuMesh(
        xy=np.zeros((3, 2)),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        n_cells=1,
        point_data={"uh": np.zeros((3, 2)), "damage": np.zeros(3)},
    )
    assert guess_scalar_field(mesh) == "damage"
    assert guess_scalar_field(mesh, "damage") == "damage"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("PIL") is None,
    reason="Pillow not installed",
)
def test_write_gif_from_synthetic_meshes(tmp_path: Path) -> None:
    frames = [
        _one_tri(np.array([0.0, 0.2 * k, 0.4 * k]), stem=f"step_{k:03d}")
        for k in range(3)
    ]
    out = tmp_path / "phase.gif"
    path = write_field_animation(frames, out, field="damage", fps=4, dpi=50, mesh_every=2)
    assert path.is_file()
    assert path.stat().st_size > 0
