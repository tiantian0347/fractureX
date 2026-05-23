"""Lightweight tests for phase-field mesh sizing (no full model0 solve)."""
from __future__ import annotations

import numpy as np

from fracturex.utilfuc.phasefield_mesh import (
    mesh_h_stats,
    phasefield_h_target,
    resolve_model0_distmesh_hmin,
)


class _FakeMesh:
  def __init__(self, hmax: float, nc: int = 100, nn: int = 50):
      self._hmax = float(hmax)
      self._nc = int(nc)
      self._nn = int(nn)

  def edge_length(self):
      return np.array([self._hmax * 0.9, self._hmax], dtype=float)

  def number_of_cells(self):
      return self._nc

  def number_of_nodes(self):
      return self._nn


def test_model0_l0_and_target():
    l0 = 0.02
    assert phasefield_h_target(l0, safety=0.92) == 0.0092
    assert l0 / 2 == 0.01


def test_resolve_model0_hmin_mock_refinement():
    """Simulate distmesh where h_max ~ 1.5 * hmin; resolver must shrink hmin."""
    l0 = 0.02
    target = phasefield_h_target(l0)

    def make_mesh(hmin: float):
        return _FakeMesh(hmax=1.5 * float(hmin))

    hmin, info = resolve_model0_distmesh_hmin(l0, make_mesh)
    assert info["h_ok"] is True
    assert info["h_max"] < target
    assert hmin < l0 / 2


def test_old_default_hmin_0_01_would_fail_mock():
    """Document: fixed hmin=0.01 is too coarse when h_max ~ 1.5*hmin for model0."""
    l0 = 0.02
    target = phasefield_h_target(l0)
    mesh = _FakeMesh(hmax=1.5 * 0.01)
    stats = mesh_h_stats(mesh)
    assert stats["h_max"] > target
    assert stats["h_max"] > l0 / 2
