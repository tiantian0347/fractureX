"""Unit tests for fracturex.ml.coarse_features (D13 learned coarse-space pipeline).

Covers the behavioral contracts the learned model depends on:
  - shape / finiteness of the per-node feature block;
  - feature semantics (dimensionless, local, mesh-resolution-invariant):
      * h/l0 halves under uniform refinement; d & g/g_max stay stable;
      * g/g_max in (0,1], with a global max == 1;
      * a sharp localized crack band produces a large negative log(g/g_bar)
        (the high-contrast signature the enrichment keys on) while a smooth
        field does not;
  - exact nodal evaluation for a P2 damage field (dofs != node values).

Run:
  PYTHONPATH=<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest fracturex/tests/test_coarse_features.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

from fracturex.ml.coarse_features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_coarse_features,
)

L0 = 0.02


class _QuadraticDamage:
    """Minimal damage stub: g(d) = (1-d)^2 + eps, matching the model floor."""

    def __init__(self, l0=L0, eps=1e-10):
        self.l0 = l0
        self._eps = eps

    def degradation(self, d):
        return (1.0 - np.asarray(d)) ** 2 + self._eps


class _State:
    def __init__(self, d):
        self.d = d


def _make(n, *, p=1, band=True, l0=L0, x_half=True):
    """Build a (mesh, state) with a damage field on a P``p`` space.

    band=True  -> sharp localized crack band along y=0.5 (width ~0.5*l0), x<=0.5;
    band=False -> smooth low-amplitude field (no sharp interface).
    """
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=n, ny=n)
    space = LagrangeFESpace(mesh, p=p)
    d = space.function()
    ip = space.interpolation_points()
    x = np.asarray(ip[:, 0])
    y = np.asarray(ip[:, 1])
    if band:
        vals = np.exp(-((y - 0.5) ** 2) / (2 * (0.5 * l0) ** 2))
        if x_half:
            vals = vals * (x <= 0.5)
        vals = np.clip(vals, 0.0, 0.999)
    else:
        vals = 0.05 + 0.02 * np.sin(np.pi * x) * np.cos(np.pi * y)  # smooth, small
    d[:] = bm.array(np.asarray(vals, dtype=float))
    return mesh, _State(d)


def test_shape_and_finiteness():
    mesh, st = _make(32)
    cf = extract_coarse_features(mesh, _QuadraticDamage(), st, l0=L0)
    nn = int(mesh.number_of_nodes())
    assert cf.phi.shape == (nn, N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert np.isfinite(cf.phi).all()
    assert cf.node.shape == (nn, mesh.geo_dimension())


def test_g_over_gmax_range():
    mesh, st = _make(32)
    cf = extract_coarse_features(mesh, _QuadraticDamage(), st, l0=L0)
    col = cf.phi[:, FEATURE_NAMES.index("g_over_gmax")]
    assert col.min() > 0.0
    assert col.max() == pytest.approx(1.0, abs=1e-12)
    assert col.max() <= 1.0 + 1e-12


def test_d_feature_in_unit_range():
    mesh, st = _make(32)
    cf = extract_coarse_features(mesh, _QuadraticDamage(), st, l0=L0)
    d = cf.phi[:, FEATURE_NAMES.index("d")]
    assert d.min() >= 0.0
    assert d.max() <= 1.0


def test_mesh_resolution_invariance():
    """h/l0 halves per refinement; d & g/g_max distributions stay stable."""
    j_h = FEATURE_NAMES.index("h_l0")
    j_g = FEATURE_NAMES.index("g_over_gmax")
    prev_h = None
    g_means = []
    for n in (32, 64, 128):
        mesh, st = _make(n)
        cf = extract_coarse_features(mesh, _QuadraticDamage(), st, l0=L0)
        h_mean = float(cf.phi[:, j_h].mean())
        g_means.append(float(cf.phi[:, j_g].mean()))
        if prev_h is not None:
            # uniform refinement halves the mesh size; tolerate boundary effects.
            assert h_mean == pytest.approx(prev_h / 2.0, rel=0.05)
        prev_h = h_mean
    # g/g_max mean is a dimensionless field statistic -> stable across meshes.
    assert max(g_means) - min(g_means) < 0.05


def test_localization_signature():
    """Sharp band -> strongly negative log(g/g_bar); smooth field -> near zero."""
    j = FEATURE_NAMES.index("log_g_over_gbar")
    mesh_b, st_b = _make(64, band=True)
    mesh_s, st_s = _make(64, band=False)
    cf_b = extract_coarse_features(mesh_b, _QuadraticDamage(), st_b, l0=L0)
    cf_s = extract_coarse_features(mesh_s, _QuadraticDamage(), st_s, l0=L0)
    min_band = float(cf_b.phi[:, j].min())
    min_smooth = float(cf_s.phi[:, j].min())
    # The localized high-contrast interface drives a large jump in g across a
    # node ring; the smooth field does not.
    assert min_band < -1.0
    assert min_smooth > -0.2
    assert min_band < min_smooth - 1.0


def test_gradd_dimensionless_scales_with_l0():
    """gradd_l0 = ||grad d|| * l0 scales linearly with the supplied l0."""
    j = FEATURE_NAMES.index("gradd_l0")
    mesh, st = _make(64, band=True, l0=L0)
    cf1 = extract_coarse_features(mesh, _QuadraticDamage(l0=L0), st, l0=L0)
    cf2 = extract_coarse_features(mesh, _QuadraticDamage(l0=2 * L0), st, l0=2 * L0)
    m1 = float(cf1.phi[:, j].max())
    m2 = float(cf2.phi[:, j].max())
    assert m2 == pytest.approx(2.0 * m1, rel=1e-6)


def test_p2_damage_nodal_eval():
    """P2 damage: dofs include edge midpoints, so raw dof array != node values.

    The extractor must evaluate the FE function at vertices. For a linear field
    d = a*x + b*y the recovered nodal d must match the analytic vertex values.
    """
    n = 16
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=n, ny=n)
    sp2 = LagrangeFESpace(mesh, p=2)
    d = sp2.function()
    ip = sp2.interpolation_points()
    a, b = 0.3, 0.4
    d[:] = bm.array(np.asarray(a * ip[:, 0] + b * ip[:, 1], dtype=float))
    # P2 has more dofs than nodes -> raw dof array is NOT the node field.
    assert sp2.number_of_global_dofs() > mesh.number_of_nodes()

    st = _State(d)
    cf = extract_coarse_features(mesh, _QuadraticDamage(), st, l0=L0)
    node = np.asarray(mesh.entity("node"), dtype=float)
    expected = np.clip(a * node[:, 0] + b * node[:, 1], 0.0, 1.0)
    assert cf.d == pytest.approx(expected, abs=1e-10)


def test_requires_positive_l0():
    mesh, st = _make(16)
    with pytest.raises(ValueError):
        extract_coarse_features(mesh, _QuadraticDamage(), st, l0=0.0)
