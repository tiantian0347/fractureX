"""Unit tests for fracturex.postprocess.dataset_export geometry/grid helpers.

Covers:
  - CircularNotchDomain.signed_distance: matches model0 geometry analytically.
  - compute_sdf / compute_valid_mask: convention (positive inside Ω) and
    consistency (mask == (sdf >= 0)).
  - compute_coords: shape, range, indexing convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from fracturex.postprocess.dataset_export import (
    SCHEMA_VERSION,
    GridSpec,
    CircularNotchDomain,
    compute_coords,
    compute_sdf,
    compute_valid_mask,
)


M0_DOMAIN = CircularNotchDomain(box=(0.0, 1.0, 0.0, 1.0),
                                 cx=0.5, cy=0.5, r=0.2)


class TestCircularNotchSDF:
    """Match the model0_circular_notch geometry (cases/model0_circular_notch.py)."""

    def test_center_is_inside_disk(self):
        # Hole center is *outside* Ω. SDF(center) = -r.
        s = M0_DOMAIN.signed_distance(np.array([[0.5, 0.5]]))
        np.testing.assert_allclose(s[0], -0.2, atol=1e-12)

    def test_disk_boundary_is_zero(self):
        # On the disk boundary: distance from center = r.
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        pts = np.stack([0.5 + 0.2 * np.cos(theta),
                        0.5 + 0.2 * np.sin(theta)], axis=-1)
        s = M0_DOMAIN.signed_distance(pts)
        np.testing.assert_allclose(s, 0.0, atol=1e-12)

    def test_box_boundary_is_zero(self):
        # On the outer box boundary (away from disk).
        pts = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]])
        s = M0_DOMAIN.signed_distance(pts)
        np.testing.assert_allclose(s, 0.0, atol=1e-12)

    def test_far_outside_box_negative(self):
        s = M0_DOMAIN.signed_distance(np.array([[-0.5, 0.5]]))
        np.testing.assert_allclose(s[0], -0.5, atol=1e-12)

    def test_just_inside_disk_negative(self):
        # Inside the disk → outside Ω → SDF negative.
        s = M0_DOMAIN.signed_distance(np.array([[0.5 + 0.1, 0.5]]))
        np.testing.assert_allclose(s[0], -(0.2 - 0.1), atol=1e-12)

    def test_safe_interior_positive(self):
        # Far from any boundary, inside Ω → positive.
        s = M0_DOMAIN.signed_distance(np.array([[0.1, 0.1]]))
        # Distances: x=0.1 from x=0; y=0.1 from y=0; from center
        # sqrt(0.4^2+0.4^2)-0.2 ≈ 0.366. Min over those = 0.1.
        np.testing.assert_allclose(s[0], 0.1, atol=1e-12)

    def test_batch_shapes_preserved(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0.0, 1.0, size=(4, 5, 2))
        s = M0_DOMAIN.signed_distance(pts)
        assert s.shape == (4, 5)


class TestComputeSDFAndMask:

    @pytest.fixture
    def grid(self) -> GridSpec:
        return GridSpec(H=64, W=64, bbox=((0.0, 1.0), (0.0, 1.0)))

    def test_sdf_shape_dtype(self, grid):
        sdf = compute_sdf(grid, M0_DOMAIN)
        assert sdf.shape == (1, grid.H, grid.W)
        assert sdf.dtype == np.float32

    def test_mask_shape_dtype(self, grid):
        m = compute_valid_mask(grid, M0_DOMAIN)
        assert m.shape == (1, grid.H, grid.W)
        assert m.dtype == np.uint8
        assert set(np.unique(m)).issubset({0, 1})

    def test_mask_consistent_with_sdf(self, grid):
        sdf = compute_sdf(grid, M0_DOMAIN)
        m = compute_valid_mask(grid, M0_DOMAIN)
        assert np.array_equal(m[0], (sdf[0] >= 0.0).astype(np.uint8))

    def test_corner_inside_box_outside_disk(self, grid):
        # (0,0) is the box corner: on box boundary → sdf == 0, mask == 1.
        sdf = compute_sdf(grid, M0_DOMAIN)
        m = compute_valid_mask(grid, M0_DOMAIN)
        # First pixel is at (x_lo=0, y_lo=0).
        np.testing.assert_allclose(sdf[0, 0, 0], 0.0, atol=1e-6)
        assert m[0, 0, 0] == 1

    def test_center_pixel_excluded(self, grid):
        # The pixel containing (0.5, 0.5) must be excluded from Ω.
        m = compute_valid_mask(grid, M0_DOMAIN)
        i_mid = grid.H // 2
        j_mid = grid.W // 2
        assert m[0, i_mid, j_mid] == 0

    def test_mask_count_matches_area(self, grid):
        # Inside-Ω pixel ratio ≈ (1 - π r²) / 1 = 1 - π·0.04.
        m = compute_valid_mask(grid, M0_DOMAIN)
        ratio = m.mean()
        expected = 1.0 - np.pi * 0.2 ** 2
        # 64×64 grid: tolerate ~3% from quantization (notch is small).
        assert abs(ratio - expected) < 0.03

    def test_callable_geometry(self, grid):
        """Accept a plain callable returning SDF (positive inside)."""
        m_ref = compute_valid_mask(grid, M0_DOMAIN)
        m_cb = compute_valid_mask(grid, M0_DOMAIN.signed_distance)
        np.testing.assert_array_equal(m_ref, m_cb)


class TestComputeCoords:

    def test_shape_and_range(self):
        grid = GridSpec(H=8, W=16, bbox=((-1.0, 1.0), (-2.0, 2.0)))
        c = compute_coords(grid)
        assert c.shape == (2, 8, 16)
        assert c.dtype == np.float32
        np.testing.assert_allclose(c[0].min(), 0.0, atol=1e-7)
        np.testing.assert_allclose(c[0].max(), 1.0, atol=1e-7)
        np.testing.assert_allclose(c[1].min(), 0.0, atol=1e-7)
        np.testing.assert_allclose(c[1].max(), 1.0, atol=1e-7)

    def test_x_increases_along_W_axis(self):
        grid = GridSpec(H=4, W=8, bbox=((0.0, 1.0), (0.0, 1.0)))
        c = compute_coords(grid)
        # coords[0] = x; should be constant along H, varying along W.
        for i in range(grid.H):
            np.testing.assert_array_equal(c[0, i, :], c[0, 0, :])
        assert (np.diff(c[0, 0, :]) > 0).all()

    def test_y_increases_along_H_axis(self):
        grid = GridSpec(H=8, W=4, bbox=((0.0, 1.0), (0.0, 1.0)))
        c = compute_coords(grid)
        for j in range(grid.W):
            np.testing.assert_array_equal(c[1, :, j], c[1, :, 0])
        assert (np.diff(c[1, :, 0]) > 0).all()


def test_schema_version_pinned():
    assert SCHEMA_VERSION == "0.1"
