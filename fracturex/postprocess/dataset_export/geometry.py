# fracturex/postprocess/dataset_export/geometry.py
"""Geometry abstraction for the surrogate exporter.

A "geometry" describes the physical domain Ω ⊂ R² used to build the
``sdf`` / ``mask`` input channels (schema §3.1). It is **model-agnostic**:
the export core only needs a signed-distance evaluator, so porting to a new
domain shape means supplying a new :class:`Geometry` (or a plain callable),
never touching the core.

Two representations are accepted (see :data:`GeometryLike`):
  1. An object exposing ``signed_distance(points) -> array`` with the schema
     convention *positive inside Ω*.
  2. A bare callable ``points -> array`` with the same convention.

``CircularNotchDomain`` is the reference implementation (matches
``cases/model0_circular_notch.py``); add sibling classes here for other shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Union, runtime_checkable

import numpy as np


@runtime_checkable
class Geometry(Protocol):
    """Domain description sufficient to build SDF / mask channels.

    Implementations must return a signed distance to ∂Ω that is **positive
    inside Ω, negative outside, zero on the boundary** (schema §3.1).
    """

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance for ``points`` of shape (..., 2); same leading shape out."""
        ...


@dataclass(frozen=True)
class CircularNotchDomain:
    """Ω = box [x0,x1]×[y0,y1] minus a disk of radius r at (cx, cy).

    Matches ``Model0CircularNotchCase`` (cases/model0_circular_notch.py).
    """

    box: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    cx: float = 0.5
    cy: float = 0.5
    r: float = 0.2

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """SDF (positive inside Ω) for points (..., 2)."""
        pts = np.asarray(points, dtype=np.float64)
        x0, x1, y0, y1 = self.box
        x = pts[..., 0]
        y = pts[..., 1]
        # box SDF (positive inside)
        d_box = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])
        # disk SDF: dist_to_circle = r - radius;  inside disk ⇒ positive.
        # The disk is a *hole*, so Ω-inside ⇔ outside disk ⇔ -d_disk_inside.
        d_to_center = np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2)
        d_outside_disk = d_to_center - self.r  # positive when outside disk
        return np.minimum(d_box, d_outside_disk)


GeometryLike = Union[Geometry, Callable[[np.ndarray], np.ndarray]]


def signed_distance(geometry: GeometryLike, points: np.ndarray) -> np.ndarray:
    """Dispatch to ``geometry.signed_distance`` or call ``geometry`` directly."""
    if hasattr(geometry, "signed_distance"):
        return np.asarray(geometry.signed_distance(points), dtype=np.float64)
    if callable(geometry):
        return np.asarray(geometry(points), dtype=np.float64)
    raise TypeError(
        f"geometry must expose signed_distance(points) or be callable; got {type(geometry)!r}"
    )


# Backwards-compatible private alias (old code referenced ``_signed_distance``).
_signed_distance = signed_distance
