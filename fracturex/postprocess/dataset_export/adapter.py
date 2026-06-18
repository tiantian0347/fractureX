# fracturex/postprocess/dataset_export/adapter.py
"""The portability seam: :class:`SolverAdapter`.

The export *core* (grid building, masking, normalization, npz/json packing,
schema invariants) is model-agnostic. Everything that knows about a specific
physical model — how to rebuild its discretization from a recorder dir, how to
evaluate its output fields on a grid, which material parameters it carries —
is isolated behind this Protocol.

**Porting the operator-learning pipeline to a new model = implementing one
`SolverAdapter`.** No change to the core, the grid/sampling utilities, or the
`fracturex/learn/` training side is required. The reference implementation is
``adapters/huzhang_phasefield.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from .grid import GridSpec
from .sampling import PixelLocator


@dataclass(frozen=True)
class FieldSpec:
    """Declares one output field the adapter produces (schema §3.2).

    Attributes:
        name:     npz key, e.g. ``"damage"`` / ``"stress"``.
        channels: channel count C of the (C, H, W) per-frame field.
        scaling:  normalization policy applied by the core:
                    ``"none"``         — store as-is (e.g. damage ∈ [0,1]);
                    ``"stress_scale"`` — divide by the (auto or configured)
                                         stress scale and record it in metadata.
    """

    name: str
    channels: int
    scaling: str = "none"


@runtime_checkable
class SolverAdapter(Protocol):
    """Model-specific operations the export core delegates to.

    Implementations are typically stateless and cheap to construct; per-run
    state lives in the opaque ``discr`` handle returned by
    :meth:`load_discretization` and threaded back through the other methods.
    """

    #: schema version this adapter targets (must match core SCHEMA_VERSION).
    schema_version: str
    #: fixed order of the material parameter vector (schema §3.5).
    material_order: tuple[str, ...]
    #: output fields produced per frame, in schema order.
    output_field_specs: tuple[FieldSpec, ...]

    def load_discretization(self, recorder_dir: Path) -> Any:
        """Rebuild the model's discretization handle from a recorder dir."""
        ...

    def mesh(self, discr: Any) -> Any:
        """Return the FE mesh of ``discr`` (used to build the pixel locator)."""
        ...

    def list_checkpoints(self, recorder_dir: Path) -> list[Path]:
        """Ordered list of per-step checkpoint files under ``recorder_dir``."""
        ...

    def material_vector(
        self, recorder_meta: dict, overrides: Optional[dict] = None
    ) -> np.ndarray:
        """Build the (k,) material vector in :attr:`material_order` from meta."""
        ...

    def evaluate_outputs(
        self,
        discr: Any,
        checkpoint: dict,
        locator: PixelLocator,
        grid: GridSpec,
    ) -> dict[str, np.ndarray]:
        """Evaluate one frame's output fields on the grid.

        Returns ``{spec.name: (spec.channels, H, W)}`` in **schema channel
        order**, for the single checkpoint ``checkpoint`` (a loaded npz mapping).
        Outside-Ω masking and normalization are applied by the core afterwards.
        """
        ...

    def geometry_meta(
        self, recorder_dir: Path, recorder_meta: dict, cfg: Any
    ) -> dict:
        """Model/geometry descriptor for ``<sample>.meta.json::geometry_params``."""
        ...
