# fracturex/learn/datasets.py
"""PyTorch Dataset adapters for the SURROGATE_DATA_SCHEMA npz format.

Schema: docs/SURROGATE_DATA_SCHEMA.md
Plan:   docs/plan_operator_learning.md §M1
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class DatasetConfig:
    """Knobs that select which npz fields a Dataset exposes."""
    include_stress: bool = True          # T3 main / T2 ablation
    include_history: bool = False        # plan §3.5 (b) variant
    rollout_length: Optional[int] = None # None → full sequence; else random window of length k
    augment_rotate90: bool = False
    augment_flip: bool = False


class PhaseFieldOperatorDataset:
    """Lazy reader of `samples/sample_XXXXXX.npz` from a dataset directory.

    Index → (inputs_dict, outputs_dict, valid_mask, metadata_dict).

    Subclassed (or wrapped) to produce framework-native batches:
      - PyTorch: see `as_torch_dataset()`.
      - JAX:     see `as_jax_iterable()` (deferred to M2).
    """

    def __init__(self, dataset_dir: Path, split: str = "train",
                 cfg: Optional[DatasetConfig] = None) -> None:
        raise NotImplementedError("M1 task: implement lazy npz reader + split lookup")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return a sample-shaped dict; tensors stay as numpy until backend wrapper."""
        raise NotImplementedError


def collate_masked(batch: list[dict]) -> dict:
    """Stack a batch while keeping mask channel; used by torch DataLoader."""
    raise NotImplementedError("M1 task: batch stacker with mask broadcast handling")
