# fracturex/learn/train.py
"""Training loop entry point — backend-agnostic skeleton.

PyTorch is the default; JAX deferred to M2+. Each stage from plan §3.5
table (A → E) is a separate train(...) invocation with different
`stage` and loss-weight overrides.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    dataset_dir: Path
    out_dir: Path
    model: str                  # 'unet' | 'fno_2d' | 'multioutput_fno' | 'deeponet'
    stage: str = "A"            # A | B | C | D | E   (plan §3.5)
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    rollout_length: Optional[int] = None
    seed: int = 0
    # Loss weights (overrides plan §3.5 defaults if non-None)
    lambda_sigma: Optional[float] = None
    lambda_h1: Optional[float] = None
    lambda_front: Optional[float] = None
    lambda_eq: Optional[float] = None
    lambda_irr: Optional[float] = None


def train(cfg: TrainConfig) -> None:
    """Single-stage training run.

    Writes:
      out_dir/
        config.json
        metrics.csv          (epoch, train_loss, val_loss, all eval metrics)
        checkpoints/model_epoch_XXX.pt
        eval_report.md       (final metric table on the held-out split)
    """
    raise NotImplementedError("M1 task: build model, dataset, loss, opt, train loop")
