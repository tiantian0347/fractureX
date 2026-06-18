"""Training for the learned coarse-space amplitude model (D13 L2-beta).

Objective (plan §5.2, supervised variant): regress the per-node amplitude model to the
``ideal_interface_amplitude`` label (spectral_labels), which marks the high-contrast
interface nodes the geometric P1 coarse space cannot resolve. This is the stable
offline target; the *real* objective — the two-level condition number — is reported as
an evaluation metric (not back-propagated; differentiating an eigensolver is unstable,
plan §5.2).

Standardization comes from the TRAIN split only (datasets.DataSplit), and is baked into
the model buffers so inference applies the identical z-score.

torch lives here and in coarse_weight_model only; the solver never imports either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from fracturex.ml.coarse_features import N_FEATURES
from fracturex.ml.coarse_weight_model import AmplitudeModelConfig, build_model
from fracturex.ml.datasets import DataSplit, FeatureSample
from fracturex.ml.spectral_labels import ideal_interface_amplitude


def _torch():
    import torch
    return torch


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    hidden: Sequence[int] = (16, 16)
    a_max: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    seed: int = 0


def _stack_xy(samples: Sequence[FeatureSample], *, use_target: bool = True):
    """Concatenate features and per-node amplitude labels over all sample nodes.

    Label priority: the stored spectral ``sample.target`` (worst-mode amplitude) when
    present and ``use_target``; otherwise the feature heuristic ``ideal_interface_amplitude``.
    The spectral target is what makes the learned model reproduce a worst-mode-like Phi
    (validated to give ~4x the niter gain of the heuristic, D13_IMPL §6.3).
    """
    import numpy as np
    from fealpy.backend import backend_manager as bm
    X, Y = [], []
    for s in samples:
        phi = np.asarray(bm.to_numpy(s.phi), dtype=np.float64)
        X.append(phi)
        if use_target and s.target is not None:
            Y.append(np.asarray(bm.to_numpy(s.target), dtype=np.float64).reshape(-1))
        else:
            Y.append(ideal_interface_amplitude(phi))
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


def train_amplitude_model(split: DataSplit, cfg: Optional[TrainConfig] = None):
    """Train the per-node amplitude model on a DataSplit (supervised to ideal label).

    Args:
        split: a :class:`DataSplit` (train/test + train-fitted standardizer).
        cfg: training hyperparameters.
    Returns:
        ``(model, history)`` where history is a dict with ``train_loss`` / ``test_loss``
        per epoch (test_loss only if the split has a test set).
    """
    torch = _torch()
    cfg = cfg or TrainConfig()
    torch.manual_seed(int(cfg.seed))

    from fealpy.backend import backend_manager as bm
    mean = np.asarray(bm.to_numpy(split.mean), dtype=np.float64)
    std = np.asarray(bm.to_numpy(split.std), dtype=np.float64)
    mcfg = AmplitudeModelConfig(
        in_dim=N_FEATURES, hidden=cfg.hidden, a_max=cfg.a_max,
        feature_mean=mean.tolist(), feature_std=std.tolist(),
    )
    model = build_model(mcfg)

    Xtr, Ytr = _stack_xy(split.train)
    xtr = torch.as_tensor(Xtr, dtype=torch.float64)
    ytr = torch.as_tensor(Ytr, dtype=torch.float64)
    has_test = len(split.test) > 0
    if has_test:
        Xte, Yte = _stack_xy(split.test)
        xte = torch.as_tensor(Xte, dtype=torch.float64)
        yte = torch.as_tensor(Yte, dtype=torch.float64)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()
    history = {"train_loss": [], "test_loss": []}

    for _ in range(int(cfg.epochs)):
        model.train()
        opt.zero_grad()
        pred = model(xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()
        history["train_loss"].append(float(loss.item()))
        if has_test:
            model.eval()
            with torch.no_grad():
                history["test_loss"].append(float(loss_fn(model(xte), yte).item()))

    return model, history
