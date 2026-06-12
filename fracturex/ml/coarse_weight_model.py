"""Small per-node amplitude model for the learned coarse-space enrichment (D13 L2-beta).

The model maps a node's dimensionless feature vector ``phi`` (see ``coarse_features``)
to a bounded scalar amplitude ``a in [0, a_max]``; the enrichment columns are then
``Phi = template * a`` (``scale_modes``), so the network only modulates a fixed,
physically-motivated interface jump template. This keeps the learned object small,
interpretable, and — combined with the Galerkin seam — provably SPD-safe regardless of
the network output (plan command 4: amplitude can never break correctness).

torch lives ONLY in this module and the training script; the solver never imports it.
Inference runs once per staggered setup (``predict_amplitude`` -> numpy), feeding the
solver-side enrichment provider (plan §3.3): no torch on the GMRES hot path.

Backend note: features come in as bm/numpy; we convert to a torch tensor at the model
boundary and return numpy amplitudes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from fracturex.ml.coarse_features import N_FEATURES


def _torch():
    """Import torch lazily so importing this module never hard-requires it."""
    import torch
    return torch


@dataclass
class AmplitudeModelConfig:
    """Hyperparameters for :class:`CoarseAmplitudeMLP`.

    Attributes:
        in_dim: feature width (defaults to N_FEATURES).
        hidden: hidden layer widths.
        a_max: output upper bound (amplitude in [0, a_max]).
        feature_mean, feature_std: optional standardization (from datasets.DataSplit),
            stored so inference applies the SAME z-score as training.
    """

    in_dim: int = N_FEATURES
    hidden: Sequence[int] = (16, 16)
    a_max: float = 1.0
    feature_mean: Optional[Sequence[float]] = None
    feature_std: Optional[Sequence[float]] = None


def build_model(cfg: AmplitudeModelConfig):
    """Build the per-node amplitude MLP (bounded output via sigmoid * a_max).

    Args:
        cfg: model configuration.
    Returns:
        a ``torch.nn.Module`` mapping ``(N, in_dim) -> (N,)`` in ``[0, a_max]``.
    """
    torch = _torch()
    nn = torch.nn

    class CoarseAmplitudeMLP(nn.Module):
        def __init__(self, cfg: AmplitudeModelConfig):
            super().__init__()
            self.a_max = float(cfg.a_max)
            dims = [int(cfg.in_dim), *map(int, cfg.hidden)]
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.GELU()]
            layers += [nn.Linear(dims[-1], 1)]
            self.net = nn.Sequential(*layers)
            mean = cfg.feature_mean if cfg.feature_mean is not None else [0.0] * cfg.in_dim
            std = cfg.feature_std if cfg.feature_std is not None else [1.0] * cfg.in_dim
            self.register_buffer("mean", torch.as_tensor(np.asarray(mean), dtype=torch.float64))
            self.register_buffer("std", torch.as_tensor(np.asarray(std), dtype=torch.float64))
            self.double()

        def forward(self, x):
            x = (x - self.mean) / self.std
            z = self.net(x).squeeze(-1)
            return self.a_max * torch.sigmoid(z)

    return CoarseAmplitudeMLP(cfg)


def predict_amplitude(model, features) -> np.ndarray:
    """Run the model on a feature block, returning per-node amplitudes as numpy.

    Inference-only (no grad); this is the once-per-setup call the solver-side provider
    uses to build ``Phi = template * amplitude``.

    Args:
        model: a built amplitude model.
        features: ``(N, in_dim)`` features (bm/numpy).
    Returns:
        ``(N,)`` numpy float64 amplitudes in ``[0, a_max]``.
    """
    torch = _torch()
    from fealpy.backend import backend_manager as bm
    x = np.asarray(bm.to_numpy(features) if not isinstance(features, np.ndarray)
                   else features, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        a = model(torch.as_tensor(x, dtype=torch.float64))
    return np.asarray(a.detach().cpu().numpy(), dtype=np.float64)


def save_model(model, cfg: AmplitudeModelConfig, path) -> None:
    """Persist model weights + config for later inference."""
    torch = _torch()
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, str(path))


def load_model(path):
    """Load a model saved by :func:`save_model`. Returns ``(model, cfg)``."""
    torch = _torch()
    blob = torch.load(str(path), weights_only=False)
    cfg = AmplitudeModelConfig(**blob["cfg"])
    model = build_model(cfg)
    model.load_state_dict(blob["state_dict"])
    return model, cfg
