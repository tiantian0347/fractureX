# fracturex/learn/models/__init__.py
"""Neural-operator architectures for the surrogate.

Roster (plan §3.9 Baselines):
  - unet:          U-Net, strong sharp-front baseline (M1 required comparator)
  - fno_2d:        FNO2d-global (scheme A, plan §3.9), M1 main
  - deeponet:      branch-trunk baseline / mesh-flexible control (M1)
  - multioutput_fno: 4-channel head (d, σ_xx, σ_yy, σ_xy), M2 Stage B
  - geo_fno:       Geo-FNO with learned diffeomorphism (M2 robustness)

Uniform M1 contract (scheme A): every model maps a static input tensor
``(B, C_in, H, W)`` to all-T damage channels ``(B, T, H, W)`` via ``forward(x)``.
Use :func:`build_model` so the training loop is model-agnostic — that registry
is the extension point for new architectures.
"""
from __future__ import annotations

from typing import Callable


class _MultiField:
    """Wrap a single-block backbone (out = T·n_fields) into (B, T, n_fields, H, W).

    Lets any (C_in, H, W)→(C_out, H, W) backbone serve as a multi-field
    (Stage B) predictor without bespoke code. Built lazily inside build_model
    so this module stays torch-free at import.
    """


def _make_multifield(backbone_factory, in_channels, T, n_fields):
    import torch.nn as nn

    class _MF(nn.Module):
        def __init__(self):
            super().__init__()
            self.T, self.n_fields = T, n_fields
            self.backbone = backbone_factory(in_channels, T * n_fields)

        def forward(self, x):
            B, _, H, W = x.shape
            return self.backbone(x).view(B, self.T, self.n_fields, H, W)

    return _MF()


def build_model(name: str, in_channels: int, out_channels: int, *,
                grid_hw: tuple[int, int] | None = None,
                n_fields: int = 1, **kwargs):
    """Construct a model by name.

    Stage A (``n_fields=1``): uniform contract (C_in, H, W) → (T, H, W).
    Stage B (``n_fields>1``, multioutput_* models): (C_in, H, W) → (T, n_fields, H, W).

    Args:
        name:        'unet' | 'fno_2d' | 'deeponet' | 'multioutput_fno' | 'multioutput_unet'.
        in_channels: C_in (= 4 + k material channels).
        out_channels: T (load steps; scheme A predicts all as channels).
        grid_hw:     (H, W); required by deeponet, ignored by others.
        n_fields:    fields per time step (Stage B: 4 = d,σxx,σyy,σxy; 5 with 𝓗).
        kwargs:      forwarded to the model constructor.
    """
    name = name.lower()
    if name == "unet":
        from .unet import UNet2d
        return UNet2d(in_channels, out_channels, **kwargs)
    if name in ("fno", "fno_2d", "fno2d"):
        from .fno_2d import FNO2d
        return FNO2d(in_channels, out_channels, **kwargs)
    if name == "deeponet":
        from .deeponet import DeepONet
        if grid_hw is None:
            raise ValueError("deeponet requires grid_hw=(H, W)")
        return DeepONet(in_channels, out_channels, grid_hw=grid_hw, **kwargs)
    if name in ("multioutput_fno", "mo_fno"):
        from .multioutput_fno import MultiOutputFNO2d
        return MultiOutputFNO2d(in_channels, T=out_channels, n_fields=n_fields, **kwargs)
    if name in ("multioutput_unet", "mo_unet"):
        from .unet import UNet2d
        return _make_multifield(
            lambda cin, cout: UNet2d(cin, cout, **kwargs), in_channels, out_channels, n_fields
        )
    raise ValueError(
        f"unknown model {name!r}; available: {', '.join(available_models())}"
    )


def available_models() -> list[str]:
    return ["unet", "fno_2d", "deeponet", "multioutput_fno", "multioutput_unet"]
