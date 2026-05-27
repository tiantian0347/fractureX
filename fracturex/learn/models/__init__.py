# fracturex/learn/models/__init__.py
"""Neural-operator architectures for the surrogate.

Roster (plan §3.9 Baselines):
  - unet:          U-Net, strong sharp-front baseline (M1 required comparator)
  - fno_2d:        FNO2d-global (scheme A, plan §3.9), M1 main
  - multioutput_fno: 4-channel head (d, σ_xx, σ_yy, σ_xy), M2 Stage B
  - deeponet:      theoretical baseline / mesh-flexible control
  - geo_fno:       Geo-FNO with learned diffeomorphism (M2 robustness)
"""
from __future__ import annotations
