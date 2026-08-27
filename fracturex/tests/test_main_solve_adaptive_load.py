"""Unit checks for adaptive load-step state rejection and restoration.

The test uses an in-memory scalar surrogate for the MainSolve load loop.  It
verifies the contract needed by fracture continuation: a failed target is
bisected, the last accepted state is restored, and only converged reactions are
committed.  No finite-element assembly is required at this unit-test layer.
"""
from __future__ import annotations

import numpy as np

from fealpy.backend import backend_manager as bm

from fracturex.phasefield.main_solve import MainSolve


class _DummyField:
    """Minimal array-like field with a dtype and mutable slice."""

    def __init__(self, value: float):
        self.array = bm.array([value], dtype=bm.float64)

    @property
    def dtype(self):
        return self.array.dtype

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value


class _DummyMaterial:
    """Record restored fields without performing constitutive work."""

    def __init__(self):
        self.H = bm.array([0.0], dtype=bm.float64)

    def update_historical_field(self, history):
        self.H = history

    def update_disp(self, displacement):
        self.uh = displacement

    def update_phase(self, damage):
        self.d = damage


class _AdaptiveLoopSurrogate(MainSolve):
    """MainSolve load loop whose nonlinear step fails for increments above 0.3."""

    def __init__(self):
        self.uh = _DummyField(0.0)
        self.d = _DummyField(0.0)
        self.H = bm.array([0.0], dtype=bm.float64)
        self.pfcm = _DummyMaterial()
        self.bc_dict = {
            "force": [
                {
                    "type": "Dirichlet",
                    "bcdof": None,
                    "value": bm.array([0.0, 1.0], dtype=bm.float64),
                    "direction": "y",
                }
            ]
        }
        self._save_vtk = False
        self._damage_npz_dir = None
        self.attempts = []

    def initialize_settings(self, p=1, q=None):
        return None

    def set_linear_solver_options(self, **kwargs):
        return None

    def _write_step_damage(self, step):
        return None

    def newton_raphson(self, maxit=50):
        current = float(np.asarray(bm.to_numpy(self.uh[:]))[0])
        target = float(np.asarray(bm.to_numpy(self._currt_force_value)))
        self.attempts.append((current, target))
        self.uh[:] = target
        self.d[:] = target
        self.H[:] = target
        self.pfcm.H = self.H
        self._Rfu = target
        converged = abs(target - current) <= 0.3 + 1.0e-12
        return converged, 1, 0.0 if converged else 1.0


def test_adaptive_load_bisects_and_restores_rejected_state():
    """A failed unit increment becomes four accepted quarter increments."""
    solver = _AdaptiveLoopSurrogate()
    solver.solve(
        adaptive_load_step=True,
        min_load_step=0.125,
        max_subdivisions=3,
    )

    assert np.isclose(float(np.asarray(bm.to_numpy(solver.uh[:]))[0]), 1.0)
    assert np.isclose(float(np.asarray(bm.to_numpy(solver.d[:]))[0]), 1.0)
    assert np.isclose(float(np.asarray(bm.to_numpy(solver.H))[0]), 1.0)
    assert np.allclose(bm.to_numpy(solver.get_residual_force()), [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(
        bm.to_numpy(solver.get_accepted_load_path()), [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    assert solver.attempts[:3] == [(0.0, 1.0), (0.0, 0.5), (0.0, 0.25)]
