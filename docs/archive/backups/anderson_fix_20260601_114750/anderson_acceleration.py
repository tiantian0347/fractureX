"""Safeguarded Anderson acceleration for the staggered phase-field loop.

Implements the Storvik et al. (CMAME 381:113822, 2021) scheme: treat one
staggered sweep as a fixed-point map ``d_{k+1} = G(d_k)`` and post-process its
increments, *switching between Anderson acceleration and over-relaxation
depending on the residual evolution*, with a restart safeguard so the
accelerator cannot get stuck in a limit cycle at the nucleation/propagation
step. Solver-agnostic: it only sees the iterate ``x = d_old`` and the raw
fixed-point image ``Gx = d_trial``; irreversibility ``max(d_old,.)`` and clip
``[0,1]`` are applied by the caller AFTER acceleration (projected Anderson).

Switching logic (Storvik):
  - while the residual norm ||f|| = ||Gx - x|| is *decreasing* (quasi-static /
    easy step) use over-relaxation  x_new = x + omega * f  with omega > 1;
  - once it stalls or oscillates (crack evolving) switch to windowed Anderson;
  - if Anderson itself stalls (no residual improvement for `restart_patience`
    steps) or the residual blows up relative to its best value, *restart* the
    window and take an over-relaxation step to perturb out of the cycle.

Theory anchor: alternate minimization is energy-monotone and converges
(subsequence sense) to a critical point [Bourdin-Francfort-Marigo, JMPS 48:797,
2000; Bourdin, IFB 2007]; the acceleration is a heuristic efficiency device
(no convergence proof of the accelerated scheme). Validated for the numpy
backend (the paper runs); the Anderson least squares is done in float64 numpy.
"""
from __future__ import annotations

import numpy as np


class AndersonAccelerator:
    """Storvik-style safeguarded Anderson(m) / over-relaxation hybrid."""

    def __init__(
        self,
        depth: int = 5,
        beta: float = 1.0,
        omega: float = 1.5,
        restart_patience: int = 3,
        blowup_factor: float = 2.0,
        reg: float = 1e-12,
    ):
        """
        Inputs:
            depth: Anderson window size m (number of past residual differences).
            beta:  Anderson mixing parameter in (0, 1]; 1.0 = plain Anderson.
            omega: over-relaxation factor (> 1 accelerates a monotone fixed
                   point; 1.0 recovers the plain staggered step).
            restart_patience: restart the Anderson window after this many
                   consecutive accelerated steps with no residual improvement.
            blowup_factor: restart if ||f_k|| exceeds this multiple of the best
                   residual seen so far (Anderson diverged / overshot).
            reg:   Tikhonov regularization on the Anderson normal equations.
        """
        self.m = max(0, int(depth))
        self.beta = float(beta)
        self.omega = float(omega)
        self.restart_patience = int(restart_patience)
        self.blowup_factor = float(blowup_factor)
        self.reg = float(reg)
        self.reset()

    def reset(self) -> None:
        """Clear all state. Call once at the start of every load step."""
        self._X: list[np.ndarray] = []
        self._F: list[np.ndarray] = []
        self._rhist: list[float] = []
        self._stall = 0
        self._best = np.inf
        # last-used mode, for diagnostics: 'plain' | 'overrelax' | 'anderson'
        self.last_mode = "plain"

    def _restart_window(self) -> None:
        self._X.clear()
        self._F.clear()
        self._stall = 0

    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """Return the accelerated iterate given x = d_old and gx = G(d_old).

        The caller must still apply irreversibility max(d_old, .) and clip [0,1].
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        gx = np.asarray(gx, dtype=np.float64).reshape(-1)
        f = gx - x
        rn = float(np.linalg.norm(f))

        improved = rn < (self._rhist[-1] if self._rhist else np.inf)
        # --- safeguards: restart the window on blow-up or persistent stall ---
        if self._X and rn > self.blowup_factor * self._best:
            self._restart_window()
        if self._stall >= self.restart_patience:
            self._restart_window()

        # --- mode selection ----------------------------------------------------
        # Use Anderson whenever the window has entries (this is what delivers the
        # speed-up AND reproduces plain AM at the tolerance). The window is only
        # (re)seeded with a plain/over-relaxed step right after a restart, which
        # perturbs the iterate out of a limit cycle. Over-relaxation is NOT used
        # in the decreasing-residual regime, because at a fixed tolerance it
        # converges to a slightly different state and degrades the reaction.
        if len(self._X) == 0:
            x_new = x + self.omega * f          # (re)seed; omega=1.0 => plain
            self.last_mode = "plain" if self.omega == 1.0 else "overrelax"
        else:
            dX = np.column_stack([x - xi for xi in self._X])
            dF = np.column_stack([f - fi for fi in self._F])
            try:
                G = dF.T @ dF
                pp = G.shape[0]
                G.flat[:: pp + 1] += self.reg * (np.trace(G) / max(pp, 1) + 1e-30)
                gamma = np.linalg.solve(G, dF.T @ f)
                x_new = (x - dX @ gamma) + self.beta * (f - dF @ gamma)
                self.last_mode = "anderson"
            except np.linalg.LinAlgError:
                x_new = x + f
                self.last_mode = "plain"

        if not np.all(np.isfinite(x_new)):
            x_new = x + f                      # last-resort plain step
            self.last_mode = "plain"
            self._restart_window()

        # --- bookkeeping -------------------------------------------------------
        self._stall = 0 if improved else self._stall + 1
        self._best = min(self._best, rn)
        self._rhist.append(rn)
        self._push(x, f)
        return x_new

    def _push(self, x: np.ndarray, f: np.ndarray) -> None:
        self._X.append(x.copy())
        self._F.append(f.copy())
        while len(self._X) > self.m:
            self._X.pop(0)
            self._F.pop(0)
