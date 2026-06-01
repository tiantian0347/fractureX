"""Safeguarded Anderson acceleration for the staggered phase-field loop.

基于 Storvik et al. (CMAME 381:113822, 2021):把一次 staggered 扫描看作不动点
映射 ``d_{k+1} = G(d_k)``,对其增量做窗口化 Anderson 外推。求解器无关——只看
迭代 ``x = d_old`` 和原始不动点像 ``Gx = d_trial``;不可逆 ``max(d_old,.)`` 与
clip ``[0,1]`` 由调用方在加速 *之后* 施加(projected Anderson)。

实际逻辑(与本文件 ``step`` 一致):
  - 窗口一旦有料(>=1 个历史增量)就 **一直用 Anderson**——这才同时给出提速与
    "在 tol 处复现 plain AM 的精度";不按残差切换 over-relaxation(固定 tol 下
    over-relax 会把解推偏、损害反力)。
  - 窗口为空时(每载荷步初、或刚 restart 后)取一步 ``x + omega*f`` 重新播种
    (``omega=1.0`` 即 plain 步)。
  - **信赖域**:把单次加速步长限制为 plain 步长 ``||f||`` 的 ``tr_factor`` 倍,
    防止裂纹突跳处病态最小二乘外推 *有限大过冲*(``isfinite`` 拦不住)。
  - **restart 兜底**:残差相对 *近期*(上一迭代)残差反弹 ``blowup_factor`` 倍,
    或连续 ``restart_patience`` 步无改善时清空窗口,打破起裂步极限环。

理论锚点:交替最小化能量单调、(子列意义)收敛到临界点 [Bourdin-Francfort-
Marigo, JMPS 48:797, 2000; Bourdin, IFB 2007];加速是启发式效率手段(加速格式本
身无收敛证明)。仅 numpy 后端验证(论文算例);Anderson 最小二乘用 float64 numpy。
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
        tr_factor: float = 20.0,
        restart_omega: float = 1.6,
        reg: float = 1e-12,
    ):
        """
        Inputs（均为标量）:
            depth: Anderson window size m (number of past residual differences).
            beta:  Anderson mixing parameter in (0, 1]; 1.0 = plain Anderson.
            omega: over-relaxation factor (> 1 accelerates a monotone fixed
                   point; 1.0 recovers the plain staggered step).
            restart_patience: restart the Anderson window after this many
                   consecutive accelerated steps with no residual improvement.
            blowup_factor: restart if ||f_k|| exceeds this multiple of the
                   *previous* residual (genuine step-over-step divergence).
            tr_factor: 信赖域系数——单次加速步长 ||x_new - x|| 上限为
                   tr_factor * ||f||(plain 步长)。挡住病态外推的有限大过冲。
            restart_omega: restart 后第一步(空窗口 reseed)的 over-relax kick
                   系数(>1)。仅在 restart 触发时用一次,破起裂极限环;稳态首
                   迭代仍用 omega(默认 1.0,plain),不损反力精度。
            reg:   Tikhonov regularization on the Anderson normal equations.
        """
        self.m = max(0, int(depth))
        self.beta = float(beta)
        self.omega = float(omega)
        self.restart_patience = int(restart_patience)
        self.blowup_factor = float(blowup_factor)
        self.tr_factor = float(tr_factor)
        self.restart_omega = float(restart_omega)
        self.reg = float(reg)
        self.reset()

    def reset(self) -> None:
        """Clear all state. Call once at the start of every load step."""
        self._X: list[np.ndarray] = []
        self._F: list[np.ndarray] = []
        self._rhist: list[float] = []
        self._stall = 0
        self._best = np.inf
        # 刚 restart 标志:下一次空窗口 reseed 用 restart_omega 做 over-relax
        # kick(仅破极限环),区别于每载荷步首迭代的 plain 播种。
        self._just_restarted = False
        # last-used mode, for diagnostics: 'plain' | 'overrelax' | 'kick' | 'anderson'
        self.last_mode = "plain"

    def _restart_window(self) -> None:
        self._X.clear()
        self._F.clear()
        self._stall = 0
        self._just_restarted = True

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
        # Blow-up 基准用 *上一迭代* 残差 _rhist[-1],不是步内全局最小 _best:
        # 峰前收敛后 _best→0,起裂残差突跳必然 > blowup_factor*_best → 几乎每步
        # restart → 起裂步退化成 plain(极限环)。改比上一步残差只在真正逐步发散
        # 时才 restart。
        prev_rn = self._rhist[-1] if self._rhist else np.inf
        if self._X and rn > self.blowup_factor * prev_rn:
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
            if self._just_restarted:
                # 破极限环:restart 后的第一步用 over-relax kick(restart_omega>1)
                # 把迭代扰动出极限环——这正是起裂步真正需要的"大跳"。仅此一步,
                # 不影响稳态。
                seed_omega = self.restart_omega
                self._just_restarted = False
                self.last_mode = "kick"
            else:
                # 每载荷步首迭代的自然播种:plain(omega 通常 1.0),稳态不 over-relax,
                # 以保稳定段反力精度(over-relax 在固定 tol 下会把解推偏)。
                seed_omega = self.omega
                self.last_mode = "plain" if self.omega == 1.0 else "overrelax"
            x_new = x + seed_omega * f
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

        # --- trust region: 限制加速步长 -------------------------------------
        # 病态最小二乘外推(dF 在裂纹突跳处近奇异)会产生 *有限大* 过冲,下面的
        # isfinite 检查拦不住。把 ||x_new - x|| 限制到 plain 步长 ||f||=rn 的
        # tr_factor 倍,沿同方向缩回。播种步 x+omega*f(omega<=1)天然在界内。
        step_norm = float(np.linalg.norm(x_new - x))
        max_step = self.tr_factor * (rn + 1e-30)
        if step_norm > max_step:
            x_new = x + (max_step / step_norm) * (x_new - x)
            self.last_mode = self.last_mode + "+tr"

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
