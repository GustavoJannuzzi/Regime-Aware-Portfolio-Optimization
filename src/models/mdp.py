"""Portfolio MDP environment (gymnasium.Env).

State vector:
  - Normalized returns (rolling window T=20)
  - Realized volatility
  - Momentum
  - SELIC direction (one-hot, 3 dims)
  - CDS (normalized)
  - Credit spread (normalized)
  - IVOL-BR (normalized)

Action: portfolio weights w in Delta^N (simplex).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.models.reward import CVaRReward


class PortfolioMDP(gym.Env):
    """Continuous portfolio allocation environment.

    Parameters
    ----------
    returns : np.ndarray
        Asset returns array, shape (T, N).
    macro : np.ndarray
        Macro features array, shape (T, M).
    reward_fn : callable
        Reward function (default: CVaRReward).
    window : int
        Lookback window for state construction (default: 20).
    rebal_freq : int
        Rebalancing frequency in days (default: 1).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,
        macro: np.ndarray,
        reward_fn=None,
        window: int = 20,
        rebal_freq: int = 1,
    ) -> None:
        super().__init__()
        self.returns = returns
        self.macro = macro
        self.reward_fn = reward_fn or CVaRReward()
        self.window = window
        self.rebal_freq = rebal_freq

        T, N = returns.shape
        self.n_assets = N
        self._T = T

        # Observation: returns (N*window) + vol (N) + momentum (N) + macro (M) + selic_onehot (3)
        macro_dim = macro.shape[1] if macro.ndim > 1 else 1
        obs_dim = N * window + N + N + macro_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N,), dtype=np.float32
        )

        self._t = window

    def _get_obs(self) -> np.ndarray:
        t = self._t
        ret_window = self.returns[t - self.window : t].flatten()
        vol = self.returns[t - self.window : t].std(axis=0)
        momentum = self.returns[t - self.window : t].sum(axis=0)
        macro_t = self.macro[t].flatten() if self.macro.ndim > 1 else np.array([self.macro[t]])
        return np.concatenate([ret_window, vol, momentum, macro_t]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.window
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Project action to simplex
        weights = np.abs(action)
        weights = weights / (weights.sum() + 1e-8)

        port_ret = self.returns[self._t] @ weights
        reward = float(self.reward_fn(self.returns[self._t - self.window : self._t] @ weights))

        self._t += self.rebal_freq
        terminated = self._t >= self._T
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape)
        return obs, reward, terminated, False, {"portfolio_return": port_ret}
