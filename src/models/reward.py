"""Reward functions for portfolio RL agents.

CVaRReward: -CVaR_{alpha}(R_p) + lambda * E[R_p]
SharpeReward: E[R_p] / std(R_p)
"""

from __future__ import annotations

import numpy as np


class CVaRReward:
    """CVaR-based reward function.

    Parameters
    ----------
    alpha : float
        Confidence level for CVaR (default 0.95).
    lam : float
        Trade-off between CVaR and expected return (default 0.5).
    """

    def __init__(self, alpha: float = 0.95, lam: float = 0.5) -> None:
        self.alpha = alpha
        self.lam = lam

    def __call__(self, portfolio_returns: np.ndarray) -> float:
        """Compute reward for a window of portfolio returns.

        Parameters
        ----------
        portfolio_returns : np.ndarray
            1-D array of portfolio daily returns.

        Returns
        -------
        float
            Reward value.
        """
        var = np.quantile(portfolio_returns, 1 - self.alpha)
        cvar = -portfolio_returns[portfolio_returns <= var].mean()
        expected = portfolio_returns.mean()
        return -cvar + self.lam * expected


class SharpeReward:
    """Annualized Sharpe ratio as reward (baseline).

    Parameters
    ----------
    risk_free : float
        Daily risk-free rate (default 0.0).
    """

    def __init__(self, risk_free: float = 0.0) -> None:
        self.risk_free = risk_free

    def __call__(self, portfolio_returns: np.ndarray) -> float:
        excess = portfolio_returns - self.risk_free
        if excess.std() == 0:
            return 0.0
        return float(np.sqrt(252) * excess.mean() / excess.std())
