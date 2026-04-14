"""Performance metrics for portfolio evaluation.

Implements: Sharpe, Sortino, CVaR, MDD, Calmar, Turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, ann: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free
    return float(np.sqrt(ann) * excess.mean() / excess.std()) if excess.std() != 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, ann: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    excess = returns - risk_free
    downside = excess[excess < 0].std()
    return float(np.sqrt(ann) * excess.mean() / downside) if downside != 0 else 0.0


def cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Historical CVaR (Expected Shortfall) at confidence level alpha."""
    var = returns.quantile(1 - alpha)
    return float(-returns[returns <= var].mean())


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from cumulative return series."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def calmar_ratio(returns: pd.Series, ann: int = 252) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    ann_ret = returns.mean() * ann
    mdd = abs(max_drawdown(returns))
    return float(ann_ret / mdd) if mdd != 0 else 0.0


def portfolio_turnover(weights: pd.DataFrame) -> float:
    """Average daily turnover across rebalancing events."""
    return float(weights.diff().abs().sum(axis=1).mean())


def compute_all_metrics(
    returns: pd.Series,
    weights: pd.DataFrame | None = None,
    alpha: float = 0.95,
) -> dict:
    """Compute all performance metrics at once.

    Returns
    -------
    dict
        Keys: sharpe, sortino, cvar, mdd, calmar, turnover (if weights given).
    """
    out = {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "cvar": cvar(returns, alpha),
        "mdd": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
    }
    if weights is not None:
        out["turnover"] = portfolio_turnover(weights)
    return out
