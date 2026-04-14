"""Walk-forward validation protocol.

Train window: 252 business days
Test window:  63 business days (quarterly rebalancing)
No look-ahead bias guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    train_start: list[str]
    train_end: list[str]
    test_start: list[str]
    test_end: list[str]
    portfolio_returns: pd.Series
    weights: pd.DataFrame


def walk_forward_splits(
    returns: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 63,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate (train, test) splits for walk-forward validation.

    Parameters
    ----------
    returns : pd.DataFrame
        Full returns DataFrame.
    train_window : int
        Number of training days.
    test_window : int
        Number of test days.

    Yields
    ------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_returns, test_returns) for each fold.
    """
    n = len(returns)
    splits = []
    start = 0
    while start + train_window + test_window <= n:
        train = returns.iloc[start : start + train_window]
        test = returns.iloc[start + train_window : start + train_window + test_window]
        splits.append((train, test))
        start += test_window
    return splits


def evaluate_strategy(
    strategy,
    returns: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 63,
) -> WalkForwardResult:
    """Run walk-forward evaluation for a PortfolioStrategy.

    Parameters
    ----------
    strategy : PortfolioStrategy
        Any object with .fit(train) and .predict() interface.
    returns : pd.DataFrame
        Full returns DataFrame.

    Returns
    -------
    WalkForwardResult
        Aggregated out-of-sample returns and weights.
    """
    splits = walk_forward_splits(returns, train_window, test_window)
    all_returns = []
    all_weights = []
    metadata = {"train_start": [], "train_end": [], "test_start": [], "test_end": []}

    for train, test in splits:
        strategy.fit(train)
        w = strategy.predict()
        port_ret = test.values @ w
        all_returns.append(pd.Series(port_ret, index=test.index))
        all_weights.append(pd.DataFrame([w] * len(test), index=test.index, columns=returns.columns))
        metadata["train_start"].append(str(train.index[0].date()))
        metadata["train_end"].append(str(train.index[-1].date()))
        metadata["test_start"].append(str(test.index[0].date()))
        metadata["test_end"].append(str(test.index[-1].date()))

    return WalkForwardResult(
        portfolio_returns=pd.concat(all_returns),
        weights=pd.concat(all_weights),
        **metadata,
    )
