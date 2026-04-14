"""Data cleaning utilities for B3 prices and macro series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def winsorize_returns(
    returns: pd.DataFrame,
    lower: float = 0.005,
    upper: float = 0.995,
) -> pd.DataFrame:
    """Winsorize extreme returns at given quantiles."""
    return returns.clip(
        lower=returns.quantile(lower),
        upper=returns.quantile(upper),
        axis=1,
    )


def align_dates(*series: pd.DataFrame | pd.Series) -> tuple:
    """Align multiple DataFrames/Series to their common date index."""
    common_idx = series[0].index
    for s in series[1:]:
        common_idx = common_idx.intersection(s.index)
    return tuple(s.loc[common_idx] for s in series)
