"""Macroeconomic data collection via BCB SGS API.

Séries coletadas:
  432  — Meta SELIC
  433  — IPCA mensal
  20786 — Spread crédito livre PF
"""

from __future__ import annotations

import pandas as pd
from bcb import sgs


BCB_SERIES = {
    "selic": 432,
    "ipca": 433,
    "credit_spread_pf": 20786,
}


def get_bcb_series(
    code: int,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
) -> pd.Series:
    """Fetch a single BCB SGS series.

    Parameters
    ----------
    code : int
        BCB SGS series code.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.Series
        Time series with daily frequency (forward-filled).
    """
    series = sgs.get(code, start=start, end=end)
    return series.squeeze()


def get_all_macro(
    start: str = "2010-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Collect all macro series and return as a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: selic, ipca, credit_spread_pf — daily, forward-filled.
    """
    frames = {}
    for name, code in BCB_SERIES.items():
        frames[name] = get_bcb_series(code, start=start, end=end)
    df = pd.DataFrame(frames)
    return df.resample("B").ffill()


def label_selic_regime(
    selic: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Classify SELIC direction as +1 (rising), 0 (stable), -1 (falling).

    Parameters
    ----------
    selic : pd.Series
        SELIC rate series.
    window : int
        Rolling window (business days) to compute trend.

    Returns
    -------
    pd.Series
        Integer series: +1 / 0 / -1.
    """
    delta = selic.diff(window)
    regime = pd.cut(
        delta,
        bins=[-float("inf"), -0.25, 0.25, float("inf")],
        labels=[-1, 0, 1],
    ).astype(int)
    return regime
