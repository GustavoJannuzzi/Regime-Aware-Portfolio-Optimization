"""B3 price data loader via yfinance.

Coleta preços ajustados dos ativos do Ibovespa com filtro de liquidez mínima.
Período padrão: 2010-01-01 a 2024-12-31.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from pathlib import Path


DATA_DIR = Path(__file__).parents[2] / "data"


def download_b3_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    min_avg_volume: float = 1e7,
) -> pd.DataFrame:
    """Download adjusted close prices for B3 tickers.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance tickers (e.g. ['PETR4.SA', 'VALE3.SA']).
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    min_avg_volume : float
        Minimum average daily volume in BRL to keep an asset.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted close prices, columns = tickers.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)
    prices = raw["Close"]
    # Filter by liquidity
    avg_vol = raw["Volume"].mean()
    liquid = avg_vol[avg_vol >= min_avg_volume].index.tolist()
    return prices[liquid].dropna(how="all")


def load_processed_prices() -> pd.DataFrame:
    """Load preprocessed prices from data/processed/prices.parquet."""
    path = DATA_DIR / "processed" / "prices.parquet"
    return pd.read_parquet(path)
