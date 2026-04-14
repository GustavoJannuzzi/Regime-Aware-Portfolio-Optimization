"""
run_nb02.py — Data pipeline for RBFin Offline RL project.

Collects stock prices (B3) and macro data (BCB SGS), computes log returns,
applies SELIC regime labeling, and saves all processed outputs.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from bcb import sgs

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
    "WEGE3.SA", "RENT3.SA", "SUZB3.SA", "B3SA3.SA", "BBAS3.SA",
]
START = "2015-01-01"
END   = "2024-12-31"
MISSING_THRESHOLD = 0.10  # drop tickers with >10% missing

# SELIC regime thresholds (annualised percentage points over 63-day window)
REGIME_UP   =  0.25
REGIME_DOWN = -0.25


# ---------------------------------------------------------------------------
# Step 1 — Stock prices & log returns
# ---------------------------------------------------------------------------
def download_prices() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Download adjusted close prices and compute log returns."""
    print("\n[1] Downloading stock prices via yfinance...")
    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single ticker fallback
        prices = raw[["Close"]].copy()
        prices.columns = TICKERS

    print(f"  Raw shape: {prices.shape}")

    # Drop tickers with too many missing values
    missing_frac = prices.isna().mean()
    bad_tickers = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()
    if bad_tickers:
        print(f"  Dropping tickers (>{MISSING_THRESHOLD*100:.0f}% missing): {bad_tickers}")
        prices.drop(columns=bad_tickers, inplace=True)

    good_tickers = prices.columns.tolist()
    print(f"  Kept tickers ({len(good_tickers)}): {good_tickers}")

    # Forward-fill then drop any remaining NaNs at the edges
    prices.ffill(inplace=True)
    prices.dropna(inplace=True)
    prices.index = pd.DatetimeIndex(prices.index)
    prices.index.name = "date"

    print(f"  Prices shape after cleaning: {prices.shape}")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  Log returns shape: {log_returns.shape}")

    return prices, log_returns, good_tickers


# ---------------------------------------------------------------------------
# Step 2 — Macro data via BCB SGS
# ---------------------------------------------------------------------------
def download_macro(stock_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Fetch SELIC, IPCA, and credit spread from BCB SGS and align to stock dates."""
    print("\n[2] Downloading macro data from BCB SGS...")

    series_map = {
        "selic": 432,    # SELIC meta (% a.a.)
        "ipca":  433,    # IPCA mensal (%)
        "spread_credito": 20786,  # Spread crédito livre PF (p.p.)
    }

    macro_frames: list[pd.DataFrame] = []

    for name, code in series_map.items():
        print(f"  Fetching series {code} ({name})...")
        fetched = False
        for attempt in range(1, 4):  # up to 3 attempts
            try:
                s = sgs.get(code, start=START, end=END)
                # sgs.get may return a DataFrame or Series
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s.name = name
                s.index = pd.DatetimeIndex(s.index)
                print(f"    -> {len(s)} obs  [{s.index[0].date()} - {s.index[-1].date()}]")
                macro_frames.append(s)
                fetched = True
                break
            except Exception as e:
                print(f"    [WARN] Attempt {attempt}/3 failed for {name} (code {code}): {e}")
                import time; time.sleep(5)
        if not fetched:
            print(f"    [ERROR] Could not fetch {name} after 3 attempts — skipping.")

    if not macro_frames:
        raise RuntimeError("No macro series could be fetched from BCB SGS.")

    macro_raw = pd.concat(macro_frames, axis=1)

    # Build a continuous daily date range (calendar days) covering full period,
    # forward-fill within that range, then select business days aligned to stock dates.
    full_idx = pd.date_range(start=START, end=END, freq="D")
    macro_daily = macro_raw.reindex(full_idx).ffill()
    # Align to actual stock trading dates
    macro_bd = macro_daily.reindex(stock_dates)
    macro_bd.index.name = "date"

    print(f"  Macro DataFrame shape (aligned to stock dates): {macro_bd.shape}")
    print(f"  Columns: {macro_bd.columns.tolist()}")
    print(f"  Missing values:\n{macro_bd.isna().sum()}")

    return macro_bd


# ---------------------------------------------------------------------------
# Step 3 — SELIC regime labeling
# ---------------------------------------------------------------------------
def add_selic_regime(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 63-day rolling difference of SELIC and label regimes:
      +1 = rising  (delta > REGIME_UP)
       0 = stable
      -1 = falling (delta < REGIME_DOWN)
    One-hot encode as selic_up, selic_stable, selic_down.
    """
    print("\n[3] Computing SELIC regime labels...")

    if "selic" not in macro.columns:
        print("  [WARN] SELIC column not found; skipping regime labeling.")
        macro["selic_up"] = 0
        macro["selic_stable"] = 1
        macro["selic_down"] = 0
        return macro

    selic_delta = macro["selic"].diff(63)

    regime = pd.Series(0, index=macro.index, dtype=int, name="selic_regime")
    regime[selic_delta > REGIME_UP]   =  1
    regime[selic_delta < REGIME_DOWN] = -1

    macro["selic_regime"] = regime
    macro["selic_up"]     = (regime ==  1).astype(int)
    macro["selic_stable"] = (regime ==  0).astype(int)
    macro["selic_down"]   = (regime == -1).astype(int)

    counts = regime.value_counts().sort_index()
    count_labels = {-1: "falling", 0: "stable", 1: "rising"}
    print("  Regime counts:")
    for k, v in counts.items():
        label = count_labels.get(int(k), str(k))
        print(f"    {label:>10}: {v:5d} days  ({v/len(regime)*100:.1f}%)")

    return macro


# ---------------------------------------------------------------------------
# Step 4 — Save outputs
# ---------------------------------------------------------------------------
def save_outputs(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    macro: pd.DataFrame,
    good_tickers: list[str],
) -> None:
    """Save parquet files and summary JSON."""
    print("\n[4] Saving processed data...")

    prices_path = PROCESSED_DIR / "prices.parquet"
    returns_path = PROCESSED_DIR / "returns.parquet"
    macro_path = PROCESSED_DIR / "macro.parquet"

    prices.to_parquet(prices_path)
    log_returns.to_parquet(returns_path)
    macro.to_parquet(macro_path)

    print(f"  Saved: {prices_path}")
    print(f"  Saved: {returns_path}")
    print(f"  Saved: {macro_path}")

    # Missing % per ticker in returns
    missing_pct = {
        t: float(log_returns[t].isna().mean())
        for t in good_tickers
        if t in log_returns.columns
    }

    # SELIC regime counts
    selic_counts: dict = {}
    if "selic_regime" in macro.columns:
        vc = macro["selic_regime"].value_counts().to_dict()
        selic_counts = {
            "rising":  int(vc.get(1, 0)),
            "stable":  int(vc.get(0, 0)),
            "falling": int(vc.get(-1, 0)),
        }

    summary = {
        "tickers": good_tickers,
        "date_range": {
            "start": str(prices.index[0].date()),
            "end":   str(prices.index[-1].date()),
        },
        "n_obs": int(len(prices)),
        "n_tickers": int(len(good_tickers)),
        "prices_shape": list(prices.shape),
        "returns_shape": list(log_returns.shape),
        "macro_shape": list(macro.shape),
        "macro_columns": macro.columns.tolist(),
        "missing_pct_returns": missing_pct,
        "selic_regime_counts": selic_counts,
    }

    summary_path = PROCESSED_DIR / "data_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("NB-02: Data Pipeline")
    print(f"Period : {START} to {END}")
    print(f"Tickers: {TICKERS}")
    print(f"Output : {PROCESSED_DIR}")
    print("=" * 60)

    prices, log_returns, good_tickers = download_prices()
    macro = download_macro(prices.index)
    macro = add_selic_regime(macro)
    save_outputs(prices, log_returns, macro, good_tickers)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print(f"  Prices     : {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"  Log returns: {log_returns.shape[0]} days x {log_returns.shape[1]} tickers")
    print(f"  Macro      : {macro.shape[0]} days x {macro.shape[1]} columns")
    print("=" * 60)


if __name__ == "__main__":
    main()
