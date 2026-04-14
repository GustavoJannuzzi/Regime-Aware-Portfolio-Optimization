"""
run_nb04.py — Benchmark Evaluation with Walk-Forward Protocol

Implements EqualWeight (1/N), MinimumVariance (MVP), and RiskParity (ERC)
strategies evaluated out-of-sample via walk-forward cross-validation.

Outputs:
    outputs/tables/benchmark_results.tex   — LaTeX booktabs table
    outputs/tables/benchmark_results.csv   — CSV version of results
    data/processed/benchmark_returns.parquet  — OOS daily returns per strategy
    data/processed/benchmark_weights.parquet  — weights at each rebalancing date

Walk-forward protocol:
    Train window: 252 business days
    Test window:  63 business days (quarterly rebalancing)
    No look-ahead bias.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
TRAIN_WINDOW: int = 252
TEST_WINDOW: int = 63
TRADING_DAYS: int = 252
CVAR_LEVEL: float = 0.05      # 5th percentile -> CVaR 95%

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = PROJECT_ROOT / "data" / "processed"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)


# ── Strategy implementations ──────────────────────────────────────────────────

def equal_weight(train_returns: pd.DataFrame) -> np.ndarray:
    """Return equal (1/N) weights — no fitting required."""
    n = train_returns.shape[1]
    return np.ones(n) / n


def minimum_variance(train_returns: pd.DataFrame) -> np.ndarray:
    """Minimum-variance portfolio (long-only, fully-invested).

    Parameters
    ----------
    train_returns : DataFrame of shape (T, N)

    Returns
    -------
    weights : ndarray of shape (N,)
    """
    n = train_returns.shape[1]
    cov = train_returns.cov().values

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        portfolio_variance,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        # fallback to equal weight
        return equal_weight(train_returns)
    w = result.x
    w = np.clip(w, 0.0, 1.0)
    return w / w.sum()


def risk_parity(train_returns: pd.DataFrame) -> np.ndarray:
    """Equal Risk Contribution (ERC) portfolio.

    Parameters
    ----------
    train_returns : DataFrame of shape (T, N)

    Returns
    -------
    weights : ndarray of shape (N,)
    """
    n = train_returns.shape[1]
    cov = train_returns.cov().values

    def objective(w: np.ndarray) -> float:
        sigma = np.sqrt(w @ cov @ w)
        if sigma < 1e-12:
            return 0.0
        rc = (w * (cov @ w)) / sigma        # risk contributions
        target = sigma / n                  # equal target
        return float(np.sum((rc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(1e-6, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        return equal_weight(train_returns)
    w = np.clip(result.x, 1e-6, 1.0)
    return w / w.sum()


STRATEGIES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "EqualWeight": equal_weight,
    "MVP": minimum_variance,
    "RiskParity": risk_parity,
}


# ── Walk-forward engine ───────────────────────────────────────────────────────

def generate_folds(
    n_obs: int, train_window: int = TRAIN_WINDOW, test_window: int = TEST_WINDOW
) -> list[tuple[int, int, int, int]]:
    """Generate (train_start, train_end, test_start, test_end) index tuples."""
    folds: list[tuple[int, int, int, int]] = []
    train_start = 0
    while True:
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window
        if test_end > n_obs:
            break
        folds.append((train_start, train_end, test_start, test_end))
        train_start += test_window   # rolling / expanding? rolling chosen here
    return folds


def walk_forward(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward evaluation for all strategies.

    Returns
    -------
    oos_returns : DataFrame (date, strategy)
    weights_df  : DataFrame (date, (strategy, ticker))
    """
    dates = returns.index
    n_obs = len(returns)
    tickers = returns.columns.tolist()
    folds = generate_folds(n_obs)

    print(f"\n{'='*60}")
    print(f"Walk-Forward Protocol")
    print(f"  Train window : {TRAIN_WINDOW} days")
    print(f"  Test window  : {TEST_WINDOW} days")
    print(f"  Total folds  : {len(folds)}")
    print(f"{'='*60}\n")

    # accumulators
    oos_records: list[pd.Series] = []
    weight_records: list[pd.DataFrame] = []

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        train_data = returns.iloc[tr_s:tr_e]
        test_data = returns.iloc[te_s:te_e]

        train_start_dt = dates[tr_s]
        train_end_dt = dates[tr_e - 1]
        test_start_dt = dates[te_s]
        test_end_dt = dates[te_e - 1]

        print(
            f"Fold {fold_idx + 1:02d}/{len(folds)} | "
            f"Train: {train_start_dt.date()} to {train_end_dt.date()} | "
            f"Test : {test_start_dt.date()} to {test_end_dt.date()}"
        )

        fold_weights: dict[str, np.ndarray] = {}
        for strat_name, strat_fn in STRATEGIES.items():
            w = strat_fn(train_data)
            fold_weights[strat_name] = w

            # OOS returns for this strategy
            oos_ret = test_data.values @ w   # shape (TEST_WINDOW,)
            for i, dt in enumerate(test_data.index):
                oos_records.append(
                    pd.Series({"date": dt, "strategy": strat_name, "ret": oos_ret[i]})
                )

        # Store weights at rebalancing date (first day of test period)
        rebal_date = test_start_dt
        row_idx = pd.MultiIndex.from_tuples(
            [(strat, tick) for strat in STRATEGIES for tick in tickers],
            names=["strategy", "ticker"],
        )
        row_vals = np.concatenate([fold_weights[s] for s in STRATEGIES])
        w_series = pd.Series(row_vals, index=row_idx, name=rebal_date)
        weight_records.append(w_series)

    # Build OOS returns DataFrame
    oos_df = pd.DataFrame(oos_records)
    oos_returns = oos_df.pivot_table(index="date", columns="strategy", values="ret")
    oos_returns = oos_returns[list(STRATEGIES.keys())]  # canonical column order
    oos_returns.index.name = "date"

    # Build weights DataFrame
    weights_df = pd.DataFrame(weight_records)
    weights_df.index.name = "rebal_date"
    weights_df.columns = pd.MultiIndex.from_tuples(
        weights_df.columns.tolist(), names=["strategy", "ticker"]
    )

    return oos_returns, weights_df


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(ret_series: pd.Series, weights_df: pd.DataFrame, strat_name: str) -> dict:
    """Compute all performance metrics for a single OOS return series."""
    r = ret_series.dropna().values

    # Annualised return and volatility
    ann_ret = float(np.mean(r) * TRADING_DAYS)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))

    # Sharpe ratio
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Sortino ratio
    downside = r[r < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    ann_downside = downside_std * np.sqrt(TRADING_DAYS)
    sortino = ann_ret / ann_downside if ann_downside > 0 else np.nan

    # CVaR 95%
    threshold = np.quantile(r, CVAR_LEVEL)
    tail = r[r <= threshold]
    cvar = float(-np.mean(tail)) if len(tail) > 0 else np.nan

    # Max drawdown
    wealth = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (wealth - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Turnover (mean absolute change in weights across rebalancing dates)
    if strat_name in weights_df.columns.get_level_values("strategy"):
        w = weights_df[strat_name].values.astype(float)   # shape (n_folds, N)
        diffs = np.abs(np.diff(w, axis=0))
        turnover = float(np.mean(diffs.sum(axis=1)))
    else:
        turnover = np.nan

    return {
        "Ann.Return": ann_ret,
        "Ann.Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "CVaR_0.95": cvar,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "Turnover": turnover,
    }


# ── LaTeX export ──────────────────────────────────────────────────────────────

def to_latex(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Export metrics DataFrame to a publication-ready booktabs LaTeX table."""
    float_fmt = "{:.4f}".format

    # Format each cell
    formatted = metrics_df.map(
        lambda x: float_fmt(x) if isinstance(x, float) and not np.isnan(x) else "—"
    )

    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Out-of-Sample Performance of Benchmark Strategies (2016--2024)}",
        r"  \label{tab:benchmark_results}",
        r"  \begin{tabular}{l" + "r" * len(metrics_df.columns) + r"}",
        r"    \toprule",
    ]

    # Header row
    header = "    Strategy & " + " & ".join(metrics_df.columns.tolist()) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"    \midrule")

    # Data rows
    for strat, row in formatted.iterrows():
        data_row = f"    {strat} & " + " & ".join(row.values.tolist()) + r" \\"
        latex_lines.append(data_row)

    latex_lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"\nLaTeX table saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading returns data...")
    returns = pd.read_parquet(DATA_PROC / "returns.parquet")
    print(f"  Shape: {returns.shape} | {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Tickers: {returns.columns.tolist()}")

    # Walk-forward evaluation
    oos_returns, weights_df = walk_forward(returns)

    print(f"\nOOS returns shape : {oos_returns.shape}")
    print(f"Weights shape     : {weights_df.shape}")

    # Compute metrics for each strategy
    metrics: dict[str, dict] = {}
    for strat in STRATEGIES:
        metrics[strat] = compute_metrics(oos_returns[strat], weights_df, strat)

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "Strategy"

    # Print final metrics table
    print(f"\n{'='*80}")
    print("FINAL METRICS TABLE (Out-of-Sample)")
    print(f"{'='*80}")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*80}\n")

    # Save outputs
    # 1. Benchmark returns parquet
    oos_path = DATA_PROC / "benchmark_returns.parquet"
    oos_returns.to_parquet(oos_path)
    print(f"Benchmark returns saved : {oos_path}")

    # 2. Benchmark weights parquet
    w_path = DATA_PROC / "benchmark_weights.parquet"
    weights_df.to_parquet(w_path)
    print(f"Benchmark weights saved : {w_path}")

    # 3. CSV table
    csv_path = OUT_TABLES / "benchmark_results.csv"
    metrics_df.to_csv(csv_path, float_format="%.4f")
    print(f"CSV table saved         : {csv_path}")

    # 4. LaTeX table
    tex_path = OUT_TABLES / "benchmark_results.tex"
    to_latex(metrics_df, tex_path)

    print("\nNB-04 complete.")


if __name__ == "__main__":
    main()
