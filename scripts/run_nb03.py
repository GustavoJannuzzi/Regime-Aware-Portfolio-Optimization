"""
run_nb03.py — NB-03: Exploratory Data Analysis (EDA)
Project: Regime-Aware Portfolio Optimization (RBFin)
Author: Gustavo (UFPR Mestrado)

Produces:
  outputs/figures/fig_cumulative_returns.{pdf,png}
  outputs/figures/fig_correlation_matrix.{pdf,png}
  outputs/figures/fig_selic_regimes.{pdf,png}
  outputs/figures/fig_return_distribution.{pdf,png}
  outputs/tables/descriptive_stats.{tex,csv}
  outputs/tables/regime_stats.{tex,csv}
  outputs/tables/stationarity_tests.{tex,csv}
  data/processed/ew_returns.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, norm
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("C:/Users/gustavo.j.siebel/Downloads/Regime-Aware Portfolio Optimization")
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
DATA_PROC = ROOT / "data" / "processed"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
prices  = pd.read_parquet(DATA_PROC / "prices.parquet")
returns = pd.read_parquet(DATA_PROC / "returns.parquet")
macro   = pd.read_parquet(DATA_PROC / "macro.parquet")

tickers = returns.columns.tolist()
print(f"  prices:  {prices.shape}")
print(f"  returns: {returns.shape}")
print(f"  macro:   {macro.shape}")
print(f"  tickers: {tickers}")

# ---------------------------------------------------------------------------
# Equal-weight portfolio returns
# ---------------------------------------------------------------------------
ew_returns: pd.Series = returns.mean(axis=1)
ew_returns.name = "ew_portfolio"
print(f"\nEqual-weight portfolio returns: {ew_returns.shape}, "
      f"mean={ew_returns.mean():.6f}, std={ew_returns.std():.6f}")

# ---------------------------------------------------------------------------
# Helper: save figure
# ---------------------------------------------------------------------------
def save_fig(name: str) -> None:
    """Save active figure as both PDF and PNG."""
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {pdf_path.name}  |  {png_path.name}")
    plt.close()

# ===========================================================================
# FIGURE 1 — Cumulative Returns
# ===========================================================================
print("\n[Figure 1] Cumulative returns...")

cum_rets = (1 + returns).cumprod()  # normalized to 1 at start

fig, ax = plt.subplots(figsize=(6.5, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))

for i, col in enumerate(tickers):
    ax.plot(cum_rets.index, cum_rets[col], linewidth=1.0,
            color=colors[i], label=col.replace(".SA", ""))

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (base = 1)")

# Legend outside right
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
          borderaxespad=0, fontsize=9)

fig.tight_layout()
save_fig("fig_cumulative_returns")

# ===========================================================================
# FIGURE 2 — Correlation Matrix
# ===========================================================================
print("\n[Figure 2] Correlation matrix...")

corr = returns.corr()
tick_labels = [t.replace(".SA", "") for t in tickers]

fig, ax = plt.subplots(figsize=(6.5, 5.2))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1,
    vmax=1,
    ax=ax,
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    annot_kws={"size": 8},
    linewidths=0.4,
    linecolor="white",
)
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)

fig.tight_layout()
save_fig("fig_correlation_matrix")

# ===========================================================================
# FIGURE 3 — SELIC Regimes
# ===========================================================================
print("\n[Figure 3] SELIC regimes...")

# Align macro to returns date range
macro_aligned = macro.reindex(prices.index)

regime_colors = {1: "red", 0: "gray", -1: "green"}
regime_labels = {1: "Rising", 0: "Stable", -1: "Falling"}

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6.5, 4), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# Top panel: SELIC rate
ax1.plot(macro_aligned.index, macro_aligned["selic"], color="#1f77b4",
         linewidth=1.2, label="SELIC (%)")
ax1.set_ylabel("SELIC Rate (%)")

# Bottom panel: regime colored bars
dates = macro_aligned.index
regime = macro_aligned["selic_regime"].values

# Draw colored spans for each regime segment
prev_regime = regime[0]
start_idx = 0
for i in range(1, len(regime)):
    if regime[i] != prev_regime or i == len(regime) - 1:
        end_idx = i if regime[i] != prev_regime else len(regime) - 1
        color = regime_colors.get(int(prev_regime), "gray")
        ax2.axvspan(dates[start_idx], dates[end_idx],
                    alpha=0.7, color=color, linewidth=0)
        start_idx = i
        prev_regime = regime[i]

ax2.set_yticks([])
ax2.set_ylabel("Regime", fontsize=9)
ax2.set_xlabel("Date")

# Legend patches
patches = [
    mpatches.Patch(color=regime_colors[k], label=regime_labels[k])
    for k in [1, 0, -1]
]
ax2.legend(handles=patches, loc="upper right", fontsize=8,
           ncol=3, bbox_to_anchor=(1.0, 1.6))

fig.tight_layout()
save_fig("fig_selic_regimes")

# ===========================================================================
# FIGURE 4 — Return Distribution with CVaR
# ===========================================================================
print("\n[Figure 4] Return distribution...")

alpha = 0.05
ew_sorted = np.sort(ew_returns.dropna().values)
var_05 = np.percentile(ew_sorted, alpha * 100)
cvar_05 = ew_sorted[ew_sorted <= var_05].mean()

mu_ew  = ew_returns.mean()
std_ew = ew_returns.std()
x_range = np.linspace(ew_returns.min() - 0.005, ew_returns.max() + 0.005, 500)

fig, ax = plt.subplots(figsize=(6.5, 4))

# Histogram + KDE
ax.hist(ew_returns.dropna(), bins=80, density=True, alpha=0.35,
        color="#4878d0", label="Daily returns")
kde = stats.gaussian_kde(ew_returns.dropna())
ax.plot(x_range, kde(x_range), color="#4878d0", linewidth=1.5, label="KDE")

# Normal overlay
ax.plot(x_range, norm.pdf(x_range, mu_ew, std_ew),
        color="orange", linewidth=1.5, linestyle="--", label="Normal fit")

# CVaR line
ax.axvline(cvar_05, color="crimson", linewidth=1.5,
           linestyle="--", label=f"CVaR$_{{0.95}}$ = {cvar_05:.4f}")

ax.set_xlabel("Daily Log Return (equal-weight portfolio)")
ax.set_ylabel("Density")
ax.legend(fontsize=9)

fig.tight_layout()
save_fig("fig_return_distribution")

# ===========================================================================
# TABLE 1 — Descriptive Statistics
# ===========================================================================
print("\n[Table 1] Descriptive statistics...")


def max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown from cumulative return series."""
    cumret = (1 + series.dropna()).cumprod()
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max
    return float(drawdown.min())


def cvar_hist(series: pd.Series, alpha: float = 0.05) -> float:
    """Historical CVaR at level alpha (expected shortfall)."""
    s = series.dropna().values
    var = np.percentile(s, alpha * 100)
    return float(s[s <= var].mean())


rows = []
for ticker in tickers:
    s = returns[ticker].dropna()
    jb_stat, jb_pval = jarque_bera(s)
    rows.append({
        "Ticker": ticker.replace(".SA", ""),
        "Mean (ann.)": s.mean() * 252,
        "Std (ann.)": s.std() * np.sqrt(252),
        "Skewness": float(s.skew()),
        "Kurtosis": float(s.kurt()),   # excess kurtosis
        "JB p-value": float(jb_pval),
        "CVaR$_{0.95}$": cvar_hist(s, 0.05),
        "Max Drawdown": max_drawdown(s),
    })

desc_df = pd.DataFrame(rows).set_index("Ticker")

# Save CSV
desc_df.to_csv(TAB_DIR / "descriptive_stats.csv")
print(f"  Saved: descriptive_stats.csv")

# Save LaTeX — booktabs
desc_df.to_latex(
    buf=str(TAB_DIR / "descriptive_stats.tex"),
    float_format="%.4f",
    escape=False,
    caption="Descriptive Statistics of B3 Assets (2015--2024)",
    label="tab:descriptive_stats",
    bold_rows=False,
    column_format="l" + "r" * len(desc_df.columns),
)
print(f"  Saved: descriptive_stats.tex")
print(desc_df.to_string())

# ===========================================================================
# TABLE 2 — Regime Statistics
# ===========================================================================
print("\n[Table 2] Regime statistics...")

# Align macro and ew_returns on common dates
common_idx = ew_returns.index.intersection(macro_aligned.index)
ew_common  = ew_returns.loc[common_idx]
mac_common = macro_aligned.loc[common_idx]

regime_map = {1: "Rising", 0: "Stable", -1: "Falling"}
reg_rows = []

for reg_code, reg_name in regime_map.items():
    mask = mac_common["selic_regime"] == reg_code
    sub   = ew_common.loc[mask].dropna()
    selic_sub = mac_common.loc[mask, "selic"].dropna()

    n_days = int(mask.sum())
    mean_r = float(sub.mean())
    std_r  = float(sub.std())
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else np.nan
    cvar   = cvar_hist(sub, 0.05)
    avg_selic = float(selic_sub.mean())

    reg_rows.append({
        "Regime": reg_name,
        "N Days": n_days,
        "Mean Return (daily)": mean_r,
        "Std (daily)": std_r,
        "Sharpe (ann.)": sharpe,
        "CVaR$_{0.95}$": cvar,
        "Avg SELIC (\\%)": avg_selic,
    })

reg_df = pd.DataFrame(reg_rows).set_index("Regime")

reg_df.to_csv(TAB_DIR / "regime_stats.csv")
print(f"  Saved: regime_stats.csv")

reg_df.to_latex(
    buf=str(TAB_DIR / "regime_stats.tex"),
    float_format="%.4f",
    escape=False,
    caption="Equal-Weight Portfolio Statistics by SELIC Regime (2015--2024)",
    label="tab:regime_stats",
    bold_rows=False,
    column_format="l" + "r" * len(reg_df.columns),
)
print(f"  Saved: regime_stats.tex")
print(reg_df.to_string())

# ===========================================================================
# TABLE 3 — Stationarity Tests (ADF)
# ===========================================================================
print("\n[Table 3] Stationarity tests (ADF)...")

adf_rows = []
for ticker in tickers:
    s = returns[ticker].dropna()
    result = adfuller(s, autolag="AIC")
    adf_stat, pval, lags_used, n_obs = result[0], result[1], result[2], result[3]
    crit = result[4]
    adf_rows.append({
        "Ticker": ticker.replace(".SA", ""),
        "ADF Stat": float(adf_stat),
        "p-value": float(pval),
        "Lags": int(lags_used),
        "N Obs": int(n_obs),
        "Crit 1\\%": float(crit["1%"]),
        "Crit 5\\%": float(crit["5%"]),
        "Crit 10\\%": float(crit["10%"]),
        "Stationary": "Yes" if pval < 0.05 else "No",
    })

adf_df = pd.DataFrame(adf_rows).set_index("Ticker")

print(adf_df.to_string())

adf_df.to_csv(TAB_DIR / "stationarity_tests.csv")
print(f"  Saved: stationarity_tests.csv")

# For LaTeX, drop "Stationary" column (text) to keep numeric formatting
adf_df_tex = adf_df.drop(columns=["Stationary", "N Obs", "Lags"])
adf_df_tex.to_latex(
    buf=str(TAB_DIR / "stationarity_tests.tex"),
    float_format="%.4f",
    escape=False,
    caption="Augmented Dickey--Fuller Stationarity Tests on Log Returns",
    label="tab:stationarity",
    bold_rows=False,
    column_format="l" + "r" * len(adf_df_tex.columns),
)
print(f"  Saved: stationarity_tests.tex")

# ===========================================================================
# Save equal-weight portfolio returns
# ===========================================================================
print("\n[Data] Saving ew_returns.parquet...")
ew_returns.to_frame().to_parquet(DATA_PROC / "ew_returns.parquet")
print(f"  Saved: data/processed/ew_returns.parquet  shape={ew_returns.shape}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print("NB-03 EDA — OUTPUTS SUMMARY")
print("=" * 60)

figures = list(FIG_DIR.glob("fig_*.p*"))
tables  = list(TAB_DIR.glob("*.tex")) + list(TAB_DIR.glob("*.csv"))

print(f"\nFigures ({len(figures)}):")
for f in sorted(figures):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:45s}  {size_kb:6.1f} KB")

print(f"\nTables ({len(tables)}):")
for t in sorted(tables):
    size_kb = t.stat().st_size / 1024
    print(f"  {t.name:45s}  {size_kb:6.1f} KB")

print(f"\nData:")
ew_path = DATA_PROC / "ew_returns.parquet"
print(f"  ew_returns.parquet  shape={ew_returns.shape}  "
      f"size={ew_path.stat().st_size/1024:.1f} KB")

print("\nDescriptive stats (annualized):")
print(desc_df[["Mean (ann.)", "Std (ann.)", "Skewness", "CVaR$_{0.95}$", "Max Drawdown"]].round(4).to_string())

print("\nRegime stats:")
print(reg_df.to_string())

print("\nAll outputs created successfully.")
sys.exit(0)
