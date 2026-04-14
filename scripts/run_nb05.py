"""
run_nb05.py — MDP Formalization and Offline Dataset Construction (NB-05)

Builds the offline RL dataset D = {(s_t, a_t, r_t, s_{t+1})} for the
CQL agent training. Uses MinimumVariance weights as the behavioral policy
and CVaR-based reward.

State vector (225 dims):
  - returns[t-20:t] normalized (200 dims)
  - realized_vol[t]             (10 dims)
  - momentum[t]                 (10 dims)
  - cds_norm[t]                 (1 dim, SELIC proxy)
  - spread_norm[t]              (1 dim)
  - selic_up[t]                 (1 dim)
  - selic_stable[t]             (1 dim)
  - selic_down[t]               (1 dim)
"""

from __future__ import annotations

import sys
import io
import warnings
from pathlib import Path
from typing import Tuple

# Force UTF-8 output on Windows to handle special characters in print
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)

LOOKBACK: int = 20          # short-term lookback for state
MVP_WINDOW: int = 252       # lookback for MVP weight computation
NORM_WINDOW: int = 252      # rolling z-score window
ALPHA: float = 0.95         # CVaR confidence level
LAMBDA_: float = 0.5        # reward blending coefficient
N_ASSETS: int = 10
STATE_DIM: int = 200 + 10 + 10 + 1 + 1 + 3   # = 225
TRAIN_END: str = "2021-12-31"
EVAL_START: str = "2022-01-01"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("NB-05 — MDP Formalization & Offline Dataset Construction")
print("=" * 60)

returns_df: pd.DataFrame = pd.read_parquet(DATA_PROCESSED / "returns.parquet")
macro_df: pd.DataFrame = pd.read_parquet(DATA_PROCESSED / "macro.parquet")

# Align macro to returns index (returns starts one day later)
macro_df = macro_df.reindex(returns_df.index)

print(f"\nReturns shape : {returns_df.shape}  [{returns_df.index[0].date()} – {returns_df.index[-1].date()}]")
print(f"Macro shape   : {macro_df.shape}  [{macro_df.index[0].date()} – {macro_df.index[-1].date()}]")

ret_vals: np.ndarray = returns_df.values          # (T, 10)
dates: pd.DatetimeIndex = returns_df.index
T: int = len(ret_vals)

# ---------------------------------------------------------------------------
# Rolling z-score normalizer
# ---------------------------------------------------------------------------

def rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling z-score along axis 0.

    Parameters
    ----------
    arr : np.ndarray, shape (T, N)
    window : int

    Returns
    -------
    np.ndarray, same shape — NaN-free (expanding for t < window)
    """
    out = np.zeros_like(arr, dtype=float)
    for t in range(len(arr)):
        start = max(0, t - window + 1)
        window_data = arr[start : t + 1]
        mu = window_data.mean(axis=0)
        sigma = window_data.std(axis=0)
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)
        out[t] = (arr[t] - mu) / sigma
    return out


# ---------------------------------------------------------------------------
# Pre-compute rolling z-score for the full returns matrix
# ---------------------------------------------------------------------------
print("\nComputing rolling z-score on returns matrix … (this may take ~30 s)")
ret_zscore: np.ndarray = rolling_zscore(ret_vals, NORM_WINDOW)
print("  Done.")

# ---------------------------------------------------------------------------
# Pre-compute realized volatility (rolling 20-day std, per asset)
# ---------------------------------------------------------------------------
realized_vol = np.zeros((T, N_ASSETS), dtype=float)
for t in range(LOOKBACK, T):
    window_data = ret_vals[t - LOOKBACK : t]
    realized_vol[t] = window_data.std(axis=0)

# Normalize realized_vol with rolling z-score
print("Computing rolling z-score on realized vol …")
realized_vol_norm: np.ndarray = rolling_zscore(realized_vol, NORM_WINDOW)
print("  Done.")

# ---------------------------------------------------------------------------
# Pre-compute momentum (sum of last 20 returns, per asset)
# ---------------------------------------------------------------------------
momentum = np.zeros((T, N_ASSETS), dtype=float)
for t in range(LOOKBACK, T):
    momentum[t] = ret_vals[t - LOOKBACK : t].sum(axis=0)

print("Computing rolling z-score on momentum …")
momentum_norm: np.ndarray = rolling_zscore(momentum, NORM_WINDOW)
print("  Done.")

# ---------------------------------------------------------------------------
# Macro features: SELIC (CDS proxy) + spread + one-hot regime
# ---------------------------------------------------------------------------
selic_vals = macro_df["selic"].fillna(method="ffill").fillna(method="bfill").values.reshape(-1, 1)
spread_vals = macro_df["spread_credito"].fillna(method="ffill").fillna(method="bfill").values.reshape(-1, 1)

print("Computing rolling z-score on SELIC and spread …")
selic_norm: np.ndarray = rolling_zscore(selic_vals, NORM_WINDOW)    # (T, 1) — CDS proxy
spread_norm: np.ndarray = rolling_zscore(spread_vals, NORM_WINDOW)  # (T, 1)
print("  Done.")

selic_up: np.ndarray = macro_df["selic_up"].fillna(0).values.reshape(-1, 1)
selic_stable: np.ndarray = macro_df["selic_stable"].fillna(0).values.reshape(-1, 1)
selic_down: np.ndarray = macro_df["selic_down"].fillna(0).values.reshape(-1, 1)

# ---------------------------------------------------------------------------
# Minimum Variance Portfolio (MVP) weights
# ---------------------------------------------------------------------------

def compute_mvp_weights(ret_window: np.ndarray) -> np.ndarray:
    """
    Compute minimum-variance portfolio weights via quadratic programming.

    Parameters
    ----------
    ret_window : np.ndarray, shape (252, N)

    Returns
    -------
    np.ndarray, shape (N,) — simplex-projected weights
    """
    n = ret_window.shape[1]
    cov = np.cov(ret_window.T)
    # Ensure PSD
    cov += np.eye(n) * 1e-8

    def portfolio_variance(w: np.ndarray) -> float:
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    result = minimize(
        portfolio_variance,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if result.success:
        w = np.clip(result.x, 0, 1)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n   # 1/N fallback


# ---------------------------------------------------------------------------
# Build state vector at time t
# ---------------------------------------------------------------------------

def build_state(t: int) -> np.ndarray:
    """
    Construct the 225-dimensional state vector at time t.

    Parameters
    ----------
    t : int — index into ret_vals (must be >= MVP_WINDOW)

    Returns
    -------
    np.ndarray, shape (225,)
    """
    # 200 dims: normalized returns over lookback window (flattened)
    ret_block = ret_zscore[t - LOOKBACK : t].flatten()   # (200,)

    # 10 dims: realized vol (normalized)
    rvol = realized_vol_norm[t]                           # (10,)

    # 10 dims: momentum (normalized)
    mom = momentum_norm[t]                                # (10,)

    # 1 dim: CDS proxy (SELIC normalized)
    cds = selic_norm[t]                                   # (1,)

    # 1 dim: spread normalized
    spr = spread_norm[t]                                  # (1,)

    # 3 dims: one-hot SELIC regime
    reg = np.array([selic_up[t, 0], selic_stable[t, 0], selic_down[t, 0]])  # (3,)

    state = np.concatenate([ret_block, rvol, mom, cds, spr, reg])
    assert state.shape == (STATE_DIM,), f"State dim mismatch: {state.shape}"
    return state.astype(np.float32)


# ---------------------------------------------------------------------------
# CVaR reward
# ---------------------------------------------------------------------------

def compute_reward(t: int, weights: np.ndarray) -> float:
    """
    CVaR-based reward for a portfolio at time t.

    Parameters
    ----------
    t       : int
    weights : np.ndarray, shape (N,)

    Returns
    -------
    float
    """
    port_returns_window = ret_vals[t - LOOKBACK : t] @ weights   # (20,)
    var = np.quantile(port_returns_window, 1.0 - ALPHA)
    tail = port_returns_window[port_returns_window <= var]
    cvar = -tail.mean() if len(tail) > 0 else 0.0
    expected = port_returns_window.mean()
    reward = -cvar + LAMBDA_ * expected
    return float(reward)


# ---------------------------------------------------------------------------
# Main dataset construction loop
# ---------------------------------------------------------------------------
START_IDX: int = MVP_WINDOW   # t=252 — need 252-day window for MVP + normalization

print(f"\nBuilding dataset starting at t={START_IDX} ({dates[START_IDX].date()}) …")

observations_list: list[np.ndarray] = []
actions_list: list[np.ndarray] = []
rewards_list: list[float] = []
next_obs_list: list[np.ndarray] = []
terminals_list: list[bool] = []
valid_dates: list[pd.Timestamp] = []

n_fallback: int = 0
total_steps: int = T - START_IDX - 1   # -1 because we need t+1 for next_state

for i, t in enumerate(range(START_IDX, T - 1)):
    if i % 200 == 0:
        print(f"  t={t:4d} / {T-2}  ({dates[t].date()})", flush=True)

    # Action: MVP weights from preceding 252-day window
    ret_window = ret_vals[t - MVP_WINDOW : t]
    weights = compute_mvp_weights(ret_window)

    # Fallback detection (1/N)
    if np.allclose(weights, 1.0 / N_ASSETS, atol=1e-4):
        n_fallback += 1

    # State
    s_t = build_state(t)

    # Reward
    r_t = compute_reward(t, weights)

    # Next state
    s_t1 = build_state(t + 1)

    # Terminal: True only at last step
    terminal = (t == T - 2)

    observations_list.append(s_t)
    actions_list.append(weights.astype(np.float32))
    rewards_list.append(r_t)
    next_obs_list.append(s_t1)
    terminals_list.append(terminal)
    valid_dates.append(dates[t])

print(f"\nDataset construction complete.")
print(f"  Total steps built : {len(valid_dates)}")
print(f"  MVP fallback count: {n_fallback} ({100*n_fallback/len(valid_dates):.1f}%)")

# Convert to arrays
observations   = np.stack(observations_list).astype(np.float32)    # (N, 225)
actions        = np.stack(actions_list).astype(np.float32)          # (N, 10)
rewards        = np.array(rewards_list, dtype=np.float32)           # (N,)
next_obs       = np.stack(next_obs_list).astype(np.float32)         # (N, 225)
terminals      = np.array(terminals_list, dtype=bool)               # (N,)
all_dates      = pd.DatetimeIndex(valid_dates)

# ---------------------------------------------------------------------------
# Train / eval split
# ---------------------------------------------------------------------------
train_mask = all_dates <= TRAIN_END
eval_mask  = all_dates >= EVAL_START

obs_train       = observations[train_mask]
act_train       = actions[train_mask]
rew_train       = rewards[train_mask]
nobs_train      = next_obs[train_mask]
term_train      = terminals[train_mask]
dates_train     = all_dates[train_mask]

obs_eval        = observations[eval_mask]
act_eval        = actions[eval_mask]
rew_eval        = rewards[eval_mask]
nobs_eval       = next_obs[eval_mask]
term_eval       = terminals[eval_mask]
dates_eval      = all_dates[eval_mask]

# Force terminal=True on last step of each split
if len(term_train) > 0:
    term_train[-1] = True
if len(term_eval) > 0:
    term_eval[-1] = True

# ---------------------------------------------------------------------------
# Print statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"\nState vector dimension breakdown:")
print(f"  Normalized returns (10 x {LOOKBACK})  : 200")
print(f"  Realized volatility (per asset)       :  10")
print(f"  Momentum (per asset)                  :  10")
print(f"  CDS proxy (SELIC normalized)          :   1")
print(f"  Spread normalized                     :   1")
print(f"  SELIC regime one-hot (up/stable/down) :   3")
print(f"  {'TOTAL':36s}: {STATE_DIM}")

print(f"\nTraining dataset:")
print(f"  Date range : {dates_train[0].date()} – {dates_train[-1].date()}")
print(f"  N steps    : {len(rew_train)}")
print(f"  Shapes     : obs={obs_train.shape}, act={act_train.shape}, rew={rew_train.shape}")

print(f"\nEval (OOS) dataset:")
print(f"  Date range : {dates_eval[0].date()} – {dates_eval[-1].date()}")
print(f"  N steps    : {len(rew_eval)}")
print(f"  Shapes     : obs={obs_eval.shape}, act={act_eval.shape}, rew={rew_eval.shape}")

print(f"\nReward statistics (TRAIN):")
print(f"  Mean   : {rew_train.mean():.6f}")
print(f"  Std    : {rew_train.std():.6f}")
print(f"  Min    : {rew_train.min():.6f}")
print(f"  Max    : {rew_train.max():.6f}")
print(f"  Median : {np.median(rew_train):.6f}")

print(f"\nReward statistics (EVAL):")
print(f"  Mean   : {rew_eval.mean():.6f}")
print(f"  Std    : {rew_eval.std():.6f}")
print(f"  Min    : {rew_eval.min():.6f}")
print(f"  Max    : {rew_eval.max():.6f}")
print(f"  Median : {np.median(rew_eval):.6f}")

# CVaR benchmark comparison (1/N portfolio)
equal_weights = np.ones(N_ASSETS) / N_ASSETS
ew_rewards_train = np.array([compute_reward(t, equal_weights)
                              for t in range(START_IDX, START_IDX + len(rew_train))])
ew_rewards_eval  = np.array([compute_reward(t, equal_weights)
                              for t in range(T - len(rew_eval) - 1, T - 1)])

reward_improvement_train = (rew_train.mean() - ew_rewards_train.mean()) / abs(ew_rewards_train.mean()) * 100
reward_improvement_eval  = (rew_eval.mean()  - ew_rewards_eval.mean())  / abs(ew_rewards_eval.mean())  * 100

print(f"\nCVaR reward: MVP vs 1/N benchmark:")
print(f"  Train - MVP mean: {rew_train.mean():.6f} | 1/N mean: {ew_rewards_train.mean():.6f} | diff: {reward_improvement_train:+.2f}%")
print(f"  Eval  - MVP mean: {rew_eval.mean():.6f} | 1/N mean: {ew_rewards_eval.mean():.6f} | diff: {reward_improvement_eval:+.2f}%")

# ---------------------------------------------------------------------------
# Save datasets
# ---------------------------------------------------------------------------
print("\nSaving datasets …")

np.savez_compressed(
    DATA_PROCESSED / "offline_dataset.npz",
    observations=obs_train,
    actions=act_train,
    rewards=rew_train,
    next_observations=nobs_train,
    terminals=term_train,
)
print(f"  Saved: {DATA_PROCESSED / 'offline_dataset.npz'}")

np.savez_compressed(
    DATA_PROCESSED / "eval_dataset.npz",
    observations=obs_eval,
    actions=act_eval,
    rewards=rew_eval,
    next_observations=nobs_eval,
    terminals=term_eval,
)
print(f"  Saved: {DATA_PROCESSED / 'eval_dataset.npz'}")

np.save(DATA_PROCESSED / "state_dates_train.npy", dates_train.values)
print(f"  Saved: {DATA_PROCESSED / 'state_dates_train.npy'}")

np.save(DATA_PROCESSED / "state_dates_eval.npy", dates_eval.values)
print(f"  Saved: {DATA_PROCESSED / 'state_dates_eval.npy'}")

# ---------------------------------------------------------------------------
# MDP summary table
# ---------------------------------------------------------------------------
summary_data = {
    "Metric": [
        "N_train",
        "N_eval",
        "state_dim",
        "action_dim",
        "Train reward mean",
        "Train reward std",
        "Train reward min",
        "Train reward max",
        "Eval reward mean",
        "Eval reward std",
        "Eval reward min",
        "Eval reward max",
        "Train CVaR improv. vs 1/N (%)",
        "Eval CVaR improv. vs 1/N (%)",
    ],
    "Value": [
        len(rew_train),
        len(rew_eval),
        STATE_DIM,
        N_ASSETS,
        f"{rew_train.mean():.6f}",
        f"{rew_train.std():.6f}",
        f"{rew_train.min():.6f}",
        f"{rew_train.max():.6f}",
        f"{rew_eval.mean():.6f}",
        f"{rew_eval.std():.6f}",
        f"{rew_eval.min():.6f}",
        f"{rew_eval.max():.6f}",
        f"{reward_improvement_train:+.2f}",
        f"{reward_improvement_eval:+.2f}",
    ],
}
summary_df = pd.DataFrame(summary_data)
csv_path = OUTPUTS_TABLES / "mdp_summary.csv"
summary_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# LaTeX table
latex_lines = [
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \caption{MDP Offline Dataset Summary}",
    r"  \label{tab:mdp_summary}",
    r"  \begin{tabular}{lr}",
    r"    \toprule",
    r"    \textbf{Metric} & \textbf{Value} \\",
    r"    \midrule",
]
for _, row in summary_df.iterrows():
    metric = str(row["Metric"]).replace("_", r"\_").replace("%", r"\%")
    value  = str(row["Value"]).replace("%", r"\%")
    latex_lines.append(f"    {metric} & {value} \\\\")

latex_lines += [
    r"    \bottomrule",
    r"  \end{tabular}",
    r"\end{table}",
]
latex_str = "\n".join(latex_lines) + "\n"

tex_path = OUTPUTS_TABLES / "mdp_summary.tex"
tex_path.write_text(latex_str, encoding="utf-8")
print(f"Saved: {tex_path}")

print("\n" + "=" * 60)
print("NB-05 COMPLETE")
print("=" * 60)
