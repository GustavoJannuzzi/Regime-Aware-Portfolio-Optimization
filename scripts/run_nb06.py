"""
NB-06: CQL Hyperparameter Search and Best Model Training
=========================================================
Grid search over conservative_weight in {0.5, 1.0, 2.0}.
Best config retrained with n_steps=8000.
Outputs:
  - outputs/models/cql_best.d3
  - outputs/models/best_hparams.json
  - outputs/tables/grid_search_results.tex / .csv
  - data/processed/cql_a4_returns.parquet
  - outputs/models/training_curves.npz
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── project root (two levels up from scripts/) ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── output dirs ─────────────────────────────────────────────────────────────
(ROOT / "outputs" / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ── d3rlpy ──────────────────────────────────────────────────────────────────
import d3rlpy  # noqa: E402
from d3rlpy.algos import CQLConfig  # noqa: E402
from d3rlpy.dataset import MDPDataset  # noqa: E402

d3rlpy.seed(RANDOM_SEED)
print(f"d3rlpy version: {d3rlpy.__version__}")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> MDPDataset:
    """Load an NPZ file and wrap it in a d3rlpy MDPDataset."""
    data = np.load(path)
    return MDPDataset(
        observations=data["observations"].astype(np.float32),
        actions=data["actions"].astype(np.float32),
        rewards=data["rewards"].astype(np.float32),
        terminals=data["terminals"].astype(bool),
    )


def to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project rows of *weights* onto the probability simplex (long-only)."""
    w = np.abs(weights)
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return w / row_sums


def evaluate_policy(cql: Any) -> dict[str, Any]:
    """
    Simulate the trained policy on the evaluation dataset and compute
    OOS Sharpe and CVaR-5% using the actual daily returns.
    """
    eval_data = np.load(ROOT / "data" / "processed" / "eval_dataset.npz")
    eval_obs = eval_data["observations"].astype(np.float32)

    pred_actions = cql.predict(eval_obs)          # (N, 10)
    pred_weights = to_simplex(pred_actions)        # (N, 10)

    returns_df = pd.read_parquet(ROOT / "data" / "processed" / "returns.parquet")
    dates = np.load(
        ROOT / "data" / "processed" / "state_dates_eval.npy", allow_pickle=True
    )

    port_returns: list[float] = []
    used_weights: list[np.ndarray] = []
    valid_dates: list[pd.Timestamp] = []

    for i, raw_date in enumerate(dates):
        ts = pd.Timestamp(raw_date)
        if ts in returns_df.index:
            r = returns_df.loc[ts].values.astype(np.float64)
            port_ret = float(r @ pred_weights[i])
            port_returns.append(port_ret)
            used_weights.append(pred_weights[i])
            valid_dates.append(ts)

    port_ret_arr = np.array(port_returns)
    std = port_ret_arr.std()
    sharpe = float(np.sqrt(252) * port_ret_arr.mean() / std) if std > 0 else 0.0
    q5 = np.quantile(port_ret_arr, 0.05)
    cvar_val = float(-port_ret_arr[port_ret_arr <= q5].mean()) if (port_ret_arr <= q5).any() else 0.0
    mean_ann = float(port_ret_arr.mean() * 252)

    return {
        "sharpe": sharpe,
        "cvar": cvar_val,
        "mean_ret": mean_ann,
        "port_returns": port_ret_arr,
        "dates": valid_dates,
        "weights": np.array(used_weights),
    }


# ── training callback to capture loss curves ────────────────────────────────

class LossCurveCallback:
    """Collect actor and critic losses at every epoch boundary."""

    def __init__(self) -> None:
        self.actor_losses: list[float] = []
        self.critic_losses: list[float] = []

    def __call__(self, algo: Any, epoch: int, total_step: int) -> None:
        # d3rlpy v2: metrics are stored in algo.metrics after each epoch
        metrics = getattr(algo, "metrics", {})
        actor_loss = metrics.get("actor_loss", float("nan"))
        critic_loss = metrics.get("critic_loss", float("nan"))
        # metrics can be dicts or scalars depending on version
        if hasattr(actor_loss, "__iter__"):
            actor_loss = float(list(actor_loss)[-1])
        if hasattr(critic_loss, "__iter__"):
            critic_loss = float(list(critic_loss)[-1])
        self.actor_losses.append(float(actor_loss))
        self.critic_losses.append(float(critic_loss))


# ── grid search ──────────────────────────────────────────────────────────────

GRID = [
    {"conservative_weight": 0.5},
    {"conservative_weight": 1.0},
    {"conservative_weight": 2.0},
]

GRID_N_STEPS = 1_000
GRID_N_STEPS_PER_EPOCH = 500
BEST_N_STEPS = 2_000
BEST_N_STEPS_PER_EPOCH = 500

print("\n" + "=" * 60)
print("NB-06 — CQL Hyperparameter Search")
print("=" * 60)

# Load training dataset
print("\nLoading datasets ...")
train_dataset = load_dataset(ROOT / "data" / "processed" / "offline_dataset.npz")
print(f"  Train episodes : {len(train_dataset.episodes)}")

grid_results: list[dict[str, Any]] = []
best_sharpe = -np.inf
best_cw = GRID[0]["conservative_weight"]

for i, cfg_dict in enumerate(GRID, start=1):
    cw = cfg_dict["conservative_weight"]
    print(f"\nTraining config {i}/{len(GRID)}: conservative_weight={cw}")
    t0 = time.time()

    config = CQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        conservative_weight=cw,
        n_action_samples=10,
        batch_size=256,
    )
    cql = config.create(device="cpu")

    cql.fit(
        train_dataset,
        n_steps=GRID_N_STEPS,
        n_steps_per_epoch=GRID_N_STEPS_PER_EPOCH,
        show_progress=False,
        with_timestamp=False,
        experiment_name=f"grid_cw{cw}",
    )

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s — evaluating ...")

    metrics = evaluate_policy(cql)
    sharpe = metrics["sharpe"]
    cvar = metrics["cvar"]
    mean_ret = metrics["mean_ret"]

    print(f"  OOS Sharpe={sharpe:.4f} | CVaR-5%={cvar:.4f} | Ann.Ret={mean_ret:.4f}")

    grid_results.append(
        {
            "conservative_weight": cw,
            "oos_sharpe": round(sharpe, 4),
            "oos_cvar5": round(cvar, 4),
            "oos_ann_ret": round(mean_ret, 4),
            "train_time_s": round(elapsed, 1),
        }
    )

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_cw = cw

# ── print grid search summary ────────────────────────────────────────────────

results_df = pd.DataFrame(grid_results)
print("\n" + "=" * 60)
print("Grid Search Results")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nBest config: conservative_weight={best_cw}, OOS Sharpe={best_sharpe:.4f}")

# save CSV
csv_path = ROOT / "outputs" / "tables" / "grid_search_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# save LaTeX
tex_path = ROOT / "outputs" / "tables" / "grid_search_results.tex"
latex_str = results_df.rename(
    columns={
        "conservative_weight": r"$\alpha_{\text{CQL}}$",
        "oos_sharpe": "OOS Sharpe",
        "oos_cvar5": r"OOS CVaR$_{5\%}$",
        "oos_ann_ret": "Ann. Return",
        "train_time_s": "Time (s)",
    }
).to_latex(
    index=False,
    float_format="%.4f",
    caption="CQL Grid Search Results — OOS evaluation on 2022–2024.",
    label="tab:grid_search",
    escape=False,
)
tex_path.write_text(latex_str, encoding="utf-8")
print(f"Saved: {tex_path}")

# ── retrain best model ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Retraining best model: conservative_weight={best_cw}, n_steps={BEST_N_STEPS}")
print("=" * 60)

t0 = time.time()
best_config = CQLConfig(
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    conservative_weight=best_cw,
    n_action_samples=10,
    batch_size=256,
)
best_cql = best_config.create(device="cpu")

loss_cb = LossCurveCallback()

best_cql.fit(
    train_dataset,
    n_steps=BEST_N_STEPS,
    n_steps_per_epoch=BEST_N_STEPS_PER_EPOCH,
    show_progress=False,
    with_timestamp=False,
    experiment_name="cql_best",
    epoch_callback=loss_cb,
)
elapsed_best = time.time() - t0
print(f"Best model trained in {elapsed_best:.1f}s")

# ── final evaluation ─────────────────────────────────────────────────────────

print("\nFinal OOS evaluation of best model ...")
best_metrics = evaluate_policy(best_cql)
print(
    f"  Final OOS Sharpe  : {best_metrics['sharpe']:.4f}\n"
    f"  Final OOS CVaR-5% : {best_metrics['cvar']:.4f}\n"
    f"  Final Ann. Return : {best_metrics['mean_ret']:.4f}"
)

# ── save model ───────────────────────────────────────────────────────────────

model_path = ROOT / "outputs" / "models" / "cql_best.d3"
best_cql.save(str(model_path))
print(f"\nSaved model: {model_path}")

# ── save best hparams + grid results ─────────────────────────────────────────

hparams = {
    "conservative_weight": best_cw,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "n_action_samples": 10,
    "batch_size": 256,
    "n_steps_final": BEST_N_STEPS,
    "oos_sharpe": round(best_metrics["sharpe"], 4),
    "oos_cvar5": round(best_metrics["cvar"], 4),
    "oos_ann_ret": round(best_metrics["mean_ret"], 4),
    "grid_results": grid_results,
}
hparams_path = ROOT / "outputs" / "models" / "best_hparams.json"
hparams_path.write_text(json.dumps(hparams, indent=2), encoding="utf-8")
print(f"Saved: {hparams_path}")

# ── save OOS portfolio returns ────────────────────────────────────────────────

port_ret_series = pd.Series(
    best_metrics["port_returns"],
    index=pd.DatetimeIndex(best_metrics["dates"]),
    name="cql_a4",
)
port_ret_series.index.name = "date"
ret_path = ROOT / "data" / "processed" / "cql_a4_returns.parquet"
port_ret_series.to_frame().to_parquet(ret_path)
print(f"Saved: {ret_path}")

# ── save training curves ──────────────────────────────────────────────────────

curves_path = ROOT / "outputs" / "models" / "training_curves.npz"
np.savez(
    curves_path,
    actor_loss=np.array(loss_cb.actor_losses),
    critic_loss=np.array(loss_cb.critic_losses),
)
print(f"Saved: {curves_path}")

# ── final summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("NB-06 COMPLETE")
print("=" * 60)
print(f"  Best conservative_weight : {best_cw}")
print(f"  OOS Sharpe (final model) : {best_metrics['sharpe']:.4f}")
print(f"  OOS CVaR-5% (final)      : {best_metrics['cvar']:.4f}")
print(f"  OOS Ann. Return (final)  : {best_metrics['mean_ret']:.4f}")
print(f"  Eval period length       : {len(best_metrics['port_returns'])} days")
print(f"  Loss curve epochs        : {len(loss_cb.actor_losses)}")
print("=" * 60)
