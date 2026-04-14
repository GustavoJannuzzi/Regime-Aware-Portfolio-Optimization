"""
NB-07: Ablation Study A1–A4
============================
A1: CQL conservative_weight≈0 + Sharpe reward + no macro  (≈DDPG/unconstrained)
A2: CQL best_cw           + Sharpe reward + no macro  (isolates offline paradigm)
A3: CQL best_cw           + CVaR reward   + no macro  (isolates reward)
A4: CQL best_cw           + CVaR reward   + full macro (full proposal)

Outputs:
  - outputs/tables/ablation_results.csv / .tex
  - data/processed/ablation_returns.parquet
  - outputs/models/ablation_metrics.json
  - outputs/models/cql_a1.d3, cql_a2.d3, cql_a3.d3
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import d3rlpy
from d3rlpy.algos import CQLConfig
from d3rlpy.dataset import MDPDataset

ROOT = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42
d3rlpy.seed(RANDOM_SEED)

print(f"d3rlpy version: {d3rlpy.__version__}")

# ── Load best hyperparameters from NB-06 ────────────────────────────────────
hparams_path = ROOT / "outputs" / "models" / "best_hparams.json"
with open(hparams_path) as f:
    hparams = json.load(f)
best_cw = hparams["conservative_weight"]
print(f"Best conservative_weight from NB-06: {best_cw}")

# ── Load offline training dataset ───────────────────────────────────────────
data = np.load(ROOT / "data" / "processed" / "offline_dataset.npz")
obs_train       = data["observations"].astype(np.float32)   # (1485, 225)
act_train       = data["actions"].astype(np.float32)         # (1485, 10)
rew_cvar        = data["rewards"].astype(np.float32)         # (1485,)
next_obs_train  = data["next_observations"].astype(np.float32)
terminals_train = data["terminals"].astype(bool)

eval_data = np.load(ROOT / "data" / "processed" / "eval_dataset.npz")
obs_eval  = eval_data["observations"].astype(np.float32)    # (748, 225)

returns_df = pd.read_parquet(ROOT / "data" / "processed" / "returns.parquet")
dates_eval = np.load(ROOT / "data" / "processed" / "state_dates_eval.npy", allow_pickle=True)
dates_train = np.load(ROOT / "data" / "processed" / "state_dates_train.npy", allow_pickle=True)

# ── Build Sharpe reward ──────────────────────────────────────────────────────
# Simple daily portfolio return as Sharpe-proxy reward
rew_sharpe = np.zeros(len(dates_train), dtype=np.float32)
for i, raw_d in enumerate(dates_train):
    ts = pd.Timestamp(raw_d)
    if ts in returns_df.index:
        rew_sharpe[i] = float(returns_df.loc[ts].values @ act_train[i])

print(f"Sharpe reward stats — mean: {rew_sharpe.mean():.5f}, std: {rew_sharpe.std():.5f}")

# ── No-macro state (drop last 5 dims: CDS_proxy, spread, selic_up/stable/down) ──
obs_train_nomacro      = obs_train[:, :220]
next_obs_train_nomacro = next_obs_train[:, :220]
obs_eval_nomacro       = obs_eval[:, :220]

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_dataset(obs, actions, rewards, next_obs, terminals):
    return MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards.reshape(-1),
        terminals=terminals.astype(np.float32).reshape(-1),
    )

def make_cql(cw: float):
    return CQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        conservative_weight=cw,
        n_action_samples=10,
        batch_size=256,
    ).create(device="cpu")

def to_simplex(w: np.ndarray) -> np.ndarray:
    w = np.abs(w)
    s = w.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return w / s

def evaluate(cql, obs_eval_arr, obs_dim_label=""):
    pred_w = to_simplex(cql.predict(obs_eval_arr))  # (N, 10)
    port_rets, valid_dates, weights_list = [], [], []
    for i, raw_d in enumerate(dates_eval):
        ts = pd.Timestamp(raw_d)
        if ts in returns_df.index:
            r = returns_df.loc[ts].values[:pred_w.shape[1]] if obs_eval_arr.shape[1] < 225 else returns_df.loc[ts].values
            # always use all 10 assets
            r_full = returns_df.loc[ts].values
            port_rets.append(float(r_full @ pred_w[i]))
            valid_dates.append(ts)
            weights_list.append(pred_w[i])

    pr = np.array(port_rets)
    ann_ret = pr.mean() * 252
    ann_vol = pr.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    sortino_down = pr[pr < 0].std() * np.sqrt(252) if (pr < 0).any() else 1e-9
    sortino = ann_ret / sortino_down
    q5      = np.quantile(pr, 0.05)
    cvar_v  = float(-pr[pr <= q5].mean()) if (pr <= q5).any() else 0.0
    wealth  = (1 + pr).cumprod()
    peak    = np.maximum.accumulate(wealth)
    mdd     = float(((wealth - peak) / peak).min())
    calmar  = ann_ret / abs(mdd) if mdd != 0 else 0.0
    return {
        "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "cvar_95": cvar_v, "max_dd": mdd, "calmar": calmar,
        "port_returns": pd.Series(port_rets, index=pd.DatetimeIndex(valid_dates)),
    }

# ── Ablation configurations ──────────────────────────────────────────────────
N_STEPS = 5_000
N_STEPS_PER_EPOCH = 500

configs = {
    "A1": dict(cw=0.001, reward=rew_sharpe, obs=obs_train_nomacro,
               next_obs=next_obs_train_nomacro, eval_obs=obs_eval_nomacro,
               label="CQL(a~0)+Sharpe+NoMacro"),
    "A2": dict(cw=best_cw, reward=rew_sharpe, obs=obs_train_nomacro,
               next_obs=next_obs_train_nomacro, eval_obs=obs_eval_nomacro,
               label=f"CQL(a={best_cw})+Sharpe+NoMacro"),
    "A3": dict(cw=best_cw, reward=rew_cvar,   obs=obs_train_nomacro,
               next_obs=next_obs_train_nomacro, eval_obs=obs_eval_nomacro,
               label=f"CQL(a={best_cw})+CVaR+NoMacro"),
    "A4": dict(cw=best_cw, reward=rew_cvar,   obs=obs_train,
               next_obs=next_obs_train,         eval_obs=obs_eval,
               label=f"CQL(a={best_cw})+CVaR+Macro"),
}

print("\n" + "=" * 60)
print("NB-07 — Ablation Study A1–A4")
print("=" * 60)

ablation_metrics = {}
ablation_returns = {}

for name, cfg in configs.items():
    print(f"\n=== {name}: {cfg['label']} ===")
    t0 = time.time()

    cql = make_cql(cfg["cw"])
    ds  = make_dataset(cfg["obs"], act_train, cfg["reward"], cfg["next_obs"], terminals_train)
    cql.fit(ds, n_steps=N_STEPS, n_steps_per_epoch=N_STEPS_PER_EPOCH,
            show_progress=False, with_timestamp=False, experiment_name=f"ablation_{name}")

    elapsed = time.time() - t0
    result  = evaluate(cql, cfg["eval_obs"])

    print(f"  Trained in {elapsed:.1f}s")
    print(f"  Sharpe={result['sharpe']:.4f}  Sortino={result['sortino']:.4f}  "
          f"CVaR={result['cvar_95']:.4f}  MDD={result['max_dd']:.4f}")

    ablation_metrics[name] = {
        k: round(v, 6) for k, v in result.items() if k != "port_returns"
    }
    ablation_returns[name] = result["port_returns"]

    # save individual models (A1–A3; A4 already saved as cql_best)
    if name != "A4":
        save_path = ROOT / "outputs" / "models" / f"cql_{name.lower()}.d3"
        cql.save(str(save_path))
        print(f"  Saved: {save_path}")

# ── Benchmark metrics (eval period 2022-2024) ────────────────────────────────
bench = pd.read_parquet(ROOT / "data" / "processed" / "benchmark_returns.parquet")
eval_start = pd.Timestamp(dates_eval[0])
eval_end   = pd.Timestamp(dates_eval[-1])
bench_eval = bench.loc[eval_start:eval_end]

bench_labels = {"EqualWeight": "1/N", "MVP": "Min. Variance", "RiskParity": "Risk Parity"}
for col, label in bench_labels.items():
    if col not in bench_eval.columns:
        continue
    pr = bench_eval[col].values
    ann_ret = pr.mean() * 252
    ann_vol = pr.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    sortino_down = pr[pr < 0].std() * np.sqrt(252) if (pr < 0).any() else 1e-9
    sortino = ann_ret / sortino_down
    q5      = np.quantile(pr, 0.05)
    cvar_v  = float(-pr[pr <= q5].mean()) if (pr <= q5).any() else 0.0
    wealth  = (1 + pr).cumprod()
    peak    = np.maximum.accumulate(wealth)
    mdd     = float(((wealth - peak) / peak).min())
    calmar  = ann_ret / abs(mdd) if mdd != 0 else 0.0
    ablation_metrics[col] = {
        "ann_return": round(ann_ret, 6), "ann_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 6), "sortino": round(sortino, 6),
        "cvar_95": round(cvar_v, 6), "max_dd": round(mdd, 6), "calmar": round(calmar, 6),
    }
    ablation_returns[col] = bench_eval[col]

# ── Build results DataFrame ───────────────────────────────────────────────────
rows = []
order = ["A1", "A2", "A3", "A4", "EqualWeight", "MVP", "RiskParity"]
for k in order:
    if k in ablation_metrics:
        m = ablation_metrics[k]
        rows.append({
            "Config": k,
            "Ann.Return": m["ann_return"],
            "Ann.Vol":    m["ann_vol"],
            "Sharpe":     m["sharpe"],
            "Sortino":    m["sortino"],
            "CVaR_0.95":  m["cvar_95"],
            "MaxDD":      m["max_dd"],
            "Calmar":     m["calmar"],
        })
results_df = pd.DataFrame(rows).set_index("Config")

print("\n" + "=" * 60)
print("Ablation + Benchmark Results (OOS 2022–2024)")
print("=" * 60)
print(results_df.round(4).to_string())

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = ROOT / "outputs" / "tables" / "ablation_results.csv"
results_df.to_csv(csv_path)
print(f"\nSaved: {csv_path}")

# ── Save LaTeX ────────────────────────────────────────────────────────────────
tex_str = results_df.to_latex(
    float_format="%.4f",
    escape=False,
    caption=(
        "Ablation study and benchmark comparison — OOS performance (2022--2024). "
        "A1: unconstrained RL + Sharpe, no macro. "
        "A2: CQL + Sharpe, no macro. "
        "A3: CQL + CVaR, no macro. "
        "A4: CQL + CVaR + macro (full proposal)."
    ),
    label="tab:ablation_results",
)
tex_path = ROOT / "outputs" / "tables" / "ablation_results.tex"
tex_path.write_text(tex_str, encoding="utf-8")
print(f"Saved: {tex_path}")

# ── Save OOS returns parquet ──────────────────────────────────────────────────
abl_ret_df = pd.DataFrame({k: v for k, v in ablation_returns.items()
                            if hasattr(v, "index")})
abl_ret_df.index.name = "date"
ret_path = ROOT / "data" / "processed" / "ablation_returns.parquet"
abl_ret_df.to_parquet(ret_path)
print(f"Saved: {ret_path}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_path = ROOT / "outputs" / "models" / "ablation_metrics.json"
json_path.write_text(json.dumps(ablation_metrics, indent=2), encoding="utf-8")
print(f"Saved: {json_path}")

print("\n" + "=" * 60)
print("NB-07 COMPLETE — Ablation A1–A4 done")
print("=" * 60)
