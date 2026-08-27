"""
NB-08 + NB-09: Final Results Tables and Publication-Ready Figures
==================================================================
Outputs:
  - outputs/tables/main_results.csv / .tex
  - outputs/tables/subperiod_results.csv / .tex
  - outputs/tables/regime_conditional.csv / .tex
  - outputs/figures/fig_wealth_curves.pdf/.png
  - outputs/figures/fig_drawdown.pdf/.png
  - outputs/figures/fig_cvar_subperiods.pdf/.png
  - outputs/figures/fig_ablation_heatmap.pdf/.png
  - outputs/figures/fig_sector_allocation.pdf/.png
  - outputs/figures/fig_training_curves.pdf/.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

FIG_DIR   = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

def save_fig(name: str, fig: plt.Figure) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: outputs/figures/{name}.pdf/.png")

def para_latex(df: pd.DataFrame) -> pd.DataFrame:
    """Troca os rotulos por versoes que o LaTeX compila.

    `to_latex` e' chamado com escape=False, entao o que sai daqui precisa ja'
    ser LaTeX valido. Subscrito Unicode e a estrela do A4 nao sao.
    """
    trocas = {
        "CVaR\u2080.\u2089\u2085": "CVaR$_{0.95}$",
        "CVaR_0.95": "CVaR$_{0.95}$",
    }
    saida = df.copy()
    saida.columns = [trocas.get(c, c) for c in saida.columns]
    limpa = lambda v: str(v).replace(" \u2605", "").replace("\u2605", "")
    if isinstance(saida.index, pd.MultiIndex):
        saida.index = pd.MultiIndex.from_tuples(
            [tuple(limpa(p) for p in t) for t in saida.index], names=saida.index.names
        )
    else:
        saida.index = [limpa(v) for v in saida.index]
    return saida


def metrics(pr: pd.Series) -> dict:
    arr = pr.dropna().values
    ann_ret  = arr.mean() * 252
    ann_vol  = arr.std()  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0
    down     = arr[arr < 0].std() * np.sqrt(252) if (arr < 0).any() else 1e-9
    sortino  = ann_ret / down
    q5       = np.quantile(arr, 0.05)
    cvar_v   = float(-arr[arr <= q5].mean()) if (arr <= q5).any() else 0.0
    wealth   = (1 + arr).cumprod()
    peak     = np.maximum.accumulate(wealth)
    mdd      = float(((wealth - peak) / peak).min())
    calmar   = ann_ret / abs(mdd) if mdd != 0 else 0.0
    return dict(ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, cvar_95=cvar_v, max_dd=mdd, calmar=calmar)

# ── Load all return series ────────────────────────────────────────────────────
STRATEGIES = ["A1", "A2", "A3", "A4", "EqualWeight", "MVP", "RiskParity"]

print("Loading data …")
abl   = pd.read_parquet(ROOT / "data" / "processed" / "ablation_returns.parquet")
bench = pd.read_parquet(ROOT / "data" / "processed" / "benchmark_returns.parquet")
macro = pd.read_parquet(ROOT / "data" / "processed" / "macro.parquet")

# NOTE (correção): a versão anterior deste script sobrescrevia a coluna "A4"
# com a série do NB-06 (o modelo do grid search, treinado com 2.000 steps).
# Isso quebrava a escada da ablação: A1, A2 e A3 vinham do NB-07 com 5.000
# steps e A4 vinha de outro protocolo, então a tabela principal comparava
# tratamento COM duração de treino, não o tratamento sozinho. O A4 usado aqui
# é o do NB-07, treinado exatamente como A1–A3.
#
# A série do NB-06 continua no repositório (cql_a4_returns.parquet) e é
# reportada separadamente como checagem de robustez, não como a linha de A4.
cql_best_path = ROOT / "data" / "processed" / "cql_a4_returns.parquet"
tuning_run = None
if cql_best_path.exists():
    tuning_run = pd.read_parquet(cql_best_path)["cql_a4"]

# Build unified returns frame (eval period overlap)
# NOTE (correcao): o NB-07 ja' grava as colunas de benchmark dentro de
# ablation_returns.parquet, recortadas na janela de avaliacao. Um join direto
# com benchmark_returns.parquet colide nos mesmos nomes e o pandas levanta
# "columns overlap but no suffix specified". Fica a versao que ja' esta' em
# `abl`, e do arquivo de benchmarks vem so' o que faltar.
novas = [c for c in bench.columns if c not in abl.columns]
all_ret = (abl.join(bench[novas], how="outer") if novas else abl.copy()).sort_index()
eval_start = abl.index.min()
eval_end   = abl.index.max()
all_ret = all_ret.loc[eval_start:eval_end]

# NOTE (correção): comparar estratégias medidas em janelas diferentes não é
# comparação. O agente tem 748 dias de avaliação (até 2024-12-27); os
# benchmarks param em 2024-11-13, porque o walk-forward precisa de uma janela
# de teste inteira e não sobrou dado para mais um fold. Antes, cada coluna era
# avaliada no seu próprio intervalo via .dropna() individual. Agora todas as
# tabelas de comparação usam a interseção das datas, e o número de dias fica
# explícito nas legendas.
COMPARE_COLS = [c for c in STRATEGIES if c in all_ret.columns]
common_idx = all_ret[COMPARE_COLS].dropna(how="any").index
compare_ret = all_ret.loc[common_idx, COMPARE_COLS]
print(f"Janela comum de comparação: {common_idx.min().date()} a {common_idx.max().date()} "
      f"({len(common_idx)} dias de pregão)")
print(f"  (a série do agente vai até {all_ret.index.max().date()}; "
      f"os {len(all_ret) - len(common_idx)} dias sem benchmark ficam de fora das comparações)")

COLORS     = ["#d62728","#ff7f0e","#2ca02c","#1f77b4","#9467bd","#8c564b","#e377c2"]
COLOR_MAP  = dict(zip(STRATEGIES, COLORS))
LABELS     = {
    "A1": "A1 (Unconstrained+Sharpe)",
    "A2": "A2 (CQL+Sharpe)",
    "A3": "A3 (CQL+CVaR)",
    "A4": "A4 (CQL+CVaR+Macro) ★",
    "EqualWeight": "1/N",
    "MVP": "Min. Variance",
    "RiskParity": "Risk Parity",
}

# ── NB-08: Main results table ─────────────────────────────────────────────────
print("\n=== NB-08: Main Results ===")
rows = []
for s in STRATEGIES:
    if s in compare_ret.columns:
        m = metrics(compare_ret[s])
        m["Strategy"] = LABELS[s]
        rows.append(m)
main_df = pd.DataFrame(rows).set_index("Strategy")
col_order = ["ann_return","ann_vol","sharpe","sortino","cvar_95","max_dd","calmar"]
main_df = main_df[col_order]
main_df.columns = ["Ann.Return","Ann.Vol","Sharpe","Sortino","CVaR₀.₉₅","MaxDD","Calmar"]
print(main_df.round(4).to_string())

# Checagem de robustez, fora da tabela principal: o modelo do grid search do
# NB-06 (2.000 steps) medido na mesma janela comum. Não entra como linha de A4
# porque não seguiu o mesmo protocolo de treino das outras três configurações.
if tuning_run is not None:
    tuning_common = tuning_run.reindex(common_idx).dropna()
    if len(tuning_common) > 10:
        tm = metrics(tuning_common)
        pd.DataFrame([tm], index=["A4 tuning run (NB-06, 2k steps)"]).to_csv(
            TABLE_DIR / "a4_tuning_run_check.csv"
        )
        print()
        print(
            "  [robustez] A4 do NB-06 (2k steps) na mesma janela: "
            f"Sharpe={tm['sharpe']:.4f}  CVaR={tm['cvar_95']:.4f}  "
            f"(A4 do NB-07, 5k steps: Sharpe={main_df.loc[LABELS['A4'], 'Sharpe']:.4f})"
        )

main_df.to_csv(TABLE_DIR / "main_results.csv")
tex = para_latex(main_df).to_latex(
    float_format="%.4f", escape=False,
    caption=(
        f"Out-of-sample performance of all strategies, common evaluation window "
        f"{common_idx.min().date()} to {common_idx.max().date()} "
        f"({len(common_idx)} trading days). "
        "All configurations are trained under the identical protocol "
        "(5{,}000 gradient steps, seed 42). "
        "A4 is the full proposal (CQL + CVaR + macro state)."
    ),
    label="tab:main_results",
)
(TABLE_DIR / "main_results.tex").write_text(tex, encoding="utf-8")
print("  Saved: outputs/tables/main_results.csv/.tex")

# ── Sub-period analysis ───────────────────────────────────────────────────────
SUB_PERIODS = {
    "2022 (Tightening)":   ("2022-01-01","2022-12-31"),
    "2023 (Peak/Stable)":  ("2023-01-01","2023-12-31"),
    "2024 (Easing)":       ("2024-01-01","2024-12-31"),
}
sub_rows = []
for period, (s, e) in SUB_PERIODS.items():
    sub = compare_ret.loc[s:e]
    for strat in ["A4","EqualWeight","MVP"]:
        if strat in sub.columns and not sub[strat].dropna().empty:
            m = metrics(sub[strat].dropna())
            m["Period"]   = period
            m["Strategy"] = LABELS[strat]
            sub_rows.append(m)
sub_df = pd.DataFrame(sub_rows).set_index(["Period","Strategy"])[col_order]
sub_df.columns = ["Ann.Return","Ann.Vol","Sharpe","Sortino","CVaR₀.₉₅","MaxDD","Calmar"]
sub_df.to_csv(TABLE_DIR / "subperiod_results.csv")
tex_sub = para_latex(sub_df).to_latex(
    float_format="%.4f", escape=False,
    caption="Sub-period performance analysis (2022--2024).",
    label="tab:subperiod_results", multirow=True,
)
(TABLE_DIR / "subperiod_results.tex").write_text(tex_sub, encoding="utf-8")
print("  Saved: outputs/tables/subperiod_results.csv/.tex")

# ── Regime-conditional performance ───────────────────────────────────────────
regime_map = macro["selic_regime"].reindex(compare_ret.index, method="ffill")
regime_labels = {-1: "Falling", 0: "Stable", 1: "Rising"}
reg_rows = []
for rv, rl in regime_labels.items():
    idx = regime_map[regime_map == rv].index
    for strat in ["A4","EqualWeight","MVP"]:
        if strat in compare_ret.columns:
            sub = compare_ret.loc[compare_ret.index.isin(idx), strat]
            if len(sub) > 10:
                m = metrics(sub)
                m["Regime"]   = rl
                m["Strategy"] = LABELS[strat]
                reg_rows.append(m)
reg_df = pd.DataFrame(reg_rows).set_index(["Regime","Strategy"])[["sharpe","cvar_95","ann_return"]]
reg_df.columns = ["Sharpe","CVaR₀.₉₅","Ann.Return"]
reg_df.to_csv(TABLE_DIR / "regime_conditional.csv")
tex_reg = para_latex(reg_df).to_latex(
    float_format="%.4f", escape=False,
    caption="Regime-conditional performance by SELIC cycle phase.",
    label="tab:regime_conditional", multirow=True,
)
(TABLE_DIR / "regime_conditional.tex").write_text(tex_reg, encoding="utf-8")
print("  Saved: outputs/tables/regime_conditional.csv/.tex")

# ── NB-09: FIGURES ────────────────────────────────────────────────────────────
print("\n=== NB-09: Figures ===")

# ── Fig 1: Wealth curves ──────────────────────────────────────────────────────
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for s in STRATEGIES:
        if s in compare_ret.columns:
            pr = compare_ret[s]
            wealth = (1 + pr).cumprod()
            wealth = wealth / wealth.iloc[0]
            ax.plot(wealth.index, wealth.values,
                    label=LABELS[s], color=COLOR_MAP[s],
                    lw=2.5 if s == "A4" else 1.2,
                    alpha=1.0 if s == "A4" else 0.75)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth Index (base = 1)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
save_fig("fig_wealth_curves", fig)

# ── Fig 2: Drawdown ───────────────────────────────────────────────────────────
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for s in ["A4", "EqualWeight", "MVP", "RiskParity"]:
        if s in compare_ret.columns:
            pr = compare_ret[s]
            wealth = (1 + pr).cumprod()
            peak   = wealth.cummax()
            dd     = (wealth - peak) / peak
            ax.fill_between(dd.index, dd.values, 0,
                            alpha=0.35 if s != "A4" else 0.6,
                            color=COLOR_MAP[s], label=LABELS[s])
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    fig.tight_layout()
save_fig("fig_drawdown", fig)

# ── Fig 3: CVaR by sub-period ─────────────────────────────────────────────────
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    periods = list(SUB_PERIODS.keys())
    x = np.arange(len(periods))
    w = 0.22
    plot_strats = ["A4", "EqualWeight", "MVP"]
    for j, s in enumerate(plot_strats):
        cvar_vals = []
        for p, (ps, pe) in SUB_PERIODS.items():
            sub = compare_ret.loc[ps:pe, s] if s in compare_ret.columns else pd.Series(dtype=float)
            cvar_vals.append(metrics(sub)["cvar_95"] if len(sub) > 5 else 0.0)
        ax.bar(x + j*w, cvar_vals, w, label=LABELS[s], color=COLOR_MAP[s], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(periods, fontsize=9)
    ax.set_ylabel("CVaR₀.₉₅ (daily)")
    ax.legend(fontsize=9)
    fig.tight_layout()
save_fig("fig_cvar_subperiods", fig)

# ── Fig 4: Ablation heatmap ───────────────────────────────────────────────────
abl_json_path = ROOT / "outputs" / "models" / "ablation_metrics.json"
with open(abl_json_path) as f:
    abl_met = json.load(f)

heat_strats = ["A1","A2","A3","A4"]
heat_metrics = ["sharpe","sortino","cvar_95","max_dd","calmar"]
heat_labels  = ["Sharpe","Sortino","CVaR₀.₉₅","MaxDD","Calmar"]
heat_data = np.array([
    [abl_met[s][m] if s in abl_met else 0.0 for m in heat_metrics]
    for s in heat_strats
], dtype=float)

# Normalize each column to [0,1] for color; flip sign of CVaR and MaxDD (lower=better)
heat_norm = heat_data.copy()
for c, m in enumerate(heat_metrics):
    col = heat_norm[:, c]
    if m in ("cvar_95", "max_dd"):
        col = -col  # lower is better → negate
    mn, mx = col.min(), col.max()
    heat_norm[:, c] = (col - mn) / (mx - mn + 1e-9)

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    im = ax.imshow(heat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat_labels))); ax.set_xticklabels(heat_labels, fontsize=10)
    ax.set_yticks(range(len(heat_strats))); ax.set_yticklabels(heat_strats, fontsize=10)
    for i in range(len(heat_strats)):
        for j in range(len(heat_metrics)):
            raw = heat_data[i, j]
            ax.text(j, i, f"{raw:.3f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Normalized score (↑ better)")
    fig.tight_layout()
save_fig("fig_ablation_heatmap", fig)

# ── Fig 5: Training curves (actor/critic loss) ────────────────────────────────
curves_path = ROOT / "outputs" / "models" / "training_curves.npz"
if curves_path.exists():
    curves = np.load(curves_path)
    actor_loss  = curves["actor_loss"]
    critic_loss = curves["critic_loss"]
    valid_mask  = np.isfinite(actor_loss) & np.isfinite(critic_loss)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
        epochs = np.arange(1, len(actor_loss)+1)
        axes[0].plot(epochs[valid_mask], actor_loss[valid_mask], color="#1f77b4", lw=1.5)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Actor Loss")
        axes[1].plot(epochs[valid_mask], critic_loss[valid_mask], color="#d62728", lw=1.5)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Critic Loss")
        fig.tight_layout()
    save_fig("fig_training_curves", fig)
else:
    print("  Training curves not found — skipping fig_training_curves")

# ── Fig 6: SELIC regime + A4 Sharpe conditional (bar chart) ──────────────────
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    reg_sharpes = {}
    for rv, rl in regime_labels.items():
        idx = regime_map[regime_map == rv].index
        for s in ["A4","EqualWeight"]:
            if s in compare_ret.columns:
                sub = compare_ret.loc[compare_ret.index.isin(idx), s]
                if len(sub) > 5:
                    reg_sharpes.setdefault(rl, {})[s] = metrics(sub)["sharpe"]

    regime_order = ["Rising","Stable","Falling"]
    x = np.arange(len(regime_order))
    w = 0.35
    for j, s in enumerate(["A4","EqualWeight"]):
        vals = [reg_sharpes.get(r, {}).get(s, 0.0) for r in regime_order]
        ax.bar(x + j*w, vals, w, label=LABELS[s], color=COLOR_MAP[s], alpha=0.85)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(x + w/2); ax.set_xticklabels(regime_order)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("SELIC Regime")
    ax.legend(fontsize=9)
    fig.tight_layout()
save_fig("fig_regime_sharpe", fig)

print("\n" + "=" * 60)
print("NB-08 + NB-09 COMPLETE")
print(f"  Tables: {len(list(TABLE_DIR.glob('*.tex')))} .tex files")
print(f"  Figures: {len(list(FIG_DIR.glob('*.pdf')))} .pdf files")
print("=" * 60)
