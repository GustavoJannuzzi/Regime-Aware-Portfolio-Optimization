"""
Substitui todos os \PLACEHOLDER{} no LaTeX com os números reais
produzidos pelo NB-07 (ablation_metrics.json) e NB-08 (main_results.csv,
subperiod_results.csv, regime_conditional.csv).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
LATEX_DIR = ROOT / "latex" / "sections"

# ── Carregar métricas ─────────────────────────────────────────────────────────
abl_path = ROOT / "outputs" / "models" / "ablation_metrics.json"
with open(abl_path) as f:
    abl = json.load(f)

# main_results
main_df = pd.read_csv(ROOT / "outputs" / "tables" / "main_results.csv", index_col=0)

# subperiod
sub_df = pd.read_csv(ROOT / "outputs" / "tables" / "subperiod_results.csv", index_col=[0, 1])

# regime conditional
reg_df = pd.read_csv(ROOT / "outputs" / "tables" / "regime_conditional.csv", index_col=[0, 1])

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"

def sub_val(df, row, col):
    try:
        return float(df.loc[row, col])
    except Exception:
        return float("nan")

# ── Construir dicionário de valores ──────────────────────────────────────────
# A4 label no main_results pode ser "A4 (CQL+CVaR+Macro) ★" ou similar
A4_label = [idx for idx in main_df.index if "A4" in str(idx)]
A4_label = A4_label[0] if A4_label else "A4"

EW_label  = [idx for idx in main_df.index if "1/N" in str(idx) or "Equal" in str(idx)]
EW_label  = EW_label[0] if EW_label else "EqualWeight"

A1_label  = [idx for idx in main_df.index if "A1" in str(idx)]
A1_label  = A1_label[0] if A1_label else "A1"

A2_label  = [idx for idx in main_df.index if "A2" in str(idx)]
A2_label  = A2_label[0] if A2_label else "A2"

A3_label  = [idx for idx in main_df.index if "A3" in str(idx)]
A3_label  = A3_label[0] if A3_label else "A3"

col_map = {
    "Ann.Return": "ann_return", "Ann.Vol": "ann_vol",
    "Sharpe": "sharpe", "Sortino": "sortino",
    "CVaR\u2080.\u2089\u2085": "cvar_95",  # unicode subscript
    "CVaR_0.95": "cvar_95",
    "MaxDD": "max_dd", "Calmar": "calmar",
}

def get_main(label, metric):
    for col in main_df.columns:
        if metric in col or col in col_map and col_map[col] == metric:
            try:
                return float(main_df.loc[label, col])
            except Exception:
                pass
    # fallback: try ablation metrics json
    key = label.split()[0]  # "A4", "A1", etc.
    if key in abl and metric in abl[key]:
        return float(abl[key][metric])
    return float("nan")

A4_sharpe    = get_main(A4_label, "sharpe")
A4_cvar      = get_main(A4_label, "cvar_95")
A4_ann_ret   = get_main(A4_label, "ann_return")
A4_maxdd     = get_main(A4_label, "max_dd")
A4_calmar    = get_main(A4_label, "calmar")
EW_sharpe    = get_main(EW_label, "sharpe")
# NOTE (correcao): o Sharpe do MVP estava fixo em 0.3692, valor do NB-04, que e'
# medido em 2016-2024. O A4 so' existe a partir de 2022, entao a diferenca
# publicada comparava janelas diferentes. Agora sai da mesma tabela, na mesma
# janela comum que o NB-08 monta.
MVP_label = [idx for idx in main_df.index if "Variance" in str(idx) or "MVP" in str(idx)]
MVP_label = MVP_label[0] if MVP_label else "MVP"
MVP_sharpe   = get_main(MVP_label, "sharpe")
A4_sharpe_diff     = A4_sharpe - EW_sharpe
A4_sharpe_vs_mvp   = A4_sharpe - MVP_sharpe

A1_sharpe    = get_main(A1_label, "sharpe")
A1_cvar      = get_main(A1_label, "cvar_95")
A2_sharpe    = get_main(A2_label, "sharpe")
A2_cvar      = get_main(A2_label, "cvar_95")
A3_sharpe    = get_main(A3_label, "sharpe")
A3_cvar      = get_main(A3_label, "cvar_95")

# Sub-period values (flexible lookup)
def get_sub(period_key, strat_key, col):
    for pidx in sub_df.index.get_level_values(0).unique():
        if period_key.lower() in str(pidx).lower():
            for sidx in sub_df.loc[pidx].index:
                if strat_key.lower() in str(sidx).lower():
                    try:
                        return float(sub_df.loc[(pidx, sidx), col])
                    except Exception:
                        pass
    return float("nan")

# NOTE (correcao): isto era uma cadeia `a or b or c`. NaN e' truthy em Python,
# entao um primeiro lookup falhado devolvia NaN e os fallbacks nunca rodavam,
# e o NaN ia parar no texto do artigo. `primeiro_valido` so' aceita numero.
def primeiro_valido(*valores: float) -> float:
    for v in valores:
        if v is not None and not np.isnan(v):
            return v
    return float("nan")

A4_cvar_2022 = primeiro_valido(
    get_sub("2022", "A4", "CVaR\u2080.\u2089\u2085"),
    get_sub("2022", "A4", "CVaR_0.95"),
    get_sub("2022", "A4", "cvar_95"),
)
EW_cvar_2022 = primeiro_valido(
    get_sub("2022", "1/N", "CVaR\u2080.\u2089\u2085"),
    get_sub("2022", "Equal", "CVaR_0.95"),
    get_sub("2022", "Equal", "cvar_95"),
)
A4_sharpe_2023 = get_sub("2023", "A4",   "Sharpe")
EW_sharpe_2023 = get_sub("2023", "1/N",  "Sharpe") or get_sub("2023", "Equal", "Sharpe")
A4_ret_2024    = get_sub("2024", "A4",   "Ann.Return")

# Regime-conditional
def get_reg(regime, strat_key, col):
    for ridx in reg_df.index.get_level_values(0).unique():
        if regime.lower() in str(ridx).lower():
            for sidx in reg_df.loc[ridx].index:
                if strat_key.lower() in str(sidx).lower():
                    try:
                        return float(reg_df.loc[(ridx, sidx), col])
                    except Exception:
                        pass
    return float("nan")

A4_sharpe_rising = get_reg("Rising", "A4", "Sharpe")
A4_sharpe_stable = get_reg("Stable", "A4", "Sharpe")

# ── Build replacement map ─────────────────────────────────────────────────────
replacements = {
    r"A4\_sharpe\_diff":  fmt(A4_sharpe_diff,  4),
    r"A4\_sharpe\_vs\_mvp": fmt(A4_sharpe_vs_mvp, 4),
    r"A4\_sharpe\_rising": fmt(A4_sharpe_rising, 4),
    r"A4\_sharpe\_stable": fmt(A4_sharpe_stable, 4),
    r"A4\_sharpe\_2023":  fmt(A4_sharpe_2023, 4),
    r"A4\_sharpe":        fmt(A4_sharpe, 4),
    r"A4\_cvar\_2022":    fmt(A4_cvar_2022, 4),
    r"A4\_cvar":          fmt(A4_cvar, 4),
    r"A4\_ann\_ret":      fmt(A4_ann_ret, 4),
    r"A4\_ret\_2024":     fmt(A4_ret_2024*100, 1) + r"\%",
    r"A4\_maxdd":         fmt(A4_maxdd, 4),
    r"A4\_calmar":        fmt(A4_calmar, 4),
    r"EW\_cvar\_2022":    fmt(EW_cvar_2022, 4),
    r"EW\_sharpe\_2023":  fmt(EW_sharpe_2023, 4),
    r"A1\_sharpe":        fmt(A1_sharpe, 4),
    r"A1\_cvar":          fmt(A1_cvar, 4),
    r"A2\_sharpe":        fmt(A2_sharpe, 4),
    r"A2\_cvar":          fmt(A2_cvar, 4),
    r"A3\_sharpe":        fmt(A3_sharpe, 4),
    r"A3\_cvar":          fmt(A3_cvar, 4),
}

# NOTE (correcao): antes, um valor que nao fosse encontrado virava a string
# "nan" e entrava no texto do artigo sem reclamar. Um numero ausente tem que
# parar o processo, nao sair impresso.
faltando = [k for k, v in {
    "A4_sharpe": A4_sharpe, "A4_cvar": A4_cvar, "A4_ann_ret": A4_ann_ret,
    "A4_maxdd": A4_maxdd, "A4_calmar": A4_calmar, "EW_sharpe": EW_sharpe,
    "MVP_sharpe": MVP_sharpe, "A1_sharpe": A1_sharpe, "A1_cvar": A1_cvar,
    "A2_sharpe": A2_sharpe, "A2_cvar": A2_cvar, "A3_sharpe": A3_sharpe,
    "A3_cvar": A3_cvar, "A4_cvar_2022": A4_cvar_2022, "EW_cvar_2022": EW_cvar_2022,
    "A4_sharpe_2023": A4_sharpe_2023, "EW_sharpe_2023": EW_sharpe_2023,
    "A4_ret_2024": A4_ret_2024, "A4_sharpe_rising": A4_sharpe_rising,
    "A4_sharpe_stable": A4_sharpe_stable,
}.items() if v is None or np.isnan(v)]
if faltando:
    raise SystemExit(f"valores nao encontrados nas tabelas: {faltando}")

print("Replacement values:")
for k, v in replacements.items():
    print(f"  \\PLACEHOLDER{{{k}}} -> {v}")

# ── Apply substitutions to all .tex section files ────────────────────────────
def replace_placeholders(text: str, repl: dict[str, str]) -> str:
    for key, val in repl.items():
        pattern = r"\\PLACEHOLDER\{" + re.escape(key) + r"\}"
        text = re.sub(pattern, val, text)
    return text

changed = []
for tex_file in LATEX_DIR.glob("*.tex"):
    original = tex_file.read_text(encoding="utf-8")
    updated  = replace_placeholders(original, replacements)
    if updated != original:
        tex_file.write_text(updated, encoding="utf-8")
        n_replaced = sum(
            len(re.findall(r"\\PLACEHOLDER\{" + re.escape(k) + r"\}", original))
            for k in replacements
        )
        changed.append((tex_file.name, n_replaced))
        print(f"  Updated: {tex_file.name} ({n_replaced} replacements)")

# Check for remaining placeholders
remaining = []
for tex_file in LATEX_DIR.glob("*.tex"):
    text = tex_file.read_text(encoding="utf-8")
    found = re.findall(r"\\PLACEHOLDER\{[^}]+\}", text)
    if found:
        remaining.extend([(tex_file.name, f) for f in found])

if remaining:
    print(f"\nWARNING: {len(remaining)} unfilled placeholders remain:")
    for fname, ph in remaining:
        print(f"  {fname}: {ph}")
else:
    print("\nAll placeholders filled successfully.")

print("\nDone.")
