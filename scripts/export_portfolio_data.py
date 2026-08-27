"""
Gera o JSON que alimenta o case "Carteira consciente do ciclo de juros" do
portfolio (github.com/GustavoJannuzzi/jannuzzi-portfolio).

Le APENAS artefatos ja' gravados pelo pipeline (NB-04 a NB-09) e nao recalcula
modelo nenhum. As unicas contas feitas aqui sao agregacao e reducao: montar as
curvas de patrimonio a partir dos retornos diarios gravados e reduzi-las a um
tamanho que cabe numa pagina estatica.

Rodar DEPOIS de run_nb07.py e run_nb08_09.py.

Uso:
    python scripts/export_portfolio_data.py [--out CAMINHO]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
PROCESSED = ROOT / "data" / "processed"
SAIDA_PADRAO = ROOT.parent / "portfolio" / "src" / "content" / "quant" / "regime-portfolio.json"

# Cada configuracao da escada da ablacao, com os tres tratamentos explicitos.
# `arm` e' o que a pagina usa para desenhar a escada: o leitor precisa ver qual
# peca entrou em cada degrau, nao so' o nome do experimento.
CONFIGS = [
    {
        "id": "A1",
        "kind": "agent",
        "conservatism": False,
        "reward": "sharpe",
        "macro": False,
    },
    {
        "id": "A2",
        "kind": "agent",
        "conservatism": True,
        "reward": "sharpe",
        "macro": False,
    },
    {
        "id": "A3",
        "kind": "agent",
        "conservatism": True,
        "reward": "cvar",
        "macro": False,
    },
    {
        "id": "A4",
        "kind": "agent",
        "conservatism": True,
        "reward": "cvar",
        "macro": True,
    },
    {"id": "EqualWeight", "kind": "benchmark"},
    {"id": "RiskParity", "kind": "benchmark"},
    {"id": "MVP", "kind": "benchmark"},
]

# Quantos pontos de cada curva de patrimonio sobrevivem ao downsample.
PONTOS_CURVA = 200


class Compacto:
    """Marca um valor para sair do json.dumps numa linha so'."""

    def __init__(self, valor) -> None:
        self.valor = valor


class CodificadorCompacto(json.JSONEncoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._trechos: dict[str, str] = {}

    def default(self, o):
        if isinstance(o, Compacto):
            marcador = f"@@compacto-{len(self._trechos)}@@"
            self._trechos[marcador] = json.dumps(
                o.valor, ensure_ascii=False, separators=(",", ":")
            )
            return marcador
        return super().default(o)

    def encode(self, o) -> str:
        texto = super().encode(o)
        for marcador, trecho in self._trechos.items():
            texto = texto.replace(f'"{marcador}"', trecho)
        return texto


def arred(valor, casas: int = 4):
    if valor is None:
        return None
    valor = float(valor)
    return None if np.isnan(valor) else round(valor, casas)


def chave_por_prefixo(indice, prefixo: str) -> str:
    """Acha o rotulo completo de uma estrategia ("A4 (CQL+CVaR+Macro) ★")."""
    for idx in indice:
        texto = str(idx)
        if texto.startswith(prefixo) or prefixo in texto:
            return texto
    raise KeyError(f"{prefixo} nao encontrado em {list(indice)}")


LABEL_BENCH = {"EqualWeight": "1/N", "MVP": "Variance", "RiskParity": "Risk Parity"}


def le_main() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "main_results.csv", index_col=0)
    df.columns = ["annReturn", "annVol", "sharpe", "sortino", "cvar95", "maxDrawdown", "calmar"]
    return df


def curva(serie: pd.Series) -> dict:
    """Patrimonio acumulado, base 100, reduzido preservando os extremos."""
    valores = 100 * (1 + serie).cumprod().values
    valores = np.concatenate([[100.0], valores])
    total = len(valores)
    passo = max(1, total / PONTOS_CURVA)

    indices: set[int] = {0, total - 1}
    inicio = 0.0
    while inicio < total:
        fim = min(total, int(inicio + passo) or 1)
        janela = range(int(inicio), max(fim, int(inicio) + 1))
        indices.add(max(janela, key=lambda i: abs(valores[i] - 100.0)))
        inicio += passo

    return {
        "bars": total,
        "points": Compacto([[i, round(float(valores[i]), 2)] for i in sorted(indices)]),
    }


def commit_atual() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args()

    for exigido in ("main_results.csv", "regime_conditional.csv", "subperiod_results.csv"):
        if not (TABLES / exigido).is_file():
            raise SystemExit(f"falta {exigido}: rode run_nb07.py e run_nb08_09.py antes")

    main_df = le_main()
    retornos = pd.read_parquet(PROCESSED / "ablation_returns.parquet")
    bench = pd.read_parquet(PROCESSED / "benchmark_returns.parquet")
    resumo_mdp = pd.read_csv(TABLES / "mdp_summary.csv").set_index("Metric")["Value"]
    regime_stats = pd.read_csv(TABLES / "regime_stats.csv")
    grid = pd.read_csv(TABLES / "grid_search_results.csv")
    reg_df = pd.read_csv(TABLES / "regime_conditional.csv", index_col=[0, 1])
    sub_df = pd.read_csv(TABLES / "subperiod_results.csv", index_col=[0, 1])
    dados = json.loads((PROCESSED / "data_summary.json").read_text(encoding="utf-8"))

    # A janela comum e' a mesma que o NB-08 usa para as tabelas: so' os dias em
    # que agente e benchmarks produzem retorno. Sem isso o grafico mostraria a
    # curva do agente indo 28 pregoes alem da dos benchmarks.
    # O NB-07 ja' grava as colunas de benchmark dentro de ablation_returns,
    # recortadas na janela de avaliacao; juntar as duas fontes direto colide
    # nos mesmos nomes. Vale a versao que ja' esta' la', e do arquivo de
    # benchmarks vem so' o que faltar.
    novas = [c for c in bench.columns if c not in retornos.columns]
    juntos = (retornos.join(bench[novas], how="outer") if novas else retornos.copy()).sort_index()
    colunas = [c["id"] for c in CONFIGS if c["id"] in juntos.columns]
    comum = juntos[colunas].dropna(how="any")

    estrategias = []
    for cfg in CONFIGS:
        ident = cfg["id"]
        if ident not in comum.columns:
            continue
        rotulo = chave_por_prefixo(main_df.index, ident if cfg["kind"] == "agent" else LABEL_BENCH[ident])
        linha = main_df.loc[rotulo]
        estrategias.append(
            {
                **{k: v for k, v in cfg.items() if k != "id"},
                "id": ident,
                "metrics": {k: arred(linha[k]) for k in main_df.columns},
                "equity": curva(comum[ident]),
            }
        )

    regimes = []
    for regime in ["Rising", "Stable", "Falling"]:
        bloco = {"regime": regime, "strategies": {}}
        for ident in ["A4", "EqualWeight", "MVP"]:
            rotulo = chave_por_prefixo(
                reg_df.loc[regime].index, ident if ident == "A4" else LABEL_BENCH[ident]
            )
            linha = reg_df.loc[(regime, rotulo)]
            bloco["strategies"][ident] = {
                "sharpe": arred(linha.iloc[0], 3),
                "cvar95": arred(linha.iloc[1]),
                "annReturn": arred(linha.iloc[2]),
            }
        regimes.append(bloco)

    subperiodos = []
    for periodo in sub_df.index.get_level_values(0).unique():
        bloco = {"period": str(periodo), "strategies": {}}
        for ident in ["A4", "EqualWeight", "MVP"]:
            rotulo = chave_por_prefixo(
                sub_df.loc[periodo].index, ident if ident == "A4" else LABEL_BENCH[ident]
            )
            linha = sub_df.loc[(periodo, rotulo)]
            bloco["strategies"][ident] = {
                "sharpe": arred(linha.iloc[2], 3),
                "cvar95": arred(linha.iloc[4]),
                "annReturn": arred(linha.iloc[0]),
            }
        subperiodos.append(bloco)

    payload = {
        "provenance": {
            "kind": "recorded",
            "repo": "github.com/GustavoJannuzzi/Regime-Aware-Portfolio-Optimization",
            "commit": commit_atual(),
            "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pipeline": "python scripts/run_nb07.py && python scripts/run_nb08_09.py",
        },
        "setup": {
            "market": "B3",
            "tickers": [t.replace(".SA", "") for t in dados["tickers"]],
            "dataStart": dados["date_range"]["start"],
            "dataEnd": dados["date_range"]["end"],
            "observations": dados["n_obs"],
            "stateDim": int(resumo_mdp["state_dim"]),
            "actionDim": int(resumo_mdp["action_dim"]),
            "nTrain": int(resumo_mdp["N_train"]),
            "nEval": int(resumo_mdp["N_eval"]),
            "trainStart": "2016-01-13",
            "trainEnd": "2021-12-30",
            "evalStart": "2022-01-03",
            "evalEnd": "2024-12-27",
            "commonStart": str(comum.index.min().date()),
            "commonEnd": str(comum.index.max().date()),
            "nCommon": int(len(comum)),
            "behavioralPolicy": "MVP",
            "gamma": 0.99,
            "lambda": 0.5,
            "cvarAlpha": 0.95,
            "conservativeWeight": float(
                json.loads((ROOT / "outputs" / "models" / "best_hparams.json").read_text())[
                    "conservative_weight"
                ]
            ),
            "gradientSteps": 5000,
            "seed": 42,
            "benchmarkFolds": 35,
            "trainWindowDays": 252,
            "testWindowDays": 63,
        },
        "strategies": estrategias,
        "regimes": regimes,
        "subperiods": subperiodos,
        # O mercado por regime, no periodo inteiro (NB-03). Nao e' resultado do
        # agente: e' o pano de fundo que explica por que o regime importa.
        "marketByRegime": [
            {
                "regime": linha["Regime"],
                "days": int(linha["N Days"]),
                "sharpe": arred(linha["Sharpe (ann.)"], 3),
                "avgSelic": arred(linha["Avg SELIC (%)"], 2),
            }
            for _, linha in regime_stats.iterrows()
        ],
        "gridSearch": [
            {
                "conservativeWeight": arred(linha["conservative_weight"], 3),
                "sharpe": arred(linha["oos_sharpe"], 3),
                "cvar95": arred(linha["oos_cvar5"]),
            }
            for _, linha in grid.iterrows()
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    texto = CodificadorCompacto(ensure_ascii=False, indent=2).encode(payload)
    args.out.write_text(texto + "\n", encoding="utf-8")

    print(f"escrito {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    print(f"\njanela comum: {payload['setup']['commonStart']} a {payload['setup']['commonEnd']} "
          f"({payload['setup']['nCommon']} pregoes)")
    print("\nranking por Sharpe:")
    for e in sorted(estrategias, key=lambda x: -(x["metrics"]["sharpe"] or -99)):
        m = e["metrics"]
        print(
            f"  {e['id']:<12} sharpe {m['sharpe']:+.4f}  cvar {m['cvar95']:.4f}  "
            f"maxDD {m['maxDrawdown']:+.4f}  calmar {m['calmar']:+.4f}  "
            f"annRet {m['annReturn']:+.4f}"
        )


if __name__ == "__main__":
    main()
