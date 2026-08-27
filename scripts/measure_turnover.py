"""
Mede giro e concentracao das carteiras que o agente produz.

Nao estava no pipeline e faz falta: o `evaluate()` do NB-07 aplica os pesos
previstos DIA A DIA sobre o retorno do dia, ou seja, o agente rebalanceia a
cada pregao. Os benchmarks rebalanceiam a cada 63 pregoes. Comparar os dois sem
custo de transacao nao e' uma simplificacao neutra, e ate' agora nao havia
numero nenhum no projeto que dissesse o tamanho da diferenca.

Roda inferencia sobre o conjunto de avaliacao, sem treinar nada.

Uso:
    python scripts/measure_turnover.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import d3rlpy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "outputs" / "models"
RANDOM_SEED = 42

# Custo por perna de uma volta completa na B3, em fracao. Corretagem
# institucional + emolumentos + escorregamento, arredondado para baixo: e' uma
# conta de ordem de grandeza, nao uma estimativa de execucao.
CUSTO_POR_GIRO = 0.0010

d3rlpy.seed(RANDOM_SEED)


def para_simplex(w: np.ndarray) -> np.ndarray:
    """Mesma projecao que o NB-07 usa antes de aplicar os pesos."""
    w = np.abs(w)
    soma = w.sum(axis=1, keepdims=True)
    return w / np.where(soma == 0, 1.0, soma)


def mede(caminho: Path, obs: np.ndarray) -> dict:
    agente = d3rlpy.load_learnable(str(caminho), device="cpu")
    pesos = para_simplex(agente.predict(obs))
    giro = np.abs(np.diff(pesos, axis=0)).sum(axis=1)
    return {
        "days": int(len(pesos)),
        "daily_turnover_mean": round(float(giro.mean()), 4),
        "daily_turnover_max": round(float(giro.max()), 4),
        "annualized_turnover": round(float(giro.mean() * 252), 2),
        "annual_cost_estimate": round(float(giro.mean() * 252 * CUSTO_POR_GIRO), 4),
        "mean_max_weight": round(float(pesos.max(axis=1).mean()), 4),
        "days_above_50pct": int((pesos.max(axis=1) > 0.5).sum()),
    }


def main() -> None:
    obs = np.load(ROOT / "data" / "processed" / "eval_dataset.npz")["observations"].astype(
        np.float32
    )
    obs_sem_macro = obs[:, :220]

    # A1 a A3 treinam sem as cinco dimensoes macro; A4 usa o estado inteiro.
    alvos = [
        ("A1", MODELS / "cql_a1.d3", obs_sem_macro),
        ("A2", MODELS / "cql_a2.d3", obs_sem_macro),
        ("A3", MODELS / "cql_a3.d3", obs_sem_macro),
        ("A4", MODELS / "cql_a4.d3", obs),
        ("A4_tuning_run", MODELS / "cql_best.d3", obs),
    ]

    resultado = {}
    for nome, caminho, entrada in alvos:
        if not caminho.is_file():
            print(f"  {nome}: sem arquivo ({caminho.name}), pulando")
            continue
        resultado[nome] = mede(caminho, entrada)
        r = resultado[nome]
        print(
            f"  {nome:<14} giro diario {r['daily_turnover_mean']:.4f}"
            f"  anualizado {r['annualized_turnover']:>6.2f}x"
            f"  custo estimado {r['annual_cost_estimate'] * 100:>5.2f}% a.a."
            f"  peso max medio {r['mean_max_weight']:.3f}"
        )

    if not resultado:
        raise SystemExit("nenhum modelo encontrado em outputs/models/")

    saida = MODELS / "turnover.json"
    saida.write_text(
        json.dumps({"cost_per_turnover": CUSTO_POR_GIRO, "agents": resultado}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {saida}")


if __name__ == "__main__":
    main()
