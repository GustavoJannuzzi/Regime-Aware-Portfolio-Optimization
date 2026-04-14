"""Ablation study runner — configurations A1 through A4.

A1: DDPG online  + Sharpe + no macro  (RL baseline)
A2: CQL offline  + Sharpe + no macro  (isolates offline paradigm)
A3: CQL offline  + CVaR   + no macro  (isolates reward function)
A4: CQL offline  + CVaR   + macro     (full proposal)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.eval.metrics import compute_all_metrics


ABLATION_CONFIGS = {
    "A1": {"paradigm": "ddpg", "reward": "sharpe", "macro": False},
    "A2": {"paradigm": "cql", "reward": "sharpe", "macro": False},
    "A3": {"paradigm": "cql", "reward": "cvar", "macro": False},
    "A4": {"paradigm": "cql", "reward": "cvar", "macro": True},
}


@dataclass
class AblationResult:
    config: str
    paradigm: str
    reward: str
    macro: bool
    metrics: dict


def run_ablation(
    returns_dict: dict[str, pd.Series],
    weights_dict: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compile ablation results into a comparison table.

    Parameters
    ----------
    returns_dict : dict
        Keys are config names (A1-A4 + benchmarks), values are return Series.
    weights_dict : dict, optional
        Portfolio weight DataFrames for turnover calculation.

    Returns
    -------
    pd.DataFrame
        Rows = configs, columns = metrics.
    """
    rows = []
    for name, rets in returns_dict.items():
        w = weights_dict.get(name) if weights_dict else None
        m = compute_all_metrics(rets, weights=w)
        m["config"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("config")
    return df[["sharpe", "sortino", "cvar", "mdd", "calmar", "turnover"]]
