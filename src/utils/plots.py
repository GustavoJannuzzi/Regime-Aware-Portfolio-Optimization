"""Publication-ready matplotlib figures.

All figures follow RBFin standards:
  - dpi=300, fontsize=11
  - figsize=(6.5, 4) single column | (13, 4) full width
  - No internal title
  - Clean legend
  - Saved to outputs/figures/*.pdf and *.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


FIGURES_DIR = Path(__file__).parents[2] / "outputs" / "figures"
STYLE = {
    "figure.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def save_figure(name: str, fig: plt.Figure | None = None) -> None:
    """Save current (or given) figure to outputs/figures/ in PDF and PNG.

    Parameters
    ----------
    name : str
        Output filename without extension.
    fig : plt.Figure, optional
        Figure to save (defaults to plt.gcf()).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: outputs/figures/{name}.pdf and .png")


def plot_wealth_curves(
    returns_dict: dict[str, pd.Series],
    figsize: tuple = (6.5, 4),
) -> plt.Figure:
    """Fig. 1: Cumulative wealth curves for all models.

    Parameters
    ----------
    returns_dict : dict
        Keys = model names, values = daily return Series.

    Returns
    -------
    plt.Figure
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        for name, rets in returns_dict.items():
            wealth = (1 + rets).cumprod()
            ax.plot(wealth.index, wealth.values, label=name)
        ax.set_xlabel("Date")
        ax.set_ylabel("Wealth Index")
        ax.legend()
        fig.tight_layout()
    return fig


def plot_drawdown(
    returns_dict: dict[str, pd.Series],
    figsize: tuple = (6.5, 4),
) -> plt.Figure:
    """Fig. 2: Maximum drawdown over time for all models."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        for name, rets in returns_dict.items():
            wealth = (1 + rets).cumprod()
            peak = wealth.cummax()
            dd = (wealth - peak) / peak
            ax.fill_between(dd.index, dd.values, 0, alpha=0.4, label=name)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend()
        fig.tight_layout()
    return fig
