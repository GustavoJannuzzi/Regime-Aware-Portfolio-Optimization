"""CQL agent wrapper using d3rlpy.

Wraps d3rlpy.algos.CQL for offline training on the portfolio MDP dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import d3rlpy
import numpy as np


OUTPUT_DIR = Path(__file__).parents[2] / "outputs"


def build_cql_agent(
    conservative_weight: float = 1.0,
    actor_hidden_units: list[int] | None = None,
    critic_hidden_units: list[int] | None = None,
    batch_size: int = 256,
) -> d3rlpy.algos.CQL:
    """Instantiate a CQL agent with given hyperparameters.

    Parameters
    ----------
    conservative_weight : float
        Alpha — controls conservatism penalty strength.
    actor_hidden_units : list[int]
        Hidden layer sizes for the actor network.
    critic_hidden_units : list[int]
        Hidden layer sizes for the critic network.
    batch_size : int
        Mini-batch size for offline training.

    Returns
    -------
    d3rlpy.algos.CQL
        Configured CQL agent (not yet trained).
    """
    actor_hidden_units = actor_hidden_units or [256, 256]
    critic_hidden_units = critic_hidden_units or [256, 256]

    return d3rlpy.algos.CQL(
        conservative_weight=conservative_weight,
        actor_encoder_factory=d3rlpy.encoders.VectorEncoderFactory(
            hidden_units=actor_hidden_units
        ),
        critic_encoder_factory=d3rlpy.encoders.VectorEncoderFactory(
            hidden_units=critic_hidden_units
        ),
        batch_size=batch_size,
    )


def save_best_hparams(hparams: dict, path: Path | None = None) -> None:
    """Save best hyperparameters to outputs/models/best_hparams.json."""
    path = path or OUTPUT_DIR / "models" / "best_hparams.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(hparams, f, indent=2)
