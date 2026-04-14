"""Benchmark portfolio strategies.

Implements:
  - EqualWeight (1/N)
  - MeanVarianceMinimum (MVP)
  - RiskParity (Equal Risk Contribution)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioStrategy:
    """Base class for all portfolio strategies."""

    def fit(self, train_returns: pd.DataFrame) -> "PortfolioStrategy":
        raise NotImplementedError

    def predict(self, state: pd.DataFrame | None = None) -> np.ndarray:
        raise NotImplementedError


class EqualWeight(PortfolioStrategy):
    """Naive 1/N equal-weight portfolio."""

    def fit(self, train_returns: pd.DataFrame) -> "EqualWeight":
        self.n = train_returns.shape[1]
        return self

    def predict(self, state=None) -> np.ndarray:
        return np.ones(self.n) / self.n


class MeanVarianceMinimum(PortfolioStrategy):
    """Minimum Variance Portfolio (MVP) via quadratic programming."""

    def fit(self, train_returns: pd.DataFrame) -> "MeanVarianceMinimum":
        self.cov = train_returns.cov().values
        self.n = train_returns.shape[1]
        return self

    def predict(self, state=None) -> np.ndarray:
        n = self.n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n
        result = minimize(
            lambda w: w @ self.cov @ w,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x


class RiskParity(PortfolioStrategy):
    """Equal Risk Contribution (Risk Parity) portfolio."""

    def fit(self, train_returns: pd.DataFrame) -> "RiskParity":
        self.cov = train_returns.cov().values
        self.n = train_returns.shape[1]
        return self

    def predict(self, state=None) -> np.ndarray:
        n = self.n
        target_rc = np.ones(n) / n

        def objective(w):
            sigma = np.sqrt(w @ self.cov @ w)
            rc = (w * (self.cov @ w)) / sigma
            return np.sum((rc - target_rc * sigma) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(1e-6, 1)] * n
        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x / result.x.sum()
