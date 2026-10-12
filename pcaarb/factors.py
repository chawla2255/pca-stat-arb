"""PCA risk factors estimated from a trailing window of returns.

Following Avellaneda & Lee (2010): standardise each stock's returns over the window,
take the eigenvectors of their correlation matrix, and turn each eigenvector into an
"eigenportfolio" by dividing its weights by each stock's volatility. The eigenportfolio
returns are the factor returns; PC1 is close to the market portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorModel:
    tickers: list[str]
    eigenvalues: np.ndarray          # all eigenvalues, largest first
    loadings: np.ndarray             # (n_stocks, k) eigenvectors of the correlation matrix
    weights: np.ndarray              # (n_stocks, k) eigenportfolio weights = loadings / volatility

    @property
    def explained(self) -> np.ndarray:
        """Share of total variance explained by each component (all components)."""
        return self.eigenvalues / self.eigenvalues.sum()

    def factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Returns of the k eigenportfolios over the given dates."""
        f = returns[self.tickers].to_numpy() @ self.weights
        return pd.DataFrame(f, index=returns.index, columns=[f"PC{j + 1}" for j in range(f.shape[1])])


def fit_pca(window: pd.DataFrame, k: int = 5) -> FactorModel:
    """Fit k PCA factors to a window of daily returns (rows = days, columns = stocks).

    Only stocks with a full history in the window are used, so the model never sees a
    partially missing column (for example a stock before its IPO).
    """
    window = window.dropna(axis=1)
    x = window.to_numpy()
    sigma = x.std(axis=0, ddof=1)
    z = (x - x.mean(axis=0)) / sigma
    corr = np.corrcoef(z, rowvar=False)
    vals, vecs = np.linalg.eigh(corr)                 # ascending order
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Fix the sign so each factor has positive average loading (PC1 then reads as "the market").
    vecs = vecs * np.where(vecs.sum(axis=0) < 0, -1.0, 1.0)
    loadings = vecs[:, :k]
    weights = loadings / sigma[:, None]
    return FactorModel(list(window.columns), vals, loadings, weights)
