import numpy as np
import pandas as pd

from pcaarb.factors import fit_pca


def _returns(n_days=300, n_stocks=12, seed=0):
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n_days)
    noise = rng.normal(0, 0.01, (n_days, n_stocks))
    betas = rng.uniform(0.5, 1.5, n_stocks)
    return pd.DataFrame(market[:, None] * betas + noise, columns=[f"S{i}" for i in range(n_stocks)])


def test_loadings_are_orthonormal():
    m = fit_pca(_returns(), k=4)
    np.testing.assert_allclose(m.loadings.T @ m.loadings, np.eye(4), atol=1e-10)


def test_explained_variance_sums_to_one_and_is_sorted():
    m = fit_pca(_returns())
    assert np.isclose(m.explained.sum(), 1.0)
    assert np.all(np.diff(m.eigenvalues) <= 1e-12)


def test_first_factor_is_the_market_with_positive_sign():
    r = _returns()
    m = fit_pca(r, k=1)
    assert (m.loadings[:, 0] > 0).all()
    f = m.factor_returns(r)["PC1"]
    assert np.corrcoef(f, r.mean(axis=1))[0, 1] > 0.9


def test_columns_with_missing_history_are_dropped():
    r = _returns()
    r.iloc[:10, 3] = np.nan
    m = fit_pca(r)
    assert "S3" not in m.tickers and len(m.tickers) == r.shape[1] - 1
