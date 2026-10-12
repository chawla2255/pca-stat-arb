import numpy as np

from pcaarb.signals import Thresholds, next_positions, s_scores


def _ou_paths(n_days=60, n_stocks=8, b=0.8, seed=1):
    """Cumulative residuals that follow AR(1) paths, returned as their daily changes."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_days + 1, n_stocks))
    for t in range(n_days):
        x[t + 1] = b * x[t] + rng.normal(0, 0.01, n_stocks)
    return np.diff(x, axis=0)


def test_mean_reverting_residuals_get_scores_and_fast_kappa():
    rng = np.random.default_rng(2)
    factors = rng.normal(0, 0.01, (60, 2))
    stocks = factors @ rng.uniform(0.5, 1.5, (2, 8)) + _ou_paths()
    sc = s_scores(stocks, factors)
    assert np.isfinite(sc.s).sum() >= 6
    assert np.nanmedian(sc.kappa) > 252 / 30
    assert sc.betas.shape == (8, 2)


def test_trending_residuals_are_not_traded():
    rng = np.random.default_rng(3)
    factors = rng.normal(0, 0.01, (60, 1))
    trend = np.cumsum(rng.normal(0.002, 0.0005, (60, 4)), axis=0)   # random walk with drift, b near 1
    sc = s_scores(factors + trend, factors)
    assert np.isnan(sc.s).all()


def test_trading_rules_open_and_close():
    th = Thresholds()
    flat = np.zeros(4)
    opened = next_positions(flat, np.array([-1.5, 1.5, 0.0, np.nan]), th)
    assert opened.tolist() == [1, -1, 0, 0]
    held = next_positions(opened, np.array([-0.9, 1.0, 0.0, 0.0]), th)
    assert held.tolist() == [1, -1, 0, 0]          # inside the bands: keep positions
    closed = next_positions(held, np.array([-0.4, 0.5, 0.0, 0.0]), th)
    assert closed.tolist() == [0, 0, 0, 0]
