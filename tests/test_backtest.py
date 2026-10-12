import numpy as np
import pandas as pd

from pcaarb.backtest import Config, metrics, run

CFG = Config(pca_window=60, estimation=30, k=2)


def _returns(n_days=140, n_stocks=10, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    market = rng.normal(0, 0.01, n_days)
    data = market[:, None] * rng.uniform(0.6, 1.4, n_stocks) + rng.normal(0, 0.01, (n_days, n_stocks))
    return pd.DataFrame(data, index=idx, columns=[f"S{i}" for i in range(n_stocks)])


def test_no_lookahead_future_returns_do_not_change_past_weights():
    r = _returns()
    base = run(r, CFG).weights
    shocked = r.copy()
    shocked.iloc[-20:] = shocked.iloc[-20:] * 50          # wildly change the last 20 days
    after = run(shocked, CFG).weights
    cutoff = r.index[-21]
    pd.testing.assert_frame_equal(base.loc[:cutoff], after.loc[:cutoff])


def test_weights_have_unit_gross_exposure_when_invested():
    w = run(_returns(), CFG).weights
    gross = w.abs().sum(axis=1)
    invested = gross > 0
    np.testing.assert_allclose(gross[invested], 1.0)


def test_returns_are_dated_on_the_day_they_are_earned():
    r = _returns()
    res = run(r, CFG)
    first_weight_day = res.weights.index[0]
    assert res.net.index[0] == r.index[r.index.get_loc(first_weight_day) + 1]


def test_metrics_on_a_constant_series():
    m = metrics(pd.Series([0.001] * 252))
    assert np.isclose(m["annual return"], 0.252)
    assert m["max drawdown"] == 0
