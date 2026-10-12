"""Daily walk-forward backtest of the PCA statistical-arbitrage strategy.

At each close t the strategy sees returns up to and including day t only:
  1. fit PCA factors on the trailing `pca_window` days,
  2. score residuals over the trailing `estimation` days,
  3. update long/short states with the trading rules,
  4. hedge the stock book's exposure to every factor with the eigenportfolios,
  5. scale to 1 dollar of gross exposure.
Those weights are held over day t + 1, and trading costs are charged on the change in weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .factors import fit_pca
from .signals import Thresholds, next_positions, s_scores

TRADING_DAYS = 252


@dataclass(frozen=True)
class Config:
    pca_window: int = 252        # days used to estimate the factors
    estimation: int = 60         # days used to fit residuals and s-scores
    k: int = 5                   # number of PCA factors
    max_reversion_days: float = 30.0
    cost_bps: float = 5.0        # one-way cost per dollar traded, in basis points
    thresholds: Thresholds = field(default_factory=Thresholds)


@dataclass
class Result:
    weights: pd.DataFrame        # weights decided at each close (held the next day)
    gross: pd.Series             # daily return before costs
    costs: pd.Series             # daily trading cost
    net: pd.Series               # daily return after costs
    n_long: pd.Series
    n_short: pd.Series


def run(returns: pd.DataFrame, cfg: Config = Config()) -> Result:
    tickers = list(returns.columns)
    col = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    states = np.zeros(n)
    rows, dates, longs, shorts = [], [], [], []

    for t in range(cfg.pca_window - 1, len(returns) - 1):
        window = returns.iloc[t - cfg.pca_window + 1 : t + 1]
        model = fit_pca(window, cfg.k)
        idx = np.array([col[c] for c in model.tickers])
        est = window[model.tickers].iloc[-cfg.estimation :]
        sc = s_scores(est.to_numpy(), model.factor_returns(est).to_numpy(), cfg.max_reversion_days)

        s = np.full(n, np.nan)
        s[idx] = sc.s
        states = next_positions(states, s, cfg.thresholds)

        w = states.astype(float)
        betas = np.zeros((n, cfg.k))
        betas[idx] = sc.betas
        q = np.zeros((n, cfg.k))
        q[idx] = model.weights
        w = w - q @ (w @ betas)              # neutralise exposure to every factor
        gross_exposure = np.abs(w).sum()
        if gross_exposure > 0:
            w = w / gross_exposure

        rows.append(w)
        dates.append(returns.index[t])
        longs.append(int((states == 1).sum()))
        shorts.append(int((states == -1).sum()))

    weights = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=tickers)
    nxt = returns.shift(-1).loc[weights.index].fillna(0.0)
    gross = (weights * nxt).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    costs = turnover * cfg.cost_bps / 1e4
    # Returns are dated on the day they are earned.
    hold_dates = returns.index[returns.index.get_indexer(weights.index) + 1]
    gross.index = costs.index = hold_dates
    net = gross - costs
    n_long = pd.Series(longs, index=hold_dates)
    n_short = pd.Series(shorts, index=hold_dates)
    return Result(weights, gross, costs, net, n_long, n_short)


def metrics(daily: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Annualised performance statistics for a daily return series."""
    daily = daily.dropna()
    ann_ret = daily.mean() * TRADING_DAYS
    ann_vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    wealth = (1 + daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    out = {
        "annual return": ann_ret,
        "annual volatility": ann_vol,
        "Sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max drawdown": drawdown.min(),
        "hit rate": (daily > 0).mean(),
    }
    if benchmark is not None:
        b = benchmark.reindex(daily.index).dropna()
        d = daily.loc[b.index]
        out["beta to S&P 500"] = np.cov(d, b)[0, 1] / b.var()
        out["correlation to S&P 500"] = np.corrcoef(d, b)[0, 1]
    return out
