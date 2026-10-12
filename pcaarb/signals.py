"""Residual s-scores and the trading rules of Avellaneda & Lee (2010).

For every stock, regress its returns over the last `estimation` days on the factor
returns. The cumulative residual X_t is modelled as an Ornstein-Uhlenbeck process,
fitted as an AR(1): X_{t+1} = a + b X_t + e. That gives

    speed of mean reversion   kappa    = -ln(b) * 252
    equilibrium level         m        = a / (1 - b)
    equilibrium volatility    sigma_eq = sqrt(var(e) / (1 - b^2))
    s-score                   s        = -(m - mean(m)) / sigma_eq

(the residuals of a regression with an intercept sum to zero, so X at the end of the
window is 0 and s measures how far the stock sits from its equilibrium; m is centred
across stocks as in the paper). A large negative s means the stock is cheap relative
to its factors, a large positive s means it is rich.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class Scores:
    s: np.ndarray        # s-score per stock (nan where the fit is not mean-reverting)
    kappa: np.ndarray    # speed of mean reversion per year
    betas: np.ndarray    # (n_stocks, k) factor loadings from the residual regression


def s_scores(stock_returns: np.ndarray, factor_returns: np.ndarray, max_reversion_days: float = 30.0) -> Scores:
    """Score every stock from aligned (days x stocks) and (days x factors) return arrays."""
    t = stock_returns.shape[0]
    x = np.column_stack([np.ones(t), factor_returns])
    coef, *_ = np.linalg.lstsq(x, stock_returns, rcond=None)
    resid = stock_returns - x @ coef
    cum = np.cumsum(resid, axis=0)

    # AR(1) fit on the cumulative residual, vectorised over stocks.
    prev, nxt = cum[:-1], cum[1:]
    prev_mean, next_mean = prev.mean(axis=0), nxt.mean(axis=0)
    cov = ((prev - prev_mean) * (nxt - next_mean)).mean(axis=0)
    var = ((prev - prev_mean) ** 2).mean(axis=0)
    b = cov / var
    a = next_mean - b * prev_mean
    err_var = (nxt - (a + b * prev)).var(axis=0)

    valid = (b > 0) & (b < 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(valid, -np.log(b) * TRADING_DAYS, np.nan)
        m = np.where(valid, a / (1 - b), np.nan)
        sigma_eq = np.where(valid, np.sqrt(err_var / (1 - b**2)), np.nan)
    # Only trade residuals that revert fast enough to be worth holding.
    fast = kappa > TRADING_DAYS / max_reversion_days
    if not fast.any():
        return Scores(s=np.full(stock_returns.shape[1], np.nan), kappa=kappa, betas=coef[1:].T)
    m_centred = m - m[fast].mean()
    s = np.where(fast, -m_centred / sigma_eq, np.nan)
    return Scores(s=s, kappa=kappa, betas=coef[1:].T)


@dataclass(frozen=True)
class Thresholds:
    open_long: float = -1.25    # buy when s falls below this
    close_long: float = -0.50   # exit the long when s rises above this
    open_short: float = 1.25    # sell short when s rises above this
    close_short: float = 0.75   # exit the short when s falls below this


def next_positions(current: np.ndarray, s: np.ndarray, th: Thresholds = Thresholds()) -> np.ndarray:
    """Apply the bang-bang rules to yesterday's positions (+1 long, -1 short, 0 flat).

    A stock without a valid score (nan) is closed out.
    """
    pos = current.copy()
    nan = np.isnan(s)
    s = np.where(nan, 0.0, s)
    pos[(current == 0) & (s < th.open_long)] = 1
    pos[(current == 0) & (s > th.open_short)] = -1
    pos[(current == 1) & (s > th.close_long)] = 0
    pos[(current == -1) & (s < th.close_short)] = 0
    pos[nan] = 0
    return pos
