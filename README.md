# PCA statistical arbitrage on US large caps

A from-scratch, walk-forward replication of the classic Avellaneda & Lee (2010) strategy:
extract risk factors with PCA, strip them out of each stock's returns, and trade the
mean reversion of what is left, fully hedged against the factors. Tested on 50 US large
caps from 2013 to 2025 with transaction costs, a no-look-ahead test suite and an honest
out-of-sample split.

**Short version:** the method works as designed. The book is market-neutral (beta 0.03),
and it has a real edge before costs (Sharpe 0.48, 0.84 in 2013 to 2019). But the edge is
thin, it decays after 2020, and at 5 bp per trade it does not survive costs in large caps.
Breakeven is about 2.4 bp. That result, and why, is the point of this repository.

![Equity curves](docs/equity_curves.png)

## How it works

Every day, using only data available at that day's close:

1. **Factors.** Standardise the last 252 days of returns and take the top 5 eigenvectors
   of their correlation matrix. Dividing each eigenvector by the stocks' volatilities gives
   tradable **eigenportfolios**; their returns are the factor returns.
2. **Residuals.** Regress each stock's last 60 days of returns on the 5 factor returns.
   The residual is the part of the move the factors do not explain.
3. **Mean reversion.** Fit the cumulative residual as an Ornstein–Uhlenbeck process (as an
   AR(1)) to get its speed of mean reversion κ, equilibrium level and equilibrium volatility.
   Only stocks that revert within about 30 days (κ > 252/30) are tradable.
4. **Signal.** The **s-score** measures how far a stock sits from equilibrium, in standard
   deviations. Open a long when s < −1.25, a short when s > 1.25; close longs above −0.50
   and shorts below 0.75.
5. **Hedge.** Compute the stock book's exposure to every factor and offset it with the
   eigenportfolios, then scale to $1 of gross exposure.
6. **Costs.** Charge 5 bp one way on every dollar traded.

| Module | Role |
| --- | --- |
| `pcaarb/data.py` | Price download and cache, 50-stock universe, SPY benchmark |
| `pcaarb/factors.py` | PCA factors and eigenportfolios |
| `pcaarb/signals.py` | Residual regression, OU fit, s-scores, trading rules |
| `pcaarb/backtest.py` | Walk-forward backtest, factor hedge, turnover controls, metrics |
| `scripts/turnover_study.py` | Out-of-sample comparison of turnover controls |
| `scripts/report.py` | Charts and summary numbers in `docs/` |
| `tests/` | No-look-ahead, return dating, OU fit, trading rules, factor properties |

## Results (Jan 2013 to Sep 2025, $1 gross)

| | Before costs | After 5 bp costs |
| --- | --- | --- |
| Annual return | 2.1% | −2.2% |
| Annual volatility | 4.3% | 4.3% |
| Sharpe | **0.48** | −0.52 |
| Max drawdown | −8.9% | −29.4% |
| Beta to S&P 500 | 0.03 | 0.03 |

About 10 longs and 8 shorts are open on a typical day, and the book turns over 34% of its
gross exposure daily, roughly 85 times a year. That is why the edge cannot pay for 5 bp.

### What the factors look like

![Explained variance](docs/explained_variance.png)

- **PC1 is the market.** It explains 20% of the co-movement in calm markets and 66% in the
  2020 crash, when correlations jump.
- **PC2 changes with the regime:** energy against consumer growth in 2015, cyclicals
  against defensives in March 2020, mega-cap tech against consumer staples in 2025.

### Turnover study, with an honest out-of-sample split

Four turnover controls were fixed in advance. The choice uses **only 2013 to 2019**, and the
2020 to 2025 numbers are shown for all of them (`docs/turnover_study.csv`).

| Candidate | Daily turnover | Net Sharpe 2013–19 (choose) | Net Sharpe 2020–25 (test) |
| --- | --- | --- | --- |
| A. Baseline | 34% | −0.35 | −0.67 |
| **B. Trade 25% toward target (chosen)** | 15% | **+0.10** | −0.28 |
| C. Wider entry bands (\|s\| > 2) | 34% | −0.31 | −0.01 |
| D. Weekly rebalancing | 16% | 0.00 | −0.55 |

Smoothing halves turnover and is the only control that is positive in the development
period, so it is the choice. Out of sample it still loses money after costs. Candidate C
looks best on 2020 to 2025, but picking it after seeing those years would be look-ahead
bias, so it is reported, not chosen.

![Rolling Sharpe](docs/rolling_sharpe.png)

The raw edge itself fades: gross Sharpe falls from 0.84 (2013 to 2019) to 0.19 (2020 to 2025),
consistent with the crowding and decay of residual-reversal strategies in liquid large caps.

## Extension: does the edge depend on the market regime?

My own follow-up question, using the hidden Markov model from my
[regime-switching-vol](https://github.com/chawla2255/regime-switching-vol) project. That model
was fitted on SPY 2000-2014 and frozen; its **filtered** turbulent-regime probability on day t
uses returns up to that close only, so it can set that close's weights without look-ahead.
Everything below was fixed before looking at results (`scripts/regime_gate_study.py`):
the base is turnover choice B, the four candidates are listed in the table, the choice uses
2013-2019 only, and 2020-2025 is reported for all of them.

![Regime gate](docs/regime_gate.png)

| Candidate (5 bp costs) | Net Sharpe 2013-19 (choose) | Net Sharpe 2020-25 (test) | Gross Sharpe 2020-25 | Daily turnover 2020-25 |
| --- | --- | --- | --- | --- |
| A. No gate | 0.10 | −0.28 | 0.16 | 15% |
| **B. Flat when Pr(turbulent) > 0.5 (chosen)** | **0.20** | **−0.18** | 0.25 | 11% |
| C. Scale by Pr(calm) | 0.20 | −0.19 | 0.24 | 10% |
| D. Trade only when turbulent (opposite hypothesis) | −0.30 | −0.31 | −0.03 | 6% |

**Where the edge lives.** The base strategy's gross Sharpe is 0.94 on calm days and −0.30 on
turbulent days in 2013-2019, and 0.41 vs −0.19 out of sample: the same sign in both periods.
The difference is not statistically significant, though (Newey-West t = −1.1 over 2013-2025),
so this is evidence, not proof (`docs/regime_edge_by_regime.csv`).

**What the gate buys.** Out of sample it raises gross Sharpe from 0.16 to 0.25, cuts turnover
by 30% and brings beta from 0.03 to 0.00, but at 5 bp the strategy still loses money. As a
robustness check (not used for any choice), gating at thresholds of 0.3, 0.5 and 0.7 beats no
gate in both periods at both 2 bp and 5 bp; at 2 bp the gated book is slightly positive out of
sample (net Sharpe 0.08 vs −0.02) (`docs/regime_gate_sensitivity.csv`).

**Reading.** Residual reversal works when correlations are stable and fails when the market
factor takes over, which is what the turbulent regime picks up. A regime filter removes the
worst periods but cannot create an edge that costs have already eaten.

## Limitations

- **Survivorship bias.** The universe is today's large caps, so stocks that failed or were
  delisted are missing. The strategy is market-neutral, which limits the effect but does not
  remove it.
- **Costs are a flat 5 bp.** Real costs vary with liquidity, volatility and trade size.
- **Close-to-close execution.** Signals and fills both use closing prices.
- **No borrow costs** for the short side.

## Ideas to take it further

- Volume-weighted returns in the residual model (the "trading time" variant from the paper)
- A broader, cheaper-to-trade-around universe such as mid caps, where residuals are less crowded
- Cost-aware position sizing: trade only when the expected reversion exceeds the cost

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 11 tests
PYTHONPATH=. python scripts/turnover_study.py
PYTHONPATH=. python scripts/report.py       # writes charts to docs/
```

The first run downloads prices from Yahoo Finance into `data/` (not committed).

## Reference

Avellaneda, M. and Lee, J.-H. (2010). *Statistical arbitrage in the U.S. equities market.*
Quantitative Finance, 10(7), 761–782.

---

Yuvraj Chawla · M.S. Quantitative Finance, Northeastern University · [yuvrajchawla.si](https://yuvrajchawla.si)
