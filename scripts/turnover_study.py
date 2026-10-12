"""Turnover study: pick a turnover control on 2013-2019, report it on 2020-2025.

The candidates are fixed in advance. Only the development-period net Sharpe is used to
choose; the test-period numbers are reported for every candidate so nothing is hidden.
"""
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pcaarb.backtest import Config, metrics, run
from pcaarb.data import daily_returns, load_benchmark, load_prices
from pcaarb.signals import Thresholds

DEV_END, TEST_START = "2019-12-31", "2020-01-01"
BASE = Config()
CANDIDATES = {
    "A. baseline (daily, full trades)": BASE,
    "B. smoothing: trade 25% toward target": replace(BASE, smoothing=0.25),
    "C. wider entry bands (|s| > 2.0)": replace(BASE, thresholds=Thresholds(open_long=-2.0, open_short=2.0)),
    "D. weekly rebalancing": replace(BASE, rebalance_every=5),
}


def main():
    r = daily_returns(load_prices())
    spy = load_benchmark()
    rows = []
    for name, cfg in CANDIDATES.items():
        res = run(r, cfg)
        turnover = res.weights.diff().abs().sum(axis=1).mean()
        for period, sl in (("dev 2013-2019", slice(None, DEV_END)), ("test 2020-2025", slice(TEST_START, None))):
            m = metrics(res.net.loc[sl], spy)
            g = metrics(res.gross.loc[sl])
            rows.append({"candidate": name, "period": period, "gross Sharpe": g["Sharpe"], "net Sharpe": m["Sharpe"],
                         "net annual return": m["annual return"], "max drawdown": m["max drawdown"],
                         "beta": m["beta to S&P 500"], "daily turnover": turnover})
    table = pd.DataFrame(rows)
    dev = table[table.period == "dev 2013-2019"].set_index("candidate")
    chosen = dev["net Sharpe"].idxmax()
    out = Path(__file__).resolve().parent.parent / "docs" / "turnover_study.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("\nChosen on the development period:", chosen)


if __name__ == "__main__":
    main()
