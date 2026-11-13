"""Does the PCA stat-arb edge depend on the market regime, and does a real-time regime gate help?

Regime signal: the turbulent-regime probability from the Markov-switching variance model
(Student-t) in the regime-switching-vol project, fitted on SPY 2000-2014 and frozen. The
value on day t is the filtered probability given returns up to day t's close, so it is known
when that close's weights are set (see data/regime_prob.csv.gz and that repo's
scripts/export_regime_prob.py).

Everything below was fixed before looking at any result:
  * base strategy = turnover-study choice B (trade 25% toward target), 5 bp costs;
  * four candidates: no gate, flat when Pr(turbulent) > 0.5, scale by Pr(calm),
    and the opposite hypothesis, trade only when Pr(turbulent) > 0.5;
  * choose on 2013-2019 net Sharpe only; report 2020-2025 for every candidate;
  * diagnostic: base strategy's daily gross return on calm vs turbulent days, with a
    Newey-West (5 lags) t-statistic for the difference.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pcaarb.backtest import Config, metrics, run
from pcaarb.data import daily_returns, load_benchmark, load_prices

ROOT = Path(__file__).resolve().parent.parent
DEV_END, TEST_START = "2019-12-31", "2020-01-01"
BASE = replace(Config(), smoothing=0.25)


def load_regime() -> pd.Series:
    p = pd.read_csv(ROOT / "data" / "regime_prob.csv.gz", index_col=0, parse_dates=True).squeeze("columns")
    return p.rename("p_turbulent")


def newey_west_diff(y: pd.Series, d: pd.Series, lags: int = 5) -> tuple[float, float]:
    """OLS of y on [1, d]; returns (coefficient on d, HAC t-statistic)."""
    X = np.column_stack([np.ones(len(d)), d.to_numpy(float)])
    yv = y.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    u = yv - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    return beta[1], beta[1] / np.sqrt(cov[1, 1])


def main():
    r = daily_returns(load_prices())
    spy = load_benchmark()
    p = load_regime().reindex(r.index).ffill()
    candidates = {
        "A. no gate (turnover choice B)": None,
        "B. flat when Pr(turbulent) > 0.5": (p <= 0.5).astype(float),
        "C. scale by Pr(calm)": 1 - p,
        "D. trade only when Pr(turbulent) > 0.5": (p > 0.5).astype(float),
    }
    rows, results = [], {}
    for name, gate in candidates.items():
        res = run(r, BASE, exposure=gate)
        results[name] = res
        for period, sl in (("dev 2013-2019", slice(None, DEV_END)), ("test 2020-2025", slice(TEST_START, None))):
            m = metrics(res.net.loc[sl], spy)
            g = metrics(res.gross.loc[sl])
            w = res.weights.loc[sl]
            rows.append({"candidate": name, "period": period, "gross Sharpe": g["Sharpe"], "net Sharpe": m["Sharpe"],
                         "net annual return": m["annual return"], "annual volatility": m["annual volatility"],
                         "max drawdown": m["max drawdown"], "beta": m["beta to S&P 500"],
                         "daily turnover": w.diff().abs().sum(axis=1).mean(),
                         "avg gross exposure": w.abs().sum(axis=1).mean()})
    table = pd.DataFrame(rows)
    dev = table[table.period == "dev 2013-2019"].set_index("candidate")
    chosen = dev["net Sharpe"].idxmax()

    # Diagnostic: is the base strategy's edge different in the two regimes?
    base = results["A. no gate (turnover choice B)"]
    decided = base.weights.index                      # close at which each day's weights were set
    turb = (p.reindex(decided) > 0.5).astype(float)
    turb.index = base.gross.index                     # align to the day the return is earned
    diag = []
    for period, sl in (("dev 2013-2019", slice(None, DEV_END)), ("test 2020-2025", slice(TEST_START, None)),
                       ("full 2013-2025", slice(None, None))):
        y, d = base.gross.loc[sl], turb.loc[sl]
        diff, t = newey_west_diff(y, d)
        for label, mask in (("calm", d == 0), ("turbulent", d == 1)):
            mm = metrics(y[mask])
            diag.append({"period": period, "regime": label, "days": int(mask.sum()),
                         "gross annual return": mm["annual return"], "gross Sharpe": mm["Sharpe"],
                         "turbulent minus calm, bp/day": 1e4 * diff, "Newey-West t": t})
    diag = pd.DataFrame(diag)

    (ROOT / "docs").mkdir(exist_ok=True)
    table.to_csv(ROOT / "docs" / "regime_gate_study.csv", index=False, float_format="%.4f")
    diag.to_csv(ROOT / "docs" / "regime_edge_by_regime.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print("\nChosen on the development period:", chosen)
    print()
    print(diag.round(3).to_string(index=False))


if __name__ == "__main__":
    main()


def chart():
    """Cumulative net return with and without the chosen gate, turbulent days shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = daily_returns(load_prices())
    p = load_regime().reindex(r.index).ffill()
    a = run(r, BASE)
    b = run(r, BASE, exposure=(p <= 0.5).astype(float))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    turb = (p.reindex(a.net.index) > 0.5)
    ax.fill_between(turb.index, 0, 1, where=turb.to_numpy(), transform=ax.get_xaxis_transform(),
                    color="#d9d9d9", step="mid", label="turbulent regime (Pr > 0.5)")
    ax.plot((1 + a.net).cumprod(), color="#888888", lw=1.4, label="no gate")
    ax.plot((1 + b.net).cumprod(), color="#1f4e79", lw=1.6, label="flat when turbulent (chosen on 2013-19)")
    ax.axvline(pd.Timestamp(TEST_START), color="black", lw=0.8, ls="--")
    ax.text(pd.Timestamp(TEST_START), ax.get_ylim()[1], "  out of sample", va="top", fontsize=9)
    ax.set_title("PCA stat-arb, $1 gross, after 5 bp costs", fontsize=11)
    ax.set_ylabel("growth of $1")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "regime_gate.png", dpi=150)


def sensitivity():
    """Robustness only (not used for any choice): gate threshold 0.3/0.5/0.7 at 2 and 5 bp."""
    r = daily_returns(load_prices())
    p = load_regime().reindex(r.index).ffill()
    rows = []
    for cost in (2.0, 5.0):
        cfg = replace(BASE, cost_bps=cost)
        for name, gate in [("no gate", None)] + [(f"flat if p > {x}", (p <= x).astype(float)) for x in (0.3, 0.5, 0.7)]:
            res = run(r, cfg, exposure=gate)
            for period, sl in (("dev", slice(None, DEV_END)), ("test", slice(TEST_START, None))):
                rows.append({"rule": name, "cost": cost, "period": period,
                             "net_sharpe": round(metrics(res.net.loc[sl])["Sharpe"], 3),
                             "gross_sharpe": round(metrics(res.gross.loc[sl])["Sharpe"], 3)})
    pd.DataFrame(rows).to_csv(ROOT / "docs" / "regime_gate_sensitivity.csv", index=False)


if __name__ == "__main__" and "--chart" in __import__("sys").argv:
    chart()
    sensitivity()
