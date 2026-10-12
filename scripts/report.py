"""Charts and summary numbers for the README (writes into docs/)."""
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcaarb.backtest import Config, metrics, run
from pcaarb.data import daily_returns, load_benchmark, load_prices
from pcaarb.factors import fit_pca

DOCS = Path(__file__).resolve().parent.parent / "docs"
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def main():
    r = daily_returns(load_prices())
    spy = load_benchmark()

    # 1. How much of the co-movement PC1 explains, refit at each month end.
    month_ends = r.resample("ME").last().index
    share = {}
    for d in month_ends:
        w = r.loc[:d].tail(252)
        if len(w) == 252:
            share[d] = fit_pca(w).explained[:5]
    share = pd.DataFrame(share, index=[f"PC{i}" for i in range(1, 6)]).T
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.fill_between(share.index, 0, share["PC1"] * 100, color="#F5A623", alpha=0.85, label="PC1 (market)")
    ax.fill_between(share.index, share["PC1"] * 100, share.sum(axis=1) * 100, color="#2EC4B6", alpha=0.6, label="PC2 to PC5")
    ax.set_ylabel("% of variance explained")
    ax.set_title("How much of 50 large caps' co-movement the top factors explain (trailing year)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(DOCS / "explained_variance.png"); plt.close(fig)

    # 2. Equity curves: baseline vs chosen turnover control, before and after costs.
    base = run(r, Config())
    chosen = run(r, replace(Config(), smoothing=0.25))
    fig, ax = plt.subplots(figsize=(7, 3))
    for res, label, color in ((base, "baseline", "#8A9BAD"), (chosen, "25% smoothing", "#F5A623")):
        ax.plot((1 + res.gross).cumprod(), color=color, lw=1, ls="--", label=f"{label}, before costs")
        ax.plot((1 + res.net).cumprod(), color=color, lw=1.6, label=f"{label}, after 5 bp costs")
    ax.axvline(pd.Timestamp("2020-01-01"), color="#C0717A", lw=1)
    ax.text(pd.Timestamp("2020-02-01"), ax.get_ylim()[1] * 0.98, "out of sample →", color="#C0717A", va="top")
    ax.set_ylabel("growth of $1 (unit gross)")
    ax.set_title("PCA stat-arb on 50 US large caps, 2013 to 2025")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(DOCS / "equity_curves.png"); plt.close(fig)

    # 3. Rolling one-year Sharpe before costs: the decay of the raw edge.
    roll = base.gross.rolling(252)
    rs = roll.mean() / roll.std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(rs, color="#2EC4B6", lw=1.3)
    ax.axhline(0, color="#495A6B", lw=0.8)
    ax.set_title("Rolling 1-year Sharpe before costs (baseline)")
    fig.tight_layout(); fig.savefig(DOCS / "rolling_sharpe.png"); plt.close(fig)

    turnover = base.weights.diff().abs().sum(axis=1).mean()
    ann_gross = base.gross.mean() * 252
    print("breakeven one-way cost (bp):", round(ann_gross / (turnover * 252) * 1e4, 2))
    for name, res in (("baseline", base), ("smoothing", chosen)):
        for lbl, ser in (("gross", res.gross), ("net", res.net)):
            print(name, lbl, {k: round(float(v), 3) for k, v in metrics(ser, spy).items()})
    print("avg positions long/short:", round(base.n_long.mean(), 1), round(base.n_short.mean(), 1))
    print("PC1 share min/median/max:", share["PC1"].min().round(3), share["PC1"].median().round(3), share["PC1"].max().round(3), share["PC1"].idxmax().date())


if __name__ == "__main__":
    main()
