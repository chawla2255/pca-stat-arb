"""Price data: download once from Yahoo Finance, then read from a local cache.

The universe is 50 US large caps that traded for the whole sample. Picking names that
survived to today is a survivorship bias; the README says so, and the strategy is
market-neutral, which limits (but does not remove) its effect.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

UNIVERSE = [
    # tech and communication
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CSCO", "INTC", "IBM", "QCOM", "TXN", "ADBE", "CRM",
    # financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK",
    # health care
    "JNJ", "PFE", "MRK", "ABT", "UNH", "AMGN", "BMY", "MDT",
    # consumer
    "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "DIS",
    # industrials and energy
    "CAT", "HON", "UNP", "BA", "MMM", "GE", "XOM", "CVX", "COP", "SLB",
]
START, END = "2012-01-01", "2025-09-30"
CACHE = Path(__file__).resolve().parent.parent / "data" / "prices.csv.gz"
BENCH_CACHE = CACHE.with_name("spy.csv.gz")


def download(tickers=UNIVERSE, start=START, end=END) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][list(tickers)].sort_index()
    return prices.dropna(how="all")


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Prices from the cache, downloading them on first use."""
    if refresh or not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True)
        download().to_csv(CACHE)
    return pd.read_csv(CACHE, index_col=0, parse_dates=True)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; the first row is dropped."""
    return prices.pct_change(fill_method=None).iloc[1:]


def load_benchmark(refresh: bool = False) -> pd.Series:
    """Daily returns of SPY (S&P 500 ETF), used only to measure market exposure."""
    if refresh or not BENCH_CACHE.exists():
        import yfinance as yf

        BENCH_CACHE.parent.mkdir(exist_ok=True)
        raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
        raw["Close"].squeeze().rename("SPY").to_csv(BENCH_CACHE)
    spy = pd.read_csv(BENCH_CACHE, index_col=0, parse_dates=True).squeeze("columns")
    return spy.pct_change(fill_method=None).iloc[1:]
