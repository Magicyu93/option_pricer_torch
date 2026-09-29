"""Download everything the pricer needs for a set of underlyings and dates.

Edit the settings below, then run ``python scripts/download_market_data.py``.

* **equity** -- option minute bars joined to the underlying's minute bars, one
  trading day at a time (stocks/ETFs or indices, routed automatically);
* **rates** -- Treasury yields from ``RATES_LOOKBACK_DAYS`` before ``START``, so
  the first day always has a publication to fall back on;
* **dividends** -- ex-dates from ``START`` to ``DIVIDEND_HORIZON_DAYS`` after
  ``END``, since an option needs the dividends paid before it expires.  Index
  underlyings are skipped.

Everything is cached under ``DATA_DIR``, so rerunning after an interruption only
fetches what is missing.  Credentials come from the environment:
``MASSIVE_S3_ACCESS_KEY`` / ``MASSIVE_S3_SECRET_KEY`` for Flat Files,
``MASSIVE_API_KEY`` for the REST API, and ``FRED_API_KEY`` for FRED.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the repo root importable when run as a file, without installing it.
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from data_pipeline.dividends import DividendsDataLoader
from data_pipeline.equity import MassiveHistoricalDataLoader
from data_pipeline.rates import TREASURY_CMT_SERIES, RatesDataLoader

# -- settings ------------------------------------------------------------------
TICKERS = ["SPY", "SPX"]
START = "2026-09-08"
END = "2026-09-11"
DATA_DIR = REPO_ROOT / "market_data"  # fixed, whichever folder you run from

DOWNLOAD_EQUITY = True
DOWNLOAD_RATES = True
DOWNLOAD_DIVIDENDS = True

RATES_PROVIDER = "massive"  # or "fred"
RATES_LOOKBACK_DAYS = 14
DIVIDEND_HORIZON_DAYS = 365
REFRESH = False  # refetch rates and dividends even if cached
# ------------------------------------------------------------------------------


def download_equity(tickers, start, end, data_dir=DATA_DIR):
    """Options joined to their underlying, one trading day at a time.

    Returns the days that failed; a failed day does not stop the others.
    """
    loader = MassiveHistoricalDataLoader(data_dir=data_dir)
    failed = []
    for day in pd.bdate_range(start, end).strftime("%Y-%m-%d"):
        try:
            rows = len(loader.load_market([day], tickers))
            print(f"equity {day}: {rows:,} rows" if rows else f"equity {day}: no data")
        except Exception as exc:
            print(f"equity {day}: FAILED -- {exc}")
            failed.append(day)
    return failed


def download_rates(start, end, data_dir=DATA_DIR):
    """Treasury yields, starting early enough to cover the first day."""
    first = pd.Timestamp(start).date() - dt.timedelta(days=RATES_LOOKBACK_DAYS)
    loader = RatesDataLoader(RATES_PROVIDER, data_dir=data_dir)
    frame = loader.load(TREASURY_CMT_SERIES, first, end, refresh=REFRESH)
    print(f"rates ({RATES_PROVIDER}) {first} to {end}: {len(frame):,} rows")
    return frame


def download_dividends(tickers, start, end, data_dir=DATA_DIR):
    """Declared dividends through the horizon after ``end``; indices skipped."""
    last = pd.Timestamp(end).date() + dt.timedelta(days=DIVIDEND_HORIZON_DAYS)
    equity = MassiveHistoricalDataLoader(data_dir=data_dir, verbose=False)
    loader = DividendsDataLoader(data_dir=data_dir)
    for ticker in tickers:
        if equity.infer_underlying_asset(ticker, date=start) == "indices":
            print(f"dividends {ticker}: index, skipped")
            continue
        frame = loader.load(ticker, start, last, refresh=REFRESH)
        print(f"dividends {ticker} {start} to {last}: {len(frame):,} rows")


if __name__ == "__main__":
    if DOWNLOAD_EQUITY:
        failed = download_equity(TICKERS, START, END)
        if failed:
            print(f"equity failed for {failed}; rerun to retry them")
    if DOWNLOAD_RATES:
        download_rates(START, END)
    if DOWNLOAD_DIVIDENDS:
        download_dividends(TICKERS, START, END)
