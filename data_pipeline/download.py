"""Download market data into the cache: the functions behind the download script and the app.

Each function fills the cache under ``data_dir`` through the loaders, so a rerun
fetches only what is missing, and returns a summary rather than printing, so a
script can print it and an app can show it. ``on_progress`` receives one line
per unit of work as it finishes.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .common.storage import DEFAULT_DATA_DIR
from .dividends import DividendsDataLoader
from .equity import MassiveHistoricalDataLoader
from .rates import (
    TREASURY_CMT_SERIES,
    FredRatesSource,
    MassiveTreasuryYieldsSource,
    RatesDataLoader,
)

Progress = Callable[[str], None] | None


@dataclass(frozen=True)
class DownloadResult:
    """One unit of work: a day of equity data, a rates range, a ticker's dividends."""

    dataset: str
    label: str
    rows: int = 0
    error: str | None = None
    note: str | None = None  # why nothing was fetched, when that is not an error

    @property
    def ok(self) -> bool:
        return self.error is None


def _report(on_progress: Progress, result: DownloadResult) -> DownloadResult:
    if on_progress is not None:
        status = (
            f"FAILED -- {result.error}" if result.error
            else result.note or f"{result.rows:,} rows"
        )
        on_progress(f"{result.dataset} {result.label}: {status}")
    return result


def credentials_status() -> dict[str, dict[str, object]]:
    """Which providers have credentials, as the loaders themselves resolve them.

    Asks each loader what it found rather than re-reading the environment, so
    this cannot disagree with what a download would actually use.
    """
    equity = MassiveHistoricalDataLoader(verbose=False)
    massive_rest, fred = MassiveTreasuryYieldsSource(), FredRatesSource()
    return {
        "Massive flat files (options, stocks, indices)": {
            "ok": bool(equity.access_key and equity.secret_key),
            "variables": "MASSIVE_S3_ACCESS_KEY + MASSIVE_S3_SECRET_KEY, or MASSIVE_API_ID + MASSIVE_API_KEY",
        },
        "Massive REST (Treasury yields, dividends, reference data)": {
            "ok": bool(massive_rest.api_key),
            "variables": massive_rest.api_key_env,
        },
        "FRED (Treasury yields, other series)": {
            "ok": bool(fred.api_key),
            "variables": fred.api_key_env,
        },
    }


def download_equity(
    tickers: Sequence[str],
    start,
    end,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    on_progress: Progress = None,
) -> list[DownloadResult]:
    """Options joined to their underlying, one trading day at a time.

    A failed day does not stop the others; its error is in its result.
    """
    loader = MassiveHistoricalDataLoader(data_dir=data_dir, verbose=False)
    results = []
    for day in pd.bdate_range(start, end).strftime("%Y-%m-%d"):
        try:
            result = DownloadResult("equity", day, rows=len(loader.load_market([day], list(tickers))))
        except Exception as exc:  # recorded, and the next day still runs
            result = DownloadResult("equity", day, error=str(exc))
        results.append(_report(on_progress, result))
    return results


def download_rates(
    start,
    end,
    *,
    provider: str = "massive",
    lookback_days: int = 14,
    refresh: bool = False,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    on_progress: Progress = None,
) -> DownloadResult:
    """Treasury yields from ``lookback_days`` before ``start``, so the first day has a print."""
    first = pd.Timestamp(start).date() - dt.timedelta(days=lookback_days)
    last = pd.Timestamp(end).date()
    label = f"({provider}) {first} to {last}"
    try:
        frame = RatesDataLoader(provider, data_dir=data_dir).load(
            TREASURY_CMT_SERIES, first, last, refresh=refresh
        )
        result = DownloadResult("rates", label, rows=len(frame))
    except Exception as exc:
        result = DownloadResult("rates", label, error=str(exc))
    return _report(on_progress, result)


def download_dividends(
    tickers: Sequence[str],
    start,
    end,
    *,
    horizon_days: int = 365,
    refresh: bool = False,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    on_progress: Progress = None,
) -> list[DownloadResult]:
    """Declared dividends from ``start`` to ``horizon_days`` after ``end``; indices skipped."""
    first = pd.Timestamp(start).date()
    last = pd.Timestamp(end).date() + dt.timedelta(days=horizon_days)
    equity = MassiveHistoricalDataLoader(data_dir=data_dir, verbose=False)
    loader = DividendsDataLoader(data_dir=data_dir)
    results = []
    for ticker in tickers:
        label = f"{ticker} {first} to {last}"
        try:
            if equity.infer_underlying_asset(ticker, date=first) == "indices":
                result = DownloadResult("dividends", ticker, note="index, no dividends: skipped")
            else:
                frame = loader.load(ticker, first, last, refresh=refresh)
                result = DownloadResult("dividends", label, rows=len(frame))
        except Exception as exc:
            result = DownloadResult("dividends", label, error=str(exc))
        results.append(_report(on_progress, result))
    return results
