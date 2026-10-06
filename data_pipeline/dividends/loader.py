"""Raw dividends pipeline: retrieve, normalize, validate, and persist."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..common.inputs import normalize_identifiers
from ..common.storage import DEFAULT_DATA_DIR, QueryResult, QueryStore, normalize_date_range
from .schema import DIVIDEND_COLUMNS, normalize_dividend_records
from .sources import MassiveDividendsSource


def _today() -> dt.date:
    """The current date where US dividends are declared."""
    return pd.Timestamp.now(tz="America/New_York").date()


class DividendsSource(Protocol):
    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict: ...


class DividendsDataLoader:
    """Load declared stock/ETF dividend events; never infer a dividend curve.

    Events are cached one calendar year of ex-dividend dates per ticker, e.g.
    ``parquet/dividends/year=2025/ticker=SPY/dividends_massive.parquet``, so
    any date range is served from the same files.  A past year is fetched once;
    the current year is refetched the first time it is needed on each new day,
    so newly declared dividends appear without ``refresh=True``.
    """

    CACHE_SCHEMA = "dividends-v2"
    PROVIDER = "massive"

    def __init__(
        self,
        *,
        source: DividendsSource | None = None,
        api_key: str | None = None,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        cache_parquet: bool = True,
        request_timeout: float = 30.0,
    ) -> None:
        self.source = source or MassiveDividendsSource(
            api_key=api_key,
            request_timeout=request_timeout,
        )
        self.query_store = QueryStore(
            data_dir,
            domain="dividends",
            provider=self.PROVIDER,
            kind="dividends",
            enabled=cache_parquet,
        )
        self.store = self.query_store  # compatibility with version 0.1

    def load(
        self,
        tickers: str | Sequence[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Events with an ex-dividend date in the range, by ticker then date."""
        start, end = normalize_date_range(start_date, end_date)
        symbols = normalize_identifiers(tickers, label="dividend ticker")
        years = range(int(start[:4]), int(end[:4]) + 1)
        frames = [
            self._load_year(ticker, year, refresh=refresh)
            for ticker in symbols
            for year in years
        ]
        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            return pd.DataFrame(columns=DIVIDEND_COLUMNS)
        frame = pd.concat(nonempty, ignore_index=True)
        in_range = frame["ex_dividend_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        return (
            frame[in_range]
            .sort_values(["underlying", "ex_dividend_date", "dividend_id"])
            .reset_index(drop=True)
        )

    def _load_year(self, ticker: str, year: int, *, refresh: bool) -> pd.DataFrame:
        query = {
            "schema": self.CACHE_SCHEMA,
            "provider": self.PROVIDER,
            "ticker": ticker,
            "year": year,
            "date_field": "ex_dividend_date",
        }
        today = _today()
        if year >= today.year:
            # Still being declared: a copy fetched on an earlier day no longer
            # matches this query, so it is refetched and overwritten.
            query["fetched_on"] = today.isoformat()

        def fetch() -> QueryResult:
            payload = self.source.fetch(ticker, f"{year}-01-01", f"{year}-12-31")
            frame = normalize_dividend_records(
                ticker,
                payload["records"],
                pd.Timestamp.now(tz="UTC"),
            )
            return QueryResult(frame=frame, raw_payload=payload["pages"])

        return self.query_store.get_or_create(
            query, fetch, partitions={"year": year, "ticker": ticker}, refresh=refresh
        )
