"""Raw dividends pipeline: retrieve, normalize, validate, and persist."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..common.inputs import normalize_identifiers
from ..common.storage import QueryResult, QueryStore, normalize_date_range
from .schema import DIVIDEND_COLUMNS, normalize_dividend_records
from .sources import MassiveDividendsSource


class DividendsSource(Protocol):
    def fetch(self, ticker: str, start_date: str, end_date: str) -> dict: ...


class DividendsDataLoader:
    """Load declared stock/ETF dividend events; never infer a dividend curve."""

    CACHE_SCHEMA = "dividends-v1"

    def __init__(
        self,
        *,
        source: DividendsSource | None = None,
        api_key: str | None = None,
        data_dir: str | Path = "./market_data",
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
            provider="massive",
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
        start, end = normalize_date_range(start_date, end_date)
        symbols = normalize_identifiers(tickers, label="dividend ticker")
        frames = [
            self._load_one(ticker, start, end, refresh=refresh) for ticker in symbols
        ]
        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            return pd.DataFrame(columns=DIVIDEND_COLUMNS)
        return (
            pd.concat(nonempty, ignore_index=True)
            .sort_values(["underlying", "ex_dividend_date", "dividend_id"])
            .reset_index(drop=True)
        )

    def _load_one(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        *,
        refresh: bool,
    ) -> pd.DataFrame:
        query = {
            "schema": self.CACHE_SCHEMA,
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "date_field": "ex_dividend_date",
        }

        def fetch() -> QueryResult:
            payload = self.source.fetch(ticker, start_date, end_date)
            frame = normalize_dividend_records(
                ticker,
                payload["records"],
                pd.Timestamp.now(tz="UTC"),
            )
            return QueryResult(frame=frame, raw_payload=payload["pages"])

        return self.query_store.get_or_create(query, fetch, refresh=refresh)
