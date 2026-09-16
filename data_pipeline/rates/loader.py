"""Raw rates pipeline: retrieve, normalize, validate, and persist."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..common.inputs import normalize_identifiers
from ..common.storage import QueryResult, QueryStore, normalize_date_range
from .schema import RATE_COLUMNS, SOFR_SERIES, normalize_rate_observations
from .sources import FredRatesSource


class RatesSource(Protocol):
    def fetch_series_metadata(self, series_id: str) -> dict: ...

    def fetch_observations(
        self, series_id: str, start_date: str, end_date: str
    ) -> dict: ...


class RatesDataLoader:
    """Load published rate observations; never construct a rate curve.

    FRED's SOFR averages are backward-looking compounded averages, not OIS
    par rates.  This class deliberately stores them under their provider series
    IDs and performs no tenor mapping, interpolation, or bootstrapping.
    """

    CACHE_SCHEMA = "rates"

    def __init__(
        self,
        *,
        source: RatesSource | None = None,
        api_key: str | None = None,
        data_dir: str | Path = "./market_data",
        cache_parquet: bool = True,
        request_timeout: float = 30.0,
    ) -> None:
        self.source = source or FredRatesSource(
            api_key=api_key,
            request_timeout=request_timeout,
        )
        self.query_store = QueryStore(
            data_dir,
            domain="rates",
            provider="fred",
            enabled=cache_parquet,
        )
        self.store = self.query_store  # compatibility with version 0.1

    def load(
        self,
        series_ids: str | Sequence[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        start, end = normalize_date_range(start_date, end_date)
        series = normalize_identifiers(series_ids, label="FRED series ID")
        frames = [self._load_one(item, start, end, refresh=refresh) for item in series]
        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            return pd.DataFrame(columns=RATE_COLUMNS)
        return (
            pd.concat(nonempty, ignore_index=True)
            .sort_values(["observation_date", "series_id"])
            .reset_index(drop=True)
        )

    def load_sofr(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        *,
        series_ids: Sequence[str] = SOFR_SERIES,
        refresh: bool = False,
    ) -> pd.DataFrame:
        return self.load(series_ids, start_date, end_date, refresh=refresh)

    def _load_one(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        *,
        refresh: bool,
    ) -> pd.DataFrame:
        query = {
            "schema": self.CACHE_SCHEMA,
            "series_id": series_id,
            "start_date": start_date,
            "end_date": end_date,
            "units": "lin",
            "output_type": 1,
        }

        def fetch() -> QueryResult:
            metadata_payload = self.source.fetch_series_metadata(series_id)
            observations_payload = self.source.fetch_observations(
                series_id, start_date, end_date
            )
            frame = normalize_rate_observations(
                series_id,
                metadata_payload["record"],
                observations_payload["observations"],
                pd.Timestamp.now(tz="UTC"),
            )
            raw = {
                "metadata": metadata_payload["response"],
                "observation_pages": observations_payload["pages"],
            }
            return QueryResult(frame=frame, raw_payload=raw)

        return self.query_store.get_or_create(query, fetch, refresh=refresh)
