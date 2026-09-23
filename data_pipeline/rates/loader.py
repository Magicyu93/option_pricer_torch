"""Raw rates pipeline: retrieve, normalize, validate, and persist."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from ..common.inputs import normalize_identifiers
from ..common.storage import QueryResult, QueryStore, normalize_date, normalize_date_range
from .schema import (
    TREASURY_CMT_SERIES,
    TREASURY_TENORS,
    empty_rate_frame,
    normalize_massive_treasury_yields,
    normalize_rate_observations,
)
from .sources import FredRatesSource, MassiveTreasuryYieldsSource

Provider = Literal["massive", "fred"]


def _today() -> dt.date:
    """The current date where US rates are published."""
    return pd.Timestamp.now(tz="America/New_York").date()


class _RatesProvider(Protocol):
    """Adapter between a thin provider client and the year-partitioned cache.

    A *dataset* is the unit a provider serves in one request: every Treasury
    tenor at once for Massive, one series for FRED.
    """

    name: str

    def datasets(self, series_ids: Sequence[str]) -> dict[str, list[str]]:
        """The datasets to fetch, each mapped to the requested series it holds."""

    def fetch(self, dataset: str, start_date: str, end_date: str) -> QueryResult: ...


class _MassiveTreasury:
    name = "massive"
    DATASET = "treasury_yields"

    def __init__(self, source: MassiveTreasuryYieldsSource) -> None:
        self.source = source

    def datasets(self, series_ids: Sequence[str]) -> dict[str, list[str]]:
        unknown = [item for item in series_ids if item not in TREASURY_TENORS]
        if unknown:
            raise ValueError(
                f"Massive serves only Treasury yields, not {unknown}; expected any "
                f"of {list(TREASURY_CMT_SERIES)}"
            )
        return {self.DATASET: list(series_ids)}

    def fetch(self, dataset: str, start_date: str, end_date: str) -> QueryResult:
        payload = self.source.fetch(start_date, end_date)
        frame = normalize_massive_treasury_yields(
            payload["records"], pd.Timestamp.now(tz="UTC")
        )
        return QueryResult(frame=frame, raw_payload=payload["pages"])


class _Fred:
    name = "fred"

    def __init__(self, source: FredRatesSource) -> None:
        self.source = source
        self._metadata: dict[str, dict[str, Any]] = {}

    def datasets(self, series_ids: Sequence[str]) -> dict[str, list[str]]:
        return {item: [item] for item in series_ids}

    def fetch(self, dataset: str, start_date: str, end_date: str) -> QueryResult:
        # Series metadata does not change with the date range, so one request
        # per series serves every yearly partition.
        if dataset not in self._metadata:
            self._metadata[dataset] = self.source.fetch_series_metadata(dataset)
        metadata = self._metadata[dataset]
        observations = self.source.fetch_observations(dataset, start_date, end_date)
        frame = normalize_rate_observations(
            dataset,
            metadata["record"],
            observations["records"],
            pd.Timestamp.now(tz="UTC"),
        )
        raw = {"metadata": metadata["response"], "observation_pages": observations["pages"]}
        return QueryResult(frame=frame, raw_payload=raw)


@dataclass(frozen=True)
class ParYieldSnapshot:
    """The Treasury par yields in force on a date, shortest tenor first.

    ``observation_date`` is the publication actually used, which is earlier than
    the requested date on weekends, bond-market holidays, or with a lag.
    """

    as_of: dt.date
    observation_date: dt.date
    series_ids: tuple[str, ...]
    tenors: np.ndarray
    yields_pct: np.ndarray


class RatesDataLoader:
    """Load published rate observations; never construct a rate curve.

    ``provider="massive"`` (the default) serves Treasury constant-maturity
    yields; ``provider="fred"`` serves any FRED series, including the same
    Treasury series and SOFR.  Rows are identical in shape either way, and
    Treasury series carry their FRED IDs (``DGS3MO``, ``DGS10``, ...) from both
    providers.  Values stay as published: no tenor mapping, interpolation,
    compounding conversion, or bootstrapping happens here.

    Data is cached one calendar year per file, e.g.
    ``parquet/rates/massive/treasury_yields/2025.parquet``, so any date range is
    served from the same files.  A past year is fetched once.  The current
    year is refetched the first time it is needed on each new day, so it picks
    up new publications without ``refresh=True``.
    """

    CACHE_SCHEMA = "rates-v3"

    def __init__(
        self,
        provider: Provider = "massive",
        *,
        source: MassiveTreasuryYieldsSource | FredRatesSource | None = None,
        api_key: str | None = None,
        data_dir: str | Path = "./market_data",
        cache_parquet: bool = True,
        request_timeout: float = 30.0,
    ) -> None:
        self.provider: _RatesProvider
        if provider == "massive":
            self.provider = _MassiveTreasury(
                source
                or MassiveTreasuryYieldsSource(api_key=api_key, request_timeout=request_timeout)
            )
        elif provider == "fred":
            self.provider = _Fred(
                source or FredRatesSource(api_key=api_key, request_timeout=request_timeout)
            )
        else:
            raise ValueError(f"provider must be 'massive' or 'fred', got {provider!r}")
        self.query_store = QueryStore(
            data_dir,
            domain="rates",
            provider=self.provider.name,
            enabled=cache_parquet,
        )

    def load(
        self,
        series_ids: str | Sequence[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Observations for ``series_ids`` over the range, by date then maturity."""
        start, end = normalize_date_range(start_date, end_date)
        series = normalize_identifiers(series_ids, label="rate series ID")
        years = range(int(start[:4]), int(end[:4]) + 1)

        frames = [
            self._load_year(dataset, year, refresh=refresh)
            for dataset in self.provider.datasets(series)
            for year in years
        ]
        frame = pd.concat([f for f in frames if not f.empty] or [empty_rate_frame()])
        in_range = frame["observation_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        frame = frame[in_range & frame["series_id"].isin(series)]
        return self._sorted(frame)

    def yields_as_of(
        self,
        as_of: str | pd.Timestamp,
        *,
        series_ids: Sequence[str] = TREASURY_CMT_SERIES,
        lag_days: int = 0,
        lookback_days: int = 14,
        refresh: bool = False,
    ) -> ParYieldSnapshot:
        """The latest Treasury par yields published on or before ``as_of - lag_days``.

        Weekends and bond-market holidays fall back to the previous publication.
        Pass ``lag_days=1`` when pricing intraday: a day's yields are end-of-day
        values published after the close.  Tenors not published on the chosen
        date are omitted.  Raises ``LookupError`` if nothing was published within
        ``lookback_days`` of the cutoff.
        """
        as_of_date = pd.Timestamp(normalize_date(as_of, name="as_of")).date()
        if lag_days < 0 or lookback_days < 0:
            raise ValueError("lag_days and lookback_days must be non-negative")
        series = normalize_identifiers(series_ids, label="Treasury series ID")
        unknown = [item for item in series if item not in TREASURY_TENORS]
        if unknown:
            raise ValueError(f"yields_as_of needs Treasury series, not {unknown}")

        cutoff = as_of_date - dt.timedelta(days=lag_days)
        window = self.load(
            series,
            cutoff - dt.timedelta(days=lookback_days),
            cutoff,
            refresh=refresh,
        ).dropna(subset=["value"])
        if window.empty:
            raise LookupError(
                f"no Treasury yields published in the {lookback_days} days to {cutoff}"
            )
        latest = window[window["observation_date"] == window["observation_date"].max()]
        return ParYieldSnapshot(
            as_of=as_of_date,
            observation_date=latest["observation_date"].iloc[0].date(),
            series_ids=tuple(latest["series_id"]),
            tenors=np.array([TREASURY_TENORS[s].years for s in latest["series_id"]]),
            yields_pct=latest["value"].to_numpy(dtype=float),
        )

    def _load_year(self, dataset: str, year: int, *, refresh: bool) -> pd.DataFrame:
        query: dict[str, Any] = {
            "schema": self.CACHE_SCHEMA,
            "provider": self.provider.name,
            "dataset": dataset,
            "year": year,
        }
        today = _today()
        if year >= today.year:
            # Still being published: a copy fetched on an earlier day no longer
            # matches this query, so it is refetched and overwritten.
            query["fetched_on"] = today.isoformat()

        def fetch() -> QueryResult:
            return self.provider.fetch(dataset, f"{year}-01-01", f"{year}-12-31")

        return self.query_store.get_or_create(
            query, fetch, key=(dataset, str(year)), refresh=refresh
        )

    @staticmethod
    def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
        """By date, then Treasury maturity, then series ID for everything else."""
        maturity = frame["series_id"].map(
            lambda s: TREASURY_TENORS[s].years if s in TREASURY_TENORS else np.inf
        )
        return (
            frame.assign(_maturity=maturity)
            .sort_values(["observation_date", "_maturity", "series_id"])
            .drop(columns="_maturity")
            .reset_index(drop=True)
        )
