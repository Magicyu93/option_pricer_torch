"""Normalized schema for provider-published rate observations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TreasuryTenor:
    """One Treasury constant-maturity series and how each provider names it.

    ``series_id`` is FRED's ID, used as the canonical name whichever provider
    supplied the data.  ``years`` is the nominal maturity -- metadata about the
    instrument, not a day count or a curve.
    """

    series_id: str
    label: str
    years: float
    massive_field: str


#: Every CMT tenor, by series ID, shortest first.  Massive documents all eleven
#: but has only ever published 1M, 3M, 1Y, 2Y, 5Y, 10Y and 30Y.
TREASURY_TENORS: dict[str, TreasuryTenor] = {
    tenor.series_id: tenor
    for tenor in (
        TreasuryTenor("DGS1MO", "1-Month", 1 / 12, "yield_1_month"),
        TreasuryTenor("DGS3MO", "3-Month", 0.25, "yield_3_month"),
        TreasuryTenor("DGS6MO", "6-Month", 0.5, "yield_6_month"),
        TreasuryTenor("DGS1", "1-Year", 1.0, "yield_1_year"),
        TreasuryTenor("DGS2", "2-Year", 2.0, "yield_2_year"),
        TreasuryTenor("DGS3", "3-Year", 3.0, "yield_3_year"),
        TreasuryTenor("DGS5", "5-Year", 5.0, "yield_5_year"),
        TreasuryTenor("DGS7", "7-Year", 7.0, "yield_7_year"),
        TreasuryTenor("DGS10", "10-Year", 10.0, "yield_10_year"),
        TreasuryTenor("DGS20", "20-Year", 20.0, "yield_20_year"),
        TreasuryTenor("DGS30", "30-Year", 30.0, "yield_30_year"),
    )
}

TREASURY_CMT_SERIES = tuple(TREASURY_TENORS)

#: One row per published observation.  Only what describes the observation
#: itself: series descriptions live in the raw JSON (FRED's series metadata) or
#: in :data:`TREASURY_TENORS`, and ``raw_record`` keeps the provider's record
#: verbatim, including FRED's ``realtime_start``/``realtime_end`` vintage.
RATE_COLUMNS = [
    "observation_date",
    "series_id",
    "value",
    "value_raw",
    "units",
    "provider",
    "retrieved_at_utc",
    "raw_record",
]


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["value"] = frame["value"].astype("float64")
    return frame


def empty_rate_frame() -> pd.DataFrame:
    """A zero-row frame with the same columns and dtypes as a loaded one."""
    return _typed(pd.DataFrame(columns=RATE_COLUMNS))


def normalize_rate_observations(
    series_id: str,
    metadata: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    retrieved_at: pd.Timestamp,
) -> pd.DataFrame:
    """Normalize FRED records without changing their published units."""
    rows = []
    for record in observations:
        value_raw = record.get("value")
        rows.append(
            {
                "observation_date": record.get("date"),
                "series_id": series_id,
                "value": pd.to_numeric(value_raw, errors="coerce"),
                "value_raw": value_raw,
                "units": metadata.get("units"),
                "provider": "FRED",
                "retrieved_at_utc": retrieved_at,
                "raw_record": json.dumps(record, sort_keys=True),
            }
        )

    # FRED uses "." for missing data. Preserve value_raw and drop no rows.
    return _typed(pd.DataFrame(rows, columns=RATE_COLUMNS))


def normalize_massive_treasury_yields(
    records: Sequence[Mapping[str, Any]],
    retrieved_at: pd.Timestamp,
) -> pd.DataFrame:
    """Unpivot Massive's one-record-per-date yields into one row per tenor.

    Values stay in published units: percent, investment-basis (par) yields.
    A tenor absent from a record was not published that day and gets no row.
    """
    rows = []
    for record in records:
        for tenor in TREASURY_TENORS.values():
            field = tenor.massive_field
            if field not in record:
                continue
            value_raw = record[field]
            rows.append(
                {
                    "observation_date": record.get("date"),
                    "series_id": tenor.series_id,
                    "value": pd.to_numeric(value_raw, errors="coerce"),
                    "value_raw": None if value_raw is None else str(value_raw),
                    "units": "Percent",
                    "provider": "Massive",
                    "retrieved_at_utc": retrieved_at,
                    "raw_record": json.dumps(
                        {"date": record.get("date"), field: value_raw}, sort_keys=True
                    ),
                }
            )

    return _typed(pd.DataFrame(rows, columns=RATE_COLUMNS))
