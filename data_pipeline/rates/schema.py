"""Normalized schema for provider-published rate observations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

SOFR_SERIES = (
    "SOFR",
    "SOFR30DAYAVG",
    "SOFR90DAYAVG",
    "SOFR180DAYAVG",
    "SOFRINDEX",
)

RATE_COLUMNS = [
    "observation_date",
    "series_id",
    "value",
    "value_raw",
    "realtime_start",
    "realtime_end",
    "title",
    "frequency",
    "units",
    "seasonal_adjustment",
    "provider",
    "series_notes",
    "retrieved_at_utc",
    "raw_record",
]


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
                "realtime_start": record.get("realtime_start"),
                "realtime_end": record.get("realtime_end"),
                "title": metadata.get("title"),
                "frequency": metadata.get("frequency"),
                "units": metadata.get("units"),
                "seasonal_adjustment": metadata.get("seasonal_adjustment"),
                "provider": "FRED",
                "series_notes": metadata.get("notes"),
                "retrieved_at_utc": retrieved_at,
                "raw_record": json.dumps(record, sort_keys=True),
            }
        )

    frame = pd.DataFrame(rows, columns=RATE_COLUMNS)
    for column in ("observation_date", "realtime_start", "realtime_end"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    # FRED uses "." for missing data. Preserve value_raw and drop no rows.
    return frame
