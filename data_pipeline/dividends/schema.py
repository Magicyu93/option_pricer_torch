"""Normalized schema for provider-published cash-dividend events."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

DIVIDEND_COLUMNS = [
    "underlying",
    "provider_ticker",
    "dividend_id",
    "declaration_date",
    "ex_dividend_date",
    "record_date",
    "pay_date",
    "cash_amount",
    "split_adjusted_cash_amount",
    "currency",
    "frequency",
    "distribution_type",
    "historical_adjustment_factor",
    "provider",
    "retrieved_at_utc",
    "raw_record",
]


def normalize_dividend_records(
    requested_ticker: str,
    records: Sequence[Mapping[str, Any]],
    retrieved_at: pd.Timestamp,
) -> pd.DataFrame:
    """Normalize Massive records without deriving yields or future payments."""
    rows = []
    for record in records:
        provider_ticker = str(record.get("ticker") or requested_ticker).upper()
        rows.append(
            {
                "underlying": provider_ticker,
                "provider_ticker": provider_ticker,
                "dividend_id": record.get("id"),
                "declaration_date": record.get("declaration_date"),
                "ex_dividend_date": record.get("ex_dividend_date"),
                "record_date": record.get("record_date"),
                "pay_date": record.get("pay_date"),
                "cash_amount": pd.to_numeric(
                    record.get("cash_amount"), errors="coerce"
                ),
                "split_adjusted_cash_amount": pd.to_numeric(
                    record.get("split_adjusted_cash_amount"), errors="coerce"
                ),
                "currency": record.get("currency"),
                "frequency": pd.to_numeric(record.get("frequency"), errors="coerce"),
                "distribution_type": record.get("distribution_type"),
                "historical_adjustment_factor": pd.to_numeric(
                    record.get("historical_adjustment_factor"), errors="coerce"
                ),
                "provider": "Massive",
                "retrieved_at_utc": retrieved_at,
                "raw_record": json.dumps(record, sort_keys=True),
            }
        )

    frame = pd.DataFrame(rows, columns=DIVIDEND_COLUMNS)
    for column in (
        "declaration_date",
        "ex_dividend_date",
        "record_date",
        "pay_date",
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame
