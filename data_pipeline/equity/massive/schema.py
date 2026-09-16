"""Normalization of Massive equity Flat File rows."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

UNDERLYING_COLUMNS = (
    "underlying_ticker",
    "underlying_asset",
    "underlying_open",
    "underlying_high",
    "underlying_low",
    "underlying_close",
    "underlying_volume",
    "underlying_transactions",
    "underlying_window_start_ns",
)

CONTRACT_METADATA_COLUMNS = (
    "exercise_style",
    "contract_underlying_asset",
    "contract_underlying_class",
    "contract_underlying_provider_ticker",
    "settlement_method",
    "standard_terms",
    "shares_per_contract",
    "primary_exchange",
    "cfi",
)


def unified_underlying_columns() -> list[str]:
    return list(UNDERLYING_COLUMNS)


def contract_metadata_columns() -> list[str]:
    return list(CONTRACT_METADATA_COLUMNS)


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA


def _timestamp(values: pd.Series, timezone: str) -> pd.Series:
    return pd.to_datetime(values, unit="ns", utc=True, errors="coerce").dt.tz_convert(
        timezone
    )


def process_options(
    frame: pd.DataFrame,
    session_date: str,
    *,
    timezone: str,
) -> pd.DataFrame:
    frame = frame.copy()
    _ensure_columns(frame, ["volume", "transactions"])

    symbols = frame["ticker"].astype("string")
    frame["underlying"] = symbols.str.slice(2, -15)
    expiration_code = symbols.str.slice(-15, -9)
    type_code = symbols.str.slice(-9, -8)
    strike_code = symbols.str.slice(-8)

    frame["expiration"] = pd.to_datetime(
        expiration_code,
        format="%y%m%d",
        errors="coerce",
    )
    frame["option_type"] = type_code.map({"C": "call", "P": "put"})
    frame["strike"] = pd.to_numeric(strike_code, errors="coerce") / 1000.0
    frame["timestamp"] = _timestamp(frame["window_start"], timezone)
    frame["session_date"] = pd.Timestamp(session_date).date()
    frame = frame.dropna(
        subset=["underlying", "expiration", "option_type", "strike", "timestamp"]
    )
    frame = frame.rename(
        columns={
            "ticker": "option_ticker",
            "open": "option_open",
            "high": "option_high",
            "low": "option_low",
            "close": "option_close",
            "volume": "option_volume",
            "transactions": "option_transactions",
            "window_start": "option_window_start_ns",
        }
    )
    columns = [
        "session_date",
        "timestamp",
        "underlying",
        "option_ticker",
        "expiration",
        "option_type",
        "strike",
        "option_open",
        "option_high",
        "option_low",
        "option_close",
        "option_volume",
        "option_transactions",
        "option_window_start_ns",
    ]
    return (
        frame[columns]
        .sort_values(["underlying", "timestamp", "expiration", "option_type", "strike"])
        .reset_index(drop=True)
    )


def process_stocks(
    frame: pd.DataFrame,
    session_date: str,
    *,
    timezone: str,
) -> pd.DataFrame:
    frame = frame.copy()
    _ensure_columns(frame, ["volume", "transactions"])
    frame["timestamp"] = _timestamp(frame["window_start"], timezone)
    frame["session_date"] = pd.Timestamp(session_date).date()
    frame["underlying_ticker"] = frame["ticker"].astype("string").str.upper()
    frame["underlying"] = frame["underlying_ticker"]
    frame["underlying_asset"] = "stocks"
    frame = frame.rename(
        columns={
            "open": "underlying_open",
            "high": "underlying_high",
            "low": "underlying_low",
            "close": "underlying_close",
            "volume": "underlying_volume",
            "transactions": "underlying_transactions",
            "window_start": "underlying_window_start_ns",
        }
    )
    columns = ["session_date", "timestamp", "underlying", *UNDERLYING_COLUMNS]
    return (
        frame[columns]
        .dropna(subset=["timestamp", "underlying"])
        .sort_values(["underlying", "timestamp"])
        .reset_index(drop=True)
    )


def process_indices(
    frame: pd.DataFrame,
    session_date: str,
    *,
    timezone: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["underlying_ticker"] = frame["ticker"].astype("string").str.upper()
    frame["underlying"] = frame["underlying_ticker"].str.removeprefix("I:")
    frame["timestamp"] = _timestamp(frame["window_start"], timezone)
    frame["session_date"] = pd.Timestamp(session_date).date()
    frame["underlying_asset"] = "indices"
    frame = frame.rename(
        columns={
            "open": "underlying_open",
            "high": "underlying_high",
            "low": "underlying_low",
            "close": "underlying_close",
            "window_start": "underlying_window_start_ns",
        }
    )
    # Index aggregates are index values rather than exchange trades.
    frame["underlying_volume"] = pd.NA
    frame["underlying_transactions"] = pd.NA
    columns = ["session_date", "timestamp", "underlying", *UNDERLYING_COLUMNS]
    return (
        frame[columns]
        .dropna(subset=["timestamp", "underlying"])
        .sort_values(["underlying", "timestamp"])
        .reset_index(drop=True)
    )
