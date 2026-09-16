"""Loss-aware storage for exact provider queries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class QueryResult:
    """One provider query represented in raw and normalized forms."""

    frame: pd.DataFrame
    raw_payload: Any


def parquet_available() -> bool:
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def normalize_date(value: str | pd.Timestamp, *, name: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError
        return timestamp.strftime("%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date, got {value!r}") from exc


def normalize_date_range(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[str, str]:
    start = normalize_date(start_date, name="start_date")
    end = normalize_date(end_date, name="end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    return start, end


class QueryStore:
    """Persist both raw provider responses and normalized Parquet results.

    The cache key describes the exact provider query.  Raw JSON remains
    available for audit/re-normalization, while the Parquet file is the fast
    read path.  No interpolation, filling, or financial calculation occurs.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        domain: str,
        provider: str,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.domain = domain
        self.provider = provider
        self.enabled = bool(enabled) and parquet_available()
        if enabled and not self.enabled:
            warnings.warn(
                "Parquet persistence requested, but no Parquet engine is installed. "
                "Install pyarrow; continuing without a query cache.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _digest(query: Mapping[str, Any]) -> str:
        payload = json.dumps(query, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def paths(self, query: Mapping[str, Any]) -> tuple[Path, Path]:
        stem = self._digest(query)
        base = self.root / "raw" / self.domain / self.provider
        normalized = self.root / "normalized" / self.domain / self.provider
        return base / f"{stem}.json", normalized / f"{stem}.parquet"

    def read(self, query: Mapping[str, Any]) -> pd.DataFrame | None:
        if not self.enabled:
            return None
        _, frame_path = self.paths(query)
        return pd.read_parquet(frame_path) if frame_path.exists() else None

    def write(
        self,
        query: Mapping[str, Any],
        frame: pd.DataFrame,
        raw_payload: Any,
    ) -> None:
        if not self.enabled:
            return
        raw_path, frame_path = self.paths(query)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.parent.mkdir(parents=True, exist_ok=True)

        raw_temp = raw_path.with_suffix(".tmp.json")
        frame_temp = frame_path.with_suffix(".tmp.parquet")
        envelope = {"query": dict(query), "response": raw_payload}
        raw_temp.write_text(
            json.dumps(envelope, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        frame.to_parquet(frame_temp, index=False, compression="zstd")
        raw_temp.replace(raw_path)
        frame_temp.replace(frame_path)

    def get_or_create(
        self,
        query: Mapping[str, Any],
        fetch: Callable[[], QueryResult],
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return an exact cached query or fetch and atomically persist it."""
        if not refresh:
            cached = self.read(query)
            if cached is not None:
                return cached
        result = fetch()
        self.write(query, result.frame, result.raw_payload)
        return result.frame
