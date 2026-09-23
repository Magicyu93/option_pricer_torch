"""Loss-aware storage for exact provider queries."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - reported by QueryStore
    pa = pq = None


@dataclass(frozen=True)
class QueryResult:
    """One provider query represented in raw and normalized forms."""

    frame: pd.DataFrame
    raw_payload: Any


def parquet_available() -> bool:
    return pq is not None


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

    Files are named for what they hold, under the same top-level layout as the
    equity flat files::

        <root>/raw/<domain>/<provider>/<key...>.json
        <root>/parquet/<domain>/<provider>/<key...>.parquet

    e.g. ``parquet/dividends/massive/SPY/2025-01-01_2025-12-31.parquet``.  The
    exact provider query is written into the Parquet metadata and the raw
    envelope; a file whose query no longer matches (a schema bump, a changed
    request parameter) is treated as a miss and overwritten.  Raw JSON remains
    available for audit/re-normalization, while the Parquet file is the fast
    read path.  No interpolation, filling, or financial calculation occurs.
    """

    METADATA_KEY = b"query"

    def __init__(
        self,
        root: str | Path,
        *,
        domain: str,
        provider: str,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.domain = _path_part(domain)
        self.provider = _path_part(provider)
        self.enabled = bool(enabled) and parquet_available()
        if enabled and not self.enabled:
            warnings.warn(
                "Parquet persistence requested, but pyarrow is not installed. "
                "Install pyarrow; continuing without a query cache.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _canonical(query: Mapping[str, Any]) -> bytes:
        return json.dumps(query, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def paths(self, key: Sequence[str]) -> tuple[Path, Path]:
        parts = [_path_part(part) for part in key]
        if not parts:
            raise ValueError("cache key needs at least one path component")
        *dirs, stem = parts
        relative = Path(self.domain, self.provider, *dirs)
        return (
            self.root / "raw" / relative / f"{stem}.json",
            self.root / "parquet" / relative / f"{stem}.parquet",
        )

    def read(self, query: Mapping[str, Any], key: Sequence[str]) -> pd.DataFrame | None:
        if not self.enabled:
            return None
        _, frame_path = self.paths(key)
        if not frame_path.exists():
            return None
        table = pq.read_table(frame_path)
        stored = (table.schema.metadata or {}).get(self.METADATA_KEY)
        if stored != self._canonical(query):
            return None
        return table.to_pandas()

    def write(
        self,
        query: Mapping[str, Any],
        key: Sequence[str],
        frame: pd.DataFrame,
        raw_payload: Any,
    ) -> None:
        if not self.enabled:
            return
        raw_path, frame_path = self.paths(key)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.parent.mkdir(parents=True, exist_ok=True)

        raw_temp = raw_path.with_suffix(".tmp.json")
        frame_temp = frame_path.with_suffix(".tmp.parquet")
        envelope = {"query": dict(query), "response": raw_payload}
        raw_temp.write_text(
            json.dumps(envelope, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        table = pa.Table.from_pandas(frame, preserve_index=False)
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), self.METADATA_KEY: self._canonical(query)}
        )
        pq.write_table(table, frame_temp, compression="zstd")
        raw_temp.replace(raw_path)
        frame_temp.replace(frame_path)

    def get_or_create(
        self,
        query: Mapping[str, Any],
        fetch: Callable[[], QueryResult],
        *,
        key: Sequence[str],
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return an exact cached query or fetch and atomically persist it.

        ``key`` names the files: every component but the last is a directory.
        """
        if not refresh:
            cached = self.read(query, key)
            if cached is not None:
                return cached
        result = fetch()
        self.write(query, key, result.frame, result.raw_payload)
        return result.frame


def _path_part(value: str) -> str:
    """One filesystem-safe path component, readable where the input already is."""
    part = re.sub(r"[^A-Za-z0-9._=-]", "_", str(value).strip())
    if part in {"", ".", ".."}:
        raise ValueError(f"invalid cache path component {value!r}")
    return part
