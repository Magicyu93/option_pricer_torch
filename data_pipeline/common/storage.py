"""Loss-aware storage for exact provider queries."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - reported by QueryStore
    pa = pq = None


#: Where every loader caches by default: ``<repo>/market_data``, wherever the
#: process runs from.  A relative default would follow the working directory,
#: so a run from ``scripts/`` or an IDE would start a second, empty cache.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "market_data"


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


def partitioned_path(
    base: str | Path,
    domain: str,
    partitions: Mapping[str, Any],
    *,
    kind: str,
    provider: str,
    suffix: str,
) -> Path:
    """The one file-naming rule shared by every market-data cache::

        <base>/<domain>/<key>=<value>/.../<kind>_<provider><suffix>

    where ``base`` is ``<root>/raw`` or ``<root>/parquet``,
    e.g. ``parquet/options/date=2026-09-08/ticker=SPY/minute_massive.parquet``
    or ``parquet/rates/year=2025/dataset=DGS10/daily_fred.parquet``.  Partitions
    are Hive-style, time first, so a whole domain reads as one dataset with the
    partitions as columns; the provider is in the file name, so two sources for
    the same partition sit side by side instead of overwriting each other.
    """
    dirs = [
        f"{_path_part(name)}={_path_part(value)}" for name, value in partitions.items()
    ]
    stem = f"{_path_part(kind)}_{_path_part(provider)}"
    return Path(base, _path_part(domain), *dirs, stem + suffix)


def parse_partitioned_path(base: str | Path, path: str | Path) -> dict[str, Any] | None:
    """The inverse of :func:`partitioned_path`: what a cached file's path says.

    Returns ``domain``, the ``partitions`` (in order), ``kind``, ``provider``
    and ``suffix`` for a path under ``base`` laid out by
    :func:`partitioned_path`, or ``None`` for anything else -- temporaries,
    markers, stray files.
    """
    parts = Path(path).relative_to(base).parts
    if len(parts) < 2 or any("=" not in p for p in parts[1:-1]):
        return None
    name = parts[-1]
    stem, dot, suffix = name.partition(".")
    kind, _, provider = stem.rpartition("_")
    if not kind or not dot or ".tmp" in name or suffix in {"part", "empty"} or name.endswith(".part"):
        return None
    return {
        "domain": parts[0],
        "partitions": dict(p.split("=", 1) for p in parts[1:-1]),
        "kind": kind,
        "provider": provider,
        "suffix": "." + suffix,
    }


class QueryStore:
    """Persist both raw provider responses and normalized Parquet results.

    Files follow :func:`partitioned_path`::

        <root>/raw/<domain>/<key>=<value>/.../<kind>_<provider>.json
        <root>/parquet/<domain>/<key>=<value>/.../<kind>_<provider>.parquet

    e.g. ``parquet/dividends/year=2025/ticker=SPY/dividends_massive.parquet``.
    The exact provider query is written into the Parquet metadata and the raw
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
        kind: str,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.domain = _path_part(domain)
        self.provider = _path_part(provider)
        self.kind = _path_part(kind)
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

    def paths(self, partitions: Mapping[str, Any]) -> tuple[Path, Path]:
        if not partitions:
            raise ValueError("cache key needs at least one partition")
        return tuple(
            partitioned_path(
                self.root / layer, self.domain, partitions,
                kind=self.kind, provider=self.provider, suffix=suffix,
            )
            for layer, suffix in (("raw", ".json"), ("parquet", ".parquet"))
        )

    def read(
        self, query: Mapping[str, Any], partitions: Mapping[str, Any]
    ) -> pd.DataFrame | None:
        if not self.enabled:
            return None
        _, frame_path = self.paths(partitions)
        if not frame_path.exists():
            return None
        # The key=value directories are for readers of a whole domain; one
        # file holds its own columns, so do not infer partition columns here.
        table = pq.read_table(frame_path, partitioning=None)
        stored = (table.schema.metadata or {}).get(self.METADATA_KEY)
        if stored != self._canonical(query):
            return None
        return table.to_pandas()

    def write(
        self,
        query: Mapping[str, Any],
        partitions: Mapping[str, Any],
        frame: pd.DataFrame,
        raw_payload: Any,
    ) -> None:
        if not self.enabled:
            return
        raw_path, frame_path = self.paths(partitions)
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
        partitions: Mapping[str, Any],
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return an exact cached query or fetch and atomically persist it.

        ``partitions`` name the directories, outermost first.
        """
        if not refresh:
            cached = self.read(query, partitions)
            if cached is not None:
                return cached
        result = fetch()
        self.write(query, partitions, result.frame, result.raw_payload)
        return result.frame


def _path_part(value: str) -> str:
    """One filesystem-safe path component, readable where the input already is."""
    part = re.sub(r"[^A-Za-z0-9._=-]", "_", str(value).strip())
    if part in {"", ".", ".."}:
        raise ValueError(f"invalid cache path component {value!r}")
    return part
