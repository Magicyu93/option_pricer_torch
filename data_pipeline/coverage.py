"""What is in the cache: every file, with its dataset, partitions and size.

Reads the directory layout that :func:`~data_pipeline.common.storage.partitioned_path`
writes -- ``<root>/<layer>/<dataset>/<key>=<value>/.../<kind>_<provider>.<ext>`` --
and touches no file contents, so it is cheap enough to call on every page load.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common.storage import DEFAULT_DATA_DIR, parse_partitioned_path

COVERAGE_COLUMNS = ["layer", "dataset", "date", "year", "ticker", "series", "kind", "provider", "bytes", "path"]


def coverage(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """One row per cached file under ``raw/`` and ``parquet/``.

    ``date`` is set for daily partitions (equity), ``year`` for yearly ones
    (rates, dividends); ``ticker`` and ``series`` for whichever partition names
    them. Paths are read with :func:`~data_pipeline.common.storage.parse_partitioned_path`,
    the inverse of the function that wrote them; anything else is skipped.
    """
    root = Path(data_dir)
    rows = []
    for layer in ("raw", "parquet"):
        base = root / layer
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            parsed = parse_partitioned_path(base, path) if path.is_file() else None
            if parsed is None:
                continue
            partitions = parsed["partitions"]
            rows.append({
                "layer": layer,
                "dataset": parsed["domain"],
                "date": partitions.get("date"),
                "year": partitions.get("year"),
                "ticker": partitions.get("ticker"),
                "series": partitions.get("dataset"),
                "kind": parsed["kind"],
                "provider": parsed["provider"],
                "bytes": path.stat().st_size,
                "path": str(path),
            })
    frame = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    return frame.sort_values(["layer", "dataset", "date", "year", "ticker"], na_position="first").reset_index(drop=True)


def cached_days(dataset: str, ticker: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> list[str]:
    """Days with a processed (parquet) file for ``ticker`` in ``dataset``, oldest first."""
    frame = coverage(data_dir)
    hit = frame[(frame["layer"] == "parquet") & (frame["dataset"] == dataset) & (frame["ticker"] == ticker)]
    return sorted(hit["date"].dropna().unique().tolist())
