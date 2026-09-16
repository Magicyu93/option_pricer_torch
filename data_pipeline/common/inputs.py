"""Input normalization shared by independent market-data domains."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def normalize_identifiers(
    values: str | Sequence[str],
    *,
    label: str,
    transform: Callable[[str], str] = str.upper,
) -> list[str]:
    """Strip, transform, validate, and de-duplicate identifiers in order."""
    candidates = [values] if isinstance(values, str) else list(values)
    normalized: list[str] = []
    for value in candidates:
        if value is None:
            continue
        item = str(value).strip()
        if item:
            normalized.append(transform(item))
    if not normalized:
        raise ValueError(f"At least one {label} is required")
    return list(dict.fromkeys(normalized))
