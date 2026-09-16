"""Public types used by the Massive equity historical loader."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

TickerInput = str | Sequence[str]
DateInput = str | pd.Timestamp | Sequence[str | pd.Timestamp]
MatchMode = Literal["exact", "backward"]
ResolvedUnderlyingAsset = Literal["stocks", "indices"]
UnderlyingAsset = Literal["auto", "stocks", "indices"]
FlatAsset = Literal["options", "stocks", "indices"]


@dataclass(frozen=True)
class UnderlyingSpec:
    """Resolved provider routing information for one option underlying.

    ``canonical_ticker`` is the internal symbol (for example ``SPX``), while
    ``provider_ticker`` retains Massive symbology (for example ``I:SPX``).
    """

    canonical_ticker: str
    asset_type: ResolvedUnderlyingAsset
    provider_ticker: str
    exercise_style: str | None = None
    settlement_method: str | None = None
    shares_per_contract: float | None = None
    source_option_ticker: str | None = None
    metadata_source: str = "contract"
