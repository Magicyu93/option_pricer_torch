"""Provider-aware raw market-data pipelines."""

from .dividends import DividendsDataLoader, MassiveDividendsSource
from .equity import MassiveHistoricalDataLoader, UnderlyingSpec
from .rates import (
    TREASURY_CMT_SERIES,
    FredRatesSource,
    MassiveTreasuryYieldsSource,
    RatesDataLoader,
)

__all__ = [
    "TREASURY_CMT_SERIES",
    "DividendsDataLoader",
    "FredRatesSource",
    "MassiveDividendsSource",
    "MassiveHistoricalDataLoader",
    "MassiveTreasuryYieldsSource",
    "RatesDataLoader",
    "UnderlyingSpec",
]
