"""Provider-aware raw market-data pipelines."""

from .coverage import cached_days, coverage
from .dividends import DividendsDataLoader, MassiveDividendsSource
from .download import (
    DownloadResult,
    credentials_status,
    download_dividends,
    download_equity,
    download_rates,
)
from .equity import MassiveHistoricalDataLoader, UnderlyingSpec
from .rates import (
    TREASURY_CMT_SERIES,
    FredRatesSource,
    MassiveTreasuryYieldsSource,
    RatesDataLoader,
)

__all__ = [
    "DownloadResult",
    "cached_days",
    "coverage",
    "credentials_status",
    "download_dividends",
    "download_equity",
    "download_rates",
    "TREASURY_CMT_SERIES",
    "DividendsDataLoader",
    "FredRatesSource",
    "MassiveDividendsSource",
    "MassiveHistoricalDataLoader",
    "MassiveTreasuryYieldsSource",
    "RatesDataLoader",
    "UnderlyingSpec",
]
