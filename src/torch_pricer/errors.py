"""Exception hierarchy.

Every error this package raises on its own derives from :class:`PricerError`, so
a caller can wrap a pricing call in one ``except`` and be sure a QuantLib
``RuntimeError`` (which is what SWIG raises for every C++ exception) does not
leak through untyped. QuantLib is only reachable from the date-convention layer,
so in practice that is the only place such a wrap is needed.
"""

from __future__ import annotations


class PricerError(Exception):
    """Base class for everything raised here."""


class ValidationError(PricerError):
    """A caller-supplied value is outside the domain the model accepts."""


class MarketDataError(PricerError):
    """Market data is missing, stale, or internally inconsistent."""


class CalibrationError(PricerError):
    """A calibration failed to converge, or converged to a rejected point."""


class PricingError(PricerError):
    """The pricing call itself failed."""


class EngineNotAvailable(PricingError):
    """No registered engine supports this (instrument, engine name) pair."""
