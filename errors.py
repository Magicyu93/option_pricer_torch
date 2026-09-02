"""Exception hierarchy.

Every error this package raises on its own derives from :class:`QLDeskError`, so
a caller can wrap a pricing call in one ``except`` and be sure a QuantLib
``RuntimeError`` (which is what SWIG raises for every C++ exception, including
"root not bracketed" and "negative discount") does not leak through untyped.
"""

from __future__ import annotations


class QLDeskError(Exception):
    """Base class for everything raised here."""


class ValidationError(QLDeskError):
    """A caller-supplied value is outside the domain the model accepts."""


class MarketDataError(QLDeskError):
    """Market data is missing, stale, or internally inconsistent."""


class CalibrationError(QLDeskError):
    """A calibration failed to converge, or converged to a rejected point."""


class PricingError(QLDeskError):
    """The pricing call itself failed."""


class EngineNotAvailable(PricingError):
    """No registered engine supports this (instrument, engine name) pair."""
