"""What the market actually publishes.

These are the inputs to :mod:`torch_pricer.calibration`, kept strictly separate
from its outputs. A quote is an observation with a bid, an ask and a timestamp;
a curve or a vol model is a fitted object with parameters. Collapsing the two --
which is what a ``MarketSnapshot`` holding raw quotes would do -- makes it
impossible to say whether a number came from the market or from a fit.

Deliberately dumb: no interpolation, no arbitrage checks, no QuantLib, no torch.
Validation that needs a model belongs in the calibrator that consumes these.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..errors import ValidationError
from ..instruments.spec import Right


def _mid(bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None:
        return 0.5 * (bid + ask)
    return last


@dataclass(frozen=True, slots=True)
class SpotQuote:
    """The underlying's level."""

    ticker: str
    value: float
    as_of: dt.date | None = None


@dataclass(frozen=True, slots=True)
class RateQuote:
    """A point on the funding or dividend curve, quoted by tenor."""

    tenor: str
    rate: float
    kind: str = "zero"  # zero | deposit | swap | dividend


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """One listed option's market.

    Either a premium (``bid``/``ask``/``last``) or an ``implied_vol`` may be
    absent; a calibrator should take whichever it is given and say so if both
    are missing.
    """

    expiry: dt.date
    strike: float
    right: Right = Right.CALL
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    implied_vol: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "right", Right(self.right))
        if self.strike <= 0:
            raise ValidationError(f"strike must be positive, got {self.strike}")
        if self.price is None and self.implied_vol is None:
            raise ValidationError(
                f"quote {self.expiry} {self.strike:g} carries neither a premium nor a vol"
            )

    @property
    def price(self) -> float | None:
        """Mid where there is a two-sided market, else the last trade."""
        return _mid(self.bid, self.ask, self.last)


@dataclass(frozen=True, slots=True)
class QuoteSet:
    """Everything observed for one underlying at one instant."""

    as_of: dt.date
    spot: SpotQuote
    options: tuple[OptionQuote, ...] = ()
    rates: tuple[RateQuote, ...] = ()
    dividends: tuple[RateQuote, ...] = ()

    def expiries(self) -> tuple[dt.date, ...]:
        return tuple(sorted({q.expiry for q in self.options}))

    def slice(self, expiry: dt.date) -> tuple[OptionQuote, ...]:
        """The quotes for one expiry, in strike order."""
        return tuple(sorted((q for q in self.options if q.expiry == expiry), key=lambda q: q.strike))
