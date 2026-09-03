"""Instrument definitions.

These are plain frozen dataclasses, deliberately free of both QuantLib and
torch types: a spec is something you can build from a JSON request, put in a
database, hash into a cache key, and diff in a test. Turning one into a payoff
over simulated paths happens in :mod:`torch_pricer.instruments.payoff`.

A spec describes one contract. Quantities belong on a position, not here.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum

from ..conventions import DEFAULT_CONTRACT_MULTIPLIER
from ..errors import ValidationError


class Right(str, Enum):
    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        """+1 for a call, -1 for a put."""
        return 1 if self is Right.CALL else -1


class Style(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"


class AverageKind(str, Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


def _parse_date(d: dt.date | str) -> dt.date:
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


@dataclass(frozen=True, slots=True)
class Instrument:
    """Base for everything that can be priced."""

    underlying: str = ""

    @property
    def kind(self) -> str:
        return type(self).__name__

    @property
    def expiry(self) -> dt.date | None:
        return None

    def describe(self) -> str:
        return self.kind


@dataclass(frozen=True, slots=True)
class Stock(Instrument):
    """A share of the underlying. Present so a hedged book is one homogeneous list."""

    @property
    def expiry(self) -> None:
        return None

    def describe(self) -> str:
        return f"{self.underlying or 'stock'}"


@dataclass(frozen=True, slots=True)
class VanillaOption(Instrument):
    """A listed equity option: call or put, European, American or Bermudan."""

    strike: float = 0.0
    maturity: dt.date | None = None
    right: Right = Right.CALL
    style: Style = Style.EUROPEAN
    exercise_dates: tuple[dt.date, ...] = ()
    multiplier: float = DEFAULT_CONTRACT_MULTIPLIER

    def __post_init__(self) -> None:
        if self.maturity is None:
            raise ValidationError("a VanillaOption needs a maturity")
        object.__setattr__(self, "maturity", _parse_date(self.maturity))
        object.__setattr__(self, "right", Right(self.right))
        object.__setattr__(self, "style", Style(self.style))
        object.__setattr__(
            self, "exercise_dates", tuple(sorted(_parse_date(d) for d in self.exercise_dates))
        )
        if self.strike <= 0:
            raise ValidationError(f"strike must be positive, got {self.strike}")
        if self.style is Style.BERMUDAN and not self.exercise_dates:
            raise ValidationError("a Bermudan option needs exercise_dates")
        if self.exercise_dates and self.exercise_dates[-1] > self.maturity:
            raise ValidationError("exercise_dates cannot fall after maturity")

    @property
    def expiry(self) -> dt.date:
        return self.maturity

    def describe(self) -> str:
        tag = f"{self.underlying} " if self.underlying else ""
        return (
            f"{tag}{self.maturity:%Y-%m-%d} {self.strike:g} "
            f"{self.right.value} ({self.style.value})"
        )


@dataclass(frozen=True, slots=True)
class AsianOption(Instrument):
    """Fixed-strike Asian on the average of the fixing dates.

    With no ``fixing_dates`` the average is continuous, which only the geometric
    analytic engine supports. Discrete fixings are what actually trades.
    """

    strike: float = 0.0
    maturity: dt.date | None = None
    right: Right = Right.CALL
    average: AverageKind = AverageKind.ARITHMETIC
    fixing_dates: tuple[dt.date, ...] = ()
    past_fixings: int = 0
    running_accumulator: float | None = None
    multiplier: float = DEFAULT_CONTRACT_MULTIPLIER

    def __post_init__(self) -> None:
        if self.maturity is None:
            raise ValidationError("an AsianOption needs a maturity")
        object.__setattr__(self, "maturity", _parse_date(self.maturity))
        object.__setattr__(self, "right", Right(self.right))
        object.__setattr__(self, "average", AverageKind(self.average))
        object.__setattr__(
            self, "fixing_dates", tuple(sorted(_parse_date(d) for d in self.fixing_dates))
        )
        if self.strike <= 0:
            raise ValidationError("strike must be positive")
        if self.fixing_dates and self.fixing_dates[-1] > self.maturity:
            raise ValidationError("fixing dates cannot fall after maturity")
        if self.past_fixings < 0:
            raise ValidationError("past_fixings cannot be negative")
        if self.past_fixings and self.running_accumulator is None:
            raise ValidationError(
                "past_fixings > 0 needs running_accumulator: the sum (arithmetic) or "
                "product (geometric) of the fixings already observed"
            )

    @property
    def expiry(self) -> dt.date:
        return self.maturity

    def describe(self) -> str:
        n = len(self.fixing_dates) or "continuous"
        return (
            f"{self.maturity:%Y-%m-%d} {self.strike:g} {self.average.value} "
            f"asian {self.right.value} ({n} fixings)"
        )

OPTION_TYPES = (VanillaOption, AsianOption)

