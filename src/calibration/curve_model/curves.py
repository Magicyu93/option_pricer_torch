from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from abc import ABC, abstractmethod
import QuantLib as ql

from conventions import (
    DEFAULT_CALENDAR,
    DEFAULT_DAY_COUNT,
    calendar,
    day_count,
    from_ql,
    to_ql,
)
from errors import MarketDataError, ValidationError

_TENOR_UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}

def parse_tenor(tenor: str) -> ql.Period:
    """Parse ``"3M"``, ``"10Y"``, ``"1W"`` into a ``ql.Period``."""
    s = tenor.strip().upper()
    if len(s) < 2 or not s[:-1].isdigit() or s[-1] not in _TENOR_UNITS:
        raise ValidationError(f"cannot parse tenor {tenor!r}; expected e.g. '3M', '10Y'")
    return ql.Period(int(s[:-1]), _TENOR_UNITS[s[-1]])


class RateCurve(ABC):
    """A discount curve with a bumpable parallel spread.

    Construct through the classmethods rather than ``__init__``.
    """

    __slots__ = ("_base", "_base_handle", "_spread", "_ts", "_handle", "_dc", "_dc_name", "_label")

    def __init__(
        self,
        base: ql.YieldTermStructure,
        spread: float = 0.0,
        day_count_name: str = DEFAULT_DAY_COUNT,
        label: str = "curve",
    ) -> None:
        self._base = base
        self._base_handle = ql.YieldTermStructureHandle(base)
        self._spread = ql.SimpleQuote(spread)
        self._dc_name = day_count_name
        self._dc = day_count(day_count_name)
        self._ts = ql.ZeroSpreadedTermStructure(
            self._base_handle, ql.QuoteHandle(self._spread), ql.Continuous, ql.NoFrequency, self._dc
        )
        self._ts.enableExtrapolation()
        self._handle = ql.YieldTermStructureHandle(self._ts)
        self._label = label

    # -- constructors ---------------------------------------------------
    @classmethod
    def flat(
        cls,
        rate: float,
        as_of: dt.date | ql.Date | str,
        day_count_name: str = DEFAULT_DAY_COUNT,
        calendar_name: str = DEFAULT_CALENDAR,
        label: str = "flat",
    ) -> RateCurve:
        """A flat continuously-compounded curve at ``rate``."""
        base = ql.FlatForward(
            to_ql(as_of), ql.QuoteHandle(ql.SimpleQuote(rate)),
            day_count(day_count_name), ql.Continuous, ql.NoFrequency,
        )
        base.enableExtrapolation()
        return cls(base, 0.0, day_count_name, label)


    @abstractmethod
    def zero_rate(self) -> float:
        pass

    @abstractmethod
    def forward_rate(self) -> float:
        pass

    @abstractmethod
    def discount_rate(self) -> float:
        pass
