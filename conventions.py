"""Dates, calendars, day counts, and the global-state discipline QuantLib needs.

QuantLib keeps the pricing date in a process-wide singleton
(``ql.Settings.instance().evaluationDate``) and every term structure built with
a *relative* reference date silently follows it. That is the single most common
way a QuantLib service returns a wrong number in production: request A moves the
evaluation date while request B is halfway through a valuation.

Two rules follow, and this module is where they are enforced:

1. Never assign to ``evaluationDate`` directly. Use :func:`evaluation_date`,
   which restores the previous value on exit even if the body raises.
2. Everything that reads the global date holds :data:`PRICING_LOCK` while it
   does. The lock is reentrant, so nesting is safe.

The lock serialises pricing across threads. That is deliberate: QuantLib's
Python bindings are not thread-safe, and a correct number computed on one core
beats four cores racing. For real parallelism, fan out over *processes*.
"""

from __future__ import annotations

import datetime as dt
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import QuantLib as ql

from .errors import ValidationError

#: Reentrant lock guarding QuantLib's global evaluation date.
PRICING_LOCK = threading.RLock()

_CALENDARS: dict[str, ql.Calendar] = {
    "NYSE": ql.UnitedStates(ql.UnitedStates.NYSE),
    "US_SETTLEMENT": ql.UnitedStates(ql.UnitedStates.Settlement),
    "US_GOVERNMENTBOND": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    "TARGET": ql.TARGET(),
    "UK": ql.UnitedKingdom(),
    "JAPAN": ql.Japan(),
    "NULL": ql.NullCalendar(),
}

_DAY_COUNTS: dict[str, ql.DayCounter] = {
    "ACT/365F": ql.Actual365Fixed(),
    "ACT/360": ql.Actual360(),
    "ACT/ACT": ql.ActualActual(ql.ActualActual.ISDA),
    "30/360": ql.Thirty360(ql.Thirty360.BondBasis),
    "BUS/252": ql.Business252(_CALENDARS["NYSE"]),
}

#: What the rest of the package uses when a caller does not say otherwise.
DEFAULT_CALENDAR = "NYSE"
DEFAULT_DAY_COUNT = "ACT/365F"
#: US equity options settle T+1; the forward used for pricing starts there.
DEFAULT_SETTLEMENT_DAYS = 1
#: Contract multiplier for a standard US listed equity option.
DEFAULT_CONTRACT_MULTIPLIER = 100.0


def calendar(name: str = DEFAULT_CALENDAR) -> ql.Calendar:
    """Look up a calendar by name. Case-insensitive."""
    try:
        return _CALENDARS[name.upper()]
    except KeyError:
        raise ValidationError(
            f"unknown calendar {name!r}; known: {sorted(_CALENDARS)}"
        ) from None


def day_count(name: str = DEFAULT_DAY_COUNT) -> ql.DayCounter:
    """Look up a day count convention by name. Case-insensitive."""
    try:
        return _DAY_COUNTS[name.upper()]
    except KeyError:
        raise ValidationError(
            f"unknown day count {name!r}; known: {sorted(_DAY_COUNTS)}"
        ) from None


def to_ql(d: dt.date | dt.datetime | ql.Date | str) -> ql.Date:
    """Coerce a date-like value to ``ql.Date``. Strings must be ISO ``YYYY-MM-DD``."""
    if isinstance(d, ql.Date):
        return d
    if isinstance(d, str):
        try:
            d = dt.date.fromisoformat(d)
        except ValueError as exc:
            raise ValidationError(f"expected ISO date YYYY-MM-DD, got {d!r}") from exc
    if isinstance(d, dt.datetime):
        d = d.date()
    if not isinstance(d, dt.date):
        raise ValidationError(f"cannot convert {type(d).__name__} to a date")
    return ql.Date(d.day, d.month, d.year)


def from_ql(d: ql.Date) -> dt.date:
    """Convert a ``ql.Date`` back to ``datetime.date``."""
    return dt.date(d.year(), d.month(), d.dayOfMonth())


def year_fraction(
    start: dt.date | ql.Date,
    end: dt.date | ql.Date,
    convention: str = DEFAULT_DAY_COUNT,
) -> float:
    """Year fraction between two dates under the named day count."""
    return day_count(convention).yearFraction(to_ql(start), to_ql(end))


@contextmanager
def evaluation_date(d: dt.date | ql.Date | str) -> Iterator[ql.Date]:
    """Set QuantLib's global evaluation date for the duration of the block.

    Holds :data:`PRICING_LOCK` throughout and restores the previous date on the
    way out, including on exception. This is the only supported way to move the
    pricing date.

    >>> with evaluation_date("2026-03-26") as d:      # doctest: +SKIP
    ...     price = pricer.price(spec, market)
    """
    qd = to_ql(d)
    with PRICING_LOCK:
        previous = ql.Settings.instance().evaluationDate
        ql.Settings.instance().evaluationDate = qd
        try:
            yield qd
        finally:
            ql.Settings.instance().evaluationDate = previous


def current_evaluation_date() -> dt.date:
    """The evaluation date QuantLib would use right now."""
    with PRICING_LOCK:
        return from_ql(ql.Settings.instance().evaluationDate)


def business_days_between(
    start: dt.date | ql.Date,
    end: dt.date | ql.Date,
    calendar_name: str = DEFAULT_CALENDAR,
) -> int:
    """Business days in ``(start, end]`` on the named calendar."""
    return calendar(calendar_name).businessDaysBetween(to_ql(start), to_ql(end))


def advance(
    d: dt.date | ql.Date,
    n: int,
    unit: str = "D",
    calendar_name: str = DEFAULT_CALENDAR,
    convention: int = ql.Following,
) -> dt.date:
    """Advance a date by ``n`` periods, rolling to a good business day."""
    units = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}
    try:
        ql_unit = units[unit.upper()]
    except KeyError:
        raise ValidationError(f"unknown unit {unit!r}; use one of {sorted(units)}") from None
    return from_ql(calendar(calendar_name).advance(to_ql(d), n, ql_unit, convention))
