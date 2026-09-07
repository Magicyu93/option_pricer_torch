"""Dates, calendars and day counts.

This is the only module that touches QuantLib, and it only ever asks it
questions about the calendar. Nothing here returns a term structure, and nothing
here reads or writes ``ql.Settings.instance().evaluationDate``.

That is deliberate. QuantLib keeps the pricing date in a process-wide singleton
and every term structure built with a *relative* reference date silently follows
it, which is the classic way a QuantLib service returns a wrong number under
concurrency. This package sidesteps the problem rather than locking around it:
curves are :class:`~torch_pricer.calibration.curve_model.curves.RateCurve`
objects holding torch tensors indexed by year fraction, and the valuation date
lives on the :class:`~torch_pricer.market.snapshot.MarketSnapshot` that the
caller passes in. There is no global pricing state to guard, so pricing calls do
not serialise against each other.
"""

from __future__ import annotations

import datetime as dt

import QuantLib as ql

from torch_pricer.errors import ValidationError

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

_TENOR_UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}

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


def parse_tenor(tenor: str) -> ql.Period:
    """Parse ``"3M"``, ``"10Y"``, ``"1W"`` into a ``ql.Period``."""
    s = tenor.strip().upper()
    if len(s) < 2 or not s[:-1].isdigit() or s[-1] not in _TENOR_UNITS:
        raise ValidationError(f"cannot parse tenor {tenor!r}; expected e.g. '3M', '10Y'")
    return ql.Period(int(s[:-1]), _TENOR_UNITS[s[-1]])


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


def to_date(d: dt.date | dt.datetime | ql.Date | str) -> dt.date:
    """Coerce a date-like value to ``datetime.date``."""
    if isinstance(d, ql.Date):
        return from_ql(d)
    if isinstance(d, str):
        try:
            return dt.date.fromisoformat(d)
        except ValueError as exc:
            raise ValidationError(f"expected ISO date YYYY-MM-DD, got {d!r}") from exc
    if isinstance(d, dt.datetime):
        return d.date()
    if not isinstance(d, dt.date):
        raise ValidationError(f"cannot convert {type(d).__name__} to a date")
    return d


def year_fraction(
    start: dt.date | ql.Date | str,
    end: dt.date | ql.Date | str,
    convention: str = DEFAULT_DAY_COUNT,
) -> float:
    """Year fraction between two dates under the named day count.

    This is the bridge between the contract's calendar dates and the simulator's
    continuous time axis: everything downstream of here measures time in years
    from the snapshot's ``as_of``.
    """
    return day_count(convention).yearFraction(to_ql(start), to_ql(end))


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
    try:
        ql_unit = _TENOR_UNITS[unit.upper()]
    except KeyError:
        raise ValidationError(
            f"unknown unit {unit!r}; use one of {sorted(_TENOR_UNITS)}"
        ) from None
    return from_ql(calendar(calendar_name).advance(to_ql(d), n, ql_unit, convention))
