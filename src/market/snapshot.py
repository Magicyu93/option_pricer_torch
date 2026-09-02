from __future__ import annotations

from ...conventions import DEFAULT_CALENDAR, DEFAULT_DAY_COUNT
from src.calibration.curve_model.curves import RateCurve

import QuantLib as ql
import datetime as dt
from dataclasses import dataclass

_TENOR_UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Spot, funding, dividends and volatility, all dated ``as_of``.

    Build one with :meth:`flat` for a quick mark, or assemble the pieces
    yourself for a real one.
    """

    as_of: dt.date
    spot: float
    discount: RateCurve
    dividend: RateCurve
    vol: VolModel
    ticker: str = ""

    # -- constructors ---------------------------------------------------
    @classmethod
    def flat(
        cls,
        as_of: dt.date | str,
        spot: float,
        rate: float = 0.0,
        dividend: float = 0.0,
        vol: float = 0.20,
        ticker: str = "",
        calendar_name: str = DEFAULT_CALENDAR,
        day_count_name: str = DEFAULT_DAY_COUNT,
    ) -> MarketSnapshot:
        """Flat curves and a constant vol — the Black-Scholes textbook market."""
        if isinstance(as_of, str):
            as_of = dt.date.fromisoformat(as_of)
        return cls(
            as_of=as_of,
            spot=float(spot),
            discount=RateCurve.flat(rate, as_of, day_count_name, calendar_name, "flat-r"),
            dividend=RateCurve.flat(dividend, as_of, day_count_name, calendar_name, "flat-q"),
            vol=FlatVol(vol, as_of, calendar_name, day_count_name),
            ticker=ticker,
        )

    

    def __repr__(self) -> str:  # pragma: no cover
        tag = f"{self.ticker} " if self.ticker else ""
        return f"MarketSnapshot({tag}{self.as_of}, S={self.spot:.4f}, {self.vol!r})"