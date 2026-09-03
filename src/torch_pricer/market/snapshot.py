"""The state of the world at one instant, as the pricer sees it.

A snapshot holds *calibrated* objects, not quotes: curves that have been fitted
and a vol model that has been calibrated. Raw observations live in
:mod:`torch_pricer.market_data`, upstream of here.

``as_of`` is an ordinary date and the only date in the pricing path. Everything
below this layer measures time as a year fraction from it, so there is no global
evaluation date for a concurrent request to move underneath a valuation.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass

import torch
from torch import Tensor

from ..calibration.curve_model.curves import RateCurve
from ..calibration.vol_model.base import VolModel
from ..calibration.vol_model.black import BlackScholesModel
from ..conventions import DEFAULT_DAY_COUNT, to_date, year_fraction
from ..tensors import as_tensor


@dataclass(frozen=True)
class MarketSnapshot:
    """Spot, funding, dividends and dynamics, all dated ``as_of``.

    Build one with :meth:`flat` for a quick mark, or assemble the pieces
    yourself for a real one.

    ``spot`` is a tensor rather than a float because delta is
    ``autograd.grad(price, spot)``. The engine replaces it with a fresh leaf per
    pricing call, so a snapshot can be reused without accumulating graph.
    """

    as_of: dt.date
    spot: Tensor
    discount: RateCurve
    dividend: RateCurve
    vol: VolModel
    ticker: str = ""
    day_count: str = DEFAULT_DAY_COUNT

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
        day_count: str = DEFAULT_DAY_COUNT,
    ) -> MarketSnapshot:
        """Flat curves and a constant vol -- the Black-Scholes textbook market."""
        return cls(
            as_of=to_date(as_of),
            spot=as_tensor(float(spot)),
            discount=RateCurve.flat(rate, "flat-r"),
            dividend=RateCurve.flat(dividend, "flat-q"),
            vol=BlackScholesModel(vol),
            ticker=ticker,
            day_count=day_count,
        )

    # -- queries --------------------------------------------------------
    def time_to(self, d: dt.date | str) -> float:
        """Year fraction from ``as_of`` to ``d`` under this snapshot's day count."""
        return year_fraction(self.as_of, to_date(d), self.day_count)

    def forward(self, t) -> Tensor:
        """Forward level to ``t`` years: ``S D_q(t) / D_r(t)``."""
        return self.spot * self.dividend.discount(t) / self.discount.discount(t)

    def to(self, device=None, dtype: torch.dtype | None = None) -> MarketSnapshot:
        """Move the tensor-bearing pieces to a device or dtype."""
        if device is None and dtype is None:
            return self
        return dataclasses.replace(
            self,
            spot=self.spot.to(device=device, dtype=dtype),
            discount=self.discount.to(device=device, dtype=dtype),
            dividend=self.dividend.to(device=device, dtype=dtype),
            vol=self.vol.to(device=device, dtype=dtype),
        )

    def with_spot(self, spot: Tensor) -> MarketSnapshot:
        """A copy marked at a different spot, sharing the curves and model."""
        return dataclasses.replace(self, spot=as_tensor(spot))

    def __repr__(self) -> str:  # pragma: no cover
        tag = f"{self.ticker} " if self.ticker else ""
        return f"MarketSnapshot({tag}{self.as_of}, S={float(self.spot):.4f}, {self.vol!r})"
