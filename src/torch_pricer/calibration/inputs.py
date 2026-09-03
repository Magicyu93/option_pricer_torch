"""What a calibration is given.

Every fit in this package needs the same four things -- a date, a spot, and the
two curves that make a forward -- plus whichever of the quotes and the implied
surface it consumes. Passing them as one frozen object rather than as a
lengthening argument list keeps
:meth:`~torch_pricer.calibration.vol_model.base.VolModel.calibrate` to a single
signature across models that need very different inputs: Black wants premiums,
Dupire wants a surface, and the leverage function in a local-stochastic model
wants both plus a simulation.

This deliberately mirrors :class:`~torch_pricer.market.snapshot.MarketSnapshot`
without being one. A snapshot holds a *calibrated* vol model; handing that to a
calibration would be circular, and would let a fit read the parameters it is
about to overwrite.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from ..conventions import DEFAULT_DAY_COUNT, to_date, year_fraction
from ..tensors import as_tensor
from .curve_model.curves import RateCurve
from .surface import VolSurface

if TYPE_CHECKING:  # pragma: no cover
    from ..market.snapshot import MarketSnapshot
    from ..market_data.quotes import QuoteSet


@dataclass(frozen=True)
class CalibrationInputs:
    """Market state a calibration reads, and the targets it fits to."""

    as_of: dt.date
    spot: Tensor
    discount: RateCurve
    dividend: RateCurve
    quotes: QuoteSet | None = None
    surface: VolSurface | None = None
    day_count: str = DEFAULT_DAY_COUNT

    @classmethod
    def from_market(
        cls,
        market: MarketSnapshot,
        quotes: QuoteSet | None = None,
        surface: VolSurface | None = None,
    ) -> CalibrationInputs:
        """Borrow the dated pieces of a snapshot, leaving its vol model behind."""
        return cls(
            as_of=market.as_of,
            spot=market.spot.detach(),
            discount=market.discount,
            dividend=market.dividend,
            quotes=quotes,
            surface=surface,
            day_count=market.day_count,
        )

    def time_to(self, d: dt.date | str) -> float:
        """Year fraction from ``as_of`` to ``d``."""
        return year_fraction(self.as_of, to_date(d), self.day_count)

    def forward(self, t) -> Tensor:
        """Forward level to ``t`` years: ``S D_q(t) / D_r(t)``."""
        t = as_tensor(t)
        return self.spot * self.dividend.discount(t) / self.discount.discount(t)
