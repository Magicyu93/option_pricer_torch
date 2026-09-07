"""What a calibration consumes.

Deliberately minimal for now. The interface that lets bucketed vega flow back
through a calibration -- returning the Jacobian and Hessian at the optimum
rather than mutating in place -- is a separate piece of design work; see the
note on :meth:`torch_pricer.models.base.Model.calibrate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch_pricer.market.market_data import QuoteSet
from torch_pricer.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class CalibrationInputs:
    """The market state a model is fitted against, plus the quotes to fit."""

    market: MarketSnapshot
    quotes: QuoteSet | None = None
    weights: dict[str, float] = field(default_factory=dict)
