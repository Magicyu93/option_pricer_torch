"""The calibration contract: what a calibration consumes, and what it returns.

Shared by every calibrator -- the curve and surface fits under
:mod:`torch_pricer.market`, and :meth:`torch_pricer.models.base.Model.calibrate`.

Deliberately minimal for now. The interface that lets bucketed vega flow back
through a calibration -- returning the Jacobian and Hessian at the optimum
rather than mutating in place -- is a separate piece of design work; see the
note on :meth:`torch_pricer.models.base.Model.calibrate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from torch_pricer.market.market_data import QuoteSet
from torch_pricer.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class CalibrationInputs:
    """The market state a model is fitted against, plus the quotes to fit."""

    market: MarketSnapshot
    quotes: QuoteSet | None = None
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationResult:
    """A fit's parameters, quality and residuals.

    ``residuals`` has one row per quote. ``diagnostics`` holds checks a reader
    should look at before trusting the fit -- per-expiry error, and whether the
    no-arbitrage conditions hold numerically, not just by construction.

    Chaining sensitivities through the fit (a ``market_sensitivity``: vega to
    quotes rather than to parameters) needs the Jacobian and Hessian at the
    optimum and is not implemented yet.
    """

    parameters: dict[str, object]
    loss: float
    rmse: float
    converged: bool
    iterations: int
    residuals: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)
