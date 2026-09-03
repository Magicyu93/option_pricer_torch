"""Black-Scholes: one constant volatility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ...errors import CalibrationError, ValidationError
from ...analytics.black import black_price
from ...simulator.gbm import GeometricBrownianMotion
from ...tensors import as_tensor
from ..inputs import CalibrationInputs
from .base import VolModel

if TYPE_CHECKING:  # pragma: no cover
    from ...market.snapshot import MarketSnapshot
    from ...simulator.simulator import SDE


class BlackScholesModel(VolModel):
    """Constant vol, lognormal spot.

    The volatility is stored as its logarithm so that an unconstrained optimiser
    cannot walk it negative. Read it through :attr:`vol`.
    """

    def __init__(self, vol: float = 0.20):
        super().__init__()
        if vol <= 0:
            raise ValidationError(f"volatility must be positive, got {vol}")
        self._log_vol = nn.Parameter(as_tensor(float(vol)).log())

    @property
    def vol(self) -> torch.Tensor:
        """Volatility, positive by construction."""
        return self._log_vol.exp()

    def to_sde(self, market: MarketSnapshot) -> SDE:
        return GeometricBrownianMotion(
            vol=self.vol, discount=market.discount, dividend=market.dividend
        )

    def calibrate(self, inputs: CalibrationInputs, tolerance: float | None = None) -> None:
        """Fit the one volatility to ``inputs.quotes``. Mutates in place.

        One parameter and a strictly monotone objective, so this is a root find
        dressed as an optimisation: LBFGS on the log-vol converges in a handful
        of iterations from any start. The residuals are vega-normalised prices
        for the same reason as in
        :mod:`torch_pricer.calibration.vol_model.heston_cal` -- unweighted price
        errors would fit the in-the-money quotes, which carry almost no vol.

        A single number cannot reproduce a smile, and this does not pretend to:
        what comes back is the vega-weighted average level of the quoted vols,
        and :attr:`fit_report` says how far the quotes were from it.
        """
        from .heston_cal import _targets  # the same quote flattening, weights and all

        targets = _targets(inputs)

        def objective() -> torch.Tensor:
            model = black_price(
                targets["forward"], targets["strike"], targets["t"], self.vol,
                targets["discount"], targets["right"],
            )
            return (((model - targets["price"]) / targets["vega"]) ** 2 * targets["weight"]).mean()

        optimiser = torch.optim.LBFGS(
            [self._log_vol], max_iter=100, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        optimiser.step(closure)

        with torch.no_grad():
            rmse = float(objective().sqrt())
        self.fit_report = {"rmse_vol": rmse, "n_quotes": int(targets["strike"].numel())}
        if tolerance is not None and rmse > tolerance:
            raise CalibrationError(
                f"a flat vol left an RMSE of {rmse:.4%} across the quotes, above {tolerance:.4%}; "
                "the market has a smile and this model has no way to hold one"
            )

    def extra_repr(self) -> str:  # pragma: no cover
        return f"vol={float(self.vol.detach()):.4f}"
