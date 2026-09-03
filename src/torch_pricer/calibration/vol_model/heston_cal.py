"""Heston calibration.

Fits ``(v0, kappa, theta, xi, rho)`` to a quote set. Every parameter is stored
under an unconstrained reparameterisation as described in
:mod:`torch_pricer.calibration.vol_model.base` -- ``exp`` for the four positive
ones, ``tanh`` for the correlation -- so a torch optimiser can run free and the
model can never be evaluated outside its feasible set.

The objective is priced by :mod:`torch_pricer.analytics.heston`, not by Monte
Carlo. A calibration evaluates the objective hundreds of times; a semi-analytic
price makes that a second's work, and it is differentiable, so the optimiser
gets exact gradients rather than a bumped Jacobian of a noisy function.

Errors are measured in *vega-normalised price*, ``(model - market) / vega``.
That is the first-order approximation to the implied-vol error, which is what a
desk means by a good fit, and it avoids putting an implied-vol solve inside
every objective evaluation. Without it a fit is dominated by the in-the-money
quotes, whose premiums are large and whose vol content is small.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ...analytics.black import black_vega
from ...analytics.heston import heston_price
from ...errors import CalibrationError, ValidationError
from ...simulator.heston import HestonProcess
from ...tensors import EPS, as_tensor
from ..inputs import CalibrationInputs
from .base import VolModel

if TYPE_CHECKING:  # pragma: no cover
    from ...market.snapshot import MarketSnapshot
    from ...simulator.simulator import SDE

#: The vega normalisation is floored, absolutely and relative to the largest
#: vega in the set. Without the relative floor a quote worth a fraction of a
#: basis point -- a one-month 40% out of the money call -- divides a rounding
#: error by a vega of ``1e-5`` and produces the largest residual in the
#: objective. Such a quote carries no information about volatility; the floor is
#: how the objective is told so.
_MIN_VEGA = 1e-4
_RELATIVE_VEGA_FLOOR = 1e-2


class HestonModel(VolModel):
    """Heston, with the five parameters an optimiser is allowed to move.

    Read the parameters through the properties, never through the underlying
    ``_log_*`` buffers: the properties are what :meth:`to_sde` hands the
    simulator, and they are the tensors risk is taken against.
    """

    def __init__(
        self,
        v0: float = 0.04,
        kappa: float = 1.5,
        theta: float = 0.04,
        xi: float = 0.5,
        rho: float = -0.7,
    ):
        super().__init__()
        for name, value in (("v0", v0), ("kappa", kappa), ("theta", theta), ("xi", xi)):
            if value <= 0:
                raise ValidationError(f"{name} must be positive, got {value}")
        if not -1.0 < rho < 1.0:
            raise ValidationError(f"rho must lie strictly inside (-1, 1), got {rho}")

        self._log_v0 = nn.Parameter(as_tensor(float(v0)).log())
        self._log_kappa = nn.Parameter(as_tensor(float(kappa)).log())
        self._log_theta = nn.Parameter(as_tensor(float(theta)).log())
        self._log_xi = nn.Parameter(as_tensor(float(xi)).log())
        self._atanh_rho = nn.Parameter(as_tensor(float(rho)).atanh())

    # -- parameters -----------------------------------------------------
    @property
    def v0(self) -> Tensor:
        """Instantaneous variance at the snapshot date."""
        return self._log_v0.exp()

    @property
    def kappa(self) -> Tensor:
        """Speed of mean reversion of the variance."""
        return self._log_kappa.exp()

    @property
    def theta(self) -> Tensor:
        """Long-run variance."""
        return self._log_theta.exp()

    @property
    def xi(self) -> Tensor:
        """Volatility of variance."""
        return self._log_xi.exp()

    @property
    def rho(self) -> Tensor:
        """Spot/variance correlation, in ``(-1, 1)`` by construction."""
        return self._atanh_rho.tanh()

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / xi^2``. Below 1 the variance process touches zero.

        Not enforced. Equity smiles routinely fit best at a ratio below 1, and a
        constraint that rejects the market's own answer is not a useful
        constraint; the simulator's full truncation is what makes that safe.
        """
        return float((2.0 * self.kappa * self.theta / self.xi**2).detach())

    # -- pricing --------------------------------------------------------
    def price(self, forward, strike, t, discount=1.0, right=1) -> Tensor:
        """Semi-analytic European price under the current parameters."""
        return heston_price(
            forward, strike, t,
            v0=self.v0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho,
            discount=discount, right=right,
        )

    def to_sde(self, market: MarketSnapshot) -> SDE:
        return HestonProcess(
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
            discount=market.discount,
            dividend=market.dividend,
        )

    # -- calibration ----------------------------------------------------
    def calibrate(
        self,
        inputs: CalibrationInputs,
        iterations: int = 400,
        lr: float = 0.05,
        tolerance: float | None = None,
    ) -> None:
        """Fit the five parameters to ``inputs.quotes``. Mutates in place.

        Adam first, then LBFGS. Adam is insensitive to a bad starting point,
        which matters because the Heston objective has long flat valleys in
        ``(kappa, theta)``; LBFGS then converges quadratically once inside one.
        Running only the second from a cold start walks off into a corner of the
        reparameterised space and stalls.

        Args:
            inputs: market state plus the quotes to fit
            iterations: Adam steps before the LBFGS refinement
            lr: Adam learning rate, in unconstrained parameter units
            tolerance: raise :class:`~torch_pricer.errors.CalibrationError` if
                the final RMSE in vol terms exceeds this; ``None`` accepts any
                fit and leaves the caller to inspect :attr:`fit_report`
        """
        targets = _targets(inputs)
        params = list(self.parameters())

        def objective() -> Tensor:
            model = self.price(
                targets["forward"], targets["strike"], targets["t"],
                targets["discount"], targets["right"],
            )
            return (((model - targets["price"]) / targets["vega"]) ** 2 * targets["weight"]).mean()

        adam = torch.optim.Adam(params, lr=lr)
        for _ in range(int(iterations)):
            adam.zero_grad()
            loss = objective()
            loss.backward()
            adam.step()

        lbfgs = torch.optim.LBFGS(params, max_iter=100, line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            lbfgs.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        lbfgs.step(closure)

        with torch.no_grad():
            rmse = float(objective().sqrt())
        #: Root mean squared fit error, in volatility points.
        self.fit_report = {"rmse_vol": rmse, "n_quotes": int(targets["strike"].numel())}
        if tolerance is not None and rmse > tolerance:
            raise CalibrationError(
                f"Heston fit left an RMSE of {rmse:.4%} in vol terms, above {tolerance:.4%}"
            )

    def extra_repr(self) -> str:  # pragma: no cover
        d = {k: float(v.detach()) for k, v in
             (("v0", self.v0), ("kappa", self.kappa), ("theta", self.theta),
              ("xi", self.xi), ("rho", self.rho))}
        return (
            f"v0={d['v0']:.4f}, kappa={d['kappa']:.4f}, theta={d['theta']:.4f}, "
            f"xi={d['xi']:.4f}, rho={d['rho']:+.4f}"
        )


def _targets(inputs: CalibrationInputs) -> dict[str, Tensor]:
    """Flatten a quote set into the tensors the objective needs.

    A quote is used at its premium when it has one and at its implied vol
    otherwise, so a surface quoted in vols and a screen quoted in prices fit
    through the same objective.
    """
    # Local imports: analytics does not depend on calibration, and this keeps it
    # that way at module load time as well as on paper.
    from ...analytics.black import black_price

    quotes = inputs.quotes
    if quotes is None or not quotes.options:
        raise CalibrationError("calibration needs option quotes; inputs.quotes is empty")

    times, strikes, rights, prices, forwards, discounts, vols = [], [], [], [], [], [], []
    for quote in quotes.options:
        t = inputs.time_to(quote.expiry)
        if t <= 0:
            continue
        forward = inputs.forward(t)
        discount = inputs.discount.discount(t)
        vol = quote.implied_vol
        if quote.price is not None:
            price = as_tensor(float(quote.price))
        else:
            price = black_price(forward, quote.strike, t, vol, discount, quote.right.sign)
        times.append(t)
        strikes.append(float(quote.strike))
        rights.append(float(quote.right.sign))
        prices.append(price.reshape(()))
        forwards.append(forward.reshape(()))
        discounts.append(discount.reshape(()))
        vols.append(float(vol) if vol is not None else float("nan"))

    if not times:
        raise CalibrationError("every quote expires on or before the snapshot date")

    t = as_tensor(times)
    strike = as_tensor(strikes)
    forward = torch.stack(forwards)
    discount = torch.stack(discounts)
    market_price = torch.stack(prices)
    vol = as_tensor(vols)

    # Where a quote came without a vol, back one out of its premium for the
    # weighting only -- an approximate vega is enough to normalise with.
    from ...analytics.black import implied_vol as _implied_vol

    solved = _implied_vol(market_price, forward, strike, t, discount, as_tensor(rights))
    vol = torch.where(torch.isnan(vol), solved, vol)
    usable = ~torch.isnan(vol)
    if not bool(usable.any()):
        raise CalibrationError("no quote in the set implies a volatility")

    vega = black_vega(forward, strike, t, vol.nan_to_num(0.2), discount)
    vega = vega.clamp_min(max(_MIN_VEGA, _RELATIVE_VEGA_FLOOR * float(vega.detach().max())))
    return {
        "t": t,
        "strike": strike,
        "right": as_tensor(rights),
        "forward": forward.detach(),
        "discount": discount.detach(),
        "price": market_price.detach(),
        "vega": vega.detach(),
        # Quotes that imply no volatility at all (a premium outside the
        # no-arbitrage band) are kept in the tensors and weighted out, rather
        # than dropped, so the fit report's count matches the input.
        "weight": usable.to(t.dtype) / usable.to(t.dtype).mean().clamp_min(EPS),
    }
