"""Dupire local volatility, derived from an SVI surface inside the graph.

Local vol is the one model where market vega needs no implicit function theorem
at all. Dupire is an *explicit* formula, not an optimisation, so the whole chain

    slice parameters -> w(k, T) -> sigma_LV(S, t) -> SDE -> paths -> payoff -> V

is a single differentiable expression and ``dV/d(slice parameter)`` falls out of
one backward pass. Chaining that to per-quote vega is then only the small
per-slice fit Jacobian; see
:class:`~torch_pricer.calibration.result.CalibrationResult`.

In total implied variance over log-moneyness, Dupire reads

    sigma_LV^2 = (dw/dT) / D,
    D = 1 - (k/w) dw/dk + 0.25 (-0.25 - 1/w + k^2/w^2) (dw/dk)^2 + 0.5 d2w/dk2

which is why :mod:`torch_pricer.market.svi` works in those coordinates: written
over strike and implied vol instead, this is a page of chain rule.

Two guards matter, and neither is cosmetic:

* **Time floor.** As ``T -> 0`` the surface's total variance goes to zero, and
  ``k^2 / w^2`` diverges for every strike but the money. The formula is
  genuinely singular there, so ``t`` is floored rather than allowed to reach the
  first step's zero.
* **Variance floor.** ``D`` is positive only for an arbitrage-free surface. A
  fitted surface with a butterfly violation produces a negative local variance,
  and the honest response is to clamp and let
  :meth:`~torch_pricer.market.svi.SVISurface.arbitrage_report` say why -- not to
  return a ``nan`` price three layers downstream.
"""

from __future__ import annotations

import torch
from torch import Tensor

from torch_pricer.calibration.inputs import CalibrationInputs
from torch_pricer.calibration.result import CalibrationResult
from torch_pricer.errors import ValidationError
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.svi import SVISurface
from torch_pricer.models.base import Model
from torch_pricer.simulator.monte_carlo.simulator import SDE
from torch_pricer.tensors import EPS, as_tensor


class LocalVolSDE(SDE):
    """``d log S = (r - q - 0.5 sigma_LV^2) dt + sigma_LV dW``, ``sigma_LV`` from Dupire."""

    n_factors = 1

    def __init__(
        self,
        surface: SVISurface,
        forward,
        mut,
        vol_floor: float = 1e-3,
        vol_cap: float = 5.0,
        t_floor: float = 1e-3,
    ):
        self.surface = surface
        self.forward = forward
        self.mut = mut
        self.vol_floor = float(vol_floor)
        self.vol_cap = float(vol_cap)
        self.t_floor = float(t_floor)

    def local_variance(self, spot: Tensor, t) -> Tensor:
        """``sigma_LV^2(S, t)``, differentiable in the surface's parameters."""
        t = as_tensor(t)
        t = torch.clamp(t, min=self.t_floor)
        k = torch.log(spot.clamp_min(EPS) / self.forward(t))

        w = self.surface.total_variance(k, t).clamp_min(EPS)
        wk = self.surface.dw_dk(k, t)
        wkk = self.surface.d2w_dk2(k, t)
        wt = self.surface.dw_dT(k, t)

        denom = (
            1.0
            - (k / w) * wk
            + 0.25 * (-0.25 - 1.0 / w + (k * k) / (w * w)) * wk * wk
            + 0.5 * wkk
        )
        var = wt / denom.clamp_min(EPS)
        return var.clamp(self.vol_floor**2, self.vol_cap**2)

    def local_vol(self, spot: Tensor, t) -> Tensor:
        return torch.sqrt(self.local_variance(spot, t))

    def _var_at(self, xt: Tensor, t) -> Tensor:
        return self.local_variance(torch.exp(xt[..., 0]), t).unsqueeze(-1)

    def drift_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        return (self.mut(t) - 0.5 * self._var_at(xt, t)) * torch.ones_like(xt)

    def diffusion_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        return torch.sqrt(self._var_at(xt, t)).unsqueeze(-1)

    def coefficients(self, xt: Tensor, t: Tensor):
        """Drift and diffusion sharing one Dupire evaluation.

        Both coefficients are functions of the same local variance, and that
        expression -- a surface lookup and a dozen tensor ops on every path --
        is the whole cost of a local-vol step. Evaluating it once per step
        rather than twice halves both the runtime and the retained graph.
        """
        var = self._var_at(xt, t)
        drift = (self.mut(t) - 0.5 * var) * torch.ones_like(xt)
        return drift, torch.sqrt(var).unsqueeze(-1)

    def asset(self, x: Tensor) -> Tensor:
        return torch.exp(x[..., 0])


class LocalVolModel(Model):
    """A Dupire local vol driven by a calibrated :class:`SVISurface`."""

    n_factors = 1

    def __init__(
        self,
        surface: SVISurface,
        vol_floor: float = 1e-3,
        vol_cap: float = 5.0,
        t_floor: float = 1e-3,
    ):
        super().__init__()
        if not isinstance(surface, SVISurface):
            raise ValidationError(
                "LocalVolModel needs an SVISurface: Dupire wants analytic k-derivatives "
                f"of total variance, which {type(surface).__name__} does not provide"
            )
        self.surface = surface
        self.vol_floor, self.vol_cap, self.t_floor = vol_floor, vol_cap, t_floor

    def initial_state(self, market: MarketSnapshot) -> Tensor:
        spot = as_tensor(market.spot)
        return torch.log(spot).reshape(1)

    def to_sde(self, market: MarketSnapshot) -> SDE:
        return LocalVolSDE(
            surface=self.surface,
            # The forward of the market being priced, not the one the surface was
            # built with; they should agree, and this makes a disagreement show up
            # as a mispriced forward rather than a silently shifted smile.
            forward=market.forward,
            mut=lambda t: (
                market.discount.instantaneous_forward(t)
                - market.dividend.instantaneous_forward(t)
            ),
            vol_floor=self.vol_floor,
            vol_cap=self.vol_cap,
            t_floor=self.t_floor,
        )

    def calibrate(self, inputs: CalibrationInputs) -> CalibrationResult:
        from torch_pricer.calibration.svi_fit import fit_surface

        return fit_surface(self.surface, inputs)
