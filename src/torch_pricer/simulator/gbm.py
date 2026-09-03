"""Geometric Brownian motion, integrated in log space.

Under the risk-neutral measure the spot follows

    dS = (r(t) - q(t)) S dt + v S dW

and applying Ito to ``x = log S`` removes the state dependence entirely:

    dx = (r(t) - q(t) - v^2/2) dt + v dW.

That matters for more than tidiness. With no ``x`` on the right-hand side, an
Euler step over ``[t, t+h]`` is the *exact* transition law rather than a first
order approximation, so a Monte Carlo price converges to Black with sampling
error and nothing else. Simulating ``S`` directly would add a discretisation
bias that has to be tuned away with step count before any benchmark means
anything -- and would let a path go negative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..tensors import as_tensor
from .simulator import SDE

if TYPE_CHECKING:  # pragma: no cover
    from ..calibration.curve_model.curves import RateCurve


class GeometricBrownianMotion(SDE):
    """Lognormal spot with a constant vol and curve-driven drift.

    The vol is whatever tensor the model handed over -- typically a derived
    quantity like ``exp(log_vol)`` sitting on an ``nn.Parameter`` -- and it is
    held by reference so that ``autograd.grad(price, sde.risk_parameters()["vol"])``
    reaches the model's parameter.
    """

    dim = 1
    n_factors = 1

    def __init__(self, vol: Tensor, discount: RateCurve, dividend: RateCurve):
        super().__init__()
        self.vol = as_tensor(vol)
        self.discount = discount
        self.dividend = dividend

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        r = self.discount.instantaneous_forward(t)
        q = self.dividend.instantaneous_forward(t)
        return (r - q - 0.5 * self.vol**2).reshape(1, 1).expand_as(x)

    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        return self.vol.reshape(1, 1, 1).expand(x.shape[0], self.dim, self.n_factors)

    def initial_state(self, spot: Tensor) -> Tensor:
        return torch.log(spot).reshape(1, 1)

    def asset(self, x: Tensor) -> Tensor:
        return torch.exp(x[..., 0])

    def risk_parameters(self) -> dict[str, Tensor]:
        return {"vol": self.vol}
