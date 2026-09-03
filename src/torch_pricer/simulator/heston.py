"""Heston dynamics: stochastic variance, correlated with the spot.

Two-dimensional state ``(log S, v)`` driven by two correlated Brownians:

    d log S = (r - q - v/2) dt + sqrt(v) dW1
    dv      = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt

The correlation lives in the ``(batch, 2, 2)`` diffusion matrix as a Cholesky
factor, which is what the matrix-valued diffusion in
:mod:`torch_pricer.simulator.simulator` exists for: the second row is
``xi sqrt(v) [rho, sqrt(1 - rho^2)]``, so the two rows have correlation ``rho``
while the driving normals stay independent.

**Euler is not exact here, and the variance is the reason.** ``sqrt(v)`` is not
Lipschitz at the origin, and a discrete step can push ``v`` below zero however
fine the grid, at which point the scheme has nowhere to go. This uses the *full
truncation* fix of Lord, Koekkoek and van Dijk: the state is allowed to go
negative but every use of it reads ``max(v, 0)``. Of the standard repairs
(absorption, reflection, partial truncation) it has the smallest discretisation
bias, and unlike the exact QE scheme it costs nothing and stays differentiable
-- which matters, because the pathwise delta is taken straight through these
steps. The bias is O(h) in the price and shows up mainly for parameters that
violate Feller badly; see :attr:`HestonProcess.feller_ratio`.

The log-spot component, in contrast, is integrated exactly given the variance
path, exactly as in :mod:`torch_pricer.simulator.gbm`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..tensors import as_tensor
from .simulator import SDE

if TYPE_CHECKING:  # pragma: no cover
    from ..calibration.curve_model.curves import RateCurve


class HestonProcess(SDE):
    """``(log S, v)`` under Heston, with a curve-driven drift.

    Every parameter is held by reference, not by value: the tensors handed in
    are the model's own, so ``autograd.grad`` against them reaches the
    calibrated parameters.

    ``vol_shift`` is what the engine takes vega against. There is no single
    "the volatility" in a stochastic vol model to differentiate with respect to,
    and the honest scalar a desk asks for is the parallel one: shift the
    instantaneous spot volatility by a constant at every time and state, and see
    what the price does. Sitting at zero it changes no price, and it is in the
    same units as Black vega -- per 1.00 of volatility.
    """

    dim = 2
    n_factors = 2

    def __init__(
        self,
        v0: Tensor,
        kappa: Tensor,
        theta: Tensor,
        xi: Tensor,
        rho: Tensor,
        discount: RateCurve,
        dividend: RateCurve,
        vol_shift: Tensor | None = None,
    ):
        super().__init__()
        self.v0 = as_tensor(v0)
        self.kappa = as_tensor(kappa)
        self.theta = as_tensor(theta)
        self.xi = as_tensor(xi)
        self.rho = as_tensor(rho)
        self.discount = discount
        self.dividend = dividend
        # A fresh graph leaf per SDE, so `autograd.grad(price, vol_shift)` has
        # something to differentiate against even though its value is zero.
        self.vol_shift = (
            as_tensor(vol_shift)
            if vol_shift is not None
            else torch.zeros_like(self.v0).requires_grad_(True)
        )

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / xi^2``. Below 1 the variance touches zero."""
        return float((2.0 * self.kappa * self.theta / self.xi**2).detach())

    def _variance(self, x: Tensor) -> Tensor:
        """The variance component, truncated at zero. Shape ``(batch, 1)``."""
        return x[:, 1:2].clamp_min(0.0)

    def spot_vol(self, x: Tensor, t: Tensor) -> Tensor:
        """Instantaneous volatility of the *spot*, shape ``(batch, 1)``.

        The one hook a local-stochastic model needs: scaling this by a leverage
        function is the whole of the difference between Heston and LSV, and it
        leaves the variance equation, the correlation and the scheme untouched.
        See :class:`~torch_pricer.simulator.lsv.LocalStochasticVol`.
        """
        return self._variance(x).sqrt() + self.vol_shift

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        v = self._variance(x)
        vol = self.spot_vol(x, t)
        r = self.discount.instantaneous_forward(t)
        q = self.dividend.instantaneous_forward(t)
        spot_drift = (r - q - 0.5 * vol**2).expand_as(v)
        return torch.cat([spot_drift, self.kappa * (self.theta - v)], dim=1)

    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        v = self._variance(x)
        vol = self.spot_vol(x, t)
        vol_of_var = self.xi * v.sqrt()
        zero = torch.zeros_like(vol)
        spot_row = torch.cat([vol, zero], dim=1)
        var_row = torch.cat([vol_of_var * self.rho, vol_of_var * (1.0 - self.rho**2).sqrt()], dim=1)
        return torch.stack([spot_row, var_row], dim=1)

    def initial_state(self, spot: Tensor) -> Tensor:
        return torch.stack([torch.log(spot).reshape(()), self.v0.reshape(())]).reshape(1, 2)

    def asset(self, x: Tensor) -> Tensor:
        return torch.exp(x[..., 0])

    def risk_parameters(self) -> dict[str, Tensor]:
        return {
            "vol": self.vol_shift,
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }
