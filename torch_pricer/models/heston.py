"""Heston stochastic volatility, simulated with full truncation.

    d log S = (r - q - v/2) dt + sqrt(v) dW1
    dv      = kappa (theta - v) dt + xi sqrt(v) dW2,     d<W1, W2> = rho dt

Two factors, correlated through the Cholesky factor of the diffusion matrix --
which is why :class:`~torch_pricer.simulator.monte_carlo.simulator.SDE` returns a matrix
rather than an elementwise vector.

**Read this before trusting a pathwise greek here.** The Euler scheme cannot
keep ``v`` positive, and the standard repair is full truncation: use ``max(v, 0)``
in both coefficients while letting the state itself go negative. Whenever a path
takes ``v`` below zero that ``max`` is a kink *in the parameters*, so pathwise
sensitivities have no theoretical guarantee there.

Measured against :mod:`torch_pricer.pricer.analytic.heston`, 100k paths x 200
steps, 40 seeds, 1y ATM call -- the effect is on **variance**, not bias:

    Feller satisfied (2*kappa*theta - xi^2 = +0.23)
        every parameter within 2.3% of analytic, all |t| <= 0.6, se(v0) = 0.023

    Feller violated  (2*kappa*theta - xi^2 = -0.13)
        differences of 3-32%, but all |t| <= 1.0 -- not significant --
        while se(v0) = 1.51, a 67x jump; se(kappa) jumps 139x

So a bias, if any, is smaller than this experiment resolves. What is
unambiguous is that violating Feller inflates the estimator's variance by one to
two orders of magnitude, which makes the greeks useless at ordinary sample sizes
long before bias becomes the problem. Market fits routinely violate Feller.

If you need parameter greeks in that regime, the options are a QE (Andersen)
scheme, a smoothed truncation, or bumping -- and whichever you pick, check it
against the characteristic-function price rather than against intuition.
:func:`~torch_pricer.pricer.analytic.heston.feller` reports the margin. Delta is
unaffected; it does not pass through the truncation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.calibration.inputs import CalibrationInputs
from torch_pricer.errors import ValidationError
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.base import Model
from torch_pricer.simulator.monte_carlo.simulator import SDE
from torch_pricer.tensors import EPS, as_tensor

HESTON_PARAMS = ("v0", "kappa", "theta", "xi", "rho")


class HestonSDE(SDE):
    """State ``(log S, v)``, driven by two independent normals."""

    n_factors = 2

    def __init__(self, mut, v0, kappa, theta, xi, rho):
        self.mut = mut
        self.v0, self.kappa, self.theta, self.xi, self.rho = v0, kappa, theta, xi, rho

    def _v_plus(self, xt: Tensor) -> Tensor:
        """Full truncation: the coefficients see ``max(v, 0)``, the state does not."""
        return xt[..., 1].clamp_min(0.0)

    def drift_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        v = self._v_plus(xt)
        return torch.stack(
            [self.mut(t).expand_as(v) - 0.5 * v, self.kappa * (self.theta - v)], dim=-1
        )

    def diffusion_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        """Cholesky factor of the covariance, shape ``(n_paths, 2, 2)``."""
        root_v = torch.sqrt(self._v_plus(xt) + EPS)
        zero = torch.zeros_like(root_v)
        perp = torch.sqrt((1.0 - self.rho**2).clamp_min(EPS))
        return torch.stack(
            [
                torch.stack([root_v, zero], dim=-1),
                torch.stack([self.xi * root_v * self.rho, self.xi * root_v * perp], dim=-1),
            ],
            dim=-2,
        )

    def coefficients(self, xt: Tensor, t: Tensor):
        """Both coefficients from one truncation and one square root."""
        v = self._v_plus(xt)
        root_v = torch.sqrt(v + EPS)
        zero = torch.zeros_like(root_v)
        perp = torch.sqrt((1.0 - self.rho**2).clamp_min(EPS))
        drift = torch.stack(
            [self.mut(t).expand_as(v) - 0.5 * v, self.kappa * (self.theta - v)], dim=-1
        )
        diffusion = torch.stack(
            [
                torch.stack([root_v, zero], dim=-1),
                torch.stack([self.xi * root_v * self.rho, self.xi * root_v * perp], dim=-1),
            ],
            dim=-2,
        )
        return drift, diffusion

    def asset(self, x: Tensor) -> Tensor:
        return torch.exp(x[..., 0])


class HestonModel(Model):
    """Heston, with all five parameters differentiable."""

    n_factors = 2

    def __init__(self, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7):
        super().__init__()
        if min(v0, kappa, theta, xi) <= 0:
            raise ValidationError(
                f"v0, kappa, theta and xi must be positive, got "
                f"v0={v0}, kappa={kappa}, theta={theta}, xi={xi}"
            )
        if not -1.0 < rho < 1.0:
            raise ValidationError(f"rho must lie in (-1, 1), got {rho}")
        for name, value in zip(HESTON_PARAMS, (v0, kappa, theta, xi, rho)):
            setattr(self, name, nn.Parameter(as_tensor(float(value))))

    @property
    def feller(self) -> float:
        """``2 kappa theta - xi^2``. Negative means ``v`` reaches zero routinely."""
        return float(
            (2.0 * self.kappa * self.theta - self.xi**2).detach()
        )

    def initial_state(self, market: MarketSnapshot) -> Tensor:
        spot = as_tensor(market.spot, dtype=self.v0.dtype, device=self.v0.device)
        return torch.stack([torch.log(spot).reshape(()), self.v0.reshape(())])

    def to_sde(self, market: MarketSnapshot) -> SDE:
        return HestonSDE(
            mut=lambda t: (
                market.discount.instantaneous_forward(t)
                - market.dividend.instantaneous_forward(t)
            ),
            v0=self.v0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho,
        )

    def calibrate(self, inputs: CalibrationInputs) -> None:
        raise NotImplementedError(
            "HestonModel.calibrate: fit against analytic.heston.heston_price and "
            "return a CalibrationResult carrying the Jacobian and Hessian"
        )
