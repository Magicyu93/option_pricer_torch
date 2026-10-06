"""Surface SVI (Gatheral & Jacquier, 2014), with the power-law ``phi``."""

from __future__ import annotations

import datetime as dt

import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.errors import ValidationError
from torch_pricer.market.surface.base import VolSurface
from torch_pricer.tensors import interp_linear


def _inverse_softplus(x: Tensor) -> Tensor:
    return x + torch.log(-torch.expm1(-x))


def _logit(p: Tensor) -> Tensor:
    return torch.log(p) - torch.log1p(-p)


class SSVISurface(VolSurface):
    """Total variance at log-moneyness ``k``::

        w(k, T) = theta/2 * (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))
        phi     = eta / (theta^gamma (1 + theta)^(1 - gamma)),   theta = theta(T)

    ``theta(T)`` is the ATM total variance: one value per fitted expiry,
    linear in ``T`` between them, ``theta_1 T / T_1`` before the first and
    ``theta_n T / T_n`` (constant ATM vol) after the last.

    Free of static arbitrage by construction, not by checking: the raw
    parameters are mapped so that

    * ``|rho| < 1``, ``0 < gamma <= 1/2`` and ``eta (1 + |rho|) < 2`` -- the
      sufficient no-butterfly condition for this ``phi``;
    * ``theta`` increases with ``T`` -- with the above, no calendar arbitrage.

    So an optimizer can move the raw parameters anywhere and every surface it
    visits is arbitrage-free. All of them are ``nn.Parameter`` s, for vega by
    autograd.
    """

    def __init__(
        self,
        expiry_times,
        theta,
        as_of: dt.date,
        rho: float = -0.7,
        eta: float = 1.0,
        gamma: float = 0.4,
    ):
        super().__init__()
        times = torch.as_tensor(expiry_times, dtype=torch.float64).flatten()
        theta = torch.as_tensor(theta, dtype=torch.float64).flatten()
        if times.numel() == 0 or times.numel() != theta.numel():
            raise ValidationError("need one ATM total variance per expiry, and at least one")
        if not bool((times[1:] > times[:-1]).all()) or not bool((times > 0).all()):
            raise ValidationError("expiry times must be positive and strictly increasing")
        if not bool((theta[1:] > theta[:-1]).all()) or not bool((theta > 0).all()):
            raise ValidationError("ATM total variance must be positive and strictly increasing")
        if not (abs(rho) < 1 and 0 < gamma <= 0.5 and 0 < eta * (1 + abs(rho)) < 2):
            raise ValidationError("initial (rho, eta, gamma) violate the no-arbitrage bounds")

        self.register_buffer("expiry_times", times)
        increments = torch.diff(theta, prepend=theta.new_zeros(1))
        self.raw_theta = nn.Parameter(_inverse_softplus(increments))
        self.raw_rho = nn.Parameter(torch.atanh(torch.tensor(rho, dtype=torch.float64)))
        self.raw_gamma = nn.Parameter(_logit(torch.tensor(2 * gamma, dtype=torch.float64)))
        self.raw_eta = nn.Parameter(
            _logit(torch.tensor(eta * (1 + abs(rho)) / 2, dtype=torch.float64))
        )
        self._ref = as_of

    @classmethod
    def initial(cls, expiry_times, atm_total_variance, as_of: dt.date) -> SSVISurface:
        """The observed ATM term structure, and the default smile shape."""
        return cls(expiry_times, atm_total_variance, as_of)

    # -- parameters, constrained ------------------------------------------
    @property
    def rho(self) -> Tensor:
        return torch.tanh(self.raw_rho)

    @property
    def gamma(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.raw_gamma)

    @property
    def eta(self) -> Tensor:
        return 2.0 / (1.0 + self.rho.abs()) * torch.sigmoid(self.raw_eta)

    @property
    def theta(self) -> Tensor:
        """ATM total variance at each expiry: cumulative, so increasing."""
        return torch.cumsum(nn.functional.softplus(self.raw_theta), 0)

    def constraints(self) -> dict[str, dict[str, float]]:
        rho, eta, gamma = (float(x.detach()) for x in (self.rho, self.eta, self.gamma))
        return {
            "|rho| < 1": {"value": abs(rho), "bound": 1.0},
            "gamma <= 1/2": {"value": gamma, "bound": 0.5},
            "eta (1 + |rho|) < 2": {"value": eta * (1 + abs(rho)), "bound": 2.0},
            "theta increasing": {
                "value": float(torch.diff(self.theta).min().detach()) if self.theta.numel() > 1 else 0.0,
                "bound": 0.0,  # smallest step between expiries; must stay above 0
            },
        }

    def describe(self) -> dict[str, float | list[float]]:
        return {
            "rho": float(self.rho.detach()),
            "eta": float(self.eta.detach()),
            "gamma": float(self.gamma.detach()),
            "theta": self.theta.detach().tolist(),
        }

    # -- the surface ----------------------------------------------------------
    def atm_variance(self, expiry) -> Tensor:
        """``theta(T)``: linear between expiries, constant ATM vol outside them."""
        t = self._as_tensor(expiry)
        times, theta = self.expiry_times, self.theta
        if times.numel() == 1:
            return theta[0] * t / times[0]
        clamped = t.clamp(float(times[0]), float(times[-1]))
        inside = interp_linear(clamped, times, theta)
        return torch.where(
            t < times[0], theta[0] * t / times[0],
            torch.where(t > times[-1], theta[-1] * t / times[-1], inside),
        )

    def total_variance(self, k, expiry) -> Tensor:
        theta = self.atm_variance(expiry)
        k = self._as_tensor(k)
        rho, gamma = self.rho, self.gamma
        phi = self.eta / (theta.pow(gamma) * (1 + theta).pow(1 - gamma))
        return 0.5 * theta * (1 + rho * phi * k + torch.sqrt((phi * k + rho) ** 2 + 1 - rho**2))

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SSVISurface({self.expiry_times.numel()} expiries, rho={float(self.rho.detach()):.3f}, "
            f"eta={float(self.eta.detach()):.3f}, gamma={float(self.gamma.detach()):.3f})"
        )
