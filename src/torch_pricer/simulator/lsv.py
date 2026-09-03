"""Local-stochastic volatility dynamics.

Heston's variance scaled by a leverage function ``L(S, t)``:

    d log S = (r - q - L^2 v / 2) dt + L(S, t) sqrt(v) dW1
    dv      = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt

This is the model that ends the argument between the other two. Local vol
reprices every vanilla exactly and gets the forward smile wrong -- its future
smile flattens, so it underprices anything that pays off on the smile's own
dynamics. Heston has a plausible forward smile and cannot fit today's surface
across all expiries at once. LSV keeps Heston's dynamics for the *shape* of the
future smile and uses the leverage function to force today's surface to be
reproduced exactly.

Structurally it is one line of difference from
:class:`~torch_pricer.simulator.heston.HestonProcess`: the same state, the same
correlation, the same full-truncation scheme, with the spot's volatility
multiplied by the leverage. Everything hard about LSV lives in the calibration
of ``L``, in :mod:`torch_pricer.calibration.vol_model.lsv_cal`, because the
condition it must satisfy involves an expectation over the model it is a
parameter of.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .heston import HestonProcess
from .local_vol import LevelSurface

if TYPE_CHECKING:  # pragma: no cover
    from ..calibration.curve_model.curves import RateCurve


class LocalStochasticVol(HestonProcess):
    """Heston, with the spot's volatility scaled by a calibrated leverage surface."""

    def __init__(
        self,
        leverage: LevelSurface,
        v0: Tensor,
        kappa: Tensor,
        theta: Tensor,
        xi: Tensor,
        rho: Tensor,
        discount: RateCurve,
        dividend: RateCurve,
        vol_shift: Tensor | None = None,
    ):
        super().__init__(
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            discount=discount,
            dividend=dividend,
            vol_shift=vol_shift,
        )
        self.leverage = leverage

    def spot_vol(self, x: Tensor, t: Tensor) -> Tensor:
        leverage = self.leverage(torch.exp(x[:, :1]), t)
        return (leverage * self._variance(x).sqrt() + self.vol_shift).clamp_min(0.0)
