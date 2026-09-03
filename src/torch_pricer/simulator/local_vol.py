"""Local volatility dynamics.

    dS = (r(t) - q(t)) S dt + sigma_loc(S, t) S dW

integrated in log space like :mod:`torch_pricer.simulator.gbm`, with
``sigma_loc`` read off a calibrated grid rather than a constant. The state
dependence in the diffusion means an Euler step is no longer the exact
transition law -- unlike GBM, the number of steps now matters.

**The grid is indexed by log-moneyness against a fixed reference forward, and
that choice is a modelling statement, not a convenience.** Dupire's local
volatility is a function of the absolute level of the asset and of time. If the
coordinate were moneyness against a *live* forward, bumping the spot to take a
delta would drag the whole surface along with it, and what came back would be
the sticky-delta delta of a different model. Holding the reference forward fixed
-- detached from the spot the engine differentiates -- keeps
``sigma_loc(S, t)`` a function of the level, so the pathwise delta is the local
vol delta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

from ..errors import ValidationError
from ..tensors import as_tensor
from .simulator import SDE

if TYPE_CHECKING:  # pragma: no cover
    from ..calibration.curve_model.curves import RateCurve


class LevelSurface:
    """Values on a ``(time, log-moneyness)`` grid, evaluated at an asset level.

    Bilinear, flat outside the grid in both coordinates, and differentiable in
    both the grid values and the query: the first is what lets a leverage
    function be calibrated by an optimiser, the second is what lets delta pass
    through ``sigma_loc(S, t)`` rather than around it.

    Holds its tensors by reference -- the values are typically an
    ``nn.Parameter`` owned by the model that built this.
    """

    def __init__(
        self,
        times: Tensor,
        log_moneyness: Tensor,
        values: Tensor,
        reference_forward: Callable[[Tensor], Tensor],
    ):
        times = as_tensor(times).flatten()
        log_moneyness = as_tensor(log_moneyness).flatten()
        if values.shape != (times.numel(), log_moneyness.numel()):
            raise ValidationError(
                f"grid values must be (n_times, n_moneyness) = "
                f"({times.numel()}, {log_moneyness.numel()}), got {tuple(values.shape)}"
            )
        if log_moneyness.numel() < 2:
            raise ValidationError("the moneyness axis needs at least two points to interpolate")
        self.times = times
        self.log_moneyness = log_moneyness
        self.values = values
        self.reference_forward = reference_forward

    def row(self, t: Tensor) -> Tensor:
        """The grid's values at time ``t``, shape ``(n_moneyness,)``."""
        tp = self.times
        if tp.numel() == 1:
            return self.values[0]
        clamped = as_tensor(t).reshape(1).clamp(float(tp[0]), float(tp[-1]))
        i = torch.searchsorted(tp, clamped.detach().contiguous()).clamp(1, tp.numel() - 1)
        t0, t1 = tp[i - 1], tp[i]
        a = (clamped - t0) / (t1 - t0)
        return (1.0 - a) * self.values[i - 1].squeeze(0) + a * self.values[i].squeeze(0)

    def __call__(self, asset: Tensor, t: Tensor) -> Tensor:
        """Interpolated value at asset level ``asset`` and time ``t``.

        Args:
            asset: ``(batch, 1)`` asset levels
            t: current time, scalar

        Returns:
            ``(batch, 1)``
        """
        row = self.row(t)
        k = torch.log(asset / self.reference_forward(as_tensor(t)))
        kp = self.log_moneyness
        clamped = k.clamp(float(kp[0]), float(kp[-1]))
        j = torch.searchsorted(kp, clamped.detach().flatten().contiguous()).clamp(1, kp.numel() - 1)
        j = j.reshape(clamped.shape)
        k0, k1 = kp[j - 1], kp[j]
        v0, v1 = row[j - 1], row[j]
        return v0 + (v1 - v0) * (clamped - k0) / (k1 - k0)


class LocalVolatility(SDE):
    """Log-spot with a level- and time-dependent volatility.

    ``vol_shift`` is the parallel bump the engine takes vega against, for the
    same reason as in :class:`~torch_pricer.simulator.heston.HestonProcess`:
    the model's parameters are a whole surface, and the scalar a desk quotes is
    what a parallel shift of it is worth.
    """

    dim = 1
    n_factors = 1

    def __init__(
        self,
        surface: LevelSurface,
        discount: RateCurve,
        dividend: RateCurve,
        vol_shift: Tensor | None = None,
    ):
        super().__init__()
        self.surface = surface
        self.discount = discount
        self.dividend = dividend
        self.vol_shift = (
            as_tensor(vol_shift)
            if vol_shift is not None
            else torch.zeros_like(surface.values.reshape(-1)[0]).requires_grad_(True)
        )

    def local_vol(self, x: Tensor, t: Tensor) -> Tensor:
        """``sigma_loc(S, t)`` at the state's asset level, shape ``(batch, 1)``."""
        return (self.surface(torch.exp(x[:, :1]), t) + self.vol_shift).clamp_min(0.0)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        vol = self.local_vol(x, t)
        r = self.discount.instantaneous_forward(t)
        q = self.dividend.instantaneous_forward(t)
        return (r - q) - 0.5 * vol**2

    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        return self.local_vol(x, t).reshape(-1, 1, 1)

    def initial_state(self, spot: Tensor) -> Tensor:
        return torch.log(spot).reshape(1, 1)

    def asset(self, x: Tensor) -> Tensor:
        return torch.exp(x[..., 0])

    def risk_parameters(self) -> dict[str, Tensor]:
        return {"vol": self.vol_shift}
