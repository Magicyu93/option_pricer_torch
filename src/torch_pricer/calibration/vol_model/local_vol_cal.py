"""Local volatility calibration: Dupire.

Dupire's result is that for *any* arbitrage-free surface of European prices
there is exactly one one-factor diffusion that reproduces it, and its diffusion
coefficient is a ratio of derivatives of the observed prices. So this
"calibration" is not an optimisation at all -- there is nothing to fit, and no
objective to minimise. Given the surface, the answer is a formula, and the only
work is evaluating it stably.

In total implied variance over log-moneyness, ``w(k, T) = sigma_imp^2 T``,
``k = log(K / F(T))``, Gatheral's form of it is

    sigma_loc^2 = dw/dT / [ 1 - k/w dw/dk
                            + 1/4 (-1/4 - 1/w + k^2/w^2) (dw/dk)^2
                            + 1/2 d2w/dk2 ]

which is what is coded below. Stated in these coordinates the formula contains
no rates at all: the forward carries them, and the numerator is a derivative at
fixed moneyness rather than at fixed strike.

**The derivatives are taken by autograd, not by finite differences.** Dupire is
the standard cautionary tale about differencing: the denominator is a
second derivative of quoted data, the numerator a first, and a bump size that
suits one expiry produces noise at another. Differentiating the fitted surface
exactly removes that entire class of error -- and it is why
:class:`~torch_pricer.calibration.surface.SVISurface` is worth having, since a
formula that differentiates its input twice needs an input that is smooth twice.

The denominator is the implied density up to a positive factor. It goes
non-positive exactly when the surface admits butterfly arbitrage, so a negative
local variance is a diagnosis of the input, not of the code; see
:meth:`~torch_pricer.calibration.surface.SVISlice.butterfly_margin`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from torch import Tensor

from ...errors import CalibrationError, ValidationError
from ...simulator.local_vol import LevelSurface, LocalVolatility
from ...tensors import EPS, as_tensor
from ..inputs import CalibrationInputs
from ..surface import VolSurface
from .base import VolModel

if TYPE_CHECKING:  # pragma: no cover
    from ...market.snapshot import MarketSnapshot
    from ...simulator.simulator import SDE

#: Local vol is floored and capped rather than left to the formula. A surface is
#: only as smooth as its fit, and a denominator that grazes zero in a far wing
#: would otherwise put an infinite volatility into a simulation that then has no
#: way to recover. The band is wide enough to be inactive on any sane surface.
MIN_LOCAL_VOL = 0.01
MAX_LOCAL_VOL = 2.00


def dupire_local_variance(
    surface: VolSurface,
    forward: Callable[[Tensor], Tensor],
    times: Tensor,
    log_moneyness: Tensor,
    min_vol: float = MIN_LOCAL_VOL,
    max_vol: float = MAX_LOCAL_VOL,
) -> tuple[Tensor, Tensor]:
    """Dupire local variance on a ``(time, log-moneyness)`` grid.

    Args:
        surface: the implied surface, differentiable in strike and expiry
        forward: ``t -> F(t)``, the same forward the surface is quoted against
        times: expiries in years, strictly positive, ``(n_t,)``
        log_moneyness: ``k = log(K / F(T))``, increasing, ``(n_k,)``
        min_vol: floor applied to the resulting volatility
        max_vol: cap applied to the resulting volatility

    Returns:
        ``(local_variance, denominator)``, both ``(n_t, n_k)``. The denominator
        is returned because its sign is the arbitrage diagnostic; a caller that
        only wants the surface can drop it.
    """
    times = as_tensor(times).flatten()
    log_moneyness = as_tensor(log_moneyness).flatten()
    if bool((times <= 0).any()):
        raise ValidationError(
            "Dupire needs strictly positive expiries; t = 0 is a limit, not a point"
        )

    n_t, n_k = times.numel(), log_moneyness.numel()
    # Both coordinates are expanded to the full grid before they are made leaves:
    # `grad(w.sum(), k)` is then elementwise, because w[i, j] sees only k[i, j].
    k = log_moneyness.reshape(1, n_k).expand(n_t, n_k).clone().requires_grad_(True)
    t = times.reshape(n_t, 1).expand(n_t, n_k).clone().requires_grad_(True)

    with torch.enable_grad():
        strike = forward(t) * torch.exp(k)
        w = surface.total_variance(strike, t).clamp_min(EPS)
        (w_t,) = torch.autograd.grad(w.sum(), t, create_graph=True)
        (w_k,) = torch.autograd.grad(w.sum(), k, create_graph=True)
        (w_kk,) = torch.autograd.grad(w_k.sum(), k, retain_graph=False)

    w, w_t, w_k, w_kk = (x.detach() for x in (w, w_t, w_k, w_kk))
    k = k.detach()
    denominator = (
        1.0
        - k * w_k / w
        + 0.25 * (-0.25 - 1.0 / w + k**2 / w**2) * w_k**2
        + 0.5 * w_kk
    )
    # A non-positive denominator is butterfly arbitrage in the input surface. The
    # clamp keeps the grid finite; the caller is told by the returned value.
    variance = w_t / denominator.clamp_min(EPS)
    return variance.clamp(min_vol**2, max_vol**2), denominator


class LocalVolModel(VolModel):
    """Dupire local volatility, held as a grid of volatilities.

    The grid is an ``nn.Parameter``, which is not there so an optimiser can move
    it -- Dupire already determined it -- but so that it serialises with
    ``state_dict()`` and moves with ``.to(device)`` like every other calibrated
    object here. It is also what a local-stochastic model differentiates when it
    fits its leverage function on top.
    """

    def __init__(
        self,
        times: Tensor,
        log_moneyness: Tensor,
        local_vol: Tensor,
        reference_spot: Tensor,
    ):
        super().__init__()
        self._install(times, log_moneyness, local_vol, reference_spot)

    def _install(self, times, log_moneyness, local_vol, reference_spot) -> None:
        times = as_tensor(times).flatten()
        log_moneyness = as_tensor(log_moneyness).flatten()
        local_vol = as_tensor(local_vol)
        if local_vol.shape != (times.numel(), log_moneyness.numel()):
            raise ValidationError(
                f"local vol grid must be ({times.numel()}, {log_moneyness.numel()}), "
                f"got {tuple(local_vol.shape)}"
            )
        self.register_buffer("times", times)
        self.register_buffer("log_moneyness", log_moneyness)
        self.register_buffer("reference_spot", as_tensor(reference_spot).detach().reshape(()))
        self.local_vol = nn.Parameter(local_vol.detach().clone())

    # -- construction ---------------------------------------------------
    @classmethod
    def from_surface(
        cls,
        inputs: CalibrationInputs,
        times: Tensor | None = None,
        log_moneyness: Tensor | None = None,
    ) -> LocalVolModel:
        """Build the Dupire surface implied by ``inputs.surface``."""
        surface = inputs.surface
        if surface is None:
            raise CalibrationError(
                "local volatility is a transform of an implied surface; none given"
            )

        times = as_tensor(_default_times(surface) if times is None else times)
        if log_moneyness is None:
            log_moneyness = torch.linspace(-1.2, 1.2, 49, dtype=times.dtype)
        log_moneyness = as_tensor(log_moneyness)
        variance, _ = dupire_local_variance(surface, inputs.forward, times, log_moneyness)
        return cls(times, log_moneyness, variance.sqrt(), inputs.spot)

    def calibrate(self, inputs: CalibrationInputs) -> None:
        """Recompute the grid from ``inputs.surface``. Mutates in place."""
        fitted = self.from_surface(inputs, self.times, self.log_moneyness)
        self._install(fitted.times, fitted.log_moneyness, fitted.local_vol, fitted.reference_spot)

    # -- queries --------------------------------------------------------
    def reference_forward(self, discount, dividend) -> Callable[[Tensor], Tensor]:
        """``t -> F(t)`` at the *calibration* spot, which fixes the grid's coordinate."""
        spot = self.reference_spot

        def forward(t: Tensor) -> Tensor:
            t = as_tensor(t)
            return spot * dividend.discount(t) / discount.discount(t)

        return forward

    def level_surface(self, discount, dividend) -> LevelSurface:
        """The interpolator the SDE reads ``sigma_loc(S, t)`` from."""
        return LevelSurface(
            self.times,
            self.log_moneyness,
            self.local_vol,
            self.reference_forward(discount, dividend),
        )

    def to_sde(self, market: MarketSnapshot) -> SDE:
        return LocalVolatility(
            surface=self.level_surface(market.discount, market.dividend),
            discount=market.discount,
            dividend=market.dividend,
        )

    def extra_repr(self) -> str:  # pragma: no cover
        lo, hi = self.local_vol.detach().min(), self.local_vol.detach().max()
        return (
            f"{self.times.numel()}x{self.log_moneyness.numel()} grid, "
            f"vol in [{float(lo):.4f}, {float(hi):.4f}]"
        )


def _default_times(surface: VolSurface) -> Tensor:
    """A time axis dense at the front, where the surface moves fastest.

    Local vol is a derivative in ``T``, and the term structure of a real surface
    does most of its work in the first few months; a uniform grid spends its
    points where nothing is happening.
    """
    quoted = getattr(surface, "expiries", None)
    horizon = float(as_tensor(quoted).max()) if quoted is not None and len(quoted) else 2.0
    return torch.linspace(0.02**0.5, max(horizon, 0.05) ** 0.5, 25) ** 2
