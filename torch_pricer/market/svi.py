"""Gatheral SVI, one slice per expiry, differentiable in its parameters.

Raw SVI parameterises total implied variance over log-moneyness ``k`` as

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))

a hyperbola in ``k`` with asymptote slopes ``b(rho -/+ 1)``. Five parameters per
expiry buy a smile with level, slope, curvature, and both wings, and every one
of the no-arbitrage conditions is checkable in closed form.

The parameters are ``nn.Parameter`` s. That is the point: local vol is derived
from this surface by Dupire, so ``dV/d(slice parameter)`` is what an autograd
pass through a Dupire-driven simulation actually produces, and market vega is
that chained through the slice fit. A float here would end the chain before it
started.

The ``k``-derivatives are analytic rather than autograd'd. Dupire needs the
second derivative, and taking it by nested ``autograd.grad`` on every step of
every path builds a second-order graph over the whole simulation for a quantity
that is three lines of algebra:

    u = k - m,  q = sqrt(u^2 + s^2)
    w   = a + b (rho u + q)
    w'  = b (rho + u / q)
    w'' = b s^2 / q^3

Time interpolation is linear in total variance at fixed ``k``, which keeps
``dw/dT >= 0`` -- calendar-arbitrage-free -- whenever the slices themselves are
ordered. Below the first slice and above the last, ``w`` is held proportional to
``T``, i.e. implied vol is flat in time, which keeps ``dw/dT`` strictly positive
rather than zero (a zero would make Dupire's numerator vanish and the local vol
with it).
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.errors import ValidationError
from torch_pricer.market.surface import VolSurface
from torch_pricer.tensors import EPS, as_tensor

#: Parameter order used by every calibration and sensitivity in this module.
SVI_PARAMS = ("a", "b", "rho", "m", "sigma")


class SVISlice(nn.Module):
    """One expiry's smile, in total implied variance over log-moneyness."""

    def __init__(self, a, b, rho, m, sigma, expiry: float):
        super().__init__()
        self.a = nn.Parameter(as_tensor(float(a)))
        self.b = nn.Parameter(as_tensor(float(b)))
        self.rho = nn.Parameter(as_tensor(float(rho)))
        self.m = nn.Parameter(as_tensor(float(m)))
        self.sigma = nn.Parameter(as_tensor(float(sigma)))
        if expiry <= 0:
            raise ValidationError(f"slice expiry must be positive, got {expiry}")
        self.register_buffer("expiry", as_tensor(float(expiry)))

    def _uq(self, k: Tensor) -> tuple[Tensor, Tensor]:
        u = k - self.m
        return u, torch.sqrt(u * u + self.sigma**2)

    def total_variance(self, k: Tensor) -> Tensor:
        """``w(k)``, total implied variance."""
        u, q = self._uq(k)
        return self.a + self.b * (self.rho * u + q)

    def d_dk(self, k: Tensor) -> Tensor:
        """``dw/dk``."""
        u, q = self._uq(k)
        return self.b * (self.rho + u / q)

    def d2_dk2(self, k: Tensor) -> Tensor:
        """``d2w/dk2``. Strictly positive for ``b, sigma > 0``, so the smile is convex."""
        _, q = self._uq(k)
        return self.b * self.sigma**2 / q**3

    def vector(self) -> Tensor:
        return torch.stack([self.a, self.b, self.rho, self.m, self.sigma])

    def butterfly_g(self, k: Tensor) -> Tensor:
        """Durrleman's function. Negative anywhere means butterfly arbitrage."""
        w, wp, wpp = self.total_variance(k), self.d_dk(k), self.d2_dk2(k)
        w = w.clamp_min(EPS)
        return (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2

    def extra_repr(self) -> str:  # pragma: no cover
        vals = ", ".join(
            f"{n}={float(getattr(self, n).detach()):.4g}"
            for n in ("a", "b", "rho", "m", "sigma")
        )
        return f"T={float(self.expiry):.4g}, {vals}"


class SVISurface(VolSurface):
    """A term structure of :class:`SVISlice` s, interpolated in total variance.

    Args:
        slices: one per expiry; sorted on construction
        forward: ``t -> F(t)``, needed to turn a strike into log-moneyness
        as_of: the observation date

    ``k`` may have any shape; ``expiry`` must be a scalar. That is what the
    local-vol simulator asks for -- one time step, every path at once -- and
    supporting a ragged time axis would mean a per-element bracket search for no
    caller that needs it.
    """

    def __init__(
        self,
        slices: Sequence[SVISlice],
        forward: Callable[[Tensor], Tensor],
        as_of: dt.date,
    ):
        super().__init__()
        if not slices:
            raise ValidationError("an SVI surface needs at least one slice")
        ordered = sorted(slices, key=lambda s: float(s.expiry))
        expiries = [float(s.expiry) for s in ordered]
        if len(set(expiries)) != len(expiries):
            raise ValidationError(f"duplicate slice expiries: {expiries}")
        self.slices = nn.ModuleList(ordered)
        self.register_buffer("expiries", as_tensor(expiries))
        self.forward = forward
        self._ref = as_of

    # -- term structure -------------------------------------------------
    def _bracket(self, t: float) -> tuple[int, int, float]:
        """``(i, j, lambda)`` with ``w = (1 - lambda) w_i + lambda w_j``."""
        ts = [float(x) for x in self.expiries]
        if t <= ts[0] or len(ts) == 1:
            return 0, 0, 0.0
        if t >= ts[-1]:
            return len(ts) - 1, len(ts) - 1, 0.0
        j = next(i for i, x in enumerate(ts) if x >= t)
        return j - 1, j, (t - ts[j - 1]) / (ts[j] - ts[j - 1])

    def _blend(self, method: str, k: Tensor, expiry) -> Tensor:
        """Interpolate ``method`` between the bracketing slices.

        Only the bracket *index* is taken from a detached ``float(t)``; the
        weight is a tensor, so an expiry the caller intends to differentiate
        against -- theta -- survives. Detaching the whole of ``t`` here would cut
        it silently, exactly as ``torch.linspace`` does with a tensor endpoint.
        """
        t = as_tensor(expiry)
        if float(t.detach()) <= 0:
            raise ValidationError(f"expiry must be positive, got {float(t.detach())}")
        i, j, _ = self._bracket(float(t.detach()))
        fi = getattr(self.slices[i], method)(k)
        if i == j:
            # Outside the quoted range, hold implied vol flat in time: w scales
            # linearly with T, which keeps dw/dT strictly positive.
            return fi * (t / self.expiries[i])
        fj = getattr(self.slices[j], method)(k)
        lam = (t - self.expiries[i]) / (self.expiries[j] - self.expiries[i])
        return (1.0 - lam) * fi + lam * fj

    def total_variance(self, k, expiry) -> Tensor:
        """``w(k, T)``."""
        return self._blend("total_variance", as_tensor(k), expiry)

    def dw_dk(self, k, expiry) -> Tensor:
        return self._blend("d_dk", as_tensor(k), expiry)

    def d2w_dk2(self, k, expiry) -> Tensor:
        return self._blend("d2_dk2", as_tensor(k), expiry)

    def dw_dT(self, k, expiry) -> Tensor:
        """``dw/dT`` at fixed ``k``. Piecewise constant, and strictly positive."""
        k = as_tensor(k)
        t = as_tensor(expiry)
        i, j, _ = self._bracket(float(t.detach()))
        if i == j:
            return self.slices[i].total_variance(k) / self.expiries[i]
        wi = self.slices[i].total_variance(k)
        wj = self.slices[j].total_variance(k)
        return (wj - wi) / (self.expiries[j] - self.expiries[i])

    # -- VolSurface interface -------------------------------------------
    def log_moneyness(self, strike, expiry) -> Tensor:
        """``k = log(K / F(T))``."""
        strike = as_tensor(strike)
        return torch.log(strike.clamp_min(EPS) / self.forward(as_tensor(expiry)))

    def vol(self, strike, expiry) -> Tensor:
        """Implied vol at ``strike`` for ``expiry`` years."""
        k = self.log_moneyness(strike, expiry)
        w = self.total_variance(k, expiry).clamp_min(EPS)
        return torch.sqrt(w / as_tensor(expiry))

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    # -- diagnostics ----------------------------------------------------
    def arbitrage_report(self, k_range: float = 1.5, n: int = 201) -> dict[str, float]:
        """Worst butterfly and calendar violations over a band of log-moneyness.

        Both numbers should be ``>= 0``. Negative means the surface admits an
        arbitrage, and any local vol derived from it will have a negative
        variance somewhere.
        """
        k = torch.linspace(-k_range, k_range, n, dtype=self.expiries.dtype)
        butterfly = min(float(s.butterfly_g(k).min().detach()) for s in self.slices)
        calendar = float("inf")
        for i in range(len(self.slices) - 1):
            gap = self.slices[i + 1].total_variance(k) - self.slices[i].total_variance(k)
            calendar = min(calendar, float(gap.min().detach()))
        return {
            "butterfly": butterfly,
            "calendar": calendar if len(self.slices) > 1 else float("inf"),
        }
