"""Implied-vol surfaces: clean, arbitrage-free quoted vol as a function of strike and expiry.

A surface is a *description of the market's quotes*. It is not the dynamics --
that is a :class:`~torch_pricer.models.base.Model`, which is fitted to a surface
and knows how to produce an SDE. Local vol is exactly the map between the two,
which is why it deserves both a surface and a model.

Everything below works in **total implied variance** ``w(k, T) = sigma(k, T)^2
T`` over log-moneyness ``k = log(K / F(T))``, and not in implied vol over
strike. Three reasons, all of which matter downstream:

* Dupire's formula is a ratio of derivatives of ``w`` in exactly these
  coordinates, and writing it any other way buys a page of chain rule;
* the no-arbitrage conditions are clean here -- calendar arbitrage is
  ``dw/dT >= 0`` at fixed ``k``, butterfly arbitrage is one inequality in ``w``
  and its two ``k``-derivatives;
* interpolating linearly in ``w`` between expiries preserves the first of those
  automatically, while interpolating in vol does not.

Surface parameters are held as tensors, never coerced with ``float()``: a
surface is where bucketed vega has to come from, and a float would cut it out
of the graph before the question could be asked.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.tensors import as_tensor


class VolSurface(nn.Module, ABC):
    """Quoted implied vol as a function of strike and expiry."""

    @abstractmethod
    def vol(self, strike, expiry) -> Tensor:
        """Implied vol at ``strike`` for ``expiry`` years, broadcast over ``strike``."""

    @property
    @abstractmethod
    def reference_date(self) -> dt.date:
        """The date the quotes were observed."""


class FlatVolSurface(VolSurface):
    """One number everywhere: the Black-Scholes textbook surface."""

    def __init__(self, vol: float, as_of: dt.date):
        super().__init__()
        self.level = nn.Parameter(as_tensor(float(vol)))
        self._ref = as_of

    def vol(self, strike, expiry) -> Tensor:
        # Broadcast against the strike so callers get the shape they passed in,
        # without letting the strike's *value* into the result.
        strike = as_tensor(strike, dtype=self.level.dtype, device=self.level.device)
        return self.level.expand(strike.shape) if strike.dim() else self.level

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    def __repr__(self) -> str:  # pragma: no cover
        return f"FlatVolSurface({float(self.level.detach()):.4f}, {self._ref})"


class SVISurface(VolSurface):
    """Gatheral SVI, one slice per expiry. Not implemented yet.

    When it lands, the slice parameters ``(a, b, rho, m, sigma)`` must be
    ``nn.Parameter`` s: they are what bucketed vega differentiates against
    before being chained back to quote sensitivities through the slice fit.
    """

    def __init__(self, as_of: dt.date):
        super().__init__()
        self._ref = as_of

    def vol(self, strike, expiry) -> Tensor:
        raise NotImplementedError("SVISurface.vol")

    def total_variance(self, k, expiry) -> Tensor:
        raise NotImplementedError("SVISurface.total_variance")

    @property
    def reference_date(self) -> dt.date:
        return self._ref
