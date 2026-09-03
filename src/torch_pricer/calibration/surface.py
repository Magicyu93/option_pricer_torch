"""Implied-vol surfaces: clean, arbitrage-free quoted vol as a function of strike and expiry.

A surface is a *description of the market's quotes*. It is not the dynamics --
that is a :class:`~torch_pricer.calibration.vol_model.base.VolModel`, which is
fitted to a surface and knows how to produce an SDE. Local vol is exactly the
map between the two, which is why it deserves both a surface and a model.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod

from torch import Tensor

from ..tensors import as_tensor


class VolSurface(ABC):
    """Implied volatility as a function of strike and expiry."""

    @abstractmethod
    def vol(self, strike, expiry) -> Tensor:
        """Implied vol at ``strike`` for ``expiry`` years out."""

    @property
    @abstractmethod
    def reference_date(self) -> dt.date:
        """The date the surface's expiries are measured from."""


class FlatVolSurface(VolSurface):
    """One number everywhere: the Black-Scholes textbook surface."""

    def __init__(self, vol: float, as_of: dt.date):
        self._vol = float(vol)
        self._ref = as_of

    def vol(self, strike, expiry) -> Tensor:
        # Broadcast against the strike so callers get the shape they passed in.
        return as_tensor(self._vol) * as_tensor(strike) ** 0

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    def __repr__(self) -> str:  # pragma: no cover
        return f"FlatVolSurface({self._vol:.4f}, {self._ref})"


class SVISurface(VolSurface):
    """Gatheral SVI, fitted per expiry.

    Stub: the parameterisation and the no-butterfly-arbitrage constraints are
    not implemented yet.
    """

    def __init__(self, as_of: dt.date):
        self._ref = as_of

    def vol(self, strike, expiry) -> Tensor:
        raise NotImplementedError("SVI fitting is not implemented yet")

    @property
    def reference_date(self) -> dt.date:
        return self._ref
