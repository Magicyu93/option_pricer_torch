"""The curve interface every rate and carry curve implements.

A curve is anything with a continuously-compounded zero rate as a function of
time in years from the snapshot's ``as_of``. Everything else -- discount
factors, forward rates, the short rate an SDE drift wants -- follows from that,
so a new curve model (pillar zeros, a spread over another curve, a parametric
Nelson-Siegel) implements :meth:`Curve.zero_rate` and nothing else, and is then
usable wherever a curve is: in a :class:`~torch_pricer.market.snapshot.MarketSnapshot`,
in a model's drift, in the engine's discounting.

Curves are ``nn.Module`` s holding their parameters as tensors, so that rho is
a backward pass rather than a bump: :attr:`Curve.risk_factors` names the tensor
the engine differentiates against.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain

import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.tensors import EPS, as_tensor


class Curve(nn.Module, ABC):
    """A term structure of continuously-compounded zero rates."""

    label: str = "curve"

    @abstractmethod
    def zero_rate(self, t) -> Tensor:
        """Continuously-compounded zero rate to ``t`` years."""

    @property
    @abstractmethod
    def risk_factors(self) -> Tensor:
        """The tensor rho is taken against: one entry per bucket of exposure."""

    def _time(self, t) -> Tensor:
        """``t`` as a tensor on this curve's dtype and device."""
        ref = next(chain(self.parameters(), self.buffers()), None)
        if ref is None:
            return as_tensor(t)
        return as_tensor(t, dtype=ref.dtype, device=ref.device)

    def discount(self, t) -> Tensor:
        """Discount factor to ``t`` years. ``discount(0) == 1`` by construction."""
        t = self._time(t)
        return torch.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1, t2) -> Tensor:
        """Continuously-compounded forward rate over ``[t1, t2]``."""
        t1, t2 = self._time(t1), self._time(t2)
        span = (t2 - t1).clamp_min(EPS)
        return (torch.log(self.discount(t1)) - torch.log(self.discount(t2))) / span

    def instantaneous_forward(self, t, bump: float = 1e-4) -> Tensor:
        """The short rate at ``t``, as a one-basis-point-of-a-year forward.

        This is what an SDE's drift wants. For a flat curve it is exact; for a
        pillared curve it is the forward over a very short window, which is the
        same approximation an Euler step already makes about the drift being
        constant across the step.
        """
        t = self._time(t)
        return self.forward_rate(t, t + bump)
