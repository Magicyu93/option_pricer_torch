"""Instrument specs to differentiable payoffs over simulated asset paths.

A payoff never sees the simulator's state coordinate. The SDE may be integrating
log-spot, or a two-component ``(spot, variance)`` vector; it hands back asset
levels through :meth:`~torch_pricer.simulator.simulator.SDE.asset` and payoffs
work in that one currency. That is what lets a single payoff serve every model
on the roadmap.

Payoffs must stay differentiable in the asset path, because delta is taken by
autograd straight through them. ``clamp_min`` is fine -- it is the kink at the
strike, and the pathwise estimator handles a kink on a null set. A *jump*, as in
a digital or a knock-out, is not: its derivative is zero almost everywhere and
autograd will confidently report a delta of zero. Those need a smoothed payoff
before they can be priced this way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..errors import ValidationError
from .spec import AsianOption, AverageKind, Instrument, Right, Stock, VanillaOption


class Payoff(ABC):
    """A contract's cashflow at expiry as a function of the simulated asset."""

    #: Whether the engine must retain the whole trajectory. Terminal-only
    #: payoffs let it keep just the final state, which is the difference between
    #: holding one tensor and holding ``n_steps`` of them with their graph.
    needs_path: bool = False

    @abstractmethod
    def __call__(self, asset: Tensor, grid: Tensor) -> Tensor:
        """Undiscounted payoff per path.

        Args:
            asset: ``(n_paths,)`` terminal levels, or ``(n_paths, n_steps + 1)``
                when :attr:`needs_path`.
            grid: ``(n_steps + 1,)`` time grid in years from the snapshot date.

        Returns:
            ``(n_paths,)`` undiscounted payoff.
        """


class ForwardPayoff(Payoff):
    """The asset itself. Present so a hedged book prices through one code path."""

    def __call__(self, asset: Tensor, grid: Tensor) -> Tensor:
        return asset


class EuropeanPayoff(Payoff):
    """``max(w (S_T - K), 0)``."""

    needs_path = False

    def __init__(self, strike: float, right: Right):
        self.strike = float(strike)
        self.sign = float(Right(right).sign)

    def __call__(self, asset: Tensor, grid: Tensor) -> Tensor:
        return torch.clamp_min(self.sign * (asset - self.strike), 0.0)


class AsianPayoff(Payoff):
    """Fixed-strike Asian on the average of the simulation grid.

    The average is taken over the grid points after ``t=0``, which is the
    continuous average the spec describes when it carries no ``fixing_dates``.
    Contract fixing dates are rejected rather than approximated: silently
    averaging over the wrong dates is the kind of error that prices a book
    plausibly and wrongly for a year.
    """

    needs_path = True

    def __init__(self, strike: float, right: Right, average: AverageKind):
        self.strike = float(strike)
        self.sign = float(Right(right).sign)
        self.average = AverageKind(average)

    def __call__(self, asset: Tensor, grid: Tensor) -> Tensor:
        body = asset[:, 1:]
        if self.average is AverageKind.GEOMETRIC:
            mean = torch.exp(torch.log(body).mean(dim=1))
        else:
            mean = body.mean(dim=1)
        return torch.clamp_min(self.sign * (mean - self.strike), 0.0)


def payoff_for(spec: Instrument) -> Payoff:
    """Map an instrument spec to its payoff."""
    if isinstance(spec, VanillaOption):
        return EuropeanPayoff(spec.strike, spec.right)
    if isinstance(spec, AsianOption):
        if spec.fixing_dates:
            raise ValidationError(
                "discrete fixing dates are not wired to the simulation grid yet; "
                "leave fixing_dates empty for the continuous average"
            )
        if spec.past_fixings:
            raise ValidationError("seasoned Asians (past_fixings > 0) are not supported yet")
        return AsianPayoff(spec.strike, spec.right, spec.average)
    if isinstance(spec, Stock):
        return ForwardPayoff()
    raise ValidationError(f"no payoff registered for {type(spec).__name__}")
