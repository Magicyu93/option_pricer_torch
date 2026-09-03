"""Discount curves, as torch modules.

A curve is a set of pillar zero rates and an interpolation rule, held as
tensors. Nothing here is a QuantLib term structure, and that is the whole point:
the discount factor multiplies the payoff, so it sits inside the autograd graph.
Because the pillars are an ``nn.Parameter``, bucketed rho falls out of the same
backward pass that produces delta -- no curve rebuild, no bump loop.

Time is a year fraction from the snapshot's ``as_of``, never a date. The
conversion happens once, in :func:`torch_pricer.conventions.year_fraction`.

Interpolation is linear on the *integrated* zero ``z(t) t`` rather than on
``z(t)``. That is the standard choice because it makes the implied forward rate
piecewise constant between pillars instead of piecewise linear, and it keeps
discount factors monotone. Outside the pillar range the zero rate is held flat.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...errors import ValidationError
from ...tensors import EPS, as_tensor


def _interp_linear(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation of ``fp`` over knots ``xp``, evaluated at ``x``.

    ``xp`` must be sorted and hold at least two points. Values outside the knot
    range are extrapolated along the nearest segment; callers that want flat
    extrapolation clamp ``x`` first.
    """
    idx = torch.searchsorted(xp, x.detach().contiguous()).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)


class RateCurve(nn.Module):
    """A zero curve carrying its own pillar rates.

    Build one with :meth:`flat` or :meth:`from_zeros` rather than ``__init__``.
    """

    def __init__(self, pillar_times: Tensor, pillar_zeros: Tensor, label: str = "curve"):
        super().__init__()
        times = as_tensor(pillar_times).flatten()
        zeros = as_tensor(pillar_zeros).flatten()
        if times.numel() != zeros.numel():
            raise ValidationError(
                f"curve {label!r}: {times.numel()} pillar times but {zeros.numel()} rates"
            )
        if times.numel() == 0:
            raise ValidationError(f"curve {label!r}: needs at least one pillar")
        if times.numel() > 1 and not bool((times[1:] > times[:-1]).all()):
            raise ValidationError(f"curve {label!r}: pillar times must be strictly increasing")
        if bool((times < 0).any()):
            raise ValidationError(f"curve {label!r}: pillar times must be non-negative")

        self.register_buffer("pillar_times", times)
        self.pillar_zeros = nn.Parameter(zeros.clone())
        self.label = label

    # -- constructors ---------------------------------------------------
    @classmethod
    def flat(cls, rate: float, label: str = "flat") -> RateCurve:
        """A flat continuously-compounded curve at ``rate``."""
        return cls(torch.tensor([1.0]), as_tensor([float(rate)]), label)

    @classmethod
    def from_zeros(cls, times, zeros, label: str = "zero") -> RateCurve:
        """A curve through continuously-compounded zero rates at the given year fractions."""
        return cls(as_tensor(times), as_tensor(zeros), label)

    # -- queries --------------------------------------------------------
    def zero_rate(self, t) -> Tensor:
        """Continuously-compounded zero rate to ``t`` years."""
        t = as_tensor(t, dtype=self.pillar_zeros.dtype, device=self.pillar_zeros.device)
        if self.pillar_times.numel() == 1:
            return self.pillar_zeros[0].expand(t.shape) if t.dim() else self.pillar_zeros[0]

        tp = self.pillar_times
        clamped = t.clamp(float(tp[0]), float(tp[-1]))
        integrated = _interp_linear(clamped, tp, self.pillar_zeros * tp)
        inside = integrated / clamped.clamp_min(EPS)
        # Flat in the zero rate beyond the end pillars.
        return torch.where(
            t < tp[0], self.pillar_zeros[0], torch.where(t > tp[-1], self.pillar_zeros[-1], inside)
        )

    def discount(self, t) -> Tensor:
        """Discount factor to ``t`` years. ``discount(0) == 1`` by construction."""
        t = as_tensor(t, dtype=self.pillar_zeros.dtype, device=self.pillar_zeros.device)
        return torch.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1, t2) -> Tensor:
        """Continuously-compounded forward rate over ``[t1, t2]``."""
        t1 = as_tensor(t1, dtype=self.pillar_zeros.dtype, device=self.pillar_zeros.device)
        t2 = as_tensor(t2, dtype=self.pillar_zeros.dtype, device=self.pillar_zeros.device)
        span = (t2 - t1).clamp_min(EPS)
        return (torch.log(self.discount(t1)) - torch.log(self.discount(t2))) / span

    def instantaneous_forward(self, t, bump: float = 1e-4) -> Tensor:
        """The short rate at ``t``, as a one-basis-point-of-a-year forward.

        This is what an SDE's drift wants. For a flat curve it is exact; for a
        pillared curve it is the forward over a very short window, which is the
        same approximation an Euler step already makes about the drift being
        constant across the step.
        """
        return self.forward_rate(t, as_tensor(t) + bump)

    def extra_repr(self) -> str:  # pragma: no cover
        n = self.pillar_times.numel()
        return f"{self.label!r}, {n} pillar{'' if n == 1 else 's'}"
