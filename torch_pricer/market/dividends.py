"""Discrete cash dividends, and the escrowed-spot transformation.

A continuous yield is the wrong model for a single name, and it is wrong exactly
where it matters most. Early exercise of an American *call* is driven almost
entirely by discrete cash dividends: it is optimal only in the instant before an
ex-date, when the holder gives up remaining time value to capture a payment.
Smearing that payment into a constant yield removes the very discontinuity the
decision turns on, so the option prices plausibly and the exercise boundary is
nonsense. (An American *put* is rate-driven and survives a yield perfectly well,
which is why puts came first.)

**The escrowed-spot model.** Cash dividends break the two things every engine
here relies on. They make the process non-lognormal, and -- because the drop is
*additive* while a lattice is multiplicative -- they stop a tree recombining, so
an n-step tree would hold 2^n nodes. The standard repair is to split the spot:

    S(t) = S*(t) + PV_t(dividends still to come before expiry)

and let the *escrowed* part ``S*`` be the lognormal one. The known cashflows are
held aside at their present value and added back; ``S*`` recombines, stays
positive, and takes the volatility. At expiry no dividends remain, so
``S(T) = S*(T)`` and a European payoff sees nothing but a reduced initial spot.
Early exercise is where the add-back earns its keep: the intrinsic at an interim
date must be taken against the *real* spot, not the escrowed one.

It is a model, not an identity. Quoted volatility is implicitly a volatility of
``S``, and this applies it to ``S*``, which overstates the volatility of the
whole; the error grows with the dividend's size relative to spot and is small for
ordinary equity yields. It is Hull's treatment and what the tree and the
simulation here both implement -- the same approximation on both sides, so the
cross-check tests the *implementation* rather than laundering a model difference
into an agreement.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import torch
from torch import Tensor

from torch_pricer.conventions import to_date, year_fraction
from torch_pricer.errors import ValidationError


@dataclass(frozen=True)
class DividendSchedule:
    """Known cash dividends, by ex-date.

    Amounts are per share in the underlying's currency, not yields. A schedule
    is a market observation like any other: it does no discounting and holds no
    curve.
    """

    ex_dates: tuple[dt.date, ...] = ()
    amounts: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ex_dates", tuple(to_date(d) for d in self.ex_dates))
        object.__setattr__(self, "amounts", tuple(float(a) for a in self.amounts))
        if len(self.ex_dates) != len(self.amounts):
            raise ValidationError(
                f"{len(self.ex_dates)} ex-dates but {len(self.amounts)} amounts"
            )
        if any(a < 0 for a in self.amounts):
            raise ValidationError("dividend amounts must be non-negative")
        if list(self.ex_dates) != sorted(self.ex_dates):
            raise ValidationError("ex-dates must be in ascending order")

    def __bool__(self) -> bool:
        return bool(self.ex_dates)

    def times(self, as_of: dt.date, day_count: str) -> list[float]:
        """Ex-dates as year fractions from ``as_of``."""
        return [year_fraction(as_of, d, day_count) for d in self.ex_dates]

    def pv_remaining(
        self,
        t: Tensor,
        times: list[float],
        discount,
        horizon: float,
    ) -> Tensor:
        """PV at each time in ``t`` of the dividends still ahead of it.

        Only dividends strictly after ``t`` and at or before ``horizon`` count: a
        payment after the option expires cannot affect it, and one already passed
        is in the holder's pocket rather than the stock.

        The result is differentiable in the discount curve -- that is where
        bucketed rho on a dividend-paying name comes from -- while the *selection*
        of which dividends are still ahead is a comparison on detached values,
        being a discrete fact about the calendar rather than a function of it.
        """
        total = torch.zeros_like(t)
        if not self.ex_dates:
            return total
        t_detached = t.detach()
        for time, amount in zip(times, self.amounts):
            if time > horizon or amount == 0.0:
                continue
            ahead = t_detached < time
            if not bool(ahead.any()):
                continue
            # D * DF(t_i) / DF(t): the payment carried back to each time in t.
            ratio = discount.discount(torch.full_like(t, time)) / discount.discount(t)
            total = total + torch.where(ahead, amount * ratio, torch.zeros_like(t))
        return total
