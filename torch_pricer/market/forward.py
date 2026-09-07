"""Forwards and discount factors implied by put-call parity.

This is the piece that decides whether a real-data accuracy claim means anything.

``fit_surface`` places each quote at ``k = log(K / F(T))``, and ``F`` comes from
``MarketSnapshot.forward`` -- that is, from the discount and dividend curves. So
the smile sits wherever those curves put it. Source them independently, a SOFR
curve from one place and a dividend forecast from another, and any error in
either shifts the whole slice sideways, where it is indistinguishable from a
model failure. Weeks can be spent debugging a pricer for a bad forward.

The repair is to stop sourcing them. For a European chain, parity is an identity:

    C(K) - P(K) = D * (F - K)

Both ``D`` and ``F`` are unknown and neither depends on ``K``, so regressing the
call-put difference on the strike recovers both at once -- slope ``-D``,
intercept ``D F``. Nothing enters but the option quotes themselves, and the
forward that results is the one the market is actually trading, dividend
forecast and borrow cost included, whether or not anyone has estimated them.

**European chains only.** For an American contract parity degrades to an
inequality, ``S - K <= C - P <= S - K e^{-rT}``, and the regression above would
be fitting a band rather than a line. That is why SPX and XSP come first and
single names need a genuine dividend and borrow estimate instead -- see
``tests/test_tree.py::test_american_parity_bounds_hold``.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

from torch_pricer.errors import MarketDataError
from torch_pricer.instruments.spec import Right
from torch_pricer.market.market_data import QuoteSet


@dataclass(frozen=True)
class ImpliedForward:
    """What parity says about one expiry."""

    expiry: dt.date
    t: float
    forward: float
    discount: float
    #: Strike pairs the regression used.
    n_pairs: int
    #: Largest absolute parity residual, in premium terms. A clean European chain
    #: sits near the tick; anything larger means stale quotes or a mis-paired row.
    max_residual: float

    @property
    def zero_rate(self) -> float:
        """Continuously-compounded rate implied by the discount factor."""
        return -math.log(self.discount) / self.t

    @property
    def dividend_rate(self) -> float:
        """The yield that reconciles this forward with spot, given the discount.

        ``F = S D_q / D_r``, so ``q = r - log(F / S) / t``. For an index this is
        the market's own dividend forecast; for a single name it would also carry
        the borrow cost, which is one reason single names are harder.
        """
        raise NotImplementedError  # needs spot; see implied_forwards


def _pairs(quotes: QuoteSet, expiry: dt.date):
    """Strikes quoted on both sides, with their call and put mids."""
    calls, puts = {}, {}
    for q in quotes.slice(expiry):
        if q.price is None:
            continue
        (calls if q.right is Right.CALL else puts)[q.strike] = q.price
    shared = sorted(set(calls) & set(puts))
    return shared, [calls[k] for k in shared], [puts[k] for k in shared]


def implied_forward(
    quotes: QuoteSet,
    expiry: dt.date,
    t: float,
    atm_window: float = 0.10,
    min_pairs: int = 3,
) -> ImpliedForward:
    """Forward and discount factor for one expiry, by regressing parity.

    Args:
        quotes: a European chain
        expiry: the expiry to fit
        t: year fraction to that expiry
        atm_window: keep strikes within this fraction of the rough forward. Deep
            wings are where one leg is nearly worthless and its quote is tick
            noise, which the regression would weight equally.
        min_pairs: fewest two-sided strikes to attempt a fit

    Returns:
        An :class:`ImpliedForward`.

    Raises:
        MarketDataError: with too few pairs, or a degenerate fit.
    """
    strikes, calls, puts = _pairs(quotes, expiry)
    if len(strikes) < min_pairs:
        raise MarketDataError(
            f"{expiry}: {len(strikes)} two-sided strike pairs, need {min_pairs}"
        )

    k = np.asarray(strikes, dtype=float)
    diff = np.asarray(calls, dtype=float) - np.asarray(puts, dtype=float)

    # A first pass over everything locates the forward; the second keeps only the
    # strikes around it, where both legs carry real premium.
    slope, intercept = np.polyfit(k, diff, 1)
    if slope >= 0:
        raise MarketDataError(
            f"{expiry}: parity regression gave a non-negative slope ({slope:.4g}); "
            "the call-put difference must fall with strike"
        )
    rough_forward = intercept / -slope

    near = np.abs(k / rough_forward - 1.0) <= atm_window
    if int(near.sum()) >= min_pairs:
        k, diff = k[near], diff[near]
        slope, intercept = np.polyfit(k, diff, 1)

    discount = float(-slope)
    if not 0.0 < discount <= 1.5:
        raise MarketDataError(
            f"{expiry}: implied discount factor {discount:.4g} is not plausible"
        )
    forward = float(intercept / discount)
    residual = float(np.max(np.abs(diff - (intercept + slope * k))))
    return ImpliedForward(
        expiry=expiry, t=t, forward=forward, discount=discount,
        n_pairs=int(k.size), max_residual=residual,
    )


def implied_forwards(
    quotes: QuoteSet,
    year_fraction_of,
    atm_window: float = 0.10,
    min_pairs: int = 3,
) -> list[ImpliedForward]:
    """Every expiry that can be fitted, in order. Unfittable ones are skipped.

    Skipping rather than raising is deliberate: a chain routinely holds one
    illiquid expiry among many, and losing the whole surface to it would be worse
    than losing the slice.
    """
    out = []
    for expiry in quotes.expiries():
        t = year_fraction_of(expiry)
        if t <= 0:
            continue
        try:
            out.append(implied_forward(quotes, expiry, t, atm_window, min_pairs))
        except MarketDataError:
            continue
    if not out:
        raise MarketDataError("no expiry had enough two-sided strikes to imply a forward")
    return out


def curves_from_forwards(forwards: list[ImpliedForward], spot: float):
    """``(discount RateCurve, dividend RateCurve)`` reproducing these forwards.

    Built so that ``MarketSnapshot.forward(t)`` returns the parity forward at
    every fitted expiry, to floating point. That is the whole objective: the
    log-moneyness a calibration sees is then the market's own, and a residual
    afterwards is the model's rather than the curve's.
    """
    from torch_pricer.market.curves import RateCurve

    times = [f.t for f in forwards]
    zeros = [f.zero_rate for f in forwards]
    # F = S D_q / D_r  =>  q = r - log(F / S) / t
    dividends = [
        f.zero_rate - math.log(f.forward / spot) / f.t for f in forwards
    ]
    return (
        RateCurve.from_zeros(times, zeros, "parity_discount"),
        RateCurve.from_zeros(times, dividends, "parity_dividend"),
    )
