"""Treasury constant-maturity par yields to a continuously-compounded zero curve.

The published CMT curve is a *par* curve on a bond-equivalent basis:

* up to six months, a bill yield -- simple interest, ``DF(T) = 1 / (1 + y T)``;
* beyond, the coupon of a semiannual bond priced at par,
  ``sum_i (c/2) DF(t_i) + DF(T) = 1`` over coupon dates ``t_i = 0.5, 1.0, ...``.

A par yield is a yield to maturity: one rate applied to every cash flow of the
bond. Reading it as a zero rate is only right on a flat curve. On an upward
sloping curve the early coupons are really discounted at lower short rates, so
they are worth more than the single yield implies, the principal must be worth
less for the bond to stay at par, and the zero rate sits *above* the par yield --
by a basis point at 2Y and ~15bp at 30Y on a typical curve.

The bootstrap therefore interpolates par yields onto the semiannual coupon grid
(linear in the par yield; flat beyond the first and last quoted tenor) and
solves for discount factors in order of maturity, each from the ones before it.
The resulting zeros are returned at the *quoted* tenors only, so the curve's
pillars -- and its bucketed rho -- line up with the instruments actually quoted.

Tenors are nominal year fractions (``1/12``, ``0.25``, ``1``, ...), not dated
cash flows under a day count. That is the level of approximation this curve is
meant for: a risk-free proxy for option pricing, not a Treasury trading curve.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from torch_pricer.errors import ValidationError
from torch_pricer.market.curves import RateCurve

#: Coupon interval of a Treasury note or bond, in years.
_COUPON_PERIOD = 0.5


def _clean(tenors: Sequence[float], par_yields: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted tenors and par yields with missing quotes dropped."""
    t = np.asarray(tenors, dtype=float).ravel()
    y = np.asarray(par_yields, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValidationError(f"{t.size} tenors but {y.size} par yields")
    keep = np.isfinite(y)
    t, y = t[keep], y[keep]
    if t.size == 0:
        raise ValidationError("no par yields to build a curve from")
    if np.any(t <= 0):
        raise ValidationError("tenors must be positive")
    order = np.argsort(t)
    t, y = t[order], y[order]
    if np.any(np.diff(t) == 0):
        raise ValidationError("tenors must be distinct")
    coupon = t > _COUPON_PERIOD
    steps = t[coupon] / _COUPON_PERIOD
    if not np.allclose(steps, np.round(steps)):
        raise ValidationError(
            f"coupon-bearing tenors must be whole half-years, got {t[coupon].tolist()}"
        )
    return t, y


def bootstrap_discount_factors(
    tenors: Sequence[float], par_yields: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Discount factors on the semiannual coupon grid out to the longest tenor.

    Args:
        tenors: year fractions of the quoted yields.
        par_yields: bond-equivalent par yields as decimals (``0.0425``), NaN
            where not quoted.

    Returns:
        ``(times, discount_factors)`` at ``0.5, 1.0, ...`` up to the longest
        tenor; empty when every tenor is six months or shorter.
    """
    t, y = _clean(tenors, par_yields)
    n = int(round(t[-1] / _COUPON_PERIOD)) if t[-1] > _COUPON_PERIOD else 0
    grid = _COUPON_PERIOD * np.arange(1, n + 1)
    par = np.interp(grid, t, y)

    dfs = np.empty(n)
    annuity = 0.0  # sum of discount factors at the coupon dates already solved
    for i, c in enumerate(par):
        coupon = c * _COUPON_PERIOD
        dfs[i] = (1.0 - coupon * annuity) / (1.0 + coupon)
        annuity += dfs[i]
    if np.any(dfs <= 0):
        raise ValidationError("par yields imply a non-positive discount factor")
    return grid, dfs


def par_to_zero(tenors: Sequence[float], par_yields: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Continuously-compounded zero rates at the quoted tenors.

    Args:
        tenors: year fractions of the quoted yields.
        par_yields: bond-equivalent par yields as decimals, NaN where not quoted.

    Returns:
        ``(tenors, zeros)``, sorted, with unquoted tenors dropped.
    """
    t, y = _clean(tenors, par_yields)
    zeros = np.empty_like(t)

    bill = t <= _COUPON_PERIOD
    zeros[bill] = np.log1p(y[bill] * t[bill]) / t[bill]

    if np.any(~bill):
        grid, dfs = bootstrap_discount_factors(t, y)
        idx = np.rint(t[~bill] / _COUPON_PERIOD).astype(int) - 1
        zeros[~bill] = -np.log(dfs[idx]) / grid[idx]
    return t, zeros


def treasury_zero_curve(
    tenors: Sequence[float],
    par_yields_pct: Sequence[float],
    label: str = "UST",
) -> RateCurve:
    """A :class:`RateCurve` from one day's CMT par yields, quoted in percent.

    Example::

        curve = treasury_zero_curve([1/12, 0.25, 1, 2, 5, 10, 30],
                                    [3.96, 4.17, 4.45, 4.76, 4.83, 4.96, 5.29])
    """
    t, zeros = par_to_zero(tenors, np.asarray(par_yields_pct, dtype=float) / 100.0)
    return RateCurve.from_zeros(t, zeros, label=label)
