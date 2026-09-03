"""Closed-form Black formulas, in torch.

Everything is expressed on the *forward*, not the spot: ``F`` already carries
the funding and dividend legs, so these formulas contain no rates. The caller
supplies the discount factor separately.

Being torch rather than numpy buys two things. Analytic prices become a
differentiable reference, so the closed-form greeks below can be checked against
autograd through :func:`black_price`; and the same code runs on whatever device
the Monte Carlo engine is using, which is what makes it usable as a control
variate later.

Degenerate inputs are clamped, not rejected: a surface fit routinely evaluates
at zero time or zero vol on its way to somewhere sensible, and raising there
turns a convergent calibration into a crash.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..tensors import EPS, as_tensor

_SQRT_HALF = math.sqrt(0.5)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF. ``erfc`` rather than ``1 + erf`` for the left tail."""
    return 0.5 * torch.erfc(-x * _SQRT_HALF)


def _norm_pdf(x: Tensor) -> Tensor:
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _clamped(t, vol) -> tuple[Tensor, Tensor, Tensor]:
    """``(t, vol, vol*sqrt(t))``, all guarded against zero.

    Every entry point runs inputs through here so that the clamping is identical
    across price, vega, delta and gamma. Computing ``vol * sqrt(t)`` a second
    time from the raw arguments is how a divide-by-zero gets into one function
    and not its neighbours.
    """
    t = as_tensor(t).clamp_min(EPS)
    vol = as_tensor(vol).clamp_min(EPS)
    return t, vol, vol * t.sqrt()


def d1_d2(forward, strike, t, vol) -> tuple[Tensor, Tensor]:
    """The two Black arguments."""
    forward = as_tensor(forward)
    strike = as_tensor(strike)
    _, _, sd = _clamped(t, vol)
    d1 = (torch.log(forward / strike) + 0.5 * sd * sd) / sd
    return d1, d1 - sd


def black_price(forward, strike, t, vol, discount=1.0, right=1) -> Tensor:
    """``D [w F N(w d1) - w K N(w d2)]``, with ``w = +1`` call, ``-1`` put."""
    w = as_tensor(right)
    d1, d2 = d1_d2(forward, strike, t, vol)
    return as_tensor(discount) * w * (
        as_tensor(forward) * _norm_cdf(w * d1) - as_tensor(strike) * _norm_cdf(w * d2)
    )


def black_vega(forward, strike, t, vol, discount=1.0) -> Tensor:
    """``dV/dsigma`` per 1.00 of vol. Identical for calls and puts."""
    t, _, _ = _clamped(t, vol)
    d1, _ = d1_d2(forward, strike, t, vol)
    return as_tensor(discount) * as_tensor(forward) * _norm_pdf(d1) * t.sqrt()


def black_delta(forward, strike, t, vol, discount=1.0, right=1) -> Tensor:
    """Delta with respect to the *forward*. Multiply by ``dF/dS = D_q/D_r`` for spot delta."""
    w = as_tensor(right)
    d1, _ = d1_d2(forward, strike, t, vol)
    return as_tensor(discount) * w * _norm_cdf(w * d1)


def black_gamma(forward, strike, t, vol, discount=1.0) -> Tensor:
    """``d2V/dF2 = D N'(d1) / (F sigma sqrt(T))``. Identical for calls and puts."""
    forward = as_tensor(forward)
    _, _, sd = _clamped(t, vol)
    d1, _ = d1_d2(forward, strike, t, vol)
    return as_tensor(discount) * _norm_pdf(d1) / (forward * sd)


def intrinsic(forward, strike, discount=1.0, right=1) -> Tensor:
    """Discounted intrinsic value, floored at zero."""
    w = as_tensor(right)
    forward = as_tensor(forward)
    strike = as_tensor(strike)
    return torch.clamp_min(as_tensor(discount) * w * (forward - strike), 0.0)


@torch.no_grad()
def implied_vol(
    price,
    forward,
    strike,
    t,
    discount=1.0,
    right=1,
    tol: float = 1e-8,
    max_newton: int = 12,
    max_bisect: int = 60,
    vol_bounds: tuple[float, float] = (1e-4, 5.0),
    vol_tolerance: float = 1e-5,
) -> Tensor:
    """Vectorised Black implied vol. ``nan`` where no volatility reproduces the price.

    Newton from a Brenner-Subrahmanyam seed converges in a handful of steps near
    the money but is unreliable in the wings, where vega underflows and the step
    explodes. Bisection on ``vol_bounds`` is unconditionally convergent because
    price is strictly increasing in vol, so both run and the bisection answer is
    kept wherever Newton failed to land within ``tol``.

    Runs under ``no_grad``: this is a root find, and differentiating through the
    iterates is both wasteful and wrong. If you need ``dsigma/dprice``, it is
    ``1 / vega`` at the solution.

    :param price: option premium, on the same discounted basis as ``discount``
    :param forward: forward level to expiry
    :param strike: option strike
    :param t: time to expiry in years
    :param discount: discount factor to the payment date
    :param right: ``+1`` for a call, ``-1`` for a put
    :param tol: absolute price tolerance for declaring Newton converged
    :param max_newton: Newton iterations before falling back to the bisection root
    :param max_bisect: bisection iterations; 60 exhausts float64 on the default bounds
    :param vol_bounds: search bracket, also the feasible range of the answer
    :param vol_tolerance: answers this close to a bracket end are treated as no solution
    :return: implied volatility, broadcast to the shape of the inputs
    """
    lo_bound, hi_bound = vol_bounds
    if not 0 < lo_bound < hi_bound:
        raise ValueError(f"vol_bounds must be an increasing positive pair, got {vol_bounds}")

    price, forward, strike, t, discount, w = torch.broadcast_tensors(
        as_tensor(price),
        as_tensor(forward),
        as_tensor(strike),
        as_tensor(t),
        as_tensor(discount),
        as_tensor(right),
    )

    # Work undiscounted, in forward space, so the no-arbitrage band is simple.
    target = price / discount.clamp_min(EPS)
    is_call = w > 0
    floor = torch.clamp_min(w * (forward - strike), 0.0)
    ceiling = torch.where(is_call, forward, strike)
    feasible = (target >= floor - tol) & (target <= ceiling + tol) & (t > 0)

    # Brenner-Subrahmanyam: exact at the money, a serviceable seed elsewhere.
    safe_t = t.clamp_min(EPS)
    seed = (target / forward.clamp_min(EPS)) * math.sqrt(2.0 * math.pi) / safe_t.sqrt()
    v = seed.clamp(lo_bound, hi_bound)

    for _ in range(max_newton):
        err = black_price(forward, strike, safe_t, v, 1.0, w) - target
        vega = black_vega(forward, strike, safe_t, v, 1.0)
        # Where vega underflows the step is meaningless; freeze and let bisection carry it.
        step = torch.where(vega > EPS, err / vega.clamp_min(EPS), torch.zeros_like(err))
        v = (v - step).clamp(lo_bound, hi_bound)

    newton_err = (black_price(forward, strike, safe_t, v, 1.0, w) - target).abs()
    converged = newton_err <= tol

    lo = torch.full_like(v, lo_bound)
    hi = torch.full_like(v, hi_bound)
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        err = black_price(forward, strike, safe_t, mid, 1.0, w) - target
        too_low = err < 0  # price rises with vol, so we are below the root
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    bisected = 0.5 * (lo + hi)

    v = torch.where(converged, v, bisected)

    # A root pinned to a bracket end is the search giving up, not an answer.
    pinned = (v <= lo_bound + vol_tolerance) | (v >= hi_bound - vol_tolerance)
    final_err = (black_price(forward, strike, safe_t, v, 1.0, w) - target).abs()
    # A looser gate than `tol`: bisection is exhausted at float64 resolution, so
    # anything still off by more than this never had a root in the bracket.
    accept = max(tol * 1e3, 1e-6)
    solved = feasible & ~pinned & (final_err <= accept)
    return torch.where(solved, v, torch.full_like(v, float("nan")))
