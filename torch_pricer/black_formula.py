# formula for bs formula with constant r and sigma
import numpy as np
from scipy.stats import norm

_EPS = 1e-12

def d1_d2(forward, strike, t, vol):
    """The two Black arguments. Degenerate inputs are clamped, not rejected."""
    forward = np.maximum(np.asarray(forward, dtype=float), _EPS)
    strike = np.maximum(np.asarray(strike, dtype=float), _EPS)
    t = np.maximum(np.asarray(t, dtype=float), _EPS)
    vol = np.maximum(np.asarray(vol, dtype=float), _EPS)
    sd = vol * np.sqrt(t)
    d1 = (np.log(forward / strike) + 0.5 * sd**2) / sd
    return d1, d1 - sd


def black_price(forward, strike, t, vol, discount=1.0, right=1):
    """Undiscounted-forward Black price, then discounted: ``D [w F N(w d1) - w K N(w d2)]``."""
    w = np.asarray(right, dtype=float)
    d1, d2 = d1_d2(forward, strike, t, vol)
    return np.asarray(discount, dtype=float) * w * (
        np.asarray(forward, dtype=float) * norm.cdf(w * d1)
        - np.asarray(strike, dtype=float) * norm.cdf(w * d2)
    )


def black_vega(forward, strike, t, vol, discount=1.0):
    """``dV/dsigma`` per 1.00 of vol. Identical for calls and puts."""
    d1, _ = d1_d2(forward, strike, t, vol)
    t = np.maximum(np.asarray(t, dtype=float), _EPS)
    forward = np.asarray(forward, dtype=float)
    return np.asarray(discount, dtype=float) * forward * norm.pdf(d1) * np.sqrt(t)


def black_delta(forward, strike, t, vol, discount=1.0, right=1):
    """Delta with respect to the *forward*. Multiply by ``dF/dS = D_q/D_r`` for spot delta."""
    w = np.asarray(right, dtype=float)
    d1, _ = d1_d2(forward, strike, t, vol)
    return np.asarray(discount, dtype=float) * w * norm.cdf(w * d1)


def black_gamma(forward, strike, t, vol, discount=1.0):
    """Gamma with respect to the *forward*: ``D N'(d1) / (F sigma sqrt(T))``.

    This is ``d2V/dF2``, not ``d2V/dS2``. For spot gamma multiply by
    ``(dF/dS)^2 = (D_q/D_r)^2``, the same conversion :func:`black_delta` needs.
    Identical for calls and puts.
    """
    d1, _ = d1_d2(forward, strike, t, vol)
    # Clamp exactly as d1_d2 does. Recomputing sd from the raw arguments would
    # divide by zero at t=0 or vol=0 while every sibling degrades gracefully.
    forward = np.maximum(np.asarray(forward, dtype=float), _EPS)
    t = np.maximum(np.asarray(t, dtype=float), _EPS)
    vol = np.maximum(np.asarray(vol, dtype=float), _EPS)
    sd = vol * np.sqrt(t)
    return np.asarray(discount, dtype=float) * norm.pdf(d1) / (forward * sd)


def intrinsic(forward, strike, discount=1.0, right=1):
    w = np.asarray(right, dtype=float)
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    return np.maximum(np.asarray(discount, dtype=float) * w * (forward - strike), 0.0)


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
):
    """Vectorised Black implied vol. ``nan`` where no volatility reproduces the price.

    Args broadcast against each other, as in :func:`black_price`; ``right`` is
    +1 for a call and -1 for a put.

    Newton on the price, kept inside a bracket that every step narrows, for
    ``max_newton`` steps; whatever has not converged then bisects for up to
    ``max_bisect`` more. Newton alone fails exactly where it matters for listed
    options -- far out of the money, where vega vanishes and a step overshoots
    the bracket -- and bisection alone is slow; the bracket makes the hand-off
    safe.

    A point has converged when the price matches to ``tol`` relative to its
    *time value*, or the bracket is narrower than ``vol_tolerance``. Relative to
    the time value, not the premium: deep in the money the premium is almost all
    intrinsic, and a premium-relative tolerance stops while the vol is still
    off by tens of points. For the same reason a time value below float64
    resolution of the premium cannot be inverted at all.

    ``nan`` marks prices outside the no-arbitrage range ``(intrinsic, F or K)``,
    time value too small to resolve, prices no vol in ``vol_bounds`` reaches,
    and points that never converged.
    """
    w = np.asarray(right, dtype=float)
    price, forward, strike, t, discount, w = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, forward, strike, t, discount)), w
    )
    target = price / discount  # undiscounted, so the bounds are F and K

    lo = np.full(target.shape, float(vol_bounds[0]))
    hi = np.full(target.shape, float(vol_bounds[1]))

    def undiscounted(vol):
        return black_price(forward, strike, t, vol, 1.0, w)

    lower, upper = np.maximum(w * (forward - strike), 0.0), np.where(w > 0, forward, strike)
    time_value = target - lower
    valid = (
        np.isfinite(target) & (t > 0) & (target < upper)
        & (time_value > 1e-12 * np.maximum(target, 1.0))
        & (undiscounted(lo) <= target) & (target <= undiscounted(hi))
    )

    # Start where the time value is ATM-like: sigma ~ sqrt(2 |ln F/K| / T),
    # floored so an exactly-ATM point does not start at the lower bound.
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = np.sqrt(2.0 * np.abs(np.log(forward / strike)) / t)
    vol = np.clip(np.where(np.isfinite(guess), np.maximum(guess, 0.2), 0.2), lo, hi)
    done = ~valid

    for step in range(max_newton + max_bisect):
        diff = undiscounted(vol) - target
        converged = (np.abs(diff) <= tol * np.maximum(time_value, 1e-300)) | (hi - lo <= vol_tolerance)
        done |= converged
        if done.all():
            break
        # Price is increasing in vol, so the sign of the error says which end moves.
        hi = np.where(~done & (diff > 0), vol, hi)
        lo = np.where(~done & (diff < 0), vol, lo)
        midpoint = 0.5 * (lo + hi)
        if step < max_newton:
            vega = black_vega(forward, strike, t, vol)
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = vol - diff / vega
            inside = np.isfinite(newton) & (newton > lo) & (newton < hi)
            proposal = np.where(inside, newton, midpoint)
        else:
            proposal = midpoint
        vol = np.where(done, vol, proposal)

    return np.where(valid & done, vol, np.nan)