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
):
    """Vectorised Black implied vol. ``nan`` where no volatility reproduces the price.

    Newton first, because vega is analytic and the iteration is quadratic near
    the root; bisection afterwards for whatever Newton left, because vega
    collapses far from the money and a Newton step there can leap outside the
    bracket entirely. Bisection cannot fail once the price is inside the
    no-arbitrage bounds, so the two together always land.

    Prices outside ``[intrinsic, forward bound]`` come back ``nan`` rather than
    clamped: there is no such volatility, and returning the nearest one would
    launder an arbitrage into a plausible number. A price *on* either bound is
    ``nan`` for the same reason -- an option with no time value has no
    recoverable vol, since vega there is zero to machine precision and every
    volatility over a wide range reproduces the same double.
    """
    price, forward, strike, t, discount, w = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, forward, strike, t, discount, right))
    )
    price, forward, strike = price.copy(), forward.copy(), strike.copy()
    t = np.maximum(t, _EPS)

    lo, hi = vol_bounds
    # No-arbitrage bounds: a call is worth between its intrinsic and the
    # discounted forward; a put between intrinsic and the discounted strike.
    floor = np.maximum(discount * w * (forward - strike), 0.0)
    cap = discount * np.where(w > 0, forward, strike)
    # Strictly interior: on the bound there is no time value, hence no vol.
    feasible = (price > floor + tol) & (price < cap - tol)

    vol = np.full(price.shape, np.nan, dtype=float)
    # Brenner-Subrahmanyam: exact at the money, a good bracket-interior start
    # everywhere else.
    guess = np.sqrt(2.0 * np.pi / t) * np.divide(
        price, discount * forward, out=np.zeros_like(price), where=discount * forward > 0
    )
    x = np.clip(np.where(np.isfinite(guess) & (guess > 0), guess, 0.2), lo, hi)

    active = feasible.copy()
    for _ in range(max_newton):
        if not active.any():
            break
        err = black_price(forward, strike, t, x, discount, w) - price
        v = black_vega(forward, strike, t, x, discount)
        done = np.abs(err) < tol
        active &= ~done
        step = np.divide(err, v, out=np.zeros_like(err), where=v > _EPS)
        nxt = x - step
        # A Newton step is only trusted while it stays inside the bracket.
        ok = active & np.isfinite(nxt) & (nxt > lo) & (nxt < hi) & (v > _EPS)
        x = np.where(ok, nxt, x)
        active &= ok

    vol = np.where(feasible, x, np.nan)
    err = black_price(forward, strike, t, x, discount, w) - price
    stubborn = feasible & (np.abs(err) > tol)
    if stubborn.any():
        a = np.full(price.shape, lo)
        b = np.full(price.shape, hi)
        for _ in range(max_bisect):
            mid = 0.5 * (a + b)
            f = black_price(forward, strike, t, mid, discount, w) - price
            a = np.where(f < 0.0, mid, a)
            b = np.where(f < 0.0, b, mid)
        vol = np.where(stubborn, 0.5 * (a + b), vol)
    return vol
