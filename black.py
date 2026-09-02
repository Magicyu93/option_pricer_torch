# formula for bs formula with constant r and sigma
import numpy as np
from scipy.stats import norm

_EPS = 1e-12

def d1_d2(forward, strike, t, vol):
    """The two Black arguments. Degenerate inputs are clamped, not rejected."""
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
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
    """''dV / dS^2': D N'(d1) / F / (sigma * sqrt(T)) Identical for calls and puts. """
    d1, _ = d1_d2(forward, strike, t, vol)
    sd = vol * np.sqrt(t)

    gamma = np.asarray(discount, dtype=float) * (
        norm.pdf(d1) / forward / np.asarray(sd, dtype=float)
    )
    return gamma


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
    '''
    Vectorised Black implied vol. ``nan`` where no volatility reproduces the price.

    :param price:
    :param forward:
    :param strike:
    :param t:
    :param discount:
    :param right:
    :param tol:
    :param max_newton:
    :param max_bisect:
    :param vol_bounds:
    :param vol_tolerance:
    :return:
    '''

    return NotImplemented