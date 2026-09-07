"""Semi-analytic Heston, as the reference a simulated Heston is checked against.

Heston has a closed-form characteristic function, so European prices follow from
one numerical integral per probability. That matters here for one reason: the
Monte Carlo scheme for the variance process is *not* exact, and a pathwise greek
taken through its truncation is biased. Without an independent price there is no
way to tell a scheme problem from a calibration problem.

The characteristic function is written in the "Little Heston Trap" form of
Albrecher et al. (2007). The textbook 1993 parameterisation takes a complex
logarithm that crosses its branch cut for moderate ``T``, which shows up as a
price that oscillates with maturity; this form moves the cut and is stable.

NumPy and SciPy rather than torch on purpose: this is the reference, and a
reference that shares an implementation with the thing it validates is not one.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def _cf_log_spot(u, spot, t, r, q, v0, kappa, theta, xi, rho):
    """``E[exp(i u log S_T)]`` under the risk-neutral measure."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    # g in the form whose |g| <= 1, keeping log() off its branch cut.
    g = (b - d) / (b + d)
    e = np.exp(-d * t)
    c_term = (kappa * theta / xi**2) * ((b - d) * t - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    d_term = (v0 / xi**2) * (b - d) * (1.0 - e) / (1.0 - g * e)
    return np.exp(iu * (np.log(spot) + (r - q) * t) + c_term + d_term)


def heston_price(
    spot, strike, t, r, q, v0, kappa, theta, xi, rho, right=1, limit: int = 200
) -> float:
    """European price under Heston. ``right`` is +1 for a call, -1 for a put."""
    cf = lambda u: _cf_log_spot(u, spot, t, r, q, v0, kappa, theta, xi, rho)
    log_k = np.log(strike)
    forward = spot * np.exp((r - q) * t)

    def integrand(u, shift):
        # shift=1 gives P1 (share measure), shift=0 gives P2 (risk-neutral).
        num = cf(u - 1j) / forward if shift else cf(u)
        return float(np.real(np.exp(-1j * u * log_k) * num / (1j * u)))

    p1 = 0.5 + quad(integrand, 0.0, np.inf, args=(1,), limit=limit)[0] / np.pi
    p2 = 0.5 + quad(integrand, 0.0, np.inf, args=(0,), limit=limit)[0] / np.pi

    call = spot * np.exp(-q * t) * p1 - strike * np.exp(-r * t) * p2
    if right > 0:
        return float(call)
    # Put-call parity, rather than a second pair of integrals.
    return float(call - spot * np.exp(-q * t) + strike * np.exp(-r * t))


def feller(kappa, theta, xi) -> float:
    """``2 kappa theta - xi^2``. Negative means the variance process reaches zero."""
    return float(2.0 * kappa * theta - xi**2)
