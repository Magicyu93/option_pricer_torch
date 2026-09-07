import datetime as dt
import math

import pytest

from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot

AS_OF = dt.date(2025, 1, 2)
EXPIRY = dt.date(2026, 1, 2)
SPOT, RATE, DIV, VOL = 100.0, 0.03, 0.01, 0.20


@pytest.fixture
def market() -> MarketSnapshot:
    return MarketSnapshot.flat(
        AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=DIV, flat_vol=VOL
    )


@pytest.fixture
def call() -> VanillaOption:
    return VanillaOption(strike=100.0, maturity=EXPIRY, right=Right.CALL, style=Style.EUROPEAN)


@pytest.fixture
def put() -> VanillaOption:
    return VanillaOption(strike=100.0, maturity=EXPIRY, right=Right.PUT, style=Style.EUROPEAN)


def analytic_inputs(market, spec):
    """``(T, forward, discount)`` for the flat market fixture."""
    t = market.time_to(spec.maturity)
    return t, SPOT * math.exp((RATE - DIV) * t), math.exp(-RATE * t)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def analytic_greeks(strike: float, t: float, right: Right) -> dict[str, float]:
    """Black-Scholes greeks in *spot* terms, for either right.

    ``black_delta`` and ``black_gamma`` are with respect to the forward, so both
    need ``dF/dS = D_q / D_r`` -- once for delta, squared for gamma. Theta is
    ``d/dt`` in calendar time, hence opposite in sign to the ``d/dT`` the engine
    differentiates.
    """
    from torch_pricer.black_formula import black_delta, black_gamma, black_price, black_vega

    w = float(right.sign)
    fwd, disc = SPOT * math.exp((RATE - DIV) * t), math.exp(-RATE * t)
    carry = math.exp((RATE - DIV) * t)
    d1 = (math.log(fwd / strike) + 0.5 * VOL**2 * t) / (VOL * math.sqrt(t))
    d2 = d1 - VOL * math.sqrt(t)
    return {
        "price": float(black_price(fwd, strike, t, VOL, disc, w)),
        "delta": float(black_delta(fwd, strike, t, VOL, disc, w)) * carry,
        "gamma": float(black_gamma(fwd, strike, t, VOL, disc)) * carry**2,
        "vega": float(black_vega(fwd, strike, t, VOL, disc)),
        "theta": (
            -SPOT * math.exp(-DIV * t) * _norm_pdf(d1) * VOL / (2.0 * math.sqrt(t))
            + w * (
                DIV * SPOT * math.exp(-DIV * t) * _norm_cdf(w * d1)
                - RATE * strike * math.exp(-RATE * t) * _norm_cdf(w * d2)
            )
        ),
        "rho": w * strike * t * math.exp(-RATE * t) * _norm_cdf(w * d2),
    }
