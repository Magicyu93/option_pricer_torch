import datetime as dt
import math

import pytest
import torch

from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot

AS_OF = dt.date(2025, 1, 2)
EXPIRY = dt.date(2026, 1, 2)
SPOT, RATE, DIV, VOL = 100.0, 0.03, 0.01, 0.20


@pytest.fixture(autouse=True)
def _release_cuda_memory():
    """Hand freed blocks back to the driver between tests.

    Torch's caching allocator keeps freed blocks for reuse, which is the right
    default in a training loop and the wrong one here: these tests run a dozen
    unrelated simulations of very different shapes in one process, and the cache
    fragments until an allocation that fits the card cannot find contiguous room.
    Without this, tests fail in file order rather than by cost -- whatever runs
    last inherits an exhausted device, and a one-line reproducibility check OOMs
    while the 200k-path simulation before it passed.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def averaged(price_fn, seeds):
    """Mean of ``price_fn(seed)`` over independent seeds.

    Peak memory is one run's; accuracy is the pooled sample's. ``k`` seeds at
    ``n/k`` paths is statistically the same estimator as one run at ``n`` --
    measured, not assumed: 8 x 50k reproduces 1 x 400k to within the spread of
    either -- while allocating a ``k``-th as much at once. That is the whole
    trick for keeping an accuracy assertion honest on a card that cannot hold
    the sample it needs.
    """
    results = [price_fn(s) for s in seeds]
    k = len(results)
    price = sum(r.price for r in results) / k

    # Independent runs, so the variances add and the mean's standard error is
    # sqrt(sum se^2)/k. Returned rather than left to the caller so an assertion
    # can stay scaled to the sampling error, the way it was before the split.
    stderr = math.sqrt(sum(r.stderr**2 for r in results)) / k

    def _mean(name):
        first = results[0].greeks[name]
        if isinstance(first, dict):  # model_params: one entry per parameter
            return {
                key: sum(float(r.greeks[name][key]) for r in results) / k
                for key in first
            }
        return sum(float(r.greeks[name]) for r in results) / k

    return price, stderr, {n: _mean(n) for n in results[0].greeks}


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
    from torch_pricer.pricer.analytic.black import black_delta, black_gamma, black_price, black_vega

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
