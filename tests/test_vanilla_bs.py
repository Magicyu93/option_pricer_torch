"""The reference test: European vanilla under constant r, q and sigma.

Euler-Maruyama on log-spot is *exact* for geometric Brownian motion, so there is
no discretisation bias between this engine and the Black formula. Any
disagreement beyond Monte Carlo error is a bug, which is what makes this the
gate every other change is measured against.

Tolerances are set from the estimators' measured single-seed spread at these
sample sizes, not guessed: price and rho are tight, pathwise vega has a ~0.4%
spread, and gamma -- differenced rather than differentiated -- has ~1.5%.
"""

import math

import pytest

from torch_pricer.black_formula import black_delta, black_gamma, black_price, black_vega
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.engine import MCConfig, price

from .conftest import DIV, RATE, SPOT, VOL, analytic_inputs

CONFIG = MCConfig(n_paths=200_000, n_steps=20, seed=7)


def _model():
    return BlackScholesModel(VOL)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# -- price ---------------------------------------------------------------


def test_price_matches_black(market, call):
    t, fwd, disc = analytic_inputs(market, call)
    res = price(call, market, _model(), CONFIG)
    expected = float(black_price(fwd, call.strike, t, VOL, disc))
    assert abs(res.price - expected) < 3 * res.stderr
    assert res.stderr > 0


def test_put_call_parity(market, call, put):
    """``C - P = D (F - K)``, to Monte Carlo accuracy on the forward."""
    t, fwd, disc = analytic_inputs(market, call)
    c = price(call, market, _model(), CONFIG)
    p = price(put, market, _model(), CONFIG)
    assert abs((c.price - p.price) - disc * (fwd - call.strike)) < 4 * (
        c.stderr + p.stderr
    )


@pytest.mark.parametrize("n_steps", [1, 5, 50, 200])
def test_step_count_invariance(market, call, n_steps):
    """Log-space GBM is exact under Euler, so the step count changes only the
    Brownian path -- never the price beyond sampling error.

    This is the assertion that catches a wrong initial state, a draws axis
    indexed as paths instead of steps, or a time grid the simulator disagrees
    with.
    """
    t, fwd, disc = analytic_inputs(market, call)
    res = price(call, market, _model(), MCConfig(n_paths=200_000, n_steps=n_steps, seed=7))
    expected = float(black_price(fwd, call.strike, t, VOL, disc))
    assert abs(res.price - expected) < 3 * res.stderr


def test_deep_itm_call_is_the_forward(market, call):
    """A call struck at ~0 is the forward, discounted."""
    from torch_pricer.instruments.spec import Right, VanillaOption

    deep = VanillaOption(strike=1e-4, maturity=call.maturity, right=Right.CALL)
    t, fwd, disc = analytic_inputs(market, deep)
    res = price(deep, market, _model(), CONFIG)
    assert abs(res.price - disc * (fwd - deep.strike)) < 4 * res.stderr


def test_reproducible_from_seed(market, call):
    assert price(call, market, _model(), CONFIG).price == (
        price(call, market, _model(), CONFIG).price
    )


# -- greeks --------------------------------------------------------------


def test_pathwise_greeks_match_black(market, call):
    """delta, vega, theta and rho all come off one backward pass."""
    t, fwd, disc = analytic_inputs(market, call)
    res = price(call, market, _model(), CONFIG, greeks=("delta", "vega", "theta", "rho"))

    d1 = (math.log(fwd / call.strike) + 0.5 * VOL**2 * t) / (VOL * math.sqrt(t))
    d2 = d1 - VOL * math.sqrt(t)

    # black_delta is with respect to the forward; dF/dS = D_q/D_r.
    assert res.greeks["delta"] == pytest.approx(
        float(black_delta(fwd, call.strike, t, VOL, disc)) * math.exp((RATE - DIV) * t),
        rel=5e-3,
    )
    assert res.greeks["vega"] == pytest.approx(
        float(black_vega(fwd, call.strike, t, VOL, disc)), rel=1.5e-2
    )
    theta = (
        -SPOT * math.exp(-DIV * t) * _norm_pdf(d1) * VOL / (2 * math.sqrt(t))
        + DIV * SPOT * math.exp(-DIV * t) * _norm_cdf(d1)
        - RATE * call.strike * math.exp(-RATE * t) * _norm_cdf(d2)
    )
    assert res.greeks["theta"] == pytest.approx(theta, rel=1e-2)
    assert float(res.greeks["rho"]) == pytest.approx(
        call.strike * t * math.exp(-RATE * t) * _norm_cdf(d2), rel=5e-3
    )


def test_gamma_by_bumping_matches_black(market, call):
    """Gamma is differenced, not differentiated: the second derivative of a kink
    is a Dirac, so a second backward pass returns exactly zero."""
    t, fwd, disc = analytic_inputs(market, call)
    res = price(
        call, market, _model(), MCConfig(n_paths=400_000, n_steps=20, seed=7),
        greeks=("gamma",),
    )
    spot_gamma = float(black_gamma(fwd, call.strike, t, VOL, disc)) * math.exp(
        2 * (RATE - DIV) * t
    )
    assert res.greeks["gamma"] == pytest.approx(spot_gamma, rel=5e-2)


def test_put_greeks_match_black(market, put):
    t, fwd, disc = analytic_inputs(market, put)
    res = price(put, market, _model(), CONFIG, greeks=("delta", "vega"))
    assert res.greeks["delta"] == pytest.approx(
        float(black_delta(fwd, put.strike, t, VOL, disc, right=-1))
        * math.exp((RATE - DIV) * t),
        rel=5e-3,
    )
    # Vega is identical for calls and puts.
    assert res.greeks["vega"] == pytest.approx(
        float(black_vega(fwd, put.strike, t, VOL, disc)), rel=1.5e-2
    )
