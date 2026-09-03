"""The simplest case, end to end: a European option with constant r, q and sigma.

Every number the Monte Carlo engine produces here has a closed form to be judged
against, which is the point -- this is the test that says the seams between
model, simulator, payoff and engine are wired up correctly.
"""

from __future__ import annotations

import datetime as dt

import pytest

import torch_pricer as tp

from conftest import MATURITY, VOL, analytic


@pytest.mark.parametrize("right,strike", [(1, 100.0), (-1, 100.0), (1, 120.0), (-1, 80.0)])
def test_price_matches_black(market, config, right, strike):
    """MC price sits within three standard errors of Black."""
    spec = tp.VanillaOption(strike=strike, maturity=MATURITY, right="call" if right > 0 else "put")
    result = tp.price(spec, market, config)
    expected = analytic(market, strike, right)["price"]
    assert abs(result.price - expected) < 3 * result.stderr


def test_autograd_delta_and_vega_match_black(market, config):
    """Delta and vega are pathwise estimators: unbiased, and tight at this sample size."""
    spec = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    got = tp.price(spec, market, config, greeks=("delta", "vega")).greeks
    want = analytic(market, 100.0, 1)

    assert got["delta"] == pytest.approx(want["delta"], abs=2e-3)
    assert got["vega"] == pytest.approx(want["vega"], rel=1e-2)


def test_gamma_matches_black(market, config):
    """Gamma comes from differencing the pathwise delta, so it earns a looser band.

    The pathwise second derivative of a kinked payoff is identically zero -- see
    the engine module docstring -- and the difference estimator that replaces it
    carries both a bump bias and a near-binomial sampling error.
    """
    spec = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    got = tp.price(spec, market, config, greeks=("gamma",)).greeks["gamma"]
    assert got == pytest.approx(analytic(market, 100.0, 1)["gamma"], rel=0.05)


def test_rho_matches_black(market, config):
    """Rho is d/dr through both the drift and the discount factor: ``K T D N(d2)``."""
    from torch_pricer.analytics.black import _norm_cdf, d1_d2

    strike = 100.0
    spec = tp.VanillaOption(strike=strike, maturity=MATURITY, right="call")
    got = tp.price(spec, market, config, greeks=("rho",)).greeks["rho"]

    t = market.time_to(MATURITY)
    _, d2 = d1_d2(market.forward(t), strike, t, VOL)
    want = strike * t * float(market.discount.discount(t)) * float(_norm_cdf(d2))
    assert got == pytest.approx(want, rel=1e-2)


def test_put_call_parity(market, config):
    """``C - P = D (F - K)``, to the Monte Carlo error in the forward."""
    call = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    put = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="put")
    c = tp.price(call, market, config).price
    p = tp.price(put, market, config).price

    t = market.time_to(MATURITY)
    want = float(market.discount.discount(t)) * (float(market.forward(t)) - 100.0)
    assert c - p == pytest.approx(want, abs=0.15)


def test_log_space_scheme_has_no_discretisation_bias(market):
    """One step and a hundred steps agree: Euler on log-spot is the exact transition law."""
    spec = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    coarse = tp.price(spec, market, tp.MCConfig(n_paths=200_000, n_steps=1, seed=3))
    fine = tp.price(spec, market, tp.MCConfig(n_paths=200_000, n_steps=100, seed=3))
    tol = 3 * (coarse.stderr**2 + fine.stderr**2) ** 0.5
    assert abs(coarse.price - fine.price) < tol


def test_price_is_reproducible_from_the_seed(market, config):
    spec = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    assert tp.price(spec, market, config).price == tp.price(spec, market, config).price


def test_deep_out_of_the_money_is_worth_almost_nothing(market, config):
    spec = tp.VanillaOption(strike=400.0, maturity=MATURITY, right="call")
    result = tp.price(spec, market, config)
    assert 0.0 <= result.price < 0.05


def test_expired_and_unknown_greeks_are_rejected(market, config):
    spec = tp.VanillaOption(strike=100.0, maturity=dt.date(2025, 1, 2), right="call")
    with pytest.raises(tp.ValidationError, match="expires on or before"):
        tp.price(spec, market, config)

    live = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    with pytest.raises(tp.ValidationError, match="unknown greeks"):
        tp.price(live, market, config, greeks=("theta",))


def test_asian_is_cheaper_than_the_vanilla(market, config):
    """Averaging cuts the terminal variance, so an Asian call is worth less."""
    vanilla = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    asian = tp.AsianOption(strike=100.0, maturity=MATURITY, right="call")
    cheap = tp.MCConfig(n_paths=20_000, n_steps=25, seed=7, device="cpu")
    assert tp.price(asian, market, cheap).price < tp.price(vanilla, market, cheap).price


def test_maturity_is_required():
    """``dt.date.today()`` as a default would be bound once, at import."""
    with pytest.raises(tp.ValidationError, match="needs a maturity"):
        tp.VanillaOption(strike=100.0)
