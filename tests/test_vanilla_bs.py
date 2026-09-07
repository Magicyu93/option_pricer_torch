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

from torch_pricer.pricer.analytic.black import black_price
from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.monte_carlo.engine import MCConfig, price

from .conftest import (
    DIV,
    EXPIRY,
    RATE,
    SPOT,
    VOL,
    analytic_greeks,
    analytic_inputs,
    averaged,
)

#: Log-space GBM is exact under Euler, so steps buy no accuracy here at all --
#: only memory, which is why this runs 5 and not the 20 it used to. See
#: examples/eu_bs_convergence.py: the fitted rate of error against step count is
#: -0.01, i.e. flat.
CONFIG = MCConfig(n_paths=100_000, n_steps=5, seed=7)
#: Same total sample as the single 400k run this replaces, taken as 8 independent
#: 50k runs so peak memory is an eighth of it. Used through ``averaged``.
GREEK_SEEDS = tuple(range(8))
GREEK_PATHS = 50_000
#: For the tests that compare two contracts under the *same* draws, where the
#: sampling error cancels and the sample size is nearly irrelevant.
PAIRED_CONFIG = MCConfig(n_paths=100_000, n_steps=5, seed=7)


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
    res = price(call, market, _model(), MCConfig(n_paths=50_000, n_steps=n_steps, seed=7))
    expected = float(black_price(fwd, call.strike, t, VOL, disc))
    assert abs(res.price - expected) < 3 * res.stderr


def test_deep_itm_call_is_the_forward(market, call):
    """A call struck at ~0 is the forward, discounted."""
    deep = VanillaOption(strike=1e-4, maturity=call.maturity, right=Right.CALL, style=Style.EUROPEAN)
    t, fwd, disc = analytic_inputs(market, deep)
    res = price(deep, market, _model(), CONFIG)
    assert abs(res.price - disc * (fwd - deep.strike)) < 4 * res.stderr


def test_reproducible_from_seed(market, call):
    assert price(call, market, _model(), CONFIG).price == (
        price(call, market, _model(), CONFIG).price
    )


# -- greeks --------------------------------------------------------------


#: Measured, not guessed: the worst relative error of the 8-seed mean over 16
#: independent groups, across six strike/right cases, on CUDA and CPU both --
#: delta 0.86%, vega 1.56%, theta 1.53%, rho 0.86%, gamma 3.97% -- then doubled
#: for the tail beyond 16 groups.
#:
#: The previous values (delta and rho 0.5%, theta 1%) were roughly half the
#: spread the estimator actually has, and passed only because seed 7 on CPU
#: happened to land well. CUDA seeds a different generator entirely, drew a
#: different sample, and failed three cases -- which was a latent flaky test
#: surfacing, not a device bug.
#:
#: These are sized to catch a structural error -- a dropped ``w``, a missing
#: carry factor, a sign -- which moves a greek by tens of percent. They do not
#: certify three-digit agreement with Black, and no Monte Carlo tolerance at
#: this sample size could.
_TOL = {"delta": 2e-2, "vega": 3e-2, "theta": 3e-2, "rho": 2e-2, "gamma": 8e-2}


@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_all_greeks_match_black(market, right, strike):
    """delta, vega, theta and rho from one backward pass; gamma differenced.

    Both rights, because the signs differ where it matters: put theta picks up
    ``+ r K D N(-d2)`` rather than losing it, and put rho is negative. A call-only
    test passes happily with the ``w`` factor dropped.
    """
    spec = VanillaOption(
        strike=strike, maturity=EXPIRY, right=right, style=Style.EUROPEAN
    )
    t = market.time_to(EXPIRY)
    ref = analytic_greeks(strike, t, right)
    mc_price, mc_stderr, greeks = averaged(
        lambda seed: price(
            spec, market, _model(),
            MCConfig(n_paths=GREEK_PATHS, n_steps=5, seed=seed),
            greeks=tuple(_TOL),
        ),
        GREEK_SEEDS,
    )

    assert abs(mc_price - ref["price"]) < 3 * mc_stderr
    for name, tol in _TOL.items():
        assert greeks[name] == pytest.approx(ref[name], rel=tol), name


def test_gamma_is_identical_for_call_and_put(market):
    """Put-call parity is linear in spot, so its second derivative vanishes."""
    kw = dict(strike=100.0, maturity=EXPIRY, style=Style.EUROPEAN)
    c = price(VanillaOption(right=Right.CALL, **kw), market, _model(),
              PAIRED_CONFIG, greeks=("gamma",))
    p = price(VanillaOption(right=Right.PUT, **kw), market, _model(),
              PAIRED_CONFIG, greeks=("gamma",))
    assert c.greeks["gamma"] == pytest.approx(p.greeks["gamma"], rel=1e-9)


@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
def test_gamma_autograd_is_exactly_zero_under_black_scholes(market, right):
    """Zero to floating-point roundoff, not merely small.

    ``S_T = S_0 M`` with ``M`` independent of ``S_0``, so ``max(S_0 M - K, 0)``
    is piecewise linear in spot and its second derivative is a Dirac that
    autograd evaluates as zero everywhere. What comes back is the residue of
    summing a great many exact zeros in floating point -- around 1e-18, i.e.
    1e-16 of the true gamma -- so the bound below is on that scale, not on any
    Monte Carlo error. This is why ``gamma`` is differenced instead, and why
    volga and vanna are not offered at all.
    """
    spec = VanillaOption(
        strike=100.0, maturity=EXPIRY, right=right, style=Style.EUROPEAN
    )
    res = price(spec, market, _model(), CONFIG, greeks=("gamma", "gamma_autograd"))
    assert abs(res.greeks["gamma_autograd"]) < 1e-12 * res.greeks["gamma"]
    assert res.greeks["gamma"] > 0.01  # the differenced one is fine


def test_put_call_delta_parity(market):
    """``delta_call - delta_put = D_q``, to Monte Carlo accuracy on the forward.

    Pathwise, the difference is ``D * mean(S_T / S_0)`` over the shared paths --
    exact against the *simulated* forward, but the simulated forward is not the
    analytic one. Antithetic sampling makes the driver sum to zero, not
    ``E[exp(sigma sqrt(T) z)]`` exact; see
    :class:`~torch_pricer.simulator.monte_carlo.rng.NormalDraws`.
    """
    kw = dict(strike=100.0, maturity=EXPIRY, style=Style.EUROPEAN)
    c = price(VanillaOption(right=Right.CALL, **kw), market, _model(),
              PAIRED_CONFIG, greeks=("delta",))
    p = price(VanillaOption(right=Right.PUT, **kw), market, _model(),
              PAIRED_CONFIG, greeks=("delta",))
    t = market.time_to(EXPIRY)
    assert c.greeks["delta"] - p.greeks["delta"] == pytest.approx(
        math.exp(-DIV * t), rel=1e-3
    )
