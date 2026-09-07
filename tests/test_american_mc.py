"""Longstaff-Schwartz, against the two references that do not share its arithmetic.

Every assertion here is anchored on the lattice or the PDE solver, never on
another Monte Carlo run. The interesting ones are about *bias*: LSM is not an
unbiased estimator of the American price, it is biased in two directions at once
for two different reasons, and both are pinned below.
"""

import datetime as dt
import math

import numpy as np
import pytest

from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.monte_carlo.engine import MCConfig, price
from torch_pricer.pricer.monte_carlo.lsm import LSMConfig, exercise_indices
from torch_pricer.pricer.pde.engine import price_pde
from torch_pricer.pricer.tree.engine import price_lattice

AS_OF, EXPIRY = dt.date(2025, 1, 2), dt.date(2026, 1, 2)
SPOT, STRIKE, VOL, RATE, DIV, T = 100.0, 100.0, 0.20, 0.05, 0.0, 1.0

#: Reference American put: the lattice and the PDE agree here to 2.6e-4, from
#: opposite sides. See tests/test_pde.py.
AMERICAN_PUT = 6.09046


def _market():
    return MarketSnapshot.flat(
        AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=DIV, flat_vol=VOL
    )


def _spec(right=Right.PUT, style=Style.AMERICAN):
    return VanillaOption(strike=STRIKE, maturity=EXPIRY, right=right, style=style)


def _price(spec, seed=0, n_paths=40_000, n_steps=50, greeks=(), lsm=None):
    return price(
        spec, _market(), BlackScholesModel(VOL),
        MCConfig(n_paths=n_paths, n_steps=n_steps, seed=seed),
        greeks=greeks, lsm=lsm or LSMConfig(),
    )


def _mean(spec, seeds, **kw):
    runs = [_price(spec, seed=s, **kw) for s in seeds]
    return float(np.mean([r.price for r in runs])), float(np.std([r.price for r in runs]))


# -- the exercise grid ----------------------------------------------------


def test_exercise_indices_never_include_inception():
    """Exercising at t=0 is a decision made before buying, not one to price."""
    assert exercise_indices(10, None) == list(range(1, 11))
    assert 0 not in exercise_indices(50, 5)
    assert exercise_indices(50, 5)[-1] == 50


def test_bermudan_style_is_rejected_with_a_route_forward():
    """A valid Bermudan spec -- it carries its dates -- still cannot be priced,
    because those dates are not aligned to the simulation grid. The message says
    what to use instead rather than just refusing."""
    spec = VanillaOption(
        strike=STRIKE, maturity=EXPIRY, right=Right.PUT, style=Style.BERMUDAN,
        exercise_dates=(dt.date(2025, 7, 1), dt.date(2025, 10, 1)),
    )
    with pytest.raises(ValidationError, match="n_exercise_dates"):
        _price(spec)


# -- against the references -----------------------------------------------


def test_american_call_without_dividends_does_not_exercise_early():
    """With ``q = 0`` early exercise is never optimal, so a correct policy must
    reproduce the European price. This is the sharpest check that the regression
    is not inventing exercise where there is none."""
    euro = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "call", "european",
                         n_steps=4096).price
    mc = _price(_spec(right=Right.CALL), seed=1, n_paths=80_000)
    assert abs(mc.price - euro) < 4 * mc.stderr


def test_american_put_is_worth_more_than_european():
    am, _ = _mean(_spec(), range(4))
    eu, _ = _mean(_spec(style=Style.EUROPEAN), range(4))
    assert am > eu


def test_american_put_matches_the_lattice():
    """Low-biased, so it is allowed below the reference but not above it by more
    than noise."""
    mean, sd = _mean(_spec(), range(6), n_paths=60_000, n_steps=100)
    stderr = sd / math.sqrt(6)
    assert mean < AMERICAN_PUT + 4 * stderr        # never materially above
    assert mean > AMERICAN_PUT - 0.05              # and close from below


def test_more_exercise_dates_raise_the_price_toward_the_american_limit():
    """A Bermudan with more dates is worth more, converging on the continuous
    right from below. Measured at 40k paths: -0.106, -0.058, -0.031, -0.009 for
    5, 10, 25 and 50 dates.
    """
    means = [_mean(_spec(), range(4), n_steps=nd)[0] for nd in (5, 10, 50)]
    assert means[0] < means[1] < means[2]
    assert means[-1] < AMERICAN_PUT + 0.02


# -- greeks through a detached policy -------------------------------------


def test_delta_matches_the_reference_methods():
    """The envelope-theorem argument, tested.

    The exercise decision is an indicator and its derivative is a Dirac -- the
    same one that makes ``gamma_autograd`` zero. Differentiating with the policy
    held fixed is valid because at the optimal boundary the holder is indifferent,
    so the boundary's own movement contributes nothing to first order. If that
    reasoning were wrong, delta would come back visibly biased rather than within
    a percent of two independent methods.
    """
    tree = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                         n_steps=4096, greeks=True)
    pde = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                    n_space=2048, n_time=1024, greeks=True)
    assert tree.delta == pytest.approx(pde.delta, rel=1e-3)   # references agree

    runs = [_price(_spec(), seed=s, n_paths=60_000, n_steps=100, greeks=("delta",))
            for s in range(4)]
    delta = float(np.mean([r.greeks["delta"] for r in runs]))
    assert delta < 0                                          # a put's delta
    assert delta == pytest.approx(tree.delta, rel=2e-2)


def test_gamma_is_differenced_and_still_lands():
    """Gamma cannot come from a second backward pass here any more than it can
    for a European payoff, so it is differenced -- through the same fixed policy
    at both bumped spots, which keeps the two sides correlated."""
    tree = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                         n_steps=4096, greeks=True)
    runs = [_price(_spec(), seed=s, n_paths=60_000, n_steps=100, greeks=("gamma",))
            for s in range(4)]
    gamma = float(np.mean([r.greeks["gamma"] for r in runs]))
    assert gamma == pytest.approx(tree.gamma, rel=8e-2)


# -- the bias that the path split exists to remove ------------------------


def test_fitting_on_the_pricing_paths_biases_the_price_up():
    """Foresight bias, made visible.

    Regressing on the same paths the policy is then applied to lets it exercise
    with knowledge of each path's own future. The bias is upward, and it shrinks
    as paths grow.

    Measured at 2,000 paths over 20 seeds, on both devices: in-sample runs +0.089
    (CUDA +0.087) above the true 6.0905, which is 4.3 and 3.9 standard errors --
    while out-of-sample sits +0.010, half a standard error out, on both. Twenty
    seeds rather than ten because the single-run spread at 2,000 paths is 0.1, so
    ten leaves the effect inside its own sampling noise on an unlucky draw.
    """
    seeds = range(20)
    n = len(list(seeds))
    in_sample, sd_in = _mean(
        _spec(), seeds, n_paths=2_000, lsm=LSMConfig(policy_paths=None))
    out_sample, sd_out = _mean(
        _spec(), seeds, n_paths=2_000, lsm=LSMConfig(policy_paths=20_000))
    se_in, se_out = sd_in / math.sqrt(n), sd_out / math.sqrt(n)

    assert in_sample > out_sample
    # And it is the in-sample one that is wrong, not merely different.
    assert in_sample > AMERICAN_PUT + 2 * se_in
    assert out_sample == pytest.approx(AMERICAN_PUT, abs=4 * se_out)
