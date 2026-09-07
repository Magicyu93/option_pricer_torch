"""The Andersen-Broadie dual bound: does the bracket actually bracket?

Longstaff-Schwartz alone yields a lower bound with an unknown shortfall. These
tests check the two properties that make the dual worth its cost -- the bracket
contains the truth, and its width tracks how good the exercise policy is -- plus
the one failure mode that would quietly invalidate it.
"""

import datetime as dt

import pytest

from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.monte_carlo.dual import Bracket, DualConfig
from torch_pricer.pricer.monte_carlo.engine import MCConfig, price_bracket
from torch_pricer.pricer.monte_carlo.lsm import LSMConfig

AS_OF, EXPIRY = dt.date(2025, 1, 2), dt.date(2026, 1, 2)
SPOT, STRIKE, VOL, RATE = 100.0, 100.0, 0.20, 0.05

#: The lattice and the PDE solver agree here to 2.6e-4, from opposite sides.
TRUE_PRICE = 6.09046


def _market():
    return MarketSnapshot.flat(
        AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=0.0, flat_vol=VOL
    )


def _put(style=Style.AMERICAN):
    return VanillaOption(strike=STRIKE, maturity=EXPIRY, right=Right.PUT, style=style)


def _bracket(degree=3, n_inner=400, n_outer=800, n_steps=25, seed=1):
    return price_bracket(
        _put(), _market(), BlackScholesModel(VOL),
        MCConfig(n_paths=40_000, n_steps=n_steps, seed=seed),
        lsm=LSMConfig(basis_degree=degree),
        dual=DualConfig(n_outer=n_outer, n_inner=n_inner),
    )


def test_bracket_reports_its_own_width():
    b = Bracket(low=6.0, high=6.2, low_stderr=0.01, high_stderr=0.02)
    assert b.gap == pytest.approx(0.2)


def test_a_european_contract_has_no_bracket():
    """There is no exercise policy to be suboptimal about."""
    with pytest.raises(ValidationError, match="American vanilla"):
        price_bracket(_put(style=Style.EUROPEAN), _market(), BlackScholesModel(VOL))


def test_the_bracket_contains_the_true_price():
    """The whole point: a two-sided statement, checked where the truth is known."""
    b = _bracket()
    assert b.low < b.high
    assert b.low - 3 * b.low_stderr <= TRUE_PRICE <= b.high + 3 * b.high_stderr


def test_the_gap_narrows_as_the_exercise_policy_improves():
    """The gap measures policy suboptimality, so a better basis must close it.

    Measured at 200 inner paths: 0.282, 0.150, 0.125 for degrees 1, 2 and 3,
    flattening by degree 5 -- which is the evidence behind LSMConfig's default of
    three, and the reason this diagnostic is worth having where no lattice exists
    to check against.
    """
    poor, better = _bracket(degree=1), _bracket(degree=3)
    assert poor.gap > better.gap
    assert poor.low < better.low          # a worse policy earns less
    assert poor.high > better.high        # and bounds it more loosely


def test_too_few_inner_paths_inflate_the_upper_bound():
    """The failure mode that would otherwise be mistaken for a bad policy.

    The conditional expectation is estimated by nesting, and its noise survives
    the maximum taken inside the outer expectation -- so it cannot average out
    and pushes the bound up. Measured excess over the true price: +0.683, +0.158
    and +0.026 at 25, 100 and 400 inner paths, roughly halving per doubling.

    The direction is the safe one -- the bracket stays valid -- but a gap read
    without this in mind blames the basis for the nesting.
    """
    coarse, fine = _bracket(n_inner=25), _bracket(n_inner=400)
    assert coarse.high > fine.high
    assert coarse.high > TRUE_PRICE          # still a valid bound, just loose
    assert fine.high > TRUE_PRICE - 3 * fine.high_stderr


def test_the_upper_bound_stays_above_the_lower_one_across_policies():
    for degree in (1, 3):
        b = _bracket(degree=degree, n_inner=200, n_outer=500)
        assert b.high > b.low
