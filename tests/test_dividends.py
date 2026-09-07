"""Discrete cash dividends: the escrowed model, and where early exercise moves.

Phase B exists because a continuous yield is wrong for a single name in exactly
the place it matters -- an American call's early exercise is driven by the
discrete payment, and a yield smears away the discontinuity the decision turns
on. These tests pin both the plumbing and that behaviour.
"""

import dataclasses
import datetime as dt
import math

import numpy as np
import pytest
import torch

from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.curves import RateCurve
from torch_pricer.market.dividends import DividendSchedule
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.monte_carlo.engine import MCConfig, price
from torch_pricer.pricer.monte_carlo.lsm import (
    LSMConfig,
    exercise_indices,
    pre_dividend_indices,
)
from torch_pricer.pricer.tree.engine import price_lattice

AS_OF, EXPIRY = dt.date(2025, 1, 2), dt.date(2026, 1, 2)
SPOT, STRIKE, VOL, RATE = 100.0, 100.0, 0.20, 0.05
SCHEDULE = DividendSchedule((dt.date(2025, 4, 2), dt.date(2025, 10, 2)), (2.0, 2.0))


def _market(schedule=SCHEDULE):
    base = MarketSnapshot.flat(
        AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=0.0, flat_vol=VOL
    )
    return dataclasses.replace(base, dividends=schedule)


def _tree(right, style, market=None):
    market = market or _market()
    divs = tuple(zip(SCHEDULE.times(AS_OF, market.day_count), SCHEDULE.amounts))
    return price_lattice(SPOT, STRIKE, 1.0, VOL, RATE, 0.0, right, style,
                         n_steps=4096, dividends=divs).price


def _mc(right, style, seeds=range(4), n_steps=200, **lsm_kw):
    spec = VanillaOption(strike=STRIKE, maturity=EXPIRY, right=right, style=style)
    runs = [
        price(spec, _market(), BlackScholesModel(VOL),
              MCConfig(n_paths=60_000, n_steps=n_steps, seed=s),
              lsm=LSMConfig(**lsm_kw) if lsm_kw else None)
        for s in seeds
    ]
    return float(np.mean([r.price for r in runs])), float(np.mean([r.stderr for r in runs]))


# -- the schedule ---------------------------------------------------------


def test_schedule_validates_its_inputs():
    with pytest.raises(ValidationError, match="ex-dates but"):
        DividendSchedule((dt.date(2025, 4, 2),), (1.0, 2.0))
    with pytest.raises(ValidationError, match="non-negative"):
        DividendSchedule((dt.date(2025, 4, 2),), (-1.0,))
    with pytest.raises(ValidationError, match="ascending"):
        DividendSchedule((dt.date(2025, 6, 2), dt.date(2025, 4, 2)), (1.0, 1.0))


def test_pv_remaining_counts_only_dividends_still_ahead():
    times = SCHEDULE.times(AS_OF, "ACT/365F")
    curve = RateCurve.flat(RATE)
    t = torch.tensor([0.0, 0.5, 0.99], dtype=torch.float64)
    pv = SCHEDULE.pv_remaining(t, times, curve, 1.0)

    expected_0 = sum(2.0 * math.exp(-RATE * x) for x in times)
    expected_half = 2.0 * math.exp(-RATE * (times[1] - 0.5))
    assert float(pv[0]) == pytest.approx(expected_0, rel=1e-12)
    assert float(pv[1]) == pytest.approx(expected_half, rel=1e-12)
    assert float(pv[2]) == 0.0            # both paid by then


def test_an_empty_schedule_changes_nothing():
    spec = VanillaOption(strike=STRIKE, maturity=EXPIRY, right=Right.CALL,
                         style=Style.EUROPEAN)
    kw = dict(n_paths=20_000, n_steps=20, seed=3)
    with_empty = price(spec, _market(DividendSchedule()), BlackScholesModel(VOL),
                       MCConfig(**kw)).price
    without = price(spec, MarketSnapshot.flat(AS_OF, spot=SPOT, flat_rate=RATE,
                                              flat_dividend=0.0, flat_vol=VOL),
                    BlackScholesModel(VOL), MCConfig(**kw)).price
    assert with_empty == without


def test_dividends_at_or_above_spot_are_refused_by_the_lattice():
    with pytest.raises(ValidationError, match="at or above spot"):
        price_lattice(SPOT, STRIKE, 1.0, VOL, RATE, 0.0, "call", "european",
                      n_steps=64, dividends=((0.5, 200.0),))


# -- the simulation against the lattice -----------------------------------


@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
def test_european_simulation_matches_the_lattice(right):
    """Both sides implement the same escrowed model, so this tests the two
    implementations rather than laundering a model difference into agreement."""
    mc, se = _mc(right, Style.EUROPEAN)
    assert abs(mc - _tree(right.value, "european")) < 4 * se


def test_dividends_move_the_prices_the_right_way():
    call_with = _tree("call", "european")
    put_with = _tree("put", "european")
    no_div = dict(n_steps=4096, dividends=())
    assert call_with < price_lattice(SPOT, STRIKE, 1.0, VOL, RATE, 0.0,
                                     "call", "european", **no_div).price
    assert put_with > price_lattice(SPOT, STRIKE, 1.0, VOL, RATE, 0.0,
                                    "put", "european", **no_div).price


# -- where early exercise actually lives ----------------------------------


def test_pre_dividend_indices_land_strictly_before_the_ex_date():
    """The point of exercising is to capture a dividend, so a step at or after
    the ex-date is already too late."""
    times = SCHEDULE.times(AS_OF, "ACT/365F")
    idx = pre_dividend_indices(times, n_steps=200, horizon=1.0)
    assert idx == (49, 149)
    for i, time in zip(idx, times):
        assert i * (1.0 / 200) < time


def test_exercise_set_always_holds_expiry_and_never_inception():
    assert exercise_indices(50, 0) == [50]
    assert exercise_indices(50, 0, required=(10, 30)) == [10, 30, 50]
    assert 0 not in exercise_indices(50, None)


def test_a_dividend_paying_call_needs_the_ex_date_instants_and_nothing_else():
    """The Phase B result worth remembering.

    A call is optimally exercised only just before an ex-date. Restricting the
    exercise set to those instants recovers 93% of the true early-exercise
    premium; allowing exercise at every step recovers 74%, because each extra
    date is another chance for a noisy regression to exercise when it should not,
    and exercising a call early when it is not optimal destroys value outright.
    """
    american, european = _tree("call", "american"), _tree("call", "european")
    premium = american - european
    assert premium > 0.1                       # there is a premium to find

    aligned, _ = _mc(Right.CALL, Style.AMERICAN, basis_degree=4, n_exercise_dates=0)
    every_step, _ = _mc(Right.CALL, Style.AMERICAN, basis_degree=4,
                        n_exercise_dates=None)
    assert (aligned - european) / premium > 0.85
    assert aligned > every_step


def test_a_put_wants_the_opposite_and_that_is_not_a_contradiction():
    """A put's exercise region is rate-driven and open at all times, so it needs
    many dates and gains almost nothing from the ex-date instants alone: 94% of
    the premium with every step against -2% with the dividend dates only. The
    two rights genuinely want opposite settings.
    """
    american, european = _tree("put", "american"), _tree("put", "european")
    premium = american - european
    every_step, _ = _mc(Right.PUT, Style.AMERICAN, basis_degree=4,
                        n_exercise_dates=None)
    aligned, _ = _mc(Right.PUT, Style.AMERICAN, basis_degree=4, n_exercise_dates=0)
    assert (every_step - european) / premium > 0.85
    assert every_step > aligned


def test_american_call_still_matches_european_without_dividends():
    """The Phase A identity has to survive Phase B."""
    market = MarketSnapshot.flat(AS_OF, spot=SPOT, flat_rate=RATE,
                                 flat_dividend=0.0, flat_vol=VOL)
    spec = VanillaOption(strike=STRIKE, maturity=EXPIRY, right=Right.CALL,
                         style=Style.AMERICAN)
    res = price(spec, market, BlackScholesModel(VOL),
                MCConfig(n_paths=60_000, n_steps=50, seed=2))
    euro = price_lattice(SPOT, STRIKE, 1.0, VOL, RATE, 0.0, "call", "european",
                         n_steps=4096).price
    assert abs(res.price - euro) < 4 * res.stderr
