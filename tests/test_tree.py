"""The CRR lattice, and the early exercise Monte Carlo cannot reach.

This is the reference the Longstaff-Schwartz engine will be checked against, so
it is worth more than a smoke test. The assertions below are chosen to be sharp
-- several are exact identities rather than tolerances -- because a reference
that is only approximately right validates nothing.
"""

import math

import pytest

from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Right, Style
from torch_pricer.pricer.analytic.black import black_delta, black_gamma, black_price
from torch_pricer.pricer.tree.engine import price_lattice
from torch_pricer.simulator.tree.crr import build

SPOT, STRIKE, T, VOL, RATE, DIV = 100.0, 100.0, 1.0, 0.20, 0.05, 0.0


def _black(strike=STRIKE, t=T, vol=VOL, rate=RATE, div=DIV, right=1):
    fwd, disc = SPOT * math.exp((rate - div) * t), math.exp(-rate * t)
    return float(black_price(fwd, strike, t, vol, disc, right))


# -- the lattice itself ---------------------------------------------------


def test_lattice_recombines():
    """An n-step tree has n + 1 terminal nodes, not 2^n."""
    lat = build(SPOT, T, VOL, RATE, DIV, n_steps=50)
    assert len(lat.levels(50)) == 51
    assert lat.levels(0) == pytest.approx([SPOT])
    # Centred: an up then a down returns exactly to spot.
    assert lat.u * lat.d == pytest.approx(1.0, rel=1e-15)


def test_rejects_a_step_too_coarse_for_its_drift():
    """p outside [0, 1] is rejected, not clamped: it is a negative probability,
    not an inaccuracy, and clamping would return a plausible wrong number."""
    with pytest.raises(ValidationError, match="outside \\[0, 1\\]"):
        build(SPOT, T, vol=0.05, rate=2.0, dividend=0.0, n_steps=2)


def test_rejects_bermudan():
    with pytest.raises(ValidationError, match="Bermudan"):
        price_lattice(SPOT, STRIKE, T, VOL, RATE, style=Style.BERMUDAN)


# -- European: against the closed form ------------------------------------


@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
def test_european_converges_to_black(right):
    w = float(right.sign)
    ref = _black(right=w)
    errs = [
        abs(price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, right, "european",
                          n_steps=n).price - ref)
        for n in (64, 256, 1024)
    ]
    # Measured, averaging adjacent counts: 2.1e-3, 4.9e-4, 1.2e-4 -- a clean
    # factor of ~4 per 4x the steps, i.e. first order in dt.
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 2e-4


def test_european_put_call_parity_on_the_tree():
    """``C - P = S e^{-qT} - K e^{-rT}``, exactly -- both legs use one lattice,
    so this catches a sign or discounting error with no tolerance to argue."""
    kw = dict(n_steps=257, average_adjacent=False)
    c = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "call", "european", **kw).price
    p = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "european", **kw).price
    parity = SPOT * math.exp(-DIV * T) - STRIKE * math.exp(-RATE * T)
    assert (c - p) == pytest.approx(parity, abs=1e-10)


def test_greeks_off_the_lattice_match_black():
    fwd, disc = SPOT * math.exp((RATE - DIV) * T), math.exp(-RATE * T)
    carry = math.exp((RATE - DIV) * T)
    res = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "call", "european",
                        n_steps=2048, greeks=True)
    assert res.delta == pytest.approx(
        float(black_delta(fwd, STRIKE, T, VOL, disc, 1)) * carry, rel=1e-3)
    assert res.gamma == pytest.approx(
        float(black_gamma(fwd, STRIKE, T, VOL, disc)) * carry**2, rel=1e-3)


def test_averaging_adjacent_counts_damps_the_oscillation():
    """CRR straddles the true price with the parity of the step count."""
    ref = _black()
    raw = [
        price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "call", "european",
                      n_steps=n, average_adjacent=False).price - ref
        for n in (200, 201)
    ]
    assert raw[0] * raw[1] < 0          # opposite sides
    averaged = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "call", "european",
                             n_steps=200).price - ref
    assert abs(averaged) < min(abs(r) for r in raw)


# -- American: what the Monte Carlo engine cannot price -------------------


@pytest.mark.parametrize("n_steps", [64, 512])
def test_american_call_without_dividends_equals_european(n_steps):
    """The sharp one. With ``q = 0`` it is never optimal to exercise a call
    early, so the exercise test never binds and the two roll back through
    identical arithmetic. Exact, not approximate -- a tolerance here would hide
    exactly the bug it is meant to catch."""
    kw = dict(n_steps=n_steps, average_adjacent=False)
    am = price_lattice(SPOT, STRIKE, T, VOL, RATE, 0.0, "call", "american", **kw)
    eu = price_lattice(SPOT, STRIKE, T, VOL, RATE, 0.0, "call", "european", **kw)
    assert am.price == eu.price


def test_american_call_with_dividends_is_worth_more():
    """Restore the dividend and the early exercise premium appears."""
    kw = dict(n_steps=512, average_adjacent=False)
    am = price_lattice(SPOT, STRIKE, T, VOL, RATE, 0.08, "call", "american", **kw)
    eu = price_lattice(SPOT, STRIKE, T, VOL, RATE, 0.08, "call", "european", **kw)
    assert am.price > eu.price * (1 + 1e-6)


def test_american_put_is_worth_more_than_european():
    """A put's early exercise is driven by rates, so it has value at q = 0."""
    kw = dict(n_steps=512, average_adjacent=False)
    am = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american", **kw)
    eu = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "european", **kw)
    assert am.price > eu.price * (1 + 1e-6)
    assert am.price >= max(STRIKE - SPOT, 0.0)      # never below intrinsic


def test_deep_itm_american_put_is_its_intrinsic_value():
    """Far enough in the money, waiting is worthless and the option is exercised
    at once -- so the tree must return exactly ``K - S``."""
    res = price_lattice(1.0, 100.0, T, VOL, RATE, DIV, "put", "american",
                        n_steps=256, average_adjacent=False)
    assert res.price == pytest.approx(99.0, abs=1e-9)


def test_american_parity_bounds_hold():
    """Parity is an inequality for American options, and it brackets tightly:
    ``S - K <= C - P <= S - K e^{-rT}`` when there are no dividends.

    This is the bound that stops the parity-implied forward trick from working
    on single names, so it is worth pinning that the tree respects it.
    """
    kw = dict(n_steps=512, average_adjacent=False)
    c = price_lattice(SPOT, STRIKE, T, VOL, RATE, 0.0, "call", "american", **kw).price
    p = price_lattice(SPOT, STRIKE, T, VOL, RATE, 0.0, "put", "american", **kw).price
    assert SPOT - STRIKE <= (c - p) + 1e-9
    assert (c - p) <= SPOT - STRIKE * math.exp(-RATE * T) + 1e-9


def test_american_put_converges():
    """No closed form to check against, so pin the shape of the convergence.

    Measured here: 6.091615, 6.090706, 6.090545, 6.090458 at 256/1024/2048/4096
    steps -- settling near 6.0904, which is the value this parameter set is
    usually quoted at. Treat that as corroboration, not proof; the real
    cross-check is the PSOR solver, which shares none of this arithmetic.
    """
    prices = [
        price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                      n_steps=n).price
        for n in (256, 1024, 2048)
    ]
    assert prices[0] > prices[1] > prices[2]        # monotone from above
    assert abs(prices[-1] - prices[-2]) < 5e-4
    assert prices[-1] == pytest.approx(6.0904, abs=2e-3)
