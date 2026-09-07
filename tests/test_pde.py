"""Crank-Nicolson with PSOR, and the tree it has to agree with.

The point of this module is the cross-check. A lattice converging to a value
proves only that the lattice agrees with itself on finer grids; this solver
discretizes a different formulation of the problem with different arithmetic, so
where the two land on the same number that number is evidence.
"""

import math

import numpy as np
import pytest

from torch_pricer.errors import PricingError, ValidationError
from torch_pricer.instruments.spec import Right, Style
from torch_pricer.pricer.analytic.black import black_delta, black_gamma, black_price
from torch_pricer.pricer.pde.engine import price_pde
from torch_pricer.pricer.tree.engine import price_lattice
from torch_pricer.simulator.pde.psor import solve as psor_solve

SPOT, STRIKE, T, VOL, RATE, DIV = 100.0, 100.0, 1.0, 0.20, 0.05, 0.0


def _black(right=1, strike=STRIKE, t=T, rate=RATE, div=DIV):
    fwd, disc = SPOT * math.exp((rate - div) * t), math.exp(-rate * t)
    return float(black_price(fwd, strike, t, VOL, disc, right))


# -- the solver ------------------------------------------------------------


def test_psor_unconstrained_matches_a_dense_solve():
    n, lo, di, up = 200, -1.0, 2.5, -1.0
    rhs = np.random.default_rng(0).normal(size=n)
    dense = (np.diag(np.full(n, di)) + np.diag(np.full(n - 1, up), 1)
             + np.diag(np.full(n - 1, lo), -1))
    assert psor_solve(lo, di, up, rhs) == pytest.approx(
        np.linalg.solve(dense, rhs), abs=1e-8)


def test_psor_respects_the_constraint_and_complementarity():
    """Either the equation holds or the bound binds -- never neither."""
    n, lo, di, up = 200, -1.0, 2.5, -1.0
    rhs = np.random.default_rng(1).normal(size=n)
    g = np.full(n, 0.2)
    v = psor_solve(lo, di, up, rhs, constraint=g)
    dense = (np.diag(np.full(n, di)) + np.diag(np.full(n - 1, up), 1)
             + np.diag(np.full(n - 1, lo), -1))
    assert np.all(v >= g - 1e-12)                    # v >= g
    assert np.all(dense @ v - rhs > -1e-8)           # A v >= b
    binding = v <= g + 1e-9
    assert np.all((dense @ v - rhs)[~binding] < 1e-8)  # equality off the bound


def test_psor_rejects_a_divergent_relaxation():
    with pytest.raises(ValidationError, match="omega"):
        psor_solve(-1.0, 2.5, -1.0, np.zeros(10), omega=2.5)


def test_psor_reports_failure_to_converge():
    with pytest.raises(PricingError, match="did not converge"):
        psor_solve(-1.0, 2.5, -1.0, np.ones(50), max_sweeps=2)


# -- European: against the closed form -------------------------------------


@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
def test_european_matches_black(right):
    res = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, right, "european")
    assert res.price == pytest.approx(_black(float(right.sign)), abs=2e-4)


def test_european_greeks_match_black():
    fwd, disc = SPOT * math.exp((RATE - DIV) * T), math.exp(-RATE * T)
    carry = math.exp((RATE - DIV) * T)
    res = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "call", "european", greeks=True)
    assert res.delta == pytest.approx(
        float(black_delta(fwd, STRIKE, T, VOL, disc, 1)) * carry, rel=1e-4)
    assert res.gamma == pytest.approx(
        float(black_gamma(fwd, STRIKE, T, VOL, disc)) * carry**2, rel=1e-3)


def test_european_put_call_parity():
    c = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "call", "european").price
    p = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "european").price
    parity = SPOT * math.exp(-DIV * T) - STRIKE * math.exp(-RATE * T)
    assert (c - p) == pytest.approx(parity, abs=5e-4)


def test_rejects_bermudan():
    with pytest.raises(ValidationError, match="Bermudan"):
        price_pde(SPOT, STRIKE, T, VOL, RATE, style=Style.BERMUDAN)


def test_rannacher_smoothing_kills_the_gamma_ringing():
    """Crank-Nicolson is stable but not damping, and the payoff kink is a
    high-frequency source. Where the time step is large against ``dx^2`` the
    oscillation is not subtle: measured here, gamma at the strike comes back
    2.55 against a true 0.0188 -- 135 times too big -- and two implicit steps
    drop the roughness by five orders of magnitude.
    """
    fwd, disc = SPOT * math.exp((RATE - DIV) * T), math.exp(-RATE * T)
    ref = float(black_gamma(fwd, STRIKE, T, VOL, disc)) * math.exp((RATE - DIV) * T) ** 2
    kw = dict(n_space=2048, n_time=16, greeks=True)

    def gammas(rannacher):
        return np.array([
            price_pde(s, STRIKE, T, VOL, RATE, DIV, "call", "european",
                      rannacher_steps=rannacher, **kw).gamma
            for s in np.linspace(96.0, 104.0, 33)
        ])

    raw, smoothed = gammas(0), gammas(4)
    roughness = lambda g: float(np.abs(np.diff(g, 2)).sum())  # noqa: E731

    assert abs(raw[16] - ref) > 1.0                      # catastrophically wrong
    assert smoothed[16] == pytest.approx(ref, rel=1e-2)  # and then right
    assert roughness(smoothed) < roughness(raw) / 1e4


# -- American, and the agreement that matters ------------------------------


def test_american_call_without_dividends_equals_european():
    """No dividend, so the constraint never binds and PSOR must land on the
    unconstrained solution -- to its own tolerance, not exactly, because the two
    take different routes to it."""
    kw = dict(n_space=512, n_time=256)
    am = price_pde(SPOT, STRIKE, T, VOL, RATE, 0.0, "call", "american", **kw).price
    eu = price_pde(SPOT, STRIKE, T, VOL, RATE, 0.0, "call", "european", **kw).price
    assert am == pytest.approx(eu, abs=1e-8)


def test_american_put_exceeds_european_and_intrinsic():
    kw = dict(n_space=512, n_time=256)
    am = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american", **kw).price
    eu = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "european", **kw).price
    assert am > eu * (1 + 1e-6)
    assert am >= max(STRIKE - SPOT, 0.0)


def test_deep_itm_american_put_is_intrinsic():
    res = price_pde(1.0, 100.0, T, VOL, RATE, DIV, "put", "american",
                    n_space=512, n_time=256)
    assert res.price == pytest.approx(99.0, abs=1e-6)


def test_pde_and_tree_agree_on_the_american_put():
    """The cross-check this module exists for. Two discretizations, two
    formulations, no shared arithmetic -- and they meet at 6.0904."""
    pde = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                    n_space=2048, n_time=1024).price
    tree = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                         n_steps=4096).price
    assert pde == pytest.approx(tree, abs=1e-3)
    assert pde == pytest.approx(6.0904, abs=2e-3)


def test_pde_and_tree_bracket_the_answer_from_opposite_sides():
    """Measured: the PDE rises 6.0835 -> 6.0903 as the mesh refines while the
    tree falls 6.0907 -> 6.0904. Converging from opposite directions is what
    makes the pair worth more than either alone -- a shared bug would have to
    push both the same way.
    """
    coarse_pde = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                           n_space=256, n_time=128).price
    fine_pde = price_pde(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                         n_space=2048, n_time=1024).price
    coarse_tree = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                                n_steps=1024).price
    fine_tree = price_lattice(SPOT, STRIKE, T, VOL, RATE, DIV, "put", "american",
                              n_steps=8192).price
    assert coarse_pde < fine_pde                 # PDE from below
    assert coarse_tree > fine_tree               # tree from above
    assert fine_pde < fine_tree                  # still bracketing at the end
