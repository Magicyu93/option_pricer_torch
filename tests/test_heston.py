"""Heston: the transform, the simulator that must agree with it, and the fit."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import torch_pricer as tp
from torch_pricer.analytics import black as B
from torch_pricer.analytics.heston import characteristic_function, heston_price

from conftest import MATURITY

PARAMS = {"v0": 0.04, "kappa": 1.5, "theta": 0.05, "xi": 0.6, "rho": -0.7}


@pytest.fixture
def contract(market):
    """``(t, forward, discount)`` for the standard one-year expiry."""
    t = market.time_to(MATURITY)
    return t, market.forward(t).detach(), market.discount.discount(t).detach()


def test_transform_is_a_probability_transform(contract):
    """``phi(0) = 1``, and ``phi(-i) = E[S_T / F] = 1`` because the forward is a martingale."""
    t, _, _ = contract
    at_zero = characteristic_function(torch.tensor(0.0 + 0j), t, **PARAMS)
    at_minus_i = characteristic_function(torch.tensor(-1j), t, **PARAMS)
    assert complex(at_zero) == pytest.approx(1.0 + 0j, abs=1e-12)
    assert complex(at_minus_i) == pytest.approx(1.0 + 0j, abs=1e-10)


@pytest.mark.parametrize("strike", [60.0, 80.0, 100.0, 120.0, 160.0])
@pytest.mark.parametrize("right", [1, -1])
def test_zero_vol_of_vol_is_black(contract, strike, right):
    """With ``xi -> 0`` the variance is deterministic and Heston must collapse onto Black.

    Taken at ``rho = 0``: the leading smile term is ``O(rho xi)``, so at any
    non-zero correlation the difference between the two models is first order in
    ``xi`` and this comparison would be measuring the model, not the code.
    """
    t, forward, discount = contract
    vol = math.sqrt(PARAMS["v0"])
    heston = heston_price(
        forward, strike, t, v0=vol**2, kappa=1.5, theta=vol**2, xi=1e-3, rho=0.0,
        discount=discount, right=right,
    )
    black = B.black_price(forward, strike, t, vol, discount, right)
    assert float(heston) == pytest.approx(float(black), abs=1e-4)


def test_price_is_put_call_parity_consistent(contract):
    t, forward, discount = contract
    strikes = torch.tensor([70.0, 100.0, 130.0])
    call = heston_price(forward, strikes, t, **PARAMS, discount=discount, right=1)
    put = heston_price(forward, strikes, t, **PARAMS, discount=discount, right=-1)
    assert torch.allclose(call - put, discount * (forward - strikes), atol=1e-10)


def test_quadrature_has_converged_at_the_default_order(contract):
    """256 nodes is not a guess: doubling them changes nothing at float64 resolution."""
    t, forward, discount = contract
    for strike in (60.0, 100.0, 160.0):
        coarse = heston_price(forward, strike, t, **PARAMS, discount=discount, n_nodes=256)
        fine = heston_price(forward, strike, t, **PARAMS, discount=discount, n_nodes=1024)
        assert float(coarse) == pytest.approx(float(fine), abs=1e-9)


def test_price_is_differentiable_in_every_parameter(contract):
    """The fit needs exact gradients, and complex arithmetic must not break them."""
    t, forward, discount = contract
    model = tp.HestonModel(**PARAMS)
    price = model.price(forward, 100.0, t, discount, 1)
    price.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_simulation_agrees_with_the_transform(market):
    """The Monte Carlo and the semi-analytic price are the same model, priced twice."""
    model = tp.HestonModel(**PARAMS)
    heston_market = dataclasses.replace(market, vol=model)
    t = market.time_to(MATURITY)
    forward, discount = market.forward(t).detach(), market.discount.discount(t).detach()
    config = tp.MCConfig(n_paths=100_000, n_steps=200, seed=3, device="cpu")

    for strike in (80.0, 100.0, 120.0):
        spec = tp.VanillaOption(strike=strike, maturity=MATURITY, right="call")
        result = tp.price(spec, heston_market, config)
        exact = float(model.price(forward, strike, t, discount, 1).detach())
        # Three standard errors, plus room for the O(h) full-truncation bias,
        # which at 200 steps is smaller than the sampling error.
        assert result.price == pytest.approx(exact, abs=3.5 * result.stderr)


def test_greeks_come_out_of_the_two_factor_simulation(market):
    """Delta, gamma and vega must survive a state that is no longer one-dimensional."""
    model = tp.HestonModel(**PARAMS)
    heston_market = dataclasses.replace(market, vol=model)
    config = tp.MCConfig(n_paths=50_000, n_steps=100, seed=5, device="cpu")
    spec = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    risk = tp.price(spec, heston_market, config, greeks=("delta", "gamma", "vega")).greeks

    assert 0.0 < risk["delta"] < 1.0
    assert risk["gamma"] > 0.0
    # Vega here is the parallel shift of the instantaneous spot vol, so it is on
    # the same scale as Black's vega at the model's own implied level.
    assert 20.0 < risk["vega"] < 60.0


def test_calibration_recovers_the_parameters_it_generated(market, smile_inputs, smile_model):
    """Fitting a model to its own prices has a known answer, and this finds it."""
    fitted = tp.HestonModel()  # cold start, at the class defaults
    fitted.calibrate(smile_inputs, tolerance=1e-3)

    assert fitted.fit_report["rmse_vol"] < 1e-4
    for name in ("v0", "kappa", "theta", "xi", "rho"):
        assert float(getattr(fitted, name).detach()) == pytest.approx(
            float(getattr(smile_model, name).detach()), rel=1e-2
        )


def test_a_flat_vol_cannot_fit_a_smile_and_says_so(smile_inputs):
    model = tp.BlackScholesModel(0.30)
    with pytest.raises(tp.CalibrationError, match="the market has a smile"):
        model.calibrate(smile_inputs, tolerance=0.01)
    # It still lands on the vega-weighted average level rather than nowhere.
    assert 0.15 < float(model.vol.detach()) < 0.25


def test_calibration_needs_quotes(market):
    model = tp.HestonModel()
    with pytest.raises(tp.CalibrationError, match="needs option quotes"):
        model.calibrate(tp.CalibrationInputs.from_market(market))


def test_parameters_cannot_be_constructed_outside_their_domain():
    with pytest.raises(tp.ValidationError, match="v0 must be positive"):
        tp.HestonModel(v0=-0.01)
    with pytest.raises(tp.ValidationError, match="rho must lie strictly inside"):
        tp.HestonModel(rho=-1.0)


def test_feller_ratio_reports_without_enforcing():
    """A ratio below one is a fact about the fit, not an error: equities fit there."""
    assert tp.HestonModel(kappa=1.5, theta=0.05, xi=0.6).feller_ratio == pytest.approx(
        2 * 1.5 * 0.05 / 0.36
    )
