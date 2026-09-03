"""Local-stochastic vol: the leverage function, and what it is supposed to fix."""

from __future__ import annotations

import dataclasses

import pytest
import torch

import torch_pricer as tp
from torch_pricer.analytics.black import implied_vol

from conftest import MATURITY, VOL


@pytest.fixture
def flat_inputs(market):
    return dataclasses.replace(
        tp.CalibrationInputs.from_market(market), surface=tp.FlatVolSurface(VOL, market.as_of)
    )


def _implied(market, model, strike, config):
    """The implied vol of a Monte Carlo price under ``model``."""
    spec = tp.VanillaOption(strike=strike, maturity=MATURITY, right="call")
    result = tp.price(spec, dataclasses.replace(market, vol=model), config)
    t = market.time_to(MATURITY)
    forward = market.forward(t).detach()
    discount = market.discount.discount(t).detach()
    return float(implied_vol(result.price, forward, strike, t, discount, 1)), result


def test_an_uncalibrated_model_is_plain_heston(market, smile_model):
    """Leverage one is the identity, and the LSV must then agree with the transform."""
    model = tp.LSVModel(smile_model)
    assert torch.allclose(model.leverage.detach(), torch.ones_like(model.leverage))

    config = tp.MCConfig(n_paths=50_000, n_steps=100, seed=3, device="cpu")
    t = market.time_to(MATURITY)
    forward, discount = market.forward(t).detach(), market.discount.discount(t).detach()
    spec = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")

    result = tp.price(spec, dataclasses.replace(market, vol=model), config)
    exact = float(smile_model.price(forward, 100.0, t, discount, 1).detach())
    assert result.price == pytest.approx(exact, abs=3.5 * result.stderr)


def test_leverage_flattens_heston_onto_a_flat_surface(market, smile_model, flat_inputs):
    """The point of the model, stated as a test.

    The stochastic part is a hard skew -- ``rho = -0.7`` -- and the target is a
    surface with no skew at all. Repriced through the calibrated leverage, the
    smile must come back flat: not merely close to the right level, but with the
    strike dependence gone. That is what says the leverage is being computed
    conditionally on the spot rather than as a level correction.
    """
    model = tp.LSVModel(smile_model)
    model.calibrate(flat_inputs, n_paths=50_000, n_steps=100, seed=5, fit_stochastic=False)

    config = tp.MCConfig(n_paths=100_000, n_steps=200, seed=17, device="cpu")
    vols = {strike: _implied(market, model, strike, config)[0] for strike in (85.0, 100.0, 115.0)}

    # Half a volatility point of the target, and a quarter of a point of skew
    # left across a 30-point strike range. Un-calibrated, the same Heston prices
    # this range with about four points of skew.
    for strike, vol in vols.items():
        assert vol == pytest.approx(VOL, abs=0.005), strike
    assert max(vols.values()) - min(vols.values()) < 0.0025


def test_uncalibrated_heston_really_does_have_the_skew_being_removed(market, smile_model):
    """The control for the test above: without leverage the smile is far from flat."""
    config = tp.MCConfig(n_paths=100_000, n_steps=200, seed=17, device="cpu")
    model = tp.LSVModel(smile_model)
    vols = {strike: _implied(market, model, strike, config)[0] for strike in (85.0, 100.0, 115.0)}
    assert max(vols.values()) - min(vols.values()) > 0.02


def test_calibration_leaves_the_stochastic_parameters_alone_when_asked(smile_model, flat_inputs):
    before = {name: float(p.detach()) for name, p in smile_model.named_parameters()}
    model = tp.LSVModel(smile_model)
    model.calibrate(flat_inputs, n_paths=20_000, n_steps=50, seed=1, fit_stochastic=False)
    after = {name: float(p.detach()) for name, p in model.stochastic.named_parameters()}
    assert before == after


def test_calibration_fits_the_stochastic_part_when_given_quotes(market, smile_inputs):
    """With quotes and a surface, both halves of the model are fitted in one call."""
    surface = tp.SVISurface.fit(smile_inputs)
    inputs = dataclasses.replace(smile_inputs, surface=surface)

    model = tp.LSVModel()  # default Heston parameters, nowhere near the truth
    model.calibrate(inputs, n_paths=20_000, n_steps=50, seed=2)

    assert model.stochastic.fit_report["rmse_vol"] < 1e-3
    assert model.leverage.shape == (51, model.log_moneyness.numel())
    assert float(model.leverage.min()) > 0.0


def test_leverage_needs_a_local_vol_surface(smile_inputs):
    model = tp.LSVModel()
    with pytest.raises(tp.CalibrationError, match="fitted to a local vol surface"):
        model.calibrate(smile_inputs)


def test_greeks_survive_the_leverage_grid(market, smile_model, flat_inputs):
    """Delta must differentiate *through* the interpolation of ``L(S, t)``, not around it.

    The check is against a central bump under the same seed rather than against
    Black, because the two are not supposed to agree. The leverage function is
    anchored in the absolute level, so moving the spot slides the model along a
    surface that is steep in ``S`` -- the leverage is what cancels Heston's
    skew, and cancelling a skew takes a slope. The model's own delta is
    therefore far from the flat-vol Black delta it reprices at, and getting
    ``0.79`` where Black says ``0.57`` is the model speaking, not a bug. What
    would be a bug is autograd disagreeing with a bump on the same model, which
    is what this pins down.
    """
    model = tp.LSVModel(smile_model)
    model.calibrate(flat_inputs, n_paths=20_000, n_steps=50, seed=7, fit_stochastic=False)
    config = tp.MCConfig(n_paths=50_000, n_steps=100, seed=9, device="cpu")
    spec = tp.VanillaOption(strike=100.0, maturity=MATURITY, right="call")
    lsv_market = dataclasses.replace(market, vol=model)

    risk = tp.price(spec, lsv_market, config, greeks=("delta", "vega")).greeks

    bump = 0.5
    up = tp.price(spec, dataclasses.replace(lsv_market, spot=market.spot + bump), config).price
    down = tp.price(spec, dataclasses.replace(lsv_market, spot=market.spot - bump), config).price
    assert risk["delta"] == pytest.approx((up - down) / (2 * bump), abs=2e-3)
    assert risk["vega"] > 0.0
