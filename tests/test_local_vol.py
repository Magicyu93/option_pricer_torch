"""Dupire: the transform from an implied surface, and the simulation it feeds."""

from __future__ import annotations

import dataclasses

import pytest
import torch

import torch_pricer as tp
from torch_pricer.calibration.vol_model.local_vol_cal import (
    MAX_LOCAL_VOL,
    dupire_local_variance,
)
from torch_pricer.simulator.local_vol import LevelSurface

from torch_pricer.analytics.black import black_vega, implied_vol

from conftest import MATURITY, VOL, analytic


@pytest.fixture
def flat_inputs(market):
    """A market whose implied surface is one flat number."""
    return dataclasses.replace(
        tp.CalibrationInputs.from_market(market), surface=tp.FlatVolSurface(VOL, market.as_of)
    )


def test_dupire_of_a_flat_surface_is_that_flat_vol(flat_inputs):
    """The one case with an answer by inspection: no skew, no term structure, no work.

    Every derivative in the formula except ``dw/dT`` vanishes and the
    denominator collapses to 1, so the local vol is the implied vol exactly --
    to the last bit, not approximately, because the derivatives are exact.
    """
    model = tp.LocalVolModel.from_surface(flat_inputs)
    assert torch.allclose(model.local_vol.detach(), torch.full_like(model.local_vol, VOL))


def test_dupire_denominator_is_positive_on_an_arbitrage_free_surface(smile_inputs):
    """The denominator is the implied density; a negative one is arbitrage upstream."""
    surface = tp.SVISurface.fit(smile_inputs)
    times = torch.linspace(0.05, 2.0, 20)
    k = torch.linspace(-1.0, 1.0, 41)
    _, denominator = dupire_local_variance(surface, smile_inputs.forward, times, k)
    assert float(denominator.min()) > 0.0


def test_dupire_needs_a_surface(market):
    with pytest.raises(tp.CalibrationError, match="transform of an implied surface"):
        tp.LocalVolModel.from_surface(tp.CalibrationInputs.from_market(market))


def test_dupire_rejects_a_zero_expiry(flat_inputs):
    with pytest.raises(tp.ValidationError, match="strictly positive expiries"):
        dupire_local_variance(
            flat_inputs.surface,
            flat_inputs.forward,
            torch.tensor([0.0, 1.0]),
            torch.linspace(-1, 1, 5),
        )


def test_local_vol_is_capped_rather_than_left_to_diverge(smile_inputs):
    """Short-dated far wings are pure extrapolation, and Dupire explodes there."""
    surface = tp.SVISurface.fit(smile_inputs)
    variance, _ = dupire_local_variance(
        surface, smile_inputs.forward, torch.tensor([0.02]), torch.linspace(-1.2, 1.2, 25)
    )
    assert float(variance.max().sqrt()) <= MAX_LOCAL_VOL


def test_grid_interpolation_is_bilinear_and_flat_outside():
    times = torch.tensor([0.5, 1.0])
    k = torch.linspace(-1.0, 1.0, 5)
    values = torch.stack([torch.full((5,), 0.2), torch.full((5,), 0.3)])
    surface = LevelSurface(times, k, values, lambda t: torch.as_tensor(100.0))
    asset = torch.tensor([[80.0], [100.0], [130.0]])

    assert torch.allclose(surface(asset, torch.tensor(0.75)), torch.full((3, 1), 0.25))
    assert torch.allclose(surface(asset, torch.tensor(0.0)), torch.full((3, 1), 0.2))
    assert torch.allclose(surface(asset, torch.tensor(9.0)), torch.full((3, 1), 0.3))


def test_grid_rejects_a_misshaped_value_array():
    with pytest.raises(tp.ValidationError, match="must be"):
        LevelSurface(
            torch.tensor([0.5, 1.0]), torch.linspace(-1, 1, 5), torch.ones(2, 4), lambda t: t
        )


def test_a_flat_local_vol_prices_like_black(market, config, flat_inputs):
    """With a constant grid the SDE is GBM again, so Black is the exact answer."""
    model = tp.LocalVolModel.from_surface(flat_inputs)
    local_market = dataclasses.replace(market, vol=model)
    strike = 100.0
    spec = tp.VanillaOption(strike=strike, maturity=MATURITY, right="call")

    result = tp.price(spec, local_market, config, greeks=("delta", "gamma", "vega"))
    exact = analytic(market, strike, 1)

    assert result.price == pytest.approx(exact["price"], abs=3 * result.stderr)
    assert result.greeks["delta"] == pytest.approx(exact["delta"], abs=2e-3)
    assert result.greeks["gamma"] == pytest.approx(exact["gamma"], rel=0.05)
    # Vega is the parallel shift of the whole grid, which for a flat grid is
    # exactly Black's vega.
    assert result.greeks["vega"] == pytest.approx(exact["vega"], rel=0.01)


def test_local_vol_reprices_the_surface_it_was_built_from(market, smile_inputs, smile_model):
    """Dupire's promise, tested end to end: fit a surface, transform it, simulate it back.

    The chain is SVI fit -> autograd derivatives -> local vol grid -> Euler
    simulation, and the answer has to come back to the Heston prices the quotes
    were generated from. Every step contributes: the tolerance below is a
    quarter of a volatility point, which is what this pipeline is worth at this
    grid resolution and path count, not what Dupire's theorem promises in the
    continuum.
    """
    surface = tp.SVISurface.fit(smile_inputs)
    model = tp.LocalVolModel.from_surface(dataclasses.replace(smile_inputs, surface=surface))
    local_market = dataclasses.replace(market, vol=model)

    t = market.time_to(MATURITY)
    forward = market.forward(t).detach()
    discount = market.discount.discount(t).detach()
    config = tp.MCConfig(n_paths=100_000, n_steps=200, seed=11, device="cpu")

    for strike in (90.0, 100.0, 110.0):
        spec = tp.VanillaOption(strike=strike, maturity=MATURITY, right="call")
        result = tp.price(spec, local_market, config)
        target = float(smile_model.price(forward, strike, t, discount, 1).detach())
        quoted = implied_vol(target, forward, strike, t, discount, 1)
        vega = float(black_vega(forward, strike, t, quoted, discount))
        assert abs(result.price - target) < 0.0025 * vega


def test_the_grid_is_a_function_of_the_level_not_of_moneyness(market, flat_inputs):
    """Bumping the spot must not drag the surface with it; that is a different model."""
    model = tp.LocalVolModel.from_surface(flat_inputs)
    reference = float(model.reference_spot)

    assert float(model.reference_spot) == pytest.approx(float(market.spot))
    # The coordinate is log(S / F_ref(t)) with F_ref built on the calibration
    # spot, so the same asset level maps to the same grid point whatever the
    # snapshot is later marked at.
    moved = dataclasses.replace(market, spot=torch.tensor(reference * 1.5))
    assert float(model.reference_forward(moved.discount, moved.dividend)(0.5)) == pytest.approx(
        float(model.reference_forward(market.discount, market.dividend)(0.5))
    )
