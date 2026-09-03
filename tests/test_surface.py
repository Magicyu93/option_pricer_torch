"""SVI slices, the surface they make, and the arbitrage conditions on both."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import pytest
import torch

import torch_pricer as tp
from torch_pricer.analytics import black as B

from conftest import AS_OF


def test_flat_surface_is_flat_in_both_coordinates():
    surface = tp.FlatVolSurface(0.2, AS_OF)
    assert float(surface.vol(80.0, 0.5)) == pytest.approx(0.2)
    assert float(surface.vol(140.0, 3.0)) == pytest.approx(0.2)
    # Total variance is the default derived quantity: sigma^2 T.
    assert float(surface.total_variance(100.0, 2.0)) == pytest.approx(0.08)


def test_slice_is_a_hyperbola_with_the_minimum_it_was_given():
    slice_ = tp.SVISlice(b=0.1, rho=-0.5, m=0.02, sigma=0.15, w_min=0.03)
    k = torch.linspace(-2.0, 2.0, 401)
    w = slice_.total_variance(k).detach()
    # The vertex sits between grid points, so the sampled minimum is a hair above it.
    assert float(w.min()) == pytest.approx(0.03, rel=1e-3)
    assert float(w.min()) >= float(slice_.w_min.detach())
    assert float(w[0]) > float(w[200]) < float(w[-1])  # convex, minimum inside


def test_slice_wings_cannot_be_steeper_than_lees_bound():
    """The parameterisation refuses it on construction and cannot reach it by fitting."""
    with pytest.raises(tp.ValidationError, match="Lee's moment bound"):
        tp.SVISlice(b=1.5, rho=0.6)

    slice_ = tp.SVISlice()
    # Push the optimiser hard at an unreachable target and check the constraint holds.
    k = torch.linspace(-1.0, 1.0, 21)
    slice_.fit(k, 0.2 + 5.0 * k.abs(), expiry=1.0, iterations=200)
    assert float(slice_.put_wing.detach()) < 2.0
    assert float(slice_.call_wing.detach()) < 2.0
    assert -1.0 < float(slice_.rho.detach()) < 1.0


def test_slice_recovers_a_smile_generated_by_another_slice():
    truth = tp.SVISlice(b=0.12, rho=-0.6, m=0.05, sigma=0.25, w_min=0.035)
    k = torch.linspace(-0.6, 0.6, 25)
    target = (truth.total_variance(k).detach() / 1.0).sqrt()

    fitted = tp.SVISlice()
    rmse = fitted.fit(k, target, expiry=1.0)
    # The *smile* is recovered to a fiftieth of a volatility point. The
    # parameters are not, and cannot be: SVI's five are close to degenerate --
    # a smaller `b` with a wider `sigma` draws almost the same curve -- so
    # asserting on the shape is the only thing that means anything.
    assert rmse < 1e-3
    assert torch.allclose(
        fitted.total_variance(k).detach(), truth.total_variance(k).detach(), atol=5e-4
    )


def test_fitted_slices_are_butterfly_free(smile_inputs):
    surface = tp.SVISurface.fit(smile_inputs)
    for slice_ in surface.slices:
        assert slice_.is_butterfly_free()


def test_butterfly_margin_catches_a_slice_that_is_not():
    """A vertex sharp enough to imply a negative density is detected, not fitted around."""
    bad = tp.SVISlice(b=1.2, rho=0.0, m=0.0, sigma=0.005, w_min=0.02)
    assert not bad.is_butterfly_free()
    assert float(bad.butterfly_margin(torch.linspace(-1.5, 1.5, 301)).min()) < 0.0


def test_surface_reproduces_the_quoted_smile(smile_inputs, smile_model):
    """A fit is only as good as the vols it returns at the strikes it was given."""
    surface = tp.SVISurface.fit(smile_inputs)

    errors = []
    for expiry in smile_inputs.quotes.expiries():
        t = smile_inputs.time_to(expiry)
        forward = smile_inputs.forward(t).detach()
        discount = smile_inputs.discount.discount(t).detach()
        group = smile_inputs.quotes.slice(expiry)
        quoted = [B.implied_vol(q.price, forward, q.strike, t, discount, 1) for q in group]
        vegas = [
            0.0 if torch.isnan(v) else float(B.black_vega(forward, q.strike, t, v, discount))
            for q, v in zip(group, quoted)
        ]
        # The fit is vega weighted, so it must be judged that way. A wing quote
        # worth a tenth of the at-the-money vega is a quote the fit was
        # deliberately told to ignore, and holding it to the same tolerance
        # would be testing the weighting rather than the fit.
        floor = 0.1 * max(vegas)
        errors += [
            abs(float(v) - float(surface.vol(q.strike, t)))
            for q, v, vega in zip(group, quoted, vegas)
            if vega >= floor
        ]
    assert max(errors) < 0.005


def test_surface_interpolates_in_total_variance_between_its_slices(smile_inputs):
    surface = tp.SVISurface.fit(smile_inputs)
    t0, t1 = float(surface.expiries[1]), float(surface.expiries[2])
    strike = float(smile_inputs.spot)

    w0 = float(surface.total_variance(strike, t0))
    w1 = float(surface.total_variance(strike, t1))
    mid = float(surface.total_variance(strike, 0.5 * (t0 + t1)))
    assert min(w0, w1) <= mid <= max(w0, w1)


def test_surface_holds_vol_flat_outside_the_quoted_expiries(smile_inputs):
    surface = tp.SVISurface.fit(smile_inputs)
    first, last = float(surface.expiries[0]), float(surface.expiries[-1])

    # Flat at fixed *moneyness*, which is the coordinate the slices live in --
    # at a fixed strike the vol still moves, because the forward does.
    def at_moneyness(k: float, t: float) -> float:
        return float(surface.vol(float(smile_inputs.forward(t).detach()) * math.exp(k), t))

    for k in (-0.2, 0.0, 0.2):
        assert at_moneyness(k, first / 3.0) == pytest.approx(at_moneyness(k, first), rel=1e-9)
        assert at_moneyness(k, last * 3.0) == pytest.approx(at_moneyness(k, last), rel=1e-9)


def test_surface_is_calendar_arbitrage_free(smile_inputs):
    """Total variance must not fall as expiry grows, at any fixed moneyness."""
    surface = tp.SVISurface.fit(smile_inputs)
    assert float(surface.calendar_margin().min()) >= 0.0


def test_a_surface_needs_enough_quotes_per_expiry(market):
    inputs = tp.CalibrationInputs.from_market(market)
    expiry = AS_OF + dt.timedelta(days=180)
    thin = tp.QuoteSet(
        as_of=AS_OF,
        spot=tp.SpotQuote("TEST", 100.0, AS_OF),
        options=tuple(
            tp.OptionQuote(expiry=expiry, strike=k, right="call", implied_vol=0.2)
            for k in (95.0, 100.0, 105.0)
        ),
    )
    with pytest.raises(tp.CalibrationError, match="five usable quotes"):
        tp.SVISurface.fit(dataclasses.replace(inputs, quotes=thin))
