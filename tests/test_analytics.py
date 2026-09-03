"""Closed-form Black formulas and the implied-vol inversion."""

from __future__ import annotations

import math

import pytest
import torch

from torch_pricer.analytics import black as B

F, K, T, V, D = 100.0, 95.0, 1.0, 0.25, 0.98


def test_closed_form_greeks_agree_with_autograd():
    """The analytic greeks and autograd through the analytic price are the same function."""
    fwd = torch.tensor(F, requires_grad=True)
    vol = torch.tensor(V, requires_grad=True)
    price = B.black_price(fwd, K, T, vol, D, 1)
    delta, vega = torch.autograd.grad(price, [fwd, vol], create_graph=True)
    (gamma,) = torch.autograd.grad(delta, [fwd])

    assert float(delta.detach()) == pytest.approx(float(B.black_delta(F, K, T, V, D, 1)), rel=1e-10)
    assert float(vega.detach()) == pytest.approx(float(B.black_vega(F, K, T, V, D)), rel=1e-10)
    assert float(gamma.detach()) == pytest.approx(float(B.black_gamma(F, K, T, V, D)), rel=1e-10)


def test_put_call_parity():
    call = B.black_price(F, K, T, V, D, 1)
    put = B.black_price(F, K, T, V, D, -1)
    assert float(call - put) == pytest.approx(D * (F - K), rel=1e-12)


def test_price_is_bounded_by_intrinsic_and_the_forward():
    price = float(B.black_price(F, K, T, V, D, 1))
    assert float(B.intrinsic(F, K, D, 1)) <= price <= D * F


@pytest.mark.parametrize("right", [1, -1])
def test_implied_vol_round_trips(right):
    strikes = torch.tensor([60.0, 80.0, 95.0, 100.0, 130.0, 180.0])
    vols = torch.tensor([0.40, 0.30, 0.25, 0.24, 0.28, 0.35])
    prices = B.black_price(F, strikes, T, vols, D, right)
    recovered = B.implied_vol(prices, F, strikes, T, D, right)
    assert torch.allclose(recovered, vols, atol=1e-6)


def test_implied_vol_is_nan_outside_the_no_arbitrage_band():
    below = B.implied_vol(torch.tensor(1e-6), F, K, T, D, 1)  # under intrinsic
    above = B.implied_vol(torch.tensor(2.0 * F), F, K, T, D, 1)  # over the forward
    assert math.isnan(float(below))
    assert math.isnan(float(above))


def test_degenerate_inputs_are_clamped_not_raised():
    """A surface fit evaluates at zero time and zero vol on its way somewhere sensible."""
    for t, vol in ((0.0, V), (T, 0.0), (0.0, 0.0)):
        assert all(
            torch.isfinite(fn(F, K, t, vol, D)).all()
            for fn in (B.black_vega, B.black_gamma)
        )
        assert torch.isfinite(B.black_price(F, K, t, vol, D, 1)).all()


def test_vega_and_gamma_are_right_agnostic():
    assert float(B.black_vega(F, K, T, V, D)) > 0
    assert float(B.black_gamma(F, K, T, V, D)) > 0


def test_formulas_broadcast():
    strikes = torch.linspace(80.0, 120.0, 9)
    assert B.black_price(F, strikes, T, V, D, 1).shape == strikes.shape
    assert B.black_delta(F, strikes, T, V, D, 1).shape == strikes.shape
