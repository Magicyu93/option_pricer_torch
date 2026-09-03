"""Shared fixtures: one textbook market, and the analytic answers for it."""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

import torch_pricer as tp
from torch_pricer.analytics import black as B

AS_OF = dt.date(2026, 1, 2)
MATURITY = dt.date(2027, 1, 2)
SPOT = 100.0
RATE = 0.03
DIVIDEND = 0.01
VOL = 0.20


@pytest.fixture
def market() -> tp.MarketSnapshot:
    return tp.MarketSnapshot.flat(
        AS_OF, SPOT, rate=RATE, dividend=DIVIDEND, vol=VOL, ticker="TEST"
    )


@pytest.fixture
def config() -> tp.MCConfig:
    # Fixed seed: these tests assert against analytic values, so they must not
    # be flaky. Tolerances below are sized from the reported standard error.
    return tp.MCConfig(n_paths=200_000, n_steps=25, seed=7, device="cpu")


def analytic(market: tp.MarketSnapshot, strike: float, right: int) -> dict[str, float]:
    """Closed-form price and *spot* greeks for a European option on ``market``.

    The formulas in :mod:`torch_pricer.analytics.black` are stated on the
    forward, so delta and gamma are converted with ``dF/dS = D_q / D_r``.
    """
    t = market.time_to(MATURITY)
    fwd = market.forward(t)
    disc = market.discount.discount(t)
    dfds = float(market.dividend.discount(t) / disc)
    return {
        "price": float(B.black_price(fwd, strike, t, VOL, disc, right)),
        "delta": float(B.black_delta(fwd, strike, t, VOL, disc, right)) * dfds,
        "gamma": float(B.black_gamma(fwd, strike, t, VOL, disc)) * dfds**2,
        "vega": float(B.black_vega(fwd, strike, t, VOL, disc)),
    }


def heston_quotes(
    inputs: "tp.CalibrationInputs",
    model: "tp.HestonModel",
    months: tuple[int, ...] = (1, 3, 6, 12, 24),
    moneyness: tuple[float, ...] = (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4),
) -> tp.QuoteSet:
    """A synthetic quote set: Heston's own prices, treated as the market's.

    Fitting something to a model's exact prices is the only way to say what a
    calibration recovers, because the answer is known. Real quotes have a
    bid-ask and a smile no model reproduces exactly, and a test written against
    them can only assert that nothing crashed.
    """
    options = []
    for m in months:
        expiry = inputs.as_of + dt.timedelta(days=int(30.4 * m))
        t = inputs.time_to(expiry)
        forward = inputs.forward(t).detach()
        discount = inputs.discount.discount(t).detach()
        for level in moneyness:
            strike = round(float(forward) * level, 4)
            price = float(model.price(forward, strike, t, discount, 1).detach())
            options.append(tp.OptionQuote(expiry=expiry, strike=strike, right="call", last=price))
    return tp.QuoteSet(
        as_of=inputs.as_of,
        spot=tp.SpotQuote("TEST", float(inputs.spot), inputs.as_of),
        options=tuple(options),
    )


@pytest.fixture
def smile_model() -> tp.HestonModel:
    """The Heston parameters every synthetic surface in these tests comes from."""
    return tp.HestonModel(v0=0.04, kappa=1.5, theta=0.05, xi=0.6, rho=-0.7)


@pytest.fixture
def smile_inputs(market: tp.MarketSnapshot, smile_model: tp.HestonModel) -> tp.CalibrationInputs:
    """Calibration inputs carrying a quoted Heston smile."""
    bare = tp.CalibrationInputs.from_market(market)
    return dataclasses.replace(bare, quotes=heston_quotes(bare, smile_model))
