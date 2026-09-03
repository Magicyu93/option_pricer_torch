"""Shared fixtures: one textbook market, and the analytic answers for it."""

from __future__ import annotations

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
