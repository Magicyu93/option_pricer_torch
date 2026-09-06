import datetime as dt
import math

import pytest

from torch_pricer.instruments.spec import Right, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot

AS_OF = dt.date(2025, 1, 2)
EXPIRY = dt.date(2026, 1, 2)
SPOT, RATE, DIV, VOL = 100.0, 0.03, 0.01, 0.20


@pytest.fixture
def market() -> MarketSnapshot:
    return MarketSnapshot.flat(
        AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=DIV, flat_vol=VOL
    )


@pytest.fixture
def call() -> VanillaOption:
    return VanillaOption(strike=100.0, maturity=EXPIRY, right=Right.CALL)


@pytest.fixture
def put() -> VanillaOption:
    return VanillaOption(strike=100.0, maturity=EXPIRY, right=Right.PUT)


def analytic_inputs(market, spec):
    """``(T, forward, discount)`` for the flat market fixture."""
    t = market.time_to(spec.maturity)
    return t, SPOT * math.exp((RATE - DIV) * t), math.exp(-RATE * t)
