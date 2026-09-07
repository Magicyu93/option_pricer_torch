"""A manufactured option chain, for tests and for demonstrating the pipeline.

Not market data, and nothing here should ever be presented as such. Its purpose
is the opposite of a real snapshot's: a live chain can only be checked for
self-consistency, while a chain generated from parameters we chose has a known
right answer, so the pipeline can be asked to *recover* a forward and a smile
rather than merely to produce plausible ones.

Both legs at a strike are priced off one volatility, so put-call parity holds
exactly in the mids. That is what lets
:func:`~torch_pricer.market.forward.implied_forward` be tested for recovery to
floating point instead of to a tolerance someone chose.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from torch_pricer.instruments.spec import Right
from torch_pricer.market.market_data import OptionQuote, QuoteSet, SpotQuote
from torch_pricer.pricer.analytic.black import black_price


def smile(strike: float, forward: float, atm: float = 0.18, skew: float = -0.06) -> float:
    """A downward-sloping smile in log-moneyness, so nothing is trivially flat."""
    return atm + skew * math.log(strike / forward)


def chain(
    as_of: dt.date,
    expiries,
    spot: float = 5500.0,
    rate: float = 0.042,
    dividend: float = 0.013,
    spread: float = 0.02,
    n_strikes: int = 21,
    width: float = 0.25,
    ticker: str = "SYNTHETIC",
) -> QuoteSet:
    """A European chain generated from known ``rate`` and ``dividend``.

    Args:
        as_of: snapshot date
        expiries: expiry dates
        spot: underlying level
        rate: continuously-compounded funding rate to recover
        dividend: continuous dividend yield to recover
        spread: bid-ask width as a fraction of the mid
        n_strikes: strikes per expiry
        width: strike range as a fraction either side of the forward
        ticker: label; deliberately not an exchange symbol

    Returns:
        A :class:`~torch_pricer.market.market_data.QuoteSet`.
    """
    options = []
    for expiry in expiries:
        t = (expiry - as_of).days / 365.0
        if t <= 0:
            continue
        forward = spot * math.exp((rate - dividend) * t)
        discount = math.exp(-rate * t)
        for strike in np.linspace(forward * (1 - width), forward * (1 + width), n_strikes):
            vol = smile(strike, forward)
            for right, w in ((Right.CALL, 1.0), (Right.PUT, -1.0)):
                mid = float(black_price(forward, strike, t, vol, discount, w))
                half = 0.5 * spread * max(mid, 1.0)
                options.append(OptionQuote(
                    expiry=expiry, strike=float(strike), right=right,
                    bid=mid - half, ask=mid + half,
                ))
    return QuoteSet(
        as_of=as_of, spot=SpotQuote(ticker, spot, as_of), options=tuple(options)
    )
