"""Premiums to implied volatilities, where a failed inversion is visible.

``fit_surface`` refuses quotes that carry no ``implied_vol`` and says why in the
error: inverting premiums is the caller's job, "so that a failed inversion is
reported where the bad quote is rather than as a mysterious wide residual here".
This is that caller.

Two things make the step worth its own module. The forward and discount come from
:mod:`torch_pricer.market.forward` rather than from a curve, so the vol is
implied against the market's own forward and a smile is not shifted sideways by
somebody's dividend estimate. And ``implied_vol`` returns ``nan`` outside the
no-arbitrage bounds rather than clamping -- a price at or through intrinsic has
no recoverable volatility, and inventing the nearest one would launder an
arbitrage into a plausible number. Those ``nan`` s are counted and dropped here,
where the strike that produced them can still be named.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from dataclasses import dataclass, field

from torch_pricer.errors import MarketDataError
from torch_pricer.market.forward import ImpliedForward
from torch_pricer.market.market_data import QuoteSet
from torch_pricer.pricer.analytic.black import implied_vol


@dataclass
class InversionReport:
    """How many premiums produced a usable volatility, and why the rest did not."""

    inverted: int = 0
    failed: Counter = field(default_factory=Counter)
    #: ``(expiry, strike, right, price)`` for each failure, for chasing one down.
    examples: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.inverted + sum(self.failed.values())

    def __repr__(self) -> str:  # pragma: no cover
        if not self.total:
            return "InversionReport(empty)"
        head = f"inverted {self.inverted}/{self.total} ({self.inverted / self.total:.1%})"
        return "InversionReport(\n" + "\n".join(
            [head] + [f"  {k}: {v}" for k, v in self.failed.most_common()]
        ) + "\n)"


def attach_implied_vols(
    quotes: QuoteSet,
    forwards: list[ImpliedForward],
    min_vol: float = 1e-3,
    max_vol: float = 3.0,
):
    """Populate ``implied_vol`` on every quote that admits one.

    Args:
        quotes: a cleaned European chain
        forwards: parity fits, one per expiry; expiries without one are dropped
        min_vol: reject a vol at or below this as a degenerate inversion
        max_vol: reject a vol at or above this the same way

    Returns:
        ``(QuoteSet with implied_vol set, InversionReport)``.

    Raises:
        MarketDataError: if nothing inverted at all.
    """
    by_expiry = {f.expiry: f for f in forwards}
    report = InversionReport()
    kept = []

    for q in quotes.options:
        fit = by_expiry.get(q.expiry)
        if fit is None:
            report.failed["no-forward"] += 1
            continue
        price = q.price
        if price is None:
            report.failed["no-price"] += 1
            continue

        w = float(q.right.sign)
        vol = float(implied_vol(price, fit.forward, q.strike, fit.t, fit.discount, w))
        if math.isnan(vol):
            # Outside [intrinsic, forward bound]: no volatility reproduces it.
            report.failed["outside-no-arbitrage-bounds"] += 1
            report.examples.append((q.expiry, q.strike, q.right.value, price))
            continue
        if not min_vol < vol < max_vol:
            report.failed["implausible-vol"] += 1
            report.examples.append((q.expiry, q.strike, q.right.value, price))
            continue

        kept.append(dataclasses.replace(q, implied_vol=vol))
        report.inverted += 1

    if not kept:
        raise MarketDataError(
            f"no quote inverted to a usable volatility ({report.failed.most_common()})"
        )
    return dataclasses.replace(quotes, options=tuple(kept)), report


def otm_only(quotes: QuoteSet, forwards: list[ImpliedForward]) -> QuoteSet:
    """Keep the out-of-the-money leg at each strike.

    Both legs imply the same volatility in theory and not in practice: the
    in-the-money one is mostly intrinsic, so its vega is small and the same tick
    of quote noise moves its implied vol much further. Market convention fits the
    OTM wing on each side, and doing otherwise weights the noisiest quotes most.
    """
    by_expiry = {f.expiry: f.forward for f in forwards}
    kept = [
        q for q in quotes.options
        if q.expiry in by_expiry
        and (q.strike >= by_expiry[q.expiry]) == (q.right.value == "call")
    ]
    return dataclasses.replace(quotes, options=tuple(kept))
