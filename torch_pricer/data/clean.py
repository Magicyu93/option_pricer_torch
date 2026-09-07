"""Rejecting quotes, and saying why.

A vendor chain is not a surface. It carries contracts that have never traded,
markets quoted one-sided or crossed, strikes so far out of the money that the tick
size exceeds the premium, and rows whose timestamp is minutes stale. Fitting to
that produces a calibration that fails for reasons no one can locate.

Two principles here. **Filtering is separate from fetching**, so a missing quote
is attributable to a rule rather than to the feed. And **every rejection is
counted**, because the useful diagnostic is not the cleaned chain but the tally:
a name that drops eighty per cent of its puts is telling you something the
residuals never will.

Nothing here needs a model. The arbitrage report is deliberately model-free -- it
tests the quotes against each other, so a violation is a fact about the data
rather than a disagreement with an assumption.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from torch_pricer.instruments.spec import Right
from torch_pricer.market.market_data import OptionQuote, QuoteSet


@dataclass(frozen=True)
class CleaningRules:
    """Thresholds for rejecting a quote."""

    #: Require a two-sided market. A lone bid is information, but it is not a mid,
    #: and treating it as one puts a systematic error into the smile.
    require_two_sided: bool = True
    #: Reject a bid at or below this. A zero bid means no one wants the contract
    #: at any price and its "mid" is half the ask, which is noise.
    min_bid: float = 0.0
    #: Reject when ``(ask - bid) / mid`` exceeds this. Wide markets are where the
    #: mid is least informative and the vol it implies least trustworthy.
    max_relative_spread: float = 0.5
    #: Reject a mid below this. Near the tick the premium is quantisation.
    min_price: float = 0.05


@dataclass
class CleaningReport:
    """What survived, and what did not."""

    kept: int = 0
    rejected: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return self.kept + sum(self.rejected.values())

    def __repr__(self) -> str:  # pragma: no cover
        if not self.total:
            return "CleaningReport(empty)"
        lines = [f"kept {self.kept}/{self.total} ({self.kept / self.total:.1%})"]
        lines += [f"  {reason}: {n}" for reason, n in self.rejected.most_common()]
        return "CleaningReport(\n" + "\n".join(lines) + "\n)"


def _reject_reason(q: OptionQuote, rules: CleaningRules) -> str | None:
    if q.bid is not None and q.ask is not None and q.bid > q.ask:
        return "crossed"
    if rules.require_two_sided and (q.bid is None or q.ask is None):
        return "one-sided"
    if q.bid is not None and q.bid <= rules.min_bid:
        return "zero-bid"
    mid = q.price
    if mid is None or mid <= 0:
        return "no-price"
    if mid < rules.min_price:
        return "below-tick"
    if q.bid is not None and q.ask is not None:
        if (q.ask - q.bid) / mid > rules.max_relative_spread:
            return "wide-spread"
    return None


def clean(quotes: QuoteSet, rules: CleaningRules | None = None):
    """Drop unusable quotes. Returns ``(cleaned QuoteSet, CleaningReport)``."""
    rules = rules or CleaningRules()
    report = CleaningReport()
    kept = []
    for q in quotes.options:
        reason = _reject_reason(q, rules)
        if reason is None:
            kept.append(q)
            report.kept += 1
        else:
            report.rejected[reason] += 1
    import dataclasses

    return dataclasses.replace(quotes, options=tuple(kept)), report


@dataclass(frozen=True)
class ArbitrageReport:
    """Model-free violations found among the quotes themselves."""

    monotonicity: tuple[str, ...] = ()
    convexity: tuple[str, ...] = ()
    calendar: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not (self.monotonicity or self.convexity or self.calendar)

    def __repr__(self) -> str:  # pragma: no cover
        if self.clean:
            return "ArbitrageReport(clean)"
        return (
            f"ArbitrageReport(monotonicity={len(self.monotonicity)}, "
            f"convexity={len(self.convexity)}, calendar={len(self.calendar)})"
        )


def static_arbitrage(quotes: QuoteSet, tolerance: float = 1e-8) -> ArbitrageReport:
    """Check the chain against itself.

    Three conditions that hold for *any* arbitrage-free market, with no model:

    * **Monotonicity** -- a call is non-increasing in strike, a put non-decreasing.
    * **Convexity** -- a butterfly costs something: ``C(K1) - 2 C(K2) + C(K3) >= 0``
      for equally spaced strikes.
    * **Calendar** -- for the same strike, a longer-dated option is worth at least
      as much as a shorter one. True for calls and puts on a forward-flat market;
      breaches usually mean a stale expiry rather than free money.

    A violation is a fact about the quotes. Tolerance exists only for float noise,
    not to excuse a breach.
    """
    monotonicity, convexity, calendar = [], [], []

    for expiry in quotes.expiries():
        for right in (Right.CALL, Right.PUT):
            row = [q for q in quotes.slice(expiry) if q.right is right and q.price]
            if len(row) < 2:
                continue
            for a, b in zip(row, row[1:]):
                rising = b.price > a.price + tolerance
                if right is Right.CALL and rising:
                    monotonicity.append(
                        f"{expiry} call {a.strike:g}->{b.strike:g}: "
                        f"{a.price:.4f} -> {b.price:.4f} rises"
                    )
                if right is Right.PUT and a.price > b.price + tolerance:
                    monotonicity.append(
                        f"{expiry} put {a.strike:g}->{b.strike:g}: "
                        f"{a.price:.4f} -> {b.price:.4f} falls"
                    )
            for x, y, z in zip(row, row[1:], row[2:]):
                gap1, gap2 = y.strike - x.strike, z.strike - y.strike
                if abs(gap1 - gap2) > 1e-9:
                    continue  # unequal spacing: the simple butterfly does not apply
                if x.price - 2 * y.price + z.price < -tolerance:
                    convexity.append(
                        f"{expiry} {right.value} butterfly "
                        f"{x.strike:g}/{y.strike:g}/{z.strike:g} is negative"
                    )

    expiries = quotes.expiries()
    for near, far in zip(expiries, expiries[1:]):
        near_by = {(q.strike, q.right): q.price for q in quotes.slice(near) if q.price}
        for q in quotes.slice(far):
            earlier = near_by.get((q.strike, q.right))
            if earlier is not None and q.price and q.price < earlier - tolerance:
                calendar.append(
                    f"{q.right.value} {q.strike:g}: {far} ({q.price:.4f}) below "
                    f"{near} ({earlier:.4f})"
                )

    return ArbitrageReport(
        monotonicity=tuple(monotonicity), convexity=tuple(convexity),
        calendar=tuple(calendar),
    )
