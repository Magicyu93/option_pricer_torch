"""Snapshots on disk, so a calibration can be repeated exactly.

A quote is a fact about one instant. Re-fetching gives a different instant, so
any result derived from a live pull is unreproducible by construction -- and a
test that depends on one fails for reasons that have nothing to do with the code.
Everything here exists so that a chain can be pulled once, written down, and read
back forever.

The format is plain JSON rather than a pickle: a frozen snapshot is a test
fixture that will be read by people and diffed by version control, and it should
survive a refactor of the classes it was produced from.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from torch_pricer.errors import MarketDataError
from torch_pricer.instruments.spec import Right
from torch_pricer.market.market_data import OptionQuote, QuoteSet, RateQuote, SpotQuote


def _quote_to_json(q: OptionQuote) -> dict:
    return {
        "expiry": q.expiry.isoformat(), "strike": q.strike, "right": q.right.value,
        "bid": q.bid, "ask": q.ask, "last": q.last, "implied_vol": q.implied_vol,
    }


def to_json(quotes: QuoteSet) -> dict:
    """A ``QuoteSet`` as plain data."""
    return {
        "as_of": quotes.as_of.isoformat(),
        "spot": {
            "ticker": quotes.spot.ticker, "value": quotes.spot.value,
            "as_of": quotes.spot.as_of.isoformat() if quotes.spot.as_of else None,
        },
        "options": [_quote_to_json(q) for q in quotes.options],
        "rates": [{"tenor": r.tenor, "rate": r.rate, "kind": r.kind}
                  for r in quotes.rates],
        "dividends": [{"tenor": r.tenor, "rate": r.rate, "kind": r.kind}
                      for r in quotes.dividends],
    }


def from_json(raw: dict) -> QuoteSet:
    """Rebuild a ``QuoteSet`` from :func:`to_json` output."""
    try:
        spot = raw["spot"]
        return QuoteSet(
            as_of=dt.date.fromisoformat(raw["as_of"]),
            spot=SpotQuote(
                ticker=spot["ticker"], value=float(spot["value"]),
                as_of=dt.date.fromisoformat(spot["as_of"]) if spot.get("as_of") else None,
            ),
            options=tuple(
                OptionQuote(
                    expiry=dt.date.fromisoformat(q["expiry"]), strike=float(q["strike"]),
                    right=Right(q["right"]), bid=q.get("bid"), ask=q.get("ask"),
                    last=q.get("last"), implied_vol=q.get("implied_vol"),
                )
                for q in raw.get("options", ())
            ),
            rates=tuple(RateQuote(**r) for r in raw.get("rates", ())),
            dividends=tuple(RateQuote(**r) for r in raw.get("dividends", ())),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise MarketDataError(f"malformed snapshot: {exc}") from exc


def save(quotes: QuoteSet, path: str | Path) -> Path:
    """Write a snapshot, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(quotes), indent=1, sort_keys=True))
    return path


def load(path: str | Path) -> QuoteSet:
    """Read a snapshot written by :func:`save`."""
    path = Path(path)
    if not path.exists():
        raise MarketDataError(f"no snapshot at {path}")
    return from_json(json.loads(path.read_text()))


class CachedSource:
    """A source that reads from disk, fetching only on a miss.

    The cache key is ``(ticker, as_of)``, so yesterday's chain is never silently
    served as today's. A miss with no live source behind it is an error rather
    than an empty result: quietly returning nothing is how an empty calibration
    gets blamed on the market.
    """

    def __init__(self, directory: str | Path, source=None):
        self.directory = Path(directory)
        self.source = source

    def path_for(self, ticker: str, as_of: dt.date) -> Path:
        safe = ticker.replace(":", "_").replace("/", "_")
        return self.directory / f"{safe}_{as_of.isoformat()}.json"

    def fetch(self, ticker: str, as_of: dt.date | None = None) -> QuoteSet:
        if as_of is not None:
            path = self.path_for(ticker, as_of)
            if path.exists():
                return load(path)
        if self.source is None:
            raise MarketDataError(
                f"no cached snapshot for {ticker} at {as_of} and no live source "
                "configured; pass one to CachedSource to fetch it"
            )
        quotes = self.source.fetch(ticker, as_of)
        save(quotes, self.path_for(ticker, quotes.as_of))
        return quotes
