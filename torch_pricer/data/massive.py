"""Massive (formerly Polygon.io) option chains.

The rebrand is cosmetic for our purposes: ``api.polygon.io`` keeps working with
existing keys and no announced cut-off, ``api.massive.com`` runs in parallel, and
both SDKs are maintained. So the base URL is a constructor argument and migrating
later is one string.

Endpoint: ``/v3/snapshot/options/{underlying}``, paginated by ``next_url``.
Index options take an ``I:`` prefix -- ``I:SPX`` -- which is how the European
chains are reached. That matters, because the liquid free-tier names are ETF
options (SPY, QQQ) and those are American, so they cannot be checked against a
European closed form at all.

**Only bids, asks and last trades are read.** The snapshot also carries the
vendor's implied vol and greeks, and taking them would mean calibrating against
their forward, their curve and their dividend assumption, then reporting the
agreement as validation of ours. Everything derived is derived here.

No SDK dependency: the REST surface is one paginated GET and ``requests`` is
already present. ``fetch_json`` is injectable, so every test below runs offline.
"""

from __future__ import annotations

import datetime as dt
import os
from collections import Counter
from typing import Callable

from torch_pricer.errors import MarketDataError
from torch_pricer.instruments.spec import Right
from torch_pricer.market.market_data import OptionQuote, QuoteSet, SpotQuote

#: Both hosts serve the same API. Kept as a default rather than a constant so a
#: migration to api.massive.com is a constructor argument.
POLYGON_BASE = "https://api.polygon.io"
MASSIVE_BASE = "https://api.massive.com"

_KEY_VARS = ("POLYGON_API_KEY", "MASSIVE_API_KEY")

#: Last transport status, so an empty chain can say whether the request even
#: succeeded. A 200 with no rows and a 401 are entirely different problems and an
#: error that cannot tell them apart sends you looking in the wrong place.
_LAST_STATUS: dict = {}


def _default_fetch(url: str, params: dict) -> dict:
    import requests

    response = requests.get(url, params=params, timeout=30)
    _LAST_STATUS["http"] = response.status_code
    if response.status_code == 401:
        raise MarketDataError(
            "Massive rejected the API key (401). Set POLYGON_API_KEY or "
            "MASSIVE_API_KEY, or pass api_key= explicitly."
        )
    if response.status_code == 429:
        raise MarketDataError(
            "Massive rate-limited the request (429); the free tier allows only a "
            "few calls per minute, and a full chain is several pages"
        )
    if not response.ok:
        raise MarketDataError(
            f"Massive returned {response.status_code} for {url}: {response.text[:200]}"
        )
    return response.json()


class MassiveSource:
    """Option chains from Massive's snapshot endpoint.

    Args:
        api_key: the key; falls back to ``POLYGON_API_KEY`` then ``MASSIVE_API_KEY``
        base_url: API host, for migrating to ``api.massive.com``
        fetch_json: injectable transport, so tests need no network
        max_pages: stop after this many pages; a full index chain is large and an
            unbounded loop against a paginated API is how a rate limit is hit
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = POLYGON_BASE,
        fetch_json: Callable[[str, dict], dict] | None = None,
        max_pages: int = 40,
    ):
        self.api_key = api_key or next(
            (os.environ[v] for v in _KEY_VARS if os.environ.get(v)), None
        )
        self.base_url = base_url.rstrip("/")
        self.fetch_json = fetch_json or _default_fetch
        self.max_pages = max_pages

    def fetch(self, ticker: str, as_of: dt.date | None = None) -> QuoteSet:
        """The whole chain for ``ticker`` (use ``I:SPX`` for an index).

        ``as_of`` labels the snapshot; the endpoint itself always returns the
        latest, so a past date cannot be requested here -- that is the cache's
        job, and asking for one is an error rather than a silent substitution of
        today's chain.
        """
        if self.api_key is None:
            raise MarketDataError(
                "no Massive API key: set POLYGON_API_KEY or MASSIVE_API_KEY, or "
                "pass api_key= to MassiveSource"
            )
        today = dt.date.today()
        if as_of is not None and as_of != today:
            raise MarketDataError(
                f"the snapshot endpoint returns only the latest chain, so {as_of} "
                "cannot be fetched; read it from a cached snapshot instead"
            )

        url = f"{self.base_url}/v3/snapshot/options/{ticker}"
        params = {"apiKey": self.api_key, "limit": 250}
        options: list[OptionQuote] = []
        spot: float | None = None
        _LAST_STATUS.clear()
        seen = 0
        rejected: Counter = Counter()
        sample_keys: list[str] = []
        remote: dict = {}

        for _ in range(self.max_pages):
            payload = self.fetch_json(url, params)
            for field in ("status", "message", "error"):
                if payload.get(field):
                    remote[field] = payload[field]
            for row in payload.get("results") or ():
                seen += 1
                if not sample_keys:
                    sample_keys = sorted(row)
                parsed, why = _parse_contract(row)
                if parsed is not None:
                    options.append(parsed)
                else:
                    rejected[why] += 1
                underlying = (row.get("underlying_asset") or {}).get("price")
                if spot is None and underlying is not None:
                    spot = float(underlying)
            nxt = payload.get("next_url")
            if not nxt:
                break
            url, params = nxt, {"apiKey": self.api_key}

        if not options:
            raise MarketDataError(_no_quotes_message(ticker, seen, rejected,
                                                     sample_keys, remote))
        if spot is None:
            raise MarketDataError(
                f"Massive returned quotes for {ticker} but no underlying price; "
                "every downstream step needs a spot, so this is not recoverable"
            )
        return QuoteSet(
            as_of=as_of or today,
            spot=SpotQuote(ticker=ticker, value=spot, as_of=as_of or today),
            options=tuple(options),
        )


def _no_quotes_message(ticker, seen, rejected, sample_keys, remote) -> str:
    """Say which of the several very different failures this actually was.

    An empty chain has at least four causes -- the plan does not cover the
    instrument, the ticker is wrong, the market has never traded these contracts,
    or the response shape is not what the parser expects -- and they send you to
    four different places. The message names the one that happened.
    """
    http = _LAST_STATUS.get("http")
    head = f"Massive returned no usable option quotes for {ticker}"
    if http:
        head += f" (HTTP {http})"

    if seen == 0:
        detail = (
            "the response carried no contracts at all. Either this plan does not "
            "include options on this underlying -- index options usually need a "
            "paid options tier -- or the ticker is wrong. Indices take the 'I:' "
            "prefix on the snapshot endpoint (I:SPX), and none on the reference "
            "endpoints. Run `python dashboard/diagnose.py` to see which "
            "underlyings this key can actually reach."
        )
    else:
        worst = ", ".join(f"{k} x{v}" for k, v in rejected.most_common())
        detail = (
            f"{seen} contracts came back and none could be read ({worst}). That is "
            f"a parser problem, not a data one. Row keys seen: {sample_keys}. "
            "Run `python dashboard/diagnose.py -v` and send the sample row."
        )

    if remote:
        detail += f" The API also said: {remote}."
    return f"{head}: {detail}"


def _parse_contract(row: dict) -> tuple[OptionQuote | None, str]:
    """One snapshot row to an ``OptionQuote``, with the reason when it is not.

    A row with no strike, expiry or right cannot be placed on a surface at all.
    One with no tradable price is dropped here rather than downstream, because
    ``OptionQuote`` refuses to hold neither a premium nor a vol -- and a vendor
    chain routinely carries contracts that have never traded. The reason is
    returned rather than discarded so an empty chain can explain itself.
    """
    details = row.get("details") or {}
    try:
        expiry = dt.date.fromisoformat(details["expiration_date"])
        strike = float(details["strike_price"])
        right = Right(details["contract_type"])
    except (KeyError, ValueError, TypeError):
        return None, "no-contract-details"

    quote = row.get("last_quote") or {}
    trade = row.get("last_trade") or {}
    bid, ask = quote.get("bid"), quote.get("ask")
    last = trade.get("price")
    if bid is None and ask is None and last is None:
        return None, "never-quoted-or-traded"
    return OptionQuote(
        expiry=expiry, strike=strike, right=right,
        bid=None if bid is None else float(bid),
        ask=None if ask is None else float(ask),
        last=None if last is None else float(last),
    ), "ok"
