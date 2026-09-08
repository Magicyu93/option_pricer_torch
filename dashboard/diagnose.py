"""Find which underlyings your Massive plan actually serves, and write it down.

    python dashboard/diagnose.py                 # probe the candidate list
    python dashboard/diagnose.py I:SPX SPY       # probe specific tickers
    python dashboard/diagnose.py --discover      # ask the API what indices exist

``dashboard/tickers.json`` carries two different kinds of fact. Exercise style
comes from the exchange's contract specifications and is true regardless of who
is asking. Availability depends on your subscription and cannot be known from
anywhere but your own key -- so this fills that in and saves it back.

The key is read from the environment, never printed, and redacted from every URL
shown.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CATALOGUE = HERE / "tickers.json"
BASE = "https://api.polygon.io"
KEY_VARS = ("POLYGON_API_KEY", "MASSIVE_API_KEY")


def _key() -> str | None:
    import os

    return next((os.environ[v] for v in KEY_VARS if os.environ.get(v)), None)


def probe(ticker: str, key: str, verbose: bool = False) -> dict:
    """One snapshot call. Returns what came back, without the key."""
    url = f"{BASE}/v3/snapshot/options/{ticker}"
    out = {"ticker": ticker}
    try:
        r = requests.get(url, params={"apiKey": key, "limit": 5}, timeout=30)
    except Exception as exc:
        out["result"] = f"transport error: {type(exc).__name__}"
        return out

    out["http"] = r.status_code
    try:
        payload = r.json()
    except Exception:
        out["result"] = f"non-JSON: {r.text[:120]}"
        return out

    for field in ("status", "message", "error"):
        if payload.get(field):
            out[field] = payload[field]

    rows = payload.get("results") or []
    out["rows"] = len(rows)
    if not rows:
        out["result"] = "empty"
        return out

    first = rows[0]
    details = first.get("details") or {}
    quote = first.get("last_quote") or {}
    trade = first.get("last_trade") or {}
    out["result"] = "ok"
    out["sample"] = {
        "expiry": details.get("expiration_date"),
        "strike": details.get("strike_price"),
        "right": details.get("contract_type"),
        "bid": quote.get("bid"), "ask": quote.get("ask"),
        "last": trade.get("price"),
        "underlying": (first.get("underlying_asset") or {}).get("price"),
    }
    out["parses"] = all(
        details.get(f) is not None
        for f in ("expiration_date", "strike_price", "contract_type")
    ) and any(v is not None for v in (quote.get("bid"), quote.get("ask"),
                                      trade.get("price")))
    out["row_keys"] = sorted(first)
    if verbose:
        out["raw_first_row"] = first
    return out


def discover(key: str, limit: int = 1000) -> list[dict]:
    """Ask the reference endpoint which indices exist on this plan."""
    url = f"{BASE}/v3/reference/tickers"
    r = requests.get(url, params={"apiKey": key, "market": "indices",
                                  "active": "true", "limit": limit}, timeout=30)
    r.raise_for_status()
    return r.json().get("results") or []


def main(argv: list[str]) -> int:
    key = _key()
    if key is None:
        print("No POLYGON_API_KEY / MASSIVE_API_KEY in this shell.")
        print("~/.zshrc is not read by non-interactive shells; use ~/.zshenv.")
        return 2
    print(f"key: found ({len(key)} chars), never printed\n")

    catalogue = json.loads(CATALOGUE.read_text())

    if "--discover" in argv:
        rows = discover(key)
        print(f"reference/tickers?market=indices -> {len(rows)} indices on this plan")
        for row in rows[:40]:
            print(f"  {row.get('ticker','?'):<12} {row.get('name','')[:60]}")
        if len(rows) > 40:
            print(f"  ... and {len(rows) - 40} more")
        return 0

    explicit = [a for a in argv if not a.startswith("-")]
    if explicit:
        entries = [{"snapshot": t, "name": "(explicit)", "exercise": "?"}
                   for t in explicit]
    else:
        entries = sorted(catalogue["candidates"], key=lambda e: e["priority"])

    print(f"{'ticker':<8} {'exercise':<9} {'http':<5} {'rows':>5}  result")
    print("-" * 74)
    results = {}
    for entry in entries:
        got = probe(entry["snapshot"], key, verbose="-v" in argv)
        results[entry["snapshot"]] = got
        note = got.get("result", "")
        if got.get("message"):
            note += f" -- {got['message'][:44]}"
        if got.get("result") == "ok":
            note += f", parses={got['parses']}"
        print(f"{entry['snapshot']:<8} {entry.get('exercise','?'):<9} "
              f"{str(got.get('http','-')):<5} {str(got.get('rows','-')):>5}  {note}")

    usable = [t for t, g in results.items() if g.get("result") == "ok"]
    print("\n" + "-" * 74)
    if usable:
        print(f"usable: {' '.join(usable)}")
        first = results[usable[0]]
        print(f"\nsample row from {usable[0]}:")
        for k, v in first["sample"].items():
            print(f"  {k:<11} {v}")
        print(f"  row keys: {first['row_keys']}")
        if not first["parses"]:
            print("\n  !! the adapter's expected fields are missing -- paste this output")
    else:
        print("nothing usable. If every index is empty but an ETF returns rows,")
        print("the plan does not include index options.")

    if not explicit:
        stamp = dt.date.today().isoformat()
        for entry in catalogue["candidates"]:
            got = results.get(entry["snapshot"], {})
            entry["available"] = got.get("result") == "ok"
            entry["last_probed"] = stamp
            if got.get("rows") is not None:
                entry["rows_seen"] = got["rows"]
        CATALOGUE.write_text(json.dumps(catalogue, indent=2) + "\n")
        print(f"\nwrote availability back to {CATALOGUE.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
