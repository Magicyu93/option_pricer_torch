"""Build a vol surface for an index from a real option chain.

    python -m examples.spx_surface                 # cached snapshot, offline
    python -m examples.spx_surface --fetch         # live, needs POLYGON_API_KEY

The whole Phase C pipeline in one script, and every stage prints what it dropped:

    chain -> clean -> static arbitrage -> parity forwards -> curves
          -> implied vols -> OTM only -> SVI fit

**Why the forward comes from parity and not from a curve.** ``fit_surface``
places each quote at ``k = log(K / F(T))``, so an error in the forward slides the
whole slice sideways, where it looks exactly like a model failure. Regressing
``C - P`` on ``K`` recovers the forward and the discount factor from the option
quotes alone -- the forward the market is actually trading, dividend forecast and
borrow included -- and removes both from the error budget before the fit begins.

**Index options only.** ``I:SPX`` is European. The liquid names on a cheap data
plan are ETF options (SPY, QQQ) and those are American, where parity is only an
inequality and the regression above would be fitting a band rather than a line.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import torch

from torch_pricer.calibration.inputs import CalibrationInputs
from torch_pricer.calibration.svi_fit import fit_surface
from torch_pricer.data.cache import CachedSource
from torch_pricer.data.clean import CleaningRules, clean, static_arbitrage
from torch_pricer.data.implied import attach_implied_vols, otm_only
from torch_pricer.data import synthetic
from torch_pricer.data.massive import MassiveSource
from torch_pricer.errors import MarketDataError, PricerError
from torch_pricer.market.forward import curves_from_forwards, implied_forwards
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.svi import SVISlice, SVISurface

TICKER = "I:SPX"
SNAPSHOTS = Path(__file__).resolve().parent.parent / "tests" / "data"


def load_chain(fetch: bool, as_of: dt.date | None):
    source = MassiveSource() if fetch else None
    cached = CachedSource(SNAPSHOTS, source=source)
    if fetch:
        return cached.fetch(TICKER, dt.date.today())
    if as_of is None:
        available = sorted(SNAPSHOTS.glob(f"{TICKER.replace(':', '_')}_*.json"))
        if not available:
            return None            # caller falls back to the synthetic chain
        as_of = dt.date.fromisoformat(available[-1].stem.split("_")[-1])
    return cached.fetch(TICKER, as_of)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true",
                        help="pull a live chain instead of reading a snapshot")
    parser.add_argument("--as-of", type=dt.date.fromisoformat, default=None)
    args = parser.parse_args()

    if args.fetch and not any(os.environ.get(v) for v in
                              ("POLYGON_API_KEY", "MASSIVE_API_KEY")):
        print("--fetch needs POLYGON_API_KEY or MASSIVE_API_KEY in the environment.")
        return 2

    try:
        chain = load_chain(args.fetch, args.as_of)
    except PricerError as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1

    if chain is None:
        today = dt.date.today()
        chain = synthetic.chain(
            as_of=today,
            expiries=[today + dt.timedelta(days=d) for d in (105, 196, 287)],
        )
        print("=" * 72)
        print("NO CACHED SNAPSHOT -- running on a MANUFACTURED chain.")
        print("These are not market prices. The numbers below demonstrate the")
        print("pipeline and recover the parameters the chain was built from")
        print("(r = 4.2%, q = 1.3%); they say nothing about SPX.")
        print("Run with --fetch and POLYGON_API_KEY set to use real quotes.")
        print("=" * 72 + "\n")

    print(f"{TICKER}  as of {chain.as_of}  spot {chain.spot.value:,.2f}")
    print(f"  {len(chain.options):,} raw quotes over {len(chain.expiries())} expiries")

    cleaned, report = clean(chain, CleaningRules(min_price=0.05))
    print(f"\n{report}")
    if not cleaned.options:
        print("nothing survived cleaning")
        return 1

    arb = static_arbitrage(cleaned)
    print(f"\n{arb}")
    for name in ("monotonicity", "convexity", "calendar"):
        for line in getattr(arb, name)[:3]:
            print(f"  {name}: {line}")

    fits = implied_forwards(cleaned, lambda e: (e - chain.as_of).days / 365.0)
    print(f"\nparity-implied forwards ({len(fits)} expiries)")
    print(f"  {'expiry':>12} {'T':>7} {'forward':>12} {'discount':>10} "
          f"{'zero':>8} {'pairs':>6} {'resid':>9}")
    for f in fits:
        print(f"  {f.expiry!s:>12} {f.t:>7.4f} {f.forward:>12,.2f} "
              f"{f.discount:>10.6f} {f.zero_rate:>8.4%} {f.n_pairs:>6} "
              f"{f.max_residual:>9.4f}")

    discount, dividend = curves_from_forwards(fits, chain.spot.value)
    snapshot = MarketSnapshot.flat(chain.as_of, spot=chain.spot.value, ticker=TICKER)
    snapshot = type(snapshot)(
        ticker=TICKER, as_of=chain.as_of, spot=snapshot.spot,
        discount=discount, dividend=dividend, vol_surface=snapshot.vol_surface,
    )

    with_vols, inversion = attach_implied_vols(cleaned, fits)
    print(f"\n{inversion}")
    quotes = otm_only(with_vols, fits)
    print(f"  {len(quotes.options):,} out-of-the-money quotes to fit")

    surface = SVISurface(
        slices=[SVISlice(0.02, 0.08, -0.4, 0.0, 0.2, f.t) for f in fits],
        forward=snapshot.forward,
        as_of=chain.as_of,
    )
    result = fit_surface(surface, CalibrationInputs(market=snapshot, quotes=quotes))
    residuals = result.residuals.abs()
    print(f"\nSVI fit: {residuals.numel()} quotes, "
          f"max {float(residuals.max()):.4%} vol, "
          f"rms {float((residuals**2).mean().sqrt()):.4%} vol")

    print(f"\n{'expiry':>8} {'k=-0.1':>9} {'ATM':>9} {'k=+0.1':>9}")
    for f in fits:
        row = [
            float(surface.vol(
                torch.tensor(f.forward * torch.tensor(k).exp().item(),
                             dtype=torch.float64), f.t).detach())
            for k in (-0.1, 0.0, 0.1)
        ]
        print(f"  {f.t:>6.3f} " + " ".join(f"{v:>9.2%}" for v in row))

    print("\nNote: the forwards above use no rate curve and no dividend estimate.")
    print("They come from the option quotes alone, so a residual in the fit is")
    print("the model's and not the curve's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
