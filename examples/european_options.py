"""Monte Carlo against the closed form for European options.

Run with::

    python -m examples.european_options

Prices a strip of strikes twice -- once by simulating the SDE in torch and once
by the Black formula -- and reports the gap in units of the Monte Carlo standard
error. A well-wired engine sits inside about three of those.

The greeks are the interesting column. Delta and vega come out of a single
backward pass through the simulation (the pathwise estimator, unbiased). Gamma
cannot: the payoff is piecewise linear in spot, so its second derivative
vanishes almost everywhere, and the engine falls back to differencing the
pathwise delta under common random numbers.
"""

from __future__ import annotations

import datetime as dt

import torch_pricer as tp
from torch_pricer.analytics import black as B

AS_OF = dt.date(2026, 1, 2)
MATURITY = dt.date(2027, 1, 2)
SPOT, RATE, DIVIDEND, VOL = 100.0, 0.03, 0.01, 0.20
STRIKES = (80.0, 90.0, 100.0, 110.0, 120.0)


def main() -> None:
    market = tp.MarketSnapshot.flat(
        AS_OF, SPOT, rate=RATE, dividend=DIVIDEND, vol=VOL, ticker="DEMO"
    )
    config = tp.MCConfig(n_paths=200_000, n_steps=50, seed=7)

    t = market.time_to(MATURITY)
    forward = market.forward(t)
    discount = market.discount.discount(t)
    dfds = float(market.dividend.discount(t) / discount)

    print(f"{market}   T={t:.4f}  F={float(forward):.4f}  D={float(discount):.6f}")
    print(f"paths={config.n_paths:,}  steps={config.n_steps}  seed={config.seed}\n")

    header = (
        f"{'strike':>7} {'right':>5} | {'MC':>9} {'Black':>9} {'err/se':>7} | "
        f"{'MC delta':>9} {'Black':>9} | {'MC gamma':>9} {'Black':>9} | "
        f"{'MC vega':>9} {'Black':>9}"
    )
    print(header)
    print("-" * len(header))

    for right in ("call", "put"):
        w = 1 if right == "call" else -1
        for strike in STRIKES:
            spec = tp.VanillaOption(
                underlying="DEMO", strike=strike, maturity=MATURITY, right=right
            )
            got = tp.price(spec, market, config, greeks=("delta", "gamma", "vega"))
            ref_price = float(B.black_price(forward, strike, t, VOL, discount, w))
            ref_delta = float(B.black_delta(forward, strike, t, VOL, discount, w)) * dfds
            ref_gamma = float(B.black_gamma(forward, strike, t, VOL, discount)) * dfds**2
            ref_vega = float(B.black_vega(forward, strike, t, VOL, discount))
            print(
                f"{strike:7.0f} {right:>5} | {got.price:9.4f} {ref_price:9.4f} "
                f"{(got.price - ref_price) / got.stderr:7.2f} | "
                f"{got.greeks['delta']:9.4f} {ref_delta:9.4f} | "
                f"{got.greeks['gamma']:9.5f} {ref_gamma:9.5f} | "
                f"{got.greeks['vega']:9.3f} {ref_vega:9.3f}"
            )

    print("\nerr/se is the pricing error in standard errors; |err/se| < 3 is agreement.")


if __name__ == "__main__":
    main()
