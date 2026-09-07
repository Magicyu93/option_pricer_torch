"""Comparing the Monte Carlo simulation with the closed-form solution for European options.

Run with ``python -m examples.european_options``.
"""

import datetime as dt
import math

from torch_pricer.black_formula import black_delta, black_price, black_vega
from torch_pricer.instruments.spec import Right, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.engine import MCConfig, price

AS_OF, EXPIRY = dt.date(2025, 1, 2), dt.date(2026, 1, 2)
SPOT, RATE, DIV, VOL = 100.0, 0.03, 0.01, 0.20


def main() -> None:
    market = MarketSnapshot.flat(
        AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=DIV, flat_vol=VOL
    )
    config = MCConfig(n_paths=200_000, n_steps=20, seed=7)

    print(f"{'strike':>7} {'right':>5} {'MC':>10} {'stderr':>8} {'Black':>10} "
          f"{'MC delta':>9} {'Black':>9} {'MC vega':>9} {'Black':>9}")
    for strike in (80.0, 100.0, 120.0):
        for right in (Right.CALL, Right.PUT):
            spec = VanillaOption(strike=strike, maturity=EXPIRY, right=right)
            res = price(spec, market, BlackScholesModel(VOL), config,
                        greeks=("delta", "vega"))

            t = market.time_to(EXPIRY)
            fwd, disc = SPOT * math.exp((RATE - DIV) * t), math.exp(-RATE * t)
            w = right.sign
            ref = float(black_price(fwd, strike, t, VOL, disc, right=w))
            # black_delta is w.r.t. the forward; dF/dS = D_q / D_r.
            ref_d = float(black_delta(fwd, strike, t, VOL, disc, right=w)) * math.exp(
                (RATE - DIV) * t
            )
            ref_v = float(black_vega(fwd, strike, t, VOL, disc))
            print(f"{strike:>7.0f} {right.value:>5} {res.price:>10.5f} {res.stderr:>8.5f} "
                  f"{ref:>10.5f} {res.greeks['delta']:>9.5f} {ref_d:>9.5f} "
                  f"{res.greeks['vega']:>9.4f} {ref_v:>9.4f}")


if __name__ == "__main__":
    main()
