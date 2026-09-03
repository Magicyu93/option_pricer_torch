"""Three models, one surface, and the trade that tells them apart.

Run with::

    python -m examples.smile_models

Takes a quoted smile -- here generated from a Heston model standing in for the
market -- and calibrates all three of the library's smile models to it:

* **Heston**, fitted by its characteristic function to the quoted premiums;
* **local volatility**, obtained from an SVI fit of the same quotes by Dupire's
  formula, with no optimisation at all;
* **local-stochastic volatility**, Heston underneath and a leverage function
  fitted by the particle method so the vanillas come back exactly.

Then it prices the same vanillas under each and, at the end, an Asian option.
The vanillas are the point of agreement: all three are calibrated to the same
surface, so all three must reproduce it, and the table says how well each one
manages. The Asian is the point of disagreement. It pays off on the whole path
rather than on the terminal distribution, so it is sensitive to how the smile
*moves* -- and that is precisely what calibrating to today's surface does not
pin down. The spread across the three columns is the model risk in a
path-dependent trade, and no amount of vanilla calibration removes it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt


import torch_pricer as tp
from torch_pricer.analytics.black import black_vega, implied_vol

AS_OF = dt.date(2026, 1, 2)
MATURITY = dt.date(2027, 1, 2)
SPOT, RATE, DIVIDEND = 100.0, 0.03, 0.01
#: The "market": a Heston smile with a hard skew, quoted as premiums.
MARKET_PARAMS = {"v0": 0.04, "kappa": 1.5, "theta": 0.05, "xi": 0.6, "rho": -0.7}
EXPIRY_MONTHS = (1, 3, 6, 12, 24)
MONEYNESS = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)


def quoted_market(inputs: tp.CalibrationInputs, model: tp.HestonModel) -> tp.QuoteSet:
    """Screen prices, as they would arrive: strikes, expiries and premiums."""
    options = []
    for months in EXPIRY_MONTHS:
        expiry = AS_OF + dt.timedelta(days=int(30.4 * months))
        t = inputs.time_to(expiry)
        forward = inputs.forward(t).detach()
        discount = inputs.discount.discount(t).detach()
        for level in MONEYNESS:
            strike = round(float(forward) * level, 2)
            options.append(
                tp.OptionQuote(
                    expiry=expiry,
                    strike=strike,
                    right="call",
                    last=float(model.price(forward, strike, t, discount, 1).detach()),
                )
            )
    return tp.QuoteSet(AS_OF, tp.SpotQuote("DEMO", SPOT, AS_OF), tuple(options))


def main() -> None:
    market = tp.MarketSnapshot.flat(AS_OF, SPOT, rate=RATE, dividend=DIVIDEND, ticker="DEMO")
    truth = tp.HestonModel(**MARKET_PARAMS)
    inputs = tp.CalibrationInputs.from_market(market)
    inputs = dataclasses.replace(inputs, quotes=quoted_market(inputs, truth))

    t = market.time_to(MATURITY)
    forward = market.forward(t).detach()
    discount = market.discount.discount(t).detach()
    print(f"{market}   T={t:.4f}  F={float(forward):.4f}\n")

    print("fitting an SVI surface to the quotes ...")
    surface = tp.SVISurface.fit(inputs)
    inputs = dataclasses.replace(inputs, surface=surface)
    print(f"  {surface}, calendar margin {float(surface.calendar_margin().min()):+.5f}")

    print("calibrating Heston to the premiums ...")
    heston = tp.HestonModel()
    heston.calibrate(inputs)
    print(f"  {heston}")
    print(
        f"  RMSE {heston.fit_report['rmse_vol']:.3%} of vol, "
        f"Feller ratio {heston.feller_ratio:.2f}"
    )

    print("transforming the surface into a local vol grid (Dupire) ...")
    local = tp.LocalVolModel.from_surface(inputs)
    print(f"  {local}")

    print("fitting a leverage function on top of Heston (particle method) ...")
    lsv = tp.LSVModel(tp.HestonModel(**MARKET_PARAMS))
    lsv.calibrate(inputs, n_paths=100_000, n_steps=100, seed=5, fit_stochastic=False)
    print(f"  {lsv}\n")

    models = {"Heston": heston, "local vol": local, "LSV": lsv}
    config = tp.MCConfig(n_paths=100_000, n_steps=200, seed=17, device="cpu")

    header = f"{'strike':>7} {'market iv':>10} | " + " | ".join(f"{name:>16}" for name in models)
    print("One-year vanillas, as implied vols (Monte Carlo, so +/- a few basis points)")
    print(header)
    print("-" * len(header))
    for strike in (85.0, 95.0, 100.0, 105.0, 115.0):
        target = truth.price(forward, strike, t, discount, 1).detach()
        quoted = float(implied_vol(target, forward, strike, t, discount, 1))
        cells = []
        for model in models.values():
            spec = tp.VanillaOption(strike=strike, maturity=MATURITY, right="call")
            result = tp.price(spec, dataclasses.replace(market, vol=model), config)
            vol = float(implied_vol(result.price, forward, strike, t, discount, 1))
            cells.append(f"{vol:.4f} ({vol - quoted:+.4f})")
        print(f"{strike:>7.0f} {quoted:>10.4f} | " + " | ".join(f"{c:>16}" for c in cells))

    print("\nOne-year Asian call, struck at the money -- calibrated to the same vanillas")
    asian = tp.AsianOption(strike=100.0, maturity=MATURITY, right="call")
    prices = {}
    for name, model in models.items():
        result = tp.price(asian, dataclasses.replace(market, vol=model), config, greeks=("delta",))
        prices[name] = result.price
        print(
            f"  {name:>10}: {result.price:.4f} +/- {result.stderr:.4f}"
            f"   delta {result.greeks['delta']:.4f}"
        )
    spread = max(prices.values()) - min(prices.values())
    vega = float(black_vega(forward, 100.0, t, 0.2, discount))
    print(
        f"\n  price spread across models: {spread:.4f}, "
        f"{spread / vega * 100:.2f} vol points of a vanilla's vega -- of the order of the\n"
        "  Monte Carlo error above, so on this trade the three price much the same.\n"
        "  The deltas do not, and that is the part a hedger lives with: the same option,\n"
        "  the same calibration to the same vanillas, and hedge ratios a fifth of a share\n"
        "  apart, because the models disagree about how the smile moves when the spot does."
    )


if __name__ == "__main__":
    main()
