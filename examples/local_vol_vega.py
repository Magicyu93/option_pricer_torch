"""Bucketed vega for a local-vol model, end to end.

    quoted vols --fit--> SVI slice --Dupire--> local vol --MC--> price

``dV/d(slice parameter)`` comes from one backward pass through the simulation,
because Dupire is an explicit formula and the whole chain is differentiable.
``d(slice parameter)/d(quote)`` comes from the implicit function theorem at the
calibrated point. Multiply them and you have the sensitivity to each quoted
volatility -- for one linear solve, rather than one recalibration per quote.

Run with ``python -m examples.local_vol_vega``.
"""

import datetime as dt

import torch

from torch_pricer.calibration.svi_fit import fit_slice
from torch_pricer.instruments.spec import Right, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.svi import SVISlice, SVISurface
from torch_pricer.models.local_vol import LocalVolModel
from torch_pricer.pricer.engine import MCConfig, price

AS_OF, EXPIRY, T = dt.date(2025, 1, 2), dt.date(2026, 1, 2), 1.0
STRIKES = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 125.0]
QUOTED = [0.2420, 0.2210, 0.2120, 0.2050, 0.2005, 0.1990, 0.2040]


def main() -> None:
    market = MarketSnapshot.flat(AS_OF, spot=100.0, flat_rate=0.03, flat_dividend=0.01)
    fwd = market.forward(torch.tensor(T, dtype=torch.float64))
    k = torch.log(torch.tensor(STRIKES, dtype=torch.float64) / fwd).detach()

    # 1. Fit the slice, keeping the Jacobian and Hessian at the optimum.
    slice_ = SVISlice(0.03, 0.06, -0.4, 0.0, 0.2, T)
    fit = fit_slice(slice_, k, torch.tensor(QUOTED, dtype=torch.float64))
    print(fit)
    print("  residuals (bp):", [round(float(r) * 1e4, 1) for r in fit.residuals])

    # 2. Price through Dupire, differentiating in the slice parameters.
    surface = SVISurface([slice_], forward=market.forward, as_of=AS_OF)
    print("  arbitrage:", {k_: round(v, 5) for k_, v in surface.arbitrage_report().items()})

    spec = VanillaOption(strike=105.0, maturity=EXPIRY, right=Right.CALL)
    res = price(
        spec, market, LocalVolModel(surface),
        MCConfig(n_paths=200_000, n_steps=250, seed=11, checkpoint_segments=16),
        greeks=("delta", "model_params"),
    )
    print(f"\n{spec.describe()}")
    print(f"  price {res.price:.5f} +/- {res.stderr:.5f}   delta {res.greeks['delta']:.5f}")

    params = res.greeks["model_params"]
    order = [n for n in fit.param_names]
    dv_dtheta = torch.stack([params[f"surface.slices.0.{n}"] for n in order])
    print("  dV/d(slice param): "
          + ", ".join(f"{n}={float(v):+.4f}" for n, v in zip(order, dv_dtheta)))

    # 3. Chain through the fit to get sensitivity per quoted vol.
    vega = fit.market_sensitivity(dv_dtheta)
    print(f"\n{'strike':>7} {'quoted vol':>11} {'vega per 1.00 vol':>18} {'per 1 vol pt':>13}")
    for strike, quote, v in zip(STRIKES, QUOTED, vega):
        print(f"{strike:>7.0f} {quote:>11.4f} {float(v):>18.4f} {float(v) / 100:>13.4f}")
    print(f"{'total':>7} {'':>11} {float(vega.sum()):>18.4f} {float(vega.sum()) / 100:>13.4f}")
    print("\nThe total is the parallel vega: what the option gains if every quoted")
    print("vol rises together. Individual buckets can be negative -- with fewer")
    print("parameters than quotes, lifting one quote can pull the fitted smile")
    print("down elsewhere.")


if __name__ == "__main__":
    main()
