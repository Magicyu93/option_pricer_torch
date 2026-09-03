# torch_pricer

Monte Carlo option pricing in PyTorch, with greeks from automatic
differentiation rather than bumps.

```python
import datetime as dt
import torch_pricer as tp

market = tp.MarketSnapshot.flat("2026-01-02", spot=100.0, rate=0.03, dividend=0.01, vol=0.20)
spec = tp.VanillaOption(strike=100.0, maturity=dt.date(2027, 1, 2), right="call")

result = tp.price(spec, market, greeks=("delta", "gamma", "vega", "rho"))
print(result)   # PricingResult(8.846386 +/- 0.030691, delta=0.5737, gamma=0.0195, ...)
```

## Layout

The pipeline runs one way:

```
market_data  ->  calibration  ->  market  ->  pricer
(raw quotes)     (curves,         (snapshot  (MC engine,
                  surfaces,        of the     autograd
                  vol models)      world)     greeks)
```

| Module | Holds |
| --- | --- |
| `market_data/` | Raw quotes. Dumb dataclasses, no torch, no QuantLib. |
| `calibration/curve_model/` | `RateCurve` — pillar zeros as an `nn.Parameter`. |
| `calibration/surface.py` | `VolSurface` — implied vol as quoted. |
| `calibration/vol_model/` | `VolModel` — the dynamics, and `to_sde()`. |
| `market/snapshot.py` | `MarketSnapshot` — calibrated state at one instant. |
| `simulator/` | `SDE` + `EulerMaruyamaSimulator`, model-agnostic. |
| `instruments/` | Contract specs, and their payoffs over paths. |
| `analytics/black.py` | Closed-form Black, in torch, as the reference. |
| `pricer/engine.py` | `price()`. |

`VolModel.to_sde(market)` is the joint between calibration and simulation: the
**model supplies the diffusion, the market supplies the drift**, since the
risk-neutral drift `(r - q) S` comes from the curves rather than the vol model.

Vol models are `nn.Module`s, so parameters are autograd leaves (vega is a
backward pass), `.parameters()` feeds any torch optimiser at calibration time,
and `state_dict()` serialises a fit. Constrained parameters are stored
unconstrained — `exp` for positive quantities, `tanh` for a correlation — so an
optimiser cannot leave the feasible set.

## Greeks

Delta, vega and rho are **pathwise** derivative estimators: unbiased, all
produced by one backward pass through the simulation, with none of the bump-size
sensitivity of finite differences. Because the discount curve's pillar rates are
an `nn.Parameter`, bucketed rho is available from the same pass.

Gamma is different, and the reason is structural rather than a limitation of the
implementation. The simulated spot is `S_T = S_0 M` with `M` independent of
`S_0`, so a vanilla payoff is piecewise *linear* in spot and its second
derivative is a Dirac at the strike — zero almost everywhere. A second backward
pass returns exactly `0`, not an approximation. The engine instead differences
the pathwise delta at bumped spots under common random numbers, which is why
`MCConfig.gamma_bump` exists.

Theta is not offered. Differentiating with respect to `as_of` means
differentiating through a calendar; the honest answer is a one-day bump.

## Precision and devices

Importing the package sets torch's default dtype to `float64`. Gamma is a
difference of differences, and in `float32` the rounding noise is the same order
as the answer. `MCConfig.device="auto"` falls back to CPU whenever CUDA is
unavailable, including the driver-mismatch case where a GPU is present but
unusable.

QuantLib is used for calendars and day counts and nothing else. It cannot carry
autograd gradients, so it never appears between the spot and the price, and
there is no global evaluation date for a concurrent request to move underneath a
valuation.

## Development

```bash
pip install -e . --no-build-isolation    # --no-build-isolation only for pip < 23
pytest -q
python -m examples.european_options
```

## Not implemented

Local vol, Heston and LSV dynamics and their calibrators; SVI fitting; curve
bootstrapping from deposit and swap quotes; discretely-monitored Asians and
seasoned averages; variance reduction beyond antithetic. Each has a named seam
and a stub module.
