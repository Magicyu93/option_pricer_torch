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
| `calibration/surface.py` | `VolSurface` — implied vol as quoted; SVI. |
| `calibration/vol_model/` | `VolModel` — the dynamics, and `to_sde()`: Black, Heston, local vol, LSV. |
| `market/snapshot.py` | `MarketSnapshot` — calibrated state at one instant. |
| `simulator/` | `SDE` + `EulerMaruyamaSimulator`, model-agnostic. |
| `instruments/` | Contract specs, and their payoffs over paths. |
| `analytics/` | Closed-form Black and semi-analytic Heston, in torch, as the reference. |
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
python -m examples.smile_models
```

## The models

Four `VolModel`s, all calibrated through one entry point,
`calibrate(CalibrationInputs)`, and all handing the simulator an SDE through
`to_sde(market)`.

| Model | Fitted by | Priced against |
| --- | --- | --- |
| `BlackScholesModel` | LBFGS on one log-vol | `analytics.black` |
| `HestonModel` | Adam then LBFGS on five parameters | `analytics.heston` |
| `LocalVolModel` | Dupire — a formula, not an optimisation | the surface it came from |
| `LSVModel` | the particle method | the same, plus Heston's forward smile |

**Heston** is priced semi-analytically, by Lewis' single integral over a
characteristic function in the branch-stable "little trap" form, with fixed
Gauss-Legendre quadrature so the price stays differentiable. Calibration is
therefore a second's work with exact gradients rather than a bumped Jacobian of
a noisy Monte Carlo. Simulation uses the full-truncation Euler scheme, and the
two agree to within the sampling error — which is the test. Vega is the parallel
shift of the instantaneous spot volatility: there is no single "the vol" in a
stochastic vol model, and a parallel bump is the scalar a desk quotes.

**SVI** is fitted per expiry in the wing-slope parameterisation, so `b > 0`,
`|rho| < 1` and Lee's moment bound hold by construction — no optimiser step can
reach an arbitrageable wing. Durrleman's butterfly condition and the calendar
condition against the previous expiry are penalised on a grid far wider than the
quotes reach, since that is where an unpenalised fit likes to hide a negative
density. Residuals are vega weighted: a quote worth a basis point implies a
volatility, but it does not carry one.

**Local vol** is not calibrated in any real sense. Dupire's formula gives the
unique diffusion consistent with an arbitrage-free surface, and the work is
evaluating it stably. The derivatives of total variance are taken **by autograd
through the fitted surface**, not by differencing quotes — which removes the
entire class of error that makes Dupire a cautionary tale. The denominator is
the implied density, so a non-positive one is a diagnosis of the input surface,
not of the formula.

**LSV** is Heston with a leverage function `L(S, t)` chosen so that
`L^2 E[v | S] = sigma_loc^2` — Gyongy's condition, which is implicit, since the
expectation is under the model whose parameter `L` is. The particle method
unrolls the fixed point in time: simulate a cloud of paths and estimate
`E[v | S]` from the particles themselves at each step, just before taking the
step that needs it. One forward pass, no iteration, no nested Monte Carlo.

`python -m examples.smile_models` calibrates all three smile models to one
quoted surface, shows them reproducing it, and then prices an Asian to show
where they part company.

## Not implemented

Curve bootstrapping from deposit and swap quotes; barriers, digitals and
American exercise; discretely-monitored Asians and seasoned averages; variance
reduction beyond antithetic sampling, including the control variate and the
quasi-random path construction the Brownian bridge exists for. Each has a named
seam.
