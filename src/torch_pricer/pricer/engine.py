"""The Monte Carlo engine, and greeks by automatic differentiation.

The price is a mean of discounted payoffs over simulated paths. Every link in
that chain is a torch operation on tensors -- the spot, the vol, the discount
factor -- so the whole thing is one differentiable function and a backward pass
gives risk directly:

    delta = d price / d spot
    vega  = d price / d vol
    rho   = d price / d (curve pillar zeros)

This is the *pathwise* derivative estimator. It is unbiased wherever the payoff
is Lipschitz in the parameter, it produces all first-order greeks in roughly one
extra forward pass, and it has none of the bump-size sensitivity of finite
differences.

Gamma is the exception, and it is worth understanding why rather than
discovering it in production. For a vanilla call the simulated terminal spot is
``S_T = S_0 M`` with ``M`` independent of ``S_0``, so the payoff
``max(S_0 M - K, 0)`` is piecewise *linear* in ``S_0``. Its second derivative is
a Dirac at the strike -- zero almost everywhere -- and a second backward pass
therefore returns exactly zero, not an approximation of gamma. Differentiating
twice through a kink does not work in any AD framework; the standard remedies
are a likelihood-ratio estimator (needs the transition density, which a generic
SDE does not expose), a smoothed payoff (biased), or differencing the pathwise
delta. This engine does the last: it reprices at bumped spots reusing the *same*
normal draws, so the two deltas are almost perfectly correlated and their
difference is far cleaner than a difference of two prices would be. Delta itself
stays exact.

Theta is deliberately absent. Differentiating with respect to ``as_of`` means
differentiating through a calendar, which is not a continuous object; the honest
implementation is a one-day bump and reprice, and it belongs in a risk layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from ..errors import PricingError, ValidationError
from ..instruments.payoff import Payoff, payoff_for
from ..instruments.spec import Instrument
from ..market.snapshot import MarketSnapshot
from ..simulator.rng import NormalDraws
from ..simulator.simulator import EulerMaruyamaSimulator

#: Greeks the engine knows how to take. Theta is not among them; see the module docstring.
SUPPORTED_GREEKS = ("delta", "gamma", "vega", "rho")


@dataclass(frozen=True)
class MCConfig:
    """Monte Carlo settings.

    ``n_paths`` x ``n_steps`` bounds memory, not just time: a differentiable
    simulation retains the graph for every step. A terminal-only payoff keeps
    only the running state, so the defaults are comfortable on CPU; a
    path-dependent payoff at these sizes is not, and wants either fewer paths or
    gradient checkpointing.
    """

    n_paths: int = 100_000
    n_steps: int = 100
    seed: int = 0
    device: str = "auto"
    dtype: torch.dtype = torch.float64
    antithetic: bool = True
    progress: bool = False
    #: Relative spot bump used to difference the pathwise delta into gamma.
    #: Differencing deltas amplifies noise as 1/h while the truncation bias grows
    #: as h^2, and the delta difference is a near-binomial count of the paths
    #: whose moneyness flipped, so too *small* a bump is as bad as too large.
    #: Measured against Black on a 1y ATM call, 200k paths, over 8 seeds:
    #: 1e-3 -> -0.6% bias, sd 8.0e-4; 1e-2 -> -0.2% bias, sd 3.2e-4;
    #: 1e-1 -> -3.2% bias, sd 0.8e-4. 1e-2 is the turning point.
    gamma_bump: float = 1e-2


@dataclass(frozen=True)
class PricingResult:
    """A price, its Monte Carlo error, and whatever risk was asked for."""

    price: float
    stderr: float
    greeks: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        risk = "".join(f", {k}={v:.6g}" for k, v in self.greeks.items())
        return f"PricingResult({self.price:.6f} +/- {self.stderr:.6f}{risk})"


def resolve_device(name: str = "auto") -> torch.device:
    """Pick a device, degrading to CPU when CUDA was asked for but is unusable.

    ``torch.cuda.is_available()`` is False for a driver mismatch as readily as
    for a machine with no GPU, so ``"auto"`` must not assume.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PricingError("device='cuda' requested but no usable CUDA device is present")
    return device


def _grad(output: Tensor, inputs: Sequence[Tensor], create_graph: bool = False):
    """``autograd.grad`` that tolerates inputs the price does not depend on."""
    return torch.autograd.grad(
        output, list(inputs), create_graph=create_graph, retain_graph=True, allow_unused=True
    )


def price(
    spec: Instrument,
    market: MarketSnapshot,
    config: MCConfig | None = None,
    greeks: Sequence[str] = (),
) -> PricingResult:
    """Price one instrument by Monte Carlo under a market snapshot.

    Args:
        spec: the contract
        market: calibrated curves, vol model and spot
        config: Monte Carlo settings; defaults to :class:`MCConfig`
        greeks: any of :data:`SUPPORTED_GREEKS`

    Returns:
        The discounted expected payoff, its standard error, and the requested
        risk.
    """
    config = config or MCConfig()
    want = set(greeks)
    unknown = want - set(SUPPORTED_GREEKS)
    if unknown:
        raise ValidationError(f"unknown greeks {sorted(unknown)}; known: {list(SUPPORTED_GREEKS)}")

    expiry = spec.expiry
    if expiry is None:
        raise ValidationError(f"{spec.describe()} has no expiry to simulate to")
    T = market.time_to(expiry)
    if T <= 0:
        raise ValidationError(f"{spec.describe()} expires on or before the snapshot date")

    device = resolve_device(config.device)
    market = market.to(device=device, dtype=config.dtype)
    payoff = payoff_for(spec)

    grid = torch.linspace(0.0, T, config.n_steps + 1, device=device, dtype=config.dtype)
    # Drawn once and reused for every repricing below: common random numbers are
    # what make the gamma difference usable.
    draws = NormalDraws(
        n_paths=config.n_paths,
        n_factors=market.vol.to_sde(market).n_factors,
        seed=config.seed,
        antithetic=config.antithetic,
        device=device,
        dtype=config.dtype,
    ).draw(config.n_steps)

    # Every supported greek needs the graph; a bare price does not and should
    # not pay for it.
    pv, spot, sde = _simulate(
        market, payoff, market.spot, grid, draws, T, config, no_grad=not want
    )
    expected = pv.mean()

    risk: dict[str, float] = {}
    if want & {"delta", "gamma"}:
        (delta,) = _grad(expected, [spot])
        if delta is None:
            raise PricingError("price is not differentiable with respect to spot")
        if "delta" in want:
            risk["delta"] = float(delta.detach())
    if "gamma" in want:
        risk["gamma"] = _gamma(market, payoff, grid, draws, T, config)
    if "vega" in want:
        params = sde.risk_parameters()
        if "vol" not in params:
            raise PricingError(f"{type(sde).__name__} exposes no 'vol' to take vega against")
        (vega,) = _grad(expected, [params["vol"]])
        risk["vega"] = 0.0 if vega is None else float(vega.detach())
    if "rho" in want:
        # Parallel rho is the sum of the bucketed pillar sensitivities.
        (buckets,) = _grad(expected, [market.discount.pillar_zeros])
        risk["rho"] = 0.0 if buckets is None else float(buckets.detach().sum())

    return PricingResult(
        price=float(expected.detach()),
        stderr=float(pv.detach().std(unbiased=True) / math.sqrt(config.n_paths)),
        greeks=risk,
    )


def _simulate(
    market: MarketSnapshot,
    payoff: Payoff,
    spot_value: Tensor,
    grid: Tensor,
    draws: Tensor,
    T: float,
    config: MCConfig,
    no_grad: bool,
) -> tuple[Tensor, Tensor, object]:
    """One simulation at ``spot_value``. Returns ``(discounted payoffs, spot leaf, sde)``.

    The spot is rebuilt as a fresh graph leaf on every call, so repricing at a
    bumped spot cannot entangle with the base run's graph.
    """
    spot = spot_value.detach().clone().requires_grad_(not no_grad)
    marked = market.with_spot(spot)
    sde = marked.vol.to_sde(marked)

    simulator = EulerMaruyamaSimulator(sde)
    x0 = sde.initial_state(spot).expand(config.n_paths, sde.dim)
    walk = simulator.simulate_with_trajectory if payoff.needs_path else simulator.simulate
    states = walk(x0, grid, draws, no_grad=no_grad, progress=config.progress)

    pv = marked.discount.discount(T) * payoff(sde.asset(states), grid)
    return pv, spot, sde


def _gamma(
    market: MarketSnapshot,
    payoff: Payoff,
    grid: Tensor,
    draws: Tensor,
    T: float,
    config: MCConfig,
) -> float:
    """Central difference of the pathwise delta, under common random numbers."""
    h = config.gamma_bump * float(market.spot.detach().abs().clamp_min(1.0))
    deltas = []
    for offset in (+h, -h):
        pv, spot, _ = _simulate(
            market, payoff, market.spot + offset, grid, draws, T, config, no_grad=False
        )
        (d,) = _grad(pv.mean(), [spot])
        if d is None:
            raise PricingError("price is not differentiable with respect to spot")
        deltas.append(float(d.detach()))
    return (deltas[0] - deltas[1]) / (2.0 * h)
