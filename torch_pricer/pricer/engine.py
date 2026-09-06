"""The Monte Carlo engine, and greeks by automatic differentiation.

The price is a mean of discounted payoffs over simulated paths. Every link in
that chain is a torch operation on tensors -- the spot, the vol, the discount
factor, the maturity -- so the whole thing is one differentiable function and a
backward pass gives risk directly:

    delta = d price / d spot
    vega  = d price / d vol
    rho   = d price / d (curve pillar zeros)     [a vector, one entry per pillar]
    theta = -d price / d (time to expiry)

This is the *pathwise* derivative estimator. It is unbiased wherever the payoff
is Lipschitz in the parameter, it produces all first-order greeks in roughly one
extra forward pass, and it has none of the bump-size sensitivity of finite
differences.

Second-order risk is the exception, and it is worth understanding why rather
than discovering it in production. For a vanilla call the simulated terminal
spot is ``S_T = S_0 M`` with ``M`` independent of ``S_0``, so the payoff
``max(S_0 M - K, 0)`` is piecewise *linear* in ``S_0``. Its second derivative is
a Dirac at the strike -- zero almost everywhere -- and a second backward pass
therefore returns exactly zero, not an approximation of gamma. The same defect
afflicts volga and vanna, and there it is far more dangerous: those come back
plausible rather than obviously broken, because the non-singular part of the
derivative survives while the density term is dropped.

Differentiating twice through a kink does not work in any AD framework; the
standard remedies are a likelihood-ratio estimator (needs the transition
density, which a generic SDE does not expose), a smoothed payoff (biased), or
differencing the pathwise first-order sensitivity. This engine does the last:
:func:`_bumped_sensitivity` reprices at bumped inputs reusing the *same* normal
draws, so the two sensitivities are almost perfectly correlated and their
difference is far cleaner than a difference of two prices would be. The
first-order greeks stay exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from torch_pricer.errors import PricingError, ValidationError
from torch_pricer.instruments.payoff import Payoff, payoff_for
from torch_pricer.instruments.spec import Instrument
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.base import Model
from torch_pricer.simulator.rng import NormalDraws
from torch_pricer.simulator.simulator import EulerMaruyamaSimulator

#: Greeks the engine knows how to take.
SUPPORTED_GREEKS = ("delta", "vega", "theta", "rho", "dividend_rho", "gamma")

#: Those taken by a single backward pass, and the leaf each differentiates against.
_PATHWISE_GREEKS = ("delta", "vega", "theta", "rho", "dividend_rho")


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
    """A price, its Monte Carlo error, and whatever risk was asked for.

    Bucketed risk (``rho``, ``dividend_rho``) stays a tensor, one entry per
    curve pillar; the shape is the useful part and collapsing it to a scalar
    would throw away where the exposure sits.
    """

    price: float
    stderr: float
    greeks: dict[str, float | Tensor] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        def fmt(v):
            return f"{v:.6g}" if isinstance(v, float) else f"[{v.numel()} buckets]"

        risk = "".join(f", {k}={fmt(v)}" for k, v in self.greeks.items())
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


def price(
    spec: Instrument,
    market: MarketSnapshot,
    model: Model,
    config: MCConfig | None = None,
    greeks: Sequence[str] = (),
) -> PricingResult:
    """Price one instrument by Monte Carlo under a market snapshot.

    Args:
        spec: the option contract
        market: spot, calibrated curves, calibrated vol surface
        model: calibrated model for the underlying stock
        config: Monte Carlo settings; defaults to :class:`MCConfig`
        greeks: any of :data:`SUPPORTED_GREEKS`

    Returns:
        The discounted expected payoff *per unit of underlying* -- the contract
        multiplier belongs to the position, not the instrument -- its standard
        error, and the requested risk.
    """
    config = config or MCConfig()
    unknown = tuple(g for g in greeks if g not in SUPPORTED_GREEKS)
    if unknown:
        raise ValidationError(f"unknown greeks {unknown}; supported: {SUPPORTED_GREEKS}")

    if spec.expiry is None:
        raise ValidationError(
            f"{type(spec).__name__} has no expiry, so there is no simulation horizon"
        )

    device = resolve_device(config.device)
    payoff = payoff_for(spec)
    market = market.to(device=device, dtype=config.dtype)
    model = model.to(device=device, dtype=config.dtype)  # nn.Module.to: in place

    # Maturity is a graph leaf, not a float, so theta comes off the same
    # backward pass as everything else. The calendar runs once, here, and is
    # never differentiated -- only the year fraction it produces is.
    maturity = torch.tensor(
        market.time_to(spec.expiry), dtype=config.dtype, device=device, requires_grad=True
    )

    # Randomness is drawn up front and handed to the simulator, so a price is
    # reproducible from a seed and antithetic pairs stay paired across steps.
    draws = NormalDraws(
        n_paths=config.n_paths,
        n_factors=model.n_factors,
        seed=config.seed,
        antithetic=config.antithetic,
        device=device,
        dtype=config.dtype,
    ).draw(config.n_steps)

    pv, spot, _ = _simulate(
        market, model, maturity, market.spot, payoff, draws, config, device
    )
    expected = pv.mean()

    leaves: dict[str, Tensor] = {}
    for name in _PATHWISE_GREEKS:
        if name in greeks:
            leaves[name] = _leaf_for(name, spot, maturity, market, model)

    risk: dict[str, float | Tensor] = {}
    if leaves:
        grads = torch.autograd.grad(
            expected, list(leaves.values()), retain_graph=True, allow_unused=True
        )
        for name, g in zip(leaves, grads):
            if g is None:
                raise PricingError(f"price is not differentiable with respect to {name}")
            # theta is the derivative in calendar time; we differentiated in
            # time-to-expiry, which runs the other way.
            g = -g if name == "theta" else g
            risk[name] = float(g) if g.numel() == 1 else g.detach()

    if "gamma" in greeks:
        risk["gamma"] = _gamma(market, model, payoff, draws, maturity, config, device)

    return PricingResult(
        price=float(expected.detach()),
        stderr=_stderr(pv.detach(), config.antithetic),
        greeks=risk,
    )


def _leaf_for(
    name: str, spot: Tensor, maturity: Tensor, market: MarketSnapshot, model: Model
) -> Tensor:
    """The graph leaf a given pathwise greek differentiates against."""
    if name == "delta":
        return spot
    if name == "theta":
        return maturity
    if name == "rho":
        return market.discount.pillar_zeros
    if name == "dividend_rho":
        return market.dividend.pillar_zeros
    if name == "vega":
        vol = getattr(model, "vol", None)
        if not isinstance(vol, Tensor):
            raise PricingError(
                f"{type(model).__name__} exposes no scalar 'vol' parameter, so vega is "
                "not defined for it. A model whose parameters are not quoted vols needs "
                "the chain rule through its calibration."
            )
        return vol
    raise PricingError(f"no leaf registered for greek {name!r}")


def _stderr(pv: Tensor, antithetic: bool) -> float:
    """Standard error of the mean, respecting antithetic pairing.

    ``NormalDraws`` builds the second half of the batch as the mirror of the
    first, so paths ``i`` and ``i + n/2`` are one draw, not two. Treating them
    as independent would misstate the error; the estimator is the spread of the
    ``n/2`` *pair means*.
    """
    n = pv.numel()
    if antithetic:
        half = n // 2
        pairs = 0.5 * (pv[:half] + pv[half:])
        return float(pairs.std(unbiased=True) / math.sqrt(half))
    return float(pv.std(unbiased=True) / math.sqrt(n))


def _simulate(
    market: MarketSnapshot,
    model: Model,
    T: Tensor,
    spot_value: Tensor,
    payoff: Payoff,
    draws: Tensor,
    config: MCConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor, object]:
    """One simulation at ``spot_value``. Returns ``(discounted payoffs, spot leaf, sde)``.

    The spot is rebuilt as a fresh graph leaf on every call, so repricing at a
    bumped spot cannot entangle with the base run's graph.
    """
    # Built multiplicatively rather than with ``torch.linspace(0, T, ...)``:
    # linspace's endpoint is a scalar, so a tensor T would be coerced and the
    # graph silently cut, taking theta with it.
    unit = torch.linspace(0.0, 1.0, config.n_steps + 1, device=device, dtype=config.dtype)
    t_grid = T * unit

    spot = spot_value.detach().clone().requires_grad_(True)
    sde = model.to_sde(market)
    simulator = EulerMaruyamaSimulator(sde)

    # In the SDE's own coordinate -- log-spot for GBM -- never raw spot.
    x0 = model.initial_state(market.with_spot(spot)).expand(config.n_paths, model.n_factors)

    if payoff.needs_path:
        states = simulator.simulate_with_trajectory(
            x0, t_grid, draws, progress=config.progress
        )
    else:
        states = simulator.simulate(x0, t_grid, draws, progress=config.progress)

    pv = market.discount.discount(T) * payoff(sde.asset(states), t_grid)
    return pv, spot, sde


def _gamma(
    market: MarketSnapshot,
    model: Model,
    payoff: Payoff,
    draws: Tensor,
    T: Tensor,
    config: MCConfig,
    device: torch.device,
) -> float:
    """Central difference of the pathwise delta, under common random numbers."""
    h = config.gamma_bump * max(float(market.spot.detach().abs()), 1e-8)
    deltas = []
    for offset in (+h, -h):
        pv, spot, _ = _simulate(
            market, model, T, market.spot + offset, payoff, draws, config, device
        )
        (d,) = torch.autograd.grad(pv.mean(), [spot], allow_unused=True)
        if d is None:
            raise PricingError("price is not differentiable with respect to spot")
        deltas.append(float(d.detach()))
    return (deltas[0] - deltas[1]) / (2.0 * h)
