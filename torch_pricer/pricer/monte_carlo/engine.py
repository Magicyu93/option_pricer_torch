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
than discovering it in production. Ask for ``gamma_autograd`` alongside
``gamma`` to see the difference on any model. For a vanilla call the simulated terminal
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

import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from torch_pricer.errors import PricingError, ValidationError
from torch_pricer.instruments.payoff import Payoff, intrinsic_for, payoff_for
from torch_pricer.instruments.spec import Instrument, Style, VanillaOption
from torch_pricer.pricer.monte_carlo import lsm as lsm_mod
from torch_pricer.pricer.monte_carlo.lsm import LSMConfig
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.base import Model
from torch_pricer.simulator.monte_carlo.rng import NormalDraws
from torch_pricer.simulator.monte_carlo.simulator import EulerMaruyamaSimulator

#: Greeks the engine knows how to take.
#:
#: ``model_params`` is ``dV/d(parameter)`` for every parameter the model exposes,
#: returned as a dict keyed by parameter name. For Black-Scholes that *is* vega,
#: because the parameter is the quoted vol. For Heston or local vol it is not:
#: those are sensitivities to fitted quantities, and turning them into market
#: vega means chaining through the calibration with
#: :meth:`~torch_pricer.calibration.result.CalibrationResult.market_sensitivity`.
SUPPORTED_GREEKS = (
    "delta", "vega", "theta", "rho", "dividend_rho", "gamma", "gamma_autograd",
    "model_params",
)

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
    #: Retain only this many intermediate states and recompute the rest during
    #: the backward pass. 0 disables it.
    #:
    #: A state-dependent diffusion retains a graph proportional to
    #: n_paths x n_steps. Measured on Dupire local vol, 100k paths x 200 steps,
    #: as bytes saved for the backward pass: 8242 MB undamped, falling roughly
    #: as 1/segments (4 -> 616 MB, 8 -> 1229, 14 -> 2149, 28 -> 4296). Peak is
    #: that plus one segment's recomputation, ~= 153 * n + 8242 / n MB here,
    #: which bottoms out near n = sqrt(8242 / 153) ~= 7 -- the usual
    #: sqrt(n_steps) rule. Cost is one extra forward pass: 2.6s -> 3.5s.
    #:
    #: Black-Scholes retains 918 MB at the same size and does not need this;
    #: its coefficients do not depend on the state.
    #:
    #: Numerically transparent: price and greeks are bit-identical either way.
    checkpoint_segments: int = 0
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
    greeks: dict[str, float | Tensor | dict[str, Tensor]] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.6g}"
            if isinstance(v, dict):
                return "{" + ", ".join(f"{k}={float(x):.4g}" for k, x in v.items()) + "}"
            return f"[{v.numel()} buckets]"

        risk = "".join(f", {k}={fmt(v)}" for k, v in self.greeks.items())
        return f"PricingResult({self.price:.6f} +/- {self.stderr:.6f}{risk})"


#: Cached ``(usable, why not)`` from the first CUDA probe of the process.
_CUDA_PROBE: tuple[bool, str] | None = None


def _cuda_available() -> tuple[bool, str]:
    """``(usable, why not)``. Probing CUDA can warn; the caller decides if that matters.

    Cached, and not only to save the call. Torch emits its diagnostic exactly
    once per process, so without this the *first* probe would capture the reason
    and every later one would see silence -- meaning an explicit
    ``device="cuda"`` would report a bare failure purely because something had
    already asked for ``"auto"``. Whether CUDA works does not change inside a
    process; fixing a driver mismatch takes a reboot.
    """
    global _CUDA_PROBE
    if _CUDA_PROBE is None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            usable = torch.cuda.is_available()
        reason = "; ".join(str(w.message).split("\n")[0] for w in caught)
        _CUDA_PROBE = (usable, reason)
    return _CUDA_PROBE


def resolve_device(name: str = "auto") -> torch.device:
    """Pick a device, degrading to CPU when CUDA was asked for but is unusable.

    ``torch.cuda.is_available()`` is False for a driver mismatch as readily as
    for a machine with no GPU, so ``"auto"`` must not assume.

    A driver/library mismatch -- the userspace NVIDIA libraries not matching the
    loaded kernel module, which is what a driver upgrade without a reboot leaves
    behind -- makes that probe emit a ``UserWarning`` before returning False.
    Under ``"auto"`` falling back to CPU *is* the contract, so the warning is
    swallowed rather than printed on every run. Under an explicit
    ``device="cuda"`` it is the answer to the question the caller asked, so it
    is carried into the error instead.
    """
    if name == "auto":
        usable, _ = _cuda_available()
        return torch.device("cuda" if usable else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        usable, reason = _cuda_available()
        if not usable:
            detail = f": {reason}" if reason else ""
            raise PricingError(
                f"device='cuda' requested but no usable CUDA device is present{detail}"
            )
    return device


def price(
    spec: Instrument,
    market: MarketSnapshot,
    model: Model,
    config: MCConfig | None = None,
    greeks: Sequence[str] = (),
    lsm: LSMConfig | None = None,
) -> PricingResult:
    """Price one instrument by Monte Carlo under a market snapshot.

    Args:
        spec: the option contract
        market: spot, calibrated curves, calibrated vol surface
        model: calibrated model for the underlying stock
        config: Monte Carlo settings; defaults to :class:`MCConfig`
        greeks: any of :data:`SUPPORTED_GREEKS`
        lsm: exercise-policy settings, used only for an American contract;
            defaults to :class:`~torch_pricer.pricer.monte_carlo.lsm.LSMConfig`

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

    # An early-exercisable contract takes a different route: payoff_for refuses
    # it outright, and rightly, so the intrinsic is asked for explicitly instead.
    american = isinstance(spec, VanillaOption) and spec.style is Style.AMERICAN
    if isinstance(spec, VanillaOption) and spec.style is Style.BERMUDAN:
        raise ValidationError(
            "Bermudan exercise needs its dates aligned to the simulation grid, "
            "which is not wired up; use Style.AMERICAN with "
            "LSMConfig.n_exercise_dates for a fixed number of evenly spaced dates"
        )
    payoff = intrinsic_for(spec) if american else payoff_for(spec)
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

    if american:
        lsm_config = lsm or LSMConfig()
        required: tuple[int, ...] = ()
        if lsm_config.align_to_dividends and market.dividends:
            required = lsm_mod.pre_dividend_indices(
                market.dividends.times(market.as_of, market.day_count),
                config.n_steps, float(maturity.detach()),
            )
        indices = lsm_mod.exercise_indices(
            config.n_steps, lsm_config.n_exercise_dates, required
        )
        coefficients = _fit_lsm_policy(
            market, model, maturity, payoff, spec.strike, indices,
            lsm_config, config, device,
        )

        def pv_of(states, sde, t_grid):
            return lsm_mod.value(
                asset_path(sde, states, t_grid, market, maturity),
                market.discount.discount(t_grid), payoff,
                spec.strike, indices, lsm_config, coefficients,
            )
    else:
        def pv_of(states, sde, t_grid):
            return _european_pv(states, sde, payoff, t_grid, market, maturity)

    states, spot, sde, t_grid = _simulate(
        market, model, maturity, market.spot, payoff, draws, config, device,
        keep_path=american,
    )
    pv = pv_of(states, sde, t_grid)
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

    if "gamma_autograd" in greeks:
        risk["gamma_autograd"] = _gamma_autograd(expected, spot)

    if "model_params" in greeks:
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not named:
            raise PricingError(f"{type(model).__name__} exposes no parameters")
        grads = torch.autograd.grad(
            expected, [p for _, p in named], retain_graph=True, allow_unused=True
        )
        risk["model_params"] = {
            n: (torch.zeros_like(p) if g is None else g.detach())
            for (n, p), g in zip(named, grads)
        }

    if "gamma" in greeks:
        risk["gamma"] = _gamma(
            market, model, payoff, draws, maturity, config, device, pv_of,
            keep_path=american,
        )

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


def _gamma_autograd(expected: Tensor, spot: Tensor) -> float:
    """Second derivative of the price in spot, straight from a second backward pass.

    Exposed as a diagnostic, not as a number to risk-manage on. What it returns
    depends on the model, and the difference is worth seeing:

    * **Constant or state-independent coefficients** (Black-Scholes, Heston):
      ``S_T = S_0 M`` with ``M`` independent of ``S_0``, so the payoff is
      piecewise *linear* in spot. Its second derivative is a Dirac at the strike,
      which autograd evaluates as zero almost everywhere. Expect zero to
      floating-point roundoff -- around ``1e-18``, or ``1e-16`` of the true
      gamma. Not small, not noisy: structurally absent.
    * **State-dependent diffusion** (local vol): ``sigma_LV(S_t, t)`` makes the
      path a nonlinear function of ``S_0``, so a genuine second-order term
      survives and the result is *not* zero. It is still incomplete -- the
      density term at the strike is dropped just the same -- so it is a lower
      bound on gamma of unknown tightness, not gamma.

    Either way :func:`_gamma` is the number to use. This one is here so the
    failure is visible rather than folklore.
    """
    (delta,) = torch.autograd.grad(expected, [spot], create_graph=True, retain_graph=True)
    (curvature,) = torch.autograd.grad(
        delta, [spot], retain_graph=True, allow_unused=True
    )
    # ``None`` means the graph carries no second-order dependence at all, which
    # is the piecewise-linear case above; report it as the zero it is.
    return 0.0 if curvature is None else float(curvature.detach())


def _fit_lsm_policy(
    market: MarketSnapshot,
    model: Model,
    T: Tensor,
    intrinsic: Payoff,
    strike: float,
    indices: list[int],
    lsm_config: LSMConfig,
    config: MCConfig,
    device: torch.device,
) -> dict[int, Tensor]:
    """Fit the exercise policy, on independent paths when asked for.

    Separate draws, separate seed. Fitting and exercising on one sample lets the
    policy see each path's own future and exercise with hindsight, which biases
    the price upward; the split costs one extra simulation and buys a number that
    is honestly a lower bound.
    """
    n_paths = lsm_config.policy_paths
    if n_paths is None:
        n_paths = config.n_paths
        seed = config.seed
    else:
        seed = lsm_config.policy_seed
    if config.antithetic and n_paths % 2:
        n_paths += 1

    draws = NormalDraws(
        n_paths=n_paths, n_factors=model.n_factors, seed=seed,
        antithetic=config.antithetic, device=device, dtype=config.dtype,
    ).draw(config.n_steps)
    # _simulate sizes the initial state from the config, not from the draws, so
    # the policy run needs a config that agrees with its own path count.
    policy_config = dataclasses.replace(config, n_paths=n_paths, seed=seed)
    with torch.no_grad():
        states, _, sde, t_grid = _simulate(
            market, model, T, market.spot, intrinsic, draws, policy_config, device,
            keep_path=True,
        )
        return lsm_mod.fit_policy(
            asset_path(sde, states, t_grid, market, T),
            market.discount.discount(t_grid), intrinsic,
            strike, indices, lsm_config,
        )


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
    keep_path: bool = False,
) -> tuple[Tensor, Tensor, object, Tensor]:
    """One simulation at ``spot_value``. Returns ``(states, spot leaf, sde, t_grid)``.

    The spot is rebuilt as a fresh graph leaf on every call, so repricing at a
    bumped spot cannot entangle with the base run's graph.

    Discounting is deliberately *not* applied here. A European payoff pays once,
    at ``T``, and one discount factor covers every path; an early-exercisable one
    pays at a stopping time that differs path by path, so there is no single
    factor to apply. Leaving it to the caller is what lets both share this.

    ``keep_path`` forces the whole trajectory to be retained even when the payoff
    would not need it -- Longstaff-Schwartz regresses on the path, so it does.
    """
    # Built multiplicatively rather than with ``torch.linspace(0, T, ...)``:
    # linspace's endpoint is a scalar, so a tensor T would be coerced and the
    # graph silently cut, taking theta with it.
    unit = torch.linspace(0.0, 1.0, config.n_steps + 1, device=device, dtype=config.dtype)
    t_grid = T * unit

    spot = spot_value.detach().clone().requires_grad_(True)

    # Escrowed-spot model: the lognormal part is the spot less the present value
    # of the dividends it will pay before expiry. Differentiating still happens
    # against the *real* spot leaf -- the escrowed amount does not depend on it,
    # so dS*/dS is one and delta stays a derivative in the observable.
    diffusing = spot - _dividend_pv(market, torch.zeros_like(spot.detach()), T)
    sde = model.to_sde(market)
    simulator = EulerMaruyamaSimulator(sde)

    # In the SDE's own coordinate -- log-spot for GBM -- never raw spot.
    x0 = model.initial_state(
        market.with_spot(diffusing)
    ).expand(config.n_paths, model.n_factors)

    if payoff.needs_path or keep_path:
        states = simulator.simulate_with_trajectory(
            x0, t_grid, draws, progress=config.progress,
            checkpoint_segments=config.checkpoint_segments,
        )
    else:
        states = simulator.simulate(
            x0, t_grid, draws, progress=config.progress,
            checkpoint_segments=config.checkpoint_segments,
        )

    return states, spot, sde, t_grid


def _dividend_pv(market: MarketSnapshot, at: Tensor, T: Tensor) -> Tensor:
    """PV of the dividends still ahead of each time in ``at``, before expiry."""
    if not market.dividends:
        return torch.zeros_like(at)
    times = market.dividends.times(market.as_of, market.day_count)
    return market.dividends.pv_remaining(
        at, times, market.discount, float(T.detach())
    )


def asset_path(sde, states: Tensor, t_grid: Tensor, market: MarketSnapshot, T: Tensor):
    """Real asset levels along the path: the escrowed process plus the add-back.

    At expiry nothing is left to add, so a European payoff is unaffected beyond
    its reduced starting point. An interim exercise decision is not: it must be
    taken against the spot the holder would actually receive.
    """
    levels = sde.asset(states)
    if not market.dividends:
        return levels
    add_back = _dividend_pv(market, t_grid, T)
    # A terminal-only payoff gets levels of shape (n_paths,) and wants the single
    # add-back at expiry -- which is zero, every dividend before T having been
    # paid. A retained trajectory is (n_paths, n_steps + 1) and wants the whole
    # grid broadcast across paths.
    if levels.dim() == 1:
        return levels + add_back[-1]
    return levels + add_back.unsqueeze(0)


def _european_pv(
    states: Tensor, sde, payoff: Payoff, t_grid: Tensor, market: MarketSnapshot, T: Tensor
) -> Tensor:
    """Discounted payoff per path for a contract that pays only at ``T``."""
    return market.discount.discount(T) * payoff(
        asset_path(sde, states, t_grid, market, T), t_grid
    )


def _gamma(
    market: MarketSnapshot,
    model: Model,
    payoff: Payoff,
    draws: Tensor,
    T: Tensor,
    config: MCConfig,
    device: torch.device,
    pv_of,
    keep_path: bool = False,
) -> float:
    """Central difference of the pathwise delta, under common random numbers.

    Takes the same ``pv_of`` the base price used, so an American contract is
    differenced through its own exercise policy rather than through a European
    payoff. The policy is held fixed across the two bumps -- it is common random
    numbers applied to the decision as well as the draws, and re-fitting at each
    bumped spot would inject regression noise straight into the difference.
    """
    h = config.gamma_bump * max(float(market.spot.detach().abs()), 1e-8)
    deltas = []
    for offset in (+h, -h):
        states, spot, sde, t_grid = _simulate(
            market, model, T, market.spot + offset, payoff, draws, config, device,
            keep_path=keep_path,
        )
        pv = pv_of(states, sde, t_grid)
        (d,) = torch.autograd.grad(pv.mean(), [spot], allow_unused=True)
        if d is None:
            raise PricingError("price is not differentiable with respect to spot")
        deltas.append(float(d.detach()))
    return (deltas[0] - deltas[1]) / (2.0 * h)
