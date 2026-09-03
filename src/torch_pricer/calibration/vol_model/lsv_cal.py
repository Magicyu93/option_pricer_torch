"""Local-stochastic volatility calibration: the leverage function.

The leverage function is pinned by one condition, due to Gyongy. A stochastic
volatility model reproduces every European price of a local volatility model
exactly when its diffusion matches the local one *in conditional expectation*:

    L^2(S, t) E[ v_t | S_t = S ] = sigma_loc^2(S, t)

so ``L(S, t) = sigma_loc(S, t) / sqrt(E[v_t | S_t = S])``. Read that carefully:
the right-hand side is an expectation *under the model whose parameter L is*.
The equation is implicit, which is what makes this the one calibration in the
package that cannot be written as a formula or handed to an optimiser.

**The particle method resolves it in one forward pass.** Because ``L`` at time
``t`` only ever needs the law of ``(S_t, v_t)``, which depends on ``L`` at
earlier times alone, the fixed point unrolls in time: simulate a cloud of
particles, and at each step estimate ``E[v | S]`` from the particles themselves
before taking the step that needs it. One simulation, no iteration, no nested
Monte Carlo. The conditional expectation is estimated by binning the particles
in log-moneyness -- a kernel regression with a top-hat kernel -- and the two
knobs that matter are the bin population, below which an estimate is noise, and
a light smoothing across bins.

The leverage surface that comes out is a function of the absolute level, for
the reason given in :mod:`torch_pricer.simulator.local_vol`: a moneyness
anchored one would move with the spot and hand back a different model's delta.
Be aware of what that means for risk here, though. The leverage is steep in
``S`` -- being steep is how it cancels the stochastic model's skew -- so an LSV
delta taken against a *fixed* leverage surface is a long way from the flat-vol
delta the same model reprices at, and further from it than a local vol delta
would be. That is a statement about the smile dynamics the model has been asked
to hold, and a desk that wants the other convention re-anchors the surface and
recalibrates rather than differentiating something else.

Where no particle goes, nothing is known. Bins that never fill fall back to the
unconditional ``E[v_t]``, which is the right answer in the limit of no
information about the conditioning and keeps the surface finite in the far wings
that the calibration's own paths never reach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ...errors import CalibrationError
from ...simulator.local_vol import LevelSurface
from ...simulator.lsv import LocalStochasticVol
from ...simulator.rng import NormalDraws
from ...simulator.simulator import EulerMaruyamaSimulator
from ...tensors import EPS, as_tensor
from ..inputs import CalibrationInputs
from .base import VolModel
from .heston_cal import HestonModel
from .local_vol_cal import LocalVolModel

if TYPE_CHECKING:  # pragma: no cover
    from ...market.snapshot import MarketSnapshot
    from ...simulator.simulator import SDE

#: A bin holding fewer particles than this is estimating a conditional mean from
#: noise; it is replaced by the unconditional mean instead.
MIN_PARTICLES_PER_BIN = 32
#: Leverage is clamped to this band. It multiplies a volatility, so 10 is
#: already far outside anything a calibrated surface asks for, and the clamp
#: exists to stop a wing with no particles from putting an infinity into a grid.
MIN_LEVERAGE, MAX_LEVERAGE = 0.05, 10.0


def _nearest_fill(values: Tensor, valid: Tensor, fallback: Tensor) -> Tensor:
    """Replace the invalid entries of ``values`` with the nearest valid one.

    Extending the nearest estimate outwards is much better than any global
    default. ``E[v | k]`` is smooth and steeply monotone in the wings -- under a
    negative spot/vol correlation it can be an order of magnitude larger three
    standard deviations down than at the money -- so an unconditional mean put
    into an empty wing bin is not a neutral choice; it is a badly wrong one, and
    the paths that later wander into that bin carry the error back into the
    price.
    """
    n = values.numel()
    index = torch.arange(n, device=values.device)
    before = torch.cummax(torch.where(valid, index, torch.full_like(index, -1)), 0).values
    after = torch.flip(
        torch.cummin(
            torch.flip(torch.where(valid, index, torch.full_like(index, 2 * n)), [0]), 0
        ).values,
        [0],
    )
    # Sentinels are further away than any real index, so the comparison picks
    # the valid side whenever there is exactly one.
    to_before = torch.where(before >= 0, index - before, torch.full_like(index, 4 * n))
    to_after = torch.where(after < n, after - index, torch.full_like(index, 4 * n))
    nearest = torch.where(to_before <= to_after, before, after).clamp(0, n - 1)
    filled = values[nearest]
    return torch.where(valid.any(), filled, fallback.expand_as(filled))


def _conditional_variance(
    k: Tensor,
    v: Tensor,
    edges: Tensor,
    n_bins: int,
    min_particles: int,
    fallback: Tensor,
) -> Tensor:
    """``E[v | k]`` on the bin centres, by binning the particles.

    A top-hat kernel regression: bins are narrow enough that the curvature of
    ``E[v | k]`` across one is negligible, and it costs a single scatter rather
    than a ``n_paths x n_bins`` kernel matrix at every step.
    """
    idx = torch.bucketize(k.flatten().contiguous(), edges)
    sums = torch.zeros(n_bins, dtype=v.dtype, device=v.device).scatter_add_(0, idx, v.flatten())
    counts = torch.zeros(n_bins, dtype=v.dtype, device=v.device).scatter_add_(
        0, idx, torch.ones_like(v.flatten())
    )
    mean = sums / counts.clamp_min(1.0)
    return _nearest_fill(mean, counts >= min_particles, fallback)


def _smooth(values: Tensor) -> Tensor:
    """One pass of a ``[1, 2, 1] / 4`` filter, edges replicated.

    Trades variance for bias, and the bias is not small: ``E[v | k]`` is convex,
    so averaging a point against its neighbours pulls it up, which pulls the
    leverage down and the whole surface with it. Off by default; see
    :meth:`LSVModel.calibrate`.
    """
    padded = torch.cat([values[:1], values, values[-1:]])
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


class LSVModel(VolModel):
    """Heston plus a leverage function fitted to a Dupire surface.

    Uncalibrated, the leverage is identically one and this *is* Heston: the
    grid it starts with is a placeholder, and :meth:`calibrate` replaces it with
    one on the simulation's own time grid.
    """

    def __init__(self, stochastic: HestonModel | None = None):
        super().__init__()
        self.stochastic = stochastic if stochastic is not None else HestonModel()
        self._install(
            times=torch.tensor([0.0, 1.0]),
            log_moneyness=torch.tensor([-1.0, 1.0]),
            leverage=torch.ones(2, 2),
            reference_spot=torch.tensor(1.0),
        )

    def _install(self, times, log_moneyness, leverage, reference_spot) -> None:
        self.register_buffer("times", as_tensor(times).flatten())
        self.register_buffer("log_moneyness", as_tensor(log_moneyness).flatten())
        self.register_buffer("reference_spot", as_tensor(reference_spot).detach().reshape(()))
        self.leverage = nn.Parameter(as_tensor(leverage).detach().clone())

    # -- the underlying Heston parameters --------------------------------
    @property
    def v0(self) -> Tensor:
        return self.stochastic.v0

    @property
    def kappa(self) -> Tensor:
        return self.stochastic.kappa

    @property
    def theta(self) -> Tensor:
        return self.stochastic.theta

    @property
    def xi(self) -> Tensor:
        return self.stochastic.xi

    @property
    def rho(self) -> Tensor:
        return self.stochastic.rho

    # -- pricing ---------------------------------------------------------
    def reference_forward(self, discount, dividend):
        """``t -> F(t)`` at the calibration spot; see :mod:`torch_pricer.simulator.local_vol`."""
        spot = self.reference_spot

        def forward(t: Tensor) -> Tensor:
            t = as_tensor(t)
            return spot * dividend.discount(t) / discount.discount(t)

        return forward

    def leverage_surface(self, discount, dividend) -> LevelSurface:
        """The interpolator the SDE reads ``L(S, t)`` from."""
        return LevelSurface(
            self.times,
            self.log_moneyness,
            self.leverage,
            self.reference_forward(discount, dividend),
        )

    def to_sde(self, market: MarketSnapshot) -> SDE:
        return LocalStochasticVol(
            leverage=self.leverage_surface(market.discount, market.dividend),
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
            discount=market.discount,
            dividend=market.dividend,
        )

    # -- calibration -----------------------------------------------------
    def calibrate(
        self,
        inputs: CalibrationInputs,
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: int = 0,
        horizon: float | None = None,
        fit_stochastic: bool = True,
        min_particles: int = MIN_PARTICLES_PER_BIN,
        smooth: bool = False,
    ) -> None:
        """Fit the leverage function by the particle method. Mutates in place.

        Args:
            inputs: market state, an implied ``surface``, and (to fit the
                stochastic part) ``quotes``
            n_paths: particles; the conditional expectation is only as good as
                the number of them that land in each bin
            n_steps: steps of the calibrating simulation, and rows of the
                leverage grid it produces
            seed: seed for the particle cloud
            horizon: last time to calibrate to, in years; defaults to the far
                end of the local vol surface
            fit_stochastic: also calibrate the Heston parameters underneath, to
                ``inputs.quotes``, before fitting the leverage. Turn it off to
                keep a chosen set -- the vol of vol and the correlation are what
                govern the forward smile, and a desk often wants to *choose*
                them rather than fit them
            min_particles: bin population below which the unconditional mean is
                used instead
            smooth: apply :func:`_smooth` to each row of the conditional
                expectation. Off by default: measured against a flat 20%
                surface it costs about half a volatility point, because
                ``E[v | k]`` is convex and averaging across neighbours biases it
                upward, which biases the leverage down. Worth turning on only
                when the particle count is too low for the bin estimates to
                stand on their own
        """
        if inputs.surface is None:
            raise CalibrationError(
                "the leverage function is fitted to a local vol surface; pass an implied "
                "surface in the inputs (SVISurface.fit builds one from quotes)"
            )
        if fit_stochastic and inputs.quotes is not None and inputs.quotes.options:
            self.stochastic.calibrate(inputs)

        local = LocalVolModel.from_surface(inputs)
        k_grid = local.log_moneyness
        n_k = k_grid.numel()
        edges = 0.5 * (k_grid[1:] + k_grid[:-1])
        horizon = float(local.times.max()) if horizon is None else float(horizon)

        grid = torch.linspace(0.0, horizon, int(n_steps) + 1, dtype=k_grid.dtype)
        leverage = torch.ones(int(n_steps) + 1, n_k, dtype=k_grid.dtype)
        forward = local.reference_forward(inputs.discount, inputs.dividend)
        local_vol = LevelSurface(local.times, k_grid, local.local_vol.detach(), forward)

        with torch.no_grad():
            # The SDE reads the leverage grid it is being handed, one row at a
            # time, and an Euler step at grid[j] only ever touches row j -- so
            # filling the rows as the simulation advances is enough to make this
            # the very model being calibrated, rather than an approximation of it.
            sde = LocalStochasticVol(
                leverage=LevelSurface(grid, k_grid, leverage, forward),
                v0=self.v0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho,
                discount=inputs.discount, dividend=inputs.dividend,
                vol_shift=torch.zeros((), dtype=k_grid.dtype),
            )
            simulator = EulerMaruyamaSimulator(sde)
            draws = NormalDraws(
                n_paths=int(n_paths), n_factors=2, seed=int(seed), dtype=k_grid.dtype
            ).draw(int(n_steps))
            x = sde.initial_state(inputs.spot.detach()).expand(int(n_paths), 2).clone()

            for j in range(int(n_steps) + 1):
                t = grid[j]
                variance = x[:, 1:2].clamp_min(0.0)
                k = x[:, :1] - torch.log(forward(t))
                conditional = _conditional_variance(
                    k, variance, edges, n_k, int(min_particles), variance.mean()
                )
                if smooth:
                    conditional = _smooth(conditional)
                leverage[j] = (local_vol.row(t) / conditional.clamp_min(EPS).sqrt()).clamp(
                    MIN_LEVERAGE, MAX_LEVERAGE
                )
                if j == int(n_steps):
                    break
                x = simulator.step(x, t, grid[j + 1] - t, draws[:, j])

        self._install(grid, k_grid, leverage, inputs.spot)

    def extra_repr(self) -> str:  # pragma: no cover
        lo, hi = self.leverage.detach().min(), self.leverage.detach().max()
        return (
            f"{self.times.numel()}x{self.log_moneyness.numel()} leverage grid, "
            f"L in [{float(lo):.4f}, {float(hi):.4f}]"
        )
