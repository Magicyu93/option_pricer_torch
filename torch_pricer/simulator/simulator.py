"""Path generation for

    dX = mu(X, t) dt + sigma(X, t) dW

The time grid is one-dimensional and shared across paths. Every SDE here has
time-deterministic coefficients, so a per-path time axis would buy nothing and
cost a broadcast on every step.

Shapes are the contract, and they are worth stating once:

* state ``x``            -- ``(n_paths, dim)``
* time grid ``ts``       -- ``(n_steps + 1,)``
* draws                  -- ``(n_paths, n_steps, n_factors)``
* drift                  -- ``(n_paths, dim)``
* diffusion              -- ``(n_paths, dim, n_factors)``  *a matrix*

The diffusion is matrix-valued rather than elementwise even for the
single-factor models, because a correlated multi-factor model -- Heston's
``(log S, v)`` driven by two correlated Brownians -- cannot be expressed any
other way, and widening the contract later would mean touching every model.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
from tqdm import tqdm


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, z: torch.Tensor):
        """Take one simulation step.

        Args:
            xt: state at time ``t``, shape ``(n_paths, dim)``
            t: time, shape ``()``
            h: step size, shape ``()``
            z: standard normals for this step, shape ``(n_paths, n_factors)``

        Returns:
            state at ``t + h``, shape ``(n_paths, dim)``
        """

    def _steps(self, ts: torch.Tensor, progress: bool):
        rng = range(ts.numel() - 1)
        return tqdm(rng, desc="simulating") if progress else rng

    def simulate(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        draws: torch.Tensor,
        no_grad: bool = False,
        progress: bool = False,
    ) -> torch.Tensor:
        """Integrate to ``ts[-1]``, keeping only the final state.

        Args:
            x: initial state at ``ts[0]``, shape ``(n_paths, dim)``
            ts: time grid, shape ``(n_steps + 1,)``
            draws: standard normals, shape ``(n_paths, n_steps, n_factors)``

        Returns:
            final state, shape ``(n_paths, dim)``
        """
        with torch.no_grad() if no_grad else nullcontext():
            for t_idx in self._steps(ts, progress):
                h = ts[t_idx + 1] - ts[t_idx]
                x = self.step(x, ts[t_idx], h, draws[:, t_idx])
            return x

    def simulate_with_trajectory(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        draws: torch.Tensor,
        no_grad: bool = False,
        progress: bool = False,
    ) -> torch.Tensor:
        """Integrate to ``ts[-1]``, retaining every state.

        Returns:
            trajectory, shape ``(n_paths, n_steps + 1, dim)``
        """
        with torch.no_grad() if no_grad else nullcontext():
            xs = [x]
            for t_idx in self._steps(ts, progress):
                h = ts[t_idx + 1] - ts[t_idx]
                x = self.step(x, ts[t_idx], h, draws[:, t_idx])
                xs.append(x)
            return torch.stack(xs, dim=1)


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift, shape ``(n_paths, dim)``, for state ``(n_paths, dim)`` at scalar ``t``."""

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Diffusion *matrix*, shape ``(n_paths, dim, n_factors)``.

        Matrix-valued so a multi-factor model can express correlation between
        its drivers; a single-factor model returns a ``(n_paths, dim, 1)``
        column.
        """

    def asset(self, x: torch.Tensor) -> torch.Tensor:
        """The asset level implied by state ``x``.

        Payoffs are written against this, never against the raw state, so a
        model is free to integrate log-spot or carry auxiliary components
        without any payoff knowing. The default reads the first component.
        """
        return x[..., 0]


class EulerMaruyamaSimulator(Simulator):
    """Euler-Maruyama: ``x + mu h + sigma sqrt(h) z``.

    First order in general, but *exact* for an SDE with state-independent
    coefficients -- which is why
    :class:`~torch_pricer.simulator.gbm.GeometricBrownianMotion` integrates
    log-spot. There, the number of steps changes nothing but the Brownian path,
    so a Monte Carlo price can be compared against a closed form without a
    discretisation bias in the way.
    """

    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, z: torch.Tensor):
        mu = self.sde.drift_coefficient(xt, t)              # (n_paths, dim)
        sigma = self.sde.diffusion_coefficient(xt, t)       # (n_paths, dim, n_factors)
        dw = (z * h.sqrt()).unsqueeze(-1)                   # (n_paths, n_factors, 1)
        return xt + mu * h + torch.bmm(sigma, dw).squeeze(-1)
