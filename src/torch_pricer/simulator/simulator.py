"""Path generation for

    dX = mu(X, t) dt + sigma(X, t) dW

with X in R^d driven by m independent Brownian motions.

Two shape decisions here are worth spelling out.

**The diffusion is a matrix**, ``(batch, d, m)``, not a vector. An elementwise
``(batch, d)`` diffusion cannot express a correlation between the components of
X, which rules out every stochastic-vol model on the roadmap. Writing it as a
matrix costs one ``bmm`` at ``d = m = 1`` and lets Heston put its spot/variance
correlation in a Cholesky factor without any change to this file.

**Gradients flow through simulation.** There is no ``torch.no_grad`` here by
default. Differentiating the payoff with respect to spot straight through the
paths is the pathwise derivative estimator: unbiased for Lipschitz payoffs, and
far lower variance than bumping and re-simulating. It is the reason the library
is written in torch at all. Pass ``no_grad=True`` when pricing without risk to
get the memory back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


class SDE(nn.Module, ABC):
    """An Ito process, in whatever coordinate the model finds convenient."""

    #: Dimension of the state vector X.
    dim: int = 1
    #: Number of independent Brownian motions driving it.
    n_factors: int = 1

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """``mu(x, t)``, shape ``(batch, dim)``.

        Args:
            x: state, ``(batch, dim)``
            t: time in years, scalar
        """

    @abstractmethod
    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        """``sigma(x, t)``, shape ``(batch, dim, n_factors)``."""

    def initial_state(self, spot: Tensor) -> Tensor:
        """The state at ``t = 0`` given the asset's spot, shape ``(1, dim)``."""
        return spot.reshape(1, 1).expand(1, self.dim)

    def asset(self, x: Tensor) -> Tensor:
        """The asset level implied by state ``x``.

        Payoffs are written against this, never against the raw state, so a
        model is free to integrate log-spot or carry auxiliary components
        without any payoff knowing. The default reads the first component.
        """
        return x[..., 0]

    def risk_parameters(self) -> dict[str, Tensor]:
        """Named tensors this SDE's price is differentiable against.

        These are the *natural-scale* quantities a desk quotes risk in (a
        volatility, not its logarithm), and they must be the same tensor objects
        used inside :meth:`drift` and :meth:`diffusion` -- the engine calls
        ``autograd.grad`` against them.
        """
        return {}


class Simulator(ABC):
    """Integrates an SDE over a time grid using pre-drawn normals."""

    @abstractmethod
    def step(self, x: Tensor, t: Tensor, h: Tensor, z: Tensor) -> Tensor:
        """One step from ``t`` to ``t + h``.

        Args:
            x: state, ``(batch, dim)``
            t: current time, scalar
            h: step length, scalar
            z: standard normals for this step, ``(batch, n_factors)``

        Returns:
            state at ``t + h``, ``(batch, dim)``
        """

    def _walk(self, grid: Tensor, draws: Tensor, progress: bool) -> Iterable:
        n_steps = grid.numel() - 1
        if draws.shape[1] < n_steps:
            raise ValueError(
                f"grid needs {n_steps} steps but only {draws.shape[1]} draws were supplied"
            )
        steps = range(n_steps)
        if progress:
            from tqdm import tqdm  # imported lazily: a progress bar is not a dependency of pricing

            steps = tqdm(steps, desc="paths")
        return steps

    def simulate(
        self,
        x: Tensor,
        grid: Tensor,
        draws: Tensor,
        no_grad: bool = False,
        progress: bool = False,
    ) -> Tensor:
        """Integrate to ``grid[-1]``, keeping only the final state.

        Args:
            x: initial state, ``(batch, dim)``
            grid: times in years, ``(n_steps + 1,)``, increasing
            draws: standard normals, ``(batch, n_steps, n_factors)``

        Returns:
            terminal state, ``(batch, dim)``
        """
        with torch.no_grad() if no_grad else nullcontext():
            for i in self._walk(grid, draws, progress):
                x = self.step(x, grid[i], grid[i + 1] - grid[i], draws[:, i])
            return x

    def simulate_with_trajectory(
        self,
        x: Tensor,
        grid: Tensor,
        draws: Tensor,
        no_grad: bool = False,
        progress: bool = False,
    ) -> Tensor:
        """Integrate over the whole grid, keeping every state.

        Returns:
            trajectory, ``(batch, n_steps + 1, dim)``
        """
        with torch.no_grad() if no_grad else nullcontext():
            xs = [x]
            for i in self._walk(grid, draws, progress):
                x = self.step(x, grid[i], grid[i + 1] - grid[i], draws[:, i])
                xs.append(x)
            return torch.stack(xs, dim=1)


class EulerMaruyamaSimulator(Simulator):
    """Euler-Maruyama: ``x + mu h + sigma sqrt(h) z``.

    First order in general, but *exact* for an SDE with state-independent
    coefficients -- which is why :class:`~torch_pricer.simulator.gbm.GeometricBrownianMotion`
    integrates log-spot. There, the number of steps changes nothing but the
    Brownian path, so a Monte Carlo price can be compared against a closed form
    without a discretisation bias in the way.
    """

    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, x: Tensor, t: Tensor, h: Tensor, z: Tensor) -> Tensor:
        mu = self.sde.drift(x, t)
        sigma = self.sde.diffusion(x, t)
        dw = (z * h.sqrt()).unsqueeze(-1)          # (batch, n_factors, 1)
        return x + mu * h + torch.bmm(sigma, dw).squeeze(-1)
