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
from torch.utils.checkpoint import checkpoint
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

    def _run(self, x, ts, draws, lo: int, hi: int):
        """Steps ``[lo, hi)`` with no bookkeeping. The unit of checkpointing."""
        for t_idx in range(lo, hi):
            x = self.step(x, ts[t_idx], ts[t_idx + 1] - ts[t_idx], draws[:, t_idx])
        return x

    @staticmethod
    def _segments(n_steps: int, segments: int):
        """Split ``n_steps`` into roughly equal contiguous blocks."""
        segments = max(1, min(int(segments), n_steps))
        edges = [round(i * n_steps / segments) for i in range(segments + 1)]
        return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]

    def simulate(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        draws: torch.Tensor,
        no_grad: bool = False,
        progress: bool = False,
        checkpoint_segments: int = 0,
    ) -> torch.Tensor:
        """Integrate to ``ts[-1]``, keeping only the final state.

        Args:
            x: initial state at ``ts[0]``, shape ``(n_paths, dim)``
            ts: time grid, shape ``(n_steps + 1,)``
            draws: standard normals, shape ``(n_paths, n_steps, n_factors)``
            checkpoint_segments: if positive, retain only this many intermediate
                states and recompute the rest during the backward pass. Trades
                one extra forward pass for a large drop in peak memory; see
                :class:`~torch_pricer.pricer.monte_carlo.engine.MCConfig`.

        Returns:
            final state, shape ``(n_paths, dim)``
        """
        n_steps = ts.numel() - 1
        with torch.no_grad() if no_grad else nullcontext():
            if checkpoint_segments and torch.is_grad_enabled() and not no_grad:
                blocks = self._segments(n_steps, checkpoint_segments)
                for lo, hi in tqdm(blocks) if progress else blocks:
                    x = checkpoint(
                        self._run, x, ts, draws, lo, hi,
                        use_reentrant=False, preserve_rng_state=False,
                    )
                return x
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
        checkpoint_segments: int = 0,
    ) -> torch.Tensor:
        """Integrate to ``ts[-1]``, retaining every state.

        Checkpointing helps less here than in :meth:`simulate`: every state is an
        *output*, so only the per-step internals can be dropped, not the states
        themselves. For a model whose step is expensive -- a Dupire surface
        lookup on every path -- those internals are still most of the memory.

        Returns:
            trajectory, shape ``(n_paths, n_steps + 1, dim)``
        """
        n_steps = ts.numel() - 1
        with torch.no_grad() if no_grad else nullcontext():
            if checkpoint_segments and torch.is_grad_enabled() and not no_grad:
                blocks = self._segments(n_steps, checkpoint_segments)
                xs = [x]
                for lo, hi in tqdm(blocks) if progress else blocks:
                    seg = checkpoint(
                        self._run_trajectory, x, ts, draws, lo, hi,
                        use_reentrant=False, preserve_rng_state=False,
                    )
                    xs.extend(seg.unbind(dim=1))
                    x = xs[-1]
                return torch.stack(xs, dim=1)
            xs = [x]
            for t_idx in self._steps(ts, progress):
                h = ts[t_idx + 1] - ts[t_idx]
                x = self.step(x, ts[t_idx], h, draws[:, t_idx])
                xs.append(x)
            return torch.stack(xs, dim=1)

    def _run_trajectory(self, x, ts, draws, lo: int, hi: int):
        """Steps ``[lo, hi)``, returning the states *after* each one."""
        out = []
        for t_idx in range(lo, hi):
            x = self.step(x, ts[t_idx], ts[t_idx + 1] - ts[t_idx], draws[:, t_idx])
            out.append(x)
        return torch.stack(out, dim=1)


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

    def coefficients(self, xt: torch.Tensor, t: torch.Tensor):
        """Both coefficients at once, as ``(drift, diffusion)``.

        The default just calls the two methods. Override it when they share
        work: a Dupire local vol evaluates the same surface expression for both,
        and computing it twice per step doubles the cost of the simulation and
        the size of the graph the backward pass has to walk.
        """
        return self.drift_coefficient(xt, t), self.diffusion_coefficient(xt, t)

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
    :class:`~torch_pricer.simulator.monte_carlo.gbm.GeometricBrownianMotion` integrates
    log-spot. There, the number of steps changes nothing but the Brownian path,
    so a Monte Carlo price can be compared against a closed form without a
    discretisation bias in the way.
    """

    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, z: torch.Tensor):
        mu, sigma = self.sde.coefficients(xt, t)  # (n_paths, dim), (n_paths, dim, n_factors)
        dw = (z * h.sqrt()).unsqueeze(-1)                   # (n_paths, n_factors, 1)
        return xt + mu * h + torch.bmm(sigma, dw).squeeze(-1)
