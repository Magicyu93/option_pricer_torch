"""Brownian bridge.

Stub. Note that a bridge is not a constant-coefficient process: pinning the
endpoint gives it the time-dependent drift ``(a - X_t) / (T - t)``, which blows
up as ``t`` approaches ``T``. Whatever constructor this ends up with will need
the terminal level and time, not the ``(mu, sigma)`` pair a plain Brownian
motion takes.

Intended uses are barrier monitoring -- correcting a discretely-monitored
barrier for the probability the path crossed between grid points -- and
quasi-random path construction, where the bridge orders the dimensions so the
leading Sobol coordinates carry most of the variance.
"""

from __future__ import annotations

from torch import Tensor

from .simulator import SDE


class BrownianBridge(SDE):
    """``dX = ((a - X) / (T - t)) dt + sigma dW``, pinned to ``a`` at ``T``."""

    dim = 1
    n_factors = 1

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError("BrownianBridge is not implemented yet")

    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError("BrownianBridge is not implemented yet")
