"""Brownian bridge.

A Brownian motion conditioned to arrive at a known level. Pinning the endpoint
turns a driftless process into one with the time-dependent drift

    dX = (a - X_t) / (T - t) dt + sigma dW,   X_T = a,

whose pull towards ``a`` diverges as ``t`` approaches ``T``. That divergence is
the whole content of the process and it has two consequences worth stating.
First, the constructor cannot look like a Brownian motion's: a bridge is
specified by where and when it lands, not by a ``(mu, sigma)`` pair. Second, an
Euler step that lands exactly on ``T`` divides by zero, so the drift's horizon
is floored; the effect is that the simulated bridge arrives at ``a`` with a
residual spread of ``sigma sqrt(h)`` from the final step, rather than exactly.

Two uses, neither of them pricing a bridge for its own sake:

* **Barrier monitoring.** A barrier checked only on the simulation grid misses
  the excursions between grid points and systematically underprices a knock-out.
  Conditional on its endpoints, a path *is* a Brownian bridge, and the
  probability that it crossed the barrier in between has a closed form -- so the
  correction is exact and costs one exponential per step.
* **Quasi-random path construction.** With a low-discrepancy sequence the
  coordinates are not equally valuable: the leading Sobol dimensions are the
  best distributed. Bridge construction fills in the midpoint of the path first
  and refines, so those leading coordinates carry the large-scale shape of the
  path and the variance concentrates where the sequence is strongest.
"""

from __future__ import annotations

from torch import Tensor

from ..errors import ValidationError
from ..tensors import as_tensor
from .simulator import SDE


class BrownianBridge(SDE):
    """``dX = ((a - X) / (T - t)) dt + sigma dW``, pinned to ``a`` at ``T``."""

    dim = 1
    n_factors = 1

    def __init__(self, terminal_level: float = 0.0, terminal_time: float = 1.0, sigma: float = 1.0):
        super().__init__()
        if terminal_time <= 0:
            raise ValidationError(
                f"the bridge's terminal time must be positive, got {terminal_time}"
            )
        if sigma < 0:
            raise ValidationError(f"sigma must be non-negative, got {sigma}")
        self.terminal_level = as_tensor(float(terminal_level))
        self.terminal_time = as_tensor(float(terminal_time))
        self.sigma = as_tensor(float(sigma))

    def horizon(self, t: Tensor) -> Tensor:
        """``T - t``, floored so that the drift stays finite at the pin."""
        return (self.terminal_time - as_tensor(t)).clamp_min(1e-10)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return (self.terminal_level - x) / self.horizon(t)

    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        return self.sigma.reshape(1, 1, 1).expand(x.shape[0], self.dim, self.n_factors)

    def variance(self, t, start_time=0.0) -> Tensor:
        """``sigma^2 (t - t0) (T - t) / (T - t0)``: the bridge's variance at ``t``.

        The exact answer for the continuous process, which is what the simulated
        one is checked against.
        """
        t = as_tensor(t)
        t0 = as_tensor(start_time)
        return self.sigma**2 * (t - t0) * (self.terminal_time - t) / (self.terminal_time - t0)

    def risk_parameters(self) -> dict[str, Tensor]:
        return {"sigma": self.sigma}
