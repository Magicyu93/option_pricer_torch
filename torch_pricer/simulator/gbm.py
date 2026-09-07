"""Geometric Brownian motion, integrated in log-space."""

from typing import Callable

import torch
from torch import Tensor

from torch_pricer.simulator.simulator import SDE


class GeometricBrownianMotion(SDE):
    """
    ``dS = S mu(t) dt + S sigma dW``, with ``sigma`` constant, integrated as

        ``d log S = (mu(t) - 0.5 sigma^2) dt + sigma dW``

    so that Euler-Maruyama is exact and the step count changes nothing but the
    Brownian path.

    ``sigma`` is held as whatever tensor the model handed over -- an
    ``nn.Parameter``, in practice -- never coerced with ``float()``, because
    vega is taken by differentiating the price with respect to it.
    """

    n_factors = 1

    def __init__(self, mut: Callable[[Tensor], Tensor], sigma: Tensor):
        self.mut = mut
        self.sigma = sigma

    def drift_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        """Drift of log-spot, shape ``(n_paths, dim)``."""
        return (self.mut(t) - 0.5 * self.sigma**2) * torch.ones_like(xt)

    def diffusion_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        """Diffusion column, shape ``(n_paths, dim, 1)``."""
        return (self.sigma * torch.ones_like(xt)).unsqueeze(-1)

    def asset(self, x: Tensor) -> Tensor:
        return torch.exp(x[..., 0])
