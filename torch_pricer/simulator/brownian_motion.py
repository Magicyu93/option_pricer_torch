"""Arithmetic Brownian motion with constant coefficients."""

import torch
from torch import Tensor

from torch_pricer.simulator.simulator import SDE


class BrownianMotion(SDE):
    """``dX = mu dt + sigma dW``, ``mu`` and ``sigma`` constant."""

    n_factors = 1

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def drift_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        """Drift, shape ``(n_paths, dim)``."""
        return self.mu * torch.ones_like(xt)

    def diffusion_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        """Diffusion column, shape ``(n_paths, dim, 1)``."""
        return (self.sigma * torch.ones_like(xt)).unsqueeze(-1)
