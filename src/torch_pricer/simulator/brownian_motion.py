"""Arithmetic Brownian motion.

Not a pricing process -- spot cannot be normal -- but the one SDE whose
transition law is known exactly in closed form with no transformation, which
makes it the natural fixture for testing the stepping machinery itself.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..tensors import as_tensor
from .simulator import SDE


class BrownianMotion(SDE):
    """``dX = mu dt + sigma dW``, mu and sigma constant."""

    dim = 1
    n_factors = 1

    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.mu = as_tensor(float(mu))
        self.sigma = as_tensor(float(sigma))

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.mu * torch.ones_like(x)

    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        return self.sigma.reshape(1, 1, 1).expand(x.shape[0], self.dim, self.n_factors)

    def risk_parameters(self) -> dict[str, Tensor]:
        return {"sigma": self.sigma}
