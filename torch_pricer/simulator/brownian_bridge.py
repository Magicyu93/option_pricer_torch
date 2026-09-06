"""Brownian bridge. Not implemented yet."""

from torch import Tensor

from torch_pricer.simulator.simulator import SDE


class BrownianBridge(SDE):
    """A Brownian motion pinned at both ends."""

    n_factors = 1

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def drift_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError("BrownianBridge.drift_coefficient")

    def diffusion_coefficient(self, xt: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError("BrownianBridge.diffusion_coefficient")
