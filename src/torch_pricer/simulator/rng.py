"""Normal draws for the engine, seeded and reproducible.

The randomness is generated up front and handed to the simulator rather than
drawn inside the stepping loop. That is what makes a price reproducible from a
seed, and it is what lets antithetic pairs stay paired across every step.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..errors import ValidationError


class NormalDraws:
    """A seeded source of standard normals shaped for :class:`Simulator`.

    Args:
        n_paths: number of paths; must be even when ``antithetic``
        n_factors: independent Brownian motions to drive
        seed: generator seed
        antithetic: pair each path with its mirror image, ``-z``
        device: device the draws land on
        dtype: dtype of the draws

    Antithetic sampling costs nothing and removes the odd moments of the
    sampling error exactly. For a payoff that is monotone in the driver -- which
    a vanilla call is -- it is a strict variance reduction. It also makes the
    Monte Carlo forward unbiased to machine precision, which is what keeps a
    put-call parity check meaningful at the sample sizes used in tests.
    """

    def __init__(
        self,
        n_paths: int,
        n_factors: int = 1,
        seed: int = 0,
        antithetic: bool = True,
        device=None,
        dtype: torch.dtype | None = None,
    ):
        if n_paths <= 0:
            raise ValidationError(f"n_paths must be positive, got {n_paths}")
        if antithetic and n_paths % 2:
            raise ValidationError(f"antithetic sampling needs an even n_paths, got {n_paths}")
        self.n_paths = int(n_paths)
        self.n_factors = int(n_factors)
        self.antithetic = bool(antithetic)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype or torch.get_default_dtype()
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    def draw(self, n_steps: int) -> Tensor:
        """Standard normals of shape ``(n_paths, n_steps, n_factors)``."""
        if n_steps <= 0:
            raise ValidationError(f"n_steps must be positive, got {n_steps}")
        rows = self.n_paths // 2 if self.antithetic else self.n_paths
        z = torch.randn(
            rows, n_steps, self.n_factors,
            generator=self.generator, device=self.device, dtype=self.dtype,
        )
        return torch.cat([z, -z], dim=0) if self.antithetic else z
