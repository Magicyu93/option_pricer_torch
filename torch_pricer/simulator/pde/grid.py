"""The mesh a finite-difference solver steps on.

State-propagation for the PDE method, the peer of
:mod:`torch_pricer.simulator.monte_carlo.simulator` and
:mod:`torch_pricer.simulator.tree.crr`. Geometry only -- no payoff, no exercise,
no time stepping.

**Log-space, for the same reason the simulator integrates log-spot.** In
``x = log S`` the Black-Scholes operator has constant coefficients:

    dV/dtau = 1/2 sigma^2 d2V/dx2 + (r - q - 1/2 sigma^2) dV/dx - r V

so a uniform mesh gives one tridiagonal matrix that is reused at every step
rather than rebuilt. In spot space the coefficients carry ``S`` and ``S^2`` and
the matrix changes with the node.

**Width.** The domain is truncated at ``+/- width`` standard deviations of
``log S_T`` around the forward. Dirichlet conditions there are only asymptotically
right, so the truncation error decays like the tail it cuts off -- exponentially
in ``width``. Ten standard deviations puts it far below the discretization error
and costs only a wider array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torch_pricer.errors import ValidationError


@dataclass(frozen=True)
class LogGrid:
    """A uniform mesh in ``log S``.

    Attributes:
        x: log-spot nodes, ascending, shape ``(n_space + 1,)``
        spot_levels: ``exp(x)``, cached because every payoff needs it
        dx: node spacing
        spot: the spot the grid is centred on
    """

    x: np.ndarray
    spot_levels: np.ndarray
    dx: float
    spot: float

    @property
    def n_space(self) -> int:
        """Number of intervals; there are ``n_space + 1`` nodes."""
        return self.x.size - 1

    @property
    def interior(self) -> slice:
        """The nodes actually solved for; the two ends are Dirichlet."""
        return slice(1, -1)


def build(
    spot: float,
    t: float,
    vol: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    n_space: int = 512,
    width: float = 10.0,
) -> LogGrid:
    """A log-spot mesh centred on the forward, ``width`` standard deviations wide.

    Centring on the forward rather than on spot keeps the domain symmetric about
    where the distribution actually is, which matters at long maturities or under
    a large carry.

    Args:
        spot: current asset level
        t: horizon in years
        vol: constant volatility
        rate: continuously-compounded funding rate
        dividend: continuous dividend yield
        n_space: number of intervals
        width: half-width in standard deviations of ``log S_T``

    Raises:
        ValidationError: on a non-positive input or too coarse a mesh.
    """
    if spot <= 0:
        raise ValidationError(f"spot must be positive, got {spot}")
    if t <= 0:
        raise ValidationError(f"t must be positive, got {t}")
    if vol <= 0:
        raise ValidationError(f"vol must be positive, got {vol}")
    if n_space < 4:
        raise ValidationError(f"n_space must be at least 4, got {n_space}")

    centre = np.log(spot) + (rate - dividend - 0.5 * vol**2) * t
    half = width * vol * np.sqrt(t)
    x = np.linspace(centre - half, centre + half, n_space + 1)
    return LogGrid(x=x, spot_levels=np.exp(x), dx=float(x[1] - x[0]), spot=float(spot))
