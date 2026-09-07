"""Crank-Nicolson finite differences, with early exercise by PSOR.

The PDE method's engine. It exists as the *second* independent American
reference: the CRR tree in :mod:`torch_pricer.pricer.tree.engine` converging to
a value only shows that lattice agreeing with itself on finer grids. This solver
shares none of that arithmetic -- a different discretization of a different
formulation of the problem -- so agreement between the two is evidence, and
disagreement localises a bug to one of them.

**Rannacher smoothing.** Crank-Nicolson is second order and unconditionally
stable, but it does not *damp*; it merely fails to amplify. A payoff with a kink
at the strike excites the highest grid frequency, and CN propagates that
oscillation forward essentially undiminished, which shows up as ringing in gamma
near the strike -- the one place gamma matters. Running the first few steps fully
implicit (first order, but strongly damping) kills the high-frequency content
before CN takes over, and the second-order accuracy of the remaining steps
survives. Four implicit steps is the usual prescription and what
``rannacher_steps`` defaults to.

**Where the American-ness lives.** Not in the boundary conditions and not in a
post-hoc clip of a European solution -- both get the free boundary wrong. It is
in the constraint handed to :func:`~torch_pricer.simulator.pde.psor.solve`,
applied inside each Gauss-Seidel sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Right, Style
from torch_pricer.simulator.pde import grid as grid_mod
from torch_pricer.simulator.pde.psor import solve as psor_solve


@dataclass(frozen=True)
class PDEResult:
    """A finite-difference price and the greeks read off the same solution."""

    price: float
    delta: float | None = None
    gamma: float | None = None
    n_space: int = 0
    n_time: int = 0


def _boundaries(
    levels: np.ndarray, strike: float, sign: float, american: bool,
    tau: float, rate: float, dividend: float,
) -> tuple[float, float]:
    """Dirichlet values at the two ends of the mesh, at time-to-expiry ``tau``.

    Deep enough in the money the option is worth its forward intrinsic; deep
    enough out of it, nothing. For an American contract the in-the-money end is
    the *immediate* intrinsic instead, since exercising there is optimal.
    """
    lo, hi = float(levels[0]), float(levels[-1])
    if american:
        deep_itm_low = max(sign * (lo - strike), 0.0)
        deep_itm_high = max(sign * (hi - strike), 0.0)
    else:
        disc_s, disc_k = np.exp(-dividend * tau), np.exp(-rate * tau)
        deep_itm_low = max(sign * (lo * disc_s - strike * disc_k), 0.0)
        deep_itm_high = max(sign * (hi * disc_s - strike * disc_k), 0.0)
    return deep_itm_low, deep_itm_high


def price_pde(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    dividend: float = 0.0,
    right: Right | str = Right.CALL,
    style: Style | str = Style.EUROPEAN,
    n_space: int = 1024,
    n_time: int = 512,
    width: float = 10.0,
    rannacher_steps: int = 4,
    omega: float = 1.5,
    greeks: bool = False,
) -> PDEResult:
    """Price a vanilla option by Crank-Nicolson on a log-spot mesh.

    Args:
        spot: current asset level
        strike: option strike
        t: time to expiry in years
        vol: constant volatility
        rate: continuously-compounded funding rate
        dividend: continuous dividend yield
        right: call or put
        style: European or American; Bermudan is not supported here
        n_space: mesh intervals
        n_time: time steps
        width: mesh half-width in standard deviations
        rannacher_steps: leading fully-implicit steps, to damp the payoff kink
        omega: PSOR over-relaxation; ignored for European
        greeks: also return delta and gamma

    Returns:
        A :class:`PDEResult`.

    Raises:
        ValidationError: for a Bermudan style or an invalid input.
    """
    style = Style(style)
    if style is Style.BERMUDAN:
        raise ValidationError(
            "Bermudan exercise needs its dates aligned to the time mesh; "
            "price_pde takes European or American only"
        )
    if strike <= 0:
        raise ValidationError(f"strike must be positive, got {strike}")
    if n_time < 1:
        raise ValidationError(f"n_time must be at least 1, got {n_time}")

    sign = float(Right(right).sign)
    american = style is Style.AMERICAN
    mesh = grid_mod.build(spot, t, vol, rate, dividend, n_space=n_space, width=width)
    levels, dx = mesh.spot_levels, mesh.dx

    intrinsic = np.maximum(sign * (levels - strike), 0.0)
    values = intrinsic.copy()

    # Constant-coefficient operator in x = log S:
    #   L V = 1/2 sigma^2 V_xx + nu V_x - r V,   nu = r - q - 1/2 sigma^2
    nu = rate - dividend - 0.5 * vol**2
    diffusion, drift = 0.5 * vol**2 / dx**2, nu / (2.0 * dx)
    a, b, c = diffusion - drift, -(vol**2) / dx**2 - rate, diffusion + drift

    dtau = t / n_time
    inner = intrinsic[1:-1]

    for step in range(n_time):
        # Fully implicit while damping the kink, Crank-Nicolson after.
        theta = 1.0 if step < rannacher_steps else 0.5
        tau_next = (step + 1) * dtau

        a_lo, a_di, a_up = -theta * dtau * a, 1.0 - theta * dtau * b, -theta * dtau * c
        explicit = 1.0 - theta
        rhs = values[1:-1] + explicit * dtau * (
            a * values[:-2] + b * values[1:-1] + c * values[2:]
        )

        lo_bc, hi_bc = _boundaries(
            levels, strike, sign, american, tau_next, rate, dividend
        )
        rhs[0] -= a_lo * lo_bc
        rhs[-1] -= a_up * hi_bc

        if american:
            interior = psor_solve(
                a_lo, a_di, a_up, rhs, constraint=inner,
                guess=values[1:-1], omega=omega,
            )
        else:
            banded = np.empty((3, rhs.size))
            banded[0, 1:], banded[1, :], banded[2, :-1] = a_up, a_di, a_lo
            interior = solve_banded((1, 1), banded, rhs)

        values = np.concatenate(([lo_bc], interior, [hi_bc]))

    # Spot is not generally a node -- the mesh is centred on the forward -- so
    # read the answer off a cubic through the solution rather than the nearest
    # node. The same spline gives the greeks, with the log-space chain rule:
    #   dV/dS = V_x / S,  d2V/dS2 = (V_xx - V_x) / S^2
    spline = CubicSpline(mesh.x, values)
    x0 = np.log(spot)
    price = float(spline(x0))
    if not greeks:
        return PDEResult(price=price, n_space=n_space, n_time=n_time)

    vx, vxx = float(spline(x0, 1)), float(spline(x0, 2))
    return PDEResult(
        price=price, delta=vx / spot, gamma=(vxx - vx) / spot**2,
        n_space=n_space, n_time=n_time,
    )
