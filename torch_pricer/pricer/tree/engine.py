"""Pricing by backward induction on a CRR lattice.

The tree method's engine, peer to :mod:`torch_pricer.pricer.monte_carlo.engine`.
It exists for the thing Monte Carlo cannot do here: **early exercise**. The MC
engine prices a terminal payoff, and ``payoff_for`` refuses a non-European style
outright rather than silently understating it. A lattice carries the whole
continuation value at every node, so the exercise decision is a comparison, not
an estimation problem.

That makes this the reference the Longstaff-Schwartz engine is checked against.
Two properties make it a good one:

* it is deterministic -- no seed, no standard error, no tolerance argument about
  whether a difference is sampling noise;
* it shares no code with the simulation. NumPy, no torch, no ``SDE``.

**Greeks come off the lattice, not from autograd.** The nodes at steps 1 and 2
already bracket the initial spot, so delta and gamma are differences over levels
the tree computed anyway -- no bump, no second pass, and no Dirac problem of the
kind that makes ``gamma_autograd`` zero under Monte Carlo. The cost is that they
are evaluated at ``2 dt`` rather than at ``0``; with the step counts used here
that offset is far below the discretization error itself.

**Convergence.** CRR is first order in ``dt`` and oscillates: whether the strike
falls near a node or between two of them alternates with the parity of the step
count, so consecutive counts straddle the true price rather than approaching it
from one side. Averaging two adjacent step counts cancels most of it, which is
what ``average_adjacent`` does and why it is on by default. It is not a
Richardson extrapolation and does not claim an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Right, Style
from torch_pricer.simulator.tree.crr import Lattice, build


@dataclass(frozen=True)
class LatticeResult:
    """A lattice price and the greeks read off the same tree."""

    price: float
    delta: float | None = None
    gamma: float | None = None
    #: Steps actually used. A tuple when two counts were averaged.
    n_steps: tuple[int, ...] = ()


def _induct(
    lattice: Lattice, strike: float, sign: float, american: bool, want_greeks: bool
):
    """Roll the payoff back to the root. Returns ``(price, delta, gamma)``."""
    values = np.maximum(sign * (lattice.levels(lattice.n_steps) - strike), 0.0)
    disc, p = lattice.discount, lattice.p
    captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for step in range(lattice.n_steps - 1, -1, -1):
        # values[j] is the node with j up moves; its children are j and j+1 one
        # step later, so the continuation value is a two-point stencil.
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            values = np.maximum(values, sign * (lattice.levels(step) - strike))
        if want_greeks and step in (1, 2):
            captured[step] = (lattice.levels(step), values.copy())

    price = float(values[0])
    if not want_greeks or 1 not in captured:
        return price, None, None

    s1, v1 = captured[1]
    delta = float((v1[1] - v1[0]) / (s1[1] - s1[0]))
    if 2 not in captured:
        return price, delta, None

    s2, v2 = captured[2]
    up = (v2[2] - v2[1]) / (s2[2] - s2[1])
    down = (v2[1] - v2[0]) / (s2[1] - s2[0])
    gamma = float((up - down) / (0.5 * (s2[2] - s2[0])))
    return price, delta, gamma


def price_lattice(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    dividend: float = 0.0,
    right: Right | str = Right.CALL,
    style: Style | str = Style.EUROPEAN,
    n_steps: int = 512,
    greeks: bool = False,
    average_adjacent: bool = True,
) -> LatticeResult:
    """Price a vanilla option on a CRR tree.

    Args:
        spot: initial asset level
        strike: option strike
        t: time to expiry in years
        vol: constant volatility
        rate: continuously-compounded funding rate
        dividend: continuous dividend yield
        right: call or put
        style: European or American; Bermudan is not supported here
        n_steps: steps in the tree
        greeks: also return delta and gamma, read off the lattice
        average_adjacent: average the ``n_steps`` and ``n_steps + 1`` trees to
            damp CRR's even-odd oscillation

    Returns:
        A :class:`LatticeResult`.

    Raises:
        ValidationError: for a Bermudan style, or an unusable lattice.
    """
    style = Style(style)
    if style is Style.BERMUDAN:
        raise ValidationError(
            "Bermudan exercise needs the exercise dates aligned to the tree's "
            "time grid; price_lattice takes European or American only"
        )
    if strike <= 0:
        raise ValidationError(f"strike must be positive, got {strike}")

    sign = float(Right(right).sign)
    american = style is Style.AMERICAN
    counts = (n_steps, n_steps + 1) if average_adjacent else (n_steps,)

    out = [
        _induct(build(spot, t, vol, rate, dividend, n), strike, sign, american, greeks)
        for n in counts
    ]
    mean = lambda i: (  # noqa: E731 - three one-line means, not worth a def
        None if out[0][i] is None else sum(o[i] for o in out) / len(out)
    )
    return LatticeResult(
        price=mean(0), delta=mean(1), gamma=mean(2), n_steps=counts
    )
