"""The Cox-Ross-Rubinstein lattice: the recombining tree GBM lives on.

This is the state-propagation scheme for the tree method, the peer of
:mod:`torch_pricer.simulator.monte_carlo.simulator` for Monte Carlo. It builds
the geometry -- where the asset can be at each step, and with what probability
it moves -- and knows nothing about payoffs or exercise. The backward induction
that turns geometry into a price lives in
:mod:`torch_pricer.pricer.tree.engine`.

NumPy rather than torch, deliberately. The tree's first job is to be the
reference that the Monte Carlo American engine is checked against, and a
reference sharing an implementation with the thing it validates is not one --
the same argument :mod:`torch_pricer.pricer.analytic.heston` makes for itself.
Greeks come off the lattice directly rather than by autograd; see
:func:`~torch_pricer.pricer.tree.engine.price_lattice`.

The CRR choice of ``u = exp(sigma sqrt(dt))``, ``d = 1/u`` makes the tree
recombine and centres it on the initial spot, so an ``n``-step tree holds
``n + 1`` terminal nodes rather than ``2^n`` paths. It converges to Black-Scholes
at first order in ``dt``, with a well-known even-odd oscillation: consecutive
step counts straddle the true price rather than approaching it from one side,
which is why :func:`~torch_pricer.pricer.tree.engine.price_lattice` averages two
adjacent step counts by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torch_pricer.errors import ValidationError


@dataclass(frozen=True)
class Lattice:
    """A recombining CRR tree.

    Attributes:
        spot: initial asset level
        u: up factor per step; ``d`` is its reciprocal
        p: risk-neutral probability of an up move
        dt: step size in years
        n_steps: number of steps
        discount: per-step discount factor ``exp(-r dt)``
    """

    spot: float
    u: float
    p: float
    dt: float
    n_steps: int
    discount: float

    @property
    def d(self) -> float:
        return 1.0 / self.u

    def levels(self, step: int) -> np.ndarray:
        """Asset levels at ``step``, ascending, shape ``(step + 1,)``.

        Node ``j`` has had ``j`` up moves and ``step - j`` down moves, so its
        level is ``S u^(2j - step)``. Written that way rather than as
        ``u**j * d**(step - j)`` to keep it one power of a number near 1.
        """
        if not 0 <= step <= self.n_steps:
            raise ValidationError(f"step {step} outside [0, {self.n_steps}]")
        return self.spot * self.u ** (2 * np.arange(step + 1) - step)


def build(
    spot: float,
    t: float,
    vol: float,
    rate: float,
    dividend: float = 0.0,
    n_steps: int = 512,
) -> Lattice:
    """A CRR lattice for GBM over ``[0, t]``.

    Args:
        spot: initial asset level
        t: horizon in years
        vol: constant volatility
        rate: continuously-compounded funding rate
        dividend: continuous dividend yield
        n_steps: steps in the tree

    Raises:
        ValidationError: on a non-positive input, or if the step is so coarse
            that the risk-neutral probability leaves ``[0, 1]``. That happens
            when ``|(r - q)| sqrt(dt) > sigma`` -- the drift outruns what a
            one-standard-deviation move can represent -- and it produces
            negative probabilities rather than a merely inaccurate price, so it
            is rejected instead of clamped.
    """
    if spot <= 0:
        raise ValidationError(f"spot must be positive, got {spot}")
    if t <= 0:
        raise ValidationError(f"t must be positive, got {t}")
    if vol <= 0:
        raise ValidationError(f"vol must be positive, got {vol}")
    if n_steps < 1:
        raise ValidationError(f"n_steps must be at least 1, got {n_steps}")

    dt = t / n_steps
    u = float(np.exp(vol * np.sqrt(dt)))
    growth = float(np.exp((rate - dividend) * dt))
    p = (growth - 1.0 / u) / (u - 1.0 / u)
    if not 0.0 <= p <= 1.0:
        raise ValidationError(
            f"CRR probability {p:.4f} outside [0, 1] at n_steps={n_steps}: the "
            f"drift |r - q| = {abs(rate - dividend):g} outruns a one-sigma move "
            f"over dt={dt:g}. Use more steps, or a lattice that is not CRR."
        )
    return Lattice(
        spot=float(spot), u=u, p=float(p), dt=dt, n_steps=int(n_steps),
        discount=float(np.exp(-rate * dt)),
    )
