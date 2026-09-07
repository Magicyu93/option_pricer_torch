"""Projected SOR: the tridiagonal solve that knows about early exercise.

An American option is not a boundary-value problem with a known boundary. It is
a *linear complementarity problem*: at every node either the PDE holds and the
option is worth more than intrinsic, or the option is at intrinsic and the PDE
holds as an inequality. Which of the two is not known in advance -- the free
boundary is part of the answer -- so the system cannot simply be inverted.

    A v >= b,   v >= g,   (A v - b) . (v - g) = 0

PSOR solves it by iterating a relaxed Gauss-Seidel sweep and projecting onto
``v >= g`` after every node update. The projection inside the sweep is the whole
point: applying it once at the end would just clip a European solution and get
the free boundary wrong.

**Red-black ordering.** A textbook sweep updates nodes left to right, so node
``i`` sees the already-updated ``i - 1`` -- sequential, and in Python that is a
per-node interpreter round trip: a 500 x 1000 grid needs millions of them. On a
tridiagonal system the odd nodes depend only on even neighbours and vice versa,
so the sweep splits into two half-sweeps that are each fully vectorised, with the
same Gauss-Seidel property. Same algorithm, two NumPy operations per iteration
instead of a loop.
"""

from __future__ import annotations

import numpy as np

from torch_pricer.errors import PricingError, ValidationError


def solve(
    lower: float,
    diag: float,
    upper: float,
    rhs: np.ndarray,
    constraint: np.ndarray | None = None,
    guess: np.ndarray | None = None,
    omega: float = 1.5,
    tol: float = 1e-10,
    max_sweeps: int = 10_000,
) -> np.ndarray:
    """Solve a constant-coefficient tridiagonal system, optionally constrained.

    Args:
        lower: sub-diagonal coefficient (same at every row)
        diag: diagonal coefficient
        upper: super-diagonal coefficient
        rhs: right-hand side, shape ``(n,)``, with any Dirichlet contribution
            already folded in
        constraint: elementwise lower bound ``g``; ``None`` solves the plain
            linear system (the European case)
        guess: starting iterate; the previous time step is a good one
        omega: over-relaxation factor in ``(0, 2)``
        tol: stop when the largest node update falls below this
        max_sweeps: give up after this many

    Returns:
        The solution, shape ``(n,)``.

    Raises:
        ValidationError: if ``omega`` is outside ``(0, 2)``, where SOR diverges.
        PricingError: if the sweeps do not converge.
    """
    if not 0.0 < omega < 2.0:
        raise ValidationError(f"omega must lie in (0, 2), got {omega}")

    n = rhs.size
    v = (rhs / diag if guess is None else guess).astype(float, copy=True)
    if constraint is not None:
        np.maximum(v, constraint, out=v)

    # Red-black split: with a tridiagonal operator each colour's update depends
    # only on the other colour, so a half-sweep is one vectorised expression.
    red, black = np.arange(0, n, 2), np.arange(1, n, 2)

    for _ in range(max_sweeps):
        delta = 0.0
        for idx in (red, black):
            left = np.where(idx > 0, v[np.maximum(idx - 1, 0)], 0.0)
            right = np.where(idx < n - 1, v[np.minimum(idx + 1, n - 1)], 0.0)
            target = (rhs[idx] - lower * left - upper * right) / diag
            updated = v[idx] + omega * (target - v[idx])
            if constraint is not None:
                updated = np.maximum(updated, constraint[idx])
            delta = max(delta, float(np.max(np.abs(updated - v[idx]))) if idx.size else 0.0)
            v[idx] = updated
        if delta < tol:
            return v

    raise PricingError(
        f"PSOR did not converge in {max_sweeps} sweeps (last update {delta:.3e}); "
        "the time step may be too large for this mesh"
    )
