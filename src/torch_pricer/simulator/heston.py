"""Heston dynamics.

Stub. Two-dimensional state ``(log S, v)`` driven by two correlated Brownians:

    d log S = (r - q - v/2) dt + sqrt(v) dW1
    dv      = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt

The correlation goes in the ``(batch, 2, 2)`` diffusion matrix as a Cholesky
factor, which is what the matrix-valued diffusion in
:mod:`torch_pricer.simulator.simulator` exists for. Note that Euler is *not*
exact here and can drive ``v`` negative; a full-truncation or QE scheme is the
usual answer.
"""

from __future__ import annotations
