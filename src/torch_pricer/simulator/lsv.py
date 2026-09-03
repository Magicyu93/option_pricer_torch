"""Local-stochastic volatility dynamics.

Stub. Heston's variance scaled by a leverage function ``L(S, t)`` chosen so the
model reprices the vanilla surface exactly:

    d log S = (r - q - L^2 v / 2) dt + L(S, t) sqrt(v) dW1

Pairs with :mod:`torch_pricer.calibration.vol_model.lsv_cal`. Calibrating ``L``
needs the conditional expectation of ``v`` given ``S``, which is a particle or
PDE step rather than a plain optimisation.
"""

from __future__ import annotations
