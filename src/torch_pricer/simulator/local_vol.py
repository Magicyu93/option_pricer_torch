"""Local volatility dynamics.

Stub. ``dS = (r - q) S dt + sigma_loc(S, t) S dW``, integrated in log space like
:mod:`torch_pricer.simulator.gbm`, with ``sigma_loc`` an interpolated surface
from :mod:`torch_pricer.calibration.vol_model.local_vol_cal`.
"""

from __future__ import annotations
