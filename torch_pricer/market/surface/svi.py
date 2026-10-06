"""Gatheral raw SVI, one slice per expiry. Not implemented yet."""

from __future__ import annotations

import datetime as dt

from torch import Tensor

from torch_pricer.market.surface.base import VolSurface


class SVISurface(VolSurface):
    """Raw SVI per expiry: ``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))``.

    When it lands, the slice parameters ``(a, b, rho, m, sigma)`` must be
    ``nn.Parameter`` s, constrained by their parametrisation (``b >= 0``,
    ``|rho| < 1``, ``sigma > 0``), and slices joined so that ``w`` does not
    decrease in ``T`` -- then :func:`~torch_pricer.market.surface.calibration.fit_surface`
    fits it unchanged.
    """

    def __init__(self, as_of: dt.date):
        super().__init__()
        self._ref = as_of

    def total_variance(self, k, expiry) -> Tensor:
        raise NotImplementedError("SVISurface.total_variance")

    @property
    def reference_date(self) -> dt.date:
        return self._ref
