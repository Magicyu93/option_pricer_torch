"""Black-Scholes: one constant volatility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ...errors import ValidationError
from ...simulator.gbm import GeometricBrownianMotion
from ...tensors import as_tensor
from ..curve_model.curves import RateCurve
from ..surface import VolSurface
from .base import VolModel

if TYPE_CHECKING:  # pragma: no cover
    from ...market.snapshot import MarketSnapshot
    from ...simulator.simulator import SDE


class BlackScholesModel(VolModel):
    """Constant vol, lognormal spot.

    The volatility is stored as its logarithm so that an unconstrained optimiser
    cannot walk it negative. Read it through :attr:`vol`.
    """

    def __init__(self, vol: float = 0.20):
        super().__init__()
        if vol <= 0:
            raise ValidationError(f"volatility must be positive, got {vol}")
        self._log_vol = nn.Parameter(as_tensor(float(vol)).log())

    @property
    def vol(self) -> torch.Tensor:
        """Volatility, positive by construction."""
        return self._log_vol.exp()

    def to_sde(self, market: MarketSnapshot) -> SDE:
        return GeometricBrownianMotion(
            vol=self.vol, discount=market.discount, dividend=market.dividend
        )

    def calibrate(self, quotes, rate_curves: RateCurve, vol_surface: VolSurface) -> None:
        raise NotImplementedError(
            "fitting a single vol to a quote set is not implemented yet; "
            "construct BlackScholesModel(vol) directly"
        )

    def extra_repr(self) -> str:  # pragma: no cover
        return f"vol={float(self.vol.detach()):.4f}"
