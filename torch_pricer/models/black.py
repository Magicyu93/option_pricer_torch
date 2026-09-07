from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.errors import ValidationError
from torch_pricer.inputs import CalibrationInputs
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.base import Model
from torch_pricer.simulator.gbm import GeometricBrownianMotion
from torch_pricer.simulator.simulator import SDE
from torch_pricer.tensors import as_tensor


class BlackScholesModel(Model):
    """Constant vol.

    ``vol`` is the model's only parameter; the rates come from the market. It is
    an ``nn.Parameter`` rather than a float so that vega is one entry in the
    same backward pass that produces delta.
    """

    n_factors = 1

    def __init__(self, vol: float = 0.20):
        super().__init__()
        if vol <= 0:
            raise ValidationError(f"volatility must be positive, got {vol}")
        self.vol = nn.Parameter(as_tensor(float(vol)))

    def initial_state(self, market: MarketSnapshot) -> Tensor:
        """``log S_0``, shape ``(1,)`` -- the coordinate the GBM integrates."""
        spot = as_tensor(market.spot, dtype=self.vol.dtype, device=self.vol.device)
        return torch.log(spot).reshape(1)

    def to_sde(self, market: MarketSnapshot) -> SDE:
        return GeometricBrownianMotion(
            mut=lambda t: (
                market.discount.instantaneous_forward(t)
                - market.dividend.instantaneous_forward(t)
            ),
            sigma=self.vol,
        )

    def calibrate(self, inputs: CalibrationInputs, tolerance: float | None = None) -> None:
        raise NotImplementedError("BlackScholesModel.calibrate")
