"""The vol model interface: what gets calibrated, and what it hands the simulator.

A :class:`VolModel` owns the parameters that a calibration moves. It is an
``nn.Module`` for four concrete reasons, all of which the alternatives (a plain
dataclass, or a dict of floats) give up:

* parameters are ``nn.Parameter``, so they are leaves of the autograd graph and
  vega is the same backward pass that produces delta;
* ``.parameters()`` hands the whole set to any torch optimiser, so calibration
  is Adam or LBFGS rather than a bespoke solver;
* ``.to(device)`` moves a model to the GPU without the caller enumerating fields;
* ``state_dict()`` serialises a calibrated model, which is what a pricing
  service needs to reload this morning's fit.

Constrained parameters are stored *unconstrained* and mapped through a
monotone function on read -- ``exp`` for anything positive, ``tanh`` for a
correlation. The optimiser then works in an unbounded space and the model can
never be evaluated outside its feasible set, with no penalty terms and no
projection step.

The seam to the simulator is :meth:`VolModel.to_sde`. Note what it takes: the
risk-neutral drift on the asset is ``(r(t) - q(t)) S``, which comes from the
curves, while the diffusion comes from the model. An SDE is a composition of the
two, which is exactly why a model is not itself an SDE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.nn as nn

from ..curve_model.curves import RateCurve
from ..surface import VolSurface

if TYPE_CHECKING:  # pragma: no cover
    from ...market.snapshot import MarketSnapshot
    from ...simulator.simulator import SDE


class VolModel(nn.Module, ABC):
    """A model of the underlying's dynamics, with calibratable parameters."""

    @abstractmethod
    def to_sde(self, market: MarketSnapshot) -> SDE:
        """Build the SDE this model implies under ``market``.

        The returned SDE must hold the very tensors the model exposes, not
        copies of their values: the engine differentiates the price with respect
        to them, and a ``float()`` anywhere in this method silently detaches
        vega.
        """

    @abstractmethod
    def calibrate(self, quotes, rate_curves: RateCurve, vol_surface: VolSurface) -> None:
        """Fit the model's parameters. Mutates in place."""
