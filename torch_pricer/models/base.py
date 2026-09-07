from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from torch_pricer.inputs import CalibrationInputs
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.simulator.simulator import SDE


class Model(nn.Module, ABC):
    """A model of the underlying's dynamics, with calibratable parameters."""

    n_factors: int

    @abstractmethod
    def initial_state(self, market: MarketSnapshot) -> Tensor:
        """The SDE's state at ``t = 0``, shape ``(n_factors,)``.

        In whatever coordinate the SDE integrates -- log-spot for
        :class:`~torch_pricer.models.black.BlackScholesModel`, not spot -- and
        built from ``market.spot`` without detaching it, since delta is taken by
        differentiating the price with respect to that tensor.
        """

    @abstractmethod
    def to_sde(self, market: MarketSnapshot) -> SDE:
        """Build the SDE this model implies under ``market``.

        The returned SDE must hold the very tensors the model exposes, not
        copies of their values: the engine differentiates the price with respect
        to them, and a ``float()`` anywhere in this method silently detaches
        vega.
        """

    @abstractmethod
    def calibrate(self, inputs: CalibrationInputs) -> None:
        """Fit the model's parameters to ``inputs``. Mutates in place.

        One argument rather than a list of them because the models differ in
        what they consume -- premiums, an implied surface, or both plus a
        simulation -- while agreeing on the market state they need; see
        :class:`~torch_pricer.inputs.CalibrationInputs`.

        Note for the models still to come: for Black-Scholes the parameter *is*
        the quoted vol, so ``dV/d(parameter)`` is vega. For Heston or local vol
        it is not, and market vega needs the chain rule through this fit --
        which means retaining the Jacobian and Hessian at the optimum rather
        than mutating and discarding them. That interface change belongs here,
        before a second calibrator exists.
        """
