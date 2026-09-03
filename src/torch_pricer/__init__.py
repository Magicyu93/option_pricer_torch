"""Torch-native Monte Carlo option pricing with autograd greeks.

The pipeline is one direction only::

    market_data  ->  calibration  ->  market  ->  pricer
    (raw quotes)     (curves,         (snapshot  (MC engine,
                      surfaces,        of the     autograd
                      vol models)      world)     greeks)

QuantLib is used for calendars and day counts and nothing else: it cannot carry
autograd gradients, so it never appears between the spot and the price.
"""

from __future__ import annotations

import torch

__version__ = "0.1.0"

#: Pricing runs in double precision. Second-order greeks (gamma especially) come
#: out of a difference of differences; in float32 the rounding noise is the same
#: order as the answer. Importing this package therefore sets torch's global
#: default dtype. Every internal tensor is also constructed with an explicit
#: dtype, so a caller who resets the global default still gets correct numbers.
DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

from .errors import (  # noqa: E402
    CalibrationError,
    EngineNotAvailable,
    MarketDataError,
    PricerError,
    PricingError,
    ValidationError,
)
from .calibration.curve_model.curves import RateCurve  # noqa: E402
from .calibration.inputs import CalibrationInputs  # noqa: E402
from .calibration.surface import (  # noqa: E402
    FlatVolSurface,
    SVISlice,
    SVISurface,
    VolSurface,
)
from .calibration.vol_model.base import VolModel  # noqa: E402
from .calibration.vol_model.black import BlackScholesModel  # noqa: E402
from .calibration.vol_model.heston_cal import HestonModel  # noqa: E402
from .calibration.vol_model.local_vol_cal import LocalVolModel  # noqa: E402
from .calibration.vol_model.lsv_cal import LSVModel  # noqa: E402
from .instruments.spec import (  # noqa: E402
    AsianOption,
    AverageKind,
    Right,
    Stock,
    Style,
    VanillaOption,
)
from .market.snapshot import MarketSnapshot  # noqa: E402
from .market_data.quotes import (  # noqa: E402
    OptionQuote,
    QuoteSet,
    RateQuote,
    SpotQuote,
)
from .pricer.engine import MCConfig, PricingResult, price  # noqa: E402

__all__ = [
    "DEFAULT_DTYPE",
    "__version__",
    "PricerError",
    "ValidationError",
    "MarketDataError",
    "CalibrationError",
    "PricingError",
    "EngineNotAvailable",
    "Right",
    "Style",
    "AverageKind",
    "Stock",
    "VanillaOption",
    "AsianOption",
    "MarketSnapshot",
    "OptionQuote",
    "RateQuote",
    "SpotQuote",
    "QuoteSet",
    "RateCurve",
    "CalibrationInputs",
    "VolSurface",
    "FlatVolSurface",
    "SVISlice",
    "SVISurface",
    "VolModel",
    "BlackScholesModel",
    "HestonModel",
    "LocalVolModel",
    "LSVModel",
    "MCConfig",
    "PricingResult",
    "price",
]
