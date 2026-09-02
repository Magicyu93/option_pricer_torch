import torch
from ..market.snapshot import MarketSnapshot
from ..instruments.spec import Instrument

####

# first only implement mc approach for simple vanilla european option, and benchmarking with the analytical results

class mc_config:
    n_paths : int = 100_000
    n_steps : int = 100
    T: float = 1.0
    seed: int = 0
    usd_gpu: bool = True
    antithetic: bool = True


def price(
        spec: Instrument,
        market: MarketSnapshot,
        engine_config: str,
):
    """torch simulation for the option price"""

    # from spec: time interval, simulation time grid

    # from market: r(t), sigma(S, t)

    # MC simulation, compute pay off

    # applying discount and compute price, greeks




