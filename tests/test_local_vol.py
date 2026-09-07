"""Local vol: Dupire consistency, and differentiability of the whole chain."""

import datetime as dt
import math

import pytest

import torch

from torch_pricer.pricer.analytic.black import implied_vol
from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.svi import SVISlice, SVISurface
from torch_pricer.models.local_vol import LocalVolModel
from torch_pricer.pricer.monte_carlo.engine import MCConfig, price

from .conftest import averaged

AS_OF = dt.date(2025, 1, 2)
SPOT, RATE, DIV = 100.0, 0.03, 0.01


def _market():
    return MarketSnapshot.flat(AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=DIV)


def _surface(market):
    return SVISurface(
        [
            SVISlice(0.010, 0.055, -0.55, 0.02, 0.14, 0.25),
            SVISlice(0.022, 0.075, -0.50, 0.02, 0.18, 1.00),
            SVISlice(0.046, 0.095, -0.45, 0.03, 0.22, 2.00),
        ],
        forward=market.forward,
        as_of=AS_OF,
    )


def test_local_vol_is_finite_and_positive_across_the_grid():
    market = _market()
    sde = LocalVolModel(_surface(market)).to_sde(market)
    spots = torch.tensor([60.0, 80.0, 100.0, 130.0, 170.0], dtype=torch.float64)
    for t in (0.01, 0.1, 0.5, 1.0, 1.9, 3.0):
        lv = sde.local_vol(spots, t)
        assert torch.isfinite(lv).all()
        assert float(lv.min()) > 0.0


@pytest.mark.parametrize("strike", [90.0, 100.0, 115.0])
def test_dupire_reproduces_the_generating_surface(strike):
    """The acid test.

    A local vol derived from a surface must, when simulated, reprice that
    surface's own implied vols. Nothing else checks Dupire, the SVI derivatives,
    the simulator and the payoff all at once.

    Euler on a state-dependent diffusion is only weakly first order, so a
    residual bias is expected and must shrink with the step count. Measured at
    100k paths, 1y, strikes 85/100/115, against an MC error of +/- 8bp:

        50 steps  -> +17 to +27 bp
        150 steps -> +18 to +20 bp
        400 steps -> + 7 to + 9 bp

    The tolerance below is set for the 250 steps this test runs.
    """
    market = _market()
    surface = _surface(market)
    expiry = dt.date(2026, 1, 2)
    spec = VanillaOption(strike=strike, maturity=expiry, right=Right.CALL, style=Style.EUROPEAN)

    # 250 steps is load-bearing -- the bias table above is quoted against it, and
    # the 35bp tolerance is set for it -- so only the path count comes down, and
    # the lost accuracy is bought back by averaging independent seeds instead.
    res_price, _, _ = averaged(
        lambda seed: price(
            spec, market, LocalVolModel(surface),
            MCConfig(n_paths=50_000, n_steps=250, seed=seed, checkpoint_segments=16),
        ),
        range(4),
    )
    t = market.time_to(expiry)
    fwd, disc = SPOT * math.exp((RATE - DIV) * t), math.exp(-RATE * t)
    mc_vol = float(implied_vol(res_price, fwd, strike, t, disc, 1))
    target = float(surface.vol(torch.tensor(strike, dtype=torch.float64), t).detach())

    assert abs(mc_vol - target) < 0.0035


def test_price_is_differentiable_in_the_slice_parameters():
    """This is what makes bucketed vega possible without differentiating a fit."""
    market = _market()
    surface = _surface(market)
    model = LocalVolModel(surface)
    spec = VanillaOption(
        strike=100.0, maturity=dt.date(2026, 1, 2), right=Right.CALL, style=Style.EUROPEAN
    )
    res = price(
        spec, market, model, MCConfig(n_paths=20_000, n_steps=50, seed=3),
        greeks=("delta", "model_params"),
    )
    params = res.greeks["model_params"]
    assert {n.split(".")[-1] for n in params} == {"a", "b", "rho", "m", "sigma"}
    # The slice bracketing the expiry must actually move the price.
    assert max(abs(float(v)) for v in params.values()) > 1e-6
    assert res.greeks["delta"] > 0


def test_checkpointing_is_numerically_transparent():
    market = _market()
    model = LocalVolModel(_surface(market))
    spec = VanillaOption(
        strike=100.0, maturity=dt.date(2026, 1, 2), right=Right.CALL, style=Style.EUROPEAN
    )
    kw = dict(n_paths=20_000, n_steps=60, seed=3)
    plain = price(spec, market, model, MCConfig(**kw), greeks=("delta",))
    ckpt = price(spec, market, model, MCConfig(checkpoint_segments=8, **kw), greeks=("delta",))
    assert plain.price == ckpt.price
    assert plain.greeks["delta"] == ckpt.greeks["delta"]


def test_gamma_autograd_is_nonzero_but_incomplete_under_local_vol():
    """The one model where a second backward pass returns something.

    ``sigma_LV(S_t, t)`` makes the path a nonlinear function of ``S_0``, so a
    genuine second-order term survives where Black-Scholes and Heston have none.
    It is still missing the density term at the strike, so it recovers only part
    of gamma -- roughly a third at this configuration -- and must not be used as
    gamma. Asserting that it is neither zero nor right is the point.
    """
    market = _market()
    spec = VanillaOption(
        strike=100.0, maturity=dt.date(2026, 1, 2), right=Right.CALL, style=Style.EUROPEAN
    )
    res = price(
        spec, market, LocalVolModel(_surface(market)),
        # gamma_autograd runs a second backward pass with create_graph, which
        # retains the double-backward graph on top of the forward one. Checkpoint
        # segments do not help: the recomputation graph is retained too. At the
        # 100k x 100 this used to run, peak demand exceeded 11.6 GB and the test
        # could not pass on the card at all -- the contrast it asserts is just as
        # visible an order of magnitude smaller.
        MCConfig(n_paths=20_000, n_steps=50, seed=7),
        greeks=("gamma", "gamma_autograd"),
    )
    assert res.greeks["gamma"] > 0.01
    assert res.greeks["gamma_autograd"] > 0.0
    assert res.greeks["gamma_autograd"] < 0.75 * res.greeks["gamma"]


def test_local_vol_needs_an_svi_surface():
    market = _market()
    with pytest.raises(ValidationError, match="needs an SVISurface"):
        LocalVolModel(market.vol_surface)
