"""Heston: the characteristic-function price, and what the simulation does to greeks."""

import datetime as dt
import math

import pytest
import torch

from torch_pricer.errors import ValidationError
from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.heston import HestonModel
from torch_pricer.pricer.analytic.heston import feller, heston_price
from torch_pricer.pricer.monte_carlo.engine import MCConfig, price

AS_OF, EXPIRY = dt.date(2025, 1, 2), dt.date(2026, 1, 2)
SPOT, RATE, DIV = 100.0, 0.03, 0.01
PARAMS = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)


def _market():
    return MarketSnapshot.flat(AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=DIV)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -- the analytic reference ----------------------------------------------


@pytest.mark.parametrize("theta", [0.04, 0.09])
def test_zero_vol_of_vol_collapses_to_black_scholes(theta):
    """``xi -> 0`` with ``v0 = theta`` is Black-Scholes at ``sqrt(theta)``."""
    vol, t = math.sqrt(theta), 1.0
    fwd, disc = SPOT * math.exp((RATE - DIV) * t), math.exp(-RATE * t)
    d1 = (math.log(fwd / 100.0) + 0.5 * vol**2 * t) / (vol * math.sqrt(t))
    black = disc * (fwd * _norm_cdf(d1) - 100.0 * _norm_cdf(d1 - vol * math.sqrt(t)))
    got = heston_price(SPOT, 100.0, t, RATE, DIV, v0=theta, kappa=2.0, theta=theta,
                       xi=1e-6, rho=0.0)
    assert got == pytest.approx(black, abs=1e-4)


@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_analytic_put_call_parity(strike):
    t = 1.0
    c = heston_price(SPOT, strike, t, RATE, DIV, right=1, **PARAMS)
    p = heston_price(SPOT, strike, t, RATE, DIV, right=-1, **PARAMS)
    parity = SPOT * math.exp(-DIV * t) - strike * math.exp(-RATE * t)
    assert (c - p) == pytest.approx(parity, abs=1e-8)


# -- the simulation ------------------------------------------------------


@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_simulation_matches_the_characteristic_function(strike):
    spec = VanillaOption(strike=strike, maturity=EXPIRY, right=Right.CALL, style=Style.EUROPEAN)
    res = price(spec, _market(), HestonModel(**PARAMS),
                MCConfig(n_paths=200_000, n_steps=200, seed=5))
    ref = heston_price(SPOT, strike, 1.0, RATE, DIV, right=1, **PARAMS)
    assert abs(res.price - ref) < 3 * res.stderr


def test_variance_process_is_truncated_not_reflected():
    """Full truncation: coefficients see ``max(v, 0)``, the state may go negative."""
    market = _market()
    sde = HestonModel(**PARAMS).to_sde(market)
    xt = torch.tensor([[4.6, 0.04], [4.6, -0.01]], dtype=torch.float64)
    drift, diffusion = sde.coefficients(xt, torch.tensor(0.5, dtype=torch.float64))
    # With v <= 0 the diffusion vanishes and the drift is pure mean reversion.
    assert torch.allclose(diffusion[1], torch.zeros(2, 2, dtype=torch.float64), atol=1e-6)
    assert float(drift[1, 1]) == pytest.approx(
        float(PARAMS["kappa"] * PARAMS["theta"]), rel=1e-9
    )


def test_two_factors_with_the_right_correlation():
    market = _market()
    sde = HestonModel(**PARAMS).to_sde(market)
    xt = torch.tensor([[4.6, 0.04]], dtype=torch.float64)
    d = sde.diffusion_coefficient(xt, torch.tensor(0.5, dtype=torch.float64))[0]
    assert d.shape == (2, 2)
    # Rebuild the covariance: the (0,1) entry over the norms is rho.
    cov = d @ d.T
    implied_rho = float(cov[0, 1] / torch.sqrt(cov[0, 0] * cov[1, 1]))
    assert implied_rho == pytest.approx(PARAMS["rho"], rel=1e-6)


def test_model_params_greek_covers_every_parameter():
    spec = VanillaOption(strike=100.0, maturity=EXPIRY, right=Right.CALL, style=Style.EUROPEAN)
    res = price(spec, _market(), HestonModel(**PARAMS),
                MCConfig(n_paths=20_000, n_steps=50, seed=5), greeks=("model_params",))
    assert set(res.greeks["model_params"]) == {"v0", "kappa", "theta", "xi", "rho"}


def test_pathwise_v0_greek_is_accurate_when_feller_holds():
    """Feller satisfied: 40 seeds put every parameter within 2.3% of analytic.

    Violated, the estimator's variance rises by one to two orders of magnitude
    (see the module docstring), so only the well-behaved regime is asserted.
    """
    safe = dict(v0=0.04, kappa=4.0, theta=0.04, xi=0.3, rho=-0.7)
    assert feller(safe["kappa"], safe["theta"], safe["xi"]) > 0
    spec = VanillaOption(strike=100.0, maturity=EXPIRY, right=Right.CALL, style=Style.EUROPEAN)
    res = price(spec, _market(), HestonModel(**safe),
                MCConfig(n_paths=200_000, n_steps=200, seed=5), greeks=("model_params",))
    h = 1e-5
    up, dn = dict(safe), dict(safe)
    up["v0"] += h
    dn["v0"] -= h
    ref = (heston_price(SPOT, 100.0, 1.0, RATE, DIV, right=1, **up)
           - heston_price(SPOT, 100.0, 1.0, RATE, DIV, right=1, **dn)) / (2 * h)
    assert float(res.greeks["model_params"]["v0"]) == pytest.approx(ref, rel=0.02)


def test_feller_margin_is_reported():
    assert HestonModel(**PARAMS).feller < 0  # the default set violates it
    assert HestonModel(v0=0.04, kappa=4.0, theta=0.04, xi=0.3, rho=-0.7).feller > 0


def test_construction_rejects_nonsense():
    with pytest.raises(ValidationError, match="must be positive"):
        HestonModel(v0=-0.01)
    with pytest.raises(ValidationError, match="rho must lie"):
        HestonModel(rho=-1.5)
