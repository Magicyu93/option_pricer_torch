"""SVI surface: analytic derivatives, arbitrage checks, and differentiability."""

import datetime as dt
import math

import pytest
import torch

from torch_pricer.errors import ValidationError
from torch_pricer.market.svi import SVISlice, SVISurface

AS_OF = dt.date(2025, 1, 2)


def _forward(t):
    return torch.as_tensor(100.0, dtype=torch.float64) * torch.exp(0.02 * torch.as_tensor(t))


def _surface():
    return SVISurface(
        [
            SVISlice(0.010, 0.055, -0.55, 0.02, 0.14, 0.25),
            SVISlice(0.022, 0.075, -0.50, 0.02, 0.18, 1.00),
            SVISlice(0.046, 0.095, -0.45, 0.03, 0.22, 2.00),
        ],
        forward=_forward,
        as_of=AS_OF,
    )


def test_k_derivatives_are_exact():
    """Analytic, not autograd -- Dupire needs the second derivative on every step."""
    sl = SVISlice(0.025, 0.081, -0.48, 0.03, 0.19, 1.0)
    k = torch.linspace(-1.2, 1.2, 9, dtype=torch.float64, requires_grad=True)
    w = sl.total_variance(k)
    (g1,) = torch.autograd.grad(w.sum(), k, create_graph=True)
    (g2,) = torch.autograd.grad(g1.sum(), k)
    assert torch.allclose(sl.d_dk(k), g1, atol=1e-14)
    assert torch.allclose(sl.d2_dk2(k), g2, atol=1e-13)


def test_dw_dT_matches_the_interpolation_it_describes():
    s = _surface()
    k = torch.linspace(-0.6, 0.6, 5, dtype=torch.float64)
    for t in (0.1, 0.5, 1.4, 3.0):
        h = 1e-6
        fd = (s.total_variance(k, t + h) - s.total_variance(k, t - h)) / (2 * h)
        assert torch.allclose(s.dw_dT(k, t), fd, atol=1e-8)


def test_dw_dT_is_positive_everywhere():
    """Calendar-arbitrage-free, and Dupire's numerator, so a zero would kill local vol."""
    s = _surface()
    k = torch.linspace(-1.5, 1.5, 41, dtype=torch.float64)
    for t in (0.05, 0.25, 0.9, 2.0, 5.0):
        assert float(s.dw_dT(k, t).min()) > 0


def test_expiry_stays_in_the_graph():
    """Only the bracket index may be detached; detaching t would cut theta."""
    s = _surface()
    t = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    k = torch.linspace(-0.5, 0.5, 5, dtype=torch.float64)
    (g,) = torch.autograd.grad(s.total_variance(k, t).sum(), t)
    assert g is not None
    assert float(g) == pytest.approx(float(s.dw_dT(k, t).sum().detach()), rel=1e-10)


def test_slice_parameters_are_differentiable():
    """Bucketed vega chains through these; a float would end it here."""
    sl = SVISlice(0.025, 0.081, -0.48, 0.03, 0.19, 1.0)
    names = {n for n, p in sl.named_parameters() if p.requires_grad}
    assert names == {"a", "b", "rho", "m", "sigma"}


def test_arbitrage_report_flags_a_bad_surface():
    good = _surface()
    report = good.arbitrage_report()
    assert report["butterfly"] > 0 and report["calendar"] > 0

    # A large b with a tiny sigma bends the smile past the butterfly bound.
    bad = SVISurface([SVISlice(0.02, 1.6, -0.9, 0.0, 0.01, 1.0)], _forward, AS_OF)
    assert bad.arbitrage_report()["butterfly"] < 0


def test_vol_round_trips_through_total_variance():
    s = _surface()
    t, k_strike = 1.0, torch.tensor([80.0, 100.0, 130.0], dtype=torch.float64)
    vol = s.vol(k_strike, t)
    k = s.log_moneyness(k_strike, t)
    assert torch.allclose(vol, torch.sqrt(s.total_variance(k, t) / t))


def test_construction_rejects_nonsense():
    with pytest.raises(ValidationError, match="at least one slice"):
        SVISurface([], _forward, AS_OF)
    with pytest.raises(ValidationError, match="duplicate"):
        SVISurface([SVISlice(0.02, 0.08, -0.4, 0, 0.2, 1.0)] * 2, _forward, AS_OF)
    with pytest.raises(ValidationError, match="expiry must be positive"):
        SVISlice(0.02, 0.08, -0.4, 0.0, 0.2, 0.0)
