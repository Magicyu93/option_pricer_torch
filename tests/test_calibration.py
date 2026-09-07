"""The vega chain: fit a slice, then differentiate through the fit."""

import datetime as dt

import pytest
import torch

from torch_pricer.calibration.svi_fit import fit_slice
from torch_pricer.errors import CalibrationError
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.svi import SVISlice

AS_OF, T = dt.date(2025, 1, 2), 1.0


def _market():
    return MarketSnapshot.flat(AS_OF, spot=100.0, flat_rate=0.03, flat_dividend=0.01)


def _k(strikes):
    fwd = _market().forward(torch.tensor(T, dtype=torch.float64))
    return torch.log(torch.as_tensor(strikes, dtype=torch.float64) / fwd).detach()


def _fit(k, vols):
    sl = SVISlice(0.03, 0.06, -0.4, 0.0, 0.2, T)
    return sl, fit_slice(sl, k, vols)


def test_exactly_determined_fit_is_exact():
    k = _k([80.0, 90.0, 100.0, 110.0, 125.0])
    vols = torch.tensor([0.242, 0.221, 0.205, 0.199, 0.204], dtype=torch.float64)
    _, res = _fit(k, vols)
    assert res.converged
    assert res.rmse < 1e-8  # 5 parameters, 5 quotes


def test_param_sensitivity_matches_bump_and_recalibrate():
    """The implicit function theorem, against the thing it replaces."""
    k = _k([70.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 135.0, 150.0])
    vols = torch.tensor(
        [0.281, 0.246, 0.234, 0.223, 0.212, 0.2035, 0.1985, 0.2015, 0.213, 0.229],
        dtype=torch.float64,
    )
    _, res = _fit(k, vols)

    eps, cols = 1e-6, []
    for i in range(vols.numel()):
        up, dn = vols.clone(), vols.clone()
        up[i] += eps
        dn[i] -= eps
        cols.append((_fit(k, up)[1].params - _fit(k, dn)[1].params) / (2 * eps))
    bump = torch.stack(cols, dim=1)
    scale = float(bump.abs().max())

    exact = res.param_sensitivity()
    assert float((exact - bump).abs().max()) / scale < 1e-4


def test_gauss_newton_is_the_worse_approximation_when_residuals_are_real():
    """``J^T W J`` drops a term that is only negligible at a perfect fit.

    With 5 parameters against 10 quotes the residuals are a few basis points and
    the shortcut is off by more than 10%, which is why
    ``param_sensitivity`` defaults to the exact Hessian.
    """
    k = _k([70.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 135.0, 150.0])
    vols = torch.tensor(
        [0.281, 0.246, 0.234, 0.223, 0.212, 0.2035, 0.1985, 0.2015, 0.213, 0.229],
        dtype=torch.float64,
    )
    _, res = _fit(k, vols)
    assert res.rmse > 1e-5  # the fit is genuinely imperfect

    eps, cols = 1e-6, []
    for i in range(vols.numel()):
        up, dn = vols.clone(), vols.clone()
        up[i] += eps
        dn[i] -= eps
        cols.append((_fit(k, up)[1].params - _fit(k, dn)[1].params) / (2 * eps))
    bump = torch.stack(cols, dim=1)
    scale = float(bump.abs().max())

    err_exact = float((res.param_sensitivity() - bump).abs().max()) / scale
    err_gn = float((res.param_sensitivity(gauss_newton=True) - bump).abs().max()) / scale
    assert err_gn > 0.05
    assert err_gn > 100 * err_exact


def test_market_sensitivity_chains_shapes_correctly():
    k = _k([80.0, 90.0, 100.0, 110.0, 125.0])
    vols = torch.tensor([0.242, 0.221, 0.205, 0.199, 0.204], dtype=torch.float64)
    _, res = _fit(k, vols)
    dv_dtheta = torch.ones(5, dtype=torch.float64)
    assert res.market_sensitivity(dv_dtheta).shape == (5,)
    with pytest.raises(CalibrationError, match="expected a sensitivity"):
        res.market_sensitivity(torch.ones(3, dtype=torch.float64))


def test_underdetermined_fit_is_refused():
    """Fewer quotes than parameters leaves the sensitivity singular."""
    with pytest.raises(CalibrationError, match="would not identify"):
        _fit(_k([90.0, 100.0, 110.0]), torch.tensor([0.22, 0.20, 0.21], dtype=torch.float64))
