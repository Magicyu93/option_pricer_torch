"""Fitting SVI slices to quoted implied vols, keeping the sensitivity.

Each expiry is fitted on its own -- a quote at ``T_i`` is priced by slice ``i``
and by nothing else -- so the surface's Jacobian is block diagonal by
construction. The blocks are assembled into one
:class:`~torch_pricer.calibration.result.CalibrationResult` anyway, because that
is what lets ``dV/d(quote)`` be taken across the whole surface in one chain rule
rather than expiry by expiry.

Residuals are in implied vol, not price. Vol is the unit the market quotes a
surface in, it is roughly homoscedastic across strikes where premium is not, and
it keeps a 5-vol wing error from being drowned out by a penny of ATM premium.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

import torch
from torch import Tensor

from torch_pricer.calibration.fit import least_squares
from torch_pricer.calibration.inputs import CalibrationInputs
from torch_pricer.calibration.result import CalibrationResult
from torch_pricer.errors import CalibrationError, ValidationError
from torch_pricer.market.svi import SVI_PARAMS, SVISlice, SVISurface
from torch_pricer.tensors import as_tensor

#: Box bounds keeping a slice inside the region where SVI is well posed:
#: b >= 0 (convex), |rho| < 1, sigma > 0.
_LOWER = (-5.0, 1e-8, -0.999, -5.0, 1e-4)
_UPPER = (5.0, 5.0, 0.999, 5.0, 5.0)


def fit_slice(
    slice_: SVISlice,
    log_moneyness: Tensor,
    target_vols: Tensor,
    weights: Tensor | None = None,
    **kwargs,
) -> CalibrationResult:
    """Fit one slice to quoted vols at the given log-moneyness. Mutates ``slice_``."""
    k = as_tensor(log_moneyness).flatten().to(torch.float64)
    target = as_tensor(target_vols).flatten().to(torch.float64)
    if k.numel() != target.numel():
        raise ValidationError(f"{k.numel()} strikes but {target.numel()} vols")
    if k.numel() < len(SVI_PARAMS):
        raise CalibrationError(
            f"an SVI slice has {len(SVI_PARAMS)} parameters but only {k.numel()} quotes; "
            "the fit would not identify them, and the quote sensitivity would be singular"
        )
    expiry = float(slice_.expiry)

    def residual(theta: Tensor) -> Tensor:
        a, b, rho, m, sigma = theta
        u = k - m
        w = a + b * (rho * u + torch.sqrt(u * u + sigma**2))
        return torch.sqrt(w.clamp_min(1e-12) / expiry) - target

    result = least_squares(
        residual_fn=residual,
        theta0=slice_.vector().detach(),
        param_names=SVI_PARAMS,
        quote_ids=tuple(f"k={float(x):+.4f}" for x in k),
        weights=weights,
        bounds=(as_tensor(_LOWER).to(torch.float64), as_tensor(_UPPER).to(torch.float64)),
        **kwargs,
    )
    with torch.no_grad():
        for name, value in zip(SVI_PARAMS, result.params):
            getattr(slice_, name).copy_(value)
    return result


def fit_surface(surface: SVISurface, inputs: CalibrationInputs) -> CalibrationResult:
    """Fit every slice to its expiry's quotes; return one block-diagonal result.

    Quotes must carry an ``implied_vol``; inverting premiums to vols is the
    caller's job, so that a failed inversion is reported where the bad quote is
    rather than as a mysterious wide residual here.
    """
    quotes = inputs.quotes
    if quotes is None or not quotes.options:
        raise CalibrationError("no option quotes to calibrate against")

    by_expiry: dict[dt.date, list] = defaultdict(list)
    for q in quotes.options:
        if q.implied_vol is None:
            raise CalibrationError(
                f"quote {q.expiry} {q.strike:g} has no implied_vol; invert its premium "
                "with black_formula.implied_vol before calibrating"
            )
        by_expiry[q.expiry].append(q)

    market = inputs.market
    blocks: list[CalibrationResult] = []
    for slice_ in surface.slices:
        target_t = float(slice_.expiry)
        match = [d for d in by_expiry if abs(market.time_to(d) - target_t) < 1e-6]
        if not match:
            raise CalibrationError(
                f"no quotes for the slice at T={target_t:g}; expiries available: "
                f"{sorted(round(market.time_to(d), 6) for d in by_expiry)}"
            )
        group = sorted(by_expiry[match[0]], key=lambda q: q.strike)
        fwd = market.forward(as_tensor(target_t))
        k = torch.log(as_tensor([q.strike for q in group]).to(torch.float64) / fwd)
        vols = as_tensor([q.implied_vol for q in group]).to(torch.float64)
        blocks.append(fit_slice(slice_, k.detach(), vols))

    return _block_diagonal(blocks, [float(s.expiry) for s in surface.slices])


def _block_diagonal(blocks: list[CalibrationResult], expiries: list[float]) -> CalibrationResult:
    """Stack per-slice results into one surface-wide result."""
    names, ids, res, wts = [], [], [], []
    for T, blk in zip(expiries, blocks):
        names += [f"T{T:g}:{n}" for n in blk.param_names]
        ids += [f"T{T:g}|{q}" for q in blk.quote_ids]
        res.append(blk.residuals)
        wts.append(blk.weights)
    return CalibrationResult(
        param_names=tuple(names),
        params=torch.cat([b.params for b in blocks]),
        jacobian=torch.block_diag(*[b.jacobian for b in blocks]),
        hessian=torch.block_diag(*[b.hessian for b in blocks]),
        residuals=torch.cat(res),
        weights=torch.cat(wts),
        quote_ids=tuple(ids),
        converged=all(b.converged for b in blocks),
        iterations=max(b.iterations for b in blocks),
    )
