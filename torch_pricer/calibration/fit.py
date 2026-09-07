"""A damped least-squares fitter that hands back its own sensitivity.

Levenberg-Marquardt rather than plain Gauss-Newton: an SVI slice fitted to a
skewed smile has a long curved valley, and an undamped step overshoots it and
walks the parameters somewhere the surface is not arbitrage-free. The damping
costs one extra solve per iteration and removes that failure mode.

The Jacobian and the exact Hessian are recomputed once at the optimum, by
autograd, and travel out in the :class:`~torch_pricer.calibration.result.CalibrationResult`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor

from torch_pricer.calibration.result import CalibrationResult
from torch_pricer.errors import CalibrationError


def least_squares(
    residual_fn: Callable[[Tensor], Tensor],
    theta0: Tensor,
    param_names: Sequence[str],
    quote_ids: Sequence[str] = (),
    weights: Tensor | None = None,
    max_iter: int = 200,
    tol: float = 1e-14,
    damping: float = 1e-6,
    bounds: tuple[Tensor, Tensor] | None = None,
) -> CalibrationResult:
    """Minimise ``0.5 * sum_i w_i r_i(theta)^2`` and report how to differentiate it.

    Args:
        residual_fn: ``theta -> residuals``, differentiable, shape ``(n_quotes,)``.
            Residuals must be ``model - target``, in that order: the sign is what
            makes ``d(theta)/d(target)`` come out positive-definite.
        theta0: starting parameters, shape ``(n_params,)``
        bounds: optional ``(lower, upper)``, applied by clamping each step

    Returns:
        The fitted point plus the Jacobian and exact Hessian at it.
    """
    theta = theta0.detach().clone().to(torch.float64)
    n_params = theta.numel()
    r0 = residual_fn(theta)
    w = torch.ones_like(r0) if weights is None else weights.to(r0.dtype)
    if w.shape != r0.shape:
        raise CalibrationError(f"weights {tuple(w.shape)} do not match residuals {tuple(r0.shape)}")

    def loss(th: Tensor) -> Tensor:
        r = residual_fn(th)
        return 0.5 * (w * r**2).sum()

    lam = damping
    prev = float(loss(theta))
    converged, iterations = False, 0

    for iterations in range(1, max_iter + 1):
        jac = torch.autograd.functional.jacobian(residual_fn, theta)  # (n_quotes, n_params)
        res = residual_fn(theta).detach()
        jtw = jac.T * w
        grad = jtw @ res
        normal = jtw @ jac

        # Try the damped step; back off until it actually improves the loss.
        for _ in range(30):
            step = torch.linalg.solve(
                normal + lam * torch.eye(n_params, dtype=theta.dtype), -grad
            )
            trial = theta + step
            if bounds is not None:
                trial = torch.clamp(trial, bounds[0], bounds[1])
            current = float(loss(trial))
            if current <= prev:
                theta, lam = trial, max(lam * 0.5, 1e-12)
                break
            lam *= 4.0
        else:
            break  # no downhill step exists; we are at the optimum or stuck

        if abs(prev - current) <= tol * max(1.0, abs(prev)):
            prev = current
            converged = True
            break
        prev = current

    theta = theta.detach()
    jac = torch.autograd.functional.jacobian(residual_fn, theta)
    hess = torch.autograd.functional.hessian(loss, theta)
    return CalibrationResult(
        param_names=tuple(param_names),
        params=theta,
        jacobian=jac,
        hessian=hess,
        residuals=residual_fn(theta).detach(),
        weights=w,
        quote_ids=tuple(quote_ids),
        converged=converged,
        iterations=iterations,
    )
