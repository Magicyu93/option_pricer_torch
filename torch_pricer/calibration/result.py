"""What a calibration hands back, and the sensitivity that hides inside it.

A calibrator that mutates parameters in place and returns ``None`` throws away
the only thing that connects fitted parameters to the quotes they were fitted
to. That link is what vega is, for every model whose parameters are not
themselves quoted vols.

For Black-Scholes the parameter *is* the quoted vol, so ``dV/d(parameter)`` is
already vega and none of this is needed. For Heston, or for local vol driven by
a fitted surface, it is not, and the chain rule has to run through the fit:

    dV/d(quote)  =  (dV/d(param))^T . (d(param)/d(quote))

The second factor comes from the implicit function theorem. Calibration solves

    theta* = argmin_theta  0.5 * sum_i w_i (M_i(theta) - sigma_i)^2

so at the optimum the first-order condition

    g(theta*, sigma) = J^T W (M(theta) - sigma) = 0

holds *identically in sigma*. Differentiating it totally and rearranging,

    d(theta)/d(sigma) = H^-1 J^T W,    H = J^T W J + sum_i w_i r_i grad^2 M_i

which costs one linear solve of size ``n_params`` -- five for Heston -- against
one full recalibration *per quote* for a bump-and-refit. That is the whole
argument for keeping ``J`` and ``H`` rather than discarding them.

``H`` is the true Hessian, not the Gauss-Newton approximation ``J^T W J``. The
two agree only at a perfect fit; with the residuals a real surface leaves
behind, dropping the second-order term is a percent-level error in the
sensitivity. :meth:`CalibrationResult.param_sensitivity` defaults to the exact
form and offers the cheaper one explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torch_pricer.errors import CalibrationError


@dataclass(frozen=True)
class CalibrationResult:
    """The fitted point, and everything needed to differentiate through it."""

    param_names: tuple[str, ...]
    params: Tensor  # (n_params,)
    jacobian: Tensor  # (n_quotes, n_params)  dM/dtheta at the optimum
    hessian: Tensor  # (n_params, n_params)   d2L/dtheta2, exact
    residuals: Tensor  # (n_quotes,)          M(theta) - target
    weights: Tensor  # (n_quotes,)
    quote_ids: tuple[str, ...]
    converged: bool
    iterations: int

    @property
    def gauss_newton_hessian(self) -> Tensor:
        """``J^T W J`` -- the Hessian with the residual term dropped."""
        return (self.jacobian.T * self.weights) @ self.jacobian

    def param_sensitivity(self, gauss_newton: bool = False) -> Tensor:
        """``d(theta)/d(quote)``, shape ``(n_params, n_quotes)``.

        Args:
            gauss_newton: drop the residual term from the Hessian. Cheaper, and
                exact only when the fit is perfect.
        """
        h = self.gauss_newton_hessian if gauss_newton else self.hessian
        try:
            return torch.linalg.solve(h, self.jacobian.T * self.weights)
        except RuntimeError as exc:  # singular: the fit does not pin the params
            raise CalibrationError(
                "calibration Hessian is singular, so quote sensitivities are not "
                "defined; the fit does not identify all parameters"
            ) from exc

    def market_sensitivity(self, d_dparams: Tensor, gauss_newton: bool = False) -> Tensor:
        """Chain ``dV/d(param)`` into ``dV/d(quote)``, shape ``(n_quotes,)``."""
        if d_dparams.shape[-1] != len(self.param_names):
            raise CalibrationError(
                f"expected a sensitivity in {len(self.param_names)} parameters, "
                f"got shape {tuple(d_dparams.shape)}"
            )
        return d_dparams @ self.param_sensitivity(gauss_newton=gauss_newton)

    @property
    def rmse(self) -> float:
        """Weighted root-mean-square residual, in the units the target was quoted in."""
        w = self.weights
        return float(torch.sqrt((w * self.residuals**2).sum() / w.sum()))

    def __repr__(self) -> str:  # pragma: no cover
        vals = ", ".join(
            f"{n}={float(v):.5g}" for n, v in zip(self.param_names, self.params)
        )
        flag = "" if self.converged else " NOT CONVERGED"
        return (
            f"CalibrationResult({vals}, rmse={self.rmse:.3e}, "
            f"iters={self.iterations}{flag})"
        )
