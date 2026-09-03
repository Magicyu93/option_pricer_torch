"""Semi-analytic Heston, in torch: the characteristic function and Lewis' integral.

Heston is the reason the rest of the library is shaped the way it is, and it is
also the model whose Monte Carlo answer is least self-evidently right. A
semi-analytic price is therefore not a luxury here: it is what the simulator is
tested against, and what makes calibration a matter of seconds rather than of
nested Monte Carlo.

The transform is stated on ``x = log(S_T / F_T)``. Because the rates are
deterministic and the variance process does not see them, the forward carries
the whole drift and the characteristic function below has none in it -- exactly
the split :mod:`torch_pricer.analytics.black` makes.

Two implementation choices matter.

**The "little trap" form.** Writing ``g = (beta - d) / (beta + d)`` rather than
its reciprocal keeps ``|g| <= 1`` for every ``u`` on the integration contour, so
the principal branch of the complex logarithm is the continuous one and the
price does not develop discontinuities at long maturities. The reciprocal form,
which is what Heston's 1993 paper prints, does.

**Lewis' single integral.** One integral over a contour through ``Im(u) = -1/2``
prices a call, rather than the two probabilities ``P1`` and ``P2`` of the
original formulation. The integrand is real, even, and damped by ``1/(u^2 +
1/4)``, so fixed Gauss-Legendre quadrature on a truncated domain converges to
machine precision -- and being fixed rather than adaptive, it is differentiable,
which is the whole point of putting it in torch.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from ..tensors import EPS, as_tensor

#: Complex arithmetic runs in double precision regardless of the caller's dtype:
#: the integrand is a difference of exponentials of large complex numbers, and
#: float32 loses the answer.
_COMPLEX = torch.complex128

_QUADRATURE: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}


def _gauss_legendre(n: int, device) -> tuple[Tensor, Tensor]:
    """Nodes and weights on ``[-1, 1]``, cached per (order, device)."""
    key = (int(n), torch.device(device))
    if key not in _QUADRATURE:
        nodes, weights = np.polynomial.legendre.leggauss(int(n))
        _QUADRATURE[key] = (
            torch.as_tensor(nodes, dtype=torch.float64, device=device),
            torch.as_tensor(weights, dtype=torch.float64, device=device),
        )
    return _QUADRATURE[key]


def _log1p(z: Tensor) -> Tensor:
    """``log(1 + z)`` for complex ``z``, accurate when ``|z|`` is tiny.

    torch has no complex ``log1p``, and this argument goes to zero like ``xi^2``
    while the coefficient in front of it grows like ``1 / xi^2``. Evaluating it
    as ``log(1 + z)`` there would return the rounding error of ``1.0``.
    """
    series = z * (1.0 - z * (0.5 - z / 3.0))
    return torch.where(z.abs() < 1e-4, series, torch.log(1.0 + z))


def characteristic_function(u: Tensor, t, v0, kappa, theta, xi, rho) -> Tensor:
    """``E[exp(i u log(S_T / F_T))]`` under Heston.

    Args:
        u: transform argument, complex; Lewis evaluates it at ``u - i/2``
        t: time to expiry in years
        v0: initial instantaneous variance
        kappa: mean reversion speed of the variance
        theta: long-run variance
        xi: volatility of variance
        rho: correlation between the spot and variance Brownians

    All arguments broadcast against each other.

    ``beta - d`` is never formed as written. It is a difference of two numbers
    that agree to ``O(xi^2)``, and it is then divided by ``xi^2``; taken
    literally the low vol-of-vol corner of parameter space -- which is where a
    calibration starts, and where the model must agree with Black -- returns
    noise. Multiplying through by the conjugate gives the algebraically
    identical ``-xi^2 (iu + u^2) / (beta + d)``, in which the cancellation has
    been done by hand and the ``xi^2`` cancels analytically.
    """
    u = as_tensor(u).to(_COMPLEX)
    t = as_tensor(t).to(_COMPLEX)
    v0 = as_tensor(v0).to(_COMPLEX)
    kappa = as_tensor(kappa).to(_COMPLEX)
    theta = as_tensor(theta).to(_COMPLEX)
    rho = as_tensor(rho).to(_COMPLEX)
    # Zero vol-of-vol is a removable singularity of this parameterisation, not a
    # modelling error: it is Black with a deterministic variance path. The floor
    # keeps the division defined; the algebra above keeps the limit accurate.
    xi = as_tensor(xi).to(_COMPLEX)
    xi = torch.where(xi.abs() < 1e-10, torch.full_like(xi, 1e-10), xi)

    iu = 1j * u
    psi = iu + u * u                       # the transform's quadratic form
    beta = kappa - rho * xi * iu
    d = torch.sqrt(beta * beta + xi * xi * psi)
    lo = -psi / (beta + d)                 # (beta - d) / xi^2, without the cancellation
    g = lo * xi * xi / (beta + d)          # |g| <= 1 keeps log on its principal branch
    edt = torch.exp(-d * t)

    c = kappa * theta * (lo * t - 2.0 * _log1p(g * (1.0 - edt) / (1.0 - g)) / (xi * xi))
    dd = lo * (1.0 - edt) / (1.0 - g * edt)
    return torch.exp(c + dd * v0)


def _truncation(t, v0, kappa, theta, xi, rho) -> float:
    """Where to stop integrating.

    The integrand carries two decays and the domain must respect the slower of
    them. Small vol-of-vol leaves the Gaussian ``exp(-u^2 w / 2)`` of Black;
    large vol-of-vol replaces it with ``exp(-c u)``, ``c = sqrt(1 - rho^2) (v0 +
    kappa theta t) / xi``, which only reaches the same size much further out.
    Both targets aim at about ``exp(-30)``, before the ``1 / u^2`` damping.
    """
    scalar = lambda x: float(as_tensor(x).detach().abs().max())  # noqa: E731
    t_, v0_, kappa_, theta_, xi_ = (scalar(v) for v in (t, v0, kappa, theta, xi))
    rho_ = min(scalar(rho), 1.0 - 1e-12)

    total_variance = max(0.5 * (v0_ + theta_) * t_, 1e-10)
    gaussian = 12.0 / math.sqrt(total_variance)

    rate = math.sqrt(1.0 - rho_**2) * (v0_ + kappa_ * theta_ * t_) / max(xi_, 1e-12)
    exponential = 35.0 / rate if rate > 1e-12 else 0.0

    return min(5000.0, max(50.0, gaussian, exponential))


def heston_price(
    forward,
    strike,
    t,
    v0,
    kappa,
    theta,
    xi,
    rho,
    discount=1.0,
    right=1,
    n_nodes: int = 256,
) -> Tensor:
    """European option price under Heston, by Lewis' integral.

    ``C / D = F - sqrt(F K) / pi * integral_0^inf Re[e^{i u k} phi(u - i/2)] / (u^2 + 1/4) du``

    with ``k = log(F / K)``. Puts come from parity rather than a second
    quadrature, which keeps them consistent with the calls to the last bit.

    Args:
        forward: forward level to expiry
        strike: option strike
        t: time to expiry in years
        v0, kappa, theta, xi, rho: Heston parameters, as in
            :func:`characteristic_function`
        discount: discount factor to the payment date
        right: ``+1`` call, ``-1`` put
        n_nodes: Gauss-Legendre order; 256 is machine precision for equity
            parameters and costs one matrix multiply

    Returns:
        Price, broadcast to the shape of ``forward``, ``strike``, ``t``,
        ``discount`` and ``right``.
    """
    forward, strike, t, discount, w = torch.broadcast_tensors(
        as_tensor(forward),
        as_tensor(strike),
        as_tensor(t),
        as_tensor(discount),
        as_tensor(right),
    )
    t = t.clamp_min(EPS)

    u_max = _truncation(t, v0, kappa, theta, xi, rho)
    nodes, weights = _gauss_legendre(n_nodes, forward.device)
    u = 0.5 * u_max * (nodes + 1.0)
    quad_weight = 0.5 * u_max * weights

    k = torch.log(forward / strike).unsqueeze(-1)
    phi = characteristic_function(u - 0.5j, t.unsqueeze(-1), v0, kappa, theta, xi, rho)
    integrand = torch.real(torch.exp(1j * u * k) * phi) / (u * u + 0.25)
    integral = (integrand * quad_weight).sum(-1)

    call = discount * (forward - torch.sqrt(forward * strike) / math.pi * integral)
    put = call - discount * (forward - strike)
    return torch.where(w > 0, call, put)
