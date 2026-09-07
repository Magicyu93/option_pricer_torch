"""Longstaff-Schwartz: early exercise by Monte Carlo.

The engine prices a terminal payoff. An American option pays at a *stopping
time*, and the rule that decides it depends on the continuation value -- a
conditional expectation the simulation does not carry. Longstaff-Schwartz
estimates it by regressing realised continuation values on functions of the
current state, working backwards, and exercises where intrinsic beats the fit.

Three things about this implementation are load-bearing.

**The regression is fitted on in-the-money paths only.** Out of the money the
exercise decision is not close, and including those paths spends the basis
fitting a region where the answer is already known, degrading it where it is not.
This is the detail from the original paper that most reimplementations drop.

**The exercise policy is detached, and that is not a shortcut.** A naive reading
says the price depends on the exercise boundary, so the boundary must be
differentiated too. It does not: at the optimal boundary the option is exactly
indifferent between exercising and holding -- smooth pasting -- so the derivative
of the value with respect to the boundary is zero, and the envelope theorem lets
the policy be held fixed while differentiating. Practically this matters twice
over, because the alternative is differentiating an indicator, which is the same
Dirac that makes ``gamma_autograd`` return zero. Detaching the decision and the
regression coefficients gives the correct pathwise delta; not detaching gives
nothing at all.

**Fitting and exercising on the same paths biases the price up.** The regression
has seen each path's own future, so the policy exercises with a sliver of
hindsight no real holder has -- foresight bias, and it is upward. Fitting the
policy on an independent set removes it, leaving only the downward bias of a
finite basis, which is honest: the result is then a genuine lower bound on the
option's value. ``LSMConfig.policy_paths`` turns that on, and it is on by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torch_pricer.errors import ValidationError


@dataclass(frozen=True)
class LSMConfig:
    """Settings for the least-squares Monte Carlo exercise policy.

    Separate from :class:`~torch_pricer.pricer.monte_carlo.engine.MCConfig`
    because it applies to a minority of contracts and every field here is
    meaningless for a European one.
    """

    #: Degree of the polynomial basis in moneyness ``S / K``. Three is the
    #: common choice and what the original paper's examples use; past about five
    #: the design matrix conditions badly without buying accuracy.
    basis_degree: int = 3
    #: Exercise opportunities, spread evenly over the simulation grid. ``None``
    #: uses every step, which is the closest this discretisation gets to a truly
    #: continuous American right.
    n_exercise_dates: int | None = None
    #: Paths used to fit the policy, simulated separately from the ones it is
    #: applied to. ``None`` reuses the pricing paths, which is faster and biased
    #: upward -- see the module docstring.
    policy_paths: int | None = 50_000
    #: Seed for those policy paths. Distinct from the pricing seed by
    #: construction: sharing it would defeat the point of the split.
    policy_seed: int = 987_654_321
    #: Fewest in-the-money paths at a date before its regression is trusted.
    #: Below this the fit is noise, and holding is the safer default.
    min_itm_paths: int = 32


def exercise_indices(n_steps: int, n_dates: int | None) -> list[int]:
    """Time-grid indices at which exercise is allowed, ascending, ending at ``n_steps``.

    Index 0 is never included: exercising at inception is a decision the holder
    makes before buying, not one the model prices.
    """
    if n_dates is None or n_dates >= n_steps:
        return list(range(1, n_steps + 1))
    if n_dates < 1:
        raise ValidationError(f"n_exercise_dates must be at least 1, got {n_dates}")
    step = n_steps / n_dates
    return sorted({max(1, min(n_steps, round((i + 1) * step))) for i in range(n_dates)})


def _basis(moneyness: Tensor, degree: int) -> Tensor:
    """Design matrix ``[1, x, x^2, ...]`` of shape ``(n_paths, degree + 1)``.

    In ``S / K`` rather than ``S`` so the columns stay near unity and the normal
    equations stay conditioned whatever the strike happens to be.
    """
    return torch.stack([moneyness**i for i in range(degree + 1)], dim=1)


def fit_policy(
    asset_path: Tensor,
    discounts: Tensor,
    intrinsic,
    strike: float,
    indices: list[int],
    lsm: LSMConfig,
) -> dict[int, Tensor]:
    """Regression coefficients for each exercise date, working backwards.

    Runs entirely under ``no_grad``: these coefficients define a *policy*, and
    the policy is held fixed when the price is differentiated.
    """
    with torch.no_grad():
        cash = discounts[indices[-1]] * intrinsic(asset_path[:, indices[-1]], None)
        coefficients: dict[int, Tensor] = {}
        for k in reversed(indices[:-1]):
            spot_k = asset_path[:, k]
            exercise = discounts[k] * intrinsic(spot_k, None)
            itm = exercise > 0
            if int(itm.sum()) < lsm.min_itm_paths:
                continue
            design = _basis(spot_k[itm] / strike, lsm.basis_degree)
            beta = torch.linalg.lstsq(design, cash[itm].unsqueeze(1)).solution.squeeze(1)
            coefficients[k] = beta
            take = exercise[itm] > design @ beta
            cash = torch.where(
                torch.zeros_like(itm).masked_scatter(itm, take), exercise, cash
            )
    return coefficients


def value(
    asset_path: Tensor,
    discounts: Tensor,
    intrinsic,
    strike: float,
    indices: list[int],
    lsm: LSMConfig,
    coefficients: dict[int, Tensor],
) -> Tensor:
    """Discounted cashflow per path under a fixed exercise policy.

    Differentiable in the path and in the discount curve; the decision itself is
    a detached mask, so delta comes off this by autograd exactly as it does for a
    European payoff.
    """
    cash = discounts[indices[-1]] * intrinsic(asset_path[:, indices[-1]], None)
    for k in reversed(indices[:-1]):
        beta = coefficients.get(k)
        if beta is None:
            continue
        spot_k = asset_path[:, k]
        exercise = discounts[k] * intrinsic(spot_k, None)
        itm = exercise > 0
        if not bool(itm.any()):
            continue
        design = _basis(spot_k[itm].detach() / strike, lsm.basis_degree)
        take = exercise[itm].detach() > design @ beta
        # The mask is the only thing the policy contributes; both branches of the
        # where() carry gradient, so the value stays differentiable through
        # whichever cashflow each path ends up paying.
        cash = torch.where(
            torch.zeros_like(itm).masked_scatter(itm, take), exercise, cash
        )
    return cash
