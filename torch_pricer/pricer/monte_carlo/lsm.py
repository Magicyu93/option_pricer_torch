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
    #: Evenly spaced exercise opportunities over the simulation grid. ``None``
    #: uses every step -- the closest this discretisation gets to a continuously
    #: exercisable right, and what an American *put* wants, since its exercise
    #: region is rate-driven and open at all times.
    #:
    #: ``0`` means no evenly spaced dates at all, leaving only expiry and
    #: whatever :attr:`align_to_dividends` contributes. That is the right setting
    #: for a dividend-paying *call*, and the reason is worth stating: such a call
    #: is optimally exercised only in the instant before an ex-date, so every
    #: additional date is an opportunity for a noisy regression to exercise when
    #: it should not -- and exercising a call early when it is not optimal
    #: destroys value outright. Measured on a 1y call with two 2.00 dividends,
    #: as the fraction of the true early-exercise premium recovered: with uniform
    #: dates 58%, 28%, 32%, -40% at 100, 200, 400 and 800 of them; with the
    #: ex-date instants alone, 87% to 93% and near-flat in the basis degree.
    #: More exercise opportunities made it monotonically worse.
    n_exercise_dates: int | None = None
    #: Also allow exercise at the last grid step before each ex-dividend date.
    #: Without this a dividend-paying call cannot be priced properly at all: the
    #: only moments its early exercise is ever optimal are missing from the set.
    align_to_dividends: bool = True
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


def exercise_indices(
    n_steps: int, n_dates: int | None, required: tuple[int, ...] = ()
) -> list[int]:
    """Time-grid indices at which exercise is allowed, ascending, ending at ``n_steps``.

    Expiry is always in the set, and index 0 never is: exercising at inception is
    a decision the holder makes before buying, not one the model prices.
    ``required`` adds dates that must be present whatever ``n_dates`` says --
    the instants before ex-dividend dates, in practice.
    """
    if n_dates is not None and n_dates < 0:
        raise ValidationError(f"n_exercise_dates cannot be negative, got {n_dates}")

    dates = {n_steps} | {i for i in required if 1 <= i <= n_steps}
    if n_dates is None or n_dates >= n_steps:
        dates |= set(range(1, n_steps + 1))
    elif n_dates > 0:
        step = n_steps / n_dates
        dates |= {max(1, min(n_steps, round((i + 1) * step))) for i in range(n_dates)}
    return sorted(dates)


def pre_dividend_indices(times, n_steps: int, horizon: float) -> tuple[int, ...]:
    """The last grid index strictly before each ex-date within ``horizon``.

    Strictly before, because the whole value of exercising is capturing a
    dividend the holder would otherwise not receive; a step at or after the
    ex-date has already lost it.
    """
    dt = horizon / n_steps
    return tuple(sorted({
        max(1, min(n_steps, int(time / dt))) for time in times
        if 0.0 < time <= horizon
    }))


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
    offset: int = 0,
) -> Tensor:
    """Discounted cashflow per path under a fixed exercise policy.

    ``offset`` is the global time index of column zero of ``asset_path``, so a
    sub-path starting partway through the grid can be valued under the same
    policy and the same global discount factors. ``indices`` and ``discounts``
    stay in global terms throughout; only the column lookup shifts.

    Differentiable in the path and in the discount curve; the decision itself is
    a detached mask, so delta comes off this by autograd exactly as it does for a
    European payoff.
    """
    cash = discounts[indices[-1]] * intrinsic(
        asset_path[:, indices[-1] - offset], None
    )
    for k in reversed(indices[:-1]):
        beta = coefficients.get(k)
        if beta is None:
            continue
        spot_k = asset_path[:, k - offset]
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
