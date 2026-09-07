"""The Andersen-Broadie dual: an upper bound, so the price comes with a bracket.

Longstaff-Schwartz only ever produces a *lower* bound. Its exercise policy comes
from a finite basis and is therefore suboptimal, and a suboptimal stopping rule
earns less than the optimal one by definition -- so, once foresight bias is
removed by fitting the policy out of sample, the number is honestly below the
truth by an unknown amount.

Under Black-Scholes that hardly matters, because the lattice and the PDE solver
say what the truth is. The point of a Monte Carlo American engine is the models
they cannot follow: local vol on a full surface, Heston, anything multi-asset.
There the lower bound arrives with no companion, and no way to tell whether it
falls short by 0.005 or by 0.05.

**The duality.** Haugh-Kogan and Rogers showed the problem has a dual,

    V = inf over martingales M of  E[ max over k of ( h_k - M_k ) ]

whose defining property is that *any* martingale ``M`` gives an upper bound. A
poor choice gives a loose one; the Doob martingale of the true value process
attains the infimum exactly. Andersen and Broadie (2004) supply the practical
recipe: build ``M`` from the policy already in hand, estimating its value
process by nested simulation.

**Why this is affordable at all.** Naively every date needs two inner
simulations -- one for the policy value at ``k`` and one for its conditional
expectation given ``k-1``. It does not, because the two coincide one step apart.
Where the policy *continues* at ``k``, the value of following it from ``k`` is by
definition the expectation of its value at the next date given the state now,
which is exactly the quantity the next date already estimates. Where the policy
*exercises*, the value is the intrinsic and costs nothing. So one inner
simulation per exercise date suffices, and the cost is linear in the number of
dates rather than quadratic.

It is still nested Monte Carlo, and still one to two orders of magnitude dearer
than the lower bound. The gap it returns is the price of that: a direct
measurement of how much value the basis is leaving on the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torch_pricer.errors import ValidationError
from torch_pricer.pricer.monte_carlo import lsm as lsm_mod
from torch_pricer.simulator.monte_carlo.rng import NormalDraws
from torch_pricer.simulator.monte_carlo.simulator import EulerMaruyamaSimulator


@dataclass(frozen=True)
class DualConfig:
    """Settings for the nested upper-bound simulation."""

    #: Outer paths. The bound is an average over these, so its standard error
    #: falls as their square root.
    n_outer: int = 2_000
    #: Inner paths per outer path per date. These estimate a conditional
    #: expectation, and too few biases the bound *upward*: noise in the
    #: martingale survives the maximum taken inside the expectation, so it cannot
    #: average out. The direction is safe -- the bracket stays valid -- but the
    #: gap then overstates how suboptimal the policy really is.
    #:
    #: Measured on the 1y ATM American put, as the upper bound's excess over the
    #: true 6.09046: +0.683, +0.352, +0.158, +0.072, +0.026, +0.004 at 25, 50,
    #: 100, 200, 400 and 800 inner paths. Every doubling roughly halves it, so
    #: the bias is O(1/n_inner) and 500 leaves about 0.02 -- small against a
    #: policy gap worth acting on, and the cost is linear in this number.
    n_inner: int = 500
    #: Seed for the inner draws. Distinct from the pricing seed.
    seed: int = 5_150_281


@dataclass(frozen=True)
class Bracket:
    """A two-sided estimate of an American price."""

    low: float
    high: float
    low_stderr: float
    high_stderr: float

    @property
    def gap(self) -> float:
        """Width of the bracket -- the cost of an imperfect exercise policy."""
        return self.high - self.low

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Bracket([{self.low:.5f}, {self.high:.5f}], gap={self.gap:.5f}, "
            f"se=({self.low_stderr:.5f}, {self.high_stderr:.5f}))"
        )


def _inner_policy_value(
    sde,
    start_states: Tensor,
    t_grid: Tensor,
    start_index: int,
    dates_ahead: list[int],
    discounts: Tensor,
    intrinsic,
    strike: float,
    lsm_config,
    coefficients: dict[int, Tensor],
    add_back,
    n_inner: int,
    seed: int,
    n_factors: int,
) -> Tensor:
    """``E[ policy value over ``dates_ahead`` | state at ``start_index`` ]``.

    Each outer state is fanned out into ``n_inner`` sub-paths, evolved to expiry
    over the remaining grid, and valued under the *same* fixed policy. The mean
    over each fan is the conditional expectation for that outer path.
    """
    n_outer, dim = start_states.shape
    x0 = start_states.repeat_interleave(n_inner, dim=0)
    sub_grid = t_grid[start_index:]
    n_steps = sub_grid.numel() - 1

    draws = NormalDraws(
        n_paths=n_outer * n_inner, n_factors=n_factors, seed=seed,
        antithetic=False, device=start_states.device, dtype=start_states.dtype,
    ).draw(n_steps)

    states = EulerMaruyamaSimulator(sde).simulate_with_trajectory(x0, sub_grid, draws)
    levels = sde.asset(states)
    if add_back is not None:
        levels = levels + add_back[start_index:].unsqueeze(0)

    cash = lsm_mod.value(
        levels, discounts, intrinsic, strike, dates_ahead, lsm_config,
        coefficients, offset=start_index,
    )
    return cash.reshape(n_outer, n_inner).mean(dim=1)


def upper_bound(
    sde,
    outer_levels: Tensor,
    outer_states: Tensor,
    t_grid: Tensor,
    indices: list[int],
    discounts: Tensor,
    intrinsic,
    strike: float,
    lsm_config,
    coefficients: dict[int, Tensor],
    dual_config: DualConfig,
    n_factors: int,
    add_back=None,
) -> tuple[float, float]:
    """Dual upper bound and its standard error.

    Args:
        sde: the dynamics, already built from the model and market
        outer_levels: asset levels along the outer paths, ``(n_outer, n_grid)``
        outer_states: raw simulator states along the same paths
        t_grid: the global time grid
        indices: global exercise-date indices, ascending, ending at expiry
        discounts: discount factor at every grid time, indexed globally
        intrinsic: immediate-exercise payoff
        strike: used to normalise the regression basis
        lsm_config: the policy's settings
        coefficients: the fitted policy
        dual_config: nesting sizes and seed
        n_factors: Brownian factors the model needs
        add_back: escrowed dividends to restore, or ``None``

    Returns:
        ``(upper bound, standard error)``.
    """
    if not indices:
        raise ValidationError("no exercise dates to bound")

    with torch.no_grad():
        h = torch.stack(
            [discounts[k] * intrinsic(outer_levels[:, k], None) for k in indices], dim=1
        )

        # C[i] = E[ value of the policy over dates >= indices[i] | state at the
        # previous date ]. One inner simulation per exercise date; see the module
        # docstring for why one is enough.
        starts = [0] + indices[:-1]
        conditional = []
        for i, k in enumerate(indices):
            conditional.append(
                _inner_policy_value(
                    sde, outer_states[:, starts[i]], t_grid, starts[i],
                    indices[i:], discounts, intrinsic, strike, lsm_config,
                    coefficients, add_back, dual_config.n_inner,
                    dual_config.seed + 7919 * i, n_factors,
                )
            )
        conditional = torch.stack(conditional, dim=1)

        # Where the policy exercises, its value is the intrinsic. Where it
        # continues, the value is the next date's conditional expectation -- the
        # telescoping that makes one inner simulation per date sufficient.
        exercises = _exercise_mask(
            outer_levels, indices, discounts, intrinsic, strike,
            lsm_config, coefficients,
        )
        value = torch.empty_like(h)
        value[:, -1] = h[:, -1]
        for i in range(len(indices) - 2, -1, -1):
            value[:, i] = torch.where(exercises[:, i], h[:, i], conditional[:, i + 1])

        martingale = torch.cumsum(value - conditional, dim=1)
        pathwise = (h - martingale).max(dim=1).values

    n = pathwise.numel()
    return float(pathwise.mean()), float(pathwise.std(unbiased=True) / math.sqrt(n))


def _exercise_mask(
    levels: Tensor,
    indices: list[int],
    discounts: Tensor,
    intrinsic,
    strike: float,
    lsm_config,
    coefficients: dict[int, Tensor],
) -> Tensor:
    """Where the fixed policy exercises, per path and date.

    Expiry always exercises when in the money, and a date whose regression was
    never fitted -- too few in-the-money paths to trust one -- always holds.
    """
    mask = torch.zeros(levels.shape[0], len(indices), dtype=torch.bool,
                       device=levels.device)
    for i, k in enumerate(indices):
        exercise = discounts[k] * intrinsic(levels[:, k], None)
        if i == len(indices) - 1:
            mask[:, i] = exercise > 0
            continue
        beta = coefficients.get(k)
        if beta is None:
            continue
        itm = exercise > 0
        if not bool(itm.any()):
            continue
        design = lsm_mod._basis(levels[itm, k] / strike, lsm_config.basis_degree)
        mask[:, i] = torch.zeros_like(itm).masked_scatter(
            itm, exercise[itm] > design @ beta
        )
    return mask
