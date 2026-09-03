"""The stepping machinery and its random driver, independent of any payoff."""

from __future__ import annotations

import pytest
import torch

from torch_pricer.errors import ValidationError
from torch_pricer.simulator.brownian_bridge import BrownianBridge
from torch_pricer.simulator.brownian_motion import BrownianMotion
from torch_pricer.simulator.rng import NormalDraws
from torch_pricer.simulator.simulator import EulerMaruyamaSimulator


def test_draws_are_reproducible_and_shaped():
    a = NormalDraws(1000, n_factors=2, seed=11).draw(5)
    b = NormalDraws(1000, n_factors=2, seed=11).draw(5)
    assert a.shape == (1000, 5, 2)
    assert torch.equal(a, b)


def test_a_different_seed_gives_different_draws():
    a = NormalDraws(1000, seed=1).draw(3)
    b = NormalDraws(1000, seed=2).draw(3)
    assert not torch.equal(a, b)


def test_antithetic_draws_cancel_exactly():
    draws = NormalDraws(2000, seed=5, antithetic=True).draw(4)
    assert float(draws.sum().abs()) < 1e-9


def test_antithetic_needs_an_even_path_count():
    with pytest.raises(ValidationError, match="even n_paths"):
        NormalDraws(999, antithetic=True)


def test_brownian_motion_reproduces_its_known_moments():
    """``X_T ~ N(x0 + mu T, sigma^2 T)``: the one process Euler integrates exactly."""
    mu, sigma, T, n = 0.5, 0.3, 2.0, 200_000
    sde = BrownianMotion(mu, sigma)
    simulator = EulerMaruyamaSimulator(sde)
    grid = torch.linspace(0.0, T, 21)
    draws = NormalDraws(n, seed=3, antithetic=False).draw(20)
    terminal = simulator.simulate(torch.zeros(n, 1), grid, draws, no_grad=True)[:, 0]

    assert float(terminal.mean()) == pytest.approx(mu * T, abs=4 * sigma * T**0.5 / n**0.5)
    assert float(terminal.std()) == pytest.approx(sigma * T**0.5, rel=0.01)


def test_trajectory_and_terminal_agree():
    sde = BrownianMotion(0.1, 0.2)
    simulator = EulerMaruyamaSimulator(sde)
    grid = torch.linspace(0.0, 1.0, 11)
    draws = NormalDraws(64, seed=4).draw(10)
    x0 = torch.zeros(64, 1)
    path = simulator.simulate_with_trajectory(x0, grid, draws, no_grad=True)
    terminal = simulator.simulate(x0, grid, draws, no_grad=True)
    assert path.shape == (64, 11, 1)
    assert torch.allclose(path[:, -1], terminal)


def test_too_few_draws_is_an_error():
    simulator = EulerMaruyamaSimulator(BrownianMotion(0.0, 0.1))
    grid = torch.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match="only 3 draws"):
        simulator.simulate(torch.zeros(8, 1), grid, NormalDraws(8, seed=1).draw(3))


def test_gradients_flow_through_simulation_by_default():
    """The absence of ``torch.no_grad`` here is the whole reason for the library."""
    sde = BrownianMotion(0.0, 0.2)
    simulator = EulerMaruyamaSimulator(sde)
    x0 = torch.zeros(128, 1, requires_grad=True)
    grid = torch.linspace(0.0, 1.0, 6)
    terminal = simulator.simulate(x0, grid, NormalDraws(128, seed=2).draw(5))
    (grad,) = torch.autograd.grad(terminal.sum(), [x0])
    assert torch.allclose(grad, torch.ones_like(grad))


def test_no_grad_flag_detaches():
    simulator = EulerMaruyamaSimulator(BrownianMotion(0.0, 0.2))
    x0 = torch.zeros(16, 1, requires_grad=True)
    grid = torch.linspace(0.0, 1.0, 4)
    out = simulator.simulate(x0, grid, NormalDraws(16, seed=2).draw(3), no_grad=True)
    assert not out.requires_grad


def test_brownian_bridge_raises_rather_than_returning_the_exception():
    bridge = BrownianBridge()
    with pytest.raises(NotImplementedError):
        bridge.drift(torch.zeros(2, 1), torch.tensor(0.0))
