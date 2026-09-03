"""Torch-native discount curves."""

from __future__ import annotations

import math

import pytest
import torch

from torch_pricer.calibration.curve_model.curves import RateCurve
from torch_pricer.errors import ValidationError


def test_flat_curve_reproduces_exp_minus_rt():
    curve = RateCurve.flat(0.03)
    for t in (0.25, 1.0, 7.5):
        assert float(curve.discount(t).detach()) == pytest.approx(math.exp(-0.03 * t), rel=1e-12)


def test_discount_at_zero_is_one():
    curve = RateCurve.from_zeros([0.5, 1.0, 5.0], [0.02, 0.025, 0.03])
    assert float(curve.discount(0.0)) == pytest.approx(1.0, abs=1e-15)


def test_zero_rate_round_trips_through_discount():
    curve = RateCurve.from_zeros([0.5, 1.0, 5.0], [0.02, 0.025, 0.03])
    t = torch.tensor([0.1, 0.5, 0.9, 1.0, 3.0, 5.0, 9.0])
    implied = -torch.log(curve.discount(t)) / t
    assert torch.allclose(implied, curve.zero_rate(t), atol=1e-12)


def test_pillars_are_hit_exactly():
    times, zeros = [0.5, 1.0, 5.0], [0.02, 0.025, 0.03]
    curve = RateCurve.from_zeros(times, zeros)
    assert torch.allclose(curve.zero_rate(torch.tensor(times)), torch.tensor(zeros), atol=1e-12)


def test_zero_rate_is_flat_outside_the_pillars():
    curve = RateCurve.from_zeros([1.0, 5.0], [0.02, 0.03])
    assert float(curve.zero_rate(0.1)) == pytest.approx(0.02, rel=1e-12)
    assert float(curve.zero_rate(50.0)) == pytest.approx(0.03, rel=1e-12)


def test_forward_rate_is_consistent_with_discounts():
    curve = RateCurve.from_zeros([1.0, 5.0], [0.02, 0.03])
    fwd = float(curve.forward_rate(1.0, 2.0))
    want = math.log(float(curve.discount(1.0)) / float(curve.discount(2.0))) / 1.0
    assert fwd == pytest.approx(want, rel=1e-12)


def test_instantaneous_forward_of_a_flat_curve_is_the_rate():
    curve = RateCurve.flat(0.035)
    assert float(curve.instantaneous_forward(2.0)) == pytest.approx(0.035, rel=1e-6)


def test_pillar_zeros_carry_gradient():
    """Bucketed rho is a backward pass, not a bump loop -- that is why pillars are a Parameter."""
    curve = RateCurve.from_zeros([1.0, 5.0], [0.02, 0.03])
    value = curve.discount(3.0)
    (grad,) = torch.autograd.grad(value, [curve.pillar_zeros])
    assert grad.shape == curve.pillar_zeros.shape
    assert torch.any(grad != 0)


@pytest.mark.parametrize(
    "times,zeros,message",
    [
        ([1.0, 2.0], [0.01], "pillar times but"),
        ([2.0, 1.0], [0.01, 0.02], "strictly increasing"),
        ([-1.0], [0.01], "non-negative"),
        ([], [], "at least one pillar"),
    ],
)
def test_malformed_curves_are_rejected(times, zeros, message):
    with pytest.raises(ValidationError, match=message):
        RateCurve.from_zeros(times, zeros)
