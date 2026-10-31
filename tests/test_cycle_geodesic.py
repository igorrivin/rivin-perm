"""Independent finite-order checks of the analytic cycle permanent."""

from fractions import Fraction
from itertools import permutations
import math

import numpy as np
import pytest

from cycle_geodesic import (
    cycle_geodesic_matrix,
    cycle_midpoint_exact,
    cycle_permanent,
    cycle_permanent_logabs,
    cycle_scaling_limit,
    cyclic_shift,
)


def brute_force_permanent(matrix):
    n = len(matrix)
    return sum(np.prod(matrix[np.arange(n), p]) for p in permutations(range(n)))


def independent_spectral_matrix(n, t, convention, pi_branch):
    """Keep Fourier modes in their original order; wrap angles individually."""
    k = np.arange(n)
    angles = 2 * np.pi * k / n
    if convention == "principal":
        cutoff = n // 2 if pi_branch == "positive" else (n - 1) // 2
        angles[k > cutoff] -= 2 * np.pi
    fourier = np.exp(2j * np.pi * np.outer(k, k) / n) / np.sqrt(n)
    return (fourier * np.exp(1j * t * angles)) @ fourier.conj().T


@pytest.mark.parametrize("n", range(1, 8))
@pytest.mark.parametrize("t", [0.0, 0.17, 0.31, 0.5, 0.77, 1.0])
@pytest.mark.parametrize("convention,pi_branch", [
    ("literal", "positive"),
    ("principal", "positive"),
    ("principal", "negative"),
])
def test_against_independent_permutation_sum(n, t, convention, pi_branch):
    matrix = independent_spectral_matrix(n, t, convention, pi_branch)
    expected = brute_force_permanent(matrix)
    actual = cycle_permanent(n, t, convention=convention, pi_branch=pi_branch)
    assert actual == pytest.approx(expected, abs=3e-13, rel=3e-12)
    np.testing.assert_allclose(
        cycle_geodesic_matrix(n, t, convention=convention, pi_branch=pi_branch),
        matrix, atol=4e-15,
    )


@pytest.mark.parametrize("n", [1, 2, 5, 6, 11, 12])
def test_endpoints_have_the_documented_orientation(n):
    np.testing.assert_array_equal(cycle_geodesic_matrix(n, 0), np.eye(n))
    expected = np.zeros((n, n), dtype=complex)
    for row in range(n):
        expected[row, (row + 1) % n] = 1
    np.testing.assert_array_equal(cycle_geodesic_matrix(n, 1), expected)
    np.testing.assert_array_equal(cyclic_shift(n), expected)
    assert cycle_permanent(n, 0) == cycle_permanent(n, 1) == 1


@pytest.mark.parametrize("n,expected", [
    (1, Fraction(1)), (2, Fraction(0)), (3, Fraction(-1, 9)),
    (4, Fraction(0)), (5, Fraction(9, 625)), (7, Fraction(-225, 117649)),
])
def test_exact_midpoints_and_branch_sign(n, expected):
    assert cycle_midpoint_exact(n, convention="literal") == expected
    assert cycle_midpoint_exact(n) == abs(expected)
    assert cycle_midpoint_exact(n, convention="principal") == abs(expected)
    assert cycle_midpoint_exact(n, convention="principal", pi_branch="negative") == abs(expected)
    assert cycle_permanent(n, 0.5, convention="literal") == pytest.approx(float(expected))
    assert cycle_permanent(n, 0.5).imag == 0
    if n % 2 == 0:
        assert cycle_permanent(n, 0.5) == 0j
        assert cycle_permanent_logabs(n, 0.5) == -math.inf


@pytest.mark.parametrize("n", [7, 8, 13, 14])
@pytest.mark.parametrize("t", [0.17, 0.31, 0.67, 0.91])
def test_phase_relation_for_both_parities(n, t):
    literal = cycle_permanent(n, t, convention="literal")
    for branch, shift in [("positive", (n - 1) // 2), ("negative", n // 2)]:
        principal = cycle_permanent(n, t, pi_branch=branch)
        expected = literal * np.exp(-2j * np.pi * shift * t)
        assert principal == pytest.approx(expected, abs=1e-14)
        assert math.exp(cycle_permanent_logabs(n, t)) == pytest.approx(abs(principal))
    if n % 2:
        assert cycle_permanent(n, t).real > 0
        assert cycle_permanent(n, t).imag == 0
    else:
        assert cycle_permanent(n, t).conjugate() == pytest.approx(
            cycle_permanent(n, t, pi_branch="negative"), abs=2e-15)


def test_logarithms_survive_underflow_and_resolve_a_nearby_even_zero():
    assert cycle_permanent(1001, 0.5) == 0j  # Floating-point underflow.
    assert math.isfinite(cycle_permanent_logabs(1001, 0.5))
    assert cycle_midpoint_exact(1001) != 0
    t = np.nextafter(0.5, 0.0)
    # For n=2 the magnitude is exactly cos(pi*t), evaluated near its zero.
    expected = math.log(math.sin(math.pi * (0.5 - t)))
    assert cycle_permanent_logabs(2, t) == pytest.approx(expected, abs=1e-14)
    assert cycle_permanent(2, t) != 0j


def test_scaling_limit_and_midpoint_parity():
    assert cycle_scaling_limit(0) == cycle_scaling_limit(1) == 0
    assert cycle_scaling_limit(0.5) == 1
    t = 1e-9
    assert cycle_scaling_limit(t) == pytest.approx(math.pi**2 * t**2 / 3, rel=1e-14)
    for t in [0.07, 0.23, 0.49, 0.5, 0.81]:
        observed = -cycle_permanent_logabs(2001, t) / 2001
        assert observed == pytest.approx(cycle_scaling_limit(t), abs=0.0004)
    assert cycle_permanent_logabs(2000, 0.5) == -math.inf


@pytest.mark.parametrize("n", [0, -1, 2.5, True])
def test_invalid_order(n):
    with pytest.raises(ValueError, match="positive integer"):
        cycle_permanent(n, 0.2)


@pytest.mark.parametrize("t", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_parameter(t):
    with pytest.raises(ValueError, match="finite"):
        cycle_permanent(3, t)


def test_invalid_conventions():
    with pytest.raises(ValueError, match="convention"):
        cycle_permanent(3, 0.2, convention="unknown")
    with pytest.raises(ValueError, match="pi_branch"):
        cycle_permanent(3, 0.2, pi_branch="unknown")


def test_cycle_callers_do_not_invoke_exponential_evaluators(monkeypatch):
    pytest.importorskip("numba")
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    geodesic_perm = pytest.importorskip("geodesic_perm")
    geodesic_complex_plot = pytest.importorskip("geodesic_complex_plot")

    def forbidden(*args, **kwargs):
        pytest.fail("Cycle path called a generic permanent evaluator")

    monkeypatch.setattr(geodesic_perm, "permanent_fast", forbidden)
    monkeypatch.setattr(geodesic_complex_plot, "permanent_batch_gpu", forbidden)
    data = geodesic_perm.geodesic_to_target(8, cyclic_shift(8), num_points=5)
    assert data["abs_perm"][2] == 0
    assert data["method"] == "analytic Cauchy product"
    t, values = geodesic_complex_plot.geodesic_complex(7, cyclic_shift(7), num_points=5)
    assert t[2] == 0.5
    assert values[2] == pytest.approx(float(cycle_midpoint_exact(7, convention="principal")))


def test_noncycle_target_preserves_the_gpu_evaluation_path(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    geodesic_complex_plot = pytest.importorskip("geodesic_complex_plot")
    calls = []

    def independent_batch(matrices):
        calls.append(matrices.shape)
        return np.array([brute_force_permanent(matrix) for matrix in matrices])

    monkeypatch.setattr(geodesic_complex_plot, "permanent_batch_gpu", independent_batch)
    target = geodesic_complex_plot.dft_unitary(3)
    times, values = geodesic_complex_plot.geodesic_complex(3, target, num_points=3)
    assert calls == [(3, 3, 3)]
    np.testing.assert_array_equal(times, [0, 0.5, 1])
    assert values[0] == pytest.approx(1)
    assert values[-1] == pytest.approx(brute_force_permanent(target))
