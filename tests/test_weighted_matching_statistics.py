"""Definition-level CPU checks for weighted occupancy and approximate models."""

from fractions import Fraction
import importlib.util
import itertools
import math
from pathlib import Path
import random

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "notes/benchmarks/weighted_matching_statistics.py"
if not SOURCE.is_file():
    pytest.skip("weighted matching study is available only in a source checkout", allow_module_level=True)
SPEC = importlib.util.spec_from_file_location("weighted_matching_statistics", SOURCE)
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def exhaustive_matchings(matrix, activity=1):
    """Enumerate subsets of edges, independently of any rook/BP recurrence."""
    nrows = len(matrix)
    ncols = len(matrix[0]) if matrix else 0
    edges = [(i, j, Fraction(w)) for i, row in enumerate(matrix)
             for j, w in enumerate(row) if w]
    coefficients = [Fraction(0)] * (min(nrows, ncols) + 1)
    edge_sums = [[Fraction(0)] * ncols for _ in range(nrows)]
    partition = Fraction(0)
    for bits in itertools.product((0, 1), repeat=len(edges)):
        chosen = [edge for bit, edge in zip(bits, edges) if bit]
        size = len(chosen)
        if len({i for i, _, _ in chosen}) != size or len({j for _, j, _ in chosen}) != size:
            continue
        weight = math.prod((w for _, _, w in chosen), start=Fraction(1))
        coefficients[size] += weight
        mass = weight * activity ** size
        partition += mass
        for i, j, _ in chosen:
            edge_sums[i][j] += mass
    marginals = [[x / partition for x in row] for row in edge_sums]
    return coefficients, partition, marginals


def test_exact_statistics_match_literal_weighted_matching_enumeration():
    rng = random.Random(260907)
    for _ in range(12):
        matrix = [[Fraction(rng.randrange(4), rng.randrange(1, 5))
                   for _ in range(3)] for _ in range(3)]
        for activity in (Fraction(0), Fraction(1, 3), Fraction(7)):
            coefficients, partition, marginals = exhaustive_matchings(matrix, activity)
            result = stats.exact_occupancy_statistics(coefficients, activity)
            probabilities = result["probabilities"]
            assert result["partition"] == partition
            assert sum(probabilities) == 1
            assert result["mean"] == sum(map(sum, marginals))
            second_moment = sum(k * k * p for k, p in enumerate(probabilities))
            assert result["variance"] == second_moment - result["mean"] ** 2
            assert result["empty_probability"] == 1 / partition
            assert result["max_size_probability"] == coefficients[-1] * activity ** 3 / partition
            assert all(isinstance(p, Fraction) for p in probabilities)


def test_empty_zero_activity_and_trailing_zero_are_unambiguous():
    empty = stats.exact_occupancy_statistics([1], 123)
    assert empty["partition"] == empty["empty_probability"] == empty["max_size_probability"] == 1
    assert empty["mean"] == empty["variance"] == empty["log_partition"] == 0
    zero = stats.exact_occupancy_statistics([1, 9, 20, 0], 0)
    assert zero["probabilities"] == (1, 0, 0, 0)
    assert zero["max_size"] == 3
    assert zero["max_size_probability"] == 0


@pytest.mark.parametrize("coefficients", [[], [0, 1], [2, 1], [1, -1], [True, 1],
                                          [1, 0.5], [1, "2"], None, "12"])
def test_exact_rejects_invalid_coefficient_input(coefficients):
    with pytest.raises((ValueError, TypeError)):
        stats.exact_occupancy_statistics(coefficients, 1)


@pytest.mark.parametrize("activity", [-1, Fraction(-1, 3), True, 0.0, 0.1, float("nan"),
                                     float("inf"), "1/2", None])
def test_all_models_reject_inexact_or_invalid_activity(activity):
    with pytest.raises((ValueError, TypeError)):
        stats.exact_occupancy_statistics([1, 2], activity)
    for function in (stats.independent_edge_statistics, stats.belief_propagation_statistics):
        with pytest.raises((ValueError, TypeError)):
            function([[1, 2]], activity)


@pytest.mark.parametrize("matrix", [[[1], [2, 3]], [[-1]], [[True]], [[1.0]],
                                   [["1"]], [1, 2], "12", None])
def test_matrix_models_reject_invalid_weights_or_shape(matrix):
    for function in (stats.independent_edge_statistics, stats.belief_propagation_statistics):
        with pytest.raises((ValueError, TypeError)):
            function(matrix, 1)


def test_independent_edges_count_forbidden_simultaneous_edges():
    # Two incident unit edges: matching Z=3, unconstrained Z=4.
    exact = stats.exact_occupancy_statistics([1, 2], 1)
    approx = stats.independent_edge_statistics([[1, 1]], 1)
    assert approx["probabilities"] == (0.25, 0.5, 0.25)
    assert approx["mean"] == 1
    assert approx["variance"] == 0.5
    assert approx["invalid_size_probability"] == 0.25
    assert approx["log_partition"] == pytest.approx(math.log(4))
    assert stats.total_variation_distance(exact["probabilities"], approx["probabilities"]) == pytest.approx(0.25)


def test_invalid_size_mass_is_not_total_invalid_matching_mass():
    # Same star, but padded with an isolated row: now the conflicting 2-edge
    # event is within min(rows,cols), although it still is not a matching.
    approx = stats.independent_edge_statistics([[1, 1], [0, 0]], 1)
    assert approx["invalid_size_probability"] == 0
    exact = stats.exact_occupancy_statistics([1, 2, 0], 1)
    assert stats.total_variation_distance(exact["probabilities"], approx["probabilities"]) == pytest.approx(0.25)


def test_independent_convolution_matches_exhaustive_bernoulli_outcomes():
    weights = [Fraction(1, 3), Fraction(2, 5), Fraction(7)]
    activity = Fraction(3, 2)
    probabilities = [activity * w / (1 + activity * w) for w in weights]
    expected = [Fraction(0)] * 4
    for bits in itertools.product((0, 1), repeat=3):
        expected[sum(bits)] += math.prod((p if b else 1 - p
                                        for p, b in zip(probabilities, bits)), start=Fraction(1))
    actual = stats.independent_edge_statistics([weights], activity)
    assert actual["probabilities"] == pytest.approx(list(map(float, expected)))
    assert actual["mean"] == pytest.approx(float(sum(probabilities)))
    assert actual["variance"] == pytest.approx(float(sum(p * (1 - p) for p in probabilities)))


@pytest.mark.parametrize("matrix", [[], [[], []], [[0, 0]], [[1, 0], [0, 2]],
                                   [[Fraction(1, 3), 2, 3, 5]],
                                   [[Fraction(1, 2), 2, 0], [0, 3, 4], [0, 0, Fraction(5, 3)]]])
@pytest.mark.parametrize("activity", [0, Fraction(1, 7), 1, 20])
def test_belief_propagation_is_exact_on_forests(matrix, activity):
    coefficients, partition, marginals = exhaustive_matchings(matrix, activity)
    exact = stats.exact_occupancy_statistics(coefficients, activity)
    bp = stats.belief_propagation_statistics(matrix, activity, tolerance=1e-12)
    assert bp["status"] == "CONVERGED"
    assert bp["converged"] and bp["residual"] <= 1e-12
    assert bp["log_partition"] == pytest.approx(math.log(partition), abs=1e-10)
    assert bp["mean"] == pytest.approx(float(exact["mean"]), abs=1e-10)
    for row, reference in zip(bp["edge_probabilities"], marginals):
        assert row == pytest.approx(list(map(float, reference)), abs=1e-10)
    # Vertex occupancy conservation uses all incident edge marginals.
    row_occupancy = [sum(row) for row in bp["edge_probabilities"]]
    ncols = len(matrix[0]) if matrix else 0
    col_occupancy = [sum(row[j] for row in bp["edge_probabilities"]) for j in range(ncols)]
    for occupancy, monomer in zip(row_occupancy + col_occupancy, bp["vertex_monomer_probabilities"]):
        assert occupancy + monomer == pytest.approx(1, abs=1e-10)


def test_cyclic_bethe_error_is_present_despite_convergence():
    # K_2,2 is a 4-cycle: Z=1+4z+2z^2=7. Uniform cavity m solves m=1/(1+m).
    bp = stats.belief_propagation_statistics([[1, 1], [1, 1]], 1, tolerance=1e-13)
    m = (math.sqrt(5) - 1) / 2
    expected_log = 4 * math.log1p(2 * m) - 4 * math.log1p(m * m)
    assert bp["converged"]
    assert bp["log_partition"] == pytest.approx(expected_log, abs=1e-12)
    assert abs(bp["log_partition"] - math.log(7)) > 0.01
    assert bp["mean"] == pytest.approx(4 * m * m / (1 + m * m), abs=1e-12)
    assert abs(bp["mean"] - 8 / 7) > 0.01


@pytest.mark.parametrize("damping,max_iterations", [(0.5, 0), (0.5, 1), (1e-20, 3)])
def test_nonconvergence_never_returns_success_or_estimates(damping, max_iterations):
    bp = stats.belief_propagation_statistics([[1, 1], [1, 1]], 1,
                                            damping=damping, max_iterations=max_iterations)
    assert bp["status"] == "NOT_CONVERGED" and not bp["converged"]
    assert bp["iterations"] == max_iterations
    assert bp["residual"] > bp["tolerance"]
    assert bp["log_partition"] is bp["mean"] is bp["edge_probabilities"] is None


def test_zero_activity_is_already_a_fixed_point_with_no_iterations():
    bp = stats.belief_propagation_statistics([[9, 8], [7, 6]], 0, max_iterations=0)
    assert bp["converged"] and bp["iterations"] == bp["residual"] == 0
    assert bp["log_partition"] == bp["mean"] == 0
    assert bp["edge_probabilities"] == [[0, 0], [0, 0]]


@pytest.mark.parametrize("kwargs", [{"damping": 0}, {"damping": -1}, {"damping": 1.1},
                                    {"damping": True}, {"tolerance": 0}, {"tolerance": float("nan")},
                                    {"max_iterations": -1}, {"max_iterations": 2.0},
                                    {"max_iterations": True}])
def test_bp_rejects_invalid_iteration_controls(kwargs):
    with pytest.raises((ValueError, TypeError)):
        stats.belief_propagation_statistics([[1]], 1, **kwargs)


def test_exact_huge_partition_and_numeric_logs_do_not_overflow():
    huge = 10 ** 400
    exact = stats.exact_occupancy_statistics([1, huge, huge * huge], Fraction(2, 3))
    assert exact["partition"] == 1 + Fraction(2 * huge, 3) + Fraction(4 * huge * huge, 9)
    assert math.isfinite(exact["log_partition"])
    assert stats.log_rational(Fraction(10 ** 100 + 1, 10 ** 100)) == pytest.approx(1e-100)
    approx = stats.independent_edge_statistics([[huge, Fraction(1, huge)]], 1)
    assert math.isfinite(approx["log_partition"])
    assert math.fsum(approx["probabilities"]) == 1
    assert approx["mean"] == 1
    bp = stats.belief_propagation_statistics([[huge]], 1)
    assert bp["converged"]
    assert bp["log_partition"] == pytest.approx(math.log(huge))
    assert bp["mean"] == 1


def test_log_updates_keep_small_other_edges_when_excluding_a_huge_edge():
    huge = 10 ** 40
    matrix = [[huge, 2], [0, 3]]  # A weighted path, so BP is exact.
    coefficients, partition, _ = exhaustive_matchings(matrix)
    bp = stats.belief_propagation_statistics(matrix, 1, tolerance=1e-12)
    assert bp["converged"]
    assert bp["log_partition"] == pytest.approx(math.log(partition), abs=1e-10)
    assert bp["edge_probabilities"][1][1] == pytest.approx(0.75, abs=1e-10)


@pytest.mark.parametrize("n", [12, 18])
def test_dense_high_activity_distinguishes_mean_from_partition_accuracy(n):
    # Closed-form rook coefficients provide an independent nonexponential
    # reference at the upper scale/activity of the planned application study.
    coefficients = [math.comb(n, k) ** 2 * math.factorial(k) * 9 ** k for k in range(n + 1)]
    exact = stats.exact_occupancy_statistics(coefficients, 1000)
    bp = stats.belief_propagation_statistics([[9] * n for _ in range(n)], 1000)
    assert bp["converged"] and bp["iterations"] < 100
    assert abs(bp["mean"] - float(exact["mean"])) < 0.1
    assert bp["log_partition"] - exact["log_partition"] < -1


def test_bp_mean_matches_log_activity_derivative_of_bethe_partition():
    matrix = [[1, 2, 0], [3, 4, 5], [0, 6, 7]]
    factor = Fraction(10001, 10000)
    center = stats.belief_propagation_statistics(matrix, 2, tolerance=1e-13)
    plus = stats.belief_propagation_statistics(matrix, 2 * factor, tolerance=1e-13)
    minus = stats.belief_propagation_statistics(matrix, 2 / factor, tolerance=1e-13)
    assert all(r["converged"] for r in (center, plus, minus))
    derivative = (plus["log_partition"] - minus["log_partition"]) / (2 * math.log(float(factor)))
    assert derivative == pytest.approx(center["mean"], abs=1e-7)


@pytest.mark.parametrize("first,second", [([1], [1, 0, 0]), ([0, 1], [1])])
def test_tv_padding(first, second):
    assert stats.total_variation_distance(first, second) == (0 if first == [1] else 1)


@pytest.mark.parametrize("bad", [[], [0.5], [-1, 2], [float("nan")], [float("inf")],
                               [True], ["1"], "1"])
def test_tv_does_not_hide_invalid_probability_input(bad):
    with pytest.raises((ValueError, TypeError)):
        stats.total_variation_distance([1], bad)
