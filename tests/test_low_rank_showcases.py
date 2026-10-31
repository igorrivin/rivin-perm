"""Definition-level checks for the public source-tool application examples."""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations, product
import json
from math import factorial, prod
from pathlib import Path
import subprocess
import sys

import pytest

import showcase_low_rank as showcase
from perm import matrix_permanent


def definition(matrix):
    return sum(prod(matrix[i][j] for i, j in enumerate(sigma))
               for sigma in permutations(range(len(matrix))))


@pytest.mark.parametrize("n", [3, 4, 6])
def test_exact_families_against_permutation_definition(n):
    matrices = [showcase.polynomial_kernel(n), showcase.gram_matrix(showcase.gaussian_factors(n)),
                showcase.profile_case(n)[0]]
    for matrix in matrices:
        assert matrix_permanent(matrix, parallel=False, method="low_rank") == definition(matrix)
    polynomial, gaussian, profile = matrices
    assert len(set(map(tuple, polynomial))) == len(set(zip(*polynomial))) == n
    assert len(set(map(tuple, gaussian))) == len(set(zip(*gaussian))) == n
    assert len(set(zip(*profile))) == 3


def test_gaussian_moment_by_explicit_latent_assignments():
    # Expand prod_i(sum_a F_ia g_a) literally, then integrate each monomial.
    # Independent proper complex Gaussian powers have E g^a conjugate(g)^b=0
    # unless a=b, in which case their moment is a!.
    factors = showcase.gaussian_factors(5)
    coefficients = defaultdict(int)
    for labels in product(range(3), repeat=len(factors)):
        powers = tuple(labels.count(a) for a in range(3))
        coefficients[powers] += prod(factors[i][a] for i, a in enumerate(labels))
    integrated = sum(coefficient ** 2 * prod(factorial(a) for a in powers)
                     for powers, coefficient in coefficients.items())
    gram = showcase.gram_matrix(factors)
    assert integrated == definition(gram)
    assert integrated == matrix_permanent(gram, method="low_rank", parallel=False)


def test_profile_probabilities_by_enumerating_every_sample_sequence():
    weights, sample_size = [1, 2, 3], 4
    probabilities = defaultdict(Fraction)
    for observations in product(range(3), repeat=sample_size):
        counts = Counter(observations)
        profile = tuple(sorted(counts.get(i, 0) for i in range(3)))
        probabilities[profile] += Fraction(prod(weights[i] for i in observations), sum(weights) ** sample_size)
    assert sum(probabilities.values()) == 1
    for counts, expected in probabilities.items():
        matrix = [[weight ** count for count in counts] for weight in weights]
        value = definition(matrix)
        assert showcase.profile_probability(value, weights, counts) == expected
        assert showcase.grouped_columns_permanent(matrix) == value
    assert showcase.profile_probability(factorial(3), weights, [0, 0, 0]) == 1


def test_grouped_columns_signed_and_empty_oracles():
    for matrix in ([], [[1]], [[1, -2, 1], [3, -4, 3], [-1, 2, -1]],
                   [[1, 0, 1], [0, 1, 0], [1, 1, 1]]):
        assert showcase.grouped_columns_permanent(matrix) == definition(matrix)
    with pytest.raises(ValueError, match="budget"):
        showcase.grouped_columns_permanent([[i + j for j in range(6)] for i in range(6)], max_states=1)


def test_runner_compares_every_exact_result_and_reports_interpretations():
    report = showcase.run_showcases(4, repeats=1, compare_generic=True)
    assert [case["name"] for case in report["cases"]] == [
        "polynomial_kernel", "gaussian_moment", "profile_likelihood"]
    for case in report["cases"]:
        assert case["permanent"] == definition(case["matrix"])
        assert case["generic_reference"] == "computed"
        assert all(len(timing["samples_seconds"]) == 1 for timing in case["measurements"].values())
    probability = report["cases"][-1]["interpretation"]["probability"]
    assert 0 <= probability <= 1


def test_optional_research_example_contains_exact_reference():
    if not showcase.APPROXIMATION_SOURCE.is_file():
        pytest.skip("full-rank certificate is checkout-only")
    report = showcase.run_showcases(4, repeats=1, compare_generic=True, include_approximation=True)
    case = report["cases"][-1]
    exact = definition(case["matrix"])
    assert case["rank"] == 4 and case["baseline_rank"] == 2
    assert case["exact_permanent"] == exact
    assert case["certificate"]["lower_bound"] <= exact <= case["certificate"]["upper_bound"]
    assert case["certificate"]["meets_tolerance"]
    strict = showcase.run_showcases(4, repeats=1, include_approximation=True, relative_tolerance=0)
    assert not strict["cases"][-1]["certificate"]["meets_tolerance"]


def test_large_example_uses_public_auto_dispatch_with_no_generic_reference():
    report = showcase.run_showcases(20, repeats=1)
    for case in report["cases"]:
        assert case["permanent"] > 0
        assert case["generic_reference"] == "not run"
        assert case["homogeneous_states_per_side"] == 231


def test_preflight_rejects_large_generic_and_missing_research_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before calculating a permanent")
    monkeypatch.setattr(showcase.perm, "matrix_permanent", forbidden)
    for kwargs in (dict(n=40), dict(n=40, allow_large=True, compare_generic=True),
                   dict(n=3, repeats=0), dict(n=3, relative_tolerance=-1)):
        with pytest.raises(ValueError):
            showcase.run_showcases(**kwargs)
    monkeypatch.setattr(showcase, "APPROXIMATION_SOURCE", Path("/nonexistent/near_low_rank.py"))
    with pytest.raises(ValueError, match="checkout"):
        showcase.run_showcases(3, include_approximation=True)
    monkeypatch.setattr(showcase.perm, "HAS_C_EXT", False)
    with pytest.raises(ValueError, match="native"):
        showcase.run_showcases(20, compare_generic=True)


def test_cli_outputs_exact_values_and_protects_existing_capture(tmp_path):
    script = Path(showcase.__file__)
    destination = tmp_path / "showcase.json"
    command = [sys.executable, str(script), "--order", "3", "--repeats", "1", "--output", str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(destination.read_text())
    assert len(report["cases"]) == 3
    assert "normalized_intensity_moment" in result.stdout
    assert "probability:" in result.stdout
    original = destination.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "output exists" in result.stderr
    assert destination.read_bytes() == original


@pytest.mark.parametrize("n", [0, 2, True, 3.5, 61])
def test_case_size_validation(n):
    for function in (showcase.polynomial_kernel, showcase.gaussian_factors, showcase.profile_case):
        with pytest.raises(ValueError):
            function(n)
