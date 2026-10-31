"""Independent exact checks for weighted-rook planning and support factors."""
from __future__ import annotations

import itertools
import math
import random
import sys

import pytest

import perm_polynomial as polynomial


def _prime(value):
    return value >= 2 and all(value % d for d in range(2, math.isqrt(value) + 1))


def _exhaustive_small_plan(target, order, signed_sum, limit):
    """Enumerate the entire admissible field set, without a pruning rule."""
    widths = [bits for bits in range(1, 32)
              if (signed_sum + 1) * (2**bits - 1) <= limit]
    if not widths:
        raise RuntimeError("no safe width")
    width = max(widths)
    fields = [p for p in range(2**width - 1, 2**(width - 1), -2)
              if p > order and _prime(p)
              and (signed_sum + 1) * p <= limit
              and (signed_sum + 2) * (2**width - p) < 2**width]
    for count in range(len(fields) + 1):
        if math.prod(fields[:count]) > target:
            return tuple(fields[:count])
    raise RuntimeError("no sufficient field set")


@pytest.mark.parametrize("limit,signed_sum,order", [
    (256, 0, 0), (256, 1, 3), (256, 2, 9), (256, 7, 3),
    (256, 14, 0), (256, 15, 0), (256, 63, 3), (256, 255, 0),
    (256, 256, 0), (4096, 0, 9), (4096, 15, 9), (4096, 63, 0),
])
@pytest.mark.parametrize("target", [0, 1, 127, 10**6])
def test_planner_matches_complete_admissible_field_inventory(
    monkeypatch, limit, signed_sum, order, target,
):
    monkeypatch.setattr(polynomial, "_UINT32_LIMIT", limit)
    try:
        expected = _exhaustive_small_plan(target, order, signed_sum, limit)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            polynomial._plan_weighted_rook_pseudo_mersenne_primes(
                target, order=order, maximum_signed_sum=signed_sum,
            )
    else:
        assert polynomial._plan_weighted_rook_pseudo_mersenne_primes(
            target, order=order, maximum_signed_sum=signed_sum,
        ) == expected


@pytest.mark.parametrize("signed_sum,expected", [
    (0, (2147483647, 2147483629, 2147483587)),
    (1, (2147483647, 2147483629, 2147483587)),
    (2, (1073741789, 1073741783, 1073741741)),
    (7, (536870909, 536870879, 536870869)),
    (255, (16777213, 16777199, 16777183)),
    (1023, (4194301, 4194287, 4194277, 4194271)),
])
def test_full_width_successful_plans_preserve_prior_field_choices(signed_sum, expected):
    assert polynomial._plan_weighted_rook_pseudo_mersenne_primes(
        10**20, order=63, maximum_signed_sum=signed_sum,
    ) == expected


@pytest.mark.parametrize("signed_sum", [65533, 65534, 65535])
def test_exhaustion_stops_at_the_strict_fold_boundary(signed_sum):
    # Width is 16. At S=65534 the very first candidate is exactly on
    # (S+2)*complement == 2**16 and must be rejected. S=65533 first tests
    # the composite 65535, then reaches the impossible region; S=65535
    # already exceeds the bound. The old loop scanned 16384 candidates.
    planner = polynomial._plan_weighted_rook_pseudo_mersenne_primes
    line_events = 0

    def guard(frame, event, argument):
        nonlocal line_events
        if frame.f_code is planner.__code__ and event == "line":
            line_events += 1
            assert line_events < 500, "planner walked the known-impossible remainder"
        return guard

    previous_trace = sys.gettrace()
    try:
        sys.settrace(guard)
        with pytest.raises(RuntimeError, match="exhausted"):
            planner(10**20, order=2, maximum_signed_sum=signed_sum)
    finally:
        sys.settrace(previous_trace)


def _matching_oracle(matrix):
    """Directly choose row sets and inject them into distinct columns."""
    n = len(matrix)
    coefficients = []
    for size in range(n + 1):
        total = 0
        for rows in itertools.combinations(range(n), size):
            for columns in itertools.permutations(range(n), size):
                total += math.prod(matrix[row][column] for row, column in zip(rows, columns))
        coefficients.append(total)
    return tuple(coefficients)


BACKENDS = ["python"] + (["native"] if polynomial.HAS_POLYNOMIAL_EXT else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_permuted_diagonal_uses_exact_factors_at_maximum_order(monkeypatch, backend):
    def forbidden(*args, **kwargs):
        raise AssertionError("isolated edges must not enter CRT enumeration")

    monkeypatch.setattr(polynomial, "_weighted_rook_polynomial_crt_validated", forbidden)
    n, present, weight = 63, 60, 10**40 + 7
    matrix = [[0] * n for _ in range(n)]
    for row in range(present):
        matrix[row][(17 * row + 5) % n] = weight
    expected = tuple(math.comb(present, k) * weight**k for k in range(present + 1)) + (0,) * (n-present)
    assert polynomial.weighted_rook_polynomial(matrix, backend=backend, num_threads=1) == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_rectangular_disconnected_components_and_isolates_match_definition(backend):
    # A 2x3 component and a 3x2 component need different row/column padding;
    # the sixth row and column are isolated. Matching degree is only four.
    matrix = [[2, 3, 5, 0, 0, 0], [7, 11, 13, 0, 0, 0],
              [0, 0, 0, 17, 19, 0], [0, 0, 0, 23, 29, 0],
              [0, 0, 0, 31, 37, 0], [0, 0, 0, 0, 0, 0]]
    expected = _matching_oracle(matrix)
    assert expected[-2:] == (0, 0)
    assert polynomial.weighted_rook_polynomial(matrix, backend=backend, num_threads=1) == expected
    rng = random.Random(2026090701)
    rows, columns = list(range(6)), list(range(6))
    rng.shuffle(rows)
    rng.shuffle(columns)
    permuted = [[matrix[row][column] for column in columns] for row in rows]
    assert polynomial.weighted_rook_polynomial(permuted, backend=backend, num_threads=1) == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_star_components_and_isolated_vertices_need_no_crt(monkeypatch, backend):
    def forbidden(*args, **kwargs):
        raise AssertionError("star components must not enter CRT enumeration")

    monkeypatch.setattr(polynomial, "_weighted_rook_polynomial_crt_validated", forbidden)
    matrix = [[2, 3, 5, 0, 0], [0, 0, 0, 7, 0], [0, 0, 0, 11, 0],
              [0, 0, 0, 0, 13], [0, 0, 0, 0, 0]]
    assert polynomial.weighted_rook_polynomial(matrix, backend=backend) == _matching_oracle(matrix)


@pytest.mark.parametrize("backend", BACKENDS)
def test_block_triangular_permanent_factors_are_not_partial_matching_factors(backend):
    # The off-diagonal edge is irrelevant to the permanent but contributes
    # to r_1. A perfect-matching SCC factorization would incorrectly lose it.
    matrix = [[1, 7], [0, 1]]
    assert polynomial.weighted_rook_polynomial(matrix, backend=backend, num_threads=1) == (1, 9, 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_huge_weight_block_factors_match_independent_matching_oracle(backend):
    h = 2**150 + 7
    matrix = [[h, 2, 0, 0], [3, h + 1, 0, 0],
              [0, 0, h + 2, 5], [0, 0, 7, h + 3]]
    assert polynomial.weighted_rook_polynomial(matrix, backend=backend, num_threads=1) == _matching_oracle(matrix)


@pytest.mark.parametrize("backend", BACKENDS)
def test_seeded_sparse_and_dense_inputs_match_definition(backend):
    rng = random.Random(2026090702)
    for n in range(2, 6):
        for density in (0.15, 0.45, 1.0):
            matrix = [[rng.randrange(1, 8) if rng.random() < density else 0
                       for _ in range(n)] for _ in range(n)]
            assert polynomial.weighted_rook_polynomial(
                matrix, backend=backend, num_threads=1,
            ) == _matching_oracle(matrix)


@pytest.mark.skipif(not polynomial.HAS_POLYNOMIAL_EXT, reason="optional native extension not built")
def test_real_exhausted_native_plan_falls_back_to_exact_affine():
    # Every entry fits uint16, but the total line sum makes the common-width
    # fused reduction impossible. The connected matrix reaches real fallback.
    matrix = [[32767, 32767], [32767, 32767]]
    with pytest.raises(RuntimeError):
        polynomial._plan_weighted_rook_pseudo_mersenne_primes(
            10**20, order=2, maximum_signed_sum=65534,
        )
    assert polynomial.weighted_rook_polynomial(matrix, backend="native", num_threads=1) == _matching_oracle(matrix)


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_structural_shortcuts_preserve_input_and_backend_validation(backend):
    with pytest.raises(ValueError, match="nonnegative"):
        polynomial.weighted_rook_polynomial([[1, 0], [0, -1]], backend=backend)
    with pytest.raises(TypeError, match="integers"):
        polynomial.weighted_rook_polynomial([[1.0, 0], [0, 1]], backend=backend)
    with pytest.raises(ValueError, match="square"):
        polynomial.weighted_rook_polynomial([[1, 0, 0], [0, 1, 0]], backend=backend)


def test_native_backend_still_requires_extension_for_diagonal(monkeypatch):
    monkeypatch.setattr(polynomial, "HAS_POLYNOMIAL_EXT", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        polynomial.weighted_rook_polynomial([[2, 0], [0, 3]], backend="native")
