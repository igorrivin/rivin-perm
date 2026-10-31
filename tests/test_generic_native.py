"""Independent checks for the optional generic modular OpenMP backend."""

import itertools
import math
import os
import random

import pytest


try:
    from perm_c_ext import permanent_mod_p, ryser_permanent_mod_p
except ImportError:
    if os.environ.get("PERM_REQUIRE_NATIVE_TESTS") == "1":
        raise
    pytest.skip(
        "perm_c_ext must be built with the generic permanent entry point",
        allow_module_level=True,
    )


INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _definition_permanent(matrix):
    """Sum permutation products with unbounded Python integer arithmetic."""
    return sum(
        math.prod(matrix[row][column] for row, column in enumerate(columns))
        for columns in itertools.permutations(range(len(matrix)))
    )


def _subset_permanent(matrix, modulus):
    """Matching DP, independent of both signed-sum permanent identities."""
    order = len(matrix)
    counts = [0] * (1 << order)
    counts[0] = 1 % modulus
    for used in range(1, len(counts)):
        row = matrix[bin(used).count("1") - 1]
        remaining = used
        total = 0
        while remaining:
            bit = remaining & -remaining
            total += counts[used ^ bit] * row[bit.bit_length() - 1]
            remaining ^= bit
        counts[used] = total % modulus
    return counts[-1]


@pytest.mark.parametrize("modulus", [1, 2, 9, 101, INT64_MAX])
def test_exhaustive_binary_matrices_through_order_three(modulus):
    for order in range(4):
        for entries in itertools.product((0, 1), repeat=order * order):
            matrix = [
                list(entries[row * order:(row + 1) * order])
                for row in range(order)
            ]
            expected = _definition_permanent(matrix) % modulus
            assert permanent_mod_p(matrix, modulus, 1) == expected


@pytest.mark.parametrize("modulus", [2, 3, 9, 15, 101, INT64_MAX])
def test_exhaustive_signed_order_two_matrices(modulus):
    for entries in itertools.product((-1, 0, 1), repeat=4):
        matrix = [list(entries[:2]), list(entries[2:])]
        assert permanent_mod_p(matrix, modulus, 1) == (
            _definition_permanent(matrix) % modulus
        )


@pytest.mark.parametrize("modulus", [9, 25, 45])
def test_odd_composite_moduli_with_zero_divisors(modulus):
    rng = random.Random(161803)
    values = [-1, 0, 1] + [
        value for value in range(2, modulus) if modulus % value == 0
    ]
    for order in (3, 5, 6):
        matrix = [[rng.choice(values) for _ in range(order)]
                  for _ in range(order)]
        assert permanent_mod_p(matrix, modulus, 1) == (
            _definition_permanent(matrix) % modulus
        )


@pytest.mark.parametrize(
    "modulus",
    [1, 2, 3, 6, 9, 15, 101, 3_037_000_499, 3_037_000_501,
     INT64_MAX - 1, INT64_MAX],
)
def test_seeded_signed_matrices_match_definition_across_modulus_ranges(modulus):
    rng = random.Random(20260904)
    for order in range(7):
        for _ in range(3):
            matrix = [
                [rng.randrange(INT64_MIN, INT64_MAX + 1)
                 for _ in range(order)]
                for _ in range(order)
            ]
            expected = _definition_permanent(matrix) % modulus
            assert permanent_mod_p(matrix, modulus, 1) == expected
            assert ryser_permanent_mod_p(matrix, modulus, 1) == expected


def _domain_matrix(domain, order=6):
    rng = random.Random(314159)
    matrix = [[rng.randrange(-5, 6) for _ in range(order)]
              for _ in range(order)]
    if domain == "dense_signed":
        return matrix
    if domain == "binary":
        return [[rng.randrange(2) for _ in range(order)]
                for _ in range(order)]
    if domain == "sparse":
        return [[value if rng.random() < 0.25 else 0 for value in row]
                for row in matrix]
    if domain == "triangular":
        return [[value if row <= column else 0
                 for column, value in enumerate(values)]
                for row, values in enumerate(matrix)]
    if domain == "rank_one":
        return [[(row - 2) * (column + 1) for column in range(order)]
                for row in range(order)]
    if domain == "repeated_rows":
        matrix[-1] = matrix[0][:]
        return matrix
    if domain == "zero_row":
        matrix[-1] = [0] * order
        return matrix
    if domain == "block_diagonal":
        return [[value if row // 3 == column // 3 else 0
                 for column, value in enumerate(values)]
                for row, values in enumerate(matrix)]
    raise AssertionError(domain)


@pytest.mark.parametrize(
    "domain",
    ["dense_signed", "binary", "sparse", "triangular", "rank_one",
     "repeated_rows", "zero_row", "block_diagonal"],
)
@pytest.mark.parametrize("modulus", [9, 101, INT64_MAX])
def test_distinct_matrix_domains_and_symmetries_match_definition(domain, modulus):
    matrix = _domain_matrix(domain)
    expected = _definition_permanent(matrix) % modulus
    transposed = [list(row) for row in zip(*matrix)]
    permuted = [row[2:] + row[:2] for row in reversed(matrix)]
    for equivalent in (matrix, transposed, permuted):
        assert permanent_mod_p(equivalent, modulus, 1) == expected


@pytest.mark.parametrize("modulus", [1, 2, 9, 101, INT64_MAX])
def test_empty_singletons_and_signed_64_bit_entries(modulus):
    assert permanent_mod_p([], modulus) == 1 % modulus
    for value in (INT64_MIN, -1, 0, 1, INT64_MAX):
        assert permanent_mod_p([[value]], modulus) == value % modulus
    matrix = [
        [INT64_MIN, INT64_MAX, -1],
        [INT64_MAX, INT64_MIN, 1],
        [0, -1, INT64_MAX],
    ]
    assert permanent_mod_p(matrix, modulus) == (
        _definition_permanent(matrix) % modulus
    )


@pytest.mark.parametrize(
    "order,modulus", [(16, 9), (17, 101), (18, 2_147_483_647), (18, INT64_MAX)]
)
def test_serial_parallel_boundary_and_uneven_blocks_match_subset_dp(order, modulus):
    rng = random.Random(271828 + order)
    matrix = [[rng.randrange(-modulus, modulus) for _ in range(order)]
              for _ in range(order)]
    expected = _subset_permanent(matrix, modulus)
    for threads in (1, 2, 3, 5):
        assert permanent_mod_p(matrix, modulus, threads) == expected
        assert ryser_permanent_mod_p(matrix, modulus, threads) == expected


@pytest.mark.parametrize("threads", [0, -1, 1, 3])
def test_thread_argument_preserves_existing_ryser_contract(threads):
    matrix = [[2, -1], [3, 4]]
    assert permanent_mod_p(matrix, 101, threads) == 5


@pytest.mark.parametrize(
    "matrix,modulus,threads,error,message",
    [
        (([1],), 101, 1, TypeError, "list of lists"),
        (1, 101, 1, TypeError, "list of lists"),
        ([(1,)], 101, 1, ValueError, "square"),
        ([[1, 2]], 101, 1, ValueError, "square"),
        ([[1], [2, 3]], 101, 1, ValueError, "square"),
        ([[0] * 64 for _ in range(64)], 101, 1, ValueError, "less than 64"),
        ([[1]], 0, 1, ValueError, "positive"),
        ([[1]], -3, 1, ValueError, "positive"),
        ([[1]], 1 << 63, 1, OverflowError, None),
        ([[1]], 101.0, 1, TypeError, None),
        ([[1.0]], 101, 1, TypeError, None),
        ([[1 << 63]], 101, 1, OverflowError, None),
        ([[INT64_MIN - 1]], 101, 1, OverflowError, None),
        ([[1]], 101, 1 << 31, OverflowError, None),
        ([[1]], 101, 1.0, TypeError, None),
    ],
)
def test_generic_and_explicit_ryser_validate_inputs_identically(
    matrix, modulus, threads, error, message
):
    for implementation in (permanent_mod_p, ryser_permanent_mod_p):
        with pytest.raises(error, match=message):
            implementation(matrix, modulus, threads)
