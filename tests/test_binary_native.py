"""Device-level tests for the optional binary OpenMP extension."""

import itertools
import math
import random

import pytest

import perm_binary


perm_binary_ext = pytest.importorskip("perm_binary_ext")


def _definition_permanent(matrix):
    return sum(
        math.prod(matrix[row][permutation[row]] for row in range(len(matrix)))
        for permutation in itertools.permutations(range(len(matrix)))
    )


def test_native_random_matrices_match_definition_across_montgomery_paths():
    rng = random.Random(20260815)
    # The second prime is immediately above the Mont32 cutoff; the third
    # exercises genuinely wide intermediates in the Mont64 path.
    primes = (
        2_147_483_647,
        2_147_483_659,
        2_305_843_009_213_693_951,
        9_223_372_036_854_775_783,
    )
    for order in range(8):
        trials = 10 if order < 7 else 4
        for _ in range(trials):
            matrix = [
                [rng.randrange(2) for _ in range(order)]
                for _ in range(order)
            ]
            expected = _definition_permanent(matrix)
            for prime in primes:
                for threads in (1, 3):
                    assert (
                        perm_binary_ext.binary_permanent_mod_p(
                            matrix, prime, threads
                        )
                        == expected % prime
                    )


def test_native_thread_blocks_and_transpose_preserve_residue():
    matrix = [
        [1, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 1],
    ]
    transpose = [list(row) for row in zip(*matrix)]
    prime = 2_147_483_647
    expected = _definition_permanent(matrix) % prime
    for threads in (1, 2, 3, 7):
        assert (
            perm_binary_ext.binary_permanent_mod_p(
                matrix, prime, threads
            )
            == expected
        )
        assert (
            perm_binary_ext.binary_permanent_mod_p(
                transpose, prime, threads
            )
            == expected
        )


def test_native_empty_singleton_and_zero_row_shortcuts():
    assert perm_binary_ext.binary_permanent_mod_p([], 3) == 1
    assert perm_binary_ext.binary_permanent_mod_p([[0]], 3) == 0
    assert perm_binary_ext.binary_permanent_mod_p([[1]], 3) == 1

    # This also proves the accepted order-63 boundary does not attempt its
    # exponential traversal when a zero row or column makes the answer zero.
    zero_matrix = [[0] * 63 for _ in range(63)]
    assert (
        perm_binary_ext.binary_permanent_mod_p(
            zero_matrix, 2_147_483_647
        )
        == 0
    )


@pytest.mark.parametrize(
    "matrix,prime,threads,error,message",
    [
        ([[1, 0]], 101, 0, ValueError, "square"),
        ([[1, 2], [0, 1]], 101, 0, ValueError, "0 or 1"),
        ([[1, 0], [0, 1]], 9, 0, ValueError, "prime"),
        (
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            3,
            0,
            ValueError,
            "greater than",
        ),
        ([[1]], 2, 0, ValueError, "odd"),
        ([[1, 0], [0, 1]], 101, -1, ValueError, "num_threads"),
        ([[1]], True, 0, TypeError, "p must be an integer"),
        ([[1]], 101, True, TypeError, "num_threads must be an integer"),
    ],
)
def test_native_rejects_invalid_inputs(matrix, prime, threads, error, message):
    with pytest.raises(error, match=message):
        perm_binary_ext.binary_permanent_mod_p(matrix, prime, threads)


def test_native_rejects_unsupported_order_before_traversal():
    matrix = [[0] * 64 for _ in range(64)]
    with pytest.raises(ValueError, match="at most 63"):
        perm_binary_ext.binary_permanent_mod_p(matrix, 101)


def test_public_native_backend_matches_portable_exact_backend():
    rng = random.Random(314159)
    for order in range(2, 9):
        for _ in range(5):
            matrix = [
                [rng.randrange(2) for _ in range(order)]
                for _ in range(order)
            ]
            assert perm_binary.binary_permanent(
                matrix, backend="native", num_threads=3
            ) == perm_binary.binary_permanent(matrix, backend="python")


def test_crt_planner_uses_two_31_bit_passes_for_periodic_grid_bound():
    bound = 1_393_140_695_040_000_000
    primes = perm_binary._crt_primes_above_bound(bound)
    assert primes == (2_147_483_647, 2_147_483_629)
    assert math.prod(primes) > bound
    assert primes[0] == (1 << 31) - 1
    assert all(prime < (1 << 31) for prime in primes)


def test_64_bit_primality_guard_rejects_known_strong_pseudoprime():
    # This composite passes the small consecutive-base checks used by several
    # incomplete Miller--Rabin implementations.
    assert not perm_binary._is_prime(341_550_071_728_321)


def test_native_reports_positive_openmp_capacity():
    assert perm_binary_ext.get_max_threads() >= 1


def test_native_caps_oversized_explicit_thread_request():
    # Order eight has enough states to enter the parallel kernel.  INT32_MAX
    # exercises the wrapper's capacity cap without allowing the runtime to
    # attempt an enormous worker team.
    matrix = [[1] * 8 for _ in range(8)]
    assert (
        perm_binary_ext.binary_permanent_mod_p(
            matrix, 2_147_483_647, (1 << 31) - 1
        )
        == math.factorial(8)
    )
