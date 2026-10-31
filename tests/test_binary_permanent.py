"""Correctness and API tests for exact zero-one matrix permanents."""

import itertools
import math

import numpy as np
import pytest

import perm_binary


def _permutation_oracle(matrix):
    """Definition-level permanent; intentionally independent of project code."""
    n = len(matrix)
    total = 0
    for permutation in itertools.permutations(range(n)):
        product = 1
        for row, column in enumerate(permutation):
            product *= matrix[row][column]
        total += product
    return total


def _block_diagonal_ones(block_size, block_count):
    n = block_size * block_count
    return [
        [
            int(row // block_size == column // block_size)
            for column in range(n)
        ]
        for row in range(n)
    ]


def test_python_backend_matches_brute_force_random_matrices():
    rng = np.random.default_rng(20260815)
    for n in range(8):
        trials = 12 if n < 7 else 4
        for _ in range(trials):
            matrix = rng.integers(0, 2, size=(n, n), dtype=np.int8)
            expected = _permutation_oracle(matrix.tolist())
            assert (
                perm_binary.binary_permanent(matrix, backend="python")
                == expected
            )


@pytest.mark.parametrize("modulus", [5, 97, 1_073_741_827])
def test_modular_python_backend_matches_definition(modulus):
    matrix = [
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
    ]
    expected = _permutation_oracle(matrix)
    assert (
        perm_binary.binary_permanent_mod_p(
            matrix, modulus, backend="python"
        )
        == expected % modulus
    )


def test_empty_and_one_by_one_conventions():
    assert perm_binary.binary_permanent([], backend="python") == 1
    assert (
        perm_binary.binary_permanent(
            np.empty((0, 0), dtype=np.int8), backend="python"
        )
        == 1
    )
    assert perm_binary.binary_permanent([[0]], backend="python") == 0
    assert perm_binary.binary_permanent([[1]], backend="python") == 1
    assert (
        perm_binary.binary_permanent_mod_p([], 7, backend="python")
        == 1
    )


def test_zero_row_or_column_and_hall_obstruction_return_zero():
    zero_row = [[1, 0, 1], [0, 0, 0], [0, 1, 1]]
    zero_column = [[1, 0, 0], [0, 0, 1], [1, 0, 1]]
    hall_obstruction = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    ]
    for matrix in (zero_row, zero_column, hall_obstruction):
        assert perm_binary.binary_permanent(matrix, backend="python") == 0


def test_arbitrary_size_python_integer_result():
    # Components make the answer cheap to compute while taking more than 64
    # bits, catching accidental fixed-width storage in the exact API.
    matrix = _block_diagonal_ones(block_size=10, block_count=5)
    expected = math.factorial(10) ** 5
    result = perm_binary.binary_permanent(matrix, backend="python")
    assert result == expected
    assert result.bit_length() > 64
    assert isinstance(result, int)


def test_bound_is_rigorous_and_uses_binary_degree_information():
    all_ones = np.ones((7, 7), dtype=np.int8)
    assert perm_binary.binary_permanent_bound(all_ones) == math.factorial(7)

    matrix = [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ]
    value = perm_binary.binary_permanent(matrix, backend="python")
    bound = perm_binary.binary_permanent_bound(matrix)
    assert value <= bound
    elementary_bound = min(
        math.factorial(4),
        math.prod(map(sum, matrix)),
        math.prod(sum(row[column] for row in matrix) for column in range(4)),
    )
    assert bound <= elementary_bound
    assert bound == 2  # matching-induced blocks make this example exact
    assert perm_binary.binary_permanent_bound([[0, 0], [1, 1]]) == 0
    assert perm_binary.binary_permanent_bound([]) == 1


def test_grouped_bregman_minc_bound_is_exactly_rounded():
    # Four degree-two rows contribute (2!)^(4/2) = 4 exactly.
    assert perm_binary._grouped_bregman_minc_bound([2, 2, 2, 2]) == 4
    # Three degree-three rows contribute (3!)^(3/3) = 6.
    assert perm_binary._grouped_bregman_minc_bound([3, 3, 3]) == 6
    # A partial group exercises the exact integer root rather than floats.
    assert perm_binary._grouped_bregman_minc_bound([3, 3]) == 4
    assert perm_binary._ceil_nth_root(36, 3) == 4
    assert perm_binary._ceil_nth_root(64, 3) == 4
    assert perm_binary._ceil_nth_root(65, 3) == 5


def test_bregman_minc_improves_simple_degree_product_bound():
    # Every row and column has degree two.  The degree-product bound is 16,
    # whereas grouped Bregman--Minc certifies an upper bound of 4.
    cycle = [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ]
    assert perm_binary.binary_permanent(cycle, backend="python") == 2
    assert perm_binary.binary_permanent_bound(cycle) == 4


def test_matching_feasibility_short_circuits_exponential_kernel(monkeypatch):
    hall_obstruction = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subset DP should not run without a perfect matching")

    monkeypatch.setattr(perm_binary, "_permanent_dp", fail_if_called)
    assert (
        perm_binary.binary_permanent(hall_obstruction, backend="python")
        == 0
    )
    assert perm_binary.binary_permanent_bound(hall_obstruction) == 0


def test_strong_components_preserve_cross_edge_finishing_order():
    adjacency = ((1, 2), (), (1,))
    components = perm_binary._strong_components(adjacency)
    assert {frozenset(component) for component in components} == {
        frozenset((0,)), frozenset((1,)), frozenset((2,))
    }


@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4), (0, 4, 3, 2, 1)])
def test_matching_scc_preprocessing_removes_forced_triangular_edges(monkeypatch, order):
    upper_triangular = [
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
    ]
    # Relabeling both sides preserves the unique matching but changes which
    # siblings the DFS encounters first.
    upper_triangular = [
        [upper_triangular[row][column] for column in order] for row in order
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("singleton forced blocks need no exponential kernel")

    monkeypatch.setattr(perm_binary, "_permanent_dp", fail_if_called)
    assert (
        perm_binary.binary_permanent(upper_triangular, backend="python")
        == 1
    )
    assert perm_binary.binary_permanent_bound(upper_triangular) == 1


def test_matching_scc_blocks_factor_the_permanent():
    matrix = _block_diagonal_ones(block_size=4, block_count=3)
    feasible, blocks = perm_binary._matching_blocks(
        perm_binary._row_masks(matrix)
    )
    assert feasible
    assert sorted(map(len, blocks)) == [4, 4, 4]
    assert (
        perm_binary.binary_permanent(matrix, backend="python")
        == math.factorial(4) ** 3
    )
    assert (
        perm_binary.binary_permanent_bound(matrix)
        == math.factorial(4) ** 3
    )


def test_matching_scc_factorization_matches_oracle_on_random_inputs():
    rng = np.random.default_rng(1729)
    for n in range(1, 8):
        for _ in range(10):
            matrix = rng.integers(0, 2, size=(n, n), dtype=np.int8).tolist()
            expected = _permutation_oracle(matrix)
            feasible, blocks = perm_binary._matching_blocks(
                perm_binary._row_masks(matrix)
            )
            if not feasible:
                assert expected == 0
                continue
            factorized = math.prod(
                perm_binary._permanent_dp(block) for block in blocks
            )
            assert factorized == expected


@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 0, 1], [0, 1, 0]],
        [[1, 0]],
        np.empty((0, 2), dtype=np.int8),
    ],
)
def test_rectangular_matrices_are_rejected(matrix):
    with pytest.raises(ValueError, match="square"):
        perm_binary.binary_permanent(matrix, backend="python")
    with pytest.raises(ValueError, match="square"):
        perm_binary.binary_permanent_bound(matrix)


@pytest.mark.parametrize(
    "matrix,message",
    [
        (np.array([0, 1]), "two-dimensional"),
        (np.zeros((1, 1, 1), dtype=np.int8), "two-dimensional"),
        ([1], "rows must be sequences"),
    ],
)
def test_malformed_matrix_shapes_are_rejected(matrix, message):
    with pytest.raises((TypeError, ValueError), match=message):
        perm_binary.binary_permanent(matrix, backend="python")


def test_custom_row_cannot_yield_a_different_width_than_it_reports():
    class MisreportedRow:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([1])

    matrix = [MisreportedRow(), MisreportedRow()]
    with pytest.raises(ValueError, match="row width changed"):
        perm_binary.binary_permanent(matrix, backend="python")


def test_strict_mode_accepts_integral_zero_one_and_booleans():
    matrix = np.array([[True, False], [False, True]], dtype=np.bool_)
    assert perm_binary.binary_permanent(matrix, backend="python") == 1
    assert (
        perm_binary.binary_permanent(
            [[np.int64(1), np.int8(0)], [False, True]], backend="python"
        )
        == 1
    )


@pytest.mark.parametrize("entry", [-1, 2, 17])
def test_strict_mode_rejects_nonbinary_integers(entry):
    with pytest.raises(ValueError, match="0 or 1"):
        perm_binary.binary_permanent([[entry]], backend="python")


@pytest.mark.parametrize("entry", [0.0, 1.0, "1", None, 1 + 0j])
def test_both_modes_reject_nonintegral_entries(entry):
    for strict in (True, False):
        with pytest.raises(TypeError, match="entries must be integers"):
            perm_binary.binary_permanent(
                [[entry]], backend="python", strict=strict
            )


def test_explicit_nonstrict_mode_constructs_support_matrix():
    weighted = [[-7, 0, 0], [0, 9, 2], [5, 0, 3]]
    support = [[1, 0, 0], [0, 1, 1], [1, 0, 1]]
    expected = _permutation_oracle(support)
    assert (
        perm_binary.binary_permanent(
            weighted, backend="python", strict=False
        )
        == expected
    )
    assert (
        perm_binary.binary_permanent_bound(weighted, strict=False)
        >= expected
    )


@pytest.mark.parametrize("strict", [0, 1, None, "yes"])
def test_strict_flag_itself_must_be_boolean(strict):
    with pytest.raises(TypeError, match="strict must be a boolean"):
        perm_binary.binary_permanent([[1]], strict=strict)


@pytest.mark.parametrize("parallel", [0, 1, None, "yes"])
def test_parallel_flag_itself_must_be_boolean(parallel):
    with pytest.raises(TypeError, match="parallel must be a boolean"):
        perm_binary.binary_permanent([[1]], parallel=parallel)


@pytest.mark.parametrize("num_threads", [-1, 1.5, True, np.bool_(True)])
def test_invalid_thread_counts_are_rejected(num_threads):
    error = (
        ValueError
        if type(num_threads) is int and num_threads == -1
        else TypeError
    )
    with pytest.raises(error, match="num_threads"):
        perm_binary.binary_permanent([[1]], num_threads=num_threads)


@pytest.mark.parametrize(
    "modulus", [0, 1, -3, 2, 4, 9, 2.0, True, np.bool_(True), 1 << 63]
)
def test_invalid_moduli_are_rejected(modulus):
    error = TypeError if isinstance(modulus, (float, bool, np.bool_)) else ValueError
    with pytest.raises(error, match="modulus"):
        perm_binary.binary_permanent_mod_p([[1]], modulus)


def test_backend_names_and_unavailable_native_backend(monkeypatch):
    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", False)
    assert perm_binary.get_binary_backend() == "python"
    assert perm_binary.available_binary_backends() == ("python",)
    with pytest.raises(RuntimeError, match="unavailable"):
        perm_binary.binary_permanent([[1, 0], [0, 1]], backend="native")
    with pytest.raises(ValueError, match="backend must be one of"):
        perm_binary.binary_permanent([[1]], backend="cuda")
    with pytest.raises(TypeError, match="backend must be a string"):
        perm_binary.binary_permanent([[1]], backend=None)


def test_native_exact_dispatch_uses_crt_and_forwards_threads(monkeypatch):
    calls = []
    matrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    expected = _permutation_oracle(matrix)

    class FakeNative:
        @staticmethod
        def binary_permanent_mod_p(argument, modulus, num_threads):
            calls.append((argument, modulus, num_threads))
            return expected % modulus

    monkeypatch.setattr(perm_binary, "_binary_ext", FakeNative())
    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", True)
    assert (
        perm_binary.binary_permanent(
            matrix, backend="auto", num_threads=3
        )
        == expected
    )
    assert calls
    assert {call[2] for call in calls} == {3}
    assert all(call[0] == matrix for call in calls)
    assert (
        math.prod(call[1] for call in calls)
        > perm_binary.binary_permanent_bound(matrix)
    )

    calls.clear()
    assert (
        perm_binary.binary_permanent(
            matrix, backend="native", parallel=False, num_threads=17
        )
        == expected
    )
    assert calls and {call[2] for call in calls} == {1}


def test_native_modular_dispatch_and_residue_guard(monkeypatch):
    calls = []

    class FakeNative:
        @staticmethod
        def binary_permanent_mod_p(argument, modulus, num_threads):
            calls.append((argument, modulus, num_threads))
            return 4

    monkeypatch.setattr(perm_binary, "_binary_ext", FakeNative())
    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", True)
    matrix = [[1, 1], [1, 1]]
    assert (
        perm_binary.binary_permanent_mod_p(
            matrix, 7, backend="native", parallel=False, num_threads=12
        )
        == 4
    )
    assert calls == [(matrix, 7, 1)]

    with pytest.raises(RuntimeError, match="invalid residue"):
        perm_binary.binary_permanent_mod_p(matrix, 3, backend="native")


@pytest.mark.parametrize("invalid_residue", [0.5, "1", True])
def test_native_residue_guard_rejects_nonintegral_values(
    monkeypatch, invalid_residue
):
    class FakeNative:
        @staticmethod
        def binary_permanent_mod_p(argument, modulus, num_threads):
            return invalid_residue

    monkeypatch.setattr(perm_binary, "_binary_ext", FakeNative())
    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", True)
    with pytest.raises(RuntimeError, match="invalid residue"):
        perm_binary.binary_permanent_mod_p(
            [[1, 1], [1, 1]], 7, backend="native"
        )


def test_auto_falls_back_for_native_unsupported_large_block(monkeypatch):
    large_block = tuple((1 << 64) - 1 for _ in range(64))
    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", True)
    monkeypatch.setattr(
        perm_binary,
        "_matching_blocks",
        lambda row_masks: (True, (large_block,)),
    )
    monkeypatch.setattr(perm_binary, "_permanent_dp", lambda *args: 123)
    matrix = np.eye(64, dtype=np.int8)

    assert perm_binary.binary_permanent(matrix, backend="auto") == 123
    with pytest.raises(ValueError, match="less than 64"):
        perm_binary.binary_permanent(matrix, backend="native")


def test_native_crt_rejects_inconsistent_residues(monkeypatch):
    class FakeNative:
        @staticmethod
        def binary_permanent_mod_p(argument, modulus, num_threads):
            # For a matrix whose bound is 2, this reconstructs to 3.
            return 3

    monkeypatch.setattr(perm_binary, "_binary_ext", FakeNative())
    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", True)
    with pytest.raises(RuntimeError, match="outside the bound"):
        perm_binary.binary_permanent(
            [[1, 1], [1, 1]], backend="native"
        )


def test_native_crt_reconstructs_arbitrary_size_python_integer(monkeypatch):
    expected = (1 << 100) + 123_456_789
    bound = 1 << 101
    calls = []

    class FakeNative:
        @staticmethod
        def binary_permanent_mod_p(argument, modulus, num_threads):
            calls.append((modulus, num_threads))
            return expected % modulus

    monkeypatch.setattr(perm_binary, "_binary_ext", FakeNative())
    result = perm_binary._native_exact_permanent([[1, 1], [1, 1]], bound, 5)
    assert result == expected
    assert result.bit_length() > 64
    assert len(calls) >= 4
    assert math.prod(modulus for modulus, _ in calls) > bound
    assert {threads for _, threads in calls} == {5}


def test_crt_planner_uses_nonnegative_uniqueness_interval():
    for bound in (0, 1, 2, (1 << 90) - 1, 1 << 90):
        primes = perm_binary._crt_primes_above_bound(bound)
        assert math.prod(primes) > bound
        assert all(perm_binary._is_prime(prime) for prime in primes)
        if primes:
            assert math.prod(primes[:-1]) <= bound
