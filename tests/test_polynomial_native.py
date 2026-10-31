"""Direct tests for the optional affine-polynomial OpenMP extension."""

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

import perm_polynomial


try:
    import perm_polynomial_ext as native
except ImportError:
    if os.environ.get("PERM_REQUIRE_NATIVE_TESTS") == "1":
        raise
    pytest.skip(
        "perm_polynomial_ext is not built",
        allow_module_level=True,
    )


def _multiply(left, right, modulus):
    result = [0] * (len(left) + len(right) - 1)
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            degree = left_degree + right_degree
            result[degree] = (
                result[degree] + left_value * right_value
            ) % modulus
    return result


def _permutation_oracle(constant, linear, modulus):
    """Definition-level oracle independent of Glynn and Gray ordering."""
    order = len(constant)
    result = [0] * (order + 1)
    for permutation in itertools.permutations(range(order)):
        product = [1]
        for row, column in enumerate(permutation):
            product = _multiply(
                product,
                [constant[row][column], linear[row][column]],
                modulus,
            )
        for degree, value in enumerate(product):
            result[degree] = (result[degree] + value) % modulus
    return tuple(result)


def _permutation_oracle_exact(constant, linear):
    order = len(constant)
    result = [0] * (order + 1)
    for permutation in itertools.permutations(range(order)):
        product = [1]
        for row, column in enumerate(permutation):
            updated = [0] * (len(product) + 1)
            for degree, coefficient in enumerate(product):
                updated[degree] += coefficient * constant[row][column]
                updated[degree + 1] += coefficient * linear[row][column]
            product = updated
        for degree, coefficient in enumerate(product):
            result[degree] += coefficient
    return tuple(result)


def _subset_oracle(constant, linear, modulus):
    """Coefficient-valued matching DP for somewhat larger thread tests."""
    order = len(constant)
    states = {0: (1,)}
    for row in range(order):
        following = {}
        for used, coefficients in states.items():
            for column in range(order):
                bit = 1 << column
                if used & bit:
                    continue
                product = _multiply(
                    coefficients,
                    [constant[row][column], linear[row][column]],
                    modulus,
                )
                target = used | bit
                prior = following.get(target)
                if prior is None:
                    following[target] = tuple(product)
                else:
                    following[target] = tuple(
                        (left + right) % modulus
                        for left, right in zip(prior, product)
                    )
        states = following
    return states.get((1 << order) - 1, (0,) * (order + 1))


def _rook_subset_oracle(matrix):
    """Sum weighted partial matchings by row DP, independent of Glynn."""
    order = len(matrix)
    states = {0: 1}
    for row in range(order):
        following = dict(states)
        for used, count in states.items():
            for column, entry in enumerate(matrix[row]):
                bit = 1 << column
                if entry and not used & bit:
                    target = used | bit
                    following[target] = (
                        following.get(target, 0) + count * entry
                    )
        states = following
    result = [0] * (order + 1)
    for used, count in states.items():
        result[used.bit_count()] += count
    return tuple(result)


@pytest.mark.parametrize(
    "modulus",
    [
        3,
        9,
        45,
        2_147_483_645,
        2_147_483_647,
        2_147_483_649,
        (1 << 63) - 1,
    ],
)
def test_random_small_matrices_match_permutation_definition(modulus):
    rng = random.Random(0xA771E + modulus)
    for order in range(6):
        for _ in range(4):
            constant = [
                [rng.randrange(-20, 21) for _ in range(order)]
                for _ in range(order)
            ]
            linear = [
                [rng.randrange(-20, 21) for _ in range(order)]
                for _ in range(order)
            ]
            expected = _permutation_oracle(constant, linear, modulus)
            assert native.affine_permanent_mod(
                constant, linear, modulus, 1
            ) == expected


def test_public_exact_native_backend_matches_random_signed_definition():
    rng = random.Random(0xE7AC7)
    for order in range(2, 7):
        for _ in range(3):
            constant = [
                [rng.randrange(-25, 26) for _ in range(order)]
                for _ in range(order)
            ]
            linear = [
                [rng.randrange(-25, 26) for _ in range(order)]
                for _ in range(order)
            ]
            assert perm_polynomial.affine_permanent(
                constant,
                linear,
                backend="native",
                num_threads=3,
            ) == _permutation_oracle_exact(constant, linear)


def test_public_exact_native_backend_reduces_arbitrary_python_integers():
    huge = 10**60 + 39
    constant = [[huge, 0], [0, -huge]]
    linear = [[0, huge + 4], [huge - 7, 0]]
    assert perm_polynomial.affine_permanent(
        constant, linear, backend="native", num_threads=2
    ) == (-huge * huge, 0, (huge + 4) * (huge - 7))


def test_public_principal_mod_native_matches_nonsymmetric_definition():
    rng = random.Random(0xC01A5E)
    for order in range(6):
        identity = [
            [int(row == column) for column in range(order)]
            for row in range(order)
        ]
        for modulus in (9, 101, 2_147_483_647):
            matrix = [
                [rng.randrange(-20, 21) for _ in range(order)]
                for _ in range(order)
            ]
            assert perm_polynomial.principal_permanent_polynomial_mod(
                matrix, modulus, backend="native", num_threads=3
            ) == _permutation_oracle(identity, matrix, modulus)


def test_public_capability_snapshot_matches_loaded_native_extension():
    expected_hooks = tuple(
        name
        for name in (
            "affine_permanent_mod",
            "rook_polynomial_mod_many",
            "weighted_rook_polynomial_mod_many",
        )
        if hasattr(native, name)
    )
    assert perm_polynomial.polynomial_backend_capabilities() == {
        "available_backends": ("native", "python"),
        "default_backend": "native",
        "native_extension_available": True,
        "native_hooks": expected_hooks,
    }


def test_signed_64_bit_entries_and_boundary_modulus():
    minimum = -(1 << 63)
    maximum = (1 << 63) - 1
    constant = [
        [minimum, maximum, -maximum],
        [maximum - 1, minimum + 1, 17],
        [-9, maximum, minimum],
    ]
    linear = [
        [maximum, minimum, 5],
        [-maximum, 11, minimum + 2],
        [minimum, maximum - 3, -1],
    ]
    for modulus in (2_147_483_647, 2_147_483_649, maximum):
        assert native.affine_permanent_mod(
            constant, linear, modulus, 2
        ) == _permutation_oracle(constant, linear, modulus)


@pytest.mark.parametrize(
    "modulus",
    [2_147_483_645, 2_147_483_647, 2_147_483_649, (1 << 63) - 1],
)
def test_large_centered_row_sums_use_the_fully_modular_kernel(modulus):
    order = 4
    half = modulus // 2
    constant = [
        [(-1) ** (row + column) * (half - 17 * row - column - 1)
         for column in range(order)]
        for row in range(order)
    ]
    linear = [
        [(-1) ** (row + 2 * column) * (half - 13 * row - 3 * column - 2)
         for column in range(order)]
        for row in range(order)
    ]
    assert native.affine_permanent_mod(
        constant, linear, modulus, 3
    ) == _permutation_oracle(constant, linear, modulus)


def test_arbitrary_gray_chunks_and_thread_capping_match_subset_oracle():
    order = 8
    rng = random.Random(20260817)
    constant = [
        [rng.randrange(-50, 51) for _ in range(order)]
        for _ in range(order)
    ]
    linear = [
        [rng.randrange(-50, 51) for _ in range(order)]
        for _ in range(order)
    ]
    modulus = 63
    expected = _subset_oracle(constant, linear, modulus)
    for threads in (1, 2, 3, 5, 10**9):
        assert native.affine_permanent_mod(
            constant, linear, modulus, threads
        ) == expected


@pytest.mark.parametrize(
    "modulus", [2_147_483_647, 2_147_483_645, 2_147_483_649]
)
def test_dense_case_exercises_checked_exact_convolution_fallback(modulus):
    order = 16
    constant = [[1] * order for _ in range(order)]
    linear = [[1] * order for _ in range(order)]
    factorial = math.factorial(order) % modulus
    expected = tuple(
        factorial * math.comb(order, degree) % modulus
        for degree in range(order + 1)
    )
    assert native.affine_permanent_mod(
        constant, linear, modulus, 3
    ) == expected


def test_fused_rook_residues_match_independent_partial_matching_dp():
    moduli = (
        67_108_859,
        67_108_837,
        67_108_819,
        67_108_777,
        67_108_763,
    )
    rng = random.Random(0xF05ED)
    for order in range(7):
        for _ in range(3):
            matrix = [
                [rng.randrange(2) for _ in range(order)]
                for _ in range(order)
            ]
            expected = _rook_subset_oracle(matrix)
            observed = native.rook_polynomial_mod_many(
                matrix, moduli, 3
            )
            assert len(observed) == len(moduli)
            assert all(len(row) == order + 1 for row in observed)
            assert observed == tuple(
                tuple(value % modulus for value in expected)
                for modulus in moduli
            )


def test_fused_rook_orientation_transpose_and_threads_are_invariant():
    order = 10
    matrix = [
        [int((3 * row + 5 * column + row * column) % 7 < 3)
         for column in range(order)]
        for row in range(order)
    ]
    # Make the row- and column-degree maxima deliberately asymmetric so one
    # call takes the internal transpose path.
    matrix[0] = [1] * order
    transposed = [list(column) for column in zip(*matrix)]
    moduli = (134_217_689, 134_217_649, 134_217_617, 134_217_613)
    expected = _rook_subset_oracle(matrix)
    for threads in (1, 2, 5):
        observed = native.rook_polynomial_mod_many(
            matrix, moduli, threads
        )
        observed_transposed = native.rook_polynomial_mod_many(
            transposed, moduli, threads
        )
        assert observed == observed_transposed
        assert observed == tuple(
            tuple(value % modulus for value in expected)
            for modulus in moduli
        )


def test_fused_weighted_rook_residues_match_partial_matching_dp():
    moduli = (
        67_108_859,
        67_108_837,
        67_108_819,
        67_108_777,
        67_108_763,
    )
    rng = random.Random(0x5E167ED)
    for order in range(7):
        for _ in range(4):
            matrix = [
                [rng.randrange(6) for _ in range(order)]
                for _ in range(order)
            ]
            expected = _rook_subset_oracle(matrix)
            observed = native.weighted_rook_polynomial_mod_many(
                matrix, moduli, 3
            )
            assert observed == tuple(
                tuple(value % modulus for value in expected)
                for modulus in moduli
            )


def test_fused_weighted_rook_exhaustive_two_by_two_matrices():
    moduli = (67_108_859, 67_108_837)
    for entries in itertools.product(range(4), repeat=4):
        matrix = [list(entries[:2]), list(entries[2:])]
        expected = _rook_subset_oracle(matrix)
        assert native.weighted_rook_polynomial_mod_many(
            matrix, moduli, 2
        ) == tuple(
            tuple(value % modulus for value in expected)
            for modulus in moduli
        )


def test_fused_weighted_rook_matches_binary_hook():
    moduli = (134_217_689, 134_217_649, 134_217_617)
    rng = random.Random(0xB1A4A7)
    for order in range(8):
        matrix = [
            [rng.randrange(2) for _ in range(order)]
            for _ in range(order)
        ]
        assert native.weighted_rook_polynomial_mod_many(
            matrix, moduli, 3
        ) == native.rook_polynomial_mod_many(matrix, moduli, 3)


def test_fused_weighted_rook_transpose_and_threads_are_invariant():
    order = 9
    matrix = [
        [((7 * row + 11 * column + row * column) % 6)
         for column in range(order)]
        for row in range(order)
    ]
    # Force asymmetric weighted line sums so one orientation transposes.
    matrix[0] = [5] * order
    transposed = [list(column) for column in zip(*matrix)]
    moduli = (67_108_859, 67_108_837, 67_108_819)
    expected = _rook_subset_oracle(matrix)
    for threads in (1, 2, 5):
        observed = native.weighted_rook_polynomial_mod_many(
            matrix, moduli, threads
        )
        observed_transposed = native.weighted_rook_polynomial_mod_many(
            transposed, moduli, threads
        )
        assert observed == observed_transposed
        assert observed == tuple(
            tuple(value % modulus for value in expected)
            for modulus in moduli
        )


def test_fused_rook_modulus_contract_rejects_unsafe_inputs():
    matrix = [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="between 1 and 16"):
        native.rook_polynomial_mod_many(matrix, (), 1)
    with pytest.raises(ValueError, match="between 1 and 16"):
        native.rook_polynomial_mod_many(
            matrix, (67_108_859,) * 17, 1
        )
    with pytest.raises(ValueError, match="distinct"):
        native.rook_polynomial_mod_many(
            matrix, (67_108_859, 67_108_859), 1
        )
    with pytest.raises(ValueError, match="prime"):
        native.rook_polynomial_mod_many(matrix, (67_108_855,), 1)
    with pytest.raises(ValueError, match="same pseudo-Mersenne"):
        native.rook_polynomial_mod_many(
            matrix, (67_108_859, 134_217_689), 1
        )
    with pytest.raises(TypeError, match="modulus"):
        native.rook_polynomial_mod_many(matrix, (True,), 1)

    # A 27-bit residue is safe through max signed sum 31, but not for this
    # degree-32 all-one matrix: (32+1)*p exceeds 2**32.
    dense = [[1] * 32 for _ in range(32)]
    with pytest.raises(ValueError, match="not safe"):
        native.rook_polynomial_mod_many(dense, (134_217_689,), 1)


def test_fused_rook_accepts_sixteen_moduli():
    moduli = (
        33_554_393, 33_554_383, 33_554_371, 33_554_347,
        33_554_341, 33_554_317, 33_554_291, 33_554_273,
        33_554_267, 33_554_249, 33_554_239, 33_554_221,
        33_554_201, 33_554_167, 33_554_159, 33_554_137,
    )
    expected = tuple((1, 7) for _ in moduli)
    assert native.rook_polynomial_mod_many([[1]], moduli, 2) == tuple(
        (1, 1) for _ in moduli
    )
    assert native.weighted_rook_polynomial_mod_many(
        [[7]], moduli, 2
    ) == expected


def test_fused_weighted_rook_modulus_safety_uses_weighted_line_sum():
    # A 25-bit modulus is safe for a binary order-32 row sum, but not for
    # the weighted sum 32*5: (160+1)*p exceeds 2**32.
    dense = [[5] * 32 for _ in range(32)]
    with pytest.raises(ValueError, match="not safe"):
        native.weighted_rook_polynomial_mod_many(
            dense, (33_554_393,), 1
        )


@pytest.mark.parametrize("entry", [-1, 2, 1.5, "1", None])
def test_fused_rook_strictly_validates_binary_matrix(entry):
    error = ValueError if isinstance(entry, int) else TypeError
    with pytest.raises(error, match="entries"):
        native.rook_polynomial_mod_many([[entry]], (67_108_859,), 1)


@pytest.mark.parametrize("entry", [-1, 65_536])
def test_fused_weighted_rook_rejects_out_of_range_integer_entries(entry):
    with pytest.raises(ValueError, match="between 0 and 65535"):
        native.weighted_rook_polynomial_mod_many(
            [[entry]], (8_191,), 1
        )


@pytest.mark.parametrize("entry", [1.5, "1", None, 2 + 0j])
def test_fused_weighted_rook_rejects_nonintegral_entries(entry):
    with pytest.raises(TypeError, match="entries must be integers"):
        native.weighted_rook_polynomial_mod_many(
            [[entry]], (8_191,), 1
        )


def test_fused_weighted_rook_accepts_booleans_and_maximum_weight():
    moduli = (8_191, 8_123)
    assert native.weighted_rook_polynomial_mod_many(
        [[True, 2], [0, False]], moduli, 2
    ) == ((1, 3, 0), (1, 3, 0))
    assert native.weighted_rook_polynomial_mod_many(
        [[65_535]], moduli, 2
    ) == ((1, 7), (1, 551))


def test_fused_rook_empty_singleton_shape_and_thread_contracts():
    moduli = (67_108_859, 67_108_837)
    assert native.rook_polynomial_mod_many([], moduli) == ((1,), (1,))
    assert native.rook_polynomial_mod_many([[0]], moduli) == (
        (1, 0), (1, 0)
    )
    assert native.rook_polynomial_mod_many([[1]], moduli) == (
        (1, 1), (1, 1)
    )
    with pytest.raises(ValueError, match="square"):
        native.rook_polynomial_mod_many([[1, 0]], moduli)
    with pytest.raises(TypeError, match="rows"):
        native.rook_polynomial_mod_many([1], moduli)
    with pytest.raises(ValueError, match="num_threads"):
        native.rook_polynomial_mod_many([[1]], moduli, -1)
    with pytest.raises(TypeError, match="num_threads"):
        native.rook_polynomial_mod_many([[1]], moduli, True)


def test_fused_weighted_rook_empty_singleton_shape_and_thread_contracts():
    moduli = (67_108_859, 67_108_837)
    hook = native.weighted_rook_polynomial_mod_many
    assert hook([], moduli) == ((1,), (1,))
    assert hook([[0]], moduli) == ((1, 0), (1, 0))
    assert hook([[5]], moduli) == ((1, 5), (1, 5))
    with pytest.raises(ValueError, match="square"):
        hook([[1, 0]], moduli)
    with pytest.raises(TypeError, match="rows"):
        hook([1], moduli)
    with pytest.raises(ValueError, match="num_threads"):
        hook([[1]], moduli, -1)
    with pytest.raises(TypeError, match="num_threads"):
        hook([[1]], moduli, True)


def test_sparse_affine_degree_and_trailing_zero_conventions():
    constant = [[1, 2, 0], [0, 1, 4], [5, 0, 1]]
    linear = [[0, 0, 0], [3, 0, 0], [0, 0, 0]]
    expected = _permutation_oracle(constant, linear, 25)
    result = native.affine_permanent_mod(constant, linear, 25, 2)
    assert result == expected
    assert len(result) == 4
    assert result[2:] == (0, 0)


def test_empty_singleton_sequences_and_boolean_entries():
    assert native.affine_permanent_mod([], [], 9) == (1,)
    assert native.affine_permanent_mod(((-2,),), ((14,),), 9) == (7, 5)
    assert native.affine_permanent_mod(
        [[True, False], [False, True]],
        [[False, True], [True, False]],
        15,
    ) == (1, 0, 1)


@pytest.mark.parametrize("modulus", [-3, 0, 1, 2, 4, 10, True])
def test_invalid_moduli_are_rejected(modulus):
    error = TypeError if modulus is True else ValueError
    with pytest.raises(error, match="modulus"):
        native.affine_permanent_mod([], [], modulus)


def test_modulus_and_entries_must_fit_signed_64_bits():
    maximum = (1 << 63) - 1
    with pytest.raises(OverflowError):
        native.affine_permanent_mod([], [], maximum + 2)
    with pytest.raises(OverflowError):
        native.affine_permanent_mod([[1 << 63]], [[0]], 9)
    with pytest.raises(OverflowError):
        native.affine_permanent_mod([[-(1 << 63) - 1]], [[0]], 9)


@pytest.mark.parametrize("entry", [1.5, "1", None, 2 + 0j])
def test_nonintegral_entries_are_rejected(entry):
    with pytest.raises(TypeError, match="entries must be integers"):
        native.affine_permanent_mod([[entry]], [[0]], 9)
    with pytest.raises(TypeError, match="entries must be integers"):
        native.affine_permanent_mod([[0]], [[entry]], 9)


def test_index_exceptions_are_not_masked():
    class ExplodingIndex:
        def __index__(self):
            raise RuntimeError("index exploded")

    with pytest.raises(RuntimeError, match="index exploded"):
        native.affine_permanent_mod([[ExplodingIndex()]], [[0]], 9)
    with pytest.raises(RuntimeError, match="index exploded"):
        native.affine_permanent_mod([], [], ExplodingIndex())


def test_mutating_index_cannot_invalidate_sequence_snapshots():
    script = r"""
import perm_polynomial_ext as native

class MutatingIndex:
    def __init__(self, *targets):
        self.targets = targets

    def __index__(self):
        for target in self.targets:
            target.clear()
        return 1

constant = []
first_row = []
first_row.extend([MutatingIndex(first_row, constant), 0])
constant.extend([first_row, [0, 1]])
linear = [[0, 0], [0, 0]]
print(native.affine_permanent_mod(constant, linear, 9, 1))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(native.__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "(1, 0, 0)"


def test_weighted_rook_mutating_index_cannot_invalidate_snapshots():
    script = r"""
import perm_polynomial_ext as native

class MutatingIndex:
    def __init__(self, *targets):
        self.targets = targets

    def __index__(self):
        for target in self.targets:
            target.clear()
        return 1

matrix = []
first_row = []
first_row.extend([MutatingIndex(first_row, matrix), 2])
matrix.extend([first_row, [3, 4]])
print(native.weighted_rook_polynomial_mod_many(
    matrix, (67108859,), 1
))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(native.__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "((1, 10, 10),)"


@pytest.mark.parametrize(
    "constant,linear,message",
    [
        ([[1, 2]], [[1]], "square"),
        ([[1]], [[1, 2]], "square"),
        ([[1]], [[1, 0], [0, 1]], "same order"),
        ([1], [[1]], "rows must be sequences"),
    ],
)
def test_malformed_or_mismatched_matrices_are_rejected(
    constant, linear, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        native.affine_permanent_mod(constant, linear, 9)


def test_order_and_thread_validation():
    matrix = [[0] * 64 for _ in range(64)]
    with pytest.raises(ValueError, match="less than 64"):
        native.affine_permanent_mod(matrix, matrix, 9)
    with pytest.raises(ValueError, match="num_threads"):
        native.affine_permanent_mod([], [], 9, -1)
    with pytest.raises(ValueError, match="num_threads"):
        native.affine_permanent_mod([], [], 9, 1 << 31)
    with pytest.raises(TypeError, match="num_threads"):
        native.affine_permanent_mod([], [], 9, True)
    with pytest.raises(TypeError, match="num_threads"):
        native.affine_permanent_mod([], [], 9, 1.5)
