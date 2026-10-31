"""Tests for affine polynomial permanents modulo an odd integer."""

import hashlib
import itertools
import math
import operator
import random

import numpy as np
import pytest

import perm_polynomial


def _multiply(left, right, modulus):
    result = [0] * (len(left) + len(right) - 1)
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            result[left_degree + right_degree] = (
                result[left_degree + right_degree]
                + left_value * right_value
            ) % modulus
    return result


def _permutation_oracle(constant, linear, modulus):
    """Definition-level polynomial permanent, independent of Glynn."""
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
        for degree, coefficient in enumerate(product):
            result[degree] = (result[degree] + coefficient) % modulus
    return tuple(result)


def _permutation_oracle_exact(constant, linear):
    """Definition-level exact polynomial permanent, independent of CRT."""
    order = len(constant)
    result = [0] * (order + 1)
    for permutation in itertools.permutations(range(order)):
        product = [1]
        for row, column in enumerate(permutation):
            constant_entry = operator.index(constant[row][column])
            linear_entry = operator.index(linear[row][column])
            updated = [0] * (len(product) + 1)
            for degree, coefficient in enumerate(product):
                updated[degree] += coefficient * constant_entry
                updated[degree + 1] += coefficient * linear_entry
            product = updated
        for degree, coefficient in enumerate(product):
            result[degree] += coefficient
    return tuple(result)


def _rook_polynomial_oracle(matrix):
    """Count all partial matchings directly, independently of permanents."""
    order = len(matrix)
    result = [0] * (order + 1)

    def visit(row, used_columns, size):
        if row == order:
            result[size] += 1
            return

        visit(row + 1, used_columns, size)
        for column, entry in enumerate(matrix[row]):
            if entry and not (used_columns >> column) & 1:
                visit(row + 1, used_columns | (1 << column), size + 1)

    visit(0, 0, 0)
    return tuple(result)


def _weighted_rook_polynomial_oracle(matrix):
    """Sum matching-weight products directly, without permanents or CRT."""
    order = len(matrix)
    result = [0] * (order + 1)

    def visit(row, used_columns, size, product):
        if row == order:
            result[size] += product
            return

        visit(row + 1, used_columns, size, product)
        for column, entry in enumerate(matrix[row]):
            if entry and not (used_columns >> column) & 1:
                visit(
                    row + 1,
                    used_columns | (1 << column),
                    size + 1,
                    product * entry,
                )

    visit(0, 0, 0, 1)
    return tuple(result)


def _numeric_permanent(matrix, modulus):
    total = 0
    for permutation in itertools.permutations(range(len(matrix))):
        product = 1
        for row, column in enumerate(permutation):
            product = product * matrix[row][column] % modulus
        total = (total + product) % modulus
    return total


def _evaluate(coefficients, value, modulus):
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * value + coefficient) % modulus
    return result


@pytest.mark.parametrize("modulus", [3, 9, 15, 97])
def test_python_backend_matches_definition_random_small_matrices(modulus):
    rng = np.random.default_rng(20260817 + modulus)
    for order in range(7):
        trials = 8 if order < 6 else 3
        for _ in range(trials):
            constant = rng.integers(-20, 21, size=(order, order)).tolist()
            linear = rng.integers(-20, 21, size=(order, order)).tolist()
            expected = _permutation_oracle(constant, linear, modulus)
            assert perm_polynomial.affine_permanent_mod(
                constant, linear, modulus, backend="python"
            ) == expected


def test_exact_backend_matches_definition_random_signed_matrices():
    rng = np.random.default_rng(20260818)
    for order in range(7):
        trials = 7 if order < 6 else 3
        for _ in range(trials):
            constant = rng.integers(-9, 10, size=(order, order)).tolist()
            linear = rng.integers(-9, 10, size=(order, order)).tolist()
            expected = _permutation_oracle_exact(constant, linear)
            assert perm_polynomial.affine_permanent(
                constant, linear, backend="python"
            ) == expected


def test_exact_backend_handles_large_signed_integers_across_many_primes():
    huge = 10**50 + 151
    constant = [[huge, 0], [0, -huge]]
    linear = [[0, huge + 2], [huge - 3, 0]]
    expected = (-huge * huge, 0, (huge + 2) * (huge - 3))
    assert perm_polynomial.affine_permanent(
        constant, linear, backend="python"
    ) == expected


def test_exact_nonnegative_result_and_trailing_zero_coefficients():
    constant = [[3, 4], [5, 6]]
    linear = [[0, 0], [0, 0]]
    assert perm_polynomial.affine_permanent(
        constant, linear, backend="python"
    ) == (38, 0, 0)


def test_principal_permanent_polynomial_known_cases_and_empty_convention():
    assert perm_polynomial.principal_permanent_polynomial(
        [], backend="python"
    ) == (1,)
    assert perm_polynomial.principal_permanent_polynomial(
        [[7]], backend="python"
    ) == (1, 7)
    assert perm_polynomial.principal_permanent_polynomial(
        [[2, -3], [5, 7]], backend="python"
    ) == (1, 9, -1)
    assert perm_polynomial.principal_permanent_polynomial(
        [[2, 0, 0], [0, 3, 0], [0, 0, 5]], backend="python"
    ) == (1, 10, 31, 30)


def test_principal_permanent_polynomial_mod_known_cases_and_empty_convention():
    assert perm_polynomial.principal_permanent_polynomial_mod(
        [], 9, backend="python"
    ) == (1,)
    assert perm_polynomial.principal_permanent_polynomial_mod(
        [[14]], 9, backend="python"
    ) == (1, 5)
    assert perm_polynomial.principal_permanent_polynomial_mod(
        [[2, -3], [5, 7]], 101, backend="python"
    ) == (1, 9, 100)
    assert perm_polynomial.principal_permanent_polynomial_mod(
        [[2, 0, 0], [0, 3, 0], [0, 0, 5]], 25, backend="python"
    ) == (1, 10, 6, 5)


def test_principal_permanent_polynomial_mod_matches_definition_nonsymmetric():
    rng = random.Random(20260901)
    for order in range(6):
        identity = [
            [int(row == column) for column in range(order)]
            for row in range(order)
        ]
        for modulus in (9, 25, 101):
            for _ in range(5):
                matrix = [
                    [rng.randrange(-9, 10) for _ in range(order)]
                    for _ in range(order)
                ]
                expected = _permutation_oracle(identity, matrix, modulus)
                assert perm_polynomial.principal_permanent_polynomial_mod(
                    matrix, modulus, backend="python"
                ) == expected


def test_principal_permanent_polynomial_mod_uses_direct_python_projection(
    monkeypatch,
):
    def fail_generic(*args, **kwargs):
        raise AssertionError("generic affine path should not run")

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_glynn", fail_generic
    )
    assert perm_polynomial.principal_permanent_polynomial_mod(
        [[2, -3], [5, 7]], 101, backend="python"
    ) == (1, 9, 100)


def test_principal_permanent_polynomial_matches_affine_identity():
    rng = random.Random(20260831)
    for order in range(7):
        identity = [
            [int(row == column) for column in range(order)]
            for row in range(order)
        ]
        for _ in range(6):
            matrix = [
                [rng.randrange(-7, 8) for _ in range(order)]
                for _ in range(order)
            ]
            assert perm_polynomial.principal_permanent_polynomial(
                matrix, backend="python"
            ) == perm_polynomial.affine_permanent(
                identity, matrix, backend="python"
            )


@pytest.mark.parametrize("parallel", [0, 1, None, "yes"])
def test_principal_permanent_polynomial_parallel_flag_is_boolean(parallel):
    with pytest.raises(TypeError, match="parallel"):
        perm_polynomial.principal_permanent_polynomial(
            [], backend="python", parallel=parallel
        )


@pytest.mark.parametrize("num_threads", [-1, 1 << 31, 1.5, True])
def test_principal_permanent_polynomial_thread_count_is_validated(
    num_threads,
):
    error = ValueError if isinstance(num_threads, int) and not isinstance(
        num_threads, bool
    ) else TypeError
    with pytest.raises(error, match="num_threads"):
        perm_polynomial.principal_permanent_polynomial(
            [], backend="python", num_threads=num_threads
        )


@pytest.mark.parametrize("modulus", [-3, 0, 1, 2, 4, 2.0, True])
def test_principal_permanent_polynomial_mod_validates_modulus(modulus):
    error = TypeError if isinstance(modulus, (float, bool)) else ValueError
    with pytest.raises(error, match="modulus"):
        perm_polynomial.principal_permanent_polynomial_mod([], modulus)


def test_principal_permanent_polynomial_mod_validates_modulus_width():
    maximum = (1 << 63) - 1
    assert perm_polynomial.principal_permanent_polynomial_mod(
        [], maximum
    ) == (1,)
    with pytest.raises(ValueError, match="signed 64-bit"):
        perm_polynomial.principal_permanent_polynomial_mod(
            [], maximum + 2
        )


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[1, 2]], "square"),
        ([1], "rows must be sequences"),
        ([[1.0]], "entries must be integers"),
    ],
)
def test_principal_permanent_polynomial_mod_validates_matrix(matrix, message):
    with pytest.raises((TypeError, ValueError), match=message):
        perm_polynomial.principal_permanent_polynomial_mod(matrix, 9)


@pytest.mark.parametrize("parallel", [0, 1, None, "yes"])
def test_principal_permanent_polynomial_mod_parallel_flag_is_boolean(parallel):
    with pytest.raises(TypeError, match="parallel"):
        perm_polynomial.principal_permanent_polynomial_mod(
            [], 9, backend="python", parallel=parallel
        )


@pytest.mark.parametrize("num_threads", [-1, 1 << 31, 1.5, True])
def test_principal_permanent_polynomial_mod_thread_count_is_validated(
    num_threads,
):
    error = ValueError if isinstance(num_threads, int) and not isinstance(
        num_threads, bool
    ) else TypeError
    with pytest.raises(error, match="num_threads"):
        perm_polynomial.principal_permanent_polynomial_mod(
            [], 9, backend="python", num_threads=num_threads
        )


def test_rook_polynomial_known_cases_and_empty_convention():
    assert perm_polynomial.rook_polynomial([], backend="python") == (1,)
    assert perm_polynomial.rook_polynomial(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]], backend="python"
    ) == (1, 0, 0, 0)
    assert perm_polynomial.rook_polynomial(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], backend="python"
    ) == (1, 3, 3, 1)
    assert perm_polynomial.rook_polynomial(
        [[1, 1, 1, 1]] * 4, backend="python"
    ) == (1, 16, 72, 96, 24)


def test_rook_polynomial_zero_matrix_short_circuits(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("zero matrix should not start an enumeration")

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_mod_validated", fail
    )
    assert perm_polynomial.rook_polynomial(
        [[0] * 63 for _ in range(63)], backend="python"
    ) == (1,) + (0,) * 63


def test_rook_polynomial_executes_multi_prime_crt(monkeypatch):
    order = 12
    matrix = [[1] * order for _ in range(order)]
    expected = tuple(
        math.comb(order, size) ** 2 * math.factorial(size)
        for size in range(order + 1)
    )
    calls = []
    original = perm_polynomial._affine_permanent_mod_validated

    def tracked(constant, linear, modulus, selected, num_threads):
        calls.append(modulus)
        return original(constant, linear, modulus, selected, num_threads)

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_mod_validated", tracked
    )
    assert perm_polynomial.rook_polynomial(
        matrix, backend="python"
    ) == expected
    assert len(calls) == 2
    assert calls == sorted(calls, reverse=True)


def test_rook_polynomial_matches_direct_random_matching_counts():
    rng = np.random.default_rng(20260819)
    for order in range(8):
        trials = 8 if order < 7 else 4
        for _ in range(trials):
            matrix = rng.integers(0, 2, size=(order, order)).tolist()
            expected = _rook_polynomial_oracle(matrix)
            assert perm_polynomial.rook_polynomial(
                matrix, backend="python"
            ) == expected
            bounds = perm_polynomial._rook_polynomial_bounds_validated(
                matrix
            )
            assert all(
                value <= bound for value, bound in zip(expected, bounds)
            )
            assert perm_polynomial._rook_polynomial_parities_validated(
                matrix
            ) == tuple(value % 2 for value in expected)


def test_weighted_rook_polynomial_known_cases_and_empty_convention():
    assert perm_polynomial.weighted_rook_polynomial(
        [], backend="python"
    ) == (1,)
    assert perm_polynomial.weighted_rook_polynomial(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]], backend="python"
    ) == (1, 0, 0, 0)
    assert perm_polynomial.weighted_rook_polynomial(
        [[2, 0, 0], [0, 3, 0], [0, 0, 5]], backend="python"
    ) == (1, 10, 31, 30)
    assert perm_polynomial.weighted_rook_polynomial(
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]], backend="python"
    ) == (1, 18, 72, 48)
    huge = 10**100 + 7
    assert perm_polynomial.weighted_rook_polynomial(
        [[huge]], backend="python"
    ) == (1, huge)


def test_weighted_rook_polynomial_zero_matrix_short_circuits(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("zero matrix should not start an enumeration")

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_mod_validated", fail
    )
    assert perm_polynomial.weighted_rook_polynomial(
        [[0] * 63 for _ in range(63)], backend="python"
    ) == (1,) + (0,) * 63


def test_weighted_rook_polynomial_matches_direct_random_oracle():
    rng = np.random.default_rng(20260825)
    for order in range(7):
        trials = 8 if order < 6 else 4
        for _ in range(trials):
            matrix = rng.integers(0, 7, size=(order, order)).tolist()
            expected = _weighted_rook_polynomial_oracle(matrix)
            observed = perm_polynomial.weighted_rook_polynomial(
                matrix, backend="python"
            )
            assert observed == expected
            bounds = (
                perm_polynomial._weighted_rook_polynomial_bounds_validated(
                    matrix
                )
            )
            assert all(
                value <= bound for value, bound in zip(expected, bounds)
            )
            assert (
                perm_polynomial._weighted_rook_polynomial_parities_validated(
                    matrix
                )
                == tuple(value % 2 for value in expected)
            )


def test_weighted_rook_parity_reduces_weights_modulo_two():
    matrix = ((2, 0), (0, 1))
    assert perm_polynomial._weighted_rook_polynomial_parities_validated(
        matrix
    ) == (1, 1, 0)
    assert perm_polynomial.weighted_rook_polynomial(
        matrix, backend="python"
    ) == (1, 3, 2)


def test_weighted_rook_polynomial_executes_multi_prime_crt(monkeypatch):
    order = 4
    weight = 10**12
    matrix = [[weight] * order for _ in range(order)]
    expected = tuple(
        math.comb(order, size) ** 2
        * math.factorial(size)
        * pow(weight, size)
        for size in range(order + 1)
    )
    calls = []
    original = perm_polynomial._affine_permanent_mod_validated

    def tracked(constant, linear, modulus, selected, num_threads):
        calls.append(modulus)
        return original(constant, linear, modulus, selected, num_threads)

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_mod_validated", tracked
    )
    assert perm_polynomial.weighted_rook_polynomial(
        matrix, backend="python"
    ) == expected
    assert len(calls) > 1
    assert calls == sorted(calls, reverse=True)


def test_weighted_rook_parity_selects_second_crt_lift(monkeypatch):
    monkeypatch.setattr(
        perm_polynomial,
        "_plan_crt_primes",
        lambda bound, *, signed: (11,),
    )
    # c_2=13 has canonical residue 2 modulo 11.  Its odd parity selects
    # 2+11; the certified degree-two bound is 18, below 2*11.
    assert perm_polynomial.weighted_rook_polynomial(
        [[2, 3], [3, 2]], backend="python"
    ) == (1, 10, 13)


def test_rook_parity_certificate_supports_maximum_order():
    order = 63
    matrix = tuple((1,) * order for _ in range(order))
    # r_0 and r_1=63**2 are odd; k! makes every later coefficient even.
    assert perm_polynomial._rook_polynomial_parities_validated(
        matrix
    ) == (1, 1) + (0,) * (order - 1)


def test_rook_parity_can_determine_unit_bound_without_crt(
    monkeypatch,
):
    def fail(*args, **kwargs):
        raise AssertionError("parity and a unit bound need no CRT traversal")

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_mod_validated", fail
    )
    order = 3
    matrix = [[0] * order for _ in range(order)]
    matrix[0][0] = 1
    assert perm_polynomial.rook_polynomial(
        matrix, backend="python"
    ) == (1, 1) + (0,) * (order - 1)


def test_rook_reconstruction_uses_parity_to_select_second_crt_lift(
    monkeypatch,
):
    monkeypatch.setattr(
        perm_polynomial,
        "_plan_crt_primes",
        lambda bound, *, signed: (11,),
    )
    matrix = [[1, 1, 1] for _ in range(3)]
    # r_2=18 has canonical residue 7 modulo 11.  Its even parity selects
    # 7+11, while ordinary unsigned reconstruction would return 7.
    assert perm_polynomial.rook_polynomial(
        matrix, backend="python"
    ) == (1, 9, 18, 6)


def test_rook_factorial_roots_are_exact_scaled_ceilings():
    scale = 1 << 32
    for degree in range(1, 64):
        root = perm_polynomial._ceil_scaled_factorial_root(
            degree, scale
        )
        target = math.factorial(degree) * pow(scale, degree)
        assert pow(root, degree) >= target
        assert pow(root - 1, degree) < target


@pytest.mark.parametrize("order", [1, 2, 7, 33, 63])
def test_rook_dummy_factors_are_exact_scaled_ceilings(order):
    scale = 1 << 32
    for omitted_rows in range(order + 1):
        factor = perm_polynomial._ceil_scaled_dummy_factor(
            order, omitted_rows, scale
        )
        omitted_factorial = math.factorial(omitted_rows)
        target = (
            pow(scale, order)
            * pow(math.factorial(order), omitted_rows)
        )
        assert pow(factor * omitted_factorial, order) >= target
        assert pow(
            (factor - 1) * omitted_factorial, order
        ) < target


def test_tight_rook_bound_and_parity_keep_sha33_to_four_crt_primes():
    label = "perm-llm-rook33-v1/0"
    matrix = []
    for row in range(33):
        normalized_row = []
        for column in range(33):
            payload = f"{label}/e/{row}/{column}".encode("ascii")
            digest = hashlib.sha256(payload).digest()
            normalized_row.append(
                int.from_bytes(digest[:2], "big") < 32768
            )
        matrix.append(normalized_row)

    maximum = max(
        perm_polynomial._rook_polynomial_bounds_validated(matrix)
    )
    assert maximum == 615817222118560083916977687304187
    assert maximum.bit_length() == 109
    assert len(
        perm_polynomial._plan_crt_primes(maximum // 2, signed=False)
    ) == 4
    maximum_degree = min(
        max(map(sum, matrix)),
        max(sum(matrix[row][column] for row in range(33))
            for column in range(33)),
    )
    assert perm_polynomial._plan_rook_pseudo_mersenne_primes(
        maximum // 2, order=33, maximum_degree=maximum_degree
    ) == (134217689, 134217649, 134217617, 134217613)


def test_tight_rook_bound_sha37_still_needs_five_crt_primes():
    label = "perm-llm-rook37-v1/0"
    matrix = []
    for row in range(37):
        normalized_row = []
        for column in range(37):
            payload = f"{label}/e/{row}/{column}".encode("ascii")
            digest = hashlib.sha256(payload).digest()
            normalized_row.append(
                int.from_bytes(digest[:2], "big") < 32768
            )
        matrix.append(normalized_row)

    maximum = max(
        perm_polynomial._rook_polynomial_bounds_validated(matrix)
    )
    assert sum(map(sum, matrix)) == 683
    assert maximum == 94493725960315786062474345776983531813
    assert maximum.bit_length() == 127
    assert len(
        perm_polynomial._plan_crt_primes(maximum // 2, signed=False)
    ) == 5
    maximum_degree = min(
        max(map(sum, matrix)),
        max(sum(matrix[row][column] for row in range(37))
            for column in range(37)),
    )
    assert maximum_degree == 26
    assert perm_polynomial._plan_rook_pseudo_mersenne_primes(
        maximum // 2, order=37, maximum_degree=maximum_degree
    ) == (
        134217689,
        134217649,
        134217617,
        134217613,
        134217593,
    )


@pytest.mark.parametrize(
    ("maximum_degree", "modulus_bits"),
    [
        (0, 31),
        (1, 31),
        (2, 30),
        (3, 30),
        (4, 29),
        (7, 29),
        (8, 28),
        (15, 28),
        (16, 27),
        (31, 27),
        (32, 26),
        (63, 26),
    ],
)
def test_rook_pseudo_mersenne_planner_selects_largest_safe_width(
    maximum_degree, modulus_bits
):
    target = 10**20
    primes = perm_polynomial._plan_rook_pseudo_mersenne_primes(
        target, order=63, maximum_degree=maximum_degree
    )
    assert math.prod(primes) > target
    assert {prime.bit_length() for prime in primes} == {modulus_bits}
    for prime in primes:
        complement = (1 << modulus_bits) - prime
        assert (maximum_degree + 1) * prime <= 1 << 32
        assert (maximum_degree + 2) * complement < 1 << modulus_bits


@pytest.mark.parametrize(
    ("maximum_signed_sum", "modulus_bits"),
    [
        (0, 31),
        (1, 31),
        (2, 30),
        (3, 30),
        (4, 29),
        (160, 24),
        (1000, 22),
    ],
)
def test_weighted_rook_planner_selects_largest_safe_width(
    maximum_signed_sum, modulus_bits
):
    target = 10**20
    primes = (
        perm_polynomial._plan_weighted_rook_pseudo_mersenne_primes(
            target,
            order=63,
            maximum_signed_sum=maximum_signed_sum,
        )
    )
    assert math.prod(primes) > target
    assert {prime.bit_length() for prime in primes} == {modulus_bits}
    for prime in primes:
        complement = (1 << modulus_bits) - prime
        assert (
            (maximum_signed_sum + 1) * prime <= 1 << 32
        )
        assert (
            (maximum_signed_sum + 2) * complement
            < 1 << modulus_bits
        )


def test_rook_polynomial_accepts_integral_zeroes_ones_and_booleans():
    matrix = [[True, np.int64(1)], [np.int32(1), False]]
    assert perm_polynomial.rook_polynomial(
        matrix, backend="python"
    ) == (1, 3, 1)


@pytest.mark.parametrize("entry", [1.0, "1", None, 2 + 0j])
def test_rook_polynomial_rejects_nonintegral_entries(entry):
    with pytest.raises(TypeError, match="entries must be integers"):
        perm_polynomial.rook_polynomial([[entry]], backend="python")


@pytest.mark.parametrize("entry", [-2, -1, 2, 99])
def test_rook_polynomial_strictly_rejects_nonbinary_integers(entry):
    with pytest.raises(ValueError, match="entries must be 0 or 1"):
        perm_polynomial.rook_polynomial([[entry]], backend="python")


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[1, 0]], "square"),
        (np.array([1, 0]), "two-dimensional"),
        ([1], "rows must be sequences"),
    ],
)
def test_rook_polynomial_rejects_malformed_shapes(matrix, message):
    with pytest.raises((TypeError, ValueError), match=message):
        perm_polynomial.rook_polynomial(matrix, backend="python")


def test_rook_polynomial_preserves_hostile_sequence_snapshot_checks():
    class ShortRow:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([1])

    with pytest.raises(ValueError, match="row width changed"):
        perm_polynomial.rook_polynomial(
            [ShortRow(), ShortRow()], backend="python"
        )

    class ShortMatrix:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([[1, 0]])

    with pytest.raises(ValueError, match="row count changed"):
        perm_polynomial.rook_polynomial(
            ShortMatrix(), backend="python"
        )


def test_rook_polynomial_checks_reconstruction_against_bound(monkeypatch):
    monkeypatch.setattr(
        perm_polynomial,
        "_affine_permanent_mod_validated",
        lambda *args, **kwargs: (100, 100, 100),
    )
    with pytest.raises(RuntimeError, match="outside.*rook bound"):
        perm_polynomial.rook_polynomial(
            [[1, 0], [0, 1]], backend="python"
        )


def test_weighted_rook_accepts_integral_entries_and_booleans():
    matrix = [[True, np.int64(2)], [np.int32(3), False]]
    assert perm_polynomial.weighted_rook_polynomial(
        matrix, backend="python"
    ) == (1, 6, 6)


@pytest.mark.parametrize("entry", [1.0, "1", None, 2 + 0j])
def test_weighted_rook_rejects_nonintegral_entries(entry):
    with pytest.raises(TypeError, match="entries must be integers"):
        perm_polynomial.weighted_rook_polynomial(
            [[entry]], backend="python"
        )


@pytest.mark.parametrize("entry", [-1, -10, np.int64(-2)])
def test_weighted_rook_rejects_negative_entries(entry):
    with pytest.raises(ValueError, match="entries must be nonnegative"):
        perm_polynomial.weighted_rook_polynomial(
            [[entry]], backend="python"
        )


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[1, 2]], "square"),
        (np.array([1, 2]), "two-dimensional"),
        ([1], "rows must be sequences"),
    ],
)
def test_weighted_rook_rejects_malformed_shapes(matrix, message):
    with pytest.raises((TypeError, ValueError), match=message):
        perm_polynomial.weighted_rook_polynomial(
            matrix, backend="python"
        )


def test_weighted_rook_preserves_hostile_sequence_snapshot_checks():
    class ShortRow:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([2])

    with pytest.raises(ValueError, match="row width changed"):
        perm_polynomial.weighted_rook_polynomial(
            [ShortRow(), ShortRow()], backend="python"
        )

    class ShortMatrix:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([[1, 2]])

    with pytest.raises(ValueError, match="row count changed"):
        perm_polynomial.weighted_rook_polynomial(
            ShortMatrix(), backend="python"
        )


def test_weighted_rook_checks_reconstruction_against_bound(monkeypatch):
    monkeypatch.setattr(
        perm_polynomial,
        "_affine_permanent_mod_validated",
        lambda *args, **kwargs: (100, 100, 100),
    )
    with pytest.raises(RuntimeError, match="outside.*weighted-rook bound"):
        perm_polynomial.weighted_rook_polynomial(
            # Connected support keeps this a CRT-bound rejection test;
            # disconnected diagonal inputs now have exact direct factors.
            [[2, 1], [0, 3]], backend="python"
        )


def test_coefficient_bounds_cover_random_definition_level_results():
    rng = np.random.default_rng(987654321)
    for order in range(7):
        for _ in range(5):
            constant = rng.integers(-12, 13, size=(order, order)).tolist()
            linear = rng.integers(-12, 13, size=(order, order)).tolist()
            exact = _permutation_oracle_exact(constant, linear)
            bounds = perm_polynomial.affine_permanent_bounds(
                constant, linear
            )
            assert len(bounds) == order + 1
            assert all(
                abs(value) <= bound
                for value, bound in zip(exact, bounds)
            )


@pytest.mark.parametrize(
    ("constant", "linear", "degree", "candidate_values", "winner"),
    [
        (
            [[4, 9, 9], [7, 7, 2], [1, 7, 7]],
            [[1, 9, 7], [0, 0, 0], [1, 6, 1]],
            2,
            (2176, 2348, 2268, 3510),
            0,
        ),
        (
            [[1, 8, 6], [0, 0, 5], [5, 5, 4]],
            [[3, 5, 3], [4, 4, 5], [0, 7, 1]],
            1,
            (4100, 3507, 3630, 3612),
            1,
        ),
        (
            [[4, 9, 9], [7, 7, 2], [1, 7, 7]],
            [[1, 9, 7], [0, 0, 0], [1, 6, 1]],
            0,
            (5280, 4968, 2646, 3402),
            2,
        ),
        (
            [[1, 8, 6], [0, 0, 5], [5, 5, 4]],
            [[3, 5, 3], [4, 4, 5], [0, 7, 1]],
            2,
            (4002, 3363, 3480, 3018),
            3,
        ),
    ],
)
def test_each_coefficient_bound_relaxation_can_win_uniquely(
    constant, linear, degree, candidate_values, winner
):
    # Candidate order: row sums, column sums, row maxima, column maxima.
    assert candidate_values[winner] == min(candidate_values)
    assert candidate_values.count(candidate_values[winner]) == 1
    assert perm_polynomial.affine_permanent_bounds(
        constant, linear
    )[degree] == candidate_values[winner]


def test_zero_bound_skips_every_modular_traversal(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a certified zero needs no CRT traversal")

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_mod_validated", fail
    )
    zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert perm_polynomial.affine_permanent(
        zero, zero, backend="python"
    ) == (0, 0, 0, 0)


def test_deterministic_crt_prime_planner_has_strict_coverage():
    first = 2_147_483_647
    assert perm_polynomial._plan_crt_primes(
        (first - 1) // 2, signed=True
    ) == (first,)
    signed_two = perm_polynomial._plan_crt_primes(
        (first + 1) // 2, signed=True
    )
    unsigned_two = perm_polynomial._plan_crt_primes(
        first, signed=False
    )
    assert signed_two == unsigned_two == (first, 2_147_483_629)
    assert first * 2_147_483_629 > 2 * ((first + 1) // 2)


def test_exact_unsigned_crt_crosses_first_prime_boundary(monkeypatch):
    first = 2_147_483_647
    calls = []
    original = perm_polynomial._affine_permanent_mod_validated

    def recording_residue(constant, linear, modulus, backend, threads):
        calls.append(modulus)
        return original(constant, linear, modulus, backend, threads)

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_mod_validated", recording_residue
    )
    assert perm_polynomial.affine_permanent(
        [[first, 0], [0, 1]],
        [[0, 0], [0, 0]],
        backend="python",
    ) == (first, 0, 0)
    assert calls == [first, 2_147_483_629]


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_signed_crt_maps_centered_boundary(sign, monkeypatch):
    first = 2_147_483_647
    midpoint = first // 2
    calls = []
    original = perm_polynomial._affine_permanent_mod_validated

    def recording_residue(constant, linear, modulus, backend, threads):
        calls.append(modulus)
        return original(constant, linear, modulus, backend, threads)

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_mod_validated", recording_residue
    )
    diagonal = [[-midpoint, 0], [0, -sign]]
    assert perm_polynomial.affine_permanent(
        diagonal,
        [[0, 0], [0, 0]],
        backend="python",
    ) == (sign * midpoint, 0, 0)
    assert calls == [first]


def test_exact_api_uses_unsigned_crt_only_for_nonnegative_inputs(
    monkeypatch,
):
    calls = []
    original = perm_polynomial._plan_crt_primes

    def recording_planner(bound, *, signed):
        calls.append((bound, signed))
        return original(bound, signed=signed)

    monkeypatch.setattr(
        perm_polynomial, "_plan_crt_primes", recording_planner
    )
    identity = [[1, 0], [0, 1]]
    off_diagonal = [[0, 1], [1, 0]]
    assert perm_polynomial.affine_permanent(
        identity, off_diagonal, backend="python"
    ) == (1, 0, 1)
    assert calls[-1][1] is False

    signed_constant = [[-1, 0], [0, 1]]
    assert perm_polynomial.affine_permanent(
        signed_constant, off_diagonal, backend="python"
    ) == (-1, 0, 1)
    assert calls[-1][1] is True


def test_exact_crt_result_is_checked_against_its_bound(monkeypatch):
    monkeypatch.setattr(
        perm_polynomial,
        "_affine_permanent_mod_validated",
        lambda *args, **kwargs: (100, 100, 100),
    )
    identity = [[1, 0], [0, 1]]
    off_diagonal = [[0, 1], [1, 0]]
    with pytest.raises(RuntimeError, match="outside.*bound"):
        perm_polynomial.affine_permanent(
            identity, off_diagonal, backend="python"
        )


def test_evaluation_matches_definition_level_numeric_permanents():
    constant = [[2, -1, 5], [7, 0, 3], [-4, 6, 1]]
    linear = [[1, 4, -2], [0, 3, 8], [5, -7, 2]]
    for modulus in (9, 25, 101):
        coefficients = perm_polynomial.affine_permanent_mod(
            constant, linear, modulus, backend="python"
        )
        for value in (-5, 0, 1, 2, modulus + 3):
            numeric = [
                [
                    (constant[row][column] + value * linear[row][column])
                    % modulus
                    for column in range(3)
                ]
                for row in range(3)
            ]
            assert _evaluate(coefficients, value, modulus) == (
                _numeric_permanent(numeric, modulus)
            )


def test_python_identity_constant_uses_principal_sign_projection(
    monkeypatch,
):
    identity = [[1, 0], [0, 1]]
    matrix = [[2, -3], [5, 7]]

    def fail_generic(*args, **kwargs):
        raise AssertionError("generic affine path should not run")

    monkeypatch.setattr(
        perm_polynomial, "_affine_permanent_glynn", fail_generic
    )
    assert perm_polynomial.affine_permanent_mod(
        identity, matrix, 101, backend="python"
    ) == (1, 9, 100)


def test_principal_sign_projection_matches_generic_affine_glynn():
    rng = random.Random(20260830)
    for order in range(2, 7):
        identity = tuple(
            tuple(int(row == column) for column in range(order))
            for row in range(order)
        )
        for modulus in (9, 25, 101):
            for _ in range(8):
                matrix = tuple(
                    tuple(rng.randrange(-8, 9) for _ in range(order))
                    for _ in range(order)
                )
                reduced = tuple(
                    tuple(value % modulus for value in row)
                    for row in matrix
                )
                assert perm_polynomial._principal_permanent_glynn(
                    reduced, modulus
                ) == perm_polynomial._affine_permanent_glynn(
                    identity, reduced, modulus
                )


def test_empty_and_one_by_one_conventions_and_trailing_zeroes():
    assert perm_polynomial.affine_permanent_mod(
        [], [], 9, backend="python"
    ) == (1,)
    assert perm_polynomial.affine_permanent_mod(
        [[-2]], [[14]], 9, backend="python"
    ) == (7, 5)
    assert perm_polynomial.affine_permanent_mod(
        [[1, 2], [3, 4]], [[0, 0], [0, 0]], 9, backend="python"
    ) == (1, 0, 0)


def test_integral_entries_are_normalized_and_booleans_are_allowed():
    huge = 10**100
    constant = [[True, huge], [np.int64(-4), False]]
    linear = [[False, -huge], [True, np.int32(7)]]
    modulus = 15
    expected = _permutation_oracle(constant, linear, modulus)
    assert perm_polynomial.affine_permanent_mod(
        constant, linear, modulus, backend="python"
    ) == expected


@pytest.mark.parametrize("entry", [1.0, "1", None, 2 + 0j])
def test_nonintegral_matrix_entries_are_rejected(entry):
    with pytest.raises(TypeError, match="entries must be integers"):
        perm_polynomial.affine_permanent_mod(
            [[entry]], [[0]], 9, backend="python"
        )
    with pytest.raises(TypeError, match="entries must be integers"):
        perm_polynomial.affine_permanent_mod(
            [[0]], [[entry]], 9, backend="python"
        )


@pytest.mark.parametrize(
    "constant,linear,message",
    [
        ([[1, 2]], [[1]], "square"),
        ([[1]], [[1, 2]], "square"),
        ([[1]], [[1, 0], [0, 1]], "same shape"),
        (np.array([1, 2]), np.array([1, 2]), "two-dimensional"),
        ([1], [[1]], "rows must be sequences"),
    ],
)
def test_malformed_or_mismatched_shapes_are_rejected(
    constant, linear, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        perm_polynomial.affine_permanent_mod(
            constant, linear, 9, backend="python"
        )


def test_hostile_rows_and_outer_sequences_cannot_change_reported_sizes():
    class ShortRow:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([1])

    with pytest.raises(ValueError, match="row width changed"):
        perm_polynomial.affine_permanent_mod(
            [ShortRow(), ShortRow()], [[0, 0], [0, 0]], 9
        )

    class ShortMatrix:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([[1, 0]])

    with pytest.raises(ValueError, match="row count changed"):
        perm_polynomial.affine_permanent_mod(
            ShortMatrix(), [[0, 0], [0, 0]], 9
        )


def test_order_must_be_less_than_64():
    matrix = np.zeros((64, 64), dtype=np.int8)
    with pytest.raises(ValueError, match="less than 64"):
        perm_polynomial.affine_permanent_mod(matrix, matrix, 9)


@pytest.mark.parametrize("modulus", [-3, 0, 1, 2, 4, 10, 2.0, True])
def test_modulus_must_be_an_odd_integer_greater_than_one(modulus):
    error = TypeError if isinstance(modulus, (float, bool)) else ValueError
    with pytest.raises(error, match="modulus"):
        perm_polynomial.affine_permanent_mod([], [], modulus)


def test_modulus_must_fit_in_positive_signed_64_bit_integer():
    maximum = (1 << 63) - 1
    assert perm_polynomial.affine_permanent_mod([], [], maximum) == (1,)
    with pytest.raises(ValueError, match="signed 64-bit"):
        perm_polynomial.affine_permanent_mod([], [], maximum + 2)


@pytest.mark.parametrize("parallel", [0, 1, None, "yes"])
def test_parallel_flag_must_be_boolean(parallel):
    with pytest.raises(TypeError, match="parallel"):
        perm_polynomial.affine_permanent_mod([], [], 9, parallel=parallel)


@pytest.mark.parametrize("num_threads", [-1, 1 << 31, 1.5, True])
def test_invalid_thread_counts_are_rejected(num_threads):
    error = ValueError if isinstance(num_threads, int) and not isinstance(
        num_threads, bool
    ) else TypeError
    with pytest.raises(error, match="num_threads"):
        perm_polynomial.affine_permanent_mod(
            [], [], 9, num_threads=num_threads
        )


def test_backend_names_and_unavailable_native_backend(monkeypatch):
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        perm_polynomial.affine_permanent_mod([], [], 9, backend="native")
    with pytest.raises(ValueError, match="backend must be one of"):
        perm_polynomial.affine_permanent_mod([], [], 9, backend="cuda")
    with pytest.raises(TypeError, match="backend must be a string"):
        perm_polynomial.affine_permanent_mod([], [], 9, backend=None)


def test_principal_mod_backend_names_and_unavailable_native(monkeypatch):
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        perm_polynomial.principal_permanent_polynomial_mod(
            [], 9, backend="native"
        )
    with pytest.raises(ValueError, match="backend must be one of"):
        perm_polynomial.principal_permanent_polynomial_mod(
            [], 9, backend="cuda"
        )
    with pytest.raises(TypeError, match="backend must be a string"):
        perm_polynomial.principal_permanent_polynomial_mod(
            [], 9, backend=None
        )


def test_polynomial_backend_capabilities_are_stable_and_fresh(monkeypatch):
    class FakeNative:
        @staticmethod
        def affine_permanent_mod(*args):
            return ()

        @staticmethod
        def weighted_rook_polynomial_mod_many(*args):
            return ()

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    first = perm_polynomial.polynomial_backend_capabilities()
    assert first == {
        "available_backends": ("native", "python"),
        "default_backend": "native",
        "native_extension_available": True,
        "native_hooks": (
            "affine_permanent_mod",
            "weighted_rook_polynomial_mod_many",
        ),
    }

    first["native_hooks"] = ()
    assert perm_polynomial.polynomial_backend_capabilities()[
        "native_hooks"
    ] == (
        "affine_permanent_mod",
        "weighted_rook_polynomial_mod_many",
    )

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", None)
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", False)
    assert perm_polynomial.polynomial_backend_capabilities() == {
        "available_backends": ("python",),
        "default_backend": "python",
        "native_extension_available": False,
        "native_hooks": (),
    }


def test_fake_native_dispatch_normalizes_and_forwards_threads(monkeypatch):
    calls = []

    class FakeNative:
        @staticmethod
        def affine_permanent_mod(constant, linear, modulus, num_threads):
            calls.append((constant, linear, modulus, num_threads))
            return [4, 5, 6]

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    constant = [[-1, 20], [False, 3]]
    linear = [[10, True], [2, -4]]
    assert perm_polynomial.affine_permanent_mod(
        constant, linear, 9, backend="auto", num_threads=7
    ) == (4, 5, 6)
    assert calls == [
        (
            [[8, 2], [0, 3]],
            [[1, 1], [2, 5]],
            9,
            7,
        )
    ]

    calls.clear()
    assert perm_polynomial.affine_permanent_mod(
        constant,
        linear,
        9,
        backend="NATIVE",
        parallel=False,
        num_threads=99,
    ) == (4, 5, 6)
    assert calls[0][-1] == 1


def test_principal_mod_native_dispatch_normalizes_and_forwards_threads(
    monkeypatch,
):
    calls = []

    class FakeNative:
        @staticmethod
        def affine_permanent_mod(constant, linear, modulus, num_threads):
            calls.append((constant, linear, modulus, num_threads))
            return [1, 0, 8]

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    matrix = [[2, -3], [5, 7]]

    assert perm_polynomial.principal_permanent_polynomial_mod(
        matrix, 9, backend="auto", num_threads=7
    ) == (1, 0, 8)
    assert calls == [
        (
            [[1, 0], [0, 1]],
            [[2, 6], [5, 7]],
            9,
            7,
        )
    ]

    calls.clear()
    assert perm_polynomial.principal_permanent_polynomial_mod(
        matrix,
        9,
        backend="NATIVE",
        parallel=False,
        num_threads=99,
    ) == (1, 0, 8)
    assert calls[0][-1] == 1


def test_rook_polynomial_native_dispatch_and_thread_forwarding(monkeypatch):
    calls = []

    class FakeNative:
        @staticmethod
        def affine_permanent_mod(constant, linear, modulus, num_threads):
            calls.append((constant, linear, modulus, num_threads))
            return [2, 3, 1]

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    matrix = [[1, 1], [1, 0]]

    assert perm_polynomial.rook_polynomial(
        matrix, backend="auto", num_threads=7
    ) == (1, 3, 1)
    assert calls == [
        (
            [[1, 1], [1, 1]],
            [[1, 1], [1, 0]],
            2_147_483_647,
            7,
        )
    ]

    calls.clear()
    assert perm_polynomial.rook_polynomial(
        matrix,
        backend="NATIVE",
        parallel=False,
        num_threads=99,
    ) == (1, 3, 1)
    assert calls[0][-1] == 1


def test_rook_polynomial_uses_fused_native_residue_hook(monkeypatch):
    calls = []

    class FakeNative:
        @staticmethod
        def affine_permanent_mod(*args):
            raise AssertionError("fused rook hook should replace affine calls")

        @staticmethod
        def rook_polynomial_mod_many(matrix, moduli, num_threads):
            calls.append((matrix, moduli, num_threads))
            exact = (1, 3, 0, 0)
            return tuple(
                tuple(value % modulus for value in exact)
                for modulus in moduli
            )

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    matrix = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]

    assert perm_polynomial.rook_polynomial(
        matrix, backend="auto", num_threads=7
    ) == (1, 3, 0, 0)
    assert calls == [
        (
            matrix,
            (2_147_483_647,),
            7,
        )
    ]


def test_rook_polynomial_chunks_large_fused_prime_plans(monkeypatch):
    calls = []
    primes = perm_polynomial._plan_crt_primes(
        1 << (31 * 16), signed=False
    )
    assert len(primes) == 17

    class FakeNative:
        @staticmethod
        def affine_permanent_mod(*args):
            raise AssertionError("fused rook hook should replace affine calls")

        @staticmethod
        def rook_polynomial_mod_many(matrix, moduli, num_threads):
            calls.append(tuple(moduli))
            exact = (1, 3, 1)
            return tuple(
                tuple(value % modulus for value in exact)
                for modulus in moduli
            )

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    monkeypatch.setattr(
        perm_polynomial,
        "_plan_rook_pseudo_mersenne_primes",
        lambda *args, **kwargs: primes,
    )
    assert perm_polynomial.rook_polynomial(
        [[1, 1], [1, 0]], backend="native"
    ) == (1, 3, 1)
    assert calls == [primes[:16], primes[16:]]


@pytest.mark.parametrize(
    "invalid",
    ["outer", "short", "bool", "negative", "too_large", "noninteger"],
)
def test_fused_native_rook_residue_contract_is_checked(
    monkeypatch, invalid
):
    class FakeNative:
        @staticmethod
        def affine_permanent_mod(*args):
            raise AssertionError("fused rook hook should replace affine calls")

        @staticmethod
        def rook_polynomial_mod_many(matrix, moduli, num_threads):
            if invalid == "outer":
                return ()
            if invalid == "short":
                return ((1, 2),)
            if invalid == "bool":
                return ((1, False, 1),)
            if invalid == "negative":
                return ((1, -1, 1),)
            if invalid == "too_large":
                return ((1, moduli[0], 1),)
            return ((1, object(), 1),)

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    with pytest.raises(RuntimeError, match="invalid rook residues"):
        perm_polynomial.rook_polynomial(
            [[1, 1], [1, 0]], backend="native"
        )


def test_weighted_rook_uses_fused_native_residue_hook(monkeypatch):
    calls = []

    class FakeNative:
        @staticmethod
        def affine_permanent_mod(*args):
            raise AssertionError(
                "fused weighted-rook hook should replace affine calls"
            )

        @staticmethod
        def weighted_rook_polynomial_mod_many(
            matrix, moduli, num_threads
        ):
            calls.append((matrix, moduli, num_threads))
            exact = (1, 8, 7)
            return tuple(
                tuple(value % modulus for value in exact)
                for modulus in moduli
            )

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    matrix = [[np.int64(2), True], [3, 2]]

    assert perm_polynomial.weighted_rook_polynomial(
        matrix, backend="auto", num_threads=7
    ) == (1, 8, 7)
    assert len(calls) == 1
    assert calls[0][0] == [[2, 1], [3, 2]]
    assert calls[0][2] == 7
    assert math.prod(calls[0][1]) > max(
        perm_polynomial._weighted_rook_polynomial_bounds_validated(
            ((2, 1), (3, 2))
        )
    ) // 2

    calls.clear()
    assert perm_polynomial.weighted_rook_polynomial(
        matrix,
        backend="NATIVE",
        parallel=False,
        num_threads=99,
    ) == (1, 8, 7)
    assert calls[0][2] == 1


@pytest.mark.parametrize("planner_failure", [False, True])
def test_weighted_rook_native_falls_back_to_affine_when_ineligible(
    monkeypatch, planner_failure
):
    calls = []

    class FakeNative:
        @staticmethod
        def affine_permanent_mod(constant, linear, modulus, num_threads):
            calls.append((constant, linear, modulus, num_threads))
            return _permutation_oracle(constant, linear, modulus)

        @staticmethod
        def weighted_rook_polynomial_mod_many(*args):
            raise AssertionError("ineligible fused hook must not be called")

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    if planner_failure:
        matrix = [[2, 3], [3, 2]]

        def fail_plan(*args, **kwargs):
            raise RuntimeError("no safe common-width plan")

        monkeypatch.setattr(
            perm_polynomial,
            "_plan_weighted_rook_pseudo_mersenne_primes",
            fail_plan,
        )
        expected = (1, 10, 13)
    else:
        # Keep the support connected so structural factorization cannot
        # bypass the oversized-entry affine fallback exercised here.
        matrix = [[65536, 1], [0, 1]]
        expected = (1, 65538, 65536)

    assert perm_polynomial.weighted_rook_polynomial(
        matrix, backend="native", num_threads=6
    ) == expected
    assert calls
    assert all(call[-1] == 6 for call in calls)


def test_weighted_rook_chunks_large_fused_prime_plans(monkeypatch):
    calls = []
    primes = perm_polynomial._plan_crt_primes(
        1 << (31 * 16), signed=False
    )
    assert len(primes) == 17

    class FakeNative:
        @staticmethod
        def affine_permanent_mod(*args):
            raise AssertionError(
                "fused weighted-rook hook should replace affine calls"
            )

        @staticmethod
        def weighted_rook_polynomial_mod_many(
            matrix, moduli, num_threads
        ):
            calls.append(tuple(moduli))
            exact = (1, 10, 13)
            return tuple(
                tuple(value % modulus for value in exact)
                for modulus in moduli
            )

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    monkeypatch.setattr(
        perm_polynomial,
        "_plan_weighted_rook_pseudo_mersenne_primes",
        lambda *args, **kwargs: primes,
    )
    assert perm_polynomial.weighted_rook_polynomial(
        [[2, 3], [3, 2]], backend="native"
    ) == (1, 10, 13)
    assert calls == [primes[:16], primes[16:]]


@pytest.mark.parametrize(
    "invalid",
    ["outer", "short", "bool", "negative", "too_large", "noninteger"],
)
def test_fused_native_weighted_rook_residue_contract_is_checked(
    monkeypatch, invalid
):
    class FakeNative:
        @staticmethod
        def affine_permanent_mod(*args):
            raise AssertionError(
                "fused weighted-rook hook should replace affine calls"
            )

        @staticmethod
        def weighted_rook_polynomial_mod_many(
            matrix, moduli, num_threads
        ):
            if invalid == "outer":
                return ()
            if invalid == "short":
                return ((1, 2),)
            if invalid == "bool":
                return ((1, False, 1),)
            if invalid == "negative":
                return ((1, -1, 1),)
            if invalid == "too_large":
                return ((1, moduli[0], 1),)
            return ((1, object(), 1),)

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    with pytest.raises(RuntimeError, match="invalid weighted-rook residues"):
        perm_polynomial.weighted_rook_polynomial(
            [[2, 3], [3, 2]], backend="native"
        )


@pytest.mark.parametrize("parallel", [0, 1, None, "yes"])
def test_rook_polynomial_parallel_flag_must_be_boolean(parallel):
    with pytest.raises(TypeError, match="parallel"):
        perm_polynomial.rook_polynomial(
            [], backend="python", parallel=parallel
        )


@pytest.mark.parametrize("num_threads", [-1, 1 << 31, 1.5, True])
def test_rook_polynomial_invalid_thread_counts_are_rejected(num_threads):
    error = ValueError if isinstance(num_threads, int) and not isinstance(
        num_threads, bool
    ) else TypeError
    with pytest.raises(error, match="num_threads"):
        perm_polynomial.rook_polynomial(
            [], backend="python", num_threads=num_threads
        )


def test_rook_polynomial_backend_contract(monkeypatch):
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        perm_polynomial.rook_polynomial([], backend="native")
    with pytest.raises(ValueError, match="backend must be one of"):
        perm_polynomial.rook_polynomial([], backend="cuda")
    with pytest.raises(TypeError, match="backend must be a string"):
        perm_polynomial.rook_polynomial([], backend=None)


@pytest.mark.parametrize("parallel", [0, 1, None, "yes"])
def test_weighted_rook_parallel_flag_must_be_boolean(parallel):
    with pytest.raises(TypeError, match="parallel"):
        perm_polynomial.weighted_rook_polynomial(
            [], backend="python", parallel=parallel
        )


@pytest.mark.parametrize("num_threads", [-1, 1 << 31, 1.5, True])
def test_weighted_rook_invalid_thread_counts_are_rejected(num_threads):
    error = ValueError if isinstance(num_threads, int) and not isinstance(
        num_threads, bool
    ) else TypeError
    with pytest.raises(error, match="num_threads"):
        perm_polynomial.weighted_rook_polynomial(
            [], backend="python", num_threads=num_threads
        )


def test_weighted_rook_backend_contract(monkeypatch):
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        perm_polynomial.weighted_rook_polynomial([], backend="native")
    with pytest.raises(ValueError, match="backend must be one of"):
        perm_polynomial.weighted_rook_polynomial([], backend="cuda")
    with pytest.raises(TypeError, match="backend must be a string"):
        perm_polynomial.weighted_rook_polynomial([], backend=None)


def test_trivial_inputs_do_not_call_available_native_extension(monkeypatch):
    class FailNative:
        @staticmethod
        def affine_permanent_mod(*args):
            raise AssertionError("trivial inputs should not reach native code")

        @staticmethod
        def weighted_rook_polynomial_mod_many(*args):
            raise AssertionError("trivial inputs should not reach native code")

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FailNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    assert perm_polynomial.affine_permanent_mod([], [], 9) == (1,)
    assert perm_polynomial.affine_permanent_mod([[11]], [[-2]], 9) == (2, 7)
    assert perm_polynomial.rook_polynomial([]) == (1,)
    assert perm_polynomial.rook_polynomial([[1]]) == (1, 1)
    assert perm_polynomial.weighted_rook_polynomial([]) == (1,)
    assert perm_polynomial.weighted_rook_polynomial(
        [[10**100]]
    ) == (1, 10**100)


@pytest.mark.parametrize(
    "invalid", [[1, 2], [1, 2, 3, 4], [0, True, 2], [0, -1, 2], 7]
)
def test_native_coefficient_contract_is_checked(monkeypatch, invalid):
    class FakeNative:
        @staticmethod
        def affine_permanent_mod(*args):
            return invalid

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FakeNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    with pytest.raises(RuntimeError, match="invalid coefficients"):
        perm_polynomial.affine_permanent_mod(
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            9,
            backend="native",
        )


def test_python_backend_does_not_call_available_native_extension(monkeypatch):
    class FailNative:
        @staticmethod
        def affine_permanent_mod(*args):
            raise AssertionError("native extension should not be called")

        @staticmethod
        def rook_polynomial_mod_many(*args):
            raise AssertionError("native extension should not be called")

        @staticmethod
        def weighted_rook_polynomial_mod_many(*args):
            raise AssertionError("native extension should not be called")

    monkeypatch.setattr(perm_polynomial, "_polynomial_ext", FailNative())
    monkeypatch.setattr(perm_polynomial, "HAS_POLYNOMIAL_EXT", True)
    assert perm_polynomial.affine_permanent_mod(
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        9,
        backend="python",
    ) == (1, 0, 1)
    assert perm_polynomial.principal_permanent_polynomial_mod(
        [[2, -3], [5, 7]], 101, backend="python"
    ) == (1, 9, 100)
    assert perm_polynomial.rook_polynomial(
        [[1, 1], [1, 0]], backend="python"
    ) == (1, 3, 1)
    assert perm_polynomial.weighted_rook_polynomial(
        [[2, 3], [3, 2]], backend="python"
    ) == (1, 10, 13)
