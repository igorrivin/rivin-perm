"""Regression tests for the public Python permanent APIs."""

import itertools
import math

import numpy as np
import pytest

import perm
import perm_float
import perm_mod3
import perm_numba


def _brute_force_permanent(matrix):
    n = len(matrix)
    total = 0
    for permutation in itertools.permutations(range(n)):
        product = 1
        for row, column in enumerate(permutation):
            product *= matrix[row][column]
        total += product
    return total


def test_exact_integer_implementations_match_brute_force():
    rng = np.random.default_rng(20260814)
    for n in range(6):
        for _ in range(3):
            array = rng.integers(-4, 5, size=(n, n), dtype=np.int64)
            matrix = array.tolist()
            expected = _brute_force_permanent(matrix)
            assert perm.matrix_permanent(matrix, parallel=False) == expected
            assert perm.ryser_permanent_mod_p(matrix, 101) == expected % 101
            assert perm.ryser_permanent_mod_p_fast(matrix, 101) == expected % 101
            assert perm_mod3.permanent_mod3_naive(array) == expected % 3
            assert perm_mod3.permanent_mod3_ryser(array) == expected % 3
            assert perm_mod3.permanent_mod3_fast(array) == expected % 3


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag(np.array([1 << 32, 1 << 32], dtype=np.int64)),
        np.diag(np.array([-(1 << 63), -(1 << 63)], dtype=np.int64)),
        np.array([[(1 << 63) - 1, (1 << 63) - 1], [1, 2]], dtype=np.int64),
        np.diag(np.array([(1 << 64) - 1, (1 << 64) - 1], dtype=np.uint64)),
        np.array([[1, 2], [3, 4]], dtype=np.int8),
    ],
)
@pytest.mark.parametrize("portable", [False, True])
def test_numpy_integer_inputs_preserve_exact_arithmetic(matrix, portable, monkeypatch):
    if portable:
        monkeypatch.setattr(perm, "HAS_C_EXT", False)
        monkeypatch.setattr(perm, "HAS_NUMBA", False)
    integers = matrix.tolist()
    expected = _brute_force_permanent(integers)
    expected_bound = math.prod(sum(abs(entry) for entry in row) for row in integers)

    # Treat overflow warnings as failures even when a wrapped value happens to
    # yield the expected residue for one of these small examples.
    with np.errstate(over="raise", invalid="raise"):
        bound = perm.matrix_permanent_bound(matrix)
        result = perm.matrix_permanent(matrix, parallel=False)
    assert type(bound) is int
    assert bound == expected_bound
    assert type(result) is int
    assert result == expected


@pytest.mark.parametrize(
    "function",
    [perm.ryser_permanent_mod_p, perm.ryser_permanent_mod_p_fast, perm._glynn_permanent_mod_p],
)
@pytest.mark.parametrize("modulus", [101, 4_294_967_291, (1 << 67) - 1])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_numpy_modular_inputs_use_python_integer_arithmetic(function, modulus, dtype):
    extreme = np.iinfo(dtype).max
    matrix = np.array([[extreme, extreme - 1], [extreme - 2, extreme - 3]], dtype=dtype)
    expected = _brute_force_permanent(matrix.tolist()) % modulus
    with np.errstate(over="raise", invalid="raise"):
        result = function(matrix, modulus)
        singleton = function(matrix[:1, :1], modulus)
    assert type(result) is int
    assert result == expected
    assert type(singleton) is int
    assert singleton == int(extreme) % modulus


def test_crt_numpy_scalars_do_not_overflow_combined_modulus():
    moduli = np.array([4_294_967_291, 4_294_967_279], dtype=np.int64)
    with np.errstate(over="raise", invalid="raise"):
        result = perm.chinese_remainder_theorem(moduli - 1, moduli)
    assert type(result) is int
    assert result == math.prod(map(int, moduli)) - 1


def test_schur_matrix_numpy_root_uses_exact_modular_products():
    modulus = 4_294_967_291
    omega = np.int64(modulus - 1)
    with np.errstate(over="raise", invalid="raise"):
        matrix = perm.build_schur_matrix_mod_p(2, omega, modulus)
    assert matrix == [[1, 1], [1, modulus - 1]]
    assert all(type(entry) is int for row in matrix for entry in row)


@pytest.mark.parametrize(
    "function",
    [
        perm.matrix_permanent,
        perm.matrix_permanent_bound,
        perm.ryser_permanent_mod_p,
        perm.ryser_permanent_mod_p_fast,
        perm_float.permanent_ryser_float,
        perm_float.permanent_ryser_parallel,
        perm_mod3.encode_matrix_f3,
        perm_mod3.permanent_mod3_naive,
        perm_mod3.permanent_mod3_ryser,
        perm_mod3.permanent_mod3_fast,
    ],
)
def test_rectangular_matrix_is_rejected_before_base_case(function):
    matrix = [[7, 8]]
    if function in (perm.ryser_permanent_mod_p, perm.ryser_permanent_mod_p_fast):
        with pytest.raises(ValueError, match="square"):
            function(matrix, 101)
    else:
        with pytest.raises(ValueError, match="square"):
            function(matrix)


def test_empty_rectangular_numpy_matrix_is_rejected():
    matrix = np.empty((0, 2), dtype=np.int64)
    for function in (
        perm.matrix_permanent,
        perm_float.permanent_ryser_float,
        perm_mod3.permanent_mod3_fast,
    ):
        with pytest.raises(ValueError, match="square"):
            function(matrix)


def test_empty_square_matrix_has_unit_permanent():
    assert perm.matrix_permanent([], parallel=False) == 1
    assert perm_float.permanent_ryser_float([]) == 1.0
    assert perm_mod3.permanent_mod3_fast([]) == 1


@pytest.mark.parametrize(
    "function,args",
    [
        (perm.matrix_permanent, ([[1.5]],)),
        (perm.matrix_permanent_bound, ([[1.5]],)),
        (perm.ryser_permanent_mod_p, ([[1.5]], 101)),
        (perm.ryser_permanent_mod_p_fast, ([[1.5]], 101)),
    ],
)
def test_exact_integer_apis_reject_non_integer_entries(function, args):
    with pytest.raises(TypeError, match="entries must be integers"):
        function(*args)


def test_schur_bound_is_an_exact_integer_ceiling():
    radicand = 27**27
    floor = math.isqrt(radicand)
    assert floor * floor != radicand
    assert perm.schur_permanent_bound(27) == floor + 1
    assert perm.schur_permanent_bound(27) ** 2 >= radicand


def test_even_schur_permanent_short_circuits(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("the CRT/residue path should not run for even n")

    monkeypatch.setattr(perm, "find_primes_congruent_to_1_mod_n", fail_if_called)
    assert perm.schur_permanent(38) == 0
    assert perm.schur_permanent_fast(38) == 0
    assert perm_float.permanent_schur_float(38) == 0j


@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_invalid_schur_size_is_rejected(n):
    expected = TypeError if isinstance(n, (float, bool)) else ValueError
    with pytest.raises(expected):
        perm.schur_permanent_fast(n)
    with pytest.raises(expected):
        perm_float.schur_matrix_complex(n)


def test_parallel_false_forces_single_threaded_residues(monkeypatch):
    thread_counts = []

    class FakeExtension:
        @staticmethod
        def get_max_threads():
            raise AssertionError("automatic OpenMP selection should not run")

        @staticmethod
        def schur_permanent_mod_p(n, omega, p, num_threads):
            thread_counts.append(num_threads)
            return (-3) % p

    monkeypatch.setattr(perm, "HAS_C_EXT", True)
    monkeypatch.setattr(perm, "perm_c_ext", FakeExtension())
    assert perm.schur_permanent_fast(3, parallel=False, num_threads=32) == -3
    assert thread_counts and set(thread_counts) == {1}


@pytest.mark.skipif(not perm_float.HAS_NUMBA, reason="Numba is not installed")
def test_parallel_float_order_21_real_and_complex_honors_thread_count():
    previous_threads = perm_float.numba.get_num_threads()
    requested_threads = min(2, previous_threads)
    real_result = perm_float.permanent_ryser_parallel(
        np.eye(21, dtype=np.float64), num_threads=requested_threads
    )
    complex_result = perm_float.permanent_ryser_parallel(
        np.eye(21, dtype=np.complex128), num_threads=requested_threads
    )
    assert real_result == pytest.approx(1.0)
    assert complex_result == pytest.approx(1.0 + 0.0j)
    assert perm_float.numba.get_num_threads() == previous_threads


@pytest.mark.skipif(not perm_numba.HAS_NUMBA, reason="Numba is not installed")
def test_numba_large_modulus_multiplication_does_not_overflow():
    p = 4_294_967_291
    matrix = np.diag(np.array([p - 1, p - 1], dtype=np.int64))
    assert perm_numba.ryser_permanent_mod_p_numba(matrix, p) == 1
    assert perm_numba.ryser_permanent_mod_p_numba_parallel(matrix, p) == 1


def test_public_preconditions_raise_explicit_exceptions():
    with pytest.raises(ValueError, match="not ≡"):
        perm.nth_root_of_unity_mod_p(3, 11)
    with pytest.raises(ValueError, match="coprime"):
        perm.chinese_remainder_theorem([0, 1], [6, 9])
    with pytest.raises(ValueError, match="64"):
        perm_mod3.encode_matrix_f3(np.zeros((65, 65), dtype=np.int64))
