"""Focused tests for public backend and parallelism dispatch."""

import numpy as np
import pytest

import perm
import perm_float
import perm_mod3


@pytest.mark.parametrize("modulus", [2, 9, 101, (1 << 63) - 1])
def test_general_matrix_native_dispatch_uses_new_hook(monkeypatch, modulus):
    calls = []

    class Extension:
        @staticmethod
        def permanent_mod_p(matrix, p, threads):
            calls.append((matrix, p, threads))
            return 1

        @staticmethod
        def ryser_permanent_mod_p(*args):
            pytest.fail("new extensions should select their general algorithm")

    monkeypatch.setattr(perm, "HAS_C_EXT", True)
    monkeypatch.setattr(perm, "perm_c_ext", Extension())
    matrix = [[1, 2], [3, 4]]
    assert perm._ryser_permanent_with_backend(matrix, modulus, 3) == 1
    assert calls == [(matrix, modulus, 3)]


def test_general_matrix_native_dispatch_supports_older_extensions(monkeypatch):
    calls = []

    class LegacyExtension:
        @staticmethod
        def ryser_permanent_mod_p(matrix, p, threads):
            calls.append((matrix, p, threads))
            return 10

    monkeypatch.setattr(perm, "HAS_C_EXT", True)
    monkeypatch.setattr(perm, "perm_c_ext", LegacyExtension())
    matrix = [[1, 2], [3, 4]]
    assert perm._ryser_permanent_with_backend(matrix, 101, 1) == 10
    assert calls == [(matrix, 101, 1)]


def test_general_matrix_native_failure_is_not_hidden_by_fallback(monkeypatch):
    class Extension:
        @staticmethod
        def permanent_mod_p(*args):
            raise RuntimeError("native computation failed")

        @staticmethod
        def ryser_permanent_mod_p(*args):
            pytest.fail("a native failure must not silently trigger a fallback")

    monkeypatch.setattr(perm, "HAS_C_EXT", True)
    monkeypatch.setattr(perm, "perm_c_ext", Extension())
    with pytest.raises(RuntimeError, match="native computation failed"):
        perm._ryser_permanent_with_backend([[1, 2], [3, 4]], 101, 1)


def test_mod3_fast_prefers_native_backend_and_forwards_threads(monkeypatch):
    calls = []

    class FakeMod3Extension:
        @staticmethod
        def permanent_mod3(matrix, num_threads):
            calls.append((matrix.copy(), num_threads))
            return 2

    matrix = np.array(
        [[(1 << 63) + 1, 4], [5, (1 << 63) + 2]], dtype=np.uint64
    )
    monkeypatch.setattr(perm_mod3, "HAS_MOD3_OMP", True)
    monkeypatch.setattr(perm_mod3, "perm_mod3_omp", FakeMod3Extension())

    assert perm_mod3.permanent_mod3_fast(matrix, num_threads=3) == 2
    assert len(calls) == 1
    native_matrix, native_threads = calls[0]
    assert native_threads == 3
    assert native_matrix.dtype == np.int64
    np.testing.assert_array_equal(native_matrix, matrix % 3)

    calls.clear()
    assert perm_mod3.permanent_mod3_fast(matrix) == 2
    assert calls[0][1] == -1


def test_mod3_fast_portable_fallback_accepts_thread_setting(monkeypatch):
    calls = []
    matrix = np.array([[1, 2], [0, 1]], dtype=np.int64)

    monkeypatch.setattr(perm_mod3, "HAS_MOD3_OMP", False)
    monkeypatch.setattr(perm_mod3, "HAS_NUMBA", False)

    def fake_portable(argument):
        calls.append(argument)
        return 1

    monkeypatch.setattr(perm_mod3, "permanent_mod3_ryser", fake_portable)
    assert perm_mod3.permanent_mod3_fast(matrix, num_threads=2) == 1
    assert calls == [matrix]


@pytest.mark.parametrize(
    "function",
    [
        perm_mod3.encode_matrix_f3,
        perm_mod3.permanent_mod3_ryser,
        perm_mod3.permanent_mod3_naive,
        perm_mod3.permanent_mod3_fast,
    ],
)
def test_mod3_public_apis_reject_order_64_before_dispatch(function):
    matrix = np.zeros((64, 64), dtype=np.int8)
    with pytest.raises(ValueError, match="less than 64"):
        function(matrix)


@pytest.mark.skipif(not perm_float.HAS_NUMBA, reason="Numba is not installed")
def test_parallel_float_order_17_uses_compiled_kernel_and_restores_threads(
    monkeypatch,
):
    calls = []
    previous_threads = perm_float.numba.get_num_threads()

    def fake_kernel(matrix, chunks):
        calls.append((matrix, chunks, perm_float.numba.get_num_threads()))
        return 7.0

    monkeypatch.setattr(perm_float, "_permanent_parallel_real", fake_kernel)
    result = perm_float.permanent_ryser_parallel(
        np.eye(17, dtype=np.float64), num_threads=1
    )

    assert result == 7.0
    assert calls and calls[0][1:] == (4, 1)
    assert perm_float.numba.get_num_threads() == previous_threads


@pytest.mark.skipif(not perm_float.HAS_NUMBA, reason="Numba is not installed")
def test_parallel_float_kernel_failure_is_not_silently_fallbacked(monkeypatch):
    previous_threads = perm_float.numba.get_num_threads()

    def explode(*args, **kwargs):
        raise RuntimeError("compiled kernel failed")

    monkeypatch.setattr(perm_float, "_permanent_parallel_real", explode)
    monkeypatch.setattr(
        perm_float,
        "permanent_ryser_float",
        lambda matrix: pytest.fail("unexpected exponential scalar fallback"),
    )

    with pytest.raises(RuntimeError, match="compiled kernel failed"):
        perm_float.permanent_ryser_parallel(
            np.eye(17, dtype=np.float64), num_threads=1
        )
    assert perm_float.numba.get_num_threads() == previous_threads


def test_schur_float_defaults_to_parallel_dispatch(monkeypatch):
    calls = []
    matrix = np.eye(3, dtype=np.complex128)

    monkeypatch.setattr(perm_float, "schur_matrix_complex", lambda n: matrix)

    def fake_parallel(argument, num_threads=None):
        calls.append((argument, num_threads))
        return 5.0 + 0.0j

    monkeypatch.setattr(perm_float, "permanent_ryser_parallel", fake_parallel)
    assert perm_float.permanent_schur_float(3, num_threads=2) == 5.0 + 0.0j
    assert calls == [(matrix, 2)]


def test_large_few_prime_matrix_uses_serial_openmp_residues(monkeypatch):
    calls = []

    class FakeExtension:
        @staticmethod
        def get_max_threads():
            return 64

        @staticmethod
        def ryser_permanent_mod_p(matrix, p, num_threads):
            calls.append((p, num_threads))
            return 0

    monkeypatch.setattr(perm, "HAS_C_EXT", True)
    monkeypatch.setattr(perm, "perm_c_ext", FakeExtension())
    monkeypatch.setattr(perm, "matrix_permanent_bound", lambda matrix: 1 << 90)
    monkeypatch.setattr(perm, "is_prime", lambda candidate: True)
    monkeypatch.setattr(
        perm, "chinese_remainder_theorem", lambda residues, moduli: 0
    )

    matrix = np.eye(17, dtype=np.int64).tolist()
    assert perm.matrix_permanent(matrix, parallel=True) == 0
    assert len(calls) == 4
    assert {num_threads for _, num_threads in calls} == {64}

    calls.clear()
    assert perm.matrix_permanent(matrix, parallel=True, num_threads=8) == 0
    assert len(calls) == 4
    assert {num_threads for _, num_threads in calls} == {8}

    calls.clear()
    assert perm.matrix_permanent(matrix, parallel=False, num_threads=8) == 0
    assert len(calls) == 4
    assert {num_threads for _, num_threads in calls} == {1}


def test_public_schur_alias_forwards_thread_count(monkeypatch):
    calls = []

    def fake_schur(n, verbose, parallel, num_threads):
        calls.append((n, verbose, parallel, num_threads))
        return 17

    monkeypatch.setattr(perm, "schur_permanent_fast", fake_schur)
    assert perm.permanent(5, verbose=True, parallel=False, num_threads=3) == 17
    assert calls == [(5, True, False, 3)]


@pytest.mark.parametrize(
    "function,args",
    [
        (perm.matrix_permanent, ([[1]],)),
        (perm_float.permanent_schur_float, (1,)),
        (perm_mod3.permanent_mod3_fast, (np.array([[1]], dtype=np.int64),)),
    ],
)
def test_new_thread_controls_reject_boolean_counts(function, args):
    with pytest.raises(TypeError, match="num_threads"):
        function(*args, num_threads=True)
