"""CPU-only tests for the source-only binary CUDA Python wrapper."""

import ctypes
import itertools
import math

import numpy as np
import pytest

import perm_binary
import perm_binary_cuda


def _definition_permanent(matrix):
    return sum(
        math.prod(matrix[row][permutation[row]] for row in range(len(matrix)))
        for permutation in itertools.permutations(range(len(matrix)))
    )


class FakeBinaryCuda:
    def __init__(self, *, available=True, result=None):
        self.available = available
        self.result = result
        self.calls = []

    def cuda_runtime_available(self):
        return int(self.available)

    def binary_permanent_modp_cuda(self, pointer, order, modulus):
        matrix = np.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8)),
            shape=(order * order,),
        ).reshape((order, order)).astype(int).tolist()
        self.calls.append((matrix, order, modulus))
        if self.result is not None:
            return self.result
        return perm_binary._permanent_dp(
            perm_binary._row_masks(matrix), modulus
        )


def test_availability_requires_library_probe_and_runtime(monkeypatch):
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", None)
    assert not perm_binary_cuda.is_binary_cuda_available()

    fake = FakeBinaryCuda(available=False)
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", fake)
    assert not perm_binary_cuda.is_binary_cuda_available()

    fake.available = True
    assert perm_binary_cuda.is_binary_cuda_available()


def test_modular_wrapper_normalizes_and_calls_binary_abi(monkeypatch):
    fake = FakeBinaryCuda()
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", fake)
    matrix = np.array(
        [[True, True, False], [False, True, True], [True, False, True]],
        dtype=np.bool_,
    )
    expected = _definition_permanent(matrix.astype(int).tolist())
    assert perm_binary_cuda.binary_permanent_mod_p_cuda(matrix, 101) == expected
    assert fake.calls == [(matrix.astype(int).tolist(), 3, 101)]


def test_exact_wrapper_uses_checked_two_prime_crt(monkeypatch):
    fake = FakeBinaryCuda()
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", fake)
    matrix = [[1] * 13 for _ in range(13)]
    expected = math.factorial(13)
    assert expected > (1 << 31) - 1
    assert perm_binary_cuda.binary_permanent_cuda(matrix) == expected
    assert [call[2] for call in fake.calls] == (
        list(perm_binary._crt_primes_above_bound(expected))
    )
    assert len(fake.calls) == 2


def test_matching_blocks_factor_before_cuda_dispatch(monkeypatch):
    fake = FakeBinaryCuda()
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", fake)
    matrix = [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]
    assert perm_binary_cuda.binary_permanent_cuda(matrix) == 4
    assert [call[1] for call in fake.calls] == [2, 2]


def test_trivial_and_infeasible_inputs_do_not_load_cuda(monkeypatch):
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", None)
    assert perm_binary_cuda.binary_permanent_cuda([]) == 1
    assert perm_binary_cuda.binary_permanent_cuda([[1]]) == 1
    assert perm_binary_cuda.binary_permanent_mod_p_cuda([[0]], 3) == 0
    hall_obstruction = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    ]
    assert perm_binary_cuda.binary_permanent_cuda(hall_obstruction) == 0


def test_nontrivial_call_reports_missing_or_device_less_cuda(monkeypatch):
    matrix = [[1, 1], [1, 1]]
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", None)
    with pytest.raises(RuntimeError, match="library is unavailable"):
        perm_binary_cuda.binary_permanent_cuda(matrix)

    monkeypatch.setattr(
        perm_binary_cuda, "_binary_cuda", FakeBinaryCuda(available=False)
    )
    with pytest.raises(RuntimeError, match="no CUDA runtime device"):
        perm_binary_cuda.binary_permanent_cuda(matrix)


@pytest.mark.parametrize("result", [-1, 101])
def test_cuda_failure_and_invalid_residue_are_not_returned(monkeypatch, result):
    fake = FakeBinaryCuda(result=result)
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", fake)
    message = "residue failed" if result < 0 else "invalid residue"
    with pytest.raises(RuntimeError, match=message):
        perm_binary_cuda.binary_permanent_mod_p_cuda(
            [[1, 1], [1, 1]], 101
        )


def test_exact_reconstruction_outside_bound_is_rejected(monkeypatch):
    fake = FakeBinaryCuda(result=3)
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", fake)
    with pytest.raises(RuntimeError, match="outside the bound"):
        perm_binary_cuda.binary_permanent_cuda([[1, 1], [1, 1]])


@pytest.mark.parametrize(
    "matrix,modulus,error,message",
    [
        ([[1, 0]], 101, ValueError, "square"),
        ([[1, 2], [0, 1]], 101, ValueError, "0 or 1"),
        ([[1, 0], [0, 1]], 9, ValueError, "odd prime"),
        ([[1, 0], [0, 1]], 2_147_483_659, ValueError, "less than"),
        ([[1]], True, TypeError, "integer"),
    ],
)
def test_wrapper_rejects_invalid_inputs(matrix, modulus, error, message):
    with pytest.raises(error, match=message):
        perm_binary_cuda.binary_permanent_mod_p_cuda(matrix, modulus)


def test_non_strict_mode_constructs_support_matrix(monkeypatch):
    fake = FakeBinaryCuda()
    monkeypatch.setattr(perm_binary_cuda, "_binary_cuda", fake)
    matrix = [[2, -1], [-7, 3]]
    assert (
        perm_binary_cuda.binary_permanent_mod_p_cuda(
            matrix, 101, strict=False
        )
        == 2
    )
    assert fake.calls[0][0] == [[1, 1], [1, 1]]
