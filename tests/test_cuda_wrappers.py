import ctypes
import math

import numpy as np
import pytest

import perm_cuda


class FakeMod3Library:
    def __init__(self, result=0):
        self.result = result
        self.values = None

    @staticmethod
    def cuda_runtime_available():
        return 1

    def permanent_mod3_matrix(self, pointer, n):
        self.values = np.ctypeslib.as_array(pointer, shape=(n * n,)).copy()
        return self.result


class FakeModPLibrary:
    def __init__(self, matrix_result=0, schur_result=0):
        self.matrix_result = matrix_result
        self.schur_result = schur_result
        self.values = None
        self.schur_args = None

    @staticmethod
    def cuda_runtime_available():
        return 1

    def permanent_modp_cuda(self, pointer, n, p):
        typed = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_longlong))
        self.values = np.ctypeslib.as_array(typed, shape=(n * n,)).copy()
        return self.matrix_result

    def schur_permanent_modp_cuda(self, n, omega, p):
        self.schur_args = (n, omega, p)
        return self.schur_result


def test_mod3_reduces_before_int32_conversion(monkeypatch):
    fake = FakeMod3Library(result=2)
    monkeypatch.setattr(perm_cuda, "_mod3_cuda", fake)
    huge = 1 << 100
    matrix = np.array([[huge + 2, -huge], [3, -4]], dtype=object)

    assert perm_cuda.permanent_mod3_cuda(matrix) == 2
    np.testing.assert_array_equal(fake.values, [0, 2, 0, 2])


def test_modp_reduces_before_int64_conversion(monkeypatch):
    p = 4_294_967_291
    values = [1 << 100, -(1 << 100), (1 << 63) - 1, -(1 << 63)]
    fake = FakeModPLibrary(matrix_result=17)
    monkeypatch.setattr(perm_cuda, "_modp_cuda", fake)

    assert perm_cuda.permanent_modp_cuda(
        np.array(values, dtype=object).reshape(2, 2), p
    ) == 17
    np.testing.assert_array_equal(fake.values, [value % p for value in values])


def test_schur_arguments_are_validated_and_normalized(monkeypatch):
    fake = FakeModPLibrary(schur_result=4)
    monkeypatch.setattr(perm_cuda, "_modp_cuda", fake)

    assert perm_cuda.schur_permanent_modp_cuda(3, -2, 7) == 4
    assert fake.schur_args == (3, 5, 7)


@pytest.mark.parametrize(
    ("function", "library_name", "library"),
    [
        (lambda: perm_cuda.permanent_mod3_cuda(np.eye(2, dtype=int)),
         "_mod3_cuda", FakeMod3Library(result=-1)),
        (lambda: perm_cuda.permanent_modp_cuda(np.eye(2, dtype=int), 5),
         "_modp_cuda", FakeModPLibrary(matrix_result=-1)),
        (lambda: perm_cuda.schur_permanent_modp_cuda(3, 2, 7),
         "_modp_cuda", FakeModPLibrary(schur_result=-1)),
    ],
)
def test_native_failure_sentinels_raise(monkeypatch, function, library_name, library):
    monkeypatch.setattr(perm_cuda, library_name, library)
    with pytest.raises(RuntimeError, match="failed"):
        function()


def test_trivial_sizes_do_not_require_loaded_libraries(monkeypatch):
    monkeypatch.setattr(perm_cuda, "_mod3_cuda", None)
    monkeypatch.setattr(perm_cuda, "_modp_cuda", None)

    assert perm_cuda.permanent_mod3_cuda([[10]]) == 1
    assert perm_cuda.permanent_modp_cuda([[-2]], 5) == 3
    assert perm_cuda.permanent_modp_cuda([], 1) == 0
    assert perm_cuda.schur_permanent_modp_cuda(1, 999, 5) == 1


@pytest.mark.parametrize(
    "matrix",
    [
        [1, 2],
        [[1, 2]],
        [[1, 2], [3]],
        np.zeros((2, 2, 1), dtype=int),
        np.zeros((64, 64), dtype=int),
    ],
)
def test_matrix_shape_and_dimension_validation(matrix):
    with pytest.raises(ValueError):
        perm_cuda.permanent_modp_cuda(matrix, 5)


def test_integer_and_modulus_validation():
    with pytest.raises(TypeError):
        perm_cuda.permanent_mod3_cuda([[1.5]])
    with pytest.raises(ValueError):
        perm_cuda.permanent_modp_cuda([[1]], 0)
    with pytest.raises(ValueError):
        perm_cuda.permanent_modp_cuda([[1]], 1 << 63)
    with pytest.raises(ValueError):
        perm_cuda.schur_permanent_modp_cuda(-1, 1, 3)


def test_cuda_schur_bound_uses_exact_integer_arithmetic():
    radicand = 27 ** 27

    bound = perm_cuda.schur_permanent_bound(27)

    assert bound * bound >= radicand
    assert (bound - 1) * (bound - 1) < radicand
    # This was underestimated by the previous float-based implementation.
    assert bound - int(math.ceil(27 ** (27 / 2))) == 1_563
