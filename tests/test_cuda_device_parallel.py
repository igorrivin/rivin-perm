"""Opt-in on-device tests for the parallel CUDA permanent backend.

Run on a CUDA host with, for example::

    PERM_RUN_CUDA_TESTS=1 PERM_CUDA_ARCH=sm_90 \
        python -m pytest -q tests/test_cuda_device_parallel.py
"""

import itertools
import os

import numpy as np
import pytest

import perm_parallel
from perm_float import permanent_ryser_float


pytestmark = pytest.mark.skipif(
    os.environ.get("PERM_RUN_CUDA_TESTS") != "1",
    reason="set PERM_RUN_CUDA_TESTS=1 to run tests on a CUDA device",
)


@pytest.fixture(scope="module", autouse=True)
def cuda_device():
    """Skip cleanly when the opt-in flag is set on a host without CUDA."""
    try:
        info = perm_parallel.gpu_info()
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"CUDA backend is unavailable: {exc}")
    assert info[0] > 0
    assert info[1] > 0
    return info


def _brute_force_permanent(matrix):
    n = matrix.shape[0]
    total = matrix.dtype.type(0)
    for permutation in itertools.permutations(range(n)):
        product = matrix.dtype.type(1)
        for row, column in enumerate(permutation):
            product *= matrix[row, column]
        total += product
    return total


@pytest.mark.parametrize("n", range(1, 8))
def test_real_random_matrices_match_brute_force(n):
    rng = np.random.default_rng(9_000 + n)
    matrix = rng.integers(-3, 4, size=(n, n)).astype(np.float64)
    expected = _brute_force_permanent(matrix)
    actual = perm_parallel.permanent_gpu(matrix)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-10)


@pytest.mark.parametrize("n", range(1, 7))
def test_complex_random_matrices_match_brute_force(n):
    rng = np.random.default_rng(19_000 + n)
    real = rng.integers(-2, 3, size=(n, n))
    imaginary = rng.integers(-2, 3, size=(n, n))
    matrix = (real + 1j * imaginary).astype(np.complex128)
    expected = _brute_force_permanent(matrix)
    actual = perm_parallel.permanent_gpu(matrix)
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-10)


def test_n10_multiblock_reduction_matches_cpu_ryser_real_and_complex():
    rng = np.random.default_rng(42_010)
    real_matrix = rng.normal(scale=0.35, size=(10, 10))
    complex_matrix = real_matrix + 1j * rng.normal(scale=0.35, size=(10, 10))

    real_expected = permanent_ryser_float(real_matrix)
    complex_expected = permanent_ryser_float(complex_matrix)

    assert perm_parallel.permanent_gpu(real_matrix) == pytest.approx(
        real_expected, rel=2e-10, abs=1e-12
    )
    assert perm_parallel.permanent_gpu(complex_matrix) == pytest.approx(
        complex_expected, rel=3e-10, abs=2e-12
    )


def test_larger_identity_exercises_device_sized_launch_and_reduction():
    # n > 20 selects an SM-count-scaled launch; n=24 remains quick enough for
    # routine opt-in validation while covering millions of Glynn sign vectors.
    matrix = np.eye(24, dtype=np.float64)
    assert perm_parallel.permanent_gpu(matrix) == pytest.approx(1.0, abs=1e-12)


def test_device_base_and_conversion_paths():
    assert perm_parallel.permanent_gpu(np.array([[7]], dtype=np.int16)) == 7.0
    assert perm_parallel.permanent_gpu(
        np.array([[2.0]]), real=False
    ) == 2.0 + 0.0j


def test_device_reports_unsupported_order():
    with pytest.raises(RuntimeError, match="rc=-1"):
        perm_parallel.permanent_gpu(np.eye(64, dtype=np.float64))
