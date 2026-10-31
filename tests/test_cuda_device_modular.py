"""Opt-in correctness tests that execute the modular CUDA kernels on a GPU.

Run with ``PERM_RUN_CUDA_TESTS=1 python -m pytest -q`` on a CUDA host.
"""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import perm_cuda


LARGE_SIGNED_64_PRIME = 9_223_372_036_854_775_783
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _build_cuda_library(nvcc, architecture, output, source):
    result = subprocess.run(
        [
            nvcc,
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-O3",
            f"-arch={architecture}",
            "-o",
            str(output),
            str(source),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "CUDA build failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


@pytest.fixture(scope="module", autouse=True)
def require_cuda_device(tmp_path_factory):
    if os.environ.get("PERM_RUN_CUDA_TESTS") != "1":
        pytest.skip("set PERM_RUN_CUDA_TESTS=1 to run CUDA device tests")

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")

    build_dir = tmp_path_factory.mktemp("cuda-device-modular")
    architecture = os.environ.get("PERM_CUDA_TEST_ARCH", "native")
    mod3_path = build_dir / "perm_mod3_cuda.so"
    modp_path = build_dir / "perm_modp_cuda.so"
    _build_cuda_library(
        nvcc, architecture, mod3_path, REPOSITORY_ROOT / "perm_mod3_cuda.cu"
    )
    _build_cuda_library(
        nvcc, architecture, modp_path, REPOSITORY_ROOT / "perm_modp_cuda.cu"
    )

    mod3_library = ctypes.CDLL(str(mod3_path))
    mod3_library.permanent_mod3_matrix.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    mod3_library.permanent_mod3_matrix.restype = ctypes.c_int
    mod3_library.cuda_runtime_available.argtypes = []
    mod3_library.cuda_runtime_available.restype = ctypes.c_int

    modp_library = ctypes.CDLL(str(modp_path))
    modp_library.permanent_modp_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.c_longlong,
    ]
    modp_library.permanent_modp_cuda.restype = ctypes.c_longlong
    modp_library.schur_permanent_modp_cuda.argtypes = [
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    modp_library.schur_permanent_modp_cuda.restype = ctypes.c_longlong
    modp_library.cuda_runtime_available.argtypes = []
    modp_library.cuda_runtime_available.restype = ctypes.c_int

    if not mod3_library.cuda_runtime_available() or not modp_library.cuda_runtime_available():
        pytest.skip("CUDA runtime or device is unavailable")

    previous_mod3 = perm_cuda._mod3_cuda
    previous_modp = perm_cuda._modp_cuda
    perm_cuda._mod3_cuda = mod3_library
    perm_cuda._modp_cuda = modp_library
    try:
        yield
    finally:
        perm_cuda._mod3_cuda = previous_mod3
        perm_cuda._modp_cuda = previous_modp


def permanent_mod_reference(matrix, p):
    """Small exact Ryser reference using Python's unbounded integers."""
    array = np.asarray(matrix, dtype=object)
    n = array.shape[0]
    row_sums = [0] * n
    total = 0
    previous_gray = 0

    for state in range(1, 1 << n):
        gray = state ^ (state >> 1)
        changed = gray ^ previous_gray
        column = changed.bit_length() - 1
        direction = 1 if gray & changed else -1
        for row in range(n):
            row_sums[row] += direction * int(array[row, column])

        product = 1
        for value in row_sums:
            product = (product * (value % p)) % p
        total += (-1 if gray.bit_count() & 1 else 1) * product
        previous_gray = gray

    if n & 1:
        total = -total
    return total % p


@pytest.mark.parametrize("n", [2, 8, 10])
def test_identity_matrices(n):
    matrix = np.eye(n, dtype=np.int64)

    assert perm_cuda.permanent_mod3_cuda(matrix) == 1
    assert perm_cuda.permanent_modp_cuda(matrix, 101) == 1
    assert perm_cuda.permanent_modp_cuda(matrix, LARGE_SIGNED_64_PRIME) == 1


@pytest.mark.parametrize("seed", range(6))
def test_random_n10_matches_cpu_reference(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.integers(-1_000_000, 1_000_001, size=(10, 10), dtype=np.int64)
    matrix[0, 0] = np.iinfo(np.int64).min
    matrix[0, 1] = np.iinfo(np.int64).max

    expected_mod3 = permanent_mod_reference(matrix, 3)
    assert perm_cuda.permanent_mod3_cuda(matrix) == expected_mod3

    for p in (4_294_967_291, LARGE_SIGNED_64_PRIME):
        expected = permanent_mod_reference(matrix, p)
        assert perm_cuda.permanent_modp_cuda(matrix, p) == expected
