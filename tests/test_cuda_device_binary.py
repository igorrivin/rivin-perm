"""Opt-in on-device tests for the source-only binary CUDA backend.

Run with ``PERM_RUN_CUDA_TESTS=1 python -m pytest -q`` on a CUDA host.
The module builds ``perm_binary_cuda.cu`` in pytest's temporary directory;
packaged installs remain independent of CUDA.
"""

import ctypes
import itertools
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

import perm_binary_cuda
from perm_binary_applications import grid_biadjacency


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_DEVICE_TESTS = os.environ.get("PERM_RUN_CUDA_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}
TARGET_GRID_PERMANENT = 2_551_276_235_535_120

pytestmark = pytest.mark.skipif(
    not RUN_DEVICE_TESTS,
    reason="set PERM_RUN_CUDA_TESTS=1 to run CUDA device tests",
)


def _definition_permanent(matrix):
    order = len(matrix)
    return sum(
        math.prod(matrix[row][permutation[row]] for row in range(order))
        for permutation in itertools.permutations(range(order))
    )


@pytest.fixture(scope="module")
def binary_cuda_library(tmp_path_factory):
    if os.name != "posix":
        pytest.skip("the binary CUDA test build currently targets POSIX hosts")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")

    build_directory = tmp_path_factory.mktemp("cuda-device-binary")
    library_path = build_directory / "libperm_binary_cuda.so"
    architecture = os.environ.get("PERM_CUDA_TEST_ARCH", "native")
    result = subprocess.run(
        [
            nvcc,
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-O3",
            "-std=c++17",
            f"-arch={architecture}",
            "-o",
            str(library_path),
            str(REPOSITORY_ROOT / "perm_binary_cuda.cu"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "binary CUDA build failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    library = ctypes.CDLL(str(library_path))
    perm_binary_cuda._configure_binary_cuda_library(library)
    if not library.cuda_runtime_available():
        pytest.skip("no CUDA runtime device is accessible")

    previous = perm_binary_cuda._binary_cuda
    perm_binary_cuda._binary_cuda = library
    try:
        yield library
    finally:
        perm_binary_cuda._binary_cuda = previous


def test_wrapper_imports_beside_plain_c_abi_library(
    binary_cuda_library, tmp_path
):
    """The ``lib`` name must not shadow ``perm_binary_cuda.py``."""
    module_directory = tmp_path / "source-checkout"
    module_directory.mkdir()
    shutil.copy2(REPOSITORY_ROOT / "perm_binary_cuda.py", module_directory)
    library_name = "libperm_binary_cuda.so"
    source_library = Path(binary_cuda_library._name)
    shutil.copy2(source_library, module_directory / library_name)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(module_directory), str(REPOSITORY_ROOT))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import perm_binary_cuda; "
            "assert perm_binary_cuda.__file__.endswith('.py'); "
            "assert perm_binary_cuda.is_binary_cuda_available()",
        ],
        cwd=module_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


@pytest.mark.parametrize("seed", range(5))
def test_random_small_matrices_match_definition(binary_cuda_library, seed):
    rng = np.random.default_rng(20260815 + seed)
    for order in (2, 3, 5, 7, 8):
        matrix = rng.integers(0, 2, size=(order, order), dtype=np.uint8)
        expected = _definition_permanent(matrix.tolist())
        for prime in (101, 2_147_483_647):
            assert (
                perm_binary_cuda.binary_permanent_mod_p_cuda(matrix, prime)
                == expected % prime
            )


def test_dense_exact_crt_matches_factorial(binary_cuda_library):
    matrix = np.ones((13, 13), dtype=np.uint8)
    assert perm_binary_cuda.binary_permanent_cuda(matrix) == math.factorial(13)


def test_direct_abi_rejections_use_negative_sentinel(binary_cuda_library):
    """Invalid ABI calls return -1; there is no output buffer to mutate."""
    call = binary_cuda_library.binary_permanent_modp_cuda
    valid = np.eye(5, dtype=np.uint8)
    pointer = valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    null = ctypes.POINTER(ctypes.c_uint8)()

    assert call(pointer, 5, 9) == -1  # composite
    assert call(pointer, 5, 5) == -1  # not greater than the order
    assert call(pointer, 5, 2_147_483_659) == -1  # outside 31-bit range
    assert call(pointer, -1, 101) == -1
    assert call(pointer, 64, 101) == -1
    assert call(null, 5, 101) == -1

    nonbinary = valid.copy()
    nonbinary[0, 0] = 2
    nonbinary_pointer = nonbinary.ctypes.data_as(
        ctypes.POINTER(ctypes.c_uint8)
    )
    assert call(nonbinary_pointer, 5, 101) == -1

    # The empty convention permits a null matrix, and a valid call after all
    # rejected inputs proves that the sentinel paths did not poison state.
    assert call(null, 0, 3) == 1
    assert call(pointer, 5, 101) == 1


def test_periodic_4_by_4_by_5_grid_exact_result(binary_cuda_library):
    matrix = grid_biadjacency((4, 4, 5), periodic_axes=(0, 1))
    assert perm_binary_cuda.binary_permanent_cuda(matrix) == TARGET_GRID_PERMANENT
