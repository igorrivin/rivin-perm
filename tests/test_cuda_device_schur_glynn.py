"""Opt-in device tests for the Schur-specific CUDA Glynn path.

Run with ``PERM_RUN_CUDA_TESTS=1 python -m pytest -q
tests/test_cuda_device_schur_glynn.py`` on a CUDA host.  The order-41 check is
additionally gated by ``PERM_RUN_CUDA_LONG_TESTS=1``.  The much longer
order-47 frontier regression has its own ``PERM_RUN_CUDA_FRONTIER_TESTS=1``
gate.
"""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _permanent_mod_reference(matrix, modulus):
    """Small exact Ryser reference using Python's unbounded integers."""
    n = len(matrix)
    row_sums = [0] * n
    total = 0
    previous_gray = 0

    for state in range(1, 1 << n):
        gray = state ^ (state >> 1)
        changed = gray ^ previous_gray
        column = changed.bit_length() - 1
        direction = 1 if gray & changed else -1
        for row in range(n):
            row_sums[row] += direction * matrix[row][column]

        product = 1
        for value in row_sums:
            product = product * value % modulus
        total += (-1 if gray.bit_count() & 1 else 1) * product
        previous_gray = gray

    if n & 1:
        total = -total
    return total % modulus


def _schur_reference(n, omega, modulus):
    matrix = [
        [pow(omega, row * column, modulus) for column in range(n)]
        for row in range(n)
    ]
    return _permanent_mod_reference(matrix, modulus)


@pytest.fixture(scope="module")
def schur_cuda(tmp_path_factory):
    if os.environ.get("PERM_RUN_CUDA_TESTS") != "1":
        pytest.skip("set PERM_RUN_CUDA_TESTS=1 to run CUDA device tests")

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")

    build_dir = tmp_path_factory.mktemp("cuda-device-schur-glynn")
    library_path = build_dir / "perm_modp_cuda.so"
    architecture = os.environ.get("PERM_CUDA_TEST_ARCH", "native")
    result = subprocess.run(
        [
            nvcc,
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-O3",
            f"-arch={architecture}",
            "-o",
            str(library_path),
            str(REPOSITORY_ROOT / "perm_modp_cuda.cu"),
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

    library = ctypes.CDLL(str(library_path))
    library.cuda_runtime_available.argtypes = []
    library.cuda_runtime_available.restype = ctypes.c_int
    library.schur_permanent_modp_cuda.argtypes = [
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    library.schur_permanent_modp_cuda.restype = ctypes.c_longlong
    if not library.cuda_runtime_available():
        pytest.skip("CUDA runtime or device is unavailable")
    return library.schur_permanent_modp_cuda


@pytest.mark.parametrize(
    ("n", "omega", "modulus"),
    [
        (3, 2, 7),
        (5, 4, 11),
        (7, 16, 29),
        (9, 4, 19),
        # omega=1 is an n-th root and also exercises a near-int64 modulus.
        (3, 1, (1 << 63) - 1),
    ],
)
def test_odd_root_of_unity_cases_match_reference(
    schur_cuda, n, omega, modulus
):
    assert pow(omega, n, modulus) == 1
    expected = _schur_reference(n, omega, modulus)
    assert schur_cuda(n, omega, modulus) == expected


@pytest.mark.parametrize(
    ("n", "omega", "modulus"),
    [
        (4, 2, 5),       # even order
        (3, 1, 2),       # even modulus: 2 is not invertible
        (3, 2, 5),       # omega is not an n-th root of unity
    ],
)
def test_non_glynn_preconditions_use_safe_fallback(
    schur_cuda, n, omega, modulus
):
    expected = _schur_reference(n, omega, modulus)
    assert schur_cuda(n, omega, modulus) == expected


def test_order_33_pseudo_mersenne_specialization(schur_cuda):
    """Exercise the checked-in CRT prime's register-resident pm32 kernel."""
    modulus = 4_294_967_161
    omega = 2_662_262
    assert pow(omega, 33, modulus) == 1
    assert schur_cuda(33, omega, modulus) == 1_428_739_342


@pytest.mark.skipif(
    os.environ.get("PERM_RUN_CUDA_LONG_TESTS") != "1",
    reason="set PERM_RUN_CUDA_LONG_TESTS=1 for the order-41 CUDA check",
)
def test_order_41_known_residue(schur_cuda):
    modulus = 4_294_966_153
    omega = 782_766_686
    assert pow(omega, 41, modulus) == 1
    assert schur_cuda(41, omega, modulus) == 3_889_040_436


@pytest.mark.skipif(
    os.environ.get("PERM_RUN_CUDA_FRONTIER_TESTS") != "1",
    reason="set PERM_RUN_CUDA_FRONTIER_TESTS=1 for the order-47 CUDA check",
)
def test_order_47_known_residue(schur_cuda):
    """Exercise the packed M46/depth-28 necklace specialization."""
    modulus = 4_294_967_161
    omega = 1_002_938_914
    assert pow(omega, 47, modulus) == 1
    assert schur_cuda(47, omega, modulus) == 4_173_217_525
