"""Opt-in, device-backed tests for the standalone CUDA utility libraries.

Set ``PERM_RUN_CUDA_TESTS=1`` to enable these tests.  The test fixture
then verifies that a CUDA runtime device is accessible and compiles fresh
shared libraries from the repository sources.  ``PERM_CUDA_TEST_ARCH`` may be
used to override nvcc's default ``native`` architecture selection.
"""

from collections import Counter
import ctypes
import itertools
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_DEVICE_TESTS = os.environ.get("PERM_RUN_CUDA_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}

pytestmark = pytest.mark.skipif(
    not RUN_DEVICE_TESTS,
    reason="set PERM_RUN_CUDA_TESTS=1 to run CUDA device tests",
)


def _run_build(command):
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "CUDA build failed:\n"
            f"command: {' '.join(map(str, command))}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


@pytest.fixture(scope="module")
def cuda_utility_libraries(tmp_path_factory):
    if os.name != "posix":
        pytest.skip("the CUDA utility test build currently targets POSIX hosts")

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")

    build_dir = tmp_path_factory.mktemp("cuda-device-utilities")

    # Probe the CUDA runtime itself instead of treating a successfully loaded
    # library or an installed nvcc as proof that a device is accessible.
    probe_source = build_dir / "cuda_runtime_probe.cu"
    probe_binary = build_dir / "cuda_runtime_probe"
    probe_source.write_text(
        "#include <cuda_runtime.h>\n"
        "int main() {\n"
        "    int count = 0;\n"
        "    cudaError_t error = cudaGetDeviceCount(&count);\n"
        "    return error == cudaSuccess && count > 0 ? 0 : 1;\n"
        "}\n",
        encoding="utf-8",
    )
    _run_build([nvcc, "-O2", "-o", str(probe_binary), str(probe_source)])
    probe = subprocess.run(
        [str(probe_binary)], capture_output=True, text=True, timeout=30, check=False
    )
    if probe.returncode != 0:
        pytest.skip("no CUDA runtime device is accessible")

    architecture = os.environ.get("PERM_CUDA_TEST_ARCH", "native")
    enum_library_path = build_dir / "perm_enum_cuda.so"
    proj_library_path = build_dir / "perm_proj_cuda.so"
    common = [nvcc, "-shared", "-Xcompiler", "-fPIC", "-O3", f"-arch={architecture}"]
    _run_build(
        common
        + [
            "-o",
            str(enum_library_path),
            str(REPOSITORY_ROOT / "perm_enum_cuda.cu"),
        ]
    )
    _run_build(
        common
        + [
            "-o",
            str(proj_library_path),
            str(REPOSITORY_ROOT / "perm_proj_cuda.cu"),
        ]
    )

    enum_library = ctypes.CDLL(str(enum_library_path))
    enum_library.enumerate_permanent_modp.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    enum_library.enumerate_permanent_modp.restype = ctypes.c_int
    enum_library.sample_permanent_modp.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    enum_library.sample_permanent_modp.restype = ctypes.c_int

    proj_library = ctypes.CDLL(str(proj_library_path))
    proj_library.proj_multiset_perm_n4.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    proj_library.proj_multiset_perm_n4.restype = ctypes.c_int

    return enum_library, proj_library


def _permanent_mod(matrix, modulus):
    order = len(matrix)
    total = 0
    for permutation in itertools.permutations(range(order)):
        product = 1
        for row, column in enumerate(permutation):
            product *= int(matrix[row][column])
        total += product
    return total % modulus


def _permanent_histogram(matrices, modulus):
    histogram = [0] * modulus
    for matrix in matrices:
        histogram[_permanent_mod(matrix, modulus)] += 1
    return histogram


def _sample_on_gpu(library, matrices, modulus):
    matrices = np.ascontiguousarray(matrices, dtype=np.int32)
    order = matrices.shape[1]
    counts = (ctypes.c_longlong * modulus)(*([-1] * modulus))
    rc = library.sample_permanent_modp(
        order,
        modulus,
        matrices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(matrices),
        counts,
    )
    return rc, list(counts)


def test_complete_enumeration_n2_p3_matches_cpu(cuda_utility_libraries):
    enum_library, _ = cuda_utility_libraries
    counts = (ctypes.c_longlong * 3)(-1, -1, -1)

    rc = enum_library.enumerate_permanent_modp(2, 3, counts)
    matrices = np.array(
        list(itertools.product(range(3), repeat=4)), dtype=np.int32
    ).reshape((-1, 2, 2))
    expected = _permanent_histogram(matrices, 3)

    assert rc == 0
    assert list(counts) == expected == [33, 24, 24]
    assert sum(counts) == 81


def test_sample_histograms_match_cpu(cuda_utility_libraries):
    enum_library, _ = cuda_utility_libraries
    known = np.array(
        [
            [[1, 0], [0, 1]],
            [[1, 2], [3, 4]],
            [[0, 0], [0, 0]],
            [[4, 4], [4, 4]],
        ],
        dtype=np.int32,
    )
    cases = [
        (known, 5),
        (
            np.random.default_rng(20260814).integers(
                0, 5, size=(513, 2, 2), dtype=np.int32
            ),
            5,
        ),
        (
            np.random.default_rng(20260815).integers(
                0, 3, size=(257, 3, 3), dtype=np.int32
            ),
            3,
        ),
    ]

    for matrices, modulus in cases:
        rc, counts = _sample_on_gpu(enum_library, matrices, modulus)
        assert rc == 0
        assert counts == _permanent_histogram(matrices, modulus)
        assert sum(counts) == len(matrices)


def _projective_rows(modulus):
    rows = []
    for leading_position in range(4):
        for tail in itertools.product(
            range(modulus), repeat=3 - leading_position
        ):
            row = [0] * 4
            row[leading_position] = 1
            row[leading_position + 1 :] = tail
            rows.append(row)
    return rows


def _projective_zero_weight(modulus):
    rows = _projective_rows(modulus)
    zero_weight = 0
    for indices in itertools.combinations_with_replacement(range(len(rows)), 4):
        matrix = [rows[index] for index in indices]
        if _permanent_mod(matrix, modulus) != 0:
            continue
        weight = math.factorial(4)
        for multiplicity in Counter(indices).values():
            weight //= math.factorial(multiplicity)
        zero_weight += weight
    return len(rows), zero_weight


@pytest.mark.parametrize(
    ("modulus", "expected_zero_weight"), [(2, 30_465), (3, 939_352)]
)
def test_projective_counts_match_cpu(
    cuda_utility_libraries, modulus, expected_zero_weight
):
    _, proj_library = cuda_utility_libraries
    zero_weight = ctypes.c_ulonglong(0)
    total_weight = ctypes.c_ulonglong(0)

    rc = proj_library.proj_multiset_perm_n4(
        modulus, ctypes.byref(zero_weight), ctypes.byref(total_weight)
    )
    row_count, cpu_zero_weight = _projective_zero_weight(modulus)

    assert rc == 0
    assert zero_weight.value == cpu_zero_weight == expected_zero_weight
    assert total_weight.value == row_count**4
    assert 0 <= zero_weight.value <= total_weight.value


def test_rejected_inputs_do_not_mutate_outputs(cuda_utility_libraries):
    enum_library, proj_library = cuda_utility_libraries

    for order, modulus in ((2, 4), (4, 17)):
        counts = (ctypes.c_longlong * 23)(*([777] * 23))
        assert enum_library.enumerate_permanent_modp(order, modulus, counts) == -1
        assert list(counts) == [777] * 23

    one_matrix = (ctypes.c_int * 1)(0)
    counts = (ctypes.c_longlong * 23)(*([888] * 23))
    assert enum_library.sample_permanent_modp(1, 4, one_matrix, 1, counts) == -1
    assert list(counts) == [888] * 23

    for modulus in (4, 41):
        zero_weight = ctypes.c_ulonglong(0x3333333333333333)
        total_weight = ctypes.c_ulonglong(0x4444444444444444)
        assert (
            proj_library.proj_multiset_perm_n4(
                modulus, ctypes.byref(zero_weight), ctypes.byref(total_weight)
            )
            == -1
        )
        assert zero_weight.value == 0x3333333333333333
        assert total_weight.value == 0x4444444444444444
