"""Source-only CUDA acceleration for exact zero-one permanents.

The companion ``perm_binary_cuda.cu`` library is experimental and is not
compiled or shipped by the Python package.  From a source checkout, build it
manually with a CUDA toolkit before using this module::

    nvcc -shared -Xcompiler -fPIC -O3 -std=c++17 \
        -o libperm_binary_cuda.so perm_binary_cuda.cu

The CUDA residue kernel accepts odd primes ``n < p < 2**31``.  Exact results
reuse the normalization, matching decomposition, rigorous nonnegative bound,
and descending-prime CRT plan from :mod:`perm_binary`.
"""

import ctypes
import os
from typing import Sequence

import numpy as np

import perm_binary


_cuda_directory = os.path.dirname(os.path.abspath(__file__))
_binary_cuda = None
_binary_cuda_load_error = None


def _configure_binary_cuda_library(library) -> None:
    """Configure the C ABI of an already-loaded binary CUDA library."""
    library.binary_permanent_modp_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_longlong,
    ]
    library.binary_permanent_modp_cuda.restype = ctypes.c_longlong
    if hasattr(library, "cuda_runtime_available"):
        library.cuda_runtime_available.argtypes = []
        library.cuda_runtime_available.restype = ctypes.c_int


def _load_binary_cuda_library() -> None:
    """Load the manually built library when it is present beside this file."""
    global _binary_cuda, _binary_cuda_load_error
    # The ``lib`` prefix is essential: naming the plain C ABI shared object
    # ``perm_binary_cuda.so`` would shadow this Python module during import.
    path = os.path.join(_cuda_directory, "libperm_binary_cuda.so")
    if not os.path.exists(path):
        return
    try:
        library = ctypes.CDLL(path)
        _configure_binary_cuda_library(library)
    except (AttributeError, OSError) as exc:
        _binary_cuda_load_error = exc
        return
    _binary_cuda = library
    _binary_cuda_load_error = None


_load_binary_cuda_library()


def is_binary_cuda_available() -> bool:
    """Return whether the manual library and a CUDA device are available."""
    if _binary_cuda is None:
        return False
    probe = getattr(_binary_cuda, "cuda_runtime_available", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except (OSError, TypeError, ValueError):
        return False


def _require_binary_cuda() -> None:
    if _binary_cuda is None:
        detail = (
            f" ({_binary_cuda_load_error})"
            if _binary_cuda_load_error is not None
            else ""
        )
        raise RuntimeError(
            "binary CUDA library is unavailable; build libperm_binary_cuda.so "
            "manually from perm_binary_cuda.cu" + detail
        )
    if getattr(_binary_cuda, "cuda_runtime_available", None) is None:
        raise RuntimeError(
            "binary CUDA library is outdated; rebuild it from the current "
            "perm_binary_cuda.cu"
        )
    if not is_binary_cuda_available():
        raise RuntimeError("no CUDA runtime device is available")


def _validate_cuda_prime(p: int, order: int) -> int:
    modulus = perm_binary._validate_residue_prime(p, order)
    if modulus >= 1 << 31:
        raise ValueError("the binary CUDA modulus must be less than 2**31")
    return modulus


def _cuda_residue(matrix: Sequence[Sequence[int]], modulus: int) -> int:
    """Call the C ABI and enforce its success/residue sentinel contract."""
    _require_binary_cuda()
    normalized = np.ascontiguousarray(matrix, dtype=np.uint8)
    order = normalized.shape[0]
    result = int(
        _binary_cuda.binary_permanent_modp_cuda(
            normalized.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            order,
            modulus,
        )
    )
    if result < 0:
        raise RuntimeError(
            "CUDA binary-permanent residue failed; see CUDA error output "
            "for details"
        )
    if result >= modulus:
        raise RuntimeError(
            "CUDA binary-permanent backend returned an invalid residue"
        )
    return result


def _cuda_exact_permanent(
    matrix: Sequence[Sequence[int]], bound: int
) -> int:
    """Reconstruct one matching block in the certified interval [0,bound]."""
    result = 0
    product = 1
    for prime in perm_binary._crt_primes_above_bound(bound):
        residue = _cuda_residue(matrix, prime)
        correction = (
            (residue - result) % prime
        ) * pow(product, -1, prime) % prime
        result += product * correction
        product *= prime
    if result > bound:
        raise RuntimeError(
            "CUDA binary-permanent residues reconstructed outside the bound"
        )
    return result


def binary_permanent_mod_p_cuda(
    matrix, p: int, *, strict: bool = True
) -> int:
    """Compute a zero-one permanent modulo an odd 31-bit prime on CUDA.

    Matching feasibility and SCC factorization are performed on the CPU first,
    exactly as in :func:`perm_binary.binary_permanent_mod_p`.  Trivial and
    infeasible inputs therefore do not require a CUDA library or device.
    """
    normalized = perm_binary._normalize_binary_matrix(matrix, strict=strict)
    modulus = _validate_cuda_prime(p, len(normalized))
    if not normalized:
        return 1 % modulus
    if len(normalized) == 1:
        return normalized[0][0] % modulus

    feasible, blocks = perm_binary._matching_blocks(
        perm_binary._row_masks(normalized)
    )
    if not feasible:
        return 0

    result = 1
    for block in blocks:
        if len(block) == 1:
            continue
        if len(block) > 63:
            raise ValueError(
                "CUDA binary-permanent blocks must have order at most 63"
            )
        block_matrix = perm_binary._unpack_masks(block)
        result = result * _cuda_residue(block_matrix, modulus) % modulus
    return result


def binary_permanent_cuda(matrix, *, strict: bool = True) -> int:
    """Compute an exact nonnegative zero-one permanent with CUDA residues."""
    normalized = perm_binary._normalize_binary_matrix(matrix, strict=strict)
    if not normalized:
        return 1
    if len(normalized) == 1:
        return normalized[0][0]

    feasible, blocks = perm_binary._matching_blocks(
        perm_binary._row_masks(normalized)
    )
    if not feasible:
        return 0

    result = 1
    for block in blocks:
        if len(block) == 1:
            continue
        if len(block) > 63:
            raise ValueError(
                "CUDA binary-permanent blocks must have order at most 63"
            )
        block_matrix = perm_binary._unpack_masks(block)
        bound = perm_binary._normalized_bound(block_matrix)
        result *= _cuda_exact_permanent(block_matrix, bound)
    return result


__all__ = [
    "binary_permanent_cuda",
    "binary_permanent_mod_p_cuda",
    "is_binary_cuda_available",
]
