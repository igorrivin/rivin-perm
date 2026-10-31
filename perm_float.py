"""
Floating-point permanent computation for real and complex matrices.

Uses Ryser's algorithm directly (no CRT needed for floating-point).
Supports NumPy arrays for efficient computation.
"""

import numpy as np
from numbers import Integral
from typing import Union, List

ArrayLike = Union[np.ndarray, List[List[float]], List[List[complex]]]

# Below this order, loading or compiling the Numba kernel can cost more than
# the scalar calculation. Orders at and above this measured crossover benefit
# even on a first call, while callers can still request the scalar path.
_NUMBA_PARALLEL_MIN_ORDER = 17


def _as_square_matrix(matrix: ArrayLike) -> np.ndarray:
    """Convert *matrix* to an ndarray and validate its dimensions."""
    try:
        matrix = np.asarray(matrix)
    except ValueError as exc:
        raise ValueError("Matrix must be a rectangular two-dimensional array") from exc
    if matrix.ndim == 1 and matrix.size == 0:
        matrix = matrix.reshape((0, 0))
    if matrix.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    return matrix


def _validate_num_threads(num_threads):
    if num_threads is None:
        return None
    if isinstance(num_threads, bool) or not isinstance(num_threads, Integral):
        raise TypeError("num_threads must be a positive integer or None")
    num_threads = int(num_threads)
    if num_threads <= 0:
        raise ValueError("num_threads must be positive")
    return num_threads


def _validate_schur_size(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be an integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    return n


def permanent_ryser_float(matrix: ArrayLike) -> Union[float, complex]:
    """
    Compute permanent of a real or complex matrix using Ryser's algorithm.

    This is the standard O(2^n * n) Ryser algorithm with Gray code optimization.
    Works with floating-point (real or complex) matrices.

    Uses Kahan compensated summation (Lundow-Markstrom, 2019) for the
    outer accumulation to reduce floating-point cancellation error.

    Args:
        matrix: n x n matrix (numpy array or list of lists)

    Returns:
        Permanent as float (for real matrices) or complex (for complex matrices)
    """
    matrix = _as_square_matrix(matrix)
    n = matrix.shape[0]

    if n == 0:
        return 1.0
    if n == 1:
        return matrix[0, 0]

    # Determine dtype
    dtype = matrix.dtype
    if np.issubdtype(dtype, np.complexfloating):
        result_dtype = np.complex128
    else:
        result_dtype = np.float64

    matrix = matrix.astype(result_dtype)

    # Row sums for current subset
    row_sums = np.zeros(n, dtype=result_dtype)
    result = result_dtype(0)
    comp = result_dtype(0)  # Kahan compensation
    sign = 1

    # Gray code iteration
    for i in range(1, 1 << n):
        # Find position of lowest set bit
        j = (i & -i).bit_length() - 1

        # Gray code value
        gray_i = i ^ (i >> 1)

        if (gray_i >> j) & 1:
            row_sums += matrix[:, j]
        else:
            row_sums -= matrix[:, j]

        sign = -sign
        prod = np.prod(row_sums)
        # Kahan compensated summation
        y = sign * prod - comp
        t = result + y
        comp = (t - result) - y
        result = t

    if n % 2 == 1:
        result = -result

    return result


def permanent_ryser_parallel(matrix: ArrayLike, num_threads: int = None) -> Union[float, complex]:
    """
    Parallel version of Ryser's algorithm for floating-point matrices.

    Splits the Gray-code traversal into independent contiguous chunks.

    Args:
        matrix: n x n matrix
        num_threads: number of threads (default: all available)

    Returns:
        Permanent as float or complex
    """
    matrix = _as_square_matrix(matrix)
    n = matrix.shape[0]
    num_threads = _validate_num_threads(num_threads)

    if n == 0:
        return 1.0
    if n == 1:
        return matrix[0, 0]

    if not HAS_NUMBA or n < _NUMBA_PARALLEL_MIN_ORDER:
        return permanent_ryser_float(matrix)

    # Determine if complex
    is_complex = np.issubdtype(matrix.dtype, np.complexfloating)

    previous_threads = None
    if num_threads is not None:
        previous_threads = numba.get_num_threads()
        numba.set_num_threads(num_threads)

    try:
        # A few Gray-code chunks per worker balance uneven scheduling without
        # allocating one temporary result for every one of the 2**n subsets.
        num_chunks = max(1, numba.get_num_threads() * 4)
        if is_complex:
            matrix = matrix.astype(np.complex128, copy=False)
            return _permanent_parallel_complex(matrix, num_chunks)
        matrix = matrix.astype(np.float64, copy=False)
        return _permanent_parallel_real(matrix, num_chunks)
    finally:
        if previous_threads is not None:
            numba.set_num_threads(previous_threads)


# Numba-accelerated versions
try:
    import numba
    from numba import njit, prange

    HAS_NUMBA = True

    @njit(parallel=True, cache=True)
    def _permanent_parallel_real(matrix, requested_chunks):
        n = matrix.shape[0]
        num_subsets = 1 << n
        total_subsets = num_subsets - 1
        num_chunks = min(requested_chunks, total_subsets)
        partial_results = np.zeros(num_chunks, dtype=np.float64)

        for chunk in prange(num_chunks):
            start = 1 + total_subsets * chunk // num_chunks
            stop = 1 + total_subsets * (chunk + 1) // num_chunks
            row_sums = np.zeros(n, dtype=np.float64)
            gray = start ^ (start >> 1)
            popcount = 0
            for j in range(n):
                if (gray >> j) & 1:
                    popcount += 1
                    for row in range(n):
                        row_sums[row] += matrix[row, j]

            sign = -1.0 if popcount & 1 else 1.0
            result = 0.0
            compensation = 0.0

            for subset in range(start, stop):
                if subset != start:
                    toggled = subset
                    j = 0
                    while (toggled & 1) == 0:
                        toggled >>= 1
                        j += 1
                    gray = subset ^ (subset >> 1)
                    if (gray >> j) & 1:
                        for row in range(n):
                            row_sums[row] += matrix[row, j]
                    else:
                        for row in range(n):
                            row_sums[row] -= matrix[row, j]
                    sign = -sign

                prod = 1.0
                for row in range(n):
                    prod *= row_sums[row]

                value = sign * prod - compensation
                updated = result + value
                compensation = (updated - result) - value
                result = updated

            partial_results[chunk] = result

        result = 0.0
        compensation = 0.0
        for chunk in range(num_chunks):
            value = partial_results[chunk] - compensation
            updated = result + value
            compensation = (updated - result) - value
            result = updated

        if n % 2 == 1:
            result = -result

        return result

    @njit(parallel=True, cache=True)
    def _permanent_parallel_complex(matrix, requested_chunks):
        n = matrix.shape[0]
        num_subsets = 1 << n
        total_subsets = num_subsets - 1
        num_chunks = min(requested_chunks, total_subsets)
        partial_results = np.zeros(num_chunks, dtype=np.complex128)

        for chunk in prange(num_chunks):
            start = 1 + total_subsets * chunk // num_chunks
            stop = 1 + total_subsets * (chunk + 1) // num_chunks
            row_sums = np.zeros(n, dtype=np.complex128)
            gray = start ^ (start >> 1)
            popcount = 0
            for j in range(n):
                if (gray >> j) & 1:
                    popcount += 1
                    for row in range(n):
                        row_sums[row] += matrix[row, j]

            sign = -1.0 if popcount & 1 else 1.0
            result = 0.0 + 0.0j
            compensation = 0.0 + 0.0j

            for subset in range(start, stop):
                if subset != start:
                    toggled = subset
                    j = 0
                    while (toggled & 1) == 0:
                        toggled >>= 1
                        j += 1
                    gray = subset ^ (subset >> 1)
                    if (gray >> j) & 1:
                        for row in range(n):
                            row_sums[row] += matrix[row, j]
                    else:
                        for row in range(n):
                            row_sums[row] -= matrix[row, j]
                    sign = -sign

                prod = 1.0 + 0.0j
                for row in range(n):
                    prod *= row_sums[row]

                value = sign * prod - compensation
                updated = result + value
                compensation = (updated - result) - value
                result = updated

            partial_results[chunk] = result

        result = 0.0 + 0.0j
        compensation = 0.0 + 0.0j
        for chunk in range(num_chunks):
            value = partial_results[chunk] - compensation
            updated = result + value
            compensation = (updated - result) - value
            result = updated

        if n % 2 == 1:
            result = -result

        return result

except ImportError:
    HAS_NUMBA = False


def schur_matrix_complex(n: int) -> np.ndarray:
    """
    Build the n x n Schur matrix with complex entries.

    S[j,k] = exp(2*pi*i*j*k/n) = omega^{jk}

    Args:
        n: matrix size

    Returns:
        n x n complex numpy array
    """
    n = _validate_schur_size(n)
    if n == 0:
        return np.empty((0, 0), dtype=np.complex128)
    omega = np.exp(2j * np.pi / n)
    j, k = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return omega ** (j * k)


def permanent_schur_float(
    n: int, parallel: bool = True, num_threads: int = None
) -> complex:
    """
    Compute permanent of n x n Schur matrix using floating-point arithmetic.

    This is less accurate than the exact integer computation but much faster
    for large n where we don't need exact results.

    Args:
        n: matrix size
        parallel: use the compiled parallel kernel when available
        num_threads: number of Numba threads, or ``None`` for its current setting

    Returns:
        Permanent as complex number (imaginary part should be ~0)
    """
    n = _validate_schur_size(n)
    _validate_num_threads(num_threads)
    if n > 0 and n % 2 == 0:
        return 0.0 + 0.0j
    S = schur_matrix_complex(n)
    return permanent_float(S, parallel=parallel, num_threads=num_threads)


# Convenience function
def permanent_float(
    matrix: ArrayLike, parallel: bool = True, num_threads: int = None
) -> Union[float, complex]:
    """
    Compute permanent of a real or complex matrix.

    Args:
        matrix: n x n matrix (numpy array or list of lists)
        parallel: use parallel computation if available
        num_threads: number of Numba threads, or ``None`` for its current setting

    Returns:
        Permanent as float or complex
    """
    if parallel:
        return permanent_ryser_parallel(matrix, num_threads=num_threads)
    _validate_num_threads(num_threads)
    return permanent_ryser_float(matrix)
