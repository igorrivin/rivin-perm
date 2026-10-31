"""
Fast permanent computation over F_3 using bipedal encoding.

Based on: "Fast computation of permanents over F_3 via F_2 arithmetic"
arXiv:2407.20205

Key insight: Encode F_3 elements as pairs (mag, sgn) in F_2 x F_2:
  0  -> (0, 0)
  1  -> (1, 0)
  -1 -> (1, 1)

For vectors in F_3^n with n < 64, pack all components into a pair of
64-bit integers. Then vector addition/subtraction becomes ~6 bitwise ops!

This gives ~80x speedup over naive modular arithmetic.
"""

from numbers import Integral
from typing import Tuple

import numpy as np


try:
    import perm_mod3_omp

    HAS_MOD3_OMP = True
except ImportError:
    perm_mod3_omp = None
    HAS_MOD3_OMP = False


def _validate_num_threads(num_threads):
    """Validate an optional native-backend thread count."""
    if num_threads is None:
        return None
    if isinstance(num_threads, bool) or not isinstance(num_threads, Integral):
        raise TypeError("num_threads must be a positive integer or None")
    num_threads = int(num_threads)
    if num_threads <= 0:
        raise ValueError("num_threads must be positive")
    return num_threads


def _as_square_integer_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert and validate a square integer matrix before any base case."""
    try:
        matrix = np.asarray(matrix)
    except ValueError as exc:
        raise ValueError("Matrix must be a rectangular two-dimensional array") from exc
    if matrix.ndim == 1 and matrix.size == 0:
        matrix = np.empty((0, 0), dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError("Matrix entries must be integers")
    return matrix


def encode_f3_element(x: int) -> Tuple[int, int]:
    """Encode a single F_3 element as (mag, sgn)."""
    x = x % 3
    if x == 0:
        return (0, 0)
    elif x == 1:
        return (1, 0)
    else:  # x == 2, which is -1 in F_3
        return (1, 1)


def decode_f3_element(mag: int, sgn: int) -> int:
    """Decode (mag, sgn) to F_3 element in {0, 1, 2}."""
    if mag == 0:
        return 0
    elif sgn == 0:
        return 1
    else:
        return 2  # -1 mod 3


def encode_matrix_f3(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode an n x n matrix over F_3 into bipedal form.

    Returns (mags, sgns) where each is an array of n 64-bit integers,
    one per column. Bit i of mags[j] is the magnitude of a[i,j].
    """
    matrix = _as_square_integer_matrix(matrix)
    n = matrix.shape[0]
    if n >= 64:
        raise ValueError("Matrix size must be less than 64 for bipedal encoding")

    # Work with matrix mod 3
    matrix = matrix % 3

    # Encode columns as pairs of 64-bit integers
    col_mags = np.zeros(n, dtype=np.uint64)
    col_sgns = np.zeros(n, dtype=np.uint64)

    for j in range(n):
        mag = np.uint64(0)
        sgn = np.uint64(0)
        for i in range(n):
            val = int(matrix[i, j]) % 3
            if val != 0:
                mag |= np.uint64(1 << i)
                if val == 2:  # -1 in F_3
                    sgn |= np.uint64(1 << i)
        col_mags[j] = mag
        col_sgns[j] = sgn

    return col_mags, col_sgns


def bipedal_add(m1: np.uint64, s1: np.uint64,
                m2: np.uint64, s2: np.uint64) -> Tuple[np.uint64, np.uint64]:
    """
    Add two bipedal-encoded F_3 vectors.

    (m1, s1) + (m2, s2) using Theorem 2.1 from the paper.
    """
    t = m1 ^ s1 ^ s2
    mag = (m2 & t) | (m1 ^ m2)
    sgn = (m2 & t) ^ s1
    return mag, sgn


def bipedal_sub(m1: np.uint64, s1: np.uint64,
                m2: np.uint64, s2: np.uint64) -> Tuple[np.uint64, np.uint64]:
    """
    Subtract two bipedal-encoded F_3 vectors.

    (m1, s1) - (m2, s2) using Theorem 2.1 from the paper.
    """
    t = s1 ^ s2
    mag = (m1 & t) | (m1 ^ m2)
    sgn = (m1 & t) ^ (m2 ^ s2)
    return mag, sgn


def permanent_mod3_ryser(matrix: np.ndarray) -> int:
    """
    Compute permanent of matrix mod 3 using bipedal Ryser algorithm.

    Uses Gray code traversal for O(n * 2^n) operations, but with
    ~80x speedup from bipedal encoding.

    Args:
        matrix: n x n integer matrix

    Returns:
        perm(matrix) mod 3, in {0, 1, 2}
    """
    matrix = _as_square_integer_matrix(matrix)
    n = matrix.shape[0]
    if n >= 64:
        raise ValueError("Matrix size must be less than 64 for bipedal encoding")
    if n == 0:
        return 1
    if n == 1:
        return int(matrix[0, 0]) % 3

    # Encode matrix columns
    col_mags, col_sgns = encode_matrix_f3(matrix)

    # Mask for the first n bits
    mask = np.uint64((1 << n) - 1)

    # Row sums start at zero: (mag=0, sgn=0)
    row_mag = np.uint64(0)
    row_sgn = np.uint64(0)

    # Accumulator for the permanent (in bipedal form)
    # We accumulate: sum of (-1)^|S| * prod(row_sums)
    result_mag = np.uint64(0)
    result_sgn = np.uint64(0)

    # Track the current sign (-1)^|S|
    subset_sign = 1  # +1 for empty set

    # Gray code iteration
    for i in range(1, 1 << n):
        # Find position of lowest set bit (the column that changes)
        j = (i & -i).bit_length() - 1

        # Gray code value tells us whether to add or subtract
        gray_i = i ^ (i >> 1)

        if (gray_i >> j) & 1:
            # Add column j
            row_mag, row_sgn = bipedal_add(row_mag, row_sgn,
                                           col_mags[j], col_sgns[j])
        else:
            # Subtract column j
            row_mag, row_sgn = bipedal_sub(row_mag, row_sgn,
                                           col_mags[j], col_sgns[j])

        # Flip the subset sign
        subset_sign = -subset_sign

        # Compute product of row sums
        # Product is nonzero iff all n magnitude bits are 1
        if (row_mag & mask) == mask:
            # All row sums are nonzero
            # Product magnitude is 1
            # Product sign is XOR (parity) of all sign bits
            prod_sgn = bin(row_sgn & mask).count('1') % 2

            # Multiply by subset_sign: if subset_sign is -1, flip prod_sgn
            if subset_sign < 0:
                prod_sgn = 1 - prod_sgn

            # Add to result: result += (1, prod_sgn)
            prod_mag = np.uint64(1)
            prod_sgn_bits = np.uint64(prod_sgn)
            result_mag, result_sgn = bipedal_add(result_mag, result_sgn,
                                                 prod_mag, prod_sgn_bits)

    # Apply (-1)^n factor
    if n % 2 == 1:
        # Negate: flip sign bit if magnitude is nonzero
        result_sgn ^= result_mag

    # Decode result
    # NumPy 1.x cannot safely coerce uint64 & a Python int in this case.
    return decode_f3_element(int(result_mag) & 1, int(result_sgn) & 1)


def permanent_mod3_naive(matrix: np.ndarray) -> int:
    """
    Compute permanent mod 3 using standard Ryser algorithm.

    This is the baseline for comparison - uses regular modular arithmetic.
    """
    matrix = _as_square_integer_matrix(matrix)
    n = matrix.shape[0]
    if n >= 64:
        raise ValueError("Matrix size must be less than 64")
    if n == 0:
        return 1
    if n == 1:
        return int(matrix[0, 0]) % 3

    matrix = matrix % 3

    row_sums = np.zeros(n, dtype=np.int64)
    result = 0
    sign = 1

    for i in range(1, 1 << n):
        j = (i & -i).bit_length() - 1
        gray_i = i ^ (i >> 1)

        if (gray_i >> j) & 1:
            row_sums = (row_sums + matrix[:, j]) % 3
        else:
            row_sums = (row_sums - matrix[:, j]) % 3

        sign = -sign

        prod = 1
        for rs in row_sums:
            prod = (prod * rs) % 3

        result = (result + sign * prod) % 3

    if n % 2 == 1:
        result = (-result) % 3

    return result % 3


# Try to create Numba-accelerated version
try:
    from numba import njit

    @njit(cache=True)
    def _permanent_mod3_numba(col_mags, col_sgns, n):
        """Numba-accelerated bipedal Ryser."""
        mask = np.uint64((1 << n) - 1)

        row_mag = np.uint64(0)
        row_sgn = np.uint64(0)
        result_mag = np.uint64(0)
        result_sgn = np.uint64(0)
        subset_sign = 1

        for i in range(1, 1 << n):
            # Find lowest set bit position
            j = 0
            temp = i
            while (temp & 1) == 0:
                temp >>= 1
                j += 1

            # Gray code
            gray_i = i ^ (i >> 1)

            m2 = col_mags[j]
            s2 = col_sgns[j]

            if (gray_i >> j) & 1:
                # Add: bipedal_add
                t = row_mag ^ row_sgn ^ s2
                new_mag = (m2 & t) | (row_mag ^ m2)
                new_sgn = (m2 & t) ^ row_sgn
            else:
                # Subtract: bipedal_sub
                t = row_sgn ^ s2
                new_mag = (row_mag & t) | (row_mag ^ m2)
                new_sgn = (row_mag & t) ^ (m2 ^ s2)

            row_mag = new_mag
            row_sgn = new_sgn
            subset_sign = -subset_sign

            # Product of row sums
            if (row_mag & mask) == mask:
                # Count bits in row_sgn & mask
                bits = row_sgn & mask
                count = 0
                while bits:
                    count += bits & 1
                    bits >>= 1
                prod_sgn = count & 1

                if subset_sign < 0:
                    prod_sgn = 1 - prod_sgn

                # Add to result
                t = result_mag ^ result_sgn ^ np.uint64(prod_sgn)
                result_mag = (np.uint64(1) & t) | (result_mag ^ np.uint64(1))
                result_sgn = (np.uint64(1) & t) ^ result_sgn

        # Apply (-1)^n
        if n & 1:
            result_sgn ^= result_mag

        # Decode
        if (result_mag & 1) == 0:
            return 0
        elif (result_sgn & 1) == 0:
            return 1
        else:
            return 2

    HAS_NUMBA = True

except ImportError:
    HAS_NUMBA = False


def permanent_mod3_fast(matrix: np.ndarray, num_threads=None) -> int:
    """Compute a permanent mod 3 with the fastest available backend.

    The optional OpenMP extension is preferred, followed by Numba and then the
    portable Python implementation. ``num_threads`` controls the OpenMP
    backend; portable fallbacks are serial.
    """
    matrix = _as_square_integer_matrix(matrix)
    num_threads = _validate_num_threads(num_threads)
    n = matrix.shape[0]
    if n >= 64:
        raise ValueError("Matrix size must be less than 64")
    if n == 0:
        return 1
    if n == 1:
        return int(matrix[0, 0]) % 3

    if HAS_MOD3_OMP:
        # Normalize before the extension's int64 conversion. This preserves
        # residues for unsigned inputs whose values exceed INT64_MAX.
        matrix_mod3 = np.asarray(matrix % 3, dtype=np.int64)
        native_threads = -1 if num_threads is None else num_threads
        return int(perm_mod3_omp.permanent_mod3(matrix_mod3, native_threads))

    if HAS_NUMBA:
        col_mags, col_sgns = encode_matrix_f3(matrix)
        return int(_permanent_mod3_numba(col_mags, col_sgns, n))

    return permanent_mod3_ryser(matrix)


def benchmark_mod3(n: int = 20, num_trials: int = 10):
    """Benchmark bipedal vs naive implementation."""
    import time

    print(f"Benchmarking permanent mod 3 for {n}x{n} matrices")
    print("=" * 50)

    # Generate random test matrices
    np.random.seed(42)
    matrices = [np.random.randint(0, 3, (n, n)) for _ in range(num_trials)]

    # Verify correctness first
    print("Verifying correctness...")
    for M in matrices[:3]:
        naive = permanent_mod3_naive(M)
        fast = permanent_mod3_ryser(M)
        if naive != fast:
            print(f"MISMATCH: naive={naive}, fast={fast}")
            return
        if HAS_NUMBA:
            numba_result = permanent_mod3_fast(M)
            if naive != numba_result:
                print(f"NUMBA MISMATCH: naive={naive}, numba={numba_result}")
                return
    print("Correctness verified.")
    print()

    # Benchmark naive
    start = time.perf_counter()
    for M in matrices:
        permanent_mod3_naive(M)
    naive_time = time.perf_counter() - start

    # Benchmark bipedal (pure Python)
    start = time.perf_counter()
    for M in matrices:
        permanent_mod3_ryser(M)
    fast_time = time.perf_counter() - start

    print(f"Naive:        {naive_time:.4f}s ({naive_time/num_trials*1000:.2f}ms per matrix)")
    print(f"Bipedal (Py): {fast_time:.4f}s ({fast_time/num_trials*1000:.2f}ms per matrix)")
    print(f"Speedup:      {naive_time/fast_time:.1f}x")

    if HAS_NUMBA:
        # Warmup
        permanent_mod3_fast(matrices[0])

        start = time.perf_counter()
        for M in matrices:
            permanent_mod3_fast(M)
        numba_time = time.perf_counter() - start

        print(f"Bipedal (Nb): {numba_time:.4f}s ({numba_time/num_trials*1000:.2f}ms per matrix)")
        print(f"Speedup:      {naive_time/numba_time:.1f}x")


def test_random_matrices():
    """Test with random integer matrices."""
    print("Testing random integer matrices mod 3")
    print("=" * 50)

    np.random.seed(123)

    for n in [5, 10, 15, 20]:
        # Generate random matrix with entries in {-10, ..., 10}
        M = np.random.randint(-10, 11, (n, n))

        # Compute permanent mod 3 both ways
        naive = permanent_mod3_naive(M)
        fast = permanent_mod3_fast(M)

        print(f"n={n:2d}: naive={naive}, fast={fast}, match={naive==fast}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ("verify", "test"):
            test_random_matrices()
        else:
            n = int(sys.argv[1])
            benchmark_mod3(n)
    else:
        n = 20
        benchmark_mod3(n)
