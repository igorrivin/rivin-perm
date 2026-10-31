"""Numba-accelerated permanent computation with overflow-safe modular math."""

from numbers import Integral

try:
    import numba
    from numba import get_num_threads, njit, prange
    import numpy as np

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    njit = lambda *args, **kwargs: lambda function: function
    prange = range


_INT64_MAX = (1 << 63) - 1
_DIRECT_MUL_MAX_MODULUS = 3_037_000_500


def _require_numba() -> None:
    if not HAS_NUMBA:
        raise ImportError("Numba is required for this function")


def _validate_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_modulus(p: int) -> int:
    p = _validate_integer(p, "p")
    if p <= 1:
        raise ValueError("p must be greater than 1")
    if p > _INT64_MAX:
        raise OverflowError("p must fit in a signed 64-bit integer")
    return p


def _as_int64_square_matrix(matrix: "np.ndarray") -> "np.ndarray":
    _require_numba()
    try:
        matrix = np.asarray(matrix)
    except ValueError as exc:
        raise ValueError("matrix must be a rectangular two-dimensional array") from exc
    if matrix.ndim == 1 and matrix.size == 0:
        matrix = np.empty((0, 0), dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError("matrix entries must be integers")
    if np.issubdtype(matrix.dtype, np.unsignedinteger) and matrix.size:
        if int(matrix.max()) > _INT64_MAX:
            raise OverflowError("matrix entries must fit in signed 64-bit integers")
    return np.asarray(matrix, dtype=np.int64)


@njit(inline="always")
def _add_mod(a: int, b: int, p: int) -> int:
    """Return (a + b) % p without overflowing signed int64."""
    if a >= p - b:
        return a - (p - b)
    return a + b


@njit(inline="always")
def _sub_mod(a: int, b: int, p: int) -> int:
    """Return (a - b) % p for 0 <= a,b < p without overflow."""
    if a >= b:
        return a - b
    return p - (b - a)


@njit(inline="always")
def _mul_mod(a: int, b: int, p: int) -> int:
    """Return (a * b) % p without overflowing signed int64."""
    a %= p
    b %= p
    if p <= _DIRECT_MUL_MAX_MODULUS:
        return (a * b) % p

    result = 0
    while b:
        if b & 1:
            result = _add_mod(result, a, p)
        b >>= 1
        if b:
            a = _add_mod(a, a, p)
    return result


@njit(cache=True)
def _ryser_permanent_mod_p_numba(matrix: "np.ndarray", p: int) -> int:
    n = matrix.shape[0]
    if n == 0:
        return 1
    if n == 1:
        return matrix[0, 0] % p

    row_sums = np.zeros(n, dtype=np.int64)
    result = 0
    sign = 1

    for i in range(1, 1 << n):
        j = 0
        temp = i
        while (temp & 1) == 0:
            temp >>= 1
            j += 1

        gray_i = i ^ (i >> 1)
        adding = (gray_i >> j) & 1
        for row in range(n):
            entry = matrix[row, j] % p
            if adding:
                row_sums[row] = _add_mod(row_sums[row], entry, p)
            else:
                row_sums[row] = _sub_mod(row_sums[row], entry, p)

        sign = -sign
        product = 1
        for row in range(n):
            product = _mul_mod(product, row_sums[row], p)

        if sign > 0:
            result = _add_mod(result, product, p)
        else:
            result = _sub_mod(result, product, p)

    if n & 1 and result:
        result = p - result
    return result


def ryser_permanent_mod_p_numba(matrix: "np.ndarray", p: int) -> int:
    """Compute the permanent of a square integer matrix modulo ``p``."""
    matrix = _as_int64_square_matrix(matrix)
    p = _validate_modulus(p)
    if matrix.shape[0] >= 63:
        raise ValueError("matrix size must be less than 63")
    return int(_ryser_permanent_mod_p_numba(matrix, p))


@njit(cache=True)
def _build_schur_matrix_numba(n: int, omega: int, p: int) -> "np.ndarray":
    matrix = np.empty((n, n), dtype=np.int64)
    omega_j = 1
    for j in range(n):
        power = 1
        for k in range(n):
            matrix[j, k] = power
            power = _mul_mod(power, omega_j, p)
        omega_j = _mul_mod(omega_j, omega, p)
    return matrix


def build_schur_matrix_numba(n: int, omega: int, p: int) -> "np.ndarray":
    """Build the modular Schur matrix with entries ``omega**(j*k)``."""
    _require_numba()
    n = _validate_integer(n, "n")
    if n < 0:
        raise ValueError("n must be non-negative")
    p = _validate_modulus(p)
    omega = _validate_integer(omega, "omega") % p
    return _build_schur_matrix_numba(n, omega, p)


def schur_permanent_mod_p_numba(n: int, omega: int, p: int) -> int:
    """Compute the permanent of a modular Schur matrix using Numba."""
    _require_numba()
    n = _validate_integer(n, "n")
    if n < 0:
        raise ValueError("n must be non-negative")
    p = _validate_modulus(p)
    omega = _validate_integer(omega, "omega") % p
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n % 2 == 0:
        return 0
    if n >= 63:
        raise ValueError("matrix size must be less than 63")
    matrix = _build_schur_matrix_numba(n, omega, p)
    return int(_ryser_permanent_mod_p_numba(matrix, p))


@njit(parallel=True)
def _ryser_permanent_mod_p_numba_parallel(matrix: "np.ndarray", p: int) -> int:
    n = matrix.shape[0]
    if n == 0:
        return 1
    if n == 1:
        return matrix[0, 0] % p

    total_subsets = (1 << n) - 1
    num_chunks = min(total_subsets, max(1, get_num_threads() * 4))
    partial_results = np.zeros(num_chunks, dtype=np.int64)

    for chunk in prange(num_chunks):
        start = 1 + total_subsets * chunk // num_chunks
        stop = 1 + total_subsets * (chunk + 1) // num_chunks
        partial = 0

        for subset in range(start, stop):
            row_sums = np.zeros(n, dtype=np.int64)
            popcount = 0
            for j in range(n):
                if (subset >> j) & 1:
                    popcount += 1
                    for row in range(n):
                        entry = matrix[row, j] % p
                        row_sums[row] = _add_mod(row_sums[row], entry, p)

            product = 1
            for row in range(n):
                product = _mul_mod(product, row_sums[row], p)

            if popcount & 1:
                partial = _sub_mod(partial, product, p)
            else:
                partial = _add_mod(partial, product, p)

        partial_results[chunk] = partial

    result = 0
    for chunk in range(num_chunks):
        result = _add_mod(result, partial_results[chunk], p)

    if n & 1 and result:
        result = p - result
    return result


def ryser_permanent_mod_p_numba_parallel(matrix: "np.ndarray", p: int) -> int:
    """Parallel subset implementation without an exponential result buffer."""
    matrix = _as_int64_square_matrix(matrix)
    p = _validate_modulus(p)
    if matrix.shape[0] >= 63:
        raise ValueError("matrix size must be less than 63")
    return int(_ryser_permanent_mod_p_numba_parallel(matrix, p))


def warmup() -> None:
    """Compile the common Numba kernels ahead of their first real use."""
    if not HAS_NUMBA:
        return
    test_matrix = np.array([[1, 2], [3, 4]], dtype=np.int64)
    ryser_permanent_mod_p_numba(test_matrix, 1_000_000_007)
    build_schur_matrix_numba(3, 2, 7)
