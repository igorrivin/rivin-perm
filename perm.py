"""
Fast exact permanent computation for Schur matrices.

The Schur matrix S_n has entries s_{jk} = omega^{jk} where omega is a primitive
n-th root of unity. The permanent of this matrix is always an integer.

We compute it exactly using:
1. Chinese Remainder Theorem with primes p ≡ 1 (mod n)
2. Ryser's algorithm with Gray code optimization: O(2^n * n) per prime

Backends (in order of preference):
- C extension (perm_c_ext): ~10-50x faster than pure Python
- Numba JIT (perm_numba): ~5-20x faster than pure Python
- Pure Python: always available, no dependencies

Usage:
    from perm import permanent, matrix_permanent, get_backend

    # Check which backend is active
    print(get_backend())  # 'c', 'numba', or 'python'

    # Compute Schur matrix permanent
    p = permanent(23)

    # Compute arbitrary integer matrix permanent
    A = [[1, 2], [3, 4]]
    p = matrix_permanent(A)
"""

from numbers import Integral
from typing import List, Optional, Tuple
import math
import os


_INT64_MAX = (1 << 63) - 1
_OPENMP_MATRIX_MIN_ORDER = 17
_SCHUR_CRT_PRIME_FLOOR = 1 << 30

# The portable coefficient kernel is substantially more expensive per update
# than a native Glynn multiply. These conservative work ratios were chosen on
# development inputs; the dispatch benchmark retains fresh holdout comparisons.
_LOW_RANK_MIN_ORDER = 20
_LOW_RANK_MAX_RANK = 4
_LOW_RANK_MAX_STATES = 50_000
_LOW_RANK_WORK_RATIO = 200


class LowRankLimitExceeded(ValueError):
    """An explicit low-rank request exceeds the rank or coefficient budget."""


def _validate_nonnegative_int(value: int, name: str) -> int:
    """Return *value* as an int after validating a non-negative size."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    """Return *value* as an int after validating a positive count."""
    value = _validate_nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_modulus(p: int) -> int:
    """Validate a modulus accepted by the modular permanent routines."""
    if isinstance(p, bool) or not isinstance(p, Integral):
        raise TypeError("p must be an integer")
    p = int(p)
    if p <= 1:
        raise ValueError("p must be greater than 1")
    return p


def _validate_num_threads(num_threads: int) -> int:
    """Validate a thread count, where zero means automatic selection."""
    if isinstance(num_threads, bool) or not isinstance(num_threads, Integral):
        raise TypeError("num_threads must be an integer")
    num_threads = int(num_threads)
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    return num_threads


def _validate_square_matrix(matrix) -> int:
    """Validate a two-dimensional square matrix and return its order."""
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError("Matrix must be two-dimensional")
        if shape[0] != shape[1]:
            raise ValueError("Matrix must be square")

    try:
        n = len(matrix)
    except TypeError as exc:
        raise TypeError("Matrix must be a sequence of rows") from exc

    for row in matrix:
        try:
            row_length = len(row)
        except TypeError as exc:
            raise TypeError("Matrix rows must be sequences") from exc
        if row_length != n:
            raise ValueError("Matrix must be square")
    return n


def _validate_integer_matrix(matrix) -> int:
    """Validate a square matrix whose entries are integral values."""
    n = _validate_square_matrix(matrix)
    for row in matrix:
        for entry in row:
            if not isinstance(entry, Integral):
                raise TypeError("Matrix entries must be integers")
    return n


# ============================================================
# Backend detection
# ============================================================

# Try to import C extension
perm_c_ext = None
HAS_C_EXT = False
try:
    import perm_c_ext
    HAS_C_EXT = True
except ImportError:
    pass

# Try to import Numba
HAS_NUMBA = False
try:
    from perm_numba import (
        ryser_permanent_mod_p_numba,
        build_schur_matrix_numba,
        schur_permanent_mod_p_numba,
        HAS_NUMBA as _HAS_NUMBA
    )
    HAS_NUMBA = _HAS_NUMBA
    import numpy as np
except ImportError:
    pass


def get_backend() -> str:
    """Return the active backend: 'c', 'numba', or 'python'."""
    if HAS_C_EXT:
        return 'c'
    if HAS_NUMBA:
        return 'numba'
    return 'python'


def is_prime(n: int) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witnesses sufficient for all 64-bit integers
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in witnesses:
        if a >= n:
            continue

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


def find_primes_congruent_to_1_mod_n(n: int, count: int, min_prime: int = 0) -> List[int]:
    """
    Find primes p ≡ 1 (mod n).

    These primes have F_p containing a primitive n-th root of unity.
    """
    n = _validate_positive_int(n, "n")
    count = _validate_nonnegative_int(count, "count")
    min_prime = _validate_nonnegative_int(min_prime, "min_prime")

    primes = []
    candidate = max(n + 1, min_prime)
    # Start at first candidate ≡ 1 (mod n)
    remainder = candidate % n
    if remainder != 1:
        candidate += (n + 1 - remainder) % n

    while len(primes) < count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += n

    return primes


def primitive_root_mod_p(p: int) -> int:
    """
    Find a primitive root (generator) modulo prime p.

    A primitive root g has order p-1 in (Z/pZ)*.
    """
    p = _validate_modulus(p)
    if not is_prime(p):
        raise ValueError(f"p={p} must be prime")
    if p == 2:
        return 1

    # Factor p-1
    phi = p - 1
    factors = []
    temp = phi
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)

    # Find generator by testing candidates
    for g in range(2, p):
        is_generator = True
        for q in factors:
            if pow(g, phi // q, p) == 1:
                is_generator = False
                break
        if is_generator:
            return g

    raise ValueError(f"No primitive root found for p={p}")


def nth_root_of_unity_mod_p(n: int, p: int) -> int:
    """
    Find a primitive n-th root of unity in F_p.

    Requires p ≡ 1 (mod n).
    Returns omega such that omega^n ≡ 1 (mod p) and omega^k ≢ 1 for 0 < k < n.
    """
    n = _validate_positive_int(n, "n")
    p = _validate_modulus(p)
    if (p - 1) % n != 0:
        raise ValueError(f"p={p} is not ≡ 1 (mod {n})")

    g = primitive_root_mod_p(p)
    omega = pow(g, (p - 1) // n, p)

    return omega


def _graham_lehmer_schur_congruence(
    n: int,
) -> Optional[Tuple[int, int]]:
    """Return a proven Graham--Lehmer congruence for an odd Schur order.

    For prime ``n >= 5``, Graham--Lehmer Theorem 2 gives

        perm(S_n) == n! (mod n**3).

    For every exact prime power ``p**a || n``, Theorem 4 gives divisibility by

        p**(((p**a - 1) * n) / ((p - 1) * p**a)).

    We combine those coprime prime-power divisors into a zero-residue modulus
    for odd composite ``n``.  The exceptional prime ``n == 3`` does not meet
    Theorem 2's precondition, but Theorem 4 still supplies ``(0, 3)``.  Even
    orders are handled by the exact-zero shortcut and need no CRT congruence.
    """
    n = _validate_nonnegative_int(n, "n")
    if n < 3 or not (n & 1):
        return None

    if n >= 5 and is_prime(n):
        modulus = n ** 3
        return math.factorial(n) % modulus, modulus

    remaining = n
    modulus = 1
    prime = 3
    while prime * prime <= remaining:
        if remaining % prime == 0:
            prime_power = 1
            while remaining % prime == 0:
                remaining //= prime
                prime_power *= prime
            exponent = (n // prime_power) * (
                (prime_power - 1) // (prime - 1)
            )
            modulus *= prime ** exponent
        prime += 2

    if remaining > 1:
        # The remaining exact prime power has exponent one.
        exponent = n // remaining
        modulus *= remaining ** exponent

    return 0, modulus


def gray_code_next(i: int) -> int:
    """
    Return the bit position that changes between Gray code i and i+1.
    This is the position of the lowest set bit in i+1.
    """
    return (i + 1) & -(i + 1)


def gray_code_bit_position(i: int) -> int:
    """Return the index of the lowest set bit in i (0-indexed)."""
    if i == 0:
        return 0
    pos = 0
    while (i & 1) == 0:
        i >>= 1
        pos += 1
    return pos


def ryser_permanent_mod_p(matrix: List[List[int]], p: int) -> int:
    """
    Compute permanent of matrix mod p using Ryser's algorithm with Gray code.

    Ryser's formula:
        perm(A) = (-1)^n * sum_{S ⊆ [n]} (-1)^|S| * prod_{i=1}^n (sum_{j in S} a_ij)

    With Gray code, we iterate through subsets such that consecutive subsets
    differ by exactly one element, allowing O(n) update per subset.

    Total: O(2^n * n) additions, O(2^n * n) multiplications.
    """
    n = _validate_integer_matrix(matrix)
    p = _validate_modulus(p)
    # Integral scalars such as np.int64 must not retain fixed-width arithmetic
    # in the portable path, even when the modulus itself is a Python integer.
    matrix = [[int(entry) for entry in row] for row in matrix]
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0] % p

    # row_sums[i] = sum of a_ij for j in current subset S
    row_sums = [0] * n

    # Start with empty set, product of row sums is 0 (or 1 if we consider empty product)
    # We'll accumulate: sum over S of (-1)^|S| * prod_i row_sums[i]

    result = 0
    sign = 1  # (-1)^|S|, starts with |S|=0

    # Iterate through 2^n subsets via Gray code
    # Gray code g(i) for i in [0, 2^n - 1]
    # g(0) = 0 (empty set)
    # Successive g(i) differ by one bit

    # For empty set (i=0), all row_sums are 0, product is 0
    # We skip this explicitly and start accumulating from first non-empty set

    for i in range(1, 1 << n):
        # Determine which column j to toggle (add or remove from S)
        j = gray_code_bit_position(i)

        # Determine if we're adding or removing column j
        # Gray code: g(i) = i ^ (i >> 1)
        # If bit j is set in g(i), we're adding; else removing
        gray_i = i ^ (i >> 1)
        if gray_i & (1 << j):
            # Adding column j to S
            for row in range(n):
                row_sums[row] = (row_sums[row] + matrix[row][j]) % p
            sign = -sign
        else:
            # Removing column j from S
            for row in range(n):
                row_sums[row] = (row_sums[row] - matrix[row][j]) % p
            sign = -sign

        # Compute product of row sums
        prod = 1
        for row in range(n):
            prod = (prod * row_sums[row]) % p

        result = (result + sign * prod) % p

    # Apply (-1)^n factor
    if n % 2 == 1:
        result = (-result) % p

    return result


def build_schur_matrix_mod_p(n: int, omega: int, p: int) -> List[List[int]]:
    """
    Build the Schur matrix S_n with entries s_{jk} = omega^{jk} mod p.

    Indices j, k run from 0 to n-1.
    """
    n = _validate_nonnegative_int(n, "n")
    p = _validate_modulus(p)
    if isinstance(omega, Integral):
        omega = int(omega)
    omega %= p

    matrix = []
    omega_j = 1
    for _ in range(n):
        row = [0] * n
        power = 1
        for k in range(n):
            row[k] = power
            power = power * omega_j % p
        matrix.append(row)
        omega_j = omega_j * omega % p
    return matrix


def _glynn_permanent_mod_p(matrix: List[List[int]], p: int) -> int:
    """Compute a modular permanent with Glynn's formula.

    This private reference path fixes the last sign at +1 and traverses the
    other signs in Gray-code order. The modulus must be odd so that the final
    power-of-two normalization is invertible.
    """
    matrix = [[int(entry) for entry in row] for row in matrix]
    n = len(matrix)
    if n == 0:
        return 1 % p
    if n == 1:
        return matrix[0][0] % p
    if p % 2 == 0:
        raise ValueError("Glynn's formula requires an odd modulus")

    row_sums = [sum(row) % p for row in matrix]
    result = 0
    state_sign = 1

    for state in range(1 << (n - 1)):
        if state:
            column = (state & -state).bit_length() - 1
            gray = state ^ (state >> 1)
            subtracting = (gray >> column) & 1
            for row in range(n):
                twice_entry = (2 * matrix[row][column]) % p
                if subtracting:
                    row_sums[row] = (row_sums[row] - twice_entry) % p
                else:
                    row_sums[row] = (row_sums[row] + twice_entry) % p
            state_sign = -state_sign

        product = 1
        for value in row_sums:
            product = product * value % p
        result = (result + state_sign * product) % p

    inverse_two = (p + 1) // 2
    return result * pow(inverse_two, n - 1, p) % p


def _schur_permanent_mod_p_ryser(n: int, omega: int, p: int) -> int:
    """Reference full-matrix Ryser path retained for checks and fallback."""
    matrix = build_schur_matrix_mod_p(n, omega, p)
    return ryser_permanent_mod_p(matrix, p)


def _schur_permanent_mod_p_glynn(n: int, omega: int, p: int) -> int:
    """Use the equal-cofactor identity and Glynn on the deleted minor."""
    omega %= p
    if n < 3 or not (n & 1) or not (p & 1) or pow(omega, n, p) != 1 % p:
        return _schur_permanent_mod_p_ryser(n, omega, p)

    minor_n = n - 1
    minor = []
    omega_j = omega
    for _ in range(minor_n):
        row = [0] * minor_n
        power = omega_j
        for column in range(minor_n):
            row[column] = power
            power = power * omega_j % p
        minor.append(row)
        omega_j = omega_j * omega % p

    cofactor = _glynn_permanent_mod_p(minor, p)
    return (n % p) * cofactor % p


def schur_permanent_mod_p(n: int, p: int) -> int:
    """
    Compute perm(S_n) mod p where p ≡ 1 (mod n).
    """
    n = _validate_nonnegative_int(n, "n")
    p = _validate_modulus(p)
    if n == 0:
        return 1
    if n == 1:
        return 1
    if (p - 1) % n != 0:
        raise ValueError(f"p={p} is not ≡ 1 (mod {n})")
    if n % 2 == 0:
        return 0

    omega = nth_root_of_unity_mod_p(n, p)
    return _schur_permanent_mod_p_glynn(n, omega, p)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return (a, 1, 0)
    g, x, y = extended_gcd(b, a % b)
    return (g, y, x - (a // b) * y)


def chinese_remainder_theorem(residues: List[int], moduli: List[int]) -> int:
    """
    Solve the system x ≡ r_i (mod m_i) for coprime moduli.

    Returns the unique solution in [0, prod(moduli)).
    """
    if len(residues) != len(moduli):
        raise ValueError("residues and moduli must have the same length")
    if len(residues) == 0:
        return 0


    for modulus in moduli:
        if isinstance(modulus, bool) or not isinstance(modulus, Integral):
            raise TypeError("moduli must be integers")
        if modulus <= 1:
            raise ValueError("moduli must be greater than 1")

    moduli = [int(modulus) for modulus in moduli]
    residues = [
        int(residue) if isinstance(residue, Integral) else residue
        for residue in residues
    ]
    result = residues[0]
    mod = moduli[0]

    for i in range(1, len(residues)):
        r_i = residues[i]
        m_i = moduli[i]

        # Solve: result + mod * t ≡ r_i (mod m_i)
        # mod * t ≡ (r_i - result) (mod m_i)
        g, x, _ = extended_gcd(mod, m_i)
        if g != 1:
            raise ValueError("Moduli must be pairwise coprime")

        diff = (r_i - result) % m_i
        t = (diff * x) % m_i

        result = result + mod * t
        mod = mod * m_i

    return result


def schur_permanent_bound(n: int) -> int:
    """
    Upper bound on |perm(S_n)|.

    The Schur matrix S_n = sqrt(n) * U where U is the unitary DFT matrix.
    For a unitary matrix, |perm(U)| <= 1.
    Therefore: |perm(S_n)| = n^{n/2} * |perm(U)| <= n^{n/2}

    This is much tighter than the naive n! bound.
    """
    n = _validate_nonnegative_int(n, "n")
    if n == 0:
        return 1

    # Compute ceil(sqrt(n**n)) with integer arithmetic. Using a float here can
    # round downward, which would invalidate the CRT reconstruction guarantee.
    power = n ** n
    root = math.isqrt(power)
    return root if root * root == power else root + 1


def schur_permanent(n: int, verbose: bool = False) -> int:
    """
    Compute perm(S_n) exactly using CRT.

    1. Estimate bound on |perm(S_n)|
    2. Find enough primes p ≡ 1 (mod n)
    3. Compute perm(S_n) mod each prime
    4. Reconstruct via CRT
    5. Adjust for sign (result might be negative)
    """
    n = _validate_nonnegative_int(n, "n")
    if n == 0:
        return 1
    if n == 1:
        return 1  # S_1 = [[1]]
    if n % 2 == 0:
        return 0

    # Bound: |perm(S_n)| <= n!
    bound = schur_permanent_bound(n)

    # We need product of primes > 2 * bound (to handle negative values)
    target = 2 * bound + 1

    primes, prime_product, congruence = _plan_schur_exact_crt(
        n, target, n + 1
    )
    theorem_modulus = congruence[1] if congruence is not None else 1
    reconstruction_modulus = prime_product * theorem_modulus

    if verbose:
        print(f"Using {len(primes)} primes: {primes[:5]}{'...' if len(primes) > 5 else ''}")
        if congruence is not None:
            print(f"Graham-Lehmer CRT modulus: {theorem_modulus}")
        print(
            f"CRT modulus product: {reconstruction_modulus:.2e}, "
            f"bound: {bound:.2e}"
        )

    # Compute residues
    residues = []
    for p in primes:
        r = schur_permanent_mod_p(n, p)
        residues.append(r)
        if verbose:
            print(f"  perm(S_{n}) mod {p} = {r}")

    # Add the theorem-derived modulus only at reconstruction time; it is not a
    # prime modulus and must never be sent to the residue kernel.
    crt_moduli = list(primes)
    if congruence is not None:
        theorem_residue, theorem_modulus = congruence
        residues.append(theorem_residue)
        crt_moduli.append(theorem_modulus)

    result = chinese_remainder_theorem(residues, crt_moduli)

    # Adjust for sign: if result > bound, the true value is negative
    if result > bound:
        result = result - reconstruction_modulus

    return result


# ============================================================
# Optimized Ryser with micro-optimizations
# ============================================================

def ryser_permanent_mod_p_fast(matrix: List[List[int]], p: int) -> int:
    """
    Compute permanent of matrix mod p using Ryser's algorithm.
    Micro-optimized version with fewer function calls and local variable caching.
    """
    n = _validate_integer_matrix(matrix)
    p = _validate_modulus(p)
    if n == 0:
        return 1
    if n == 1:
        return int(matrix[0][0]) % p

    # Transpose matrix for column-major access (better cache locality)
    cols = [[int(matrix[row][col]) for row in range(n)] for col in range(n)]

    row_sums = [0] * n
    result = 0
    sign = 1

    # Precompute masks for Gray code
    for i in range(1, 1 << n):
        # Find toggled bit: position of lowest set bit in i
        j = (i & -i).bit_length() - 1

        # Gray code value
        gray_i = i ^ (i >> 1)
        col = cols[j]

        if gray_i & (1 << j):
            # Adding column j
            for row in range(n):
                row_sums[row] += col[row]
        else:
            # Removing column j
            for row in range(n):
                row_sums[row] -= col[row]

        sign = -sign

        # Product of row sums mod p
        # Take mod only at the end to reduce operations
        prod = 1
        for val in row_sums:
            prod = prod * val % p

        result = (result + sign * prod) % p

    if n % 2 == 1:
        result = -result % p

    return result


def schur_permanent_mod_p_fast(n: int, p: int, num_threads: int = 0) -> int:
    """
    Compute perm(S_n) mod p using the best available backend.

    Args:
        n: matrix size
        p: prime modulus (must have p ≡ 1 mod n)
        num_threads: OpenMP threads (0 = auto, 1 = serial kernel)
    """
    n = _validate_nonnegative_int(n, "n")
    p = _validate_modulus(p)
    num_threads = _validate_num_threads(num_threads)
    if n == 0:
        return 1
    if n == 1:
        return 1
    if (p - 1) % n != 0:
        raise ValueError(f"p={p} is not ≡ 1 (mod {n})")
    if n % 2 == 0:
        return 0

    omega = nth_root_of_unity_mod_p(n, p)

    if HAS_C_EXT and p <= _INT64_MAX:
        return perm_c_ext.schur_permanent_mod_p(n, omega, p, num_threads)

    if HAS_NUMBA and p <= _INT64_MAX:
        return schur_permanent_mod_p_numba(n, omega, p)

    # Portable reference fallback.
    return _schur_permanent_mod_p_glynn(n, omega, p)


def _compute_residue(args):
    """Compute one Schur residue with a serial native kernel."""
    n, p = args
    # Avoid process-by-OpenMP oversubscription in the multiprocessing path.
    return schur_permanent_mod_p_fast(n, p, num_threads=1)


def _plan_schur_crt_primes(n: int, target: int,
                           min_prime: int) -> Tuple[List[int], int]:
    """Choose q == 1 (mod n) primes until their product reaches target."""
    primes = []
    prime_product = 1

    while prime_product < target:
        new_primes = find_primes_congruent_to_1_mod_n(n, 10, min_prime)
        for p in new_primes:
            if prime_product >= target:
                break
            primes.append(p)
            prime_product *= p
            min_prime = p + 1

    return primes, prime_product


def _plan_schur_exact_crt(
    n: int, target: int, min_prime: int
) -> Tuple[List[int], int, Optional[Tuple[int, int]]]:
    """Plan kernel primes plus an optional theorem-only CRT congruence."""
    congruence = _graham_lehmer_schur_congruence(n)
    if congruence is None:
        primes, prime_product = _plan_schur_crt_primes(
            n, target, min_prime
        )
        return primes, prime_product, None

    theorem_modulus = congruence[1]
    remaining_target = (target + theorem_modulus - 1) // theorem_modulus
    primes, prime_product = _plan_schur_crt_primes(
        n, remaining_target, min_prime
    )
    return primes, prime_product, congruence


def schur_permanent_fast(n: int, verbose: bool = False, parallel: bool = True,
                         num_threads: int = 0) -> int:
    """
    Compute perm(S_n) exactly using CRT with optimizations.

    Parallelization strategy:
    - For n <= 20: multiprocessing across primes (each with a serial kernel)
    - For n > 20: serial primes, each with the OpenMP Schur kernel

    Args:
        n: Matrix size
        verbose: Print progress
        parallel: Enable parallelization (auto-selects best strategy)
        num_threads: OpenMP threads (0 = auto-detect)
    """
    n = _validate_nonnegative_int(n, "n")
    num_threads = _validate_num_threads(num_threads)
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n % 2 == 0:
        return 0

    bound = schur_permanent_bound(n)
    target = 2 * bound + 1

    # About 30 bits per residue keeps the number of exponential kernel passes
    # small while leaving a wide margin below the native signed-64-bit limit.
    primes, prime_product, congruence = _plan_schur_exact_crt(
        n, target, max(n + 1, _SCHUR_CRT_PRIME_FLOOR)
    )
    theorem_modulus = congruence[1] if congruence is not None else 1
    reconstruction_modulus = prime_product * theorem_modulus

    if verbose:
        print(f"Using {len(primes)} primes: {primes[:5]}{'...' if len(primes) > 5 else ''}")
        if congruence is not None:
            print(f"Graham-Lehmer CRT modulus: {theorem_modulus}")
        print(
            f"CRT modulus product: {reconstruction_modulus:.2e}, "
            f"bound: {bound:.2e}"
        )

    # Disabling parallelism also disables OpenMP inside each residue. When
    # multiprocessing is enabled, every worker below likewise uses one thread.
    if not parallel:
        num_threads = 1
    elif num_threads == 0 and HAS_C_EXT:
        num_threads = perm_c_ext.get_max_threads()

    # Choose the parallelization level based on n. Large exponential kernels
    # use the full OpenMP team; small ones parallelize independent residues.

    if parallel and n > 20 and HAS_C_EXT and num_threads > 1:
        # For large n: serial primes with full OpenMP parallelization within each
        # This is faster than hybrid because the parallel Ryser doesn't scale well with fewer threads
        if verbose:
            print(f"Using OpenMP with {num_threads} threads per prime (serial primes)")
        residues = []
        for i, p in enumerate(primes):
            r = schur_permanent_mod_p_fast(n, p, num_threads=num_threads)
            residues.append(r)
            if verbose:
                print(f"  [{i+1}/{len(primes)}] perm(S_{n}) mod {p} = {r}")
    elif parallel and len(primes) > 1:
        # Small n: multiprocessing across primes with serial Gray code
        from multiprocessing import Pool, cpu_count
        num_workers = min(cpu_count(), len(primes))
        if verbose:
            print(f"Using multiprocessing with {num_workers} workers")
        with Pool(num_workers) as pool:
            residues = pool.map(_compute_residue, [(n, p) for p in primes])
    else:
        # Serial fallback
        residues = []
        for p in primes:
            r = schur_permanent_mod_p_fast(n, p, num_threads=num_threads)
            residues.append(r)
            if verbose:
                print(f"  perm(S_{n}) mod {p} = {r}")

    # The Graham--Lehmer modulus contributes only to CRT reconstruction; the
    # modular permanent backends continue to receive prime moduli exclusively.
    crt_moduli = list(primes)
    if congruence is not None:
        theorem_residue, theorem_modulus = congruence
        residues.append(theorem_residue)
        crt_moduli.append(theorem_modulus)

    result = chinese_remainder_theorem(residues, crt_moduli)

    if result > bound:
        result = result - reconstruction_modulus

    return result


# ============================================================
# General integer matrix permanent
# ============================================================

def _ryser_permanent_with_backend(
    matrix: List[List[int]], p: int, num_threads: int = 0
) -> int:
    """
    Compute permanent of matrix mod p using the best available backend.
    """
    p = _validate_modulus(p)
    num_threads = _validate_num_threads(num_threads)

    if HAS_C_EXT and p <= _INT64_MAX:
        # Older installed extensions expose only the explicit Ryser entry
        # point. New builds choose the general algorithm inside the extension.
        native_permanent = getattr(
            perm_c_ext, "permanent_mod_p", perm_c_ext.ryser_permanent_mod_p
        )
        return native_permanent(matrix, p, num_threads)

    if HAS_NUMBA and p <= _INT64_MAX:
        mat_np = np.array(matrix, dtype=np.int64)
        return ryser_permanent_mod_p_numba(mat_np, p)

    # Fall back to pure Python
    return ryser_permanent_mod_p_fast(matrix, p)


def _compute_matrix_residue(args):
    """Helper for multiprocessing with general matrices."""
    matrix, p = args
    return _ryser_permanent_with_backend(matrix, p, num_threads=1)


def matrix_permanent_bound(matrix: List[List[int]]) -> int:
    """Upper bound on |perm(A)| for integer matrix A using Hadamard-like bound."""
    n = _validate_integer_matrix(matrix)
    if n == 0:
        return 1

    # Use row-product bound: perm(A) <= prod_i (sum_j |a_ij|)
    # This is often tighter than n! * max|a_ij|^n
    bound = 1
    for row in matrix:
        # Convert before abs as well: abs(np.int64(-(1 << 63))) overflows.
        row_sum = sum(abs(int(x)) for x in row)
        bound *= max(1, row_sum)

    return bound


def _low_rank_limit(order: int, parallel_width: int) -> int:
    """Largest affordable rank, or zero to bypass even the rank probe.

    Counts both products' dense coefficient updates and elimination work.
    Account for the existing backend's parallel capacity, conservatively
    assuming ideal scaling rather than using idle-host timing observations.
    """
    if order < _LOW_RANK_MIN_ORDER:
        return 0
    generic_work = order * (1 << (order - 1))
    limit = 0
    for rank in range(1, _LOW_RANK_MAX_RANK + 1):
        states = math.comb(order + rank - 1, rank - 1)
        updates = 2 * rank * math.comb(order + rank - 1, rank)
        work = updates + order * order * rank
        if (states > _LOW_RANK_MAX_STATES or
                generic_work < _LOW_RANK_WORK_RATIO * parallel_width * work):
            break
        limit = rank
    return limit


def _low_rank_factor_mod(matrix, prime: int, max_rank: int):
    """Exact A=UV over F_p; stop as soon as the rank/state budget is exceeded.

    The echelon basis has unit pivots. Coordinates of earlier rows are padded
    when a new basis row appears. Convert entries before modular arithmetic so
    NumPy fixed-width scalars cannot overflow an intermediate expression.
    """
    order = len(matrix)
    basis, pivots, coordinates = [], [], []
    for original in matrix:
        row = [int(entry) % prime for entry in original]
        coefficients = []
        for pivot, vector in zip(pivots, basis):
            coefficient = row[pivot]
            coefficients.append(coefficient)
            if coefficient:
                row = [(entry - coefficient * value) % prime
                       for entry, value in zip(row, vector)]
        pivot = next((index for index, entry in enumerate(row) if entry), None)
        if pivot is not None:
            rank = len(basis) + 1
            states = math.comb(order + rank - 1, rank - 1)
            if rank > max_rank or states > _LOW_RANK_MAX_STATES:
                raise LowRankLimitExceeded(
                    f"rank is at least {rank}; coefficient layer needs {states} "
                    f"states (limits: rank {max_rank}, states {_LOW_RANK_MAX_STATES})"
                )
            coefficient = row[pivot]
            inverse = pow(coefficient, -1, prime)
            basis.append(tuple(entry * inverse % prime for entry in row))
            pivots.append(pivot)
            for previous in coordinates:
                previous.append(0)
            coefficients.append(coefficient)
        coordinates.append(coefficients)
    return coordinates, basis


def _low_rank_coefficients(rows, rank: int, prime: int):
    """Homogeneous coefficient layers of a product of rank-variable forms."""
    states = {(0,) * rank: 1}
    for row in rows:
        nonzero = [(index, entry) for index, entry in enumerate(row) if entry]
        following = {}
        for powers, value in states.items():
            for index, entry in nonzero:
                target = powers[:index] + (powers[index] + 1,) + powers[index + 1:]
                following[target] = (following.get(target, 0) + value * entry) % prime
        states = {powers: value for powers, value in following.items() if value}
        if not states:
            break
    return states


def _low_rank_permanent_from_factors(order: int, prime: int, factors) -> int:
    """Sum alpha! [x^alpha] prod_i(U_i x) [x^alpha] prod_j(V_:j x).

    There is no factorial division. This is an exact polynomial identity in
    every characteristic, including primes at which the integer rank drops.
    """
    coordinates, basis = factors
    rank = len(basis)
    if rank == 0:
        return int(order == 0)
    left = _low_rank_coefficients(coordinates, rank, prime)
    right = _low_rank_coefficients(zip(*basis), rank, prime)
    factorials = [1] * (order + 1)
    for index in range(2, order + 1):
        factorials[index] = factorials[index - 1] * index % prime
    result = 0
    for powers, value in left.items():
        term = value * right.get(powers, 0) % prime
        for exponent in powers:
            term = term * factorials[exponent] % prime
        result = (result + term) % prime
    return result


def _low_rank_residues(matrix, primes, max_rank: int, *, required: bool):
    """Preflight every prime before summation; None requests generic dispatch.

    A small rank modulo the first prime does not certify small rational rank.
    If any reconstruction prime exceeds the budget, auto returns to the full
    existing generic/parallel path. Explicit requests raise instead. Only this
    expected budget exception is caught; arithmetic/backend errors propagate.
    """
    factors = []
    for prime in primes:
        try:
            factors.append(_low_rank_factor_mod(matrix, prime, max_rank))
        except LowRankLimitExceeded:
            if required:
                raise
            return None
    return [_low_rank_permanent_from_factors(len(matrix), prime, factor)
            for prime, factor in zip(primes, factors)]


def matrix_permanent(
    matrix: List[List[int]],
    verbose: bool = False,
    parallel: bool = True,
    num_threads: int = 0,
    *,
    method: str = "auto",
) -> int:
    """
    Compute the permanent of an integer matrix exactly.

    Uses certified CRT with an exact low-rank specialization when cost-effective,
    otherwise the available general Ryser/Glynn backend.

    Args:
        matrix: Square integer matrix as list of lists
        verbose: Print progress information
        parallel: Use available process or OpenMP parallelism
        num_threads: OpenMP threads per residue (0 = auto-detect)
        method: 'auto', 'generic' to bypass low-rank dispatch, or 'low_rank' to
            require the specialization. The explicit method supports rank at
            most four and at most 50,000 coefficients per side, or raises
            LowRankLimitExceeded. Low-rank residues currently run serially;
            parallel and num_threads control the generic path.

    Returns:
        The exact permanent as a Python integer
    """
    n = _validate_integer_matrix(matrix)
    num_threads = _validate_num_threads(num_threads)
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.lower()
    if method not in ("auto", "generic", "low_rank"):
        raise ValueError("method must be one of: 'auto', 'generic', 'low_rank'")
    if n == 0:
        return 1
    if n == 1:
        return int(matrix[0][0])

    bound = matrix_permanent_bound(matrix)
    target = 2 * bound + 1

    # Find enough primes (any primes work for integer matrices)
    primes = []
    prime_product = 1
    candidate = 2**30  # Start with large primes for efficiency

    while prime_product < target:
        if is_prime(candidate):
            primes.append(candidate)
            prime_product *= candidate
        candidate += 1

    if verbose:
        print(f"Using {len(primes)} primes")
        print(f"Prime product: {prime_product:.2e}, bound: {bound:.2e}")

    # Reduce matrix entries modulo each prime
    def reduce_matrix(mat, p):
        return [[int(x) % p for x in row] for row in mat]

    if not parallel or method == "low_rank":
        backend_threads = 1
    elif HAS_C_EXT:
        backend_threads = num_threads or perm_c_ext.get_max_threads()
    else:
        backend_threads = num_threads

    # Large Gray-code jobs scale better by giving all OpenMP threads to one
    # residue at a time. For smaller matrices, process-level parallelism avoids
    # repeated OpenMP startup overhead and remains useful to portable backends.
    threaded_residues = (
        parallel
        and HAS_C_EXT
        and n >= _OPENMP_MATRIX_MIN_ORDER
        and backend_threads > 1
        and len(primes) < backend_threads
    )

    residues = None
    if method == "low_rank":
        residues = _low_rank_residues(matrix, primes, _LOW_RANK_MAX_RANK, required=True)
    elif method == "auto":
        if threaded_residues:
            parallel_width = backend_threads
        elif parallel and len(primes) > 1:
            parallel_width = min(os.cpu_count() or 1, len(primes))
        else:
            parallel_width = 1
        rank_limit = _low_rank_limit(n, parallel_width)
        if rank_limit:
            residues = _low_rank_residues(matrix, primes, rank_limit, required=False)

    # Compute residues
    if residues is not None:
        if verbose:
            print("Using exact low-rank coefficient computation (serial residues)")
    elif threaded_residues:
        if verbose:
            print(
                f"Using OpenMP with {backend_threads} threads per prime "
                "(serial primes)"
            )
        residues = []
        for p in primes:
            mat_mod_p = reduce_matrix(matrix, p)
            residues.append(
                _ryser_permanent_with_backend(
                    mat_mod_p, p, num_threads=backend_threads
                )
            )
    elif parallel and len(primes) > 1:
        from multiprocessing import Pool, cpu_count
        num_workers = min(cpu_count(), len(primes))
        if verbose:
            print(f"Using multiprocessing with {num_workers} workers")
        with Pool(num_workers) as pool:
            residues = pool.map(
                _compute_matrix_residue,
                [(reduce_matrix(matrix, p), p) for p in primes]
            )
    else:
        residues = []
        for p in primes:
            mat_mod_p = reduce_matrix(matrix, p)
            r = _ryser_permanent_with_backend(
                mat_mod_p, p, num_threads=backend_threads
            )
            residues.append(r)

    result = chinese_remainder_theorem(residues, primes)

    if result > bound:
        result = result - prime_product

    return result


# ============================================================
# Main interface
# ============================================================

def schur_permanent_of_size(
    n: int,
    verbose: bool = False,
    parallel: bool = True,
    num_threads: int = 0,
) -> int:
    """
    Compute the permanent of the n x n Schur matrix.

    The Schur matrix has entries S[j,k] = exp(2*pi*i*j*k/n).
    The permanent is always an integer.

    Args:
        n: Size of the matrix
        verbose: Print progress information
        parallel: Use available process or OpenMP parallelism
        num_threads: OpenMP threads per residue (0 = auto-detect)

    Returns:
        The exact permanent as a Python integer
    """
    return schur_permanent_fast(n, verbose, parallel, num_threads)


# Backwards compatibility alias
permanent = schur_permanent_of_size


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        verbose = "-v" in sys.argv or "--verbose" in sys.argv
    else:
        n = 5
        verbose = True

    print(f"Computing perm(S_{n})...")
    result = permanent(n, verbose)
    print(f"perm(S_{n}) = {result}")
