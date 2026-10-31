"""
Python wrapper for CUDA-accelerated permanent computation.

Provides GPU-accelerated versions of:
- permanent_mod3_cuda: Fast mod 3 using bipedal encoding
- permanent_modp_cuda: General mod p for Schur matrices
- permanent_cuda: Full Schur permanent using CRT with GPU
"""

import numpy as np
import ctypes
import operator
import os
from typing import Optional, List, Tuple

# Find and load the CUDA libraries
_cuda_dir = os.path.dirname(os.path.abspath(__file__))

_mod3_cuda = None
_modp_cuda = None

def _load_cuda_libs():
    """Load CUDA shared libraries."""
    global _mod3_cuda, _modp_cuda

    mod3_path = os.path.join(_cuda_dir, 'perm_mod3_cuda.so')
    modp_path = os.path.join(_cuda_dir, 'perm_modp_cuda.so')

    if os.path.exists(mod3_path) and _mod3_cuda is None:
        _mod3_cuda = ctypes.CDLL(mod3_path)
        _mod3_cuda.permanent_mod3_matrix.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int
        ]
        _mod3_cuda.permanent_mod3_matrix.restype = ctypes.c_int
        if hasattr(_mod3_cuda, "cuda_runtime_available"):
            _mod3_cuda.cuda_runtime_available.argtypes = []
            _mod3_cuda.cuda_runtime_available.restype = ctypes.c_int

    if os.path.exists(modp_path) and _modp_cuda is None:
        _modp_cuda = ctypes.CDLL(modp_path)
        _modp_cuda.schur_permanent_modp_cuda.argtypes = [
            ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong
        ]
        _modp_cuda.schur_permanent_modp_cuda.restype = ctypes.c_longlong

        _modp_cuda.permanent_modp_cuda.argtypes = [
            ctypes.POINTER(ctypes.c_longlong), ctypes.c_int, ctypes.c_longlong
        ]
        _modp_cuda.permanent_modp_cuda.restype = ctypes.c_longlong

        if hasattr(_modp_cuda, "cuda_runtime_available"):
            _modp_cuda.cuda_runtime_available.argtypes = []
            _modp_cuda.cuda_runtime_available.restype = ctypes.c_int

# Try to load on import
try:
    _load_cuda_libs()
except Exception as e:
    print(f"Warning: Could not load CUDA libraries: {e}")


def is_cuda_available() -> bool:
    """Check that the CUDA library and at least one runtime device are available."""
    if _modp_cuda is None:
        return False
    probe = getattr(_modp_cuda, "cuda_runtime_available", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except (OSError, TypeError, ValueError):
        return False


def _require_current_cuda_backend(library, name: str) -> None:
    if getattr(library, "cuda_runtime_available", None) is None:
        raise RuntimeError(
            f"{name} CUDA library is outdated; rebuild it from the current sources"
        )


def _as_index(value, name: str) -> int:
    """Return an exact Python integer without accepting lossy float casts."""
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _validate_modulus(p: int) -> int:
    modulus = _as_index(p, "p")
    if not 0 < modulus <= (1 << 63) - 1:
        raise ValueError("p must be a positive signed 64-bit integer")
    return modulus


def _validate_dimension(n: int) -> int:
    dimension = _as_index(n, "n")
    if not 0 <= dimension < 64:
        raise ValueError("n must satisfy 0 <= n < 64")
    return dimension


def _normalized_square_matrix(matrix: np.ndarray, modulus: int,
                              dtype: np.dtype) -> Tuple[int, np.ndarray]:
    """Validate an integer square matrix and reduce it before fixed-width conversion."""
    try:
        array = np.asarray(matrix, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix must be a rectangular square array") from exc

    if array.ndim == 1 and array.size == 0:
        array = array.reshape((0, 0))
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array")

    n = _validate_dimension(array.shape[0])
    normalized = np.empty(array.size, dtype=dtype)
    for index, value in enumerate(array.flat):
        normalized[index] = _as_index(value, f"matrix entry {index}") % modulus

    return n, np.ascontiguousarray(normalized.reshape((n, n)))


def _raise_on_cuda_failure(result: int, operation: str) -> int:
    value = int(result)
    if value < 0:
        raise RuntimeError(f"{operation} failed; see CUDA error output for details")
    return value


def permanent_mod3_cuda(matrix: np.ndarray) -> int:
    """
    Compute permanent of integer matrix mod 3 using GPU.

    Uses bipedal F2 encoding for massive speedup.
    """
    n, normalized = _normalized_square_matrix(matrix, 3, np.int32)
    if n == 0:
        return 1
    if n == 1:
        return int(normalized[0, 0])
    if _mod3_cuda is None:
        raise RuntimeError("CUDA mod3 library not loaded. Run: "
                           "nvcc -shared -Xcompiler -fPIC -O3 -o perm_mod3_cuda.so perm_mod3_cuda.cu")
    _require_current_cuda_backend(_mod3_cuda, "mod3")

    result = _mod3_cuda.permanent_mod3_matrix(
        normalized.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), n
    )
    return _raise_on_cuda_failure(result, "CUDA permanent modulo 3")


def schur_permanent_modp_cuda(n: int, omega: int, p: int) -> int:
    """
    Compute Schur matrix permanent mod p using GPU.

    Args:
        n: Matrix size
        omega: Primitive n-th root of unity mod p
        p: Prime modulus (must satisfy p ≡ 1 mod n)

    Returns:
        perm(S_n) mod p
    """
    n = _validate_dimension(n)
    p = _validate_modulus(p)
    omega = _as_index(omega, "omega") % p
    if n == 0:
        return 1 % p
    if n == 1:
        return 1 % p
    if _modp_cuda is None:
        raise RuntimeError("CUDA modp library not loaded. Run: "
                           "nvcc -shared -Xcompiler -fPIC -O3 -o perm_modp_cuda.so perm_modp_cuda.cu")
    _require_current_cuda_backend(_modp_cuda, "modp")

    result = _modp_cuda.schur_permanent_modp_cuda(n, omega, p)
    return _raise_on_cuda_failure(result, "CUDA Schur permanent")


def permanent_modp_cuda(matrix: np.ndarray, p: int) -> int:
    """
    Compute permanent of integer matrix mod p using GPU.
    """
    p = _validate_modulus(p)
    n, normalized = _normalized_square_matrix(matrix, p, np.int64)
    if n == 0:
        return 1 % p
    if n == 1:
        return int(normalized[0, 0])
    if _modp_cuda is None:
        raise RuntimeError("CUDA modp library not loaded")
    _require_current_cuda_backend(_modp_cuda, "modp")

    result = _modp_cuda.permanent_modp_cuda(
        normalized.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)), n, p
    )
    return _raise_on_cuda_failure(result, "CUDA permanent modulo p")


# CRT primes are kept below 2**32.  This makes every product of two reduced
# residues strictly smaller than 2**64, which is the fast arithmetic range of
# the CUDA kernel.
_CRT_PRIME_LIMIT = 1 << 32

# Deterministic Miller--Rabin witnesses for every unsigned 64-bit integer.
_MILLER_RABIN_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _is_prime(value: int) -> bool:
    """Return whether *value* is prime, deterministically for 64-bit inputs."""
    value = _as_index(value, "value")
    if value < 2:
        return False

    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value == prime:
            return True
        if value % prime == 0:
            return False

    odd_part = value - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1

    for base in _MILLER_RABIN_BASES_64:
        base %= value
        if base == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def _distinct_prime_factors(value: int) -> List[int]:
    """Return the distinct prime factors of a positive integer."""
    value = _as_index(value, "value")
    if value < 1:
        raise ValueError("value must be positive")

    factors = []
    if value % 2 == 0:
        factors.append(2)
        while value % 2 == 0:
            value //= 2

    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 2
    if value > 1:
        factors.append(value)
    return factors


def _primitive_root(p: int) -> int:
    """Find a primitive root modulo the prime *p* using factors of p - 1."""
    p = _as_index(p, "p")
    if not _is_prime(p):
        raise ValueError("p must be prime")
    if p == 2:
        return 1

    group_order = p - 1
    factors = _distinct_prime_factors(group_order)
    candidate = 2
    while candidate < p:
        if all(pow(candidate, group_order // factor, p) != 1
               for factor in factors):
            return candidate
        candidate += 1
    raise ValueError(f"No primitive root found for {p}")


def _schur_prime_candidates(n: int):
    """Yield descending odd q < 2**32 satisfying q ≡ 1 (mod n)."""
    largest = _CRT_PRIME_LIMIT - 1
    candidate = largest - ((largest - 1) % n)

    # For odd n, only every other member of the progression is odd.  Skip the
    # even half up front instead of passing it through the primality test.
    if n % 2 == 1 and candidate % 2 == 0:
        candidate -= n
    step = n if n % 2 == 0 else 2 * n

    while candidate >= 2:
        yield candidate
        candidate -= step


def _find_primes_for_schur(n: int, count: int) -> List[int]:
    """Find the largest deterministic 32-bit primes q ≡ 1 (mod n)."""
    n = _validate_dimension(n)
    count = _as_index(count, "count")
    if n == 0:
        raise ValueError("n must be positive")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []

    primes = []
    for candidate in _schur_prime_candidates(n):
        if _is_prime(candidate):
            primes.append(candidate)
            if len(primes) == count:
                return primes
    raise ValueError(f"Could not find {count} suitable CRT primes for n={n}")


def _graham_lehmer_schur_congruence(n: int) -> Optional[Tuple[int, int]]:
    """Return a proven Graham--Lehmer ``(residue, modulus)`` when available.

    For prime Schur order n >= 5, Graham--Lehmer Theorem 2 gives

        perm(S_n) ≡ n! (mod n**3)

    For every exact prime power p**a || n, Theorem 4 gives divisibility by

        p**(((p**a - 1) * n) / ((p - 1) * p**a)).

    Those coprime factors form a zero-residue CRT modulus for odd composite
    n.  The exceptional prime n=3 uses the valid Theorem 4 seed (0, 3), not
    the inapplicable factorial congruence.  Even orders short-circuit to zero.
    """
    n = _validate_dimension(n)
    if n < 3 or not (n & 1):
        return None

    if n >= 5 and _is_prime(n):
        import math

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
        exponent = n // remaining
        modulus *= remaining ** exponent

    return 0, modulus


def _plan_schur_crt_primes(n: int, bound: int) -> List[int]:
    """Plan GPU primes completing exact signed CRT coverage for Schur order n."""
    n = _validate_dimension(n)
    bound = _as_index(bound, "bound")
    if n == 0:
        raise ValueError("n must be positive")
    if bound < 0:
        raise ValueError("bound must be non-negative")

    # Graham--Lehmer supplies a theorem-only CRT modulus for every odd n >= 3:
    # n**3 with a factorial residue at eligible prime orders, or a zero-residue
    # prime-power divisor for composite orders (and n == 3).  It is never a GPU
    # modulus.  Every GPU q is 1 modulo n, so it is coprime to every prime
    # factor of the theorem modulus.
    congruence = _graham_lehmer_schur_congruence(n)
    product = congruence[1] if congruence is not None else 1

    # A combined modulus product M >= 2*bound + 1 distinguishes all integers
    # in the inclusive signed interval.  Keep the exact integer condition so
    # that a bit-length estimate can never stop one prime too early.
    target = 2 * bound + 1
    if product >= target:
        return []

    primes = []
    for candidate in _schur_prime_candidates(n):
        if not _is_prime(candidate):
            continue
        primes.append(candidate)
        product *= candidate
        if product >= target:
            return primes
    raise ValueError(f"Could not cover the CRT bound for n={n}")


def _compute_omega(n: int, p: int) -> int:
    """Compute an element of exact order n modulo the prime p."""
    n = _as_index(n, "n")
    p = _as_index(p, "p")
    if n < 1:
        raise ValueError("n must be positive")
    if (p - 1) % n != 0:
        raise ValueError("p must satisfy p ≡ 1 (mod n)")

    generator = _primitive_root(p)
    omega = pow(generator, (p - 1) // n, p)

    # This follows from generator having order p - 1, but retaining the cheap
    # checks here guards the exact-order contract if the construction changes.
    if pow(omega, n, p) != 1 or any(
        pow(omega, n // factor, p) == 1
        for factor in _distinct_prime_factors(n)
    ):
        raise RuntimeError("failed to construct an n-th root of exact order")
    return omega


def _crt_reconstruct(residues: List[int], moduli: List[int], bound: int) -> int:
    """Chinese Remainder Theorem reconstruction."""
    from functools import reduce

    M = reduce(lambda a, b: a * b, moduli)
    result = 0

    for residue, modulus in zip(residues, moduli):
        Mi = M // modulus
        # Modular inverse of Mi modulo this pairwise-coprime CRT modulus.
        yi = pow(Mi, -1, modulus)
        result += residue * Mi * yi

    result = result % M

    # Adjust for signed result
    if result > M // 2:
        result -= M

    return result


def schur_permanent_bound(n: int) -> int:
    """Upper bound on |perm(S_n)|."""
    n = _validate_dimension(n)
    if n == 0:
        return 1

    # Compute ceil(sqrt(n**n)) exactly.  A floating-point approximation can
    # round down here, leaving the CRT modulus too small to guarantee a unique
    # reconstruction.
    import math
    power = n ** n
    root = math.isqrt(power)
    return root if root * root == power else root + 1


def permanent_cuda(n: int, verbose: bool = True) -> int:
    """
    Compute exact Schur matrix permanent using GPU-accelerated CRT.

    Args:
        n: Matrix size (odd n for non-zero result)
        verbose: Print progress

    Returns:
        Exact integer permanent of S_n
    """
    import time

    n = _validate_dimension(n)
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n % 2 == 0:
        return 0  # Even n always gives 0
    # Compute bound and find required primes
    bound = schur_permanent_bound(n)

    gpu_primes = _plan_schur_crt_primes(n, bound)
    congruence = _graham_lehmer_schur_congruence(n)
    if gpu_primes and not is_cuda_available():
        raise RuntimeError("CUDA libraries not available")

    if verbose:
        print(f"Computing perm(S_{n}) using {len(gpu_primes)} primes on GPU")
        print(f"GPU primes: {gpu_primes}")
        if congruence is not None:
            print(f"Graham-Lehmer CRT modulus: {congruence[1]}")
        print(f"Bound: {bound:.2e}")
        print()

    # Compute permanent mod each prime
    residues = []
    total_start = time.perf_counter()

    for i, p in enumerate(gpu_primes):
        omega = _compute_omega(n, p)

        start = time.perf_counter()
        r = schur_permanent_modp_cuda(n, omega, p)
        elapsed = time.perf_counter() - start

        residues.append(r)

        if verbose:
            print(f"  Prime {i+1}/{len(gpu_primes)}: p={p}, omega={omega}, "
                  f"perm≡{r} (mod {p}), time={elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start

    # Add the theorem-derived residue only at reconstruction time.  Neither a
    # prime-order n**3 modulus nor a composite-order divisibility modulus is
    # ever passed to the CUDA prime-modulus kernel.
    crt_residues = list(residues)
    crt_moduli = list(gpu_primes)
    if congruence is not None:
        theorem_residue, theorem_modulus = congruence
        crt_residues.append(theorem_residue)
        crt_moduli.append(theorem_modulus)

    result = _crt_reconstruct(crt_residues, crt_moduli, bound)

    if verbose:
        print()
        print(f"perm(S_{n}) = {result}")
        print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 37

    if not is_cuda_available():
        print("Building CUDA libraries...")
        os.system("nvcc -shared -Xcompiler -fPIC -O3 -o perm_mod3_cuda.so perm_mod3_cuda.cu")
        os.system("nvcc -shared -Xcompiler -fPIC -O3 -o perm_modp_cuda.so perm_modp_cuda.cu")
        _load_cuda_libs()

    result = permanent_cuda(n)
    print(f"\nFinal result: perm(S_{n}) = {result}")
