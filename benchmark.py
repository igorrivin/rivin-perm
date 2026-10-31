"""
Benchmark all backends for permanent computation.
"""

import time
import sys

# Force reload to pick up Numba
if 'perm' in sys.modules:
    del sys.modules['perm']
if 'perm_numba' in sys.modules:
    del sys.modules['perm_numba']

from perm import (
    schur_permanent,  # Pure Python
    schur_permanent_fast,
    nth_root_of_unity_mod_p,
    find_primes_congruent_to_1_mod_n,
    ryser_permanent_mod_p_fast,
    HAS_C_EXT,
    HAS_NUMBA,
    get_backend,
)

if HAS_C_EXT:
    import perm_c_ext

if HAS_NUMBA:
    from perm_numba import (
        ryser_permanent_mod_p_numba,
        build_schur_matrix_numba,
        warmup as numba_warmup,
    )
    import numpy as np
    # Warm up Numba JIT
    print("Warming up Numba JIT...", end=" ", flush=True)
    numba_warmup()
    print("done")


def benchmark_single_prime(n: int, num_runs: int = 3):
    """
    Compare general Ryser implementations on one matrix and one CPU thread.

    The optimized Python reference is portable Python. Use the generic C
    entry point as well, so the comparison does not include Schur-specific
    algorithm reductions or parallel execution.
    """
    # Find a suitable prime
    primes = find_primes_congruent_to_1_mod_n(n, 1)
    p = primes[0]
    omega = nth_root_of_unity_mod_p(n, p)

    results = {}

    # Pure Python
    # Build matrix
    matrix = []
    omega_j = 1
    for j in range(n):
        row = [0] * n
        power = 1
        for k in range(n):
            row[k] = power
            power = power * omega_j % p
        matrix.append(row)
        omega_j = omega_j * omega % p

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result_py = ryser_permanent_mod_p_fast(matrix, p)
        times.append(time.perf_counter() - start)
    results['python'] = (min(times), result_py)

    # Numba
    if HAS_NUMBA:
        mat_np = build_schur_matrix_numba(n, omega, p)
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result_numba = ryser_permanent_mod_p_numba(mat_np, p)
            times.append(time.perf_counter() - start)
        results['numba'] = (min(times), result_numba)

    # C extension
    if HAS_C_EXT:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result_c = perm_c_ext.ryser_permanent_mod_p(matrix, p, 1)
            times.append(time.perf_counter() - start)
        results['c'] = (min(times), result_c)

    return results, p


def benchmark_full(n: int, num_runs: int = 3):
    """
    Benchmark full permanent computation (with CRT).
    """
    results = {}

    # Pure Python (serial)
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = schur_permanent(n, verbose=False)
        times.append(time.perf_counter() - start)
    results['python_serial'] = (min(times), result)

    # Fast (with best backend, parallel)
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = schur_permanent_fast(n, verbose=False, parallel=True)
        times.append(time.perf_counter() - start)
    results['fast_parallel'] = (min(times), result)

    # Fast (serial, for comparison)
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = schur_permanent_fast(n, verbose=False, parallel=False)
        times.append(time.perf_counter() - start)
    results['fast_serial'] = (min(times), result)

    return results


if __name__ == "__main__":
    print(f"Active backend: {get_backend()}")
    print(f"C extension available: {HAS_C_EXT}")
    print(f"Numba available: {HAS_NUMBA}")
    print()

    print("=" * 70)
    print("SINGLE PRIME BENCHMARK (general Ryser, one CPU thread)")
    print("=" * 70)
    print(f"{'n':>4}  {'Python':>12}  {'Numba':>12}  {'C':>12}  {'C speedup':>10}")
    print("-" * 70)

    for n in [11, 13, 15, 17, 19, 21, 23]:
        results, p = benchmark_single_prime(n)

        py_time = results['python'][0]
        numba_time = results.get('numba', (None, None))[0]
        c_time = results.get('c', (None, None))[0]

        py_str = f"{py_time*1000:.2f}ms"
        numba_str = f"{numba_time*1000:.2f}ms" if numba_time else "N/A"
        c_str = f"{c_time*1000:.2f}ms" if c_time else "N/A"
        speedup_str = f"{py_time/c_time:.1f}x" if c_time else "N/A"

        print(f"{n:>4}  {py_str:>12}  {numba_str:>12}  {c_str:>12}  {speedup_str:>10}")

        # Verify all results match
        vals = [r[1] for r in results.values()]
        if len(set(vals)) > 1:
            print(f"  WARNING: Results don't match! {results}")

    print()
    print("=" * 70)
    print("FULL COMPUTATION BENCHMARK (with CRT, parallel)")
    print("=" * 70)
    print(f"{'n':>4}  {'Python serial':>14}  {'Fast parallel':>14}  {'Speedup':>10}")
    print("-" * 70)

    for n in [13, 17, 19, 21, 23]:
        results = benchmark_full(n, num_runs=2)

        py_time = results['python_serial'][0]
        fast_time = results['fast_parallel'][0]

        py_str = f"{py_time:.3f}s"
        fast_str = f"{fast_time:.3f}s"
        speedup_str = f"{py_time/fast_time:.1f}x"

        print(f"{n:>4}  {py_str:>14}  {fast_str:>14}  {speedup_str:>10}")

        # Verify
        if results['python_serial'][1] != results['fast_parallel'][1]:
            print(f"  WARNING: Results don't match!")

    print()
    print(f"Backend used for 'fast': {get_backend()}")
