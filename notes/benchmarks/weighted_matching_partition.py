#!/usr/bin/env python3
"""Independent compiled exact per-activity weighted-matching baseline.

For square nonnegative integer A of order n<=20 and rational activity p/q,
the C++ subset DP produces three modular sums F_j= sum_k k^j r_k p^k q^(n-k)
(with F_0 using factor 1, including k=0). Python reconstructs them exactly.
The bound B=prod_i(q+p*sum_j Aij) drops column exclusion, so F0<=B,
F1<=nB and F2<=n^2 B. CRT primes need not be coprime to q.

This computes only a partition function and two moments per query, not the
complete coefficient vector. Build/loader time is separate; evaluate timing
includes validation, bound/prime planning, input reduction, allocations, C++
calls and CRT. The C++ calls include DP allocation/initialization. Execution
is serial CPU only. This is a research baseline, not an installed API.
"""
from __future__ import annotations

import ctypes
from fractions import Fraction
import hashlib
import json
import math
import operator
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from notes.benchmarks.polynomial_compare import crt_vectors, plan_primes


SCHEMA = "weighted-matching-partition-v1"
CPP_SOURCE = Path(__file__).with_suffix(".cpp")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_partition_baseline(output):
    """Compile into output directory, returning explicit build provenance.

    Refuse to overwrite an existing library or build records. No compilation
    or child process is performed by PartitionBaseline.evaluate itself.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = output / "weighted_matching_partition.so"
    stdout_path, stderr_path = output / "compile.stdout.log", output / "compile.stderr.log"
    manifest_path = output / "build.json"
    for path in (library, stdout_path, stderr_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"build output must be new: {path}")
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required to build the independent query baseline")
    source_hash = _sha(CPP_SOURCE)
    command = [compiler, "-O3", "-std=c++17", "-shared", "-fPIC", str(CPP_SOURCE), "-o", str(library)]
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True, text=True, timeout=30)
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - start
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"baseline compilation failed; see {stderr_path}")
    if _sha(CPP_SOURCE) != source_hash:
        raise RuntimeError("baseline source changed during compilation")
    report = {"schema": SCHEMA, "status": "BUILT", "library": str(library),
              "library_sha256": _sha(library), "source": str(CPP_SOURCE), "source_sha256": source_hash,
              "wrapper_sha256": _sha(__file__), "crt_helper_sha256": _sha(ROOT / "notes/benchmarks/polynomial_compare.py"),
              "compiler": compiler, "compiler_sha256": _sha(compiler),
              "compiler_version": version.stdout, "command": command, "compile_seconds": elapsed,
              "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
              "stdout_sha256": _sha(stdout_path), "stderr_sha256": _sha(stderr_path)}
    manifest_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def _matrix(matrix):
    if isinstance(matrix, (str, bytes)):
        raise TypeError("matrix must contain integer rows")
    try:
        rows = tuple(tuple(row) for row in matrix)
    except TypeError as error:
        raise TypeError("matrix must contain integer rows") from error
    n = len(rows)
    if n > 20:
        raise ValueError("query baseline supports order at most 20")
    if any(len(row) != n for row in rows):
        raise ValueError("query baseline requires a square matrix")
    normalized = []
    for row in rows:
        values = []
        for value in row:
            try:
                number = operator.index(value)
            except TypeError as error:
                raise TypeError("matrix entries must be integers") from error
            if number < 0:
                raise ValueError("matrix entries must be nonnegative")
            values.append(number)
        normalized.append(tuple(values))
    return tuple(normalized)


def _activity(value):
    if isinstance(value, Fraction):
        result = value
    else:
        try:
            result = Fraction(operator.index(value), 1)
        except TypeError as error:
            raise TypeError("activity must be an exact Fraction or integer") from error
    if result < 0:
        raise ValueError("activity must be nonnegative")
    return result


class PartitionBaseline:
    def __init__(self, library):
        self.library = Path(library).resolve(strict=True)
        self.library_sha256 = _sha(self.library)
        self._library = ctypes.CDLL(str(self.library))
        self._function = self._library.weighted_matching_partition_mod
        self._function.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
                                  ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.POINTER(ctypes.c_uint32)]
        self._function.restype = ctypes.c_int

    def evaluate(self, matrix, activity=1):
        """Return exact integer numerators and Fraction-valued query answers."""
        start = time.perf_counter()
        rows = _matrix(matrix)
        activity = _activity(activity)
        p, q = activity.numerator, activity.denominator
        n = len(rows)
        bound = math.prod(q + p * sum(row) for row in rows)
        bounds = (bound, n * bound, n * n * bound)
        primes = plan_primes(max(bounds), signed=False)
        setup_seconds = time.perf_counter() - start
        residues = []
        reduction_seconds = call_seconds = 0.0
        for prime in primes:
            stage = time.perf_counter()
            matrix_buffer = (ctypes.c_uint32 * (n * n))(*(value % prime for row in rows for value in row))
            result_buffer = (ctypes.c_uint32 * 3)()
            numerator, denominator = p % prime, q % prime
            reduction_seconds += time.perf_counter() - stage
            stage = time.perf_counter()
            code = self._function(matrix_buffer, n, prime, numerator, denominator, result_buffer)
            call_seconds += time.perf_counter() - stage
            if code:
                raise RuntimeError(f"compiled partition baseline failed with status {code}")
            residues.append(tuple(result_buffer))
        stage = time.perf_counter()
        moments = tuple(crt_vectors(residues, primes, bounds, signed=False))
        f0, f1, f2 = moments
        if f0 <= 0:
            raise RuntimeError("partition numerator must include a positive empty matching")
        denominator = q**n
        partition = Fraction(f0, denominator)
        mean = Fraction(f1, f0)
        variance = Fraction(f2, f0) - mean * mean
        if variance < 0:
            raise RuntimeError("reconstructed variance is negative")
        crt_seconds = time.perf_counter() - stage
        return {"schema": SCHEMA, "status": "COMPLETE", "exact": True,
                "algorithm": "cpp-occupied-column-dp-per-activity", "order": n,
                "activity": activity, "numerator_moments": moments, "denominator": denominator,
                "partition": partition, "mean": mean, "variance": variance,
                "bounds": bounds, "primes": tuple(primes), "residue_rows": tuple(residues),
                "modulus": math.prod(primes), "library_sha256": self.library_sha256,
                "setup_seconds": setup_seconds, "reduction_seconds": reduction_seconds,
                "residue_seconds": reduction_seconds + call_seconds, "call_seconds": call_seconds,
                "crt_seconds": crt_seconds, "total_seconds": time.perf_counter() - start}
