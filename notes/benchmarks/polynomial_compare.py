#!/usr/bin/env python3
"""Reproducible one-thread CPU comparisons of complete permanent polynomials.

No evaluator inputs, historical answer files, or experimental GPU kernels are
imported. See README.md for contracts, timing boundaries, and limitations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shlex
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "permanent-polynomial-comparison-v1"
FAMILIES = ("dense", "binary", "sparse", "banded", "diagonal", "block",
            "repeated_rows", "rank_one", "signed_dense")
CONTRACTS = ("affine", "principal", "rook")
MAX_ORDER = 20


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(canonical(data)).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_record(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump({"sha256": digest(payload), "payload": payload}, stream,
                  sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_record(path):
    record = json.loads(Path(path).read_text())
    if record.get("sha256") != digest(record.get("payload")):
        raise ValueError(f"checkpoint digest mismatch: {path}")
    return record["payload"]


def is_prime(number):
    """Deterministic Miller--Rabin for the benchmark's <=31-bit moduli."""
    if not 2 <= number <= 2147483647:
        return False
    for prime in (2, 3, 5, 7, 11):
        if number % prime == 0:
            return number == prime
    odd, shift = number - 1, 0
    while odd % 2 == 0:
        shift += 1
        odd //= 2
    for base in (2, 3, 5, 7, 11):
        value = pow(base, odd, number)
        if value in (1, number - 1):
            continue
        for _ in range(shift - 1):
            value = value * value % number
            if value == number - 1:
                break
        else:
            return False
    return True


def plan_primes(bound, signed):
    target = bound * (2 if signed else 1)
    primes, product, candidate = [], 1, 2147483647
    while product <= target or not primes:
        if is_prime(candidate):
            primes.append(candidate)
            product *= candidate
        candidate -= 2
    return primes


def multiply_linear(coefficients, constant, linear):
    result = [0] * (len(coefficients) + 1)
    for degree, value in enumerate(coefficients):
        result[degree] += constant * value
        result[degree + 1] += linear * value
    return result


def instance(family, n, scale, seed, contract):
    if family not in FAMILIES or contract not in CONTRACTS:
        raise ValueError("unknown matrix family or contract")
    if not 0 <= n <= MAX_ORDER or scale < 1:
        raise ValueError("invalid order or scale")
    if family == "signed_dense" and contract == "rook":
        raise ValueError("the public rook API requires nonnegative weights")
    # Hash-derived streams make each case independent of sweep ordering.
    case_seed = int(digest([family, n, scale, seed, contract]), 16)
    rng = random.Random(case_seed)

    def matrix():
        if family == "rank_one":
            left = [rng.randint(1, 7) for _ in range(n)]
            right = [rng.randint(1, 7) for _ in range(n)]
            return [[scale * left[i] * right[j] for j in range(n)] for i in range(n)]
        if family == "repeated_rows":
            rows = [[scale * rng.randint(1, 7) for _ in range(n)]
                    for _ in range(min(n, 2))]
            return [rows[i % len(rows)][:] for i in range(n)]
        result = []
        for i in range(n):
            row = []
            for j in range(n):
                present = (family in ("dense", "signed_dense") or
                           family == "binary" and rng.random() < 0.5 or
                           family == "sparse" and rng.random() < 0.2 or
                           family == "banded" and abs(i - j) <= 1 or
                           family == "diagonal" and i == j or
                           family == "block" and i // 3 == j // 3)
                value = scale * (1 if family == "binary" else rng.randint(1, 7)) if present else 0
                if family == "signed_dense" and rng.randrange(2):
                    value = -value
                row.append(value)
            result.append(row)
        return result

    linear = matrix()
    if contract == "affine":
        constant = matrix()
    elif contract == "principal":
        constant = [[int(i == j) for j in range(n)] for i in range(n)]
    else:
        constant = [[1] * n for _ in range(n)]
    return {"family": family, "n": n, "scale": scale, "seed": seed,
            "contract": contract, "constant": constant, "linear": linear}


def coefficient_bounds(case):
    """Independent row-product upper bounds; no production bound helper."""
    bounds = [1]
    for constant, linear in zip(case["constant"], case["linear"]):
        a = 1 if case["contract"] == "rook" else sum(map(abs, constant))
        bounds = multiply_linear(bounds, a, sum(map(abs, linear)))
    return bounds


def crt_vectors(rows, primes, bounds, signed):
    if not primes or len(rows) != len(primes):
        raise ValueError("missing CRT residue rows")
    product = math.prod(primes)
    if product <= max(bounds) * (2 if signed else 1):
        raise ValueError("insufficient CRT coverage")
    coefficients, modulus = [0] * len(bounds), 1
    for prime, row in zip(primes, rows):
        require_vector(row, len(bounds), prime)
        inverse = pow(modulus, -1, prime)
        for k, value in enumerate(row):
            coefficients[k] += modulus * ((value - coefficients[k]) * inverse % prime)
        modulus *= prime
    if signed:
        coefficients = [x - product if x > product // 2 else x for x in coefficients]
    if any(abs(x) > bound or not signed and x < 0
           for x, bound in zip(coefficients, bounds)):
        raise ValueError("reconstructed coefficients exceed independent bounds")
    return coefficients


def require_vector(values, length, modulus=None):
    if len(values) != length or any(type(x) is not int for x in values):
        raise ValueError("invalid complete coefficient vector")
    if modulus is not None and any(not 0 <= x < modulus for x in values):
        raise ValueError("noncanonical modular coefficient")


def require_match(actual, expected, label):
    if list(actual) != list(expected):
        raise ValueError(f"coefficient mismatch: {label}")


def interpolate(values, prime):
    """Newton forward differences at 0,...,n, converted to monomials."""
    if not is_prime(prime) or prime <= len(values) - 1:
        raise ValueError("interpolation needs a prime exceeding the degree")
    differences = list(values)
    answer, basis, factorial = [0] * len(values), [1], 1
    for degree in range(len(values)):
        weight = differences[0] * pow(factorial, -1, prime) % prime
        for k, value in enumerate(basis):
            answer[k] = (answer[k] + weight * value) % prime
        differences = [(b - a) % prime for a, b in zip(differences, differences[1:])]
        basis = [x % prime for x in multiply_linear(basis, -degree, 1)]
        factorial = factorial * (degree + 1) % prime
    return answer


class Baselines:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        pointer = ctypes.POINTER(ctypes.c_uint32)
        self.library.polynomial_ryser.argtypes = [pointer, pointer, ctypes.c_int,
                                                  ctypes.c_uint32, pointer]
        self.library.polynomial_ryser.restype = ctypes.c_int
        self.library.weighted_rook_dp.argtypes = [pointer, ctypes.c_int,
                                                 ctypes.c_uint32, pointer]
        self.library.weighted_rook_dp.restype = ctypes.c_int

    def evaluate(self, method, case, prime):
        start = time.perf_counter()
        n = case["n"]
        if not 0 <= n <= MAX_ORDER or not 3 <= prime <= 2147483647 or prime % 2 == 0:
            raise ValueError("unsupported baseline dimensions/modulus")
        arrays = []
        for key in (("linear",) if method == "rook_dp" else ("constant", "linear")):
            matrix = case[key]
            if len(matrix) != n or any(len(row) != n for row in matrix):
                raise ValueError("baseline matrices must be square")
            arrays.append((ctypes.c_uint32 * (n*n))(
                *(value % prime for row in matrix for value in row)))
        output = (ctypes.c_uint32 * (n + 1))()
        setup_end = time.perf_counter()
        if method == "rook_dp":
            status = self.library.weighted_rook_dp(arrays[0], n, prime, output)
        elif method == "polynomial_ryser":
            status = self.library.polynomial_ryser(*arrays, n, prime, output)
        else:
            raise ValueError("unknown compiled baseline")
        kernel_end = time.perf_counter()
        if status:
            raise RuntimeError(f"compiled baseline failed with status {status}")
        values = list(output)
        if case["contract"] == "rook" and method != "rook_dp":
            values = unscale_rook(values, prime)
        end = time.perf_counter()
        return {"coefficients": values, "setup_seconds": setup_end - start,
                "kernel_seconds": kernel_end - setup_end,
                "postprocess_seconds": end - kernel_end,
                "total_seconds": end - start,
                "workspace_bytes_model": (4*(1 << n) + 4*n if method == "rook_dp"
                                           else 4*(3*n + 1)),
                "workspace_model_scope": "C++ vector payload only; excludes input/output buffers, allocator, stack and Python"}


def build_baselines(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("polynomial_baselines.cpp")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    command = compiler + ["-O3", "-std=c++17", "-fPIC", "-shared",
                          str(source), "-o", str(output / "polynomial_baselines.so")]
    identity = {"source_sha256": file_digest(source), "command": command,
                "compiler": subprocess.check_output(compiler + ["--version"], text=True).splitlines()[0]}
    record_path = output / "build.json"
    if record_path.exists():
        existing = read_record(record_path)
        if existing["identity"] != identity:
            raise ValueError("baseline build identity changed; select a new output directory")
        if file_digest(output / "polynomial_baselines.so") != existing["binary_sha256"]:
            raise ValueError("baseline binary digest mismatch")
        return output / "polynomial_baselines.so", existing
    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    record = {"identity": identity, "seconds": time.perf_counter() - start,
              "stdout": completed.stdout, "stderr": completed.stderr,
              "binary_sha256": file_digest(output / "polynomial_baselines.so")}
    atomic_record(record_path, record)
    return output / "polynomial_baselines.so", record


def unscale_rook(coefficients, prime):
    n = len(coefficients) - 1
    if prime <= n:
        raise ValueError("rook factorial unscaling requires prime > order")
    return [value * pow(math.factorial(n-k), -1, prime) % prime
            for k, value in enumerate(coefficients)]


def modular_result(method, case, prime, baselines):
    if method in ("polynomial_ryser", "rook_dp"):
        return baselines.evaluate(method, case, prime)
    polynomial = importlib.import_module("perm_polynomial")
    start = time.perf_counter()
    if method == "native_polynomial":
        if case["contract"] == "principal":
            values = polynomial.principal_permanent_polynomial_mod(
                case["linear"], prime, backend="native", num_threads=1)
        else:
            values = polynomial.affine_permanent_mod(
                case["constant"], case["linear"], prime, backend="native", num_threads=1)
        call_end = time.perf_counter()
        values = list(values)
        if case["contract"] == "rook":
            values = unscale_rook(values, prime)
        end = time.perf_counter()
        return {"coefficients": values, "native_api_seconds": call_end - start,
                "postprocess_seconds": end - call_end, "total_seconds": end - start}
    if method != "scalar_interpolation":
        raise ValueError("unknown modular method")
    native = importlib.import_module("perm_c_ext")
    scalar = getattr(native, "permanent_mod_p", native.ryser_permanent_mod_p)
    setup_seconds, kernel_seconds, samples = 0., 0., []
    for x in range(case["n"] + 1):
        if x == 0 and case["contract"] in ("principal", "rook"):
            value = 1 if case["contract"] == "principal" else math.factorial(case["n"]) % prime
            samples.append({"x": 0, "value": value, "matrix_setup_seconds": 0.,
                            "scalar_native_api_seconds": 0.,
                            "source": "per(I)=1" if case["contract"] == "principal" else "per(J)=n!"})
            continue
        point_start = time.perf_counter()
        matrix = [[(c + x*l) % prime for c, l in zip(crow, lrow)]
                  for crow, lrow in zip(case["constant"], case["linear"])]
        built = time.perf_counter()
        value = int(scalar(matrix, prime, 1))
        evaluated = time.perf_counter()
        samples.append({"x": x, "value": value,
                        "matrix_setup_seconds": built - point_start,
                        "scalar_native_api_seconds": evaluated - built,
                        "source": "native_scalar_call"})
        setup_seconds += built - point_start
        kernel_seconds += evaluated - built
    interpolation_start = time.perf_counter()
    values = interpolate([sample["value"] for sample in samples], prime)
    interpolation_end = time.perf_counter()
    if case["contract"] == "rook":
        values = unscale_rook(values, prime)
    end = time.perf_counter()
    return {"coefficients": values, "scalar_samples": samples,
            "setup_seconds": setup_seconds, "scalar_native_api_seconds": kernel_seconds,
            "interpolation_seconds": interpolation_end - interpolation_start,
            "postprocess_seconds": end - interpolation_end,
            "total_seconds": end - start,
            "scalar_entry_point": scalar.__name__}


@contextmanager
def capture_public_residues(polynomial):
    """Record public exact API residues and exact small-component results.

    Benchmark process is single-threaded at the Python level. Wrappers retain
    output but do not alter computation or private production implementation.
    Instrumentation overhead is included in the public end-to-end timing.
    """
    records, saved = [], {}
    component_index = 0
    names = ("_affine_permanent_mod_validated", "_native_rook_polynomial_many",
             "_native_weighted_rook_polynomial_many")
    try:
        for name in names:
            original = getattr(polynomial, name)
            saved[name] = original

            def wrapper(*args, _original=original, _name=name, **kwargs):
                begin = time.perf_counter()
                result = _original(*args, **kwargs)
                elapsed = time.perf_counter() - begin
                affine = _name == "_affine_permanent_mod_validated"
                moduli = [args[2]] if affine else list(args[1])
                rows = [result] if affine else result
                records.append({"entry_point": _name, "primes": moduli,
                                "residue_rows": [list(row) for row in rows],
                                "format": "affine_coefficients" if affine else "rook_coefficients",
                                "call_seconds": elapsed})
                return result

            setattr(polynomial, name, wrapper)
        component_name = "_weighted_rook_polynomial_crt_validated"
        if hasattr(polynomial, component_name):
            original_component = getattr(polynomial, component_name)
            saved[component_name] = original_component

            def component_wrapper(matrix, selected, threads):
                nonlocal component_index
                begin = len(records)
                result = original_component(matrix, selected, threads)
                metadata = {"index": component_index,
                            "matrix": [list(row) for row in matrix],
                            "coefficients": list(result)}
                for capture in records[begin:]:
                    capture["weighted_component"] = metadata
                component_index += 1
                return result

            setattr(polynomial, component_name, component_wrapper)
        small_name = "_weighted_rook_small_component_validated"
        if hasattr(polynomial, small_name):
            original_small = getattr(polynomial, small_name)
            saved[small_name] = original_small

            def small_wrapper(matrix, rows, columns):
                nonlocal component_index
                begin = time.perf_counter()
                result = original_small(matrix, rows, columns)
                elapsed = time.perf_counter() - begin
                metadata = {"index": component_index, "rows": list(rows), "columns": list(columns),
                            "matrix": [[matrix[row][column] for column in columns] for row in rows],
                            "coefficients": list(result)}
                records.append({"entry_point": small_name, "primes": [], "residue_rows": [],
                                "format": "exact_rook_coefficients", "call_seconds": elapsed,
                                "weighted_component": metadata})
                component_index += 1
                return result

            setattr(polynomial, small_name, small_wrapper)
        yield records
    finally:
        for name, original in saved.items():
            setattr(polynomial, name, original)


def _weighted_support_components(matrix):
    """Independent bipartite BFS for auditing structural dispatch."""
    n = len(matrix)
    neighbors = [[] for _ in range(2*n)]
    for row in range(n):
        for column in range(n):
            if matrix[row][column]:
                neighbors[row].append(n+column)
                neighbors[n+column].append(row)
    visited, components = set(), []
    for first in range(n):
        if first in visited or not neighbors[first]:
            continue
        pending, vertices = [first], set()
        while pending:
            vertex = pending.pop()
            if vertex in vertices:
                continue
            vertices.add(vertex)
            pending.extend(neighbors[vertex])
        visited.update(vertices)
        components.append((sorted(v for v in vertices if v < n),
                           sorted(v-n for v in vertices if v >= n)))
    return components


def _small_weighted_factor_by_assignments(matrix):
    """Definition-level replay, independent of the production subset DP.

    Choose up to three vertices on the smaller side and inject them into the
    other side. Each partial matching appears exactly once. No modular or
    production evaluation is called here.
    """
    row_count = len(matrix)
    column_count = len(matrix[0]) if matrix else 0
    if (not row_count or not column_count or min(row_count, column_count) > 3 or
            any(len(row) != column_count for row in matrix)):
        raise ValueError("exact small-factor transcript has invalid dimensions")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError("exact small-factor transcript has invalid weights")
    if row_count > column_count:
        matrix = list(zip(*matrix))
        row_count, column_count = column_count, row_count
    coefficients = [1]
    for degree in range(1, row_count+1):
        total = 0
        for rows in itertools.combinations(range(row_count), degree):
            for columns in itertools.permutations(range(column_count), degree):
                total += math.prod(matrix[row][column] for row, column in zip(rows, columns))
        coefficients.append(total)
    return coefficients


def validate_public_residues(values, captures, case):
    """Replay public residue evidence, including disconnected weighted factors.

    Old captures refer to the complete polynomial. Weighted captures identify
    each exact small factor or square CRT input; the latter uses the public rigorous bound
    and characteristic-two certificate, with separate CRT arithmetic here.
    Support decomposition and coefficient convolution are checked independently
    of the production dispatcher. No native evaluation is performed by this
    validator. Compiled whole-polynomial comparisons remain separate evidence.
    """
    require_vector(values, case["n"] + 1)
    tagged = any("weighted_component" in capture for capture in captures)
    if not tagged and captures:
        # Backward-compatible full-vector transcripts, including old weighted
        # API versions and the unchanged generic affine/principal/binary APIs.
        primes = []
        for capture in captures:
            if capture["format"] not in ("affine_coefficients", "rook_coefficients"):
                raise ValueError("untagged exact component or unknown residue format")
            if len(capture["primes"]) != len(capture["residue_rows"]):
                raise ValueError("invalid public residue transcript")
            for prime, row in zip(capture["primes"], capture["residue_rows"]):
                require_vector(row, case["n"] + 1, prime)
                expected = [value % prime for value in values]
                if case["contract"] == "rook" and capture["format"] == "affine_coefficients":
                    expected = [value * math.factorial(case["n"] - k) % prime
                                for k, value in enumerate(expected)]
                require_match(row, expected, "public exact residue transcript")
                primes.append(prime)
        return primes
    if case["contract"] != "rook":
        if tagged:
            raise ValueError("weighted components attached to a non-rook contract")
        return []
    groups = []
    for capture in captures:
        metadata = capture.get("weighted_component")
        if not isinstance(metadata, dict) or type(metadata.get("index")) is not int:
            raise ValueError("missing or invalid weighted component identity")
        index = metadata["index"]
        if index == len(groups):
            groups.append({"metadata": metadata, "captures": []})
        if not groups or index != len(groups)-1 or groups[index]["metadata"] != metadata:
            raise ValueError("weighted component order or metadata changed")
        groups[index]["captures"].append(capture)
    recorded_primes = []

    def exact_group(group):
        return any(capture.get("format") == "exact_rook_coefficients" for capture in group["captures"])

    def replay_small(group, expected_rows, expected_columns):
        metadata, component_captures = group["metadata"], group["captures"]
        if len(component_captures) != 1:
            raise ValueError("exact small component requires exactly one capture")
        capture = component_captures[0]
        if (capture["format"] != "exact_rook_coefficients" or capture["primes"] != [] or
                capture["residue_rows"] != [] or
                capture["entry_point"] != "_weighted_rook_small_component_validated"):
            raise ValueError("exact small component cannot contain modular evidence")
        if (canonical(metadata.get("rows")) != canonical(expected_rows) or
                canonical(metadata.get("columns")) != canonical(expected_columns)):
            raise ValueError("exact small component vertex indices changed")
        matrix = [[case["linear"][row][column] for column in expected_columns] for row in expected_rows]
        if canonical(metadata["matrix"]) != canonical(matrix):
            raise ValueError("exact small component matrix changed")
        exact = metadata["coefficients"]
        require_vector(exact, min(len(expected_rows), len(expected_columns))+1)
        require_match(exact, _small_weighted_factor_by_assignments(matrix), "exact small-factor assignment replay")
        return exact

    def replay(group, expected_matrix):
        metadata, component_captures = group["metadata"], group["captures"]
        if canonical(metadata["matrix"]) != canonical(expected_matrix):
            raise ValueError("weighted component matrix does not match input support")
        size = len(expected_matrix)
        exact = metadata["coefficients"]
        require_vector(exact, size+1)
        rows, primes = [], []
        for capture in component_captures:
            if len(capture["primes"]) != len(capture["residue_rows"]):
                raise ValueError("invalid weighted component residue inventory")
            if capture["format"] not in ("affine_coefficients", "rook_coefficients"):
                raise ValueError("invalid weighted component residue format")
            for prime, row in zip(capture["primes"], capture["residue_rows"]):
                if type(prime) is not int or not is_prime(prime) or prime <= size:
                    raise ValueError("invalid weighted component prime")
                require_vector(row, size+1, prime)
                rows.append(unscale_rook(row, prime) if capture["format"] == "affine_coefficients" else row)
                primes.append(prime)
        if not primes or len(set(primes)) != len(primes):
            raise ValueError("missing or duplicate weighted component primes")
        polynomial = importlib.import_module("perm_polynomial")
        bounds = polynomial._weighted_rook_polynomial_bounds_validated(expected_matrix)
        parities = polynomial._weighted_rook_polynomial_parities_validated(expected_matrix)
        product = math.prod(primes)
        if 2*product <= max(bounds):
            raise ValueError("insufficient weighted component CRT/parity coverage")
        canonical_lift = crt_vectors(rows, primes, [product-1]*(size+1), False)
        lifted = [value if value % 2 == parity else value+product
                  for value, parity in zip(canonical_lift, parities)]
        if any(value > bound for value, bound in zip(lifted, bounds)):
            raise ValueError("weighted component lift exceeds its bound")
        require_match(exact, lifted, "weighted component CRT replay")
        recorded_primes.extend(primes)
        return exact

    # A single connected component that still spans the original order can
    # retain the original row/column layout, including isolated vertices.
    if (len(groups) == 1 and not exact_group(groups[0]) and
            canonical(groups[0]["metadata"]["matrix"]) == canonical(case["linear"])):
        require_match(values, replay(groups[0], case["linear"]), "complete weighted CRT input")
        return recorded_primes
    components = _weighted_support_components(case["linear"])
    if (any(exact_group(group) for group in groups) and len(components) == 1 and
            max(map(len, components[0])) == case["n"] and min(map(len, components[0])) > 1):
        raise ValueError("exact small-factor capture cannot replace full-order connected CRT")
    coefficients, group_index = [1], 0
    for rows, columns in components:
        degree = min(len(rows), len(columns))
        explicit_small = (group_index < len(groups) and exact_group(groups[group_index]) and
                          groups[group_index]["metadata"].get("rows") == rows and
                          groups[group_index]["metadata"].get("columns") == columns)
        if explicit_small:
            factor = replay_small(groups[group_index], rows, columns)
            group_index += 1
        elif degree == 1:
            # Historical dispatch did not record arithmetic for direct stars.
            factor = [1, sum(case["linear"][row][column] for row in rows for column in columns)]
        else:
            size = max(len(rows), len(columns))
            matrix = [[case["linear"][rows[i]][columns[j]] if i < len(rows) and j < len(columns) else 0
                       for j in range(size)] for i in range(size)]
            if group_index >= len(groups):
                raise ValueError("missing weighted component transcript")
            exact = replay(groups[group_index], matrix)
            if any(exact[degree+1:]):
                raise ValueError("weighted component exceeds matching degree")
            factor = exact[:degree+1]
            group_index += 1
        product = [0]*(len(coefficients)+len(factor)-1)
        for i, a in enumerate(coefficients):
            for j, b in enumerate(factor):
                product[i+j] += a*b
        coefficients = product
    if group_index != len(groups):
        raise ValueError("extra weighted component transcript")
    coefficients += [0]*(case["n"]+1-len(coefficients))
    require_match(values, coefficients, "weighted component convolution")
    return recorded_primes


def public_exact_result(case):
    polynomial = importlib.import_module("perm_polynomial")
    start = time.perf_counter()
    with capture_public_residues(polynomial) as captures:
        if case["contract"] == "affine":
            values = polynomial.affine_permanent(case["constant"], case["linear"],
                                                 backend="native", num_threads=1)
            entry = "affine_permanent"
        elif case["contract"] == "principal":
            values = polynomial.principal_permanent_polynomial(
                case["linear"], backend="native", num_threads=1)
            entry = "principal_permanent_polynomial"
        else:
            binary = all(x in (0, 1) for row in case["linear"] for x in row)
            entry = "rook_polynomial" if binary else "weighted_rook_polynomial"
            values = getattr(polynomial, entry)(case["linear"], backend="native", num_threads=1)
    end = time.perf_counter()
    residue_seconds = sum(row["call_seconds"] for row in captures if row["format"] != "exact_rook_coefficients")
    exact_component_seconds = sum(row["call_seconds"] for row in captures if row["format"] == "exact_rook_coefficients")
    primes = [prime for capture in captures for prime in capture["primes"]]
    return {"coefficients": list(values), "entry_point": entry,
            "residue_transcript": captures, "primes": primes, "prime_count": len(primes),
            "residue_api_seconds": residue_seconds,
            "exact_component_api_seconds": exact_component_seconds,
            "planning_crt_validation_instrumentation_seconds": end - start - residue_seconds - exact_component_seconds,
            "total_seconds": end - start}


def method_result(method, case, primes, bounds, signed, baselines):
    if method == "public_exact":
        return public_exact_result(case)
    start = time.perf_counter()
    residues = []
    for prime in primes:
        result = modular_result(method, case, prime, baselines)
        require_vector(result["coefficients"], case["n"] + 1, prime)
        residues.append({"prime": prime, **result})
    crt_start = time.perf_counter()
    coefficients = crt_vectors([x["coefficients"] for x in residues], primes, bounds, signed)
    end = time.perf_counter()
    return {"coefficients": coefficients, "residues": residues,
            "crt_seconds": end - crt_start, "total_seconds": end - start}


def validate_result(result, case, primes, bounds, signed):
    values = result["coefficients"]
    require_vector(values, case["n"] + 1)
    if "residues" in result:
        if [row["prime"] for row in result["residues"]] != primes:
            raise ValueError("checkpoint uses different common primes")
        exact = crt_vectors([row["coefficients"] for row in result["residues"]],
                            primes, bounds, signed)
        require_match(values, exact, "checkpoint CRT replay")
        for row in result["residues"]:
            if "scalar_samples" in row:
                samples = row["scalar_samples"]
                if [sample["x"] for sample in samples] != list(range(case["n"] + 1)):
                    raise ValueError("invalid scalar sample grid")
                replay = interpolate([sample["value"] for sample in samples], row["prime"])
                if case["contract"] == "rook":
                    replay = unscale_rook(replay, row["prime"])
                require_match(row["coefficients"], replay, "checkpoint interpolation replay")
    else:
        recorded_primes = validate_public_residues(values, result["residue_transcript"], case)
        if recorded_primes != result["primes"] or len(recorded_primes) != result["prime_count"]:
            raise ValueError("public exact prime summary mismatch")
    if any(abs(x) > bound or not signed and x < 0 for x, bound in zip(values, bounds)):
        raise ValueError("checkpoint result exceeds coefficient bounds")


def methods_for(case):
    methods = ["polynomial_ryser", "native_polynomial", "scalar_interpolation"]
    if case["contract"] == "rook":
        methods.append("rook_dp")
    methods.append("public_exact")
    return methods


def run_case(case, output, baselines, repeats, warmups, manifest_sha, deadline=None):
    start = time.perf_counter()
    bounds = coefficient_bounds(case)
    signed = any(value < 0 for key in ("constant", "linear") for row in case[key] for value in row)
    primes = plan_primes(max(bounds), signed)
    setup_seconds = time.perf_counter() - start
    identity = {"case": case, "bounds": bounds, "signed": signed, "primes": primes,
                "repeats": repeats, "warmups": warmups, "manifest_sha256": manifest_sha}
    path = Path(output) / "cases" / (digest(case) + ".json")
    if path.exists():
        record = read_record(path)
        if record["identity"] != identity:
            raise ValueError("case checkpoint configuration mismatch")
    else:
        record = {"identity": identity, "shared_planning_seconds": setup_seconds,
                  "samples": [], "complete": False}
    methods = methods_for(case)
    schedule = []
    for index in range(warmups + repeats):
        # Rotate starting method, reversing traversal every round.
        order = methods[index % len(methods):] + methods[:index % len(methods)]
        if index % 2:
            order.reverse()
        for method in order:
            schedule.append(("warmup" if index < warmups else "measurement", index, method))
    if len(record["samples"]) > len(schedule):
        raise ValueError("too many checkpoint samples")
    expected = None
    for position, sample in enumerate(record["samples"]):
        if (sample["phase"], sample["round"], sample["method"]) != schedule[position]:
            raise ValueError("checkpoint execution order mismatch")
        validate_result(sample["result"], case, primes, bounds, signed)
        if expected is None:
            expected = sample["result"]["coefficients"]
        require_match(sample["result"]["coefficients"], expected, "resumed checkpoint")
    if record["complete"] and len(record["samples"]) != len(schedule):
        raise ValueError("incomplete checkpoint marked complete")
    for phase, round_index, method in schedule[len(record["samples"]):]:
        if deadline is not None and time.monotonic() >= deadline:
            atomic_record(path, record)
            return record
        result = method_result(method, case, primes, bounds, signed, baselines)
        try:
            validate_result(result, case, primes, bounds, signed)
            if expected is None:
                expected = result["coefficients"]
            require_match(result["coefficients"], expected,
                          f"{case['contract']}/{case['family']}/n={case['n']}/{method}")
        except (ValueError, RuntimeError) as exc:
            record["failure"] = {"phase": phase, "round": round_index, "method": method,
                                 "result": result, "error": str(exc)}
            atomic_record(path, record)
            raise
        record["samples"].append({"phase": phase, "round": round_index,
                                  "method": method, "result": result})
        atomic_record(path, record)
    record["complete"] = True
    record["coefficients"] = expected
    record["median_seconds"] = {
        method: statistics.median(sample["result"]["total_seconds"]
                                  for sample in record["samples"]
                                  if sample["phase"] == "measurement" and sample["method"] == method)
        for method in methods}
    record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_record(path, record)
    return record


def run_campaign(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process owns this campaign") from exc
        return _run_campaign_locked(args, output)


def _run_campaign_locked(args, output):
    if max(args.sizes) > 12 and not args.allow_large:
        raise ValueError("sizes above 12 require --allow-large after pilot calibration")
    if min(args.sizes) < 0 or max(args.sizes) > MAX_ORDER:
        raise ValueError(f"sizes must lie between 0 and {MAX_ORDER}")
    if args.repeats < 1 or args.warmups < 1 or min(args.scales) < 1:
        raise ValueError("positive scales, repeats, and warmups are required")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    polynomial = importlib.import_module("perm_polynomial")
    scalar = importlib.import_module("perm_c_ext")
    if not polynomial.HAS_POLYNOMIAL_EXT:
        raise RuntimeError("build the native polynomial extension before benchmarking")
    library, build = build_baselines(output / "build")
    baselines = Baselines(library)
    sources = [Path(__file__), Path(__file__).with_name("polynomial_baselines.cpp"),
               ROOT / "perm_polynomial.py", ROOT / "perm_polynomial_omp.c", ROOT / "perm_c_omp.c"]
    source_hashes = {str(path.relative_to(ROOT)): file_digest(path) for path in sources}
    config = {"sizes": args.sizes, "scales": args.scales, "families": args.families,
              "contracts": args.contracts, "seed": args.seed, "repeats": args.repeats,
              "warmups": args.warmups, "threads": 1}
    cpu_info_path = Path("/proc/cpuinfo")
    cpu_info = cpu_info_path.read_text().split("\n\n")[0] if cpu_info_path.exists() else None
    identity = {"schema": SCHEMA, "config": config, "sources": source_hashes,
                "baseline_build_identity": build["identity"],
                "baseline_binary_sha256": build["binary_sha256"],
                "native_binaries": {Path(module.__file__).name: file_digest(module.__file__)
                                    for module in (scalar, polynomial._polynomial_ext)},
                "python": sys.version, "platform": platform.platform(),
                "machine": platform.machine(), "cpu_info_first_processor": cpu_info,
                "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "thread_environment": {key: os.environ[key] for key in
                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                        "NUMEXPR_NUM_THREADS", "OMP_DYNAMIC")}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_record(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("campaign inputs, sources, hardware, or build changed; choose a new output directory")
        for relative, expected_hash in source_hashes.items():
            archived = output / "sources" / relative
            if not archived.exists() or file_digest(archived) != expected_hash:
                raise ValueError(f"archived source digest mismatch: {relative}")
    else:
        manifest = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "initial_invocation": sys.argv,
                    "build_environment": {key: os.environ.get(key) for key in
                                          ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")},
                    "timing_scope": "warm imports; each complete method includes all common-prime calls and CRT; public exact includes its own planning and captured actual primes; excludes compilation, shared planning, checkpoint IO",
                    "memory_scope": "process RSS is cumulative across methods/cases, not per-method peak; C++ workspace formulas count vector payload only",
                    "contention": "CPU comparisons may overlap independent GPU verification on the same host; timings are pilot evidence, not isolated publication measurements"}
        for source in sources:
            destination = output / "sources" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        atomic_record(manifest_path, manifest)
    manifest_sha = digest(manifest)
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
    summaries, complete = [], True
    for n in args.sizes:
        for scale in args.scales:
            for contract in args.contracts:
                for family in args.families:
                    if contract == "rook" and family == "signed_dense":
                        continue
                    case = instance(family, n, scale, args.seed, contract)
                    record = run_case(case, output, baselines, args.repeats, args.warmups,
                                      manifest_sha, deadline)
                    summaries.append({"case_sha256": digest(case), "contract": contract,
                                      "family": family, "n": n, "scale": scale,
                                      "complete": record["complete"],
                                      "common_prime_count": len(record["identity"]["primes"]),
                                      "median_seconds": record.get("median_seconds")})
                    print(json.dumps(summaries[-1], sort_keys=True), flush=True)
                    if not record["complete"]:
                        complete = False
                        break
                if not complete: break
            if not complete: break
        if not complete: break
    summary = {"schema": SCHEMA, "manifest_sha256": manifest_sha,
               "complete": complete, "cases": summaries,
               "last_updated_utc": datetime.now(timezone.utc).isoformat()}
    atomic_record(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 9, 12])
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 100])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--contracts", nargs="+", choices=CONTRACTS, default=list(CONTRACTS))
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=600,
                        help="stop/resume between methods; not a per-kernel timeout (0 disables)")
    args = parser.parse_args(argv)
    summary = run_campaign(args)
    print(json.dumps({"complete": summary["complete"], "case_count": len(summary["cases"]),
                      "output": str(Path(args.output).resolve())}), flush=True)
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
