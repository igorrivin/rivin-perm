#!/usr/bin/env python3
"""Compiled exact weighted partial matchings with a bounded native workspace.

The C++ recurrence keeps occupied columns at the current row boundary and a
cardinality polynomial for each state. Arithmetic is modulo odd 31-bit primes;
Python CRT recovers nonnegative integer coefficients with a rigorous bound.
Input weights may be arbitrary nonnegative integers. Reusable bit positions
allow more than 64 global columns, but at most 64 columns may be live during
one row. This is a serial research/source tool, not a public API backend.

max_workspace_bytes caps simultaneous requested C++ heap allocation bytes:
hash nodes/buckets and coefficient arrays, including both layers and transient
rehashes. It excludes allocator overhead, stack, Python objects, input/output
buffers, planning and retained CRT transcripts. It is not a process RSS cap or
a time limit. max_states separately caps states in each layer. Limits raise
exceptions and never return a partial coefficient vector as an exact result.

The existing Python frontier shares only input normalization and row ordering;
its counting recurrence remains a separate reference. Source and build identity
are recorded. Building/loading is explicit and outside evaluate's timings.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import operator
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from notes.benchmarks.polynomial_compare import crt_vectors, is_prime
from notes.benchmarks.weighted_matching_frontier import _greedy_order, _snapshot


SCHEMA = "weighted-matching-frontier-native-v1"
CPP_SOURCE = Path(__file__).with_suffix(".cpp")
DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
WORKSPACE_SCOPE = (
    "Simultaneous requested C++ heap bytes for hash nodes, bucket arrays and "
    "coefficient vectors, including both layers and transient reallocation. "
    "Excludes allocator bookkeeping, stack, Python planning/CRT/transcripts "
    "and caller-owned input/output buffers; not an RSS or time limit."
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    paths = (Path(__file__), CPP_SOURCE,
             ROOT / "notes/benchmarks/weighted_matching_frontier.py",
             ROOT / "notes/benchmarks/polynomial_compare.py")
    return {str(path.resolve()): _sha(path) for path in paths}


def build_frontier_baseline(output):
    """Build once in a new directory; keep the compiled source and provenance."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = output / "weighted_matching_frontier_native.so"
    snapshot = output / CPP_SOURCE.name
    stdout_path, stderr_path = output / "compile.stdout.log", output / "compile.stderr.log"
    manifest = output / "build.json"
    for path in (library, snapshot, stdout_path, stderr_path, manifest):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"build output must be new: {path}")
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required to build the weighted frontier baseline")
    hashes = _source_hashes()
    source_bytes = CPP_SOURCE.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != hashes[str(CPP_SOURCE.resolve())]:
        raise RuntimeError("frontier source changed before compilation")
    snapshot.write_bytes(source_bytes)
    command = [compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
               "-Wall", "-Wextra", str(snapshot), "-o", str(library)]
    def save_logs(stdout, stderr):
        def decoded(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        stdout_path.write_text(decoded(stdout))
        stderr_path.write_text(decoded(stderr))

    try:
        version = subprocess.run([compiler, "--version"], capture_output=True,
                                 text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        save_logs(error.stdout, error.stderr)
        raise RuntimeError(f"compiler version query timed out; see {stderr_path}") from error
    if version.returncode:
        save_logs(version.stdout, version.stderr)
        raise RuntimeError(f"compiler version query failed; see {stderr_path}")
    start = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        save_logs(error.stdout, error.stderr)
        raise RuntimeError(f"frontier compilation timed out; see {stderr_path}") from error
    elapsed = time.perf_counter() - start
    save_logs(result.stdout, result.stderr)
    if result.returncode:
        raise RuntimeError(f"frontier compilation failed; see {stderr_path}")
    if _source_hashes() != hashes or _sha(snapshot) != hashes[str(CPP_SOURCE.resolve())]:
        raise RuntimeError("frontier sources changed during compilation")
    report = {
        "schema": SCHEMA, "status": "BUILT", "abi_version": 1,
        "library": str(library), "library_sha256": _sha(library),
        "source": str(CPP_SOURCE), "source_sha256": _sha(snapshot),
        "source_snapshot": str(snapshot), "source_hashes": hashes,
        "wrapper_sha256": _sha(__file__), "compiler": compiler,
        "compiler_sha256": _sha(compiler), "compiler_version": version.stdout,
        "command": command, "compile_seconds": elapsed,
        "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
        "stdout_sha256": _sha(stdout_path), "stderr_sha256": _sha(stderr_path),
        "platform": platform.platform(), "python": sys.version,
    }
    manifest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


class FrontierMemoryLimitError(RuntimeError):
    def __init__(self, limit, stats):
        self.limit, self.stats = limit, stats
        super().__init__(f"native frontier workspace limit {limit} bytes exceeded")


class FrontierStateLimitError(RuntimeError):
    def __init__(self, limit, stats):
        self.limit, self.stats = limit, stats
        self.attempted_states = stats["attempted_states"]
        self.processed_rows = stats["processed_rows"]
        self.row_index = stats["failure_row"]
        super().__init__(f"native frontier state limit {limit} exceeded")


class FrontierWidthLimitError(RuntimeError):
    def __init__(self, width, row_index):
        self.limit, self.width, self.row_index = 64, width, row_index
        super().__init__(f"row {row_index} needs {width} frontier slots; limit is 64")


def _positive_u64(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a positive integer") from error
    if not 1 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be in 1..2^64-1")
    return value


def _plan(rows, width, row_order):
    """Assign slots until the last incident row finishes, then recycle them."""
    supports = tuple(tuple((j, value) for j, value in enumerate(row) if value)
                     for row in rows)
    remaining = [0] * width
    for row in supports:
        for column, _ in row:
            remaining[column] += 1
    if row_order is None:
        masks = tuple(sum(1 << j for j, _ in row) for row in supports)
        order = _greedy_order(supports, masks, remaining)
        ordering = "greedy"
    else:
        supplied = tuple(row_order)
        if any(isinstance(index, bool) for index in supplied):
            raise TypeError("boolean row indices are not allowed")
        order = tuple(operator.index(index) for index in supplied)
        if len(order) != len(rows) or set(order) != set(range(len(rows))):
            raise ValueError("row_order must be a permutation of all row indices")
        ordering = "explicit"
    slots, offsets, bits, weights, live_masks = {}, [0], [], [], []
    free = UINT64_MAX
    peak_frontier = peak_row_frontier = 0
    for index in order:
        support = supports[index]
        during = len(slots) + sum(column not in slots for column, _ in support)
        if during > 64:
            raise FrontierWidthLimitError(during, index)
        peak_row_frontier = max(peak_row_frontier, during)
        for column, value in support:
            if column not in slots:
                bit = free & -free
                free ^= bit
                slots[column] = bit
            bits.append(slots[column])
            weights.append(value)
        offsets.append(len(bits))
        # Do not reuse an expired column's bit within its closing row.
        for column, _ in support:
            remaining[column] -= 1
            if not remaining[column]:
                free |= slots.pop(column)
        live_masks.append(UINT64_MAX ^ free)
        peak_frontier = max(peak_frontier, len(slots))
    if slots:
        raise RuntimeError("frontier planning did not close")
    return {"row_order": tuple(order), "ordering": ordering, "offsets": offsets,
            "bits": bits, "weights": weights, "live_masks": live_masks,
            "peak_frontier": peak_frontier, "peak_row_frontier": peak_row_frontier}


def _bounds(rows, width):
    """Each endpoint-relaxed polynomial bounds every matching coefficient."""
    degree = min(len(rows), width)
    def elementary(values):
        result = [1] + [0] * degree
        for count, value in enumerate(values):
            for k in range(min(count + 1, degree), 0, -1):
                result[k] += value * result[k - 1]
        return result
    row_bound = elementary(sum(row) for row in rows)
    column_bound = elementary(sum(row[j] for row in rows) for j in range(width))
    return tuple(min(a, b) for a, b in zip(row_bound, column_bound))


def _crt_primes(bound):
    # Keep exhaustion explicit rather than searching forever below 3. This
    # local guard leaves the retained benchmark helper and its hashes intact.
    primes, product, candidate = [], 1, 2147483647
    while product <= bound or not primes:
        if candidate < 3:
            raise RuntimeError("31-bit prime range cannot cover the frontier coefficient bound")
        if is_prime(candidate):
            primes.append(candidate)
            product *= candidate
        candidate -= 2
    return primes


class FrontierBaseline:
    def __init__(self, library):
        self.library = Path(library).resolve(strict=True)
        self.library_sha256 = _sha(self.library)
        self.source_hashes = _source_hashes()
        manifest = self.library.parent / "build.json"
        self.build = json.loads(manifest.read_text())
        if (self.build.get("schema") != SCHEMA or self.build.get("status") != "BUILT"
                or self.build.get("abi_version") != 1
                or self.build.get("library_sha256") != self.library_sha256
                or self.build.get("source_hashes") != self.source_hashes
                or self.build.get("source_sha256") != self.source_hashes[str(CPP_SOURCE.resolve())]
                or self.build.get("wrapper_sha256") != self.source_hashes[str(Path(__file__).resolve())]):
            raise ValueError("frontier build/source identity mismatch; use a fresh build")
        try:
            for path_key, hash_key in (("source_snapshot", "source_sha256"),
                                       ("stdout_log", "stdout_sha256"),
                                       ("stderr_log", "stderr_sha256")):
                if _sha(self.build[path_key]) != self.build[hash_key]:
                    raise ValueError(f"frontier build artifact identity mismatch: {path_key}")
        except (KeyError, TypeError, OSError) as error:
            raise ValueError("missing or invalid frontier build provenance") from error
        self._library = ctypes.CDLL(str(self.library))
        version = self._library.weighted_matching_frontier_abi_version
        version.argtypes, version.restype = [], ctypes.c_uint32
        if version() != 1:
            raise ValueError("unsupported frontier ABI version")
        u64, u32 = ctypes.c_uint64, ctypes.c_uint32
        self._function = self._library.weighted_matching_frontier_mod
        self._function.argtypes = [u64, u64, u64, ctypes.POINTER(u64), ctypes.POINTER(u64),
                                  ctypes.POINTER(u32), ctypes.POINTER(u64), u32, u64, u64,
                                  ctypes.POINTER(u32), ctypes.POINTER(u64)]
        self._function.restype = ctypes.c_int

    def evaluate(self, matrix, *, row_order=None, max_states=1_000_000,
                 max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
        """Return exact coefficients, CRT evidence and native workspace peaks.

        Total timing includes normalization, ordering, bounds, input buffers,
        all modular calls (including native allocation) and reconstruction.
        It excludes the explicit build and loader. No fallback or subprocess
        runs here. A resource failure raises and supplies no exact result.
        """
        start = time.perf_counter()
        max_states = _positive_u64(max_states, "max_states")
        max_workspace_bytes = _positive_u64(max_workspace_bytes, "max_workspace_bytes")
        rows, width = _snapshot(matrix)
        plan = _plan(rows, width, row_order)
        bounds = _bounds(rows, width)
        primes = _crt_primes(max(bounds))
        u64, u32 = ctypes.c_uint64, ctypes.c_uint32
        offsets = (u64 * len(plan["offsets"]))(*plan["offsets"])
        bits = (u64 * len(plan["bits"]))(*plan["bits"])
        live = (u64 * len(rows))(*plan["live_masks"])
        out = (u32 * len(bounds))()
        stats_buffer = (u64 * 5)()
        setup_seconds = time.perf_counter() - start
        residues, native_stats = [], []
        reduction_seconds = call_seconds = 0.0
        for prime in primes:
            stage = time.perf_counter()
            weights = (u32 * len(plan["weights"]))(*(w % prime for w in plan["weights"]))
            reduction_seconds += time.perf_counter() - stage
            stage = time.perf_counter()
            code = self._function(len(rows), len(bounds) - 1, len(bits), offsets,
                                  bits, weights, live, prime, max_states,
                                  max_workspace_bytes, out, stats_buffer)
            call_seconds += time.perf_counter() - stage
            failure = int(stats_buffer[3])
            stats = {"peak_workspace_bytes": int(stats_buffer[0]),
                     "peak_states": int(stats_buffer[1]),
                     "processed_rows": int(stats_buffer[2]),
                     "failure_step": None if failure == UINT64_MAX else failure,
                     "failure_row": (plan["row_order"][failure]
                                     if failure < len(rows) else None),
                     "attempted_states": int(stats_buffer[4])}
            if code == 2:
                raise FrontierMemoryLimitError(max_workspace_bytes, stats)
            if code == 3:
                raise FrontierStateLimitError(max_states, stats)
            if code:
                raise RuntimeError(f"compiled frontier failed with status {code}")
            if (stats["processed_rows"] != len(rows) or stats["failure_step"] is not None
                    or not 1 <= stats["peak_states"] <= max_states
                    or stats["attempted_states"] != stats["peak_states"]
                    or stats["peak_workspace_bytes"] > max_workspace_bytes):
                raise RuntimeError("invalid compiled frontier resource transcript")
            residues.append(tuple(out))
            native_stats.append(stats)
        stage = time.perf_counter()
        coefficients = tuple(crt_vectors(residues, primes, bounds, signed=False))
        if coefficients[0] != 1:
            raise RuntimeError("frontier result lost the empty matching")
        crt_seconds = time.perf_counter() - stage
        return {
            "schema": SCHEMA, "status": "COMPLETE", "exact": True,
            "algorithm": "cpp-row-frontier-cardinality-dp-crt",
            "shape": (len(rows), width), "coefficients": coefficients,
            "row_order": plan["row_order"], "ordering": plan["ordering"],
            "peak_frontier": plan["peak_frontier"],
            "peak_row_frontier": plan["peak_row_frontier"],
            "peak_states": max(s["peak_states"] for s in native_stats),
            "peak_workspace_bytes": max(s["peak_workspace_bytes"] for s in native_stats),
            "max_states": max_states, "max_workspace_bytes": max_workspace_bytes,
            "workspace_scope": WORKSPACE_SCOPE, "native_stats": tuple(native_stats),
            "bounds": bounds, "primes": tuple(primes), "residue_rows": tuple(residues),
            "modulus": math.prod(primes), "library_sha256": self.library_sha256,
            "source_hashes": dict(self.source_hashes),
            "setup_seconds": setup_seconds, "reduction_seconds": reduction_seconds,
            "call_seconds": call_seconds, "residue_seconds": reduction_seconds + call_seconds,
            "crt_seconds": crt_seconds, "total_seconds": time.perf_counter() - start,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="compile into a new output directory")
    build.add_argument("output", type=Path)
    run = commands.add_parser("evaluate", help="evaluate a rectangular integer JSON matrix")
    run.add_argument("library", type=Path)
    run.add_argument("matrix", type=Path)
    run.add_argument("--max-states", type=int, default=1_000_000)
    run.add_argument("--max-workspace-bytes", type=int, default=DEFAULT_WORKSPACE_BYTES)
    run.add_argument("--row-order", help="comma-separated row indices; default: greedy")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_frontier_baseline(args.output)
        else:
            order = (None if args.row_order is None else [] if not args.row_order
                     else [int(x) for x in args.row_order.split(",")])
            result = FrontierBaseline(args.library).evaluate(
                json.loads(args.matrix.read_text()), row_order=order,
                max_states=args.max_states, max_workspace_bytes=args.max_workspace_bytes)
    except (FrontierMemoryLimitError, FrontierStateLimitError, FrontierWidthLimitError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "RESOURCE_LIMIT", "exact": False,
                          "error": str(error), "limit": error.limit,
                          "details": getattr(error, "stats", {"width": getattr(error, "width", None),
                                                                "row_index": getattr(error, "row_index", None)})}))
        return 2
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
