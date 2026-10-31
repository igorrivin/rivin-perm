"""Compare native Ryser with general dispatch across unrelated matrix families.

Quick default: python benchmark_native_dispatch.py --output dispatch.json
Broader run (exponential): python benchmark_native_dispatch.py --allow-large \
    --orders 2 4 8 12 16 18 --threads 1 4 --output dispatch-broad.json

Modular calls use the requested threads. Exact calls use serial CRT to isolate
the dispatch change without process-pool startup or inherited monkeypatches.
All comparisons alternate execution order, assert repeat/result agreement,
and retain raw paired timings. Tiny calls are batched to reduce timer noise.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time


ROOT = Path(__file__).resolve().parent
INT64_MAX = (1 << 63) - 1
MODULI = {
    "odd31": (1 << 31) - 1,
    "even31": (1 << 31) - 2,
    "composite31": 3 * ((1 << 29) - 1),
    "odd63": (1 << 63) - 25,
    "even63": (1 << 63) - 2,
    "composite63": INT64_MAX,
}
FAMILIES = (
    "dense-positive", "dense-signed", "binary-sparse", "binary-dense",
    "diagonal", "permutation", "triangular", "block-diagonal",
    "repeated-rows", "zero-row", "zero-column", "large-residues", "boundary",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def matrix_for(family, n, seed, modulus):
    rng = random.Random(f"native-dispatch-v1/{seed}/{family}/{n}")
    matrix = [[rng.randint(1, 9) for _ in range(n)] for _ in range(n)]
    if family == "dense-signed":
        matrix = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(n)]
    elif family.startswith("binary-"):
        probability = 0.15 if family == "binary-sparse" else 0.8
        matrix = [[int(rng.random() < probability) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1  # Keep a matching in both binary densities.
    elif family == "permutation":
        columns = list(range(n))
        rng.shuffle(columns)
        matrix = [[int(j == columns[i]) for j in range(n)] for i in range(n)]
    elif family in ("diagonal", "triangular", "block-diagonal"):
        for i in range(n):
            for j in range(n):
                if ((family == "diagonal" and i != j)
                        or (family == "triangular" and i > j)
                        or (family == "block-diagonal" and (i < n // 2) != (j < n // 2))):
                    matrix[i][j] = 0
    elif family == "repeated-rows":
        matrix = [matrix[0][:] for _ in range(n)]
    elif family == "zero-row":
        matrix[n // 2] = [0] * n
    elif family == "zero-column":
        for row in matrix:
            row[n // 2] = 0
    elif family == "large-residues":
        matrix = [[rng.randrange(max(1, modulus - 1024), modulus)
                   for _ in range(n)] for _ in range(n)]
    elif family == "boundary":
        values = [-(1 << 63), INT64_MAX, -modulus, modulus, -1, 0, 1]
        matrix = [[rng.choice(values) for _ in range(n)] for _ in range(n)]
    return matrix


def subset_oracle(matrix):
    """Independent exact subset DP, used only at orders at most eight."""
    n = len(matrix)
    values = [0] * (1 << n)
    values[0] = 1
    for mask in range(1, 1 << n):
        row = bin(mask).count("1") - 1
        values[mask] = sum(matrix[row][j] * values[mask ^ (1 << j)]
                           for j in range(n) if mask & (1 << j))
    return values[-1]


def measure_pair(functions, args, contexts=None):
    expected = None

    def timed(name, batch):
        nonlocal expected
        context = contexts[name]() if contexts and name in contexts else nullcontext()
        with context:
            started = time.perf_counter_ns()
            for _ in range(batch):
                result = functions[name]()
                if expected is None:
                    expected = result
                elif result != expected:
                    raise AssertionError(f"{name} returned {result}, expected {expected}")
            return (time.perf_counter_ns() - started) / 1e9

    warmups = []
    for index in range(args.warmups):
        order = ["ryser", "dispatch"][::1 if index % 2 == 0 else -1]
        warmups.append({name: timed(name, 1) for name in order})
    fastest = min(warmups[-1].values())
    batch = min(10000, max(1, math.ceil(args.min_sample_seconds / fastest)))
    calibration_batch = batch
    calibration = {name: timed(name, batch) for name in functions}
    if min(calibration.values()) < args.min_sample_seconds:
        batch = min(10000, math.ceil(batch * args.min_sample_seconds / min(calibration.values())))
    pairs = []
    for index in range(args.repeats):
        order = ["ryser", "dispatch"][::1 if index % 2 == 0 else -1]
        pairs.append({"order": order, "seconds": {name: timed(name, batch) for name in order}})
    medians = {name: statistics.median(pair["seconds"][name] / batch for pair in pairs)
               for name in functions}
    return {"result": str(expected), "batch_size": batch,
            "warmup_seconds": warmups, "calibration_batch_size": calibration_batch,
            "calibration_seconds": calibration, "pairs": pairs,
            "median_seconds_per_call": medians,
            "speedup": medians["ryser"] / medians["dispatch"]}


@contextmanager
def old_dispatch(native):
    """Restore the original native hook even when an exact call fails."""
    current = native.permanent_mod_p
    native.permanent_mod_p = native.ryser_permanent_mod_p
    try:
        yield
    finally:
        native.permanent_mod_p = current


def command_output(argv):
    try:
        return subprocess.check_output(argv, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def provenance(native):
    binary = Path(native.__file__).resolve()
    compiler = shlex.split(os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc")
    executable = shutil.which(compiler[0])
    cpu = command_output(["lscpu"])
    return {
        "command": shlex.join([sys.executable, *sys.argv]),
        "imported_perm": str(Path(sys.modules["perm"].__file__).resolve()),
        "python": sys.version, "platform": platform.platform(),
        "cpu": cpu, "cpu_sha256": digest(cpu.encode()), "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "compiler_command": compiler, "compiler_version": command_output([*compiler, "--version"]),
        "compiler_sha256": digest(Path(executable).read_bytes()) if executable else None,
        "binary_compiler_comment": command_output(["readelf", "-p", ".comment", str(binary)]),
        "build_config": {key: sysconfig.get_config_var(key) for key in ("CC", "CFLAGS", "LDSHARED")},
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("OMP_", "GOMP_", "KMP_")) or key in
                        ("CC", "CFLAGS", "LDFLAGS", "PERM_BUILD_OPENMP", "PYTHONHASHSEED")},
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "source_sha256": {name: digest((ROOT / name).read_bytes()) for name in
                          ("perm.py", "perm_c_omp.c", "setup.py", Path(__file__).name)},
        "binary": str(binary), "binary_sha256": digest(binary.read_bytes()),
        "max_native_threads": native.get_max_threads() if hasattr(native, "get_max_threads") else 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 3, 4, 5, 8, 12])
    parser.add_argument("--threads", nargs="+", type=int, default=[1], help="modular thread counts; exact CRT uses one")
    parser.add_argument("--moduli", nargs="+", default=list(MODULI), help="profile names or positive signed 64-bit integers")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--mode", choices=("modular", "exact", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--min-sample-seconds", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    if any(n < 2 or n >= 64 for n in args.orders) or any(t < 1 for t in args.threads):
        parser.error("orders must be between 2 and 63; threads must be positive")
    if args.repeats < 1 or args.warmups < 1 or not math.isfinite(args.min_sample_seconds) or args.min_sample_seconds <= 0:
        parser.error("repeats, warmups, and finite minimum sample time must be positive")
    try:
        moduli = [(label, MODULI[label] if label in MODULI else int(label, 0)) for label in args.moduli]
    except ValueError:
        parser.error(f"moduli must be integers or one of {', '.join(MODULI)}")
    if any(p < 2 or p > INT64_MAX for _, p in moduli):
        parser.error("moduli must be between 2 and 2**63 - 1")
    work = sum(n * (1 << n) for n in args.orders) * len(args.families) * 2 * (args.repeats + args.warmups)
    work *= ((len(moduli) * len(args.threads) if args.mode != "exact" else 0)
             + (max(args.orders) * 3 if args.mode != "modular" else 0))
    if not args.allow_large and (max(args.orders) > 20 or work > 2_000_000_000):
        parser.error("this exponential workload requires --allow-large; reduce orders/cases/repeats for a quick run")

    import perm
    if not perm.HAS_C_EXT or not hasattr(perm.perm_c_ext, "permanent_mod_p"):
        parser.error("rebuild perm_c_ext with the general permanent_mod_p hook first")
    native = perm.perm_c_ext
    report = {"schema": "native-dispatch-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "arguments": {**vars(args), "output": str(args.output) if args.output else None},
              "provenance": provenance(native), "cases": []}
    for n in args.orders:
        for family in args.families:
            jobs = []
            if args.mode != "exact":
                jobs.extend(("modular", label, p, threads) for label, p in moduli for threads in args.threads)
            if args.mode != "modular":
                jobs.append(("exact", "exact", INT64_MAX, 1))
            for mode, label, p, threads in jobs:
                matrix = matrix_for(family, n, args.seed, p)

                exact_call = lambda: perm.matrix_permanent(matrix, parallel=False, num_threads=1)
                functions = ({"ryser": lambda: native.ryser_permanent_mod_p(matrix, p, threads),
                              "dispatch": lambda: native.permanent_mod_p(matrix, p, threads)}
                             if mode == "modular" else
                             {"ryser": exact_call, "dispatch": exact_call})
                contexts = {"ryser": lambda: old_dispatch(native)} if mode == "exact" else None
                result = measure_pair(functions, args, contexts)
                if n <= 8:
                    expected = subset_oracle(matrix)
                    if int(result["result"]) != (expected % p if mode == "modular" else expected):
                        raise AssertionError(f"independent subset oracle failed: {family}, n={n}, {label}")
                report["cases"].append({"mode": mode, "family": family, "order": n,
                                        "modulus_profile": label, "modulus": p if mode == "modular" else None,
                                        "threads": threads, "matrix_sha256": digest(json.dumps(matrix, separators=(",", ":")).encode()),
                                        "oracle_checked": n <= 8, **result})
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
