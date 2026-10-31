"""Exact whole-call low-rank dispatch comparisons on public synthetic inputs.

Quick default (native optional): python benchmark_low_rank.py --output report.json
Development grid: --phase development --allow-large --threads 1 4 --repeats 3
Fresh holdout: --phase holdout --allow-large --threads 1 4 0 --repeats 3

All outputs, method choices, raw samples and losses are retained. --threads 1
uses serial CRT. Larger/automatic thread settings allow the public generic
parallel policy, including process-pool startup. Explicit low-rank calls are
timed on eligible constructed families as a crossover counterfactual.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import perm


ROOT = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_for(family, order, seed, rank=3):
    rng = random.Random(f"low-rank-dispatch-v1/{family}/{order}/{seed}/{rank}")
    if family in ("rank", "positive_rank", "perturbed_rank"):
        entries = (1, 2, 3, 4) if family == "positive_rank" else (-3, -2, -1, 1, 2, 3)
        u = [[rng.choice(entries) for _ in range(rank)] for _ in range(order)]
        v = [[rng.choice(entries) for _ in range(order)] for _ in range(rank)]
        matrix = [[sum(u[i][k] * v[k][j] for k in range(rank))
                   for j in range(order)] for i in range(order)]
        if family == "perturbed_rank":
            for i in range(order):
                matrix[i][i] += 1
        return matrix
    if family == "identity":
        return [[int(i == j) for j in range(order)] for i in range(order)]
    if family == "triangular":
        return [[rng.randint(1, 4) if i <= j else 0 for j in range(order)]
                for i in range(order)]
    if family == "binary":
        return [[int(i == j or rng.random() < .7) for j in range(order)]
                for i in range(order)]
    if family == "dense":
        return [[rng.randint(-9, 9) for _ in range(order)] for _ in range(order)]
    raise ValueError(f"unknown family: {family}")


def case_specs(phase):
    if phase == "quick":
        return [("rank", 16, r, 101) for r in (2, 3, 4)] + [("dense", 16, 0, 101)]
    if phase == "development":
        return [("rank", n, r, 2026103001) for n in (18, 20, 22) for r in (2, 3, 4)]
    # This corpus is disjoint from the development grid and earlier prototype
    # inputs. Do not retune the candidate after examining its results.
    return ([("rank", 20, 2, 8123081), ("rank", 22, 3, 8123082),
             ("positive_rank", 22, 4, 8123083), ("rank", 24, 3, 8123084),
             ("rank", 24, 4, 8123085), ("positive_rank", 20, 1, 8123086)]
            + [(family, 20, 3, 9123000 + i) for i, family in enumerate(
                ("dense", "binary", "identity", "triangular", "perturbed_rank"))]
            + [("positive_rank", 20, 5, 9123009)])


@contextmanager
def observe_choice():
    """Trace choice outside timed calls; always restore the production hook."""
    original = perm._low_rank_residues
    choices = []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        choices.append(result is not None)
        return result
    perm._low_rank_residues = observed
    try:
        yield choices
    finally:
        perm._low_rank_residues = original


def measure(matrix, methods, threads, repeats):
    parallel = threads != 1
    expected, samples, orders = None, {method: [] for method in methods}, []
    permutations = list(itertools.permutations(methods))
    for iteration in range(repeats + 1):
        order = permutations[iteration % len(permutations)]
        orders.append(list(order))
        for method in order:
            start = time.perf_counter_ns()
            value = perm.matrix_permanent(matrix, parallel=parallel, num_threads=threads, method=method)
            elapsed = (time.perf_counter_ns() - start) / 1e9
            if expected is None:
                expected = value
            if value != expected:
                raise AssertionError(f"exact disagreement for {method}")
            if iteration:
                samples[method].append(elapsed)
    medians = {method: statistics.median(values) for method, values in samples.items()}
    return {"value": str(expected), "method_orders": orders, "samples_seconds": samples,
            "median_seconds": medians, "speedup": medians["generic"] / medians["auto"],
            "slower_seconds": medians["auto"] - medians["generic"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("quick", "development", "holdout"), default="quick")
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or not args.threads or any(t < 0 for t in args.threads):
        parser.error("positive repeats and nonnegative thread settings are required")
    if args.phase != "quick" and not args.allow_large:
        parser.error("development and holdout exponential runs require --allow-large")
    if args.phase != "quick" and not perm.HAS_C_EXT:
        parser.error("larger comparisons require the existing C extension")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output already exists; use a fresh capture path")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "phase": args.phase, "python": sys.version,
              "platform": platform.platform(), "logical_cpus": os.cpu_count(),
              "thread_environment": {key: os.environ.get(key) for key in (
                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "backend": perm.get_backend(), "repeats": args.repeats, "warmups": 1,
              "source_sha256": {path.name: file_hash(path) for path in (ROOT / "perm.py", Path(__file__))},
              "policy": {"min_order": perm._LOW_RANK_MIN_ORDER, "max_rank": perm._LOW_RANK_MAX_RANK,
                         "max_states": perm._LOW_RANK_MAX_STATES, "work_ratio": perm._LOW_RANK_WORK_RATIO},
              "native_threads": perm.perm_c_ext.get_max_threads() if perm.HAS_C_EXT else None,
              "native_library_sha256": file_hash(perm.perm_c_ext.__file__) if perm.HAS_C_EXT else None,
              "cases": []}
    try:
        report["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        report["cpu"] = platform.processor() or "unavailable"
    try:
        report["compiler"] = subprocess.check_output(["clang", "--version"],
                                                     text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        report["compiler"] = "unavailable"
    for family, n, rank, seed in case_specs(args.phase):
        matrix = matrix_for(family, n, seed, rank)
        methods = ["generic", "auto"]
        # Strict rank-5 and perturbed/full-rank matrices are control fallbacks.
        if family in ("rank", "positive_rank") and rank <= 4:
            methods.append("low_rank")
        for threads in args.threads:
            with observe_choice() as choices:
                perm.matrix_permanent(matrix, parallel=threads != 1, num_threads=threads)
            record = {"family": family, "order": n, "requested_rank": rank, "seed": seed,
                      "matrix_sha256": hashlib.sha256(json.dumps(matrix).encode()).hexdigest(),
                      "threads": threads, "selected_low_rank": bool(choices and choices[-1]),
                      **measure(matrix, methods, threads, args.repeats)}
            report["cases"].append(record)
            # Atomic checkpoint retains every completed case if a later job fails.
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)
            print(json.dumps({key: record[key] for key in (
                "family", "order", "requested_rank", "threads", "selected_low_rank", "speedup")}), flush=True)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "COMPLETE"
    report["losses"] = [i for i, c in enumerate(report["cases"]) if c["speedup"] < 1]
    report["material_slowdowns"] = [i for i, c in enumerate(report["cases"])
        if c["slower_seconds"] > .0001 and c["speedup"] < 1 / 1.1]
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
