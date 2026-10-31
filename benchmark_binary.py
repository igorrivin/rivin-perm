"""Reproducible CPU benchmark for the specialized zero-one backend.

Example on a 64-core GH200 host::

    OMP_PROC_BIND=close OMP_PLACES=cores \
        python benchmark_binary.py --orders 26 28 30 --repeats 3

The benchmark compares one identical 31-bit residue.  Exact calls may need
multiple residues; :mod:`perm_binary` determines that count from its rigorous
nonnegative bound.
"""

import argparse
import json
import random
import statistics
import time

import perm
import perm_binary


MODULUS = (1 << 31) - 1


def _regular_matrix(order: int, degree: int, seed: int):
    offsets = random.Random(seed).sample(range(order), degree)
    return [
        [int((column - row) % order in offsets) for column in range(order)]
        for row in range(order)
    ]


def _dense_matrix(order: int, probability: float, seed: int):
    rng = random.Random(seed)
    matrix = [
        [int(rng.random() < probability) for _ in range(order)]
        for _ in range(order)
    ]
    # Plant one matching so a zero result cannot make a broken benchmark look
    # plausible and so matching preprocessing does not short-circuit the case.
    for index in range(order):
        matrix[index][index] = 1
    return matrix


def _measure(function, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        current = function()
        samples.append(time.perf_counter() - started)
        if result is not None and current != result:
            raise RuntimeError("benchmark backend returned inconsistent residues")
        result = current
    return {
        "residue": result,
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int, default=[26, 28, 30])
    parser.add_argument("--degrees", nargs="+", type=int, default=[3, 6])
    parser.add_argument("--dense-probability", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    if not perm_binary.HAS_BINARY_EXT:
        raise RuntimeError("build perm_binary_ext before running this benchmark")
    if not perm.HAS_C_EXT:
        raise RuntimeError("build perm_c_ext before running this benchmark")
    if args.threads < 0 or args.repeats < 1:
        parser.error("threads must be nonnegative and repeats must be positive")
    if not 0.0 <= args.dense_probability <= 1.0:
        parser.error("dense probability must lie between zero and one")

    threads = args.threads or perm_binary._binary_ext.get_max_threads()
    cases = []
    for order in args.orders:
        if not 2 <= order < 64:
            parser.error("every order must satisfy 2 <= n < 64")
        for degree in args.degrees:
            if not 1 <= degree <= order:
                parser.error("every degree must satisfy 1 <= d <= n")
            cases.append((
                f"n{order}_d{degree}",
                _regular_matrix(order, degree, args.seed + order + degree),
            ))
        cases.append((
            f"n{order}_dense{args.dense_probability:g}",
            _dense_matrix(
                order, args.dense_probability, args.seed + 10 * order
            ),
        ))

    for label, matrix in cases:
        order = len(matrix)
        generic = _measure(
            lambda: perm.perm_c_ext.ryser_permanent_mod_p(
                matrix, MODULUS, threads
            ),
            args.repeats,
        )
        binary = _measure(
            lambda: perm_binary._binary_ext.binary_permanent_mod_p(
                matrix, MODULUS, threads
            ),
            args.repeats,
        )
        if generic["residue"] != binary["residue"]:
            raise RuntimeError("generic and binary residues disagree")
        print(json.dumps({
            "case": label,
            "order": order,
            "nonzeros": sum(map(sum, matrix)),
            "modulus": MODULUS,
            "threads": threads,
            "generic": generic,
            "binary": binary,
            "speedup": (
                generic["median_seconds"] / binary["median_seconds"]
            ),
        }))


if __name__ == "__main__":
    main()
