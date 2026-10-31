"""Reproducible scaling benchmark for weighted rook polynomials.

The benchmark uses dense SHA-256-derived matrices with weights ``1, 2, 4,
5``, matching the power profile in the phase-faded MIMO evaluation.  It times
the specialized :func:`perm_polynomial.weighted_rook_polynomial` API and, at
deliberately small orders only, the generic identity

``[x**k] per(J + x*W) = (n-k)! * r_k(W)``.

The safe default is intentionally tiny and normally completes in seconds::

    python benchmark_weighted_rook.py

Larger exponential runs require an explicit ``--allow-large`` flag.  Output
is one JSON object on stdout so runs can be archived without scraping tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from math import factorial

import perm_polynomial


DEFAULT_LABEL = "perm-weighted-rook-scaling-v1/0"
DEFAULT_ORDERS = (10, 12, 14)
DEFAULT_GENERIC_MAX_ORDER = 14
SAFE_MAX_ORDER = 24
SAFE_GENERIC_MAX_ORDER = 18
SAFE_FALLBACK_MAX_ORDER = 18
WEIGHTS = (1, 2, 4, 5)

# These hashes were generated independently with a subset dynamic program.
# They pin both the SHA generator and the complete exact answer for the quick
# default cases.  Custom labels and other orders still receive repeat and
# generic-baseline checks, but naturally have no built-in golden hash.
PINNED_CASES = {
    8: (
        "6bc46e5c431f9f772c2d1ae4075b556b2275a8f80a84367f6cf21df698990a7e",
        "511511ce95df5f338e56e76fa5b0b3bf320725988da695635de33a6db123b2c7",
    ),
    10: (
        "552c80bc58762419a2be7ce10205d0901c5fe1c729756aae47a653c574da68e8",
        "ed513af918ab3ccdcb6375ac65e7c11ec46c9ab835913daac5dc6457c7018f85",
    ),
    12: (
        "114b24de97c7e2942d438dfe14aba6b015b48cc2fe53791bfe56ab138d37674f",
        "8638ffe6910dce00777b6066555dae0635392244171292d4defa5c136dbf626c",
    ),
    14: (
        "e93c4ffb4c777b5d011c81d0713c6ce8931d4e616d5cb9fdc13abe3a6409f74e",
        "7bcebcd04fa506a97177baa6d1c3c8a425910d797417ebbcc982ffcf7aec25d8",
    ),
    16: (
        "8821177de299b277a8f89d8dc8f7994ef9b2a3ffc0909d9b1d316af51395dd91",
        "19749358e8d6b70c11c7b2143b1e493262a70879041eb0121144bb76940568da",
    ),
}


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _generate_matrix(order: int, label: str):
    matrix = []
    codes = bytearray()
    profile = bytearray()
    histogram = [0] * len(WEIGHTS)
    for row in range(order):
        values = []
        for column in range(order):
            text = f"{label}/{order}/e/{row}/{column}".encode("ascii")
            code = hashlib.sha256(text).digest()[0] >> 6
            weight = WEIGHTS[code]
            codes.append(code)
            profile.append(weight)
            histogram[code] += 1
            values.append(weight)
        matrix.append(values)
    return matrix, {
        "amplitude_code_counts_q0_through_q3": histogram,
        "amplitude_code_sha256": _sha256_hex(bytes(codes)),
        "power_sum": sum(profile),
        "power_profile_sha256": _sha256_hex(bytes(profile)),
    }


def _coefficient_hash(coefficients) -> str:
    csv = ",".join(str(value) for value in coefficients)
    return _sha256_hex(csv.encode("ascii"))


def _has_native_weighted_hook() -> bool:
    capabilities = perm_polynomial.polynomial_backend_capabilities()
    return "weighted_rook_polynomial_mod_many" in capabilities["native_hooks"]


def _low_coefficients(matrix):
    order = len(matrix)
    total = sum(sum(row) for row in matrix)
    square_sum = sum(value * value for row in matrix for value in row)
    row_square_sum = sum(sum(row) ** 2 for row in matrix)
    column_square_sum = sum(
        sum(matrix[row][column] for row in range(order)) ** 2
        for column in range(order)
    )
    twice_c2 = (
        total * total - row_square_sum - column_square_sum + square_sum
    )
    if twice_c2 % 2:
        raise RuntimeError("internal c2 formula did not produce an integer")
    return (1, total, twice_c2 // 2)


def _generic_weighted_rook(matrix, backend, parallel, num_threads):
    order = len(matrix)
    all_ones = [[1] * order for _ in range(order)]
    affine = perm_polynomial.affine_permanent(
        all_ones,
        matrix,
        backend=backend,
        parallel=parallel,
        num_threads=num_threads,
    )
    coefficients = []
    for degree, coefficient in enumerate(affine):
        divisor = factorial(order - degree)
        quotient, remainder = divmod(coefficient, divisor)
        if remainder:
            raise RuntimeError(
                "generic affine coefficient was not factorial-divisible"
            )
        coefficients.append(quotient)
    return tuple(coefficients)


def _measure(function, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        current = tuple(function())
        samples.append(time.perf_counter() - started)
        if result is not None and current != result:
            raise RuntimeError("repeated benchmark calls returned different results")
        result = current
    return result, {
        "seconds": samples,
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
    }


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders", nargs="+", type=int, default=list(DEFAULT_ORDERS)
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="native worker count; zero lets the backend choose",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--backend", choices=("auto", "native", "python"), default="auto"
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="request the public serial path (equivalent to one worker)",
    )
    parser.add_argument(
        "--generic-max-order",
        type=int,
        default=DEFAULT_GENERIC_MAX_ORDER,
        help="skip the generic affine baseline above this order; use 0 to skip it",
    )
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help=(
            "acknowledge exponential runs above the conservative safety "
            "limits"
        ),
    )
    args = parser.parse_args()

    if args.threads < 0:
        parser.error("threads must be nonnegative")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.generic_max_order < 0:
        parser.error("generic-max-order must be nonnegative")
    if not args.orders:
        parser.error("at least one order is required")
    if len(set(args.orders)) != len(args.orders):
        parser.error("orders must not contain duplicates")
    for order in args.orders:
        if not 2 <= order < 64:
            parser.error("every order must satisfy 2 <= n < 64")
    if not args.allow_large:
        if max(args.orders) > SAFE_MAX_ORDER:
            parser.error(
                f"orders above {SAFE_MAX_ORDER} require --allow-large"
            )
        fused_selected = (
            args.backend in ("auto", "native")
            and _has_native_weighted_hook()
        )
        if not fused_selected and max(args.orders) > SAFE_FALLBACK_MAX_ORDER:
            parser.error(
                "orders above "
                f"{SAFE_FALLBACK_MAX_ORDER} without the fused weighted "
                "native hook require --allow-large"
            )
        if args.generic_max_order > SAFE_GENERIC_MAX_ORDER:
            parser.error(
                "generic baselines above "
                f"{SAFE_GENERIC_MAX_ORDER} require --allow-large"
            )
    try:
        args.label.encode("ascii")
    except UnicodeEncodeError:
        parser.error("label must contain ASCII characters only")
    if not args.label:
        parser.error("label must not be empty")
    return args


def main():
    args = _parse_arguments()
    backend_capabilities = perm_polynomial.polynomial_backend_capabilities()
    weighted_api = getattr(
        perm_polynomial, "weighted_rook_polynomial", None
    )
    if weighted_api is None:
        raise RuntimeError(
            "this checkout does not provide weighted_rook_polynomial"
        )

    parallel = not args.serial
    cases = []
    for order in args.orders:
        matrix, instance = _generate_matrix(order, args.label)
        expected_low = _low_coefficients(matrix)

        weighted, weighted_timing = _measure(
            lambda: weighted_api(
                matrix,
                backend=args.backend,
                parallel=parallel,
                num_threads=args.threads,
            ),
            args.repeats,
        )
        if len(weighted) != order + 1:
            raise RuntimeError("weighted API returned the wrong coefficient count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in weighted
        ):
            raise RuntimeError("weighted API returned invalid coefficients")
        if weighted[:3] != expected_low:
            raise RuntimeError("weighted API failed the independent c0/c1/c2 check")

        coefficient_hash = _coefficient_hash(weighted)
        pinned_checked = None
        if args.label == DEFAULT_LABEL and order in PINNED_CASES:
            expected_profile_hash, expected_coefficient_hash = PINNED_CASES[order]
            if instance["power_profile_sha256"] != expected_profile_hash:
                raise RuntimeError("SHA-generated matrix changed from its pinned case")
            if coefficient_hash != expected_coefficient_hash:
                raise RuntimeError("weighted result failed its pinned hash check")
            pinned_checked = True

        generic_timing = None
        generic_checked = False
        if args.generic_max_order and order <= args.generic_max_order:
            generic, generic_timing = _measure(
                lambda: _generic_weighted_rook(
                    matrix, args.backend, parallel, args.threads
                ),
                args.repeats,
            )
            if generic != weighted:
                raise RuntimeError(
                    "weighted and generic affine computations disagree"
                )
            generic_checked = True

        speedup = None
        if generic_timing is not None:
            speedup = (
                generic_timing["median_seconds"]
                / weighted_timing["median_seconds"]
            )
        cases.append({
            "order": order,
            "instance": instance,
            "coefficient_csv_sha256": coefficient_hash,
            "coefficient_summary": {
                "count": len(weighted),
                "c0": str(weighted[0]),
                "c1": str(weighted[1]),
                "c2": str(weighted[2]),
                "leading": str(weighted[-1]),
                "maximum_bit_length": max(value.bit_length() for value in weighted),
            },
            "validation": {
                "independent_c0_c1_c2": True,
                "repeat_consistency": (
                    True if args.repeats > 1 else None
                ),
                "pinned_profile_and_coefficient_hash": pinned_checked,
                "generic_affine_exact_match": (
                    True if generic_checked else None
                ),
            },
            "weighted_rook_polynomial": weighted_timing,
            "generic_affine": generic_timing,
            "generic_over_weighted_speedup": speedup,
        })

    print(json.dumps({
        "schema_version": 1,
        "benchmark": "weighted-rook-scaling",
        "generator": {
            "label": args.label,
            "entry_string": "{label}/{order}/e/{row}/{column}",
            "code_rule": "SHA256(entry_string)[0] >> 6",
            "code_weights_q0_through_q3": list(WEIGHTS),
        },
        "configuration": {
            "orders": args.orders,
            "backend": args.backend,
            "parallel": parallel,
            "requested_threads": args.threads,
            "repeats": args.repeats,
            "generic_max_order": args.generic_max_order,
            "allow_large": args.allow_large,
            "native_extension_available": backend_capabilities[
                "native_extension_available"
            ],
            "native_weighted_hook_available": _has_native_weighted_hook(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "omp_places": os.environ.get("OMP_PLACES"),
        },
        "cases": cases,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
