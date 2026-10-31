"""Reproducible low-rank permanent applications; see LOW_RANK_SHOWCASES.md.

The three exact cases run from the checkout or unpacked source distribution.
--include-approximation additionally needs the checkout-only research helper.
No exponential generic reference above order 22 is permitted by this tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
from math import comb, factorial, prod
from numbers import Integral
from pathlib import Path
import platform
import statistics
import sys
import time

import perm


ROOT = Path(__file__).resolve().parent
APPROXIMATION_SOURCE = ROOT / "notes/experiments/near_low_rank_permanent.py"


def _order(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or not 3 <= n <= 60:
        raise ValueError("showcase order must be an integer between 3 and 60")
    return int(n)


def polynomial_kernel(n):
    """Positive integer degree-two kernel: 1 + 2ij + i^2 j^2, rank exactly 3."""
    n = _order(n)
    return [[(1 + i * j) ** 2 for j in range(1, n + 1)] for i in range(1, n + 1)]


def gaussian_factors(n):
    """Three nonnegative integer loadings per sensor; every sensor differs."""
    n = _order(n)
    return [[1, i + 1, 1 + (i * i + 3 * i) % 7] for i in range(n)]


def gram_matrix(factors):
    """FF^T; real loadings, proper complex latent Gaussian variables."""
    return [[sum(x * y for x, y in zip(a, b)) for b in factors] for a in factors]


def profile_case(n):
    """Known n-category alphabet, unequal candidate probabilities, counts 0,1,2."""
    n = _order(n)
    weights = list(range(1, n + 1))
    counts = [j % 3 for j in range(n)]
    matrix = [[weight ** count for count in counts] for weight in weights]
    return matrix, weights, counts


def profile_probability(value, weights, counts):
    """Normalize the permanent into a probability of an unlabeled count profile.

    Categories are known, including zero-count categories. This sums multinomial
    probabilities over distinct assignments of the count multiset to labels.
    """
    if (len(weights) != len(counts) or not weights
            or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in weights)
            or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in counts)):
        raise ValueError("need positive integer category weights and matching nonnegative integer counts")
    sample_size = sum(counts)
    multiplicities = Counter(counts)
    denominator = (sum(weights) ** sample_size * prod(factorial(x) for x in counts)
                   * prod(factorial(m) for m in multiplicities.values()))
    return Fraction(value * factorial(sample_size), denominator)


def grouped_columns_permanent(matrix, *, max_states=200_000):
    """Independent exact reference exploiting identical columns, not matrix rank.

    State a records how many columns of each type have been used. The final
    coefficient is multiplied by the factorial of each type multiplicity.
    """
    if isinstance(max_states, bool) or not isinstance(max_states, Integral) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    rows = [list(row) for row in matrix]
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise ValueError("matrix must be square")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for row in rows for x in row):
        raise TypeError("matrix entries must be integers")
    columns = Counter(zip(*rows))
    types = list(columns)
    limits = tuple(columns[column] for column in types)
    if prod(m + 1 for m in limits) > max_states:
        raise ValueError("grouped-column coefficient budget exceeded")
    states = {(0,) * len(types): 1}
    for i in range(n):
        updated = {}
        for powers, value in states.items():
            for a, column in enumerate(types):
                if powers[a] < limits[a] and column[i]:
                    key = powers[:a] + (powers[a] + 1,) + powers[a + 1:]
                    updated[key] = updated.get(key, 0) + int(column[i]) * value
        states = updated
    return states.get(limits, 0) * prod(factorial(m) for m in limits)


def _scientific(value):
    value = Fraction(value)
    with localcontext() as context:
        context.prec = 12
        return format(Decimal(value.numerator) / Decimal(value.denominator), ".8E")


def _measure(methods, repeats):
    """One warmup; rotate method order; retain every complete-call sample."""
    names = list(methods)
    answers, timings, orders = {}, {name: [] for name in names}, []
    for iteration in range(repeats + 1):
        shift = iteration % len(names)
        order = names[shift:] + names[:shift]
        orders.append(order)
        for name in order:
            start = time.perf_counter()
            current = methods[name]()
            elapsed = time.perf_counter() - start
            if name in answers and answers[name] != current:
                raise ArithmeticError("inconsistent repeated result")
            answers[name] = current
            if iteration:
                timings[name].append(elapsed)
    measurements = {name: dict(samples_seconds=timings[name],
                               median_seconds=statistics.median(timings[name])) for name in names}
    return answers, measurements, orders


def _research_helper():
    if not APPROXIMATION_SOURCE.is_file():
        raise ValueError("--include-approximation needs the repository checkout's notes/experiments helper")
    spec = importlib.util.spec_from_file_location("showcase_near_low_rank", APPROXIMATION_SOURCE)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def run_showcases(n=20, *, repeats=3, compare_generic=False, include_approximation=False,
                  allow_large=False, relative_tolerance=Fraction(1, 1000)):
    n = _order(n)
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if n > 22 and not allow_large:
        raise ValueError("orders above 22 need --allow-large, even for bounded low-rank work")
    if compare_generic and n > 22:
        raise ValueError("generic exponential comparisons are limited to order 22")
    if compare_generic and n > 14 and not perm.HAS_C_EXT:
        raise ValueError("generic comparisons above order 14 need the native extension")
    relative_tolerance = Fraction(relative_tolerance)
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative")
    helper = _research_helper() if include_approximation else None
    factors = gaussian_factors(n)
    gram = gram_matrix(factors)
    profile, weights, counts = profile_case(n)
    sources = [Path(__file__), ROOT / "perm.py"]
    if helper:
        sources.append(APPROXIMATION_SOURCE)
    report = dict(schema=1, order=n, repeats=int(repeats), warmups=1,
                  python=sys.version, platform=platform.platform(), backend=perm.get_backend(),
                  native_extension_available=perm.HAS_C_EXT,
                  timing_scope="complete serial permanent calls; input construction and interpretation excluded",
                  timing_order="rotated by method, one warmup; descriptive development measurements",
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sources}, cases=[])
    specs = [("polynomial_kernel", polynomial_kernel(n)),
             ("gaussian_moment", gram), ("profile_likelihood", profile)]
    for name, matrix in specs:
        methods = dict(auto=lambda: perm.matrix_permanent(matrix, parallel=False),
                       low_rank=lambda: perm.matrix_permanent(matrix, parallel=False, method="low_rank"))
        if compare_generic:
            methods["generic"] = lambda: perm.matrix_permanent(matrix, parallel=False, method="generic")
        if name == "profile_likelihood":
            methods["grouped_columns"] = lambda: grouped_columns_permanent(matrix)
        answers, measurements, orders = _measure(methods, repeats)
        if len(set(answers.values())) != 1:
            raise ArithmeticError("exact permanent methods disagree")
        value = answers["auto"]
        case = dict(name=name, matrix=matrix, rank=3,
                    distinct_rows=len(set(map(tuple, matrix))), distinct_columns=len(set(zip(*matrix))),
                    homogeneous_states_per_side=comb(n + 2, 2), permanent=value,
                    measurements=measurements, method_orders=orders,
                    generic_reference="computed" if compare_generic else "not run")
        if name == "polynomial_kernel":
            case["interpretation"] = dict(quantity="weighted assignment partition function",
                                           mean_weight_per_assignment=Fraction(value, factorial(n)))
        elif name == "gaussian_moment":
            case["factors"] = factors
            case["interpretation"] = dict(quantity="E product |z_i|^2 for three proper complex Gaussian factors",
                                           normalized_intensity_moment=Fraction(value, prod(gram[i][i] for i in range(n))))
        else:
            probability = profile_probability(value, weights, counts)
            if not 0 <= probability <= 1:
                raise ArithmeticError("profile probability outside [0,1]")
            case["interpretation"] = dict(quantity="probability of an unlabeled count profile on a known alphabet",
                                           category_weights=weights, count_profile=counts,
                                           sample_size=sum(counts), probability=probability)
        report["cases"].append(case)
    if helper:
        # A covariance matrix: two shared Gaussian modes plus independent sensor noise.
        u = [[1, i + 1] for i in range(n)]
        v = list(map(list, zip(*u)))
        diagonal = [Fraction(1, 1024)] * n
        baseline = gram_matrix(u)
        matrix = [[Fraction(baseline[i][j]) + (diagonal[i] if i == j else 0)
                   for j in range(n)] for i in range(n)]
        methods = dict(structured_exact=lambda: helper.diagonal_plus_low_rank_permanent(diagonal, u, v),
                       certified_approximation=lambda: helper.certify_permanent(
                           matrix, u, v, relative_tolerance=relative_tolerance))
        denominator = 1024
        integers = [[int(x * denominator) for x in row] for row in matrix]
        if compare_generic:
            methods["generic"] = lambda: Fraction(perm.matrix_permanent(
                integers, method="generic", parallel=False), denominator ** n)
        answers, measurements, orders = _measure(methods, repeats)
        truth, certificate = answers["structured_exact"], answers["certified_approximation"]
        if "generic" in answers and answers["generic"] != truth:
            raise ArithmeticError("structured and generic exact results disagree")
        if not certificate["lower_bound"] <= truth <= certificate["upper_bound"]:
            raise ArithmeticError("exact moment outside certificate")
        actual_error = abs(certificate["estimate"] - truth) / truth
        if certificate["relative_error_bound"] is not None and actual_error > certificate["relative_error_bound"]:
            raise ArithmeticError("relative error exceeds certificate")
        report["cases"].append(dict(name="noisy_gaussian_moment", matrix=matrix, factors=u,
            diagonal=diagonal, rank=n, baseline_rank=2,
            rank_justification="positive diagonal plus positive semidefinite Gram matrix is positive definite",
            research_only=True, exact_permanent=truth, certificate=certificate,
            actual_relative_error=actual_error, requested_relative_tolerance=relative_tolerance,
            measurements=measurements, method_orders=orders,
            generic_reference="computed" if compare_generic else "not run"))
    return report


def _jsonable(value):
    if isinstance(value, Fraction):
        return dict(numerator=str(value.numerator), denominator=str(value.denominator))
    if isinstance(value, int) and not isinstance(value, bool):
        return value  # Showcase orders/weights keep decimal conversion comfortably bounded.
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compare-generic", action="store_true")
    parser.add_argument("--include-approximation", action="store_true")
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--relative-tolerance", type=Fraction, default=Fraction(1, 1000))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("output exists; choose a fresh capture path")
    try:
        report = run_showcases(args.order, repeats=args.repeats, compare_generic=args.compare_generic,
                               include_approximation=args.include_approximation, allow_large=args.allow_large,
                               relative_tolerance=args.relative_tolerance)
    except ValueError as error:
        parser.error(str(error))
    for case in report["cases"]:
        print(f'{case["name"]}: order {report["order"]}, rank {case["rank"]}')
        value = case.get("permanent", case.get("exact_permanent"))
        print(f"  exact permanent: {_scientific(value)}")
        for method, timing in case["measurements"].items():
            print(f'  {method}: {timing["median_seconds"] * 1000:.3f} ms')
        for key, item in case.get("interpretation", {}).items():
            if isinstance(item, Fraction):
                print(f"  {key}: {_scientific(item)}")
        if "certificate" in case:
            certificate = case["certificate"]
            bound = certificate["relative_error_bound"]
            print("  certified relative error:", _scientific(bound) if bound is not None else "unbounded")
            print("  meets requested accuracy:", certificate["meets_tolerance"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
        print(f"Exact values and raw measurements saved to {args.output}")


if __name__ == "__main__":
    main()
