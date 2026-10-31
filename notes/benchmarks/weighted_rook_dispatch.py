"""Opt-in exact weighted-rook selector for a source checkout.

This preserves the public square-matrix contract and uses its validated helpers
without repeating normalization/component discovery on fallback. The original
public entry point and both independent frontier implementations are unchanged.
No compilation or automatic library discovery occurs. A verified frontier
library is optional and must be passed explicitly to WeightedRookDispatcher.
"""
from __future__ import annotations

import operator
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import perm_polynomial as public
from notes.benchmarks import weighted_matching_frontier as python_frontier
from notes.benchmarks import weighted_matching_frontier_native as native_frontier

SCHEMA = "weighted-rook-source-dispatch-v1"
POLICY = {
    "minimum_order": 8, "maximum_order": 18, "maximum_weight_bits": 10,
    "python_candidate_max_row_degree": 3, "python_max_cut_width": 4,
    "native_minimum_order": 16, "native_max_row_degree": 6,
    "native_density_denominator": 4, "native_min_cut_width": 6,
    "native_max_cut_width": 11, "native_max_row_width": 12,
}


def _positive(value, name):
    if isinstance(value, bool):
        raise TypeError(name + " must be a positive integer")
    value = operator.index(value)
    if value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _frontier_plan(matrix, native_available):
    """Bounded structural inspection, with no family/seed/reference lookup."""
    n = len(matrix)
    if not POLICY["minimum_order"] <= n <= POLICY["maximum_order"]:
        return {"method": "public", "reason": "uncalibrated_order"}
    # A full first row already gives the original public dense shortcut.
    if all(matrix[0]):
        return {"method": "public", "reason": "dense_first_row"}
    supports = []
    masks = []
    counts = [0] * n
    maximum_degree = edge_count = maximum_bits = 0
    for row in matrix:
        supported = tuple((j, weight) for j, weight in enumerate(row) if weight)
        maximum_degree = max(maximum_degree, len(supported))
        if maximum_degree > POLICY["native_max_row_degree"]:
            return {"method": "public", "reason": "high_row_degree"}
        for j, weight in supported:
            counts[j] += 1
            maximum_bits = max(maximum_bits, weight.bit_length())
        if maximum_bits > POLICY["maximum_weight_bits"]:
            return {"method": "public", "reason": "large_weights"}
        supports.append(supported)
        masks.append(sum(1 << j for j, _ in supported))
        edge_count += len(supported)
    python_candidate = maximum_degree <= POLICY["python_candidate_max_row_degree"]
    native_candidate = (native_available and n >= POLICY["native_minimum_order"]
                        and POLICY["native_density_denominator"] * edge_count <= n * n)
    if not python_candidate and not native_candidate:
        return {"method": "public", "reason": "no_cheap_candidate"}
    order = python_frontier._greedy_order(supports, masks, counts)
    live = peak_cut = peak_row = 0
    for index in order:
        live |= masks[index]
        peak_row = max(peak_row, python_frontier._bit_count(live))
        for j, _ in supports[index]:
            counts[j] -= 1
            if counts[j] == 0:
                live &= ~(1 << j)
        peak_cut = max(peak_cut, python_frontier._bit_count(live))
    details = {"row_order": order, "peak_cut_width": peak_cut,
               "peak_row_width": peak_row, "state_upper_bound": 1 << peak_cut,
               "edge_count": edge_count, "maximum_row_degree": maximum_degree,
               "maximum_weight_bits": maximum_bits}
    if python_candidate and peak_cut <= POLICY["python_max_cut_width"]:
        return dict(details, method="python", reason="narrow_frontier")
    if (native_candidate and POLICY["native_min_cut_width"] <= peak_cut <= POLICY["native_max_cut_width"]
            and peak_row <= POLICY["native_max_row_width"]):
        return dict(details, method="native", reason="bounded_sparse_frontier")
    return dict(details, method="public", reason="frontier_outside_policy")


class WeightedRookDispatcher:
    """Explicit source API; all results use exact nonnegative integer arithmetic.

    `evaluate` returns diagnostics and coefficients; `weighted_rook_polynomial`
    returns only the coefficient tuple. Selection is enabled only for backend
    auto resolving to native and an effective thread count of one. Other backend
    and thread settings follow the unchanged public implementation. The native
    option requires an explicitly supplied existing verified library.
    """

    def __init__(self, native_library=None, *, python_max_states=64,
                 native_max_states=4096, native_max_workspace_bytes=16 * 1024**2):
        self.python_max_states = _positive(python_max_states, "python_max_states")
        self.native_max_states = native_frontier._positive_u64(native_max_states, "native_max_states")
        self.native_max_workspace_bytes = native_frontier._positive_u64(
            native_max_workspace_bytes, "native_max_workspace_bytes")
        self.native = (None if native_library is None
                       else native_frontier.FrontierBaseline(native_library))

    def weighted_rook_polynomial(self, matrix, *, backend="auto", parallel=False, num_threads=1):
        return self.evaluate(matrix, backend=backend, parallel=parallel,
                             num_threads=num_threads)["coefficients"]

    def evaluate(self, matrix, *, backend="auto", parallel=False, num_threads=1):
        if not isinstance(parallel, bool):
            raise TypeError("parallel must be a boolean")
        num_threads = public._validate_num_threads(num_threads)
        selected = public._select_backend(backend)
        normalized = public._snapshot_nonnegative_integer_matrix(matrix)
        order = len(normalized)
        decision = {"method": "public", "reason": "public_fast_path"}
        fallback = None
        frontier_record = None

        def result(coefficients):
            coefficients = tuple(coefficients)
            if (len(coefficients) != order + 1 or coefficients[0] != 1
                    or any(type(x) is not int or x < 0 for x in coefficients)):
                raise RuntimeError("invalid complete exact matching vector")
            return {"schema": SCHEMA, "status": "COMPLETE", "exact": True,
                    "coefficients": coefficients, "decision": decision,
                    "resource_fallback": fallback, "frontier_record": frontier_record}

        if order == 0:
            return result((1,))
        if order == 1:
            return result((1, normalized[0][0]))
        if not any(any(row) for row in normalized):
            return result((1,) + (0,) * order)

        threads = num_threads if parallel else 1
        components = public._weighted_rook_components_validated(normalized)
        if (len(components) == 1 and max(map(len, components[0])) == order
                and min(map(len, components[0])) > 1):
            if backend.lower() != "auto" or selected != "native":
                decision = {"method": "public", "reason": "explicit_or_unavailable_backend"}
            elif threads != 1:
                decision = {"method": "public", "reason": "uncalibrated_threads"}
            elif min(map(len, components[0])) != order:
                decision = {"method": "public", "reason": "isolated_endpoints"}
            else:
                decision = _frontier_plan(normalized, self.native is not None)
            try:
                if decision["method"] == "python":
                    frontier_record = python_frontier.weighted_matching_frontier(
                        normalized, row_order=decision["row_order"], max_states=self.python_max_states)
                elif decision["method"] == "native":
                    frontier_record = self.native.evaluate(
                        normalized, row_order=decision["row_order"], max_states=self.native_max_states,
                        max_workspace_bytes=self.native_max_workspace_bytes)
            except (python_frontier.FrontierStateLimitError,
                    native_frontier.FrontierStateLimitError,
                    native_frontier.FrontierMemoryLimitError,
                    native_frontier.FrontierWidthLimitError) as error:
                fallback = {"attempted_method": decision["method"],
                            "kind": type(error).__name__, "details": vars(error)}
                decision = dict(decision, method="public", reason="frontier_resource_fallback")
            if frontier_record is not None:
                if frontier_record.get("status") != "COMPLETE" or frontier_record.get("exact") is not True:
                    raise RuntimeError("frontier returned an incomplete or inexact result")
                return result(frontier_record["coefficients"])
            return result(public._weighted_rook_polynomial_crt_validated(normalized, selected, threads))

        # Keep the original component cases and convolution, using the same
        # normalized input and helpers. No second public normalization/planning.
        decision = {"method": "public", "reason": "factorable_support"}
        coefficients = [1]
        for rows, columns in components:
            degree = min(len(rows), len(columns))
            if degree == 1:
                factor = (1, sum(normalized[row][column] for row in rows for column in columns))
            elif degree <= 3:
                factor = public._weighted_rook_small_component_validated(normalized, rows, columns)
            else:
                size = max(len(rows), len(columns))
                block = tuple(tuple(normalized[rows[row]][columns[column]]
                                    if row < len(rows) and column < len(columns) else 0
                                    for column in range(size)) for row in range(size))
                exact = public._weighted_rook_polynomial_crt_validated(block, selected, threads)
                if any(exact[degree + 1:]):
                    raise RuntimeError("weighted-rook component exceeds its matching degree")
                factor = exact[:degree + 1]
            product = [0] * (len(coefficients) + len(factor) - 1)
            for left_degree, left in enumerate(coefficients):
                for right_degree, right in enumerate(factor):
                    product[left_degree + right_degree] += left * right
            coefficients = product
        return result(tuple(coefficients) + (0,) * (order + 1 - len(coefficients)))
