#!/usr/bin/env python3
"""Independent exact weighted-matching frontier baseline (research only).

For a nonnegative integer m-by-n matrix, coefficient k is the sum of
products of edge weights over all size-k bipartite matchings. Every row may
be skipped or matched to one unused supported column. A column is forgotten
after its final incident row even when it was never matched: these are
partial matchings, so forgetting does not impose a perfect-cover condition.

This independently implements the nilpotent/frontier approach in Butera and
Pernici, "Sums of permanental minors using Grassmann algebra", Section II A,
equation (13), https://arxiv.org/abs/1406.5337. We do not run or measure their
Python/Sage implementation, and do not claim a new algorithm.

States contain only the used columns that still have incident unprocessed
rows, and each state carries an exact cardinality polynomial. Row ordering
and frontier forgetting are independent of the production permanent code.
Time depends on both the reachable states and polynomial lengths; max_states
limits states in each DP layer, not bytes, coefficient sizes, or elapsed time.
The previous and next layers can coexist in memory. Greedy ordering itself
examines all remaining rows at every step and can take quadratic row work.

CLI: python3 notes/benchmarks/weighted_matching_frontier.py matrix.json
The input is a rectangular JSON array; no CUDA, compiler or third-party
package is required. This is an exact pure-Python baseline, not a compiled
performance competitor or an installed public API.
"""
from __future__ import annotations

import argparse
import json
import operator
from pathlib import Path


SCHEMA = "weighted-matching-frontier-v1"
_bit_count = getattr(int, "bit_count", lambda value: bin(value).count("1"))


class FrontierStateLimitError(RuntimeError):
    """A new DP layer would exceed the explicit state limit."""

    def __init__(self, limit, attempted_states, row_index, processed_rows):
        self.limit = limit
        self.attempted_states = attempted_states
        self.row_index = row_index
        self.processed_rows = processed_rows
        super().__init__(
            f"frontier state limit {limit} exceeded while processing row "
            f"{row_index}: at least {attempted_states} states required"
        )


def _snapshot(matrix):
    if isinstance(matrix, (str, bytes)):
        raise TypeError("matrix must be an iterable of integer rows")
    try:
        rows = tuple(tuple(row) for row in matrix)
    except TypeError as error:
        raise TypeError("matrix must be an iterable of integer rows") from error
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError("matrix must be rectangular")
    normalized = []
    for row in rows:
        values = []
        for value in row:
            try:
                integer = operator.index(value)
            except TypeError as error:
                raise TypeError("matrix entries must be integers") from error
            if integer < 0:
                raise ValueError("matrix entries must be nonnegative")
            values.append(integer)
        normalized.append(tuple(values))
    return tuple(normalized), width


def _greedy_order(supports, masks, column_count):
    """Minimize next-cut width, then row width, support size, original index."""
    remaining = list(column_count)
    pending = set(range(len(supports)))
    live = 0
    order = []
    while pending:
        def score(index):
            closing = sum(1 << column for column, _ in supports[index]
                          if remaining[column] == 1)
            during = live | masks[index]
            after = during & ~closing
            return (_bit_count(after), _bit_count(during), len(supports[index]), index)
        index = min(pending, key=score)
        pending.remove(index)
        order.append(index)
        live |= masks[index]
        for column, _ in supports[index]:
            remaining[column] -= 1
            if remaining[column] == 0:
                live &= ~(1 << column)
    return tuple(order)


def weighted_matching_frontier(matrix, *, row_order=None, max_states=1_000_000):
    """Return exact coefficients and structural/resource statistics.

    ``row_order=None`` selects a deterministic greedy ordering. Otherwise it
    must be a permutation of all row indices. Entries accept ``__index__``
    integers, including booleans, and may be arbitrarily large. Coefficients
    are a tuple of length ``min(m, n) + 1`` including trailing zeros; the
    zero-row or zero-column result is ``(1,)``. Inputs are snapshotted.

    ``frontier_widths`` counts columns incident on both sides of each row
    cut. ``peak_row_frontier`` also includes columns needed only in the
    currently processed row. ``state_counts`` counts reachable used-column
    masks after each row, not individual polynomial coefficients. A state
    limit failure raises ``FrontierStateLimitError`` and returns no result.
    """
    if isinstance(max_states, bool):
        raise TypeError("max_states must be a positive integer")
    try:
        max_states = operator.index(max_states)
    except TypeError as error:
        raise TypeError("max_states must be a positive integer") from error
    if max_states < 1:
        raise ValueError("max_states must be a positive integer")
    rows, width = _snapshot(matrix)
    height = len(rows)
    supports = tuple(tuple((j, value) for j, value in enumerate(row) if value)
                     for row in rows)
    masks = tuple(sum(1 << column for column, _ in row) for row in supports)
    remaining = [0] * width
    for row in supports:
        for column, _ in row:
            remaining[column] += 1
    if row_order is None:
        order = _greedy_order(supports, masks, remaining)
        ordering = "greedy"
    else:
        try:
            supplied = tuple(row_order)
            if any(isinstance(value, bool) for value in supplied):
                raise TypeError("boolean row indices are not allowed")
            order = tuple(operator.index(value) for value in supplied)
        except TypeError as error:
            raise TypeError("row_order must contain integer row indices") from error
        if len(order) != height or set(order) != set(range(height)):
            raise ValueError("row_order must be a permutation of all row indices")
        ordering = "explicit"

    states = {0: [1]}
    live = 0
    peak_states = 1
    peak_frontier = peak_row_frontier = 0
    state_counts, frontier_widths = [], []
    degree_limit = min(height, width)
    for step, index in enumerate(order):
        during = live | masks[index]
        peak_row_frontier = max(peak_row_frontier, _bit_count(during))
        live = during
        for column, _ in supports[index]:
            remaining[column] -= 1
            if remaining[column] == 0:
                live &= ~(1 << column)
        length = min(step + 1, degree_limit) + 1
        next_states = {}

        def accumulate(mask, polynomial, shift, weight):
            # Forgetting merges states regardless of whether an expired
            # column was used. It must not demand that every column is used.
            key = mask & live
            target = next_states.get(key)
            if target is None:
                if len(next_states) >= max_states:
                    raise FrontierStateLimitError(max_states, len(next_states) + 1, index, step)
                target = [0] * length
                next_states[key] = target
            for degree, coefficient in enumerate(polynomial):
                if coefficient and degree + shift < length:
                    target[degree + shift] += weight * coefficient

        for used, polynomial in states.items():
            accumulate(used, polynomial, 0, 1)
            for column, weight in supports[index]:
                bit = 1 << column
                if not used & bit:
                    accumulate(used | bit, polynomial, 1, weight)
        states = next_states
        peak_states = max(peak_states, len(states))
        frontier = _bit_count(live)
        peak_frontier = max(peak_frontier, frontier)
        state_counts.append(len(states))
        frontier_widths.append(frontier)
    # All columns have expired, leaving one accumulated cardinality vector.
    if set(states) != {0} or live:
        raise RuntimeError("internal frontier did not close")
    coefficients = tuple(states[0])
    return {
        "schema": SCHEMA, "status": "COMPLETE", "exact": True,
        "algorithm": "python-row-frontier-cardinality-dp",
        "shape": (height, width), "coefficients": coefficients,
        "ordering": ordering, "row_order": order, "max_states": max_states,
        "peak_states": peak_states, "peak_frontier": peak_frontier,
        "peak_row_frontier": peak_row_frontier,
        "state_counts": tuple(state_counts), "frontier_widths": tuple(frontier_widths),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="rectangular nonnegative integer JSON matrix")
    parser.add_argument("--max-states", type=int, default=1_000_000)
    parser.add_argument("--row-order", help="comma-separated row permutation; default: greedy")
    args = parser.parse_args(argv)
    try:
        matrix = json.loads(args.matrix.read_text())
        order = (None if args.row_order is None else
                 [] if args.row_order == "" else [int(x) for x in args.row_order.split(",")])
        result = weighted_matching_frontier(matrix, row_order=order, max_states=args.max_states)
    except FrontierStateLimitError as error:
        print(json.dumps({"schema": SCHEMA, "status": "RESOURCE_LIMIT", "exact": False,
                          "error": str(error), "max_states": error.limit,
                          "attempted_states": error.attempted_states, "row_index": error.row_index}))
        return 2
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
