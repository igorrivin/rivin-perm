"""Sample exact permanents of random cubic bipartite graphs.

The sampled object is a uniformly random *simple labeled* 3-regular
bipartite graph with fixed parts of size 20+20.  Its 20x20 biadjacency
permanent is its number of perfect matchings.  A configuration-model draw is
conditioned on simplicity; every accepted graph has the same ``(3!)**40``
number of labeled-stub preimages, so the conditioned sampler is uniform.

Example reproducing the recorded experiment::

    python experiment_random_cubic.py --samples 100000 --threads 32 \
        --seed 0x3c6b142820260815 --output cubic_100000.npz

NetworkX is used for independent structural checks on a small prefix, not in
the hot sampling/permanent loop.
"""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from math import comb, factorial
import time

import numpy as np

import perm_binary


SIDE = 20
DEGREE = 3
DEFAULT_SEED = 0x3C6B_1428_2026_0815


def draw_simple_cubic_biadjacency(rng):
    """Draw one uniform simple cubic biadjacency matrix and packed rows."""
    proposals = 0
    while True:
        proposals += 1
        neighbors = (
            rng.permutation(DEGREE * SIDE) // DEGREE
        ).reshape(SIDE, DEGREE)
        simple = np.all(
            (neighbors[:, 0] != neighbors[:, 1])
            & (neighbors[:, 0] != neighbors[:, 2])
            & (neighbors[:, 1] != neighbors[:, 2])
        )
        if simple:
            break

    matrix = np.zeros((SIDE, SIDE), dtype=np.uint8)
    matrix[np.arange(SIDE)[:, None], neighbors] = 1
    row_masks = np.bitwise_or.reduce(
        np.left_shift(np.uint32(1), neighbors.astype(np.uint32)), axis=1
    )
    return matrix, row_masks, proposals


def count_four_cycles(row_masks):
    """Count the K_2,2 subgraphs, equivalently the four-cycles, once each."""
    total = 0
    for first in range(len(row_masks)):
        for second in range(first + 1, len(row_masks)):
            common = bin(
                int(row_masks[first]) & int(row_masks[second])
            ).count("1")
            total += common * (common - 1) // 2
    return total


def component_count(row_masks):
    """Count connected components of a packed bipartite graph."""
    n = len(row_masks)
    column_masks = [0] * n
    for row, packed in enumerate(row_masks):
        remaining = int(packed)
        while remaining:
            bit = remaining & -remaining
            column_masks[bit.bit_length() - 1] |= 1 << row
            remaining ^= bit

    remaining_left = (1 << n) - 1
    components = 0
    while remaining_left:
        components += 1
        reached_left = remaining_left & -remaining_left
        reached_right = 0
        while True:
            next_right = reached_right
            left = reached_left
            while left:
                bit = left & -left
                next_right |= int(row_masks[bit.bit_length() - 1])
                left ^= bit

            next_left = reached_left
            right = next_right
            while right:
                bit = right & -right
                next_left |= column_masks[bit.bit_length() - 1]
                right ^= bit

            if next_left == reached_left and next_right == reached_right:
                break
            reached_left, reached_right = next_left, next_right
        remaining_left &= ~reached_left
    return components


def count_simple_cubic_bipartite_graphs(n):
    """Count labeled n+n simple cubic bipartite graphs by column margins."""
    states = {(n, 0, 0, 0): 1}
    for _row in range(n):
        next_states = defaultdict(int)
        for (count0, count1, count2, count3), multiplicity in states.items():
            for take0 in range(min(3, count0) + 1):
                for take1 in range(min(3 - take0, count1) + 1):
                    take2 = 3 - take0 - take1
                    if take2 > count2:
                        continue
                    target = (
                        count0 - take0,
                        count1 + take0 - take1,
                        count2 + take1 - take2,
                        count3 + take2,
                    )
                    ways = (
                        comb(count0, take0)
                        * comb(count1, take1)
                        * comb(count2, take2)
                    )
                    next_states[target] += multiplicity * ways
        states = next_states
    return states[(0, 0, 0, n)]


def count_degree_two_with_forbidden_diagonal(n):
    """Count degree-two residual matrices avoiding one fixed matching."""
    states = {(0, 0, 0, n, 0, 0): Fraction(1)}
    for row in range(n):
        next_states = defaultdict(Fraction)
        future_count = n - row
        for state, multiplicity in states.items():
            past = list(state[:3])
            future = list(state[3:])
            for diagonal_degree in range(3):
                if future[diagonal_degree] == 0:
                    continue
                weight = (
                    multiplicity
                    * future[diagonal_degree]
                    / future_count
                )
                remaining_future = future.copy()
                remaining_future[diagonal_degree] -= 1
                categories = [
                    (0, 0, past[0]),
                    (0, 1, past[1]),
                    (1, 0, remaining_future[0]),
                    (1, 1, remaining_future[1]),
                ]
                for first in range(len(categories)):
                    for second in range(first, len(categories)):
                        side1, degree1, count1 = categories[first]
                        side2, degree2, count2 = categories[second]
                        ways = (
                            comb(count1, 2)
                            if first == second
                            else count1 * count2
                        )
                        if ways == 0:
                            continue
                        new_past = past.copy()
                        new_future = remaining_future.copy()
                        for side, degree in (
                            (side1, degree1),
                            (side2, degree2),
                        ):
                            target = new_future if side else new_past
                            target[degree] -= 1
                            target[degree + 1] += 1
                        new_past[diagonal_degree] += 1
                        next_states[tuple(new_past + new_future)] += (
                            weight * ways
                        )
        states = next_states
    answer = states[(0, 0, n, 0, 0, 0)]
    if answer.denominator != 1:
        raise RuntimeError("margin-state count unexpectedly became fractional")
    return answer.numerator


def exact_ensemble_theory(n=SIDE):
    """Return the exact graph count, acceptance probability, and mean."""
    graph_count = count_simple_cubic_bipartite_graphs(n)
    acceptance = Fraction(
        graph_count * factorial(DEGREE) ** (2 * n),
        factorial(DEGREE * n),
    )
    containing_fixed_matching = count_degree_two_with_forbidden_diagonal(n)
    mean = Fraction(
        factorial(n) * containing_fixed_matching,
        graph_count,
    )
    return graph_count, acceptance, containing_fixed_matching, mean


def average_ranks(values):
    """Return one-based average ranks, including tied groups."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def validate_with_networkx(row_masks, limit):
    """Independently validate a prefix with NetworkX."""
    if limit <= 0:
        return {"cases": 0, "passed": True, "version": None}
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "install NetworkX for structural checks, or pass "
            "--networkx-checks 0"
        ) from exc

    for packed_rows in row_masks[:limit]:
        graph = nx.Graph()
        left = range(SIDE)
        right = range(SIDE, 2 * SIDE)
        graph.add_nodes_from(left, bipartite=0)
        graph.add_nodes_from(right, bipartite=1)
        for row, packed in enumerate(packed_rows):
            remaining = int(packed)
            while remaining:
                bit = remaining & -remaining
                graph.add_edge(row, SIDE + bit.bit_length() - 1)
                remaining ^= bit
        if graph.number_of_edges() != DEGREE * SIDE:
            raise RuntimeError("sampled graph is not simple")
        if not nx.is_bipartite(graph):
            raise RuntimeError("sampled graph is not bipartite")
        if any(degree != DEGREE for _, degree in graph.degree()):
            raise RuntimeError("sampled graph is not cubic")
        matching = nx.algorithms.bipartite.maximum_matching(
            graph, top_nodes=left
        )
        if len(matching) != 2 * SIDE:
            raise RuntimeError("cubic bipartite graph lacks a perfect matching")
    return {"cases": limit, "passed": True, "version": nx.__version__}


def summarize(values, four_cycles, components):
    values_float = values.astype(float)
    mean = float(np.mean(values_float))
    centered = values_float - mean
    variance = float(np.mean(centered ** 2))
    standard_deviation = variance ** 0.5
    if standard_deviation:
        skewness = float(np.mean(centered ** 3) / standard_deviation ** 3)
        excess_kurtosis = float(
            np.mean(centered ** 4) / standard_deviation ** 4 - 3.0
        )
    else:
        skewness = None
        excess_kurtosis = None
    frequencies = Counter(map(int, values))
    mode_frequency = max(frequencies.values())
    modes = sorted(
        value for value, count in frequencies.items()
        if count == mode_frequency
    )
    quantile_points = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)
    quantiles = np.quantile(values_float, quantile_points)

    if (
        len(values) > 1
        and standard_deviation
        and float(np.std(four_cycles))
    ):
        pearson = float(np.corrcoef(values_float, four_cycles)[0, 1])
        spearman = float(np.corrcoef(
            average_ranks(values_float), average_ranks(four_cycles)
        )[0, 1])
    else:
        pearson = None
        spearman = None
    total_sum_squares = float(np.sum(centered ** 2))
    explained = 0.0
    for cycle_count in np.unique(four_cycles):
        group = values_float[four_cycles == cycle_count]
        explained += len(group) * (float(np.mean(group)) - mean) ** 2

    low = 100 * (int(values.min()) // 100)
    high = 100 * (int(values.max()) // 100 + 1)
    edges = np.arange(low, high + 100, 100)
    counts, _ = np.histogram(values, bins=edges)
    histogram = [
        [int(edges[index]), int(edges[index + 1]), int(count)]
        for index, count in enumerate(counts)
        if count
    ]
    return {
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": mean,
        "median": float(np.median(values_float)),
        "population_sd": standard_deviation,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "distinct_values": len(frequencies),
        "modes": modes,
        "mode_frequency": mode_frequency,
        "quantiles": {
            str(point): float(value)
            for point, value in zip(quantile_points, quantiles)
        },
        "histogram_width_100": histogram,
        "components": {
            str(key): value
            for key, value in sorted(Counter(map(int, components)).items())
        },
        "four_cycles": {
            "mean": float(np.mean(four_cycles)),
            "minimum": int(four_cycles.min()),
            "maximum": int(four_cycles.max()),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "variance_explained_by_group_means": (
                explained / total_sum_squares
                if total_sum_squares else None
            ),
        },
    }


def parse_integer(text):
    return int(text, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=parse_integer, default=DEFAULT_SEED)
    parser.add_argument("--networkx-checks", type=int, default=100)
    parser.add_argument("--output", help="optional compressed NumPy output")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.threads <= 0:
        parser.error("threads must be positive")
    if args.networkx_checks < 0:
        parser.error("networkx checks must be non-negative")
    if not perm_binary.HAS_BINARY_EXT:
        raise RuntimeError("build perm_binary_ext before this experiment")

    rng = np.random.default_rng(args.seed)
    values = np.empty(args.samples, dtype=np.int64)
    row_masks = np.empty((args.samples, SIDE), dtype=np.uint32)
    proposal_ids = np.empty(args.samples, dtype=np.int64)
    four_cycles = np.empty(args.samples, dtype=np.int16)
    components = np.empty(args.samples, dtype=np.int8)

    total_proposals = 0
    permanent_seconds = 0.0
    started = time.perf_counter()
    for sample in range(args.samples):
        matrix, masks, proposals = draw_simple_cubic_biadjacency(rng)
        total_proposals += proposals
        call_started = time.perf_counter()
        values[sample] = perm_binary.binary_permanent(
            matrix,
            backend="native",
            parallel=True,
            num_threads=args.threads,
        )
        permanent_seconds += time.perf_counter() - call_started
        row_masks[sample] = masks
        proposal_ids[sample] = total_proposals
        four_cycles[sample] = count_four_cycles(masks)
        components[sample] = component_count(masks)
    elapsed = time.perf_counter() - started

    graph_count, acceptance, fixed_count, exact_mean = exact_ensemble_theory()
    networkx_validation = validate_with_networkx(
        row_masks, min(args.networkx_checks, args.samples)
    )
    summary = {
        "ensemble": (
            "uniform labeled simple 3-regular bipartite graphs with "
            "fixed parts of size 20+20"
        ),
        "samples": args.samples,
        "seed": args.seed,
        "seed_hex": hex(args.seed),
        "threads": args.threads,
        "total_proposals": total_proposals,
        "acceptance_rate": args.samples / total_proposals,
        "elapsed_seconds": elapsed,
        "permanent_seconds": permanent_seconds,
        "mean_permanent_microseconds": (
            permanent_seconds * 1e6 / args.samples
        ),
        "theory": {
            "number_of_graphs": graph_count,
            "acceptance_probability": float(acceptance),
            "graphs_containing_fixed_matching": fixed_count,
            "exact_mean_fraction": [
                exact_mean.numerator, exact_mean.denominator
            ],
            "exact_mean": float(exact_mean),
        },
        "distribution": summarize(values, four_cycles, components),
        "networkx_validation": networkx_validation,
    }

    if args.output:
        np.savez_compressed(
            args.output,
            values=values,
            proposal_ids=proposal_ids,
            row_masks=row_masks,
            four_cycles=four_cycles,
            components=components,
            master_seed=np.array([args.seed], dtype=np.uint64),
            total_proposals=np.array([total_proposals], dtype=np.int64),
            threads=np.array([args.threads], dtype=np.int64),
        )
        summary["output"] = args.output
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
