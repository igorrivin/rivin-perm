"""Definition-level checks of weighted matching structural dispatch.

The oracle enumerates subsets of supported edges and accepts exactly those
with distinct row and column endpoints. It uses neither component factors,
dynamic programming, permanents nor modular reconstruction.
"""
from __future__ import annotations

import itertools
import math
import random

import pytest

import perm_polynomial as polynomial


BACKENDS = ["python", pytest.param("native", marks=pytest.mark.skipif(
    not polynomial.HAS_POLYNOMIAL_EXT, reason="optional polynomial extension unavailable"))]


def matching_definition(matrix):
    n = len(matrix)
    edges = [(i, j, int(w)) for i, row in enumerate(matrix)
             for j, w in enumerate(row) if w]
    result = [1] + [0] * n
    maximum = min(len({i for i, _, _ in edges}), len({j for _, j, _ in edges}))
    for k in range(1, maximum + 1):
        for chosen in itertools.combinations(edges, k):
            if len({i for i, _, _ in chosen}) == k and len({j for _, j, _ in chosen}) == k:
                result[k] += math.prod(w for _, _, w in chosen)
    return tuple(result)


def graph_components(matrix):
    """Ordinary set-valued graph traversal with explicit row/column labels."""
    adjacency = {}
    for i, row in enumerate(matrix):
        for j, w in enumerate(row):
            if w:
                a, b = ("row", i), ("column", j)
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
    remaining, result = set(adjacency), set()
    while remaining:
        pending, found = [next(iter(remaining))], set()
        while pending:
            vertex = pending.pop()
            if vertex not in found:
                found.add(vertex)
                pending.extend(adjacency[vertex] - found)
        remaining -= found
        result.add(frozenset(found))
    return result


def component_vertices(components):
    result = []
    for rows, columns in components:
        assert rows and columns
        assert len(rows) == len(set(rows)) and len(columns) == len(set(columns))
        result.append(frozenset([*(('row', i) for i in rows),
                                *(('column', j) for j in columns)]))
    assert len(result) == len(set(result)), "a support component was emitted twice"
    return set(result)


def permuted(matrix, seed):
    rng = random.Random(seed)
    rows, columns = list(range(len(matrix))), list(range(len(matrix)))
    rng.shuffle(rows)
    rng.shuffle(columns)
    return [[matrix[i][j] for j in columns] for i in rows]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_two_by_two_weight_pattern_matches_edge_subset_definition(backend):
    for values in itertools.product((0, 2, 5), repeat=4):
        matrix = [list(values[:2]), list(values[2:])]
        assert polynomial.weighted_rook_polynomial(
            matrix, backend=backend, num_threads=1,
        ) == matching_definition(matrix)


def test_all_three_by_three_support_components_match_labeled_graph():
    for pattern in range(1 << 9):
        # Unequal positive weights must leave support connectivity unchanged.
        matrix = [[(2 + 3 * (3*i+j)) if pattern & (1 << (3*i+j)) else 0
                   for j in range(3)] for i in range(3)]
        actual = polynomial._weighted_rook_components_validated(matrix)
        assert component_vertices(actual) == graph_components(matrix)


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_binary_three_by_three_matrices_match_definition(backend):
    for pattern in range(1 << 9):
        matrix = [[int(bool(pattern & (1 << (3*i+j)))) for j in range(3)] for i in range(3)]
        assert polynomial.weighted_rook_polynomial(
            matrix, backend=backend, num_threads=1,
        ) == matching_definition(matrix)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fresh_random_weights_permutations_and_transposes(backend):
    rng = random.Random(2026090807)
    for n in range(2, 6):
        for density in (0.12, 0.35, 0.7, 1.0):
            matrix = [[rng.randrange(1, 18) if rng.random() < density else 0
                       for _ in range(n)] for _ in range(n)]
            expected = matching_definition(matrix)
            original = [row[:] for row in matrix]
            for candidate in (matrix, transpose(matrix), permuted(matrix, 71*n + int(100*density))):
                result = polynomial.weighted_rook_polynomial(
                    candidate, backend=backend, parallel=False, num_threads=2,
                )
                assert result == expected
                assert isinstance(result, tuple) and len(result) == n + 1
                assert all(type(value) is int for value in result)
            assert matrix == original


def rectangular_component_case():
    # Dense 2x3 and 3x2 components plus an isolated edge, row and column.
    matrix = [[0] * 7 for _ in range(7)]
    for i, row in enumerate(((2, 7, 11), (13, 17, 19))):
        matrix[i][:3] = row
    for i, row in enumerate(((23, 29), (31, 37), (41, 43)), 2):
        matrix[i][3:5] = row
    matrix[5][5] = 47
    return matrix


@pytest.mark.parametrize("backend", BACKENDS)
def test_rectangular_factors_have_independent_row_and_column_permutations(backend):
    matrix = rectangular_component_case()
    expected = matching_definition(matrix)
    assert expected[-2:] == (0, 0)
    for candidate in (matrix, permuted(matrix, 2026090811), transpose(permuted(matrix, 2026090813))):
        assert polynomial.weighted_rook_polynomial(
            candidate, backend=backend, num_threads=1,
        ) == expected
        assert component_vertices(polynomial._weighted_rook_components_validated(candidate)) == graph_components(candidate)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orientation", ["rows", "columns"])
def test_one_rectangular_connected_component_with_isolates(backend, orientation):
    # All five columns but only two nonzero rows. This reaches the dispatch
    # boundary where maximum component side equals the original order.
    matrix = [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29]] + [[0] * 5 for _ in range(3)]
    if orientation == "columns":
        matrix = transpose(matrix)
    result = polynomial.weighted_rook_polynomial(matrix, backend=backend, num_threads=1)
    assert result == matching_definition(matrix)
    assert result[3:] == (0, 0, 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_stars_same_numeric_vertex_labels_and_unmatched_isolates(backend):
    matrix = [[0] * 8 for _ in range(8)]
    matrix[0][1:5] = [2, 3, 5, 7]
    matrix[1][0], matrix[2][0], matrix[3][0] = 11, 13, 17
    matrix[5][6] = 19
    # Row 0 and column 0 belong to DIFFERENT support components.
    assert len(graph_components(matrix)) == 3
    for candidate in (matrix, permuted(matrix, 2026090817)):
        assert polynomial.weighted_rook_polynomial(
            candidate, backend=backend, num_threads=1,
        ) == matching_definition(matrix)


@pytest.mark.parametrize("backend", BACKENDS)
def test_block_triangular_edges_contribute_to_partial_matchings(backend):
    matrix = [[2, 3, 7, 0], [5, 11, 0, 13], [0, 0, 17, 19], [0, 0, 23, 29]]
    disconnected = [row[:] for row in matrix]
    disconnected[0][2] = disconnected[1][3] = 0
    expected, false_factorization = matching_definition(matrix), matching_definition(disconnected)
    assert expected[-1] == false_factorization[-1]
    assert expected[1] == false_factorization[1] + 20
    assert len(graph_components(matrix)) == 1
    for candidate in (matrix, permuted(matrix, 2026090819)):
        assert polynomial.weighted_rook_polynomial(
            candidate, backend=backend, num_threads=1,
        ) == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_huge_weight_rectangular_factors_and_convolution(backend):
    h = 10**110 + 2**219 + 31
    matrix = [[h, 2, 3, 0, 0], [5, h + 7, 11, 0, 0],
              [0, 0, 0, h + 13, 17], [0, 0, 0, 19, h + 23],
              [0, 0, 0, 29, 31]]
    expected = matching_definition(matrix)
    assert expected[-1] == 0 and expected[-2] > h**3
    for candidate in (matrix, permuted(matrix, 2026090823)):
        assert polynomial.weighted_rook_polynomial(
            candidate, backend=backend, num_threads=1,
        ) == expected


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("matrix", [[], [[0]], [[37]], [[0]*4 for _ in range(4)]])
def test_empty_zero_and_singleton_return_full_square_cardinality_vector(backend, matrix):
    assert polynomial.weighted_rook_polynomial(
        matrix, backend=backend, num_threads=1,
    ) == matching_definition(matrix)


@pytest.mark.parametrize("backend", BACKENDS)
def test_validation_cannot_be_bypassed_by_empty_or_disconnected_support(backend):
    for matrix, error in (([[0, 0], [0, -3]], ValueError),
                          ([[0, 0], [0, 2.0]], TypeError),
                          ([[0, 0], [0]], ValueError),
                          ([[0, 0, 0], [0, 0, 0]], ValueError),
                          ([[0]*64 for _ in range(64)], ValueError)):
        with pytest.raises(error):
            polynomial.weighted_rook_polynomial(matrix, backend=backend)
    for matrix in ([], [[0, 0], [0, 0]], [[2, 0], [0, 3]]):
        with pytest.raises(TypeError, match="parallel"):
            polynomial.weighted_rook_polynomial(matrix, backend=backend, parallel=1)
        with pytest.raises(ValueError, match="num_threads"):
            polynomial.weighted_rook_polynomial(matrix, backend=backend, num_threads=-1)


def test_portable_auto_dispatch_requires_no_native_extension(monkeypatch):
    monkeypatch.setattr(polynomial, "HAS_POLYNOMIAL_EXT", False)
    matrix = permuted(rectangular_component_case(), 2026090829)
    assert polynomial.weighted_rook_polynomial(
        matrix, backend="auto", parallel=False,
    ) == matching_definition(matrix)
    with pytest.raises(RuntimeError, match="unavailable"):
        polynomial.weighted_rook_polynomial(matrix, backend="native")


@pytest.mark.parametrize("backend", BACKENDS)
def test_decomposed_three_by_three_and_star_use_exact_tiny_factors(monkeypatch, backend):
    h = 2**251 + 53
    matrix = [[h, 2, 3, 0, 0], [5, h + 7, 11, 0, 0],
              [13, 17, h + 19, 0, 0], [0, 0, 0, 23, h + 29],
              [0, 0, 0, 0, 0]]
    expected = matching_definition(matrix)

    def forbidden(*args, **kwargs):
        pytest.fail("decomposed components of minimum side at most three must avoid CRT")

    monkeypatch.setattr(polynomial, "_weighted_rook_polynomial_crt_validated", forbidden)
    for candidate in (matrix, permuted(matrix, 2026090831)):
        assert polynomial.weighted_rook_polynomial(
            candidate, backend=backend, num_threads=1,
        ) == expected


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orientation", ["three_rows", "three_columns"])
def test_decomposed_three_by_four_with_star_uses_short_side_threshold(monkeypatch, backend, orientation):
    matrix = [[2, 3, 5, 7, 0, 0], [11, 13, 17, 19, 0, 0],
              [23, 29, 31, 37, 0, 0], [0, 0, 0, 0, 41, 0],
              [0, 0, 0, 0, 43, 0], [0, 0, 0, 0, 0, 0]]
    if orientation == "three_columns":
        matrix = transpose(matrix)
    matrix = permuted(matrix, 2026090837)
    expected = matching_definition(matrix)

    def forbidden(*args, **kwargs):
        pytest.fail("a decomposed 3x4/4x3 component must use at most eight exact states")

    monkeypatch.setattr(polynomial, "_weighted_rook_polynomial_crt_validated", forbidden)
    assert polynomial.weighted_rook_polynomial(
        matrix, backend=backend, parallel=False,
    ) == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_decomposed_long_rectangular_factor_is_bounded_by_three_vertices(monkeypatch, backend):
    # A complete weighted K_(3,60) plus a disjoint K_(60,1), embedded at
    # the public order limit. The closed form chooses k endpoints on each
    # side and a bijection; edge-subset enumeration would be inappropriate.
    n, width, weight, star_weight = 63, 60, 2**170 + 61, 10**70 + 67
    matrix = [[weight] * width + [0] * (n-width) for _ in range(3)]
    matrix += [[0] * width + [star_weight, 0, 0] for _ in range(n-3)]
    first = [math.comb(3, k) * math.comb(width, k) * math.factorial(k) * weight**k
             for k in range(4)]
    star_sum = (n-3) * star_weight
    expected = tuple((first[k] if k < len(first) else 0)
                     + (first[k-1] * star_sum if 1 <= k <= len(first) else 0)
                     for k in range(n+1))

    def forbidden(*args, **kwargs):
        pytest.fail("a 3x60 factor must use its short side, without padding into CRT")

    monkeypatch.setattr(polynomial, "_weighted_rook_polynomial_crt_validated", forbidden)
    assert polynomial.weighted_rook_polynomial(
        permuted(matrix, 2026090839), backend=backend, num_threads=1,
    ) == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_component_with_both_sides_above_three_retains_crt(monkeypatch, backend):
    matrix = [[2, 3, 5, 7, 0], [11, 13, 17, 19, 0],
              [23, 29, 31, 37, 0], [41, 43, 47, 53, 0],
              [0, 0, 0, 0, 59]]
    matrix = permuted(matrix, 2026090841)
    expected = matching_definition(matrix)
    actual_crt = polynomial._weighted_rook_polynomial_crt_validated
    calls = []

    def record(block, selected, threads):
        calls.append((len(block), selected, threads))
        return actual_crt(block, selected, threads)

    monkeypatch.setattr(polynomial, "_weighted_rook_polynomial_crt_validated", record)
    assert polynomial.weighted_rook_polynomial(
        matrix, backend=backend, parallel=False, num_threads=2,
    ) == expected
    assert calls == [(4, backend, 1)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [2, 3])
def test_standalone_connected_small_input_retains_existing_crt_contract(monkeypatch, backend, n):
    matrix = [[2 + 3*i + 5*j for j in range(n)] for i in range(n)]
    actual_crt = polynomial._weighted_rook_polynomial_crt_validated
    calls = []

    def record(block, selected, threads):
        calls.append((tuple(tuple(row) for row in block), selected, threads))
        return actual_crt(block, selected, threads)

    monkeypatch.setattr(polynomial, "_weighted_rook_polynomial_crt_validated", record)
    assert polynomial.weighted_rook_polynomial(
        matrix, backend=backend, parallel=False, num_threads=2,
    ) == matching_definition(matrix)
    assert calls == [(tuple(tuple(row) for row in matrix), backend, 1)]
