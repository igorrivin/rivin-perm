"""Exact frontier baseline versus definition-level matching enumeration."""
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "notes/benchmarks/weighted_matching_frontier.py"
if not SOURCE.is_file():
    pytest.skip("checkout-only independent weighted frontier baseline", allow_module_level=True)
SPEC = importlib.util.spec_from_file_location("weighted_frontier_test", SOURCE)
frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier)


def exhaustive_matchings(matrix):
    """Select edge sets directly, enforcing distinct endpoints by definition."""
    rows, columns = len(matrix), len(matrix[0]) if matrix else 0
    edges = [(i, j, w) for i, row in enumerate(matrix) for j, w in enumerate(row) if w]
    result = [1] + [0] * min(rows, columns)
    for k in range(1, len(result)):
        for chosen in itertools.combinations(edges, k):
            if len({i for i, _, _ in chosen}) == k and len({j for _, j, _ in chosen}) == k:
                result[k] += math.prod(w for _, _, w in chosen)
    return tuple(result)


@pytest.mark.parametrize("matrix, expected", [
    ([], (1,)), ([[]], (1,)), ([[], []], (1,)),
    ([[0, 0, 0], [0, 0, 0]], (1, 0, 0)),
    ([[2, 3, 5]], (1, 10)), ([[2], [3], [5]], (1, 10)),
    ([[2, 0], [0, 3]], (1, 5, 6)),
    ([[1, 2, 0], [0, 3, 4]], (1, 10, 15)),
])
def test_rectangular_empty_zero_and_partial_matching_examples(matrix, expected):
    report = frontier.weighted_matching_frontier(matrix)
    assert report["coefficients"] == expected == exhaustive_matchings(matrix)
    assert report["status"] == "COMPLETE" and report["exact"] is True
    assert len(report["state_counts"]) == len(matrix)
    assert report["peak_states"] == max((1,) + report["state_counts"])
    assert report["peak_frontier"] == max((0,) + report["frontier_widths"])
    if matrix:
        assert report["state_counts"][-1] == 1 and report["frontier_widths"][-1] == 0


def test_forgetting_columns_does_not_require_them_to_be_matched():
    diagonal = [[2, 0, 0], [0, 3, 0], [0, 0, 5]]
    report = frontier.weighted_matching_frontier(diagonal, max_states=1)
    assert report["coefficients"] == (1, 10, 31, 30)
    assert report["state_counts"] == (1, 1, 1)
    assert report["peak_frontier"] == 0 and report["peak_row_frontier"] == 1


def test_shared_column_cannot_be_selected_twice():
    report = frontier.weighted_matching_frontier([[7, 0], [11, 0]], row_order=[0, 1])
    assert report["coefficients"] == (1, 18, 0)
    assert report["state_counts"] == (2, 1)
    assert report["frontier_widths"] == (1, 0)


def test_greedy_order_reduces_interleaved_block_frontier():
    matrix = [[1, 2, 0, 0], [0, 0, 3, 4], [5, 6, 0, 0], [0, 0, 7, 8]]
    original = frontier.weighted_matching_frontier(matrix, row_order=range(4))
    reordered = frontier.weighted_matching_frontier(matrix)
    assert reordered["coefficients"] == original["coefficients"] == exhaustive_matchings(matrix)
    assert reordered["peak_states"] < original["peak_states"]
    assert reordered["peak_frontier"] == 2 < original["peak_frontier"] == 4
    assert frontier.weighted_matching_frontier(matrix) == reordered


def test_seeded_rectangular_random_cases_against_independent_edge_sets():
    rng = random.Random(2026090701)
    for m in range(1, 5):
        for n in range(6):
            for _ in range(3):
                matrix = [[rng.randrange(1, 8) if rng.random() < 0.45 else 0 for _ in range(n)]
                          for _ in range(m)]
                order = list(range(m))
                rng.shuffle(order)
                expected = exhaustive_matchings(matrix)
                assert frontier.weighted_matching_frontier(matrix)["coefficients"] == expected
                assert frontier.weighted_matching_frontier(matrix, row_order=order)["coefficients"] == expected
                transposed = [list(row) for row in zip(*matrix)]
                assert frontier.weighted_matching_frontier(transposed)["coefficients"] == expected


def test_matches_existing_weighted_rook_with_zero_padding_for_rectangles():
    from perm_polynomial import weighted_rook_polynomial
    rng = random.Random(2026090702)
    for m, n in [(1, 4), (4, 1), (2, 3), (3, 2), (3, 3), (4, 4), (5, 5)]:
        for _ in range(2):
            matrix = [[rng.randrange(5) for _ in range(n)] for _ in range(m)]
            size = max(m, n)
            padded = [[matrix[i][j] if i < m and j < n else 0 for j in range(size)]
                      for i in range(size)]
            actual = frontier.weighted_matching_frontier(matrix)["coefficients"]
            expected = weighted_rook_polynomial(padded, backend="python", parallel=False)
            assert actual == expected[:min(m, n) + 1]
            assert all(value == 0 for value in expected[min(m, n) + 1:])


def test_disconnected_rectangular_blocks_and_isolated_vertices():
    left = [[2, 3, 0], [0, 5, 7]]
    right = [[11, 13]]
    matrix = [row + [0, 0, 0] for row in left] + [[0, 0, 0] + right[0] + [0]] + [[0] * 6]
    expected = [0] * 5
    for i, a in enumerate(exhaustive_matchings(left)):
        for j, b in enumerate(exhaustive_matchings(right)):
            expected[i + j] += a * b
    assert frontier.weighted_matching_frontier(matrix)["coefficients"] == tuple(expected)


def test_arbitrary_precision_weights_booleans_and_index_objects():
    huge = 10**200
    class Integer:
        def __index__(self):
            return huge
    matrix = [[Integer(), True], [False, huge + 1]]
    original = [row[:] for row in matrix]
    result = frontier.weighted_matching_frontier(matrix)["coefficients"]
    assert result == (1, 2 * huge + 2, huge * (huge + 1))
    assert matrix == original


def test_consumes_generator_input_and_explicit_order():
    matrix = [[2, 3], [5, 7], [11, 13]]
    result = frontier.weighted_matching_frontier((iter(row) for row in matrix), row_order=iter([2, 0, 1]))
    assert result["coefficients"] == exhaustive_matchings(matrix)
    assert result["row_order"] == (2, 0, 1) and result["ordering"] == "explicit"


@pytest.mark.parametrize("matrix, exception", [
    ([[1], [2, 3]], ValueError), ([[-1]], ValueError), ([[1.0]], TypeError),
    ([["1"]], TypeError), ([[None]], TypeError), ([1, 2], TypeError), (None, TypeError),
    ("123", TypeError),
])
def test_invalid_matrix_rejected(matrix, exception):
    with pytest.raises(exception):
        frontier.weighted_matching_frontier(matrix)


@pytest.mark.parametrize("order, exception", [
    ([0], ValueError), ([0, 0], ValueError), ([0, 2], ValueError),
    ([-1, 0], ValueError), ([0, 1.0], TypeError), ([False, 1], TypeError), (2, TypeError),
])
def test_invalid_row_order_rejected(order, exception):
    with pytest.raises(exception):
        frontier.weighted_matching_frontier([[1, 2], [3, 4]], row_order=order)


@pytest.mark.parametrize("limit, exception", [(0, ValueError), (-1, ValueError), (True, TypeError),
                                            (2.5, TypeError), (None, TypeError)])
def test_invalid_resource_limit_rejected(limit, exception):
    with pytest.raises(exception):
        frontier.weighted_matching_frontier([], max_states=limit)


def test_limit_rejects_before_inserting_extra_state():
    with pytest.raises(frontier.FrontierStateLimitError) as captured:
        frontier.weighted_matching_frontier([[1, 2], [3, 4]], max_states=2)
    error = captured.value
    assert error.limit == 2 and error.attempted_states == 3
    assert error.row_index == 0 and error.processed_rows == 0
    assert frontier.weighted_matching_frontier([[1, 2], [3, 4]], max_states=3)["coefficients"] == (1, 10, 10)


def test_cli_complete_and_explicit_resource_failure(tmp_path, capsys):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps([[1, 2], [3, 4]]))
    assert frontier.main([str(path), "--row-order", "1,0", "--max-states", "3"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["coefficients"] == [1, 10, 10] and output["status"] == "COMPLETE"
    assert frontier.main([str(path), "--max-states", "2"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "RESOURCE_LIMIT" and output["exact"] is False
    assert "coefficients" not in output
