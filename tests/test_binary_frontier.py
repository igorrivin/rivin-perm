"""Tests for the exact live-column frontier permanent solver."""

import itertools
import math

import numpy as np
import pytest

import perm_binary


def _permutation_oracle(matrix):
    return sum(
        math.prod(matrix[row][permutation[row]] for row in range(len(matrix)))
        for permutation in itertools.permutations(range(len(matrix)))
    )


def _cycle_adjacency(order):
    matrix = [[0] * order for _ in range(order)]
    for vertex in range(order):
        neighbor = (vertex + 1) % order
        matrix[vertex][neighbor] = 1
        matrix[neighbor][vertex] = 1
    return matrix


def _block_diagonal_ones(block_size, block_count):
    order = block_size * block_count
    return [
        [
            int(row // block_size == column // block_size)
            for column in range(order)
        ]
        for row in range(order)
    ]


def test_frontier_matches_definition_on_random_small_matrices():
    rng = np.random.default_rng(20260820)
    for order in range(8):
        trials = 10 if order < 7 else 4
        for _ in range(trials):
            matrix = rng.integers(0, 2, size=(order, order)).tolist()
            expected = _permutation_oracle(matrix)
            assert perm_binary.binary_permanent_frontier(matrix) == expected
            assert perm_binary.binary_permanent_frontier(
                matrix, row_order=list(reversed(range(order)))
            ) == expected


def test_frontier_empty_singleton_and_infeasible_conventions():
    assert perm_binary.binary_permanent_frontier([]) == 1
    assert perm_binary.binary_permanent_frontier([[0]]) == 0
    assert perm_binary.binary_permanent_frontier([[1]]) == 1

    hall_obstruction = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    ]
    assert perm_binary.binary_permanent_frontier(
        hall_obstruction, max_states=1
    ) == 0


def test_frontier_uses_arbitrary_size_python_integers():
    matrix = _block_diagonal_ones(10, 5)
    expected = math.factorial(10) ** 5
    result = perm_binary.binary_permanent_frontier(
        matrix, max_states=1_000
    )
    assert result == expected
    assert result.bit_length() > 64
    assert isinstance(result, int)


def test_frontier_state_cap_raises_typed_exception_immediately():
    matrix = [[1] * 8 for _ in range(8)]
    with pytest.raises(
        perm_binary.FrontierStateLimitExceeded,
        match="state limit 5 exceeded",
    ) as caught:
        perm_binary.binary_permanent_frontier(
            matrix, row_order=range(8), max_states=5
        )
    assert caught.value.max_states == 5
    assert caught.value.observed_states == 6
    assert caught.value.processed_rows == 1


@pytest.mark.parametrize(
    "max_states", [0, -1, 1.5, True, np.bool_(True), None]
)
def test_frontier_rejects_invalid_state_caps(max_states):
    error = ValueError if type(max_states) is int and max_states <= 0 else TypeError
    with pytest.raises(error, match="max_states"):
        perm_binary.binary_permanent_frontier(
            [[1]], max_states=max_states
        )


@pytest.mark.parametrize(
    "row_order",
    [
        [0, 1],
        [0, 1, 1],
        [0, 1, 3],
        [0, -1, 2],
        [0, True, 2],
        [0, np.bool_(True), 2],
        [0, 1.0, 2],
        7,
    ],
)
def test_frontier_rejects_invalid_row_orders(row_order):
    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises((TypeError, ValueError), match="row_order"):
        perm_binary.binary_permanent_frontier(
            matrix, row_order=row_order
        )


def test_frontier_accepts_integral_row_order_and_snapshots_its_length():
    identity = [[1, 0], [0, 1]]
    assert perm_binary.binary_permanent_frontier(
        identity, row_order=np.array([np.int64(1), np.int32(0)])
    ) == 1

    class ShortOrder:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([0])

    with pytest.raises(ValueError, match="length changed"):
        perm_binary.binary_permanent_frontier(
            identity, row_order=ShortOrder()
        )


def test_frontier_preserves_strict_support_and_hostile_matrix_semantics():
    weighted = [[-7, 0], [4, 9]]
    with pytest.raises(ValueError, match="0 or 1"):
        perm_binary.binary_permanent_frontier(weighted)
    assert perm_binary.binary_permanent_frontier(
        weighted, strict=False
    ) == 1

    class ShortRow:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([1])

    with pytest.raises(ValueError, match="row width changed"):
        perm_binary.binary_permanent_frontier(
            [ShortRow(), ShortRow()]
        )

    class ShortMatrix:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([[1, 0]])

    with pytest.raises(ValueError, match="row count changed"):
        perm_binary.binary_permanent_frontier(ShortMatrix())


def test_greedy_order_is_deterministic_and_small_on_an_odd_cycle():
    matrix = _cycle_adjacency(21)
    row_masks = perm_binary._row_masks(matrix)
    first = perm_binary._frontier_greedy_order(row_masks)
    second = perm_binary._frontier_greedy_order(row_masks)
    assert first == second
    assert sorted(first[0]) == list(range(21))
    assert first[1] <= 4
    assert perm_binary.binary_permanent_frontier(matrix) == 2


def test_auto_uses_frontier_for_eligible_sparse_block(monkeypatch):
    matrix = _cycle_adjacency(21)

    def fail_native(*args, **kwargs):
        raise AssertionError("eligible sparse auto input should use frontier")

    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", True)
    monkeypatch.setattr(perm_binary, "_native_exact_permanent", fail_native)
    assert perm_binary.binary_permanent(matrix, backend="auto") == 2


def test_auto_frontier_cap_falls_back_to_native(monkeypatch):
    matrix = _cycle_adjacency(21)
    calls = []

    def fake_native(argument, bound, threads):
        calls.append((argument, bound, threads))
        return 2

    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", True)
    monkeypatch.setattr(perm_binary, "_AUTO_FRONTIER_MAX_STATES", 1)
    monkeypatch.setattr(perm_binary, "_native_exact_permanent", fake_native)
    assert perm_binary.binary_permanent(matrix, backend="auto") == 2
    assert len(calls) == 1


def test_explicit_native_backend_bypasses_frontier(monkeypatch):
    matrix = _cycle_adjacency(21)
    calls = []

    def fail_plan(*args, **kwargs):
        raise AssertionError("explicit native must not plan a frontier")

    def fake_native(argument, bound, threads):
        calls.append((argument, bound, threads))
        return 2

    monkeypatch.setattr(perm_binary, "HAS_BINARY_EXT", True)
    monkeypatch.setattr(perm_binary, "_auto_frontier_plan", fail_plan)
    monkeypatch.setattr(perm_binary, "_native_exact_permanent", fake_native)
    assert perm_binary.binary_permanent(matrix, backend="native") == 2
    assert len(calls) == 1
