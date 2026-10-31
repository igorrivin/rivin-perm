"""Source selector contracts, exact combinatorics, and bounded fallbacks.

Fixtures are synthetic and do not read the dispatch study's holdout corpus.
The subset oracle retains every occupied column until the end; it shares no
frontier forgetting, row ordering, modular arithmetic, or CRT implementation.
"""
from functools import lru_cache
import itertools
from pathlib import Path

import pytest


SOURCE = (Path(__file__).resolve().parents[1]
          / "notes/benchmarks/weighted_rook_dispatch.py")
if not SOURCE.is_file():
    pytest.skip("checkout-only weighted-rook selector", allow_module_level=True)

from notes.benchmarks import weighted_rook_dispatch as dispatch


@lru_cache(maxsize=256)
def matching_definition(matrix):
    n = len(matrix)
    states = {0: 1}
    for row in matrix:
        following = dict(states)
        for occupied, count in states.items():
            for column, weight in enumerate(row):
                bit = 1 << column
                if weight and not occupied & bit:
                    following[occupied | bit] = (following.get(occupied | bit, 0)
                                                 + count * weight)
        states = following
    coefficients = [0] * (n + 1)
    for occupied, count in states.items():
        coefficients[bin(occupied).count("1")] += count
    return tuple(coefficients)


def path_matrix(n=8, weight=1):
    return tuple(tuple(weight * (1 + (i + j) % 7)
                       if j in (i, i + 1) else 0 for j in range(n))
                 for i in range(n))


def native_matrix(n=16):
    return tuple(tuple(1 + (i + j) % 7
                       if j in (i, (i + 1) % n, (i + 3) % n) else 0
                       for j in range(n)) for i in range(n))


@pytest.fixture
def routing(monkeypatch):
    """Make routing tests portable without pretending to execute native code."""
    monkeypatch.setattr(dispatch.public, "HAS_POLYNOMIAL_EXT", True)
    calls = []

    def original_crt(matrix, backend, threads):
        matrix = tuple(tuple(row) for row in matrix)
        calls.append((matrix, backend, threads))
        return matching_definition(matrix)

    monkeypatch.setattr(dispatch.public, "_weighted_rook_polynomial_crt_validated", original_crt)
    return calls


def test_all_binary_three_by_three_inputs_match_definition_and_public():
    selector = dispatch.WeightedRookDispatcher()
    for entries in itertools.product((0, 1), repeat=9):
        matrix = tuple(tuple(entries[3*i:3*i+3]) for i in range(3))
        expected = matching_definition(matrix)
        assert selector.weighted_rook_polynomial(matrix) == expected
        assert dispatch.public.weighted_rook_polynomial(matrix, parallel=False) == expected


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_all_small_weights_and_large_disconnected_factors(backend):
    selector = dispatch.WeightedRookDispatcher()
    for entries in itertools.product((0, 2, 7), repeat=4):
        matrix = (entries[:2], entries[2:])
        assert selector.weighted_rook_polynomial(matrix, backend=backend) == matching_definition(matrix)
    huge = 10**90 + 39
    matrix = ((huge, 2, 0, 0, 0), (3, huge + 1, 0, 0, 0),
              (0, 0, 7, 11, 13), (0, 0, 17, 19, 23), (0, 0, 0, 0, 0))
    assert selector.weighted_rook_polynomial(matrix, backend=backend) == matching_definition(matrix)


@pytest.mark.parametrize("matrix", [(), ((0,),), ((19,),), ((0, 0), (0, 0))])
def test_trivial_full_cardinality_vectors(matrix):
    result = dispatch.WeightedRookDispatcher().evaluate(matrix)
    assert result["coefficients"] == matching_definition(matrix)
    assert result["status"] == "COMPLETE" and result["exact"] is True
    assert result["frontier_record"] is None


@pytest.mark.parametrize("matrix,error", [
    (((0, 0), (0, -1)), ValueError), (((0, 0), (0, 1.0)), TypeError),
    (((0, 0), (0, "1")), TypeError), (((0, 0), (0,)), ValueError),
    (((0, 0, 0), (0, 0, 0)), ValueError),
    (tuple((0,) * 64 for _ in range(64)), ValueError),
])
def test_invalid_matrix_rejected_before_any_shortcut(matrix, error):
    with pytest.raises(error):
        dispatch.WeightedRookDispatcher().evaluate(matrix)


@pytest.mark.parametrize("kwargs,error", [
    ({"parallel": 1}, TypeError), ({"num_threads": True}, TypeError),
    ({"num_threads": -1}, ValueError), ({"num_threads": 2**31}, ValueError),
    ({"num_threads": 1.0}, TypeError), ({"backend": 1}, TypeError),
    ({"backend": "unknown"}, ValueError),
])
@pytest.mark.parametrize("matrix", [(), ((0, 0), (0, 0)), ((2, 0), (0, 3))])
def test_invalid_arguments_rejected_even_on_trivial_inputs(matrix, kwargs, error):
    with pytest.raises(error):
        dispatch.WeightedRookDispatcher().evaluate(matrix, **kwargs)


def test_declared_shape_and_generator_contracts_match_public():
    class WrongShape(list):
        shape = (2, 3)

    class WrongOrder(list):
        shape = (3, 3)

    for value in (WrongShape([[1, 0], [0, 1]]), WrongOrder([[1, 0], [0, 1]])):
        with pytest.raises(ValueError):
            dispatch.WeightedRookDispatcher().evaluate(value)
    for constructor in (lambda: (row for row in ((1, 0), (0, 1))),
                        lambda: [(x for x in (1, 0)), (x for x in (0, 1))]):
        with pytest.raises(TypeError):
            dispatch.WeightedRookDispatcher().evaluate(constructor())


def test_index_values_booleans_and_sized_one_shot_sequences(routing):
    class Indexed:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    class Once:
        def __init__(self, values):
            self.values, self.iterations = tuple(values), 0

        def __len__(self):
            return len(self.values)

        def __iter__(self):
            self.iterations += 1
            assert self.iterations == 1, "caller input must be snapshotted once"
            return iter(self.values)

    original = path_matrix()
    rows = [Once(Indexed(x) if x != 1 else True for x in row) for row in original]
    matrix = Once(rows)
    actual = dispatch.WeightedRookDispatcher().evaluate(matrix, num_threads=Indexed(1))
    assert actual["coefficients"] == matching_definition(original)
    assert matrix.iterations == 1 and all(row.iterations == 1 for row in rows)


def test_python_selection_matches_permutations_transpose_and_scaling(routing):
    matrix = path_matrix()
    row_order, column_order = (6, 1, 3, 7, 0, 4, 2, 5), (2, 7, 0, 5, 1, 4, 6, 3)
    transformed = tuple(tuple(matrix[i][j] for j in column_order) for i in row_order)
    transposed = tuple(zip(*matrix))
    scaled = tuple(tuple(13*x for x in row) for row in matrix)
    expected = matching_definition(matrix)
    selector = dispatch.WeightedRookDispatcher()
    for candidate in (matrix, transformed, transposed, scaled):
        result = selector.evaluate(candidate)
        assert result["decision"]["method"] == "python"
        assert result["coefficients"] == matching_definition(candidate)
    assert matching_definition(transformed) == matching_definition(transposed) == expected
    assert matching_definition(scaled) == tuple(value * 13**degree for degree, value in enumerate(expected))
    assert routing == []


def test_frontier_ordering_happens_once(routing, monkeypatch):
    calls = []
    original = dispatch.python_frontier._greedy_order

    def once(*args):
        calls.append(args)
        assert len(calls) == 1, "frontier must reuse the selected explicit order"
        return original(*args)

    monkeypatch.setattr(dispatch.python_frontier, "_greedy_order", once)
    result = dispatch.WeightedRookDispatcher().evaluate(path_matrix())
    assert len(calls) == 1
    assert result["frontier_record"]["ordering"] == "explicit"
    assert result["frontier_record"]["row_order"] == result["decision"]["row_order"]


@pytest.mark.parametrize("kwargs,expected_backend,expected_threads", [
    ({"backend": "python"}, "python", 1), ({"backend": "native"}, "native", 1),
    ({"parallel": True, "num_threads": 0}, "native", 0),
    ({"parallel": True, "num_threads": 3}, "native", 3),
])
def test_explicit_backend_and_unmeasured_threads_avoid_planning(routing, monkeypatch, kwargs,
                                                              expected_backend, expected_threads):
    monkeypatch.setattr(dispatch, "_frontier_plan", lambda *args: pytest.fail("planning forbidden"))
    result = dispatch.WeightedRookDispatcher().evaluate(path_matrix(), **kwargs)
    assert result["decision"]["method"] == "public"
    assert routing == [(path_matrix(), expected_backend, expected_threads)]


def test_parallel_false_overrides_threads_and_backend_is_case_insensitive(routing):
    result = dispatch.WeightedRookDispatcher().evaluate(path_matrix(), backend="AuTo",
                                                        parallel=False, num_threads=7)
    assert result["decision"]["method"] == "python"
    assert routing == []


@pytest.mark.parametrize("kind", ["dense", "small", "factorable", "heavy"])
def test_cheap_guards_avoid_greedy_work(routing, monkeypatch, kind):
    matrices = {
        "dense": tuple((1,) * 8 for _ in range(8)), "small": path_matrix(7),
        "factorable": tuple(tuple(2 if i == j else 0 for j in range(8)) for i in range(8)),
        "heavy": path_matrix(weight=2**20),
    }
    monkeypatch.setattr(dispatch.python_frontier, "_greedy_order", lambda *args: pytest.fail("greedy forbidden"))
    result = dispatch.WeightedRookDispatcher().evaluate(matrices[kind])
    assert result["decision"]["method"] == "public"
    assert result["coefficients"] == matching_definition(matrices[kind])


def test_isolated_endpoint_preserves_original_full_component_crt(routing, monkeypatch):
    matrix = tuple(tuple(1 + i + j if i < 7 else 0 for j in range(8)) for i in range(8))
    monkeypatch.setattr(dispatch, "_frontier_plan", lambda *args: pytest.fail("planning forbidden"))
    result = dispatch.WeightedRookDispatcher().evaluate(matrix)
    assert result["decision"]["reason"] == "isolated_endpoints"
    assert result["coefficients"] == matching_definition(matrix)
    assert result["coefficients"][-1] == 0
    assert routing == [(matrix, "native", 1)]


def test_decomposed_large_rectangle_is_padded_only_for_original_crt(routing, monkeypatch):
    # A connected K_(4,5), a disjoint K_(3,1), and one isolated column.
    matrix = tuple(tuple(2 + i + j if i < 4 and j < 5 else
                         13 if i >= 4 and j == 5 else 0 for j in range(7))
                   for i in range(7))
    monkeypatch.setattr(dispatch, "_frontier_plan", lambda *args: pytest.fail("planning forbidden"))
    result = dispatch.WeightedRookDispatcher().evaluate(matrix)
    assert result["decision"]["reason"] == "factorable_support"
    assert result["coefficients"] == matching_definition(matrix)
    assert len(routing) == 1 and len(routing[0][0]) == 5
    assert routing[0][0][-1] == (0,) * 5


@pytest.mark.parametrize("n,method", [(7, "public"), (8, "python"),
                                       (18, "python"), (19, "public")])
def test_order_policy_boundaries(n, method):
    assert dispatch._frontier_plan(path_matrix(n), False)["method"] == method


@pytest.mark.parametrize("weight,method", [(1023, "python"), (1024, "public")])
def test_weight_bit_policy_boundaries(weight, method):
    matrix = tuple(tuple(weight if value else 0 for value in row) for row in path_matrix())
    assert dispatch._frontier_plan(matrix, False)["method"] == method


def test_native_availability_does_not_relax_python_row_degree_limit():
    matrix = tuple(tuple(1 if i <= j <= i + 3 else 0 for j in range(16)) for i in range(16))
    plan = dispatch._frontier_plan(matrix, True)
    assert plan["maximum_row_degree"] == 4 and plan["peak_cut_width"] <= 4
    assert plan["method"] == "public"


def test_absent_native_library_preserves_exact_public_fallback(routing, monkeypatch):
    monkeypatch.setattr(dispatch.native_frontier, "FrontierBaseline",
                        lambda *args: pytest.fail("implicit loading forbidden"))
    result = dispatch.WeightedRookDispatcher().evaluate(native_matrix())
    assert result["decision"]["method"] == "public"
    assert result["coefficients"] == matching_definition(native_matrix())
    assert routing == [(native_matrix(), "native", 1)]


def test_missing_public_native_backend_retains_portable_and_explicit_contract(monkeypatch):
    monkeypatch.setattr(dispatch.public, "HAS_POLYNOMIAL_EXT", False)
    selector = dispatch.WeightedRookDispatcher()
    matrix = ((2, 3), (5, 7))
    assert selector.weighted_rook_polynomial(matrix) == matching_definition(matrix)
    for candidate in ((), ((0,),), matrix):
        with pytest.raises(RuntimeError, match="unavailable"):
            selector.evaluate(candidate, backend="native")


def test_constructor_loads_only_explicit_library_and_never_compiles(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(dispatch.native_frontier, "build_frontier_baseline",
                        lambda *args: pytest.fail("implicit compilation forbidden"))

    def loader(path):
        calls.append(path)
        return sentinel

    monkeypatch.setattr(dispatch.native_frontier, "FrontierBaseline", loader)
    assert dispatch.WeightedRookDispatcher().native is None and calls == []
    assert dispatch.WeightedRookDispatcher("/explicit/library.so").native is sentinel
    assert calls == ["/explicit/library.so"]


def test_constructor_provenance_failures_propagate(monkeypatch):
    def invalid(path):
        raise ValueError("source identity mismatch")

    monkeypatch.setattr(dispatch.native_frontier, "FrontierBaseline", invalid)
    with pytest.raises(ValueError, match="identity mismatch"):
        dispatch.WeightedRookDispatcher("/explicit/library.so")


@pytest.mark.parametrize("parameter", ["python_max_states", "native_max_states", "native_max_workspace_bytes"])
@pytest.mark.parametrize("value,error", [(True, TypeError), (1.0, TypeError), (0, ValueError), (-1, ValueError)])
def test_constructor_resource_limits_are_positive_integers(parameter, value, error):
    with pytest.raises(error):
        dispatch.WeightedRookDispatcher(**{parameter: value})


@pytest.mark.parametrize("parameter", ["native_max_states", "native_max_workspace_bytes"])
def test_native_limits_reject_unsigned_overflow_before_loading(parameter, monkeypatch):
    monkeypatch.setattr(dispatch.native_frontier, "FrontierBaseline",
                        lambda *args: pytest.fail("invalid caps must fail before loading"))
    with pytest.raises(ValueError, match="2\\^64"):
        dispatch.WeightedRookDispatcher("/explicit/library.so", **{parameter: 2**64})


def test_actual_python_state_limit_falls_back_once(routing):
    selector = dispatch.WeightedRookDispatcher(python_max_states=1)
    result = selector.evaluate(path_matrix())
    assert result["coefficients"] == matching_definition(path_matrix())
    assert result["resource_fallback"]["attempted_method"] == "python"
    assert result["resource_fallback"]["kind"] == "FrontierStateLimitError"
    assert result["decision"]["method"] == "public"
    assert len(routing) == 1 and result["frontier_record"] is None


def test_native_routing_passes_explicit_order_and_caps(routing, monkeypatch):
    calls = []

    class Native:
        def evaluate(self, matrix, **kwargs):
            calls.append((matrix, kwargs))
            return {"status": "COMPLETE", "exact": True,
                    "coefficients": matching_definition(matrix)}

    monkeypatch.setattr(dispatch.native_frontier, "FrontierBaseline", lambda path: Native())
    selector = dispatch.WeightedRookDispatcher("/explicit/library.so")
    result = selector.evaluate(native_matrix())
    assert result["decision"]["method"] == "native"
    assert result["decision"]["peak_cut_width"] == 6
    assert result["coefficients"] == matching_definition(native_matrix())
    assert calls == [(native_matrix(), {"row_order": result["decision"]["row_order"],
                                       "max_states": selector.native_max_states,
                                       "max_workspace_bytes": selector.native_max_workspace_bytes})]
    assert routing == []


@pytest.mark.parametrize("kind", ["state", "memory", "width"])
def test_native_known_resource_failures_fall_back_once(routing, monkeypatch, kind):
    stats = {"attempted_states": 4097, "processed_rows": 3, "failure_row": 3}
    errors = {
        "state": dispatch.native_frontier.FrontierStateLimitError(4096, stats),
        "memory": dispatch.native_frontier.FrontierMemoryLimitError(1, stats),
        "width": dispatch.native_frontier.FrontierWidthLimitError(65, 3),
    }

    class Native:
        def evaluate(self, matrix, **kwargs):
            raise errors[kind]

    monkeypatch.setattr(dispatch.native_frontier, "FrontierBaseline", lambda path: Native())
    result = dispatch.WeightedRookDispatcher("/explicit/library.so").evaluate(native_matrix())
    assert result["coefficients"] == matching_definition(native_matrix())
    assert result["resource_fallback"]["attempted_method"] == "native"
    assert result["resource_fallback"]["kind"] == type(errors[kind]).__name__
    assert len(routing) == 1 and result["frontier_record"] is None


@pytest.mark.parametrize("error", [RuntimeError("arithmetic corruption"), ValueError("bad provenance")])
def test_unexpected_frontier_failures_are_not_hidden(routing, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(dispatch.python_frontier, "weighted_matching_frontier", fail)
    with pytest.raises(type(error), match=str(error)):
        dispatch.WeightedRookDispatcher().evaluate(path_matrix())
    assert routing == []


@pytest.mark.parametrize("change", [
    {"status": "RESOURCE_LIMIT"}, {"exact": False}, {"coefficients": (1,)},
    {"coefficients": (0,) * 9}, {"coefficients": (1,) + (-1,) * 8},
    {"coefficients": (True,) + (0,) * 8}, {"coefficients": (1,) + (0.0,) * 8},
])
def test_corrupt_or_incomplete_frontier_records_raise(routing, monkeypatch, change):
    report = {"status": "COMPLETE", "exact": True, "coefficients": matching_definition(path_matrix())}
    report.update(change)
    monkeypatch.setattr(dispatch.python_frontier, "weighted_matching_frontier", lambda *args, **kwargs: report)
    with pytest.raises(RuntimeError):
        dispatch.WeightedRookDispatcher().evaluate(path_matrix())
    assert routing == []
