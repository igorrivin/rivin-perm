"""Compiled weighted frontier checked against independent matching definitions."""
import ctypes
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil

import pytest


SOURCE = (Path(__file__).resolve().parents[1]
          / "notes/benchmarks/weighted_matching_frontier_native.py")
if not SOURCE.is_file():
    pytest.skip("checkout-only compiled weighted frontier baseline", allow_module_level=True)
SPEC = importlib.util.spec_from_file_location("weighted_frontier_native_test", SOURCE)
frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier)


def edge_set_coefficients(matrix):
    """Choose complete edge sets, independently of any row-frontier recurrence."""
    height = len(matrix)
    width = len(matrix[0]) if matrix else 0
    edges = [(i, j, weight) for i, row in enumerate(matrix)
             for j, weight in enumerate(row) if weight]
    result = [1] + [0] * min(height, width)
    for degree in range(1, len(result)):
        for selected in itertools.combinations(edges, degree):
            if (len({i for i, _, _ in selected}) == degree
                    and len({j for _, j, _ in selected}) == degree):
                result[degree] += math.prod(weight for _, _, weight in selected)
    return tuple(result)


def check_exact_report(result, matrix, expected):
    shape = (len(matrix), len(matrix[0]) if matrix else 0)
    assert result["shape"] == shape
    assert result["coefficients"] == expected
    assert len(result["coefficients"]) == min(shape) + 1
    assert result["status"] == "COMPLETE" and result["exact"] is True
    assert result["modulus"] == math.prod(result["primes"])
    assert result["modulus"] > max(result["bounds"])
    assert len(result["bounds"]) == len(expected)
    assert all(0 <= value <= bound for value, bound in zip(expected, result["bounds"]))
    assert len(result["primes"]) == len(result["residue_rows"])
    for prime, residues in zip(result["primes"], result["residue_rows"]):
        assert tuple(residues) == tuple(value % prime for value in expected)
    assert 1 <= result["peak_states"] <= result["max_states"]
    assert 0 <= result["peak_frontier"] <= result["peak_row_frontier"] <= 64
    assert 0 <= result["peak_workspace_bytes"] <= result["max_workspace_bytes"]


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    if shutil.which("g++") is None:
        if os.environ.get("PERM_REQUIRE_NATIVE_TESTS") == "1":
            pytest.fail("required compiled frontier tests need g++")
        pytest.skip("g++ is required for compiled research frontier baseline tests")
    return frontier.build_frontier_baseline(tmp_path_factory.mktemp("weighted-frontier-build"))


@pytest.fixture(scope="module")
def baseline(build):
    return frontier.FrontierBaseline(build["library"])


def test_build_records_source_binary_compiler_command_and_logs(build):
    assert Path(build["library"]).is_absolute()
    for key, checksum in [("library", "library_sha256"), ("source", "source_sha256"),
                          ("compiler", "compiler_sha256"), ("stdout_log", "stdout_sha256"),
                          ("stderr_log", "stderr_sha256")]:
        assert hashlib.sha256(Path(build[key]).read_bytes()).hexdigest() == build[checksum]
    assert build["command"][-2:] == ["-o", build["library"]]
    assert hashlib.sha256(Path(build["source_snapshot"]).read_bytes()).hexdigest() == build["source_sha256"]
    assert json.loads((Path(build["library"]).parent / "build.json").read_text()) == build
    with pytest.raises(FileExistsError):
        frontier.build_frontier_baseline(Path(build["library"]).parent)


@pytest.mark.parametrize("field, corrupted", [
    ("library_sha256", "0" * 64), ("source_hashes", {"unrelated_source": "0" * 64}),
    ("abi_version", 2), ("source_sha256", "0" * 64), ("wrapper_sha256", "0" * 64),
])
def test_loader_rejects_mismatched_library_or_source_identity(build, tmp_path, field, corrupted):
    library = tmp_path / Path(build["library"]).name
    shutil.copy2(build["library"], library)
    manifest = dict(build)
    manifest[field] = corrupted
    (tmp_path / "build.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        frontier.FrontierBaseline(library)


def test_loader_rejects_changed_library_bytes(build, tmp_path):
    library = tmp_path / Path(build["library"]).name
    library.write_bytes(Path(build["library"]).read_bytes() + b"changed")
    (tmp_path / "build.json").write_text(json.dumps(build))
    with pytest.raises(ValueError, match="identity mismatch"):
        frontier.FrontierBaseline(library)


@pytest.mark.parametrize("artifact", ["source_snapshot", "stdout_log", "stderr_log"])
def test_loader_rejects_changed_retained_build_artifacts(build, tmp_path, artifact):
    manifest = dict(build)
    # Preserve all original fixture assets. This copy changes only one saved
    # artifact, with library and current-source identity still valid.
    for key in ("library", "source_snapshot", "stdout_log", "stderr_log"):
        copied = tmp_path / Path(build[key]).name
        shutil.copy2(build[key], copied)
        manifest[key] = str(copied)
    changed = Path(manifest[artifact])
    changed.write_bytes(changed.read_bytes() + b"changed")
    (tmp_path / "build.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        frontier.FrontierBaseline(manifest["library"])


def test_loader_rejects_missing_retained_build_artifact(build, tmp_path):
    library = tmp_path / Path(build["library"]).name
    shutil.copy2(build["library"], library)
    manifest = dict(build)
    manifest["source_snapshot"] = str(tmp_path / "missing.cpp")
    (tmp_path / "build.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid frontier build provenance"):
        frontier.FrontierBaseline(library)


@pytest.mark.parametrize("stage, failure", [
    ("version", "timeout"), ("compile", "timeout"),
    ("version", "nonzero"), ("compile", "nonzero"),
])
def test_build_failures_retain_logs_and_never_publish_success(tmp_path, monkeypatch, stage, failure):
    calls = []
    monkeypatch.setattr(frontier.shutil, "which", lambda _: "/mock/g++")

    def fake_run(command, **kwargs):
        calls.append(command)
        current = "version" if "--version" in command else "compile"
        if current != stage:
            return frontier.subprocess.CompletedProcess(command, 0, "mock compiler\n", "")
        if failure == "timeout":
            raise frontier.subprocess.TimeoutExpired(command, kwargs["timeout"],
                                                     output=b"partial stdout\xff\n",
                                                     stderr=b"partial stderr\n")
        return frontier.subprocess.CompletedProcess(command, 17, "failed stdout\n", "failed stderr\n")

    monkeypatch.setattr(frontier.subprocess, "run", fake_run)
    expected = "compiler version query" if stage == "version" else "frontier compilation"
    with pytest.raises(RuntimeError, match=expected):
        frontier.build_frontier_baseline(tmp_path)
    assert len(calls) == (1 if stage == "version" else 2)
    assert (tmp_path / frontier.CPP_SOURCE.name).is_file()
    assert not (tmp_path / "build.json").exists()
    assert not (tmp_path / "weighted_matching_frontier_native.so").exists()
    stdout = (tmp_path / "compile.stdout.log").read_text()
    stderr = (tmp_path / "compile.stderr.log").read_text()
    assert stdout == ("partial stdout\ufffd\n" if failure == "timeout" else "failed stdout\n")
    assert stderr == ("partial stderr\n" if failure == "timeout" else "failed stderr\n")


def test_cli_build_timeout_reports_error_with_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(frontier.shutil, "which", lambda _: "/mock/g++")

    def timeout(command, **kwargs):
        raise frontier.subprocess.TimeoutExpired(command, kwargs["timeout"], stderr=b"version timeout\n")

    monkeypatch.setattr(frontier.subprocess, "run", timeout)
    with pytest.raises(SystemExit) as captured:
        frontier.main(["build", str(tmp_path)])
    assert captured.value.code == 2
    messages = capsys.readouterr()
    assert messages.out == ""
    assert "compiler version query timed out" in messages.err
    assert "Traceback" not in messages.err
    assert (tmp_path / "compile.stderr.log").read_text() == "version timeout\n"


@pytest.mark.parametrize("matrix", [
    [], [[]], [[], []], [[0]], [[7]], [[0, 0, 0], [0, 0, 0]],
    [[2, 3, 5]], [[2], [3], [5]], [[2, 0], [0, 3]],
    [[1, 2, 0], [0, 3, 4]], [[7, 0], [11, 0]],
])
def test_empty_rectangular_and_partial_matchings(baseline, matrix):
    original = [row[:] for row in matrix]
    actual = baseline.evaluate(matrix)
    check_exact_report(actual, matrix, edge_set_coefficients(matrix))
    assert matrix == original


def test_all_81_weighted_two_by_two_matrices(baseline):
    for entries in itertools.product(range(3), repeat=4):
        matrix = [list(entries[:2]), list(entries[2:])]
        check_exact_report(baseline.evaluate(matrix), matrix, edge_set_coefficients(matrix))


def test_all_512_binary_three_by_three_matrices(baseline):
    for entries in itertools.product(range(2), repeat=9):
        matrix = [list(entries[start:start + 3]) for start in (0, 3, 6)]
        check_exact_report(baseline.evaluate(matrix), matrix, edge_set_coefficients(matrix))


def test_seeded_rectangles_orders_transposes_and_isolates(baseline):
    rng = random.Random(2026090807)
    for height, width in [(1, 5), (5, 1), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]:
        for _ in range(4):
            matrix = [[rng.randrange(1, 8) if rng.random() < 0.55 else 0
                       for _ in range(width)] for _ in range(height)]
            order = list(range(height))
            rng.shuffle(order)
            expected = edge_set_coefficients(matrix)
            check_exact_report(baseline.evaluate(matrix), matrix, expected)
            explicit = baseline.evaluate(matrix, row_order=order)
            check_exact_report(explicit, matrix, expected)
            assert explicit["row_order"] == tuple(order) and explicit["ordering"] == "explicit"
            transposed = [list(row) for row in zip(*matrix)]
            check_exact_report(baseline.evaluate(transposed), transposed, expected)


def test_medium_rectangles_match_separate_python_counting_recurrence(baseline):
    from notes.benchmarks.weighted_matching_frontier import weighted_matching_frontier

    rng = random.Random(2026090811)
    for height, width in [(5, 8), (8, 5), (7, 8), (8, 8)]:
        for banded in (False, True):
            matrix = [[rng.randrange(7) if not banded or abs(i - j) <= 1 else 0
                       for j in range(width)] for i in range(height)]
            order = list(range(height))
            rng.shuffle(order)
            for row_order in (None, order):
                expected = weighted_matching_frontier(matrix, row_order=row_order)
                actual = baseline.evaluate(matrix, row_order=row_order)
                check_exact_report(actual, matrix, expected["coefficients"])
                assert actual["peak_states"] == expected["peak_states"]
                assert actual["row_order"] == expected["row_order"]


def test_deterministic_greedy_order_closes_interleaved_components(baseline):
    matrix = [[1, 2, 0, 0], [0, 0, 3, 4], [5, 6, 0, 0], [0, 0, 7, 8]]
    sequential = baseline.evaluate(matrix, row_order=range(4))
    greedy = baseline.evaluate(matrix)
    again = baseline.evaluate(matrix)
    check_exact_report(greedy, matrix, edge_set_coefficients(matrix))
    assert greedy["coefficients"] == sequential["coefficients"]
    assert greedy["ordering"] == "greedy"
    assert greedy["row_order"] == again["row_order"]
    assert greedy["peak_states"] < sequential["peak_states"]
    assert greedy["peak_frontier"] == 2 < sequential["peak_frontier"] == 4


def test_expired_unmatched_columns_are_allowed_and_high_slots_are_reused(baseline):
    # One hundred independent edges require only one simultaneous mask slot.
    n = 100
    matrix = [[2 if i == j else 0 for j in range(n)] for i in range(n)]
    expected = tuple(math.comb(n, k) * 2**k for k in range(n + 1))
    actual = baseline.evaluate(matrix, row_order=range(n), max_states=1)
    check_exact_report(actual, matrix, expected)
    assert actual["peak_states"] == 1
    assert actual["peak_frontier"] == 0 and actual["peak_row_frontier"] == 1


def test_banded_matrix_over_64_global_columns_matches_path_formula(baseline):
    n = 70
    matrix = [[int(j == i or j == i + 1) for j in range(n)] for i in range(n)]
    # This bipartite graph is a path on 2n vertices: choosing k nonadjacent
    # path edges has binomial(2n-k, k) possibilities.
    expected = tuple(math.comb(2 * n - k, k) for k in range(n + 1))
    actual = baseline.evaluate(matrix, row_order=range(n), max_states=2)
    check_exact_report(actual, matrix, expected)
    assert actual["peak_frontier"] == 1 and actual["peak_row_frontier"] == 2


def test_bit_63_is_usable_without_sign_or_shift_errors(baseline):
    matrix = [[1] * 64]
    actual = baseline.evaluate(matrix)
    check_exact_report(actual, matrix, (1, 64))
    assert actual["peak_row_frontier"] == 64
    # Preserve bit 63 across a cut, then test it when selecting a second row.
    matrix = [[1] * 64, [1] * 64]
    actual = baseline.evaluate(matrix, max_states=65)
    check_exact_report(actual, matrix, (1, 128, 64 * 63))
    assert actual["peak_states"] == 65
    assert actual["peak_frontier"] == actual["peak_row_frontier"] == 64


def test_arbitrary_weights_booleans_index_objects_and_generators(baseline):
    huge = 10**200

    class Integer:
        def __index__(self):
            return huge

    matrix = [[Integer(), True], [False, huge + 1], [0, 5]]
    normalized = [[huge, 1], [0, huge + 1], [0, 5]]
    actual = baseline.evaluate((iter(row) for row in matrix), row_order=iter([2, 0, 1]))
    check_exact_report(actual, normalized, edge_set_coefficients(normalized))
    assert actual["row_order"] == (2, 0, 1)
    assert len(actual["primes"]) > 20


def test_zero_modular_weight_does_not_change_integer_support(baseline):
    prime = 2147483647
    matrix = [[prime, 2], [3, prime * 5]]
    actual = baseline.evaluate(matrix)
    check_exact_report(actual, matrix, edge_set_coefficients(matrix))
    assert actual["primes"][0] == prime
    assert tuple(actual["residue_rows"][0]) == (1, 5, 6)
    assert actual["peak_frontier"] == 2


def test_isolates_and_rectangular_components_retain_trailing_zero_coefficients(baseline):
    matrix = [[2, 3, 0, 0, 0, 0], [0, 5, 7, 0, 0, 0],
              [0, 0, 0, 11, 13, 0], [0, 0, 0, 0, 0, 0]]
    actual = baseline.evaluate(matrix)
    check_exact_report(actual, matrix, edge_set_coefficients(matrix))
    assert len(actual["coefficients"]) == 5 and actual["coefficients"][-1] == 0


def test_state_limit_is_enforced_per_layer(baseline):
    matrix = [[1, 2], [3, 4]]
    with pytest.raises(frontier.FrontierStateLimitError):
        baseline.evaluate(matrix, max_states=2)
    actual = baseline.evaluate(matrix, max_states=3)
    check_exact_report(actual, matrix, (1, 10, 10))
    assert actual["peak_states"] == 3


@pytest.mark.parametrize("matrix", [
    [[1, 2], [3, 4]],
    [[1] * 8 for _ in range(8)],
    [[int(i == j) for j in range(128)] for i in range(128)],
])
def test_workspace_limit_accounts_for_layers_growth_and_coefficient_degree(baseline, matrix):
    # Derive the boundary from a complete execution rather than depending on
    # a particular hash-table layout. Dense n=8 traverses several growths;
    # the long diagonal has one state but large cardinality polynomials.
    generous = baseline.evaluate(matrix)
    required = generous["peak_workspace_bytes"]
    assert required > 1
    bounded = baseline.evaluate(matrix, max_workspace_bytes=required)
    assert bounded["coefficients"] == generous["coefficients"]
    assert bounded["peak_workspace_bytes"] == required
    with pytest.raises(frontier.FrontierMemoryLimitError):
        baseline.evaluate(matrix, max_workspace_bytes=required - 1)
    with pytest.raises(frontier.FrontierMemoryLimitError):
        baseline.evaluate(matrix, max_workspace_bytes=1)


def test_coefficient_storage_counts_toward_workspace_even_with_one_state(baseline):
    peaks = []
    for n in (4, 128):
        matrix = [[int(i == j) for j in range(n)] for i in range(n)]
        report = baseline.evaluate(matrix, max_states=1)
        assert report["peak_states"] == 1
        peaks.append(report["peak_workspace_bytes"])
    assert peaks[1] > peaks[0]


def test_simultaneous_frontier_above_native_mask_width_is_rejected(baseline):
    with pytest.raises(frontier.FrontierWidthLimitError):
        baseline.evaluate([[1] * 65])
    # No row or cut exceeds 64, but the isolated middle-row column needs a
    # sixty-fifth slot while the first row's supported columns remain live.
    matrix = [[1] * 64 + [0], [0] * 64 + [1], [1] * 64 + [0]]
    with pytest.raises(frontier.FrontierWidthLimitError):
        baseline.evaluate(matrix, row_order=range(3))


@pytest.mark.parametrize("matrix, exception", [
    ([[1], [2, 3]], ValueError), ([[-1]], ValueError), ([[1.0]], TypeError),
    ([["1"]], TypeError), ([[None]], TypeError), ([1, 2], TypeError),
    (None, TypeError), ("123", TypeError),
])
def test_invalid_matrix_is_rejected_before_native_call(baseline, matrix, exception, monkeypatch):
    def fail(*args):
        pytest.fail("invalid Python matrix reached the native kernel")
    monkeypatch.setattr(baseline, "_function", fail)
    with pytest.raises(exception):
        baseline.evaluate(matrix)


@pytest.mark.parametrize("order, exception", [
    ([0], ValueError), ([0, 0], ValueError), ([0, 2], ValueError),
    ([-1, 0], ValueError), ([0, 1.0], TypeError), ([False, 1], TypeError),
    (2, TypeError),
])
def test_invalid_row_order_is_rejected_before_native_call(baseline, order, exception, monkeypatch):
    def fail(*args):
        pytest.fail("invalid Python row order reached the native kernel")
    monkeypatch.setattr(baseline, "_function", fail)
    with pytest.raises(exception):
        baseline.evaluate([[1, 2], [3, 4]], row_order=order)


@pytest.mark.parametrize("parameter", ["max_states", "max_workspace_bytes"])
@pytest.mark.parametrize("value, exception", [
    (0, ValueError), (-1, ValueError), (True, TypeError), (2.5, TypeError),
    (None, TypeError), (2**65, ValueError),
])
def test_invalid_limits_are_rejected_before_native_call(baseline, parameter, value, exception, monkeypatch):
    def fail(*args):
        pytest.fail("invalid Python resource limit reached the native kernel")
    monkeypatch.setattr(baseline, "_function", fail)
    with pytest.raises(exception):
        baseline.evaluate([], **{parameter: value})


def test_unknown_native_status_does_not_return_an_exact_result(baseline, monkeypatch):
    monkeypatch.setattr(baseline, "_function", lambda *args: 999)
    with pytest.raises(RuntimeError, match="999"):
        baseline.evaluate([[1]])


@pytest.mark.parametrize("modulus", [3, 101, 2147483647, 2147483629, 2147450881, 2147450879])
def test_c_abi_canonical_arithmetic_including_zero_residue_edges(baseline, modulus):
    # Integer support is supplied separately from residues, which may be zero.
    # Exercise small and near-31-bit odd moduli without assuming primality.
    matrix = [[modulus - 1, 0], [modulus - 3, modulus - 2]]
    offsets = (ctypes.c_uint64 * 3)(0, 2, 4)
    bits = (ctypes.c_uint64 * 4)(1, 2, 1, 2)
    weights = (ctypes.c_uint32 * 4)(*(value for row in matrix for value in row))
    live_after = (ctypes.c_uint64 * 2)(3, 0)
    out = (ctypes.c_uint32 * 3)()
    stats = (ctypes.c_uint64 * 5)()
    status = baseline._function(2, 2, 4, offsets, bits, weights, live_after,
                                modulus, 1000, 1 << 20, out, stats)
    assert status == 0
    assert tuple(out) == tuple(value % modulus for value in edge_set_coefficients(matrix))
    assert 0 < stats[0] <= 1 << 20
    assert stats[1] == 3 and stats[2] == 2
    assert stats[3] == (1 << 64) - 1


def abi_arguments():
    """A valid two-row call with sentinel output, retaining ctypes owners."""
    return [
        2, 2, 4,
        (ctypes.c_uint64 * 3)(0, 2, 4),
        (ctypes.c_uint64 * 4)(1, 2, 1, 2),
        (ctypes.c_uint32 * 4)(2, 3, 5, 7),
        (ctypes.c_uint64 * 2)(3, 0),
        101, 1000, 1 << 20,
        (ctypes.c_uint32 * 3)(91, 92, 93),
        (ctypes.c_uint64 * 5)(),
    ]


@pytest.mark.parametrize("index, replacement", [
    (0, (1 << 64) - 1), (1, 3), (2, (1 << 64) - 1),
    (3, None), (4, None), (5, None), (6, None), (11, None),
    (3, (ctypes.c_uint64 * 3)(1, 2, 4)),
    (3, (ctypes.c_uint64 * 3)(0, 3, 2)),
    (3, (ctypes.c_uint64 * 3)(0, 2, 5)),
    (4, (ctypes.c_uint64 * 4)(0, 2, 1, 2)),
    (4, (ctypes.c_uint64 * 4)(3, 2, 1, 2)),
    (4, (ctypes.c_uint64 * 4)(1, 1, 1, 2)),
    (5, (ctypes.c_uint32 * 4)(101, 3, 5, 7)),
    (6, (ctypes.c_uint64 * 2)(7, 0)),
    (6, (ctypes.c_uint64 * 2)(3, 1)),
    (7, 0), (7, 2), (7, 100), (7, 1 << 31),
    (8, 0), (9, 0),
])
def test_c_abi_rejects_malformed_inputs_without_publishing_output(baseline, index, replacement):
    args = abi_arguments()
    args[index] = replacement
    assert baseline._function(*args) == 1
    assert tuple(args[10]) == (91, 92, 93)


def test_c_abi_rejects_missing_output_pointer(baseline):
    args = abi_arguments()
    args[10] = None
    assert baseline._function(*args) == 1


@pytest.mark.parametrize("index, value, status", [(8, 2, 3), (9, 1, 2)])
def test_c_abi_resource_failure_leaves_output_untouched(baseline, index, value, status):
    args = abi_arguments()
    args[index] = value
    assert baseline._function(*args) == status
    assert tuple(args[10]) == (91, 92, 93)
    assert args[11][0] <= args[9]


def test_c_abi_does_not_require_modulus_to_exceed_polynomial_degree(baseline):
    n, modulus = 10, 3
    offsets = (ctypes.c_uint64 * (n + 1))(*range(n + 1))
    bits = (ctypes.c_uint64 * n)(*([1] * n))
    weights = (ctypes.c_uint32 * n)(*([1] * n))
    live_after = (ctypes.c_uint64 * n)()
    out = (ctypes.c_uint32 * (n + 1))()
    stats = (ctypes.c_uint64 * 5)()
    assert baseline._function(n, n, n, offsets, bits, weights, live_after,
                              modulus, 1, 1 << 20, out, stats) == 0
    assert tuple(out) == tuple(math.comb(n, k) % modulus for k in range(n + 1))


def test_noncanonical_native_residue_is_rejected(baseline, monkeypatch):
    def corrupt(rows, degree, edge_count, offsets, bits, weights, live_after,
                modulus, max_states, max_bytes, out, stats):
        out[0] = modulus
        stats[1] = 1
        stats[2] = rows
        stats[3] = (1 << 64) - 1
        stats[4] = 1
        return 0

    monkeypatch.setattr(baseline, "_function", corrupt)
    with pytest.raises(ValueError, match="noncanonical"):
        baseline.evaluate([[1]])


def test_cli_success_and_memory_failure_have_distinct_exactness(build, tmp_path, capsys):
    matrix_file = tmp_path / "matrix.json"
    matrix_file.write_text(json.dumps([[1, 2], [3, 4]]))
    arguments = ["evaluate", build["library"], str(matrix_file), "--row-order", "1,0"]
    assert frontier.main(arguments) == 0
    success = json.loads(capsys.readouterr().out)
    assert success["status"] == "COMPLETE" and success["exact"] is True
    assert success["coefficients"] == [1, 10, 10]
    assert success["row_order"] == [1, 0]
    assert frontier.main(arguments + ["--max-workspace-bytes", "1"]) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "RESOURCE_LIMIT" and failure["exact"] is False
    assert failure["limit"] == 1
    assert "coefficients" not in failure
    assert failure["details"]["peak_workspace_bytes"] <= 1
