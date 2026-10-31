"""Independent correctness and evidence-integrity checks for public CPU suite."""
import copy
from argparse import Namespace
import itertools
import math
from pathlib import Path
import random
import shutil
import time

import pytest

from notes.benchmarks import polynomial_compare as benchmark
from notes.benchmarks import summarize_polynomial_comparison as summarizer


@pytest.fixture(scope="module")
def baselines(tmp_path_factory):
    if shutil.which("c++") is None:
        pytest.skip("C++ compiler unavailable")
    library, _ = benchmark.build_baselines(tmp_path_factory.mktemp("polynomial_baselines"))
    return benchmark.Baselines(library)


def permutation_coefficients(constant, linear):
    n = len(constant)
    result = [0] * (n + 1)
    # Enumerate permutations and binary choices of constant/linear factors.
    for columns in itertools.permutations(range(n)):
        for linear_rows in itertools.product((False, True), repeat=n):
            value = math.prod((linear if selected else constant)[row][column]
                              for row, (column, selected) in enumerate(zip(columns, linear_rows)))
            result[sum(linear_rows)] += value
    return result


def partial_matching_coefficients(matrix):
    n = len(matrix)
    result = [0] * (n + 1)
    for degree in range(n + 1):
        for rows in itertools.combinations(range(n), degree):
            for columns in itertools.permutations(range(n), degree):
                result[degree] += math.prod(matrix[row][column]
                                            for row, column in zip(rows, columns))
    return result


@pytest.mark.parametrize("prime", [3, 9, 2147483629, 2147483647])
def test_compiled_ryser_against_permutation_definition(baselines, prime):
    rng = random.Random(710 + prime)
    for n in range(6):
        case = {"n": n, "contract": "affine",
                "constant": [[rng.randrange(-2*prime, 2*prime) for _ in range(n)] for _ in range(n)],
                "linear": [[rng.randrange(-2*prime, 2*prime) for _ in range(n)] for _ in range(n)]}
        expected = [x % prime for x in permutation_coefficients(case["constant"], case["linear"])]
        assert baselines.evaluate("polynomial_ryser", case, prime)["coefficients"] == expected


@pytest.mark.parametrize("prime", [3, 9, 2147483629, 2147483647])
def test_compiled_rook_dp_against_partial_matchings(baselines, prime):
    rng = random.Random(821 + prime)
    for n in range(6):
        matrix = [[rng.randrange(prime*2) if rng.randrange(3) else 0
                   for _ in range(n)] for _ in range(n)]
        case = {"n": n, "contract": "rook", "linear": matrix}
        expected = [x % prime for x in partial_matching_coefficients(matrix)]
        assert baselines.evaluate("rook_dp", case, prime)["coefficients"] == expected


def test_interpolation_recovers_signed_high_degree_coefficients():
    rng = random.Random(1307)
    for prime in (7, 101, 2147483647):
        coefficients = [rng.randrange(-10**30, 10**30) for _ in range(6)]
        values = [sum(value * x**k for k, value in enumerate(coefficients)) % prime
                  for x in range(6)]
        assert benchmark.interpolate(values, prime) == [x % prime for x in coefficients]
    with pytest.raises(ValueError, match="prime exceeding"):
        benchmark.interpolate([1] * 8, 7)
    with pytest.raises(ValueError, match="prime exceeding"):
        benchmark.interpolate([1, 2], 9)


@pytest.mark.parametrize("signed", [False, True])
def test_crt_certifies_full_large_integer_vector(signed):
    coefficients = [10**70 + 7, 0, 7**80, -(10**68 + 71) if signed else 10**68 + 71]
    bounds = [abs(x) + 1 for x in coefficients]
    primes = benchmark.plan_primes(max(bounds), signed)
    rows = [[x % prime for x in coefficients] for prime in primes]
    assert benchmark.crt_vectors(rows, primes, bounds, signed) == coefficients
    with pytest.raises(ValueError, match="coverage"):
        benchmark.crt_vectors(rows[:-1], primes[:-1], bounds, signed)


@pytest.mark.parametrize("contract", benchmark.CONTRACTS)
def test_native_and_compiled_complete_vectors_match_definition(baselines, contract):
    pytest.importorskip("perm_polynomial_ext")
    pytest.importorskip("perm_c_ext")
    family = "dense" if contract == "rook" else "signed_dense"
    case = benchmark.instance(family, 4, 100, 5910, contract)
    expected = (partial_matching_coefficients(case["linear"]) if contract == "rook"
                else permutation_coefficients(case["constant"], case["linear"]))
    bounds = benchmark.coefficient_bounds(case)
    signed = contract != "rook"
    primes = benchmark.plan_primes(max(bounds), signed)
    for method in benchmark.methods_for(case):
        result = benchmark.method_result(method, case, primes, bounds, signed, baselines)
        benchmark.validate_result(result, case, primes, bounds, signed)
        assert result["coefficients"] == expected


def test_public_binary_rook_residues_are_recorded():
    pytest.importorskip("perm_polynomial_ext")
    matrix = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    case = {"n": 3, "contract": "rook", "constant": [[1]*3 for _ in range(3)],
            "linear": matrix}
    result = benchmark.public_exact_result(case)
    assert result["entry_point"] == "rook_polynomial"
    assert result["coefficients"] == partial_matching_coefficients(matrix)
    assert result["prime_count"] > 0
    assert result["residue_transcript"]
    benchmark.validate_result(result, case, [], benchmark.coefficient_bounds(case), False)


@pytest.mark.parametrize("matrix", [
    [[2, 0, 0], [0, 3, 0], [0, 0, 5]],
    [[2, 3, 0, 0], [5, 7, 0, 0], [0, 0, 11, 13], [0, 0, 17, 19]],
    [[2, 3, 5, 0, 0], [7, 11, 13, 0, 0], [0, 0, 0, 17, 19],
     [0, 0, 0, 23, 29], [0, 0, 0, 31, 37]],
    [[2, 3, 0], [0, 0, 5], [0, 0, 7]],
])
def test_public_weighted_components_replay_full_exact_polynomial(matrix):
    pytest.importorskip("perm_polynomial_ext")
    n = len(matrix)
    case = {"n": n, "contract": "rook", "constant": [[1]*n for _ in range(n)], "linear": matrix}
    result = benchmark.public_exact_result(case)
    assert result["entry_point"] == "weighted_rook_polynomial"
    assert result["coefficients"] == partial_matching_coefficients(matrix)
    benchmark.validate_result(result, case, [], benchmark.coefficient_bounds(case), False)
    assert benchmark.validate_public_residues(result["coefficients"], result["residue_transcript"], case) == result["primes"]


@pytest.mark.parametrize("defect", ["matrix", "component_values", "residue", "index", "missing", "full_values"])
def test_weighted_component_transcript_corruption_is_rejected(defect):
    pytest.importorskip("perm_polynomial_ext")
    # Four-vertex factors still exercise CRT after the small-factor dispatch.
    matrix = [[2+i+j if i//4 == j//4 else 0 for j in range(8)] for i in range(8)]
    case = {"n": 8, "contract": "rook", "constant": [[1]*8 for _ in range(8)], "linear": matrix}
    result = copy.deepcopy(benchmark.public_exact_result(case))
    first = result["residue_transcript"][0]
    if defect == "matrix":
        first["weighted_component"]["matrix"][0][0] += 1
    elif defect == "component_values":
        first["weighted_component"]["coefficients"][1] += 1
    elif defect == "residue":
        first["residue_rows"][0][1] += 1
    elif defect == "index":
        first["weighted_component"]["index"] = 1
    elif defect == "missing":
        result["residue_transcript"].pop()
    else:
        result["coefficients"][1] += 1
    with pytest.raises(ValueError):
        benchmark.validate_result(result, case, [], benchmark.coefficient_bounds(case), False)


def small_factor_case_and_captures():
    matrix = [[2, 3, 0, 0], [5, 7, 0, 0], [0, 0, 11, 13], [0, 0, 17, 19]]
    case = {"n": 4, "contract": "rook", "constant": [[1]*4 for _ in range(4)], "linear": matrix}
    captures = []
    for index, indices in enumerate(([0, 1], [2, 3])):
        submatrix = [[matrix[row][column] for column in indices] for row in indices]
        captures.append({"entry_point": "_weighted_rook_small_component_validated",
            "primes": [], "residue_rows": [], "format": "exact_rook_coefficients", "call_seconds": 0.,
            "weighted_component": {"index": index, "rows": indices, "columns": indices,
                                   "matrix": submatrix, "coefficients": partial_matching_coefficients(submatrix)}})
    return case, captures


def test_explicit_small_factors_replay_without_production_evaluation(monkeypatch):
    case, captures = small_factor_case_and_captures()
    def forbidden(*args, **kwargs):
        raise AssertionError("exact transcript replay must not call production")
    monkeypatch.setattr(benchmark.importlib, "import_module", forbidden)
    assert benchmark.validate_public_residues(partial_matching_coefficients(case["linear"]), captures, case) == []


@pytest.mark.parametrize("defect", ["matrix", "coefficients", "rows", "columns", "bool_index", "index",
    "missing", "duplicate", "primes", "residue_rows", "entry_point", "strip_metadata", "full_values"])
def test_explicit_small_factor_corruption_is_rejected(defect):
    case, captures = small_factor_case_and_captures()
    values = partial_matching_coefficients(case["linear"])
    first = captures[0]
    metadata = first["weighted_component"]
    if defect == "matrix": metadata["matrix"][0][0] += 1
    elif defect == "coefficients": metadata["coefficients"][1] += 1
    elif defect == "rows": metadata["rows"] = [0, 2]
    elif defect == "columns": metadata["columns"] = [1, 0]
    elif defect == "bool_index": metadata["rows"] = [False, 1]
    elif defect == "index": metadata["index"] = 1
    elif defect == "missing": captures.pop()
    elif defect == "duplicate": captures.insert(0, copy.deepcopy(first))
    elif defect == "primes": first["primes"] = [101]
    elif defect == "residue_rows": first["residue_rows"] = [[1, 17, 29]]
    elif defect == "entry_point": first["entry_point"] = "unrecorded_shortcut"
    elif defect == "strip_metadata":
        for capture in captures: capture.pop("weighted_component")
    elif defect == "full_values": values[1] += 1
    with pytest.raises(ValueError):
        benchmark.validate_public_residues(values, captures, case)


@pytest.mark.parametrize("shape", [(1, 4), (2, 4), (4, 2), (3, 5), (5, 3)])
def test_assignment_replay_handles_rectangular_factors_and_huge_weights(shape):
    rows, columns = shape
    matrix = [[(10**80+i*columns+j) if (i+j)%3 else 0 for j in range(columns)] for i in range(rows)]
    size = max(shape)
    padded = [[matrix[i][j] if i < rows and j < columns else 0 for j in range(size)] for i in range(size)]
    expected = partial_matching_coefficients(padded)
    assert benchmark._small_weighted_factor_by_assignments(matrix) == expected[:min(shape)+1]
    assert not any(expected[min(shape)+1:])


def test_small_capture_cannot_substitute_for_connected_full_order_crt():
    case, captures = small_factor_case_and_captures()
    matrix = captures[0]["weighted_component"]["matrix"]
    case = {"n": 2, "contract": "rook", "constant": [[1]*2 for _ in range(2)], "linear": matrix}
    with pytest.raises(ValueError, match="full-order connected"):
        benchmark.validate_public_residues(partial_matching_coefficients(matrix), captures[:1], case)


def test_historical_tagged_small_crt_factors_remain_supported():
    pytest.importorskip("perm_polynomial_ext")
    import perm_polynomial as polynomial
    case, _ = small_factor_case_and_captures()
    with benchmark.capture_public_residues(polynomial) as captures:
        for indices in ([0, 1], [2, 3]):
            matrix = tuple(tuple(case["linear"][row][column] for column in indices) for row in indices)
            polynomial._weighted_rook_polynomial_crt_validated(matrix, "native", 1)
    assert captures and all(c["format"] != "exact_rook_coefficients" for c in captures)
    assert benchmark.validate_public_residues(partial_matching_coefficients(case["linear"]), captures, case)


def test_small_factor_capture_hook_restores_after_failure(monkeypatch):
    import perm_polynomial as polynomial
    def fail(matrix, rows, columns):
        raise RuntimeError("test helper failure")
    monkeypatch.setattr(polynomial, "_weighted_rook_small_component_validated", fail, raising=False)
    original_crt = polynomial._weighted_rook_polynomial_crt_validated
    with pytest.raises(RuntimeError, match="helper failure"):
        with benchmark.capture_public_residues(polynomial):
            polynomial._weighted_rook_small_component_validated([[2]], [0], [0])
    assert polynomial._weighted_rook_small_component_validated is fail
    assert polynomial._weighted_rook_polynomial_crt_validated is original_crt


def test_public_small_factor_capture_is_explicit_and_has_no_residue_time():
    pytest.importorskip("perm_polynomial_ext")
    import perm_polynomial as polynomial
    assert hasattr(polynomial, "_weighted_rook_small_component_validated")
    case, _ = small_factor_case_and_captures()
    result = benchmark.public_exact_result(case)
    assert len(result["residue_transcript"]) == 2
    assert all(c["format"] == "exact_rook_coefficients" for c in result["residue_transcript"])
    assert result["prime_count"] == 0 and result["primes"] == []
    assert result["residue_api_seconds"] == 0
    assert result["exact_component_api_seconds"] > 0
    assert result["coefficients"] == partial_matching_coefficients(case["linear"])
    benchmark.validate_result(result, case, [], benchmark.coefficient_bounds(case), False)


@pytest.mark.parametrize("small_indices", [[0, 1], [0, 3], [4, 5]])
def test_mixed_exact_and_crt_component_indices_and_convolution(small_indices):
    pytest.importorskip("perm_polynomial_ext")
    small = set(small_indices)
    matrix = [[2+i+j if ((i in small) == (j in small)) else 0 for j in range(6)] for i in range(6)]
    case = {"n": 6, "contract": "rook", "constant": [[1]*6 for _ in range(6)], "linear": matrix}
    result = benchmark.public_exact_result(case)
    captures = result["residue_transcript"]
    formats = {c["format"] for c in captures}
    assert "exact_rook_coefficients" in formats
    assert formats-{"exact_rook_coefficients"} in ({"rook_coefficients"}, {"affine_coefficients"})
    assert sorted({c["weighted_component"]["index"] for c in captures}) == [0, 1]
    assert result["coefficients"] == partial_matching_coefficients(matrix)
    assert result["prime_count"] > 0
    benchmark.validate_result(result, case, [], benchmark.coefficient_bounds(case), False)


def test_explicit_stars_remain_compatible_with_implicit_historical_stars():
    import perm_polynomial as polynomial
    matrix = [[2, 3, 0], [0, 0, 5], [0, 0, 7]]
    case = {"n": 3, "contract": "rook", "constant": [[1]*3 for _ in range(3)], "linear": matrix}
    values = partial_matching_coefficients(matrix)
    with benchmark.capture_public_residues(polynomial) as captures:
        polynomial._weighted_rook_small_component_validated(matrix, [0], [0, 1])
        polynomial._weighted_rook_small_component_validated(matrix, [1, 2], [2])
    assert len(captures) == 2
    assert benchmark.validate_public_residues(values, captures, case) == []
    assert benchmark.validate_public_residues(values, [], case) == []


@pytest.mark.parametrize("matrix", [[], [[]], [[1]*4 for _ in range(4)], [[1], [2, 3]],
                                     [[-1]], [[True]], [[1.0]]])
def test_assignment_replay_rejects_out_of_contract_factors(matrix):
    with pytest.raises(ValueError):
        benchmark._small_weighted_factor_by_assignments(matrix)


def test_legacy_full_weighted_capture_remains_supported():
    pytest.importorskip("perm_polynomial_ext")
    matrix = [[2, 3], [5, 7]]
    case = {"n": 2, "contract": "rook", "constant": [[1]*2 for _ in range(2)], "linear": matrix}
    result = benchmark.public_exact_result(case)
    for capture in result["residue_transcript"]:
        capture.pop("weighted_component", None)
    benchmark.validate_result(result, case, [], benchmark.coefficient_bounds(case), False)


@pytest.mark.parametrize("contract", benchmark.CONTRACTS)
def test_interpolation_uses_known_constant_term_without_scalar_call(monkeypatch, contract):
    native = pytest.importorskip("perm_c_ext")
    pytest.importorskip("perm_polynomial_ext")
    calls = []

    def scalar(matrix, prime, threads):
        calls.append(matrix)
        return sum(math.prod(matrix[i][j] for i, j in enumerate(columns))
                   for columns in itertools.permutations(range(len(matrix)))) % prime

    monkeypatch.setattr(native, "permanent_mod_p", scalar)
    case = benchmark.instance("dense", 4, 1, 778, contract)
    result = benchmark.modular_result("scalar_interpolation", case, 101, None)
    expected = (partial_matching_coefficients(case["linear"]) if contract == "rook"
                else permutation_coefficients(case["constant"], case["linear"]))
    assert result["coefficients"] == [value % 101 for value in expected]
    assert len(calls) == (5 if contract == "affine" else 4)
    if contract != "affine":
        assert result["scalar_samples"][0]["scalar_native_api_seconds"] == 0
        assert result["scalar_samples"][0]["source"] in ("per(I)=1", "per(J)=n!")


@pytest.mark.parametrize("family", benchmark.FAMILIES)
def test_seeded_matrix_contracts_and_bound_are_independently_valid(family):
    contract = "affine" if family == "signed_dense" else "rook"
    case = benchmark.instance(family, 4, 2, 707, contract)
    assert case == benchmark.instance(family, 4, 2, 707, contract)
    exact = (partial_matching_coefficients(case["linear"]) if contract == "rook"
             else permutation_coefficients(case["constant"], case["linear"]))
    assert all(abs(value) <= bound for value, bound in
               zip(exact, benchmark.coefficient_bounds(case)))


def fake_result(method, case, primes, bounds, signed, baselines):
    coefficients = permutation_coefficients(case["constant"], case["linear"])
    if method == "public_exact":
        return {"coefficients": coefficients, "residue_transcript": [], "primes": [],
                "prime_count": 0, "total_seconds": 1.}
    return {"coefficients": coefficients, "total_seconds": 1.,
            "residues": [{"prime": p, "coefficients": [x % p for x in coefficients]}
                         for p in primes]}


def test_resume_replays_partial_transcript_without_reexecuting_completed_samples(tmp_path, monkeypatch):
    case = benchmark.instance("dense", 2, 1, 99, "principal")
    calls = []

    def evaluate(*args):
        calls.append(args[0])
        return fake_result(*args)

    monkeypatch.setattr(benchmark, "method_result", evaluate)
    first = benchmark.run_case(case, tmp_path, None, 1, 1, "manifest")
    assert first["complete"] and len(calls) == 8
    path = next((tmp_path / "cases").glob("*.json"))
    partial = copy.deepcopy(first)
    partial["samples"] = partial["samples"][:3]
    partial["complete"] = False
    benchmark.atomic_record(path, partial)
    calls.clear()
    second = benchmark.run_case(case, tmp_path, None, 1, 1, "manifest")
    assert second["complete"] and len(calls) == 5
    calls.clear()
    assert benchmark.run_case(case, tmp_path, None, 1, 1, "manifest")["complete"]
    assert not calls


def test_resume_rejects_configuration_and_even_rehashed_forged_coefficients(tmp_path, monkeypatch):
    case = benchmark.instance("dense", 2, 1, 99, "principal")
    monkeypatch.setattr(benchmark, "method_result", fake_result)
    benchmark.run_case(case, tmp_path, None, 1, 1, "manifest")
    with pytest.raises(ValueError, match="configuration mismatch"):
        benchmark.run_case(case, tmp_path, None, 2, 1, "manifest")
    path = next((tmp_path / "cases").glob("*.json"))
    record = benchmark.read_record(path)
    record["samples"][0]["result"]["coefficients"][0] += 1
    benchmark.atomic_record(path, record)
    with pytest.raises(ValueError, match="coefficient mismatch"):
        benchmark.run_case(case, tmp_path, None, 1, 1, "manifest")


def test_mismatch_fails_before_accepting_sample(tmp_path, monkeypatch):
    case = benchmark.instance("dense", 2, 1, 99, "principal")

    def inconsistent(*args):
        result = fake_result(*args)
        if args[0] == "public_exact":
            result["coefficients"][0] = 0
        return result

    monkeypatch.setattr(benchmark, "method_result", inconsistent)
    with pytest.raises(ValueError, match="coefficient mismatch"):
        benchmark.run_case(case, tmp_path, None, 1, 1, "manifest")
    record = benchmark.read_record(next((tmp_path / "cases").glob("*.json")))
    assert not record["complete"]
    assert all(sample["method"] != "public_exact" for sample in record["samples"])
    assert record["failure"]["method"] == "public_exact"
    assert record["failure"]["result"]["coefficients"][0] == 0
    assert "coefficient mismatch" in record["failure"]["error"]


def test_expired_budget_checkpoints_before_next_kernel(tmp_path, monkeypatch):
    case = benchmark.instance("dense", 2, 1, 99, "principal")
    monkeypatch.setattr(benchmark, "method_result", lambda *args: pytest.fail("kernel called"))
    record = benchmark.run_case(case, tmp_path, None, 1, 1, "manifest", time.monotonic() - 1)
    assert not record["complete"] and not record["samples"]


def test_digest_corruption_is_rejected(tmp_path):
    path = tmp_path / "record.json"
    benchmark.atomic_record(path, {"count": 7})
    path.write_text(path.read_text().replace('"count": 7', '"count": 8'))
    with pytest.raises(ValueError, match="digest mismatch"):
        benchmark.read_record(path)


def test_campaign_resume_rejects_modified_archived_source(tmp_path, monkeypatch, baselines):
    pytest.importorskip("perm_polynomial_ext")
    pytest.importorskip("perm_c_ext")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    monkeypatch.setenv("OMP_DYNAMIC", "FALSE")
    args = Namespace(output=str(tmp_path), sizes=[2], scales=[1], families=["dense"],
                     contracts=["affine"], seed=331, repeats=1, warmups=1,
                     allow_large=False, max_seconds=10)
    assert benchmark.run_campaign(args)["complete"]
    rows = summarizer.comparison_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["n"] == 2 and rows[0]["public_prime_count"] == 1
    assert rows[0]["common_prime_count"] == 1
    archived = tmp_path / "sources" / "notes" / "benchmarks" / "polynomial_baselines.cpp"
    archived.write_text(archived.read_text() + "\n// corruption\n")
    with pytest.raises(ValueError, match="archived source digest mismatch"):
        benchmark.run_campaign(args)
