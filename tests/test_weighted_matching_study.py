"""Study-contract checks using tiny exact CPU jobs and corrupted records."""
from __future__ import annotations

from argparse import Namespace
import copy
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from notes.benchmarks import weighted_matching_study as study


def matching_oracle(matrix):
    n = len(matrix)
    return [sum(math.prod(matrix[r][c] for r, c in zip(rows, columns))
                for rows in itertools.combinations(range(n), k)
                for columns in itertools.permutations(range(n), k))
            for k in range(n+1)]


@pytest.mark.parametrize("family", study.FAMILIES)
def test_make_case_preserves_identical_graph_and_weights_under_scaling(family):
    small = study.make_case(family, 7, 2026090703)
    repeated = study.make_case(family, 7, 2026090703)
    scaled = study.make_case(family, 7, 2026090703, 100)
    assert small == repeated
    assert scaled["matrix"] == [[100*x for x in row] for row in small["matrix"]]
    assert small["matrix_sha256"] == study.reference.digest(small["matrix"])
    assert scaled["matrix_sha256"] == study.reference.digest(scaled["matrix"])
    assert study.make_case(family, 7, 2026090704)["matrix"] != small["matrix"]
    small["matrix"][0][0] = -1
    assert study.make_case(family, 7, 2026090703) == repeated


@pytest.mark.parametrize("family,n,seed,scale", [
    ("unknown", 3, 1, 1), ("dense", 0, 1, 1), ("dense", 21, 1, 1),
    ("dense", True, 1, 1), ("dense", 3, True, 1),
    ("dense", 3, 1, 0), ("dense", 3, 1, True), ("dense", 3, 1, 1.5),
])
def test_make_case_guards_input_contract(family, n, seed, scale):
    with pytest.raises(ValueError):
        study.make_case(family, n, seed, scale)


def test_fraction_json_roundtrip_is_exact_for_large_signed_values():
    values = [Fraction(0), Fraction(2, 6), Fraction(-2**200, 3**73), Fraction(10**100+7, 1)]
    payload = json.loads(json.dumps(study.jsonable({"values": tuple(values)}), allow_nan=False))
    assert [study.rational(item) for item in payload["values"]] == values
    assert payload["values"][1] == {"numerator": 1, "denominator": 3}
    with pytest.raises(ZeroDivisionError):
        study.rational({"numerator": 1, "denominator": 0})


@pytest.fixture(scope="module")
def baseline_library(tmp_path_factory):
    if shutil.which("c++") is None:
        pytest.skip("C++ compiler unavailable")
    library, _ = study.reference.build_baselines(tmp_path_factory.mktemp("weighted-study-baseline"))
    return library


def request(case, method, **extra):
    return {"kind": "benchmark", "case": case, "method": method,
            "warmups": 1, "repeats": 2, "max_states": 64, **extra}


@pytest.fixture(scope="module")
def compiled_record(baseline_library):
    case = study.make_case("dense", 3, 2026090701, 100)
    methods = {method: study.benchmark_worker(request(case, method, library=str(baseline_library)))
               for method in ("compiled_dp", "polynomial_ryser")}
    return {"case": case, "methods": methods, "complete": False}


def test_compiled_workers_retain_complete_exact_crt_and_memory(compiled_record):
    result = study.validate_benchmark(compiled_record, 2, 1)
    assert result == matching_oracle(compiled_record["case"]["matrix"])
    for method in compiled_record["methods"].values():
        assert method["baseline_peak_rss_bytes"] > 0
        assert method["peak_process_rss_bytes"] >= method["baseline_peak_rss_bytes"]
        assert [s["phase"] for s in method["samples"]] == ["warmup", "measurement", "measurement"]
        for sample in method["samples"]:
            assert sample["cpu_seconds"] >= 0 and sample["seconds"] >= 0
            assert sample["result"]["residues"]


@pytest.mark.parametrize("defect", [
    "case", "unknown_method", "method_label", "sample_count", "phase", "sample_vector", "sample_float",
    "residue", "prime", "bound", "median", "wall_negative", "wall_nan", "wall_bool", "cpu_negative",
    "cpu_infinite", "cpu_bool", "rss_negative", "rss_bool", "rss_decreased", "summary", "missing_complete_method",
])
def test_corrupted_benchmark_record_is_rejected(compiled_record, defect):
    record = copy.deepcopy(compiled_record)
    result = record["methods"]["compiled_dp"]
    sample = result["samples"][1]
    item = sample["result"]
    if defect == "case":
        record["case"]["matrix"][0][0] += 1
    elif defect == "unknown_method":
        record["methods"]["unregistered"] = record["methods"].pop("compiled_dp")
    elif defect == "method_label":
        result["method"] = "native"
    elif defect == "sample_count":
        result["samples"].pop()
    elif defect == "phase":
        sample["phase"] = "warmup"
    elif defect == "sample_vector":
        item["coefficients"][1] += 1
    elif defect == "sample_float":
        item["coefficients"][0] = 1.0
    elif defect == "residue":
        item["residues"][0]["coefficients"][1] += 1
    elif defect == "prime":
        item["primes"][0] -= 2
    elif defect == "bound":
        item["bounds"][1] += 1
    elif defect == "median":
        result["median_seconds"] += 1
    elif defect.startswith("wall_"):
        sample["seconds"] = {"wall_negative": -1, "wall_nan": float("nan"), "wall_bool": True}[defect]
        if defect == "wall_bool":
            result["median_seconds"] = sum(s["seconds"] for s in result["samples"][1:])/2
    elif defect.startswith("cpu_"):
        sample["cpu_seconds"] = {"cpu_negative": -1, "cpu_infinite": float("inf"), "cpu_bool": True}[defect]
    elif defect == "rss_negative":
        result["baseline_peak_rss_bytes"] = -1
    elif defect == "rss_bool":
        result["baseline_peak_rss_bytes"] = True
    elif defect == "rss_decreased":
        result["peak_process_rss_bytes"] = result["baseline_peak_rss_bytes"]-1
    elif defect == "summary":
        record["coefficients"] = list(result["coefficients"])
        record["coefficients"][1] += 1
    else:
        record["complete"] = True
    with pytest.raises((ValueError, RuntimeError)):
        study.validate_benchmark(record, 2, 1)


def test_worker_rejects_different_answers_across_repeats(monkeypatch):
    case = study.make_case("dense", 1, 1)
    values = iter(([1, 2], [1, 3]))
    monkeypatch.setattr(study, "prepare_method", lambda request: None)
    monkeypatch.setattr(study, "evaluate_method", lambda request, implementation: {"coefficients": next(values)})
    with pytest.raises(ValueError, match="within-worker repeats"):
        study.benchmark_worker(request(case, "native", warmups=0, repeats=2))


def test_frontier_resource_limit_is_retained_as_a_limit_not_a_correct_answer():
    case = study.make_case("dense", 4, 2)
    result = study.benchmark_worker(request(case, "frontier", max_states=1))
    assert result["status"] == "STATE_LIMIT"
    assert "coefficients" not in result and result["limit"] == 1
    assert result["peak_process_rss_bytes"] >= result["baseline_peak_rss_bytes"] > 0
    assert study.validate_benchmark({"case": case, "methods": {"frontier": result}, "complete": False}, 2, 1) is None


def test_unknown_method_cannot_be_prepared():
    with pytest.raises(ValueError, match="unknown benchmark method"):
        study.prepare_method({"method": "unknown"})


@pytest.mark.parametrize("method", ["native", "previous_native"])
def test_native_and_explicit_previous_source_workers_match_definition(tmp_path, method):
    pytest.importorskip("perm_polynomial_ext")
    case = study.make_case("block", 4, 2026090703)
    # A copied public source exercises the explicit previous-module loader;
    # compatibility with genuinely older untagged captures has its own tests.
    source = tmp_path/"previous.py"
    shutil.copyfile(study.ROOT/"perm_polynomial.py", source)
    result = study.benchmark_worker(request(case, method, previous_source=str(source)))
    assert result["coefficients"] == matching_oracle(case["matrix"])
    study.validate_benchmark({"case": case, "methods": {method: result}, "complete": False}, 2, 1)
    assert result["samples"][0]["result"]["component_residue_calls"]


def test_native_component_transcript_corruption_is_rejected():
    pytest.importorskip("perm_polynomial_ext")
    # Connected full-order inputs retain CRT rather than the exact small-factor hook.
    case = study.make_case("dense", 4, 2026090703)
    result = study.benchmark_worker(request(case, "native"))
    first = result["samples"][0]["result"]["component_residue_calls"][0]
    first["residue_rows"][0][1] += 1
    with pytest.raises(ValueError):
        study.validate_benchmark({"case": case, "methods": {"native": result}, "complete": False}, 2, 1)


def test_real_isolated_compiled_worker_reports_process_memory(baseline_library):
    case = study.make_case("dense", 2, 2026090701)
    result = study.execute_worker(request(case, "compiled_dp", library=str(baseline_library)),
                                  timeout=15, cpu=study.default_cpu())
    assert result["status"] == "COMPLETE"
    assert result["coefficients"] == matching_oracle(case["matrix"])
    assert result["process_wall_seconds"] > 0
    assert result["peak_process_rss_bytes"] >= result["baseline_peak_rss_bytes"] > 0
    assert result["stderr"] == ""
    assert result["cpu_affinity"]["pinned"] == (study.default_cpu() is not None)
    assert result["cpu_affinity"]["supported"] == (study.available_affinity() is not None)


def test_isolated_worker_retains_error_and_timeout_details(monkeypatch):
    case = study.make_case("dense", 1, 1)
    result = study.execute_worker(request(case, "unknown"), 15, study.default_cpu())
    assert result["status"] == "ERROR" and result["returncode"] != 0
    assert "unknown benchmark method" in result["stderr"]

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.5, output=b"partial output", stderr=b"partial error")

    monkeypatch.setattr(study.subprocess, "run", timeout)
    result = study.execute_worker(request(case, "native"), 0.5, study.default_cpu())
    assert result["status"] == "TIMEOUT" and result["timeout_seconds"] == 0.5
    assert "partial output" in result["stdout"] and "partial error" in result["stderr"]


@pytest.mark.parametrize("override", [{"sizes": [13]}, {"repeats": 0}, {"warmups": -1},
                                     {"max_states": 0}, {"timeout": -1}, {"application_size": 4}])
def test_study_resource_guards_fail_before_build_or_output(tmp_path, monkeypatch, override):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid run controls must not compile")

    monkeypatch.setattr(study.reference, "build_baselines", forbidden)
    args = Namespace(output=str(tmp_path/"not-created"), sizes=[3], seeds=[1], scales=[1],
                     families=["dense"], repeats=1, warmups=0, max_states=64, timeout=5,
                     application_size=3, cpu=study.default_cpu(), allow_large=False,
                     previous_source=None)
    vars(args).update(override)
    with pytest.raises(ValueError):
        study.run(args)
    assert not Path(args.output).exists()


def test_missing_affinity_api_is_reported_and_explicit_pinning_rejected(monkeypatch):
    monkeypatch.delattr(study.os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(study.os, "sched_setaffinity", raising=False)
    assert study.available_affinity() is None
    assert study.default_cpu() is None
    assert study.configure_cpu(None) == {
        "requested_cpu": None, "effective_cpus": None,
        "pinned": False, "supported": False,
    }
    with pytest.raises(ValueError, match="omit --cpu"):
        study.configure_cpu(0)


def test_affinity_uses_allowed_ids_and_checks_applied_mask(monkeypatch):
    state = {"mask": {3, 7}}
    monkeypatch.setattr(study.os, "sched_getaffinity", lambda pid: state["mask"], raising=False)
    monkeypatch.setattr(study.os, "sched_setaffinity",
                        lambda pid, mask: state.update(mask=set(mask)), raising=False)
    assert study.default_cpu() == 7
    with pytest.raises(ValueError, match="outside available"):
        study.validate_cpu(0)
    assert study.configure_cpu(3) == {
        "requested_cpu": 3, "effective_cpus": [3], "pinned": True, "supported": True,
    }
    state["mask"] = {3, 7}
    monkeypatch.setattr(study.os, "sched_setaffinity", lambda pid, mask: None)
    with pytest.raises(RuntimeError, match="not applied"):
        study.configure_cpu(3)


@pytest.mark.parametrize("cpu", [True, -1, 1.5, "0"])
def test_invalid_affinity_request_is_rejected(cpu):
    with pytest.raises(ValueError, match="CPU must be"):
        study.validate_cpu(cpu)


@pytest.mark.parametrize("system,multiplier", [("darwin", 1), ("linux", 1024)])
def test_peak_rss_converts_host_units_to_bytes(monkeypatch, system, multiplier):
    monkeypatch.setattr(study.sys, "platform", system)
    monkeypatch.setattr(study.resource, "getrusage", lambda who: Namespace(ru_maxrss=12345))
    assert study.peak_rss_bytes() == 12345 * multiplier


@pytest.fixture(scope="module")
def partition_build(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    from notes.benchmarks.weighted_matching_partition import build_partition_baseline
    return build_partition_baseline(tmp_path_factory.mktemp("weighted-study-query"))


@pytest.fixture
def application_record(partition_build, monkeypatch):
    activities = ("0", "1/3", "2")
    monkeypatch.setattr(study, "ACTIVITIES", activities)
    case = study.make_case("dense", 2, 2026090701)
    coefficients = matching_oracle(case["matrix"])
    result = study.application_worker({"case": case, "coefficients": coefficients,
        "activities": activities, "repeats": 1, "partition_library": partition_build["library"]})
    curve = {"case": case, "coefficients": coefficients, "native_setup_seconds": 0.001, **result}
    benchmark = {"case": case, "coefficients": coefficients,
                 "methods": {"native": {"median_seconds": 0.001}}}
    return curve, benchmark


def test_application_worker_exact_statistics_agree_with_independent_scalar_queries(application_record):
    curve, benchmark = application_record
    study.validate_application(curve, benchmark, 1)
    assert [point["activity"] for point in curve["points"]] == ["0", "1/3", "2"]
    for point in curve["points"]:
        assert sum(study.rational(p) for p in point["exact"]["probabilities"]) == 1
        for name in ("partition", "mean", "variance"):
            assert study.rational(point["exact"][name]) == study.rational(point["direct_query"][name])


@pytest.mark.parametrize("defect", ["coefficients", "activity", "exact_mean", "residue", "direct_mean", "timing", "median"])
def test_saved_application_arithmetic_corruption_is_rejected(application_record, defect):
    curve, benchmark = copy.deepcopy(application_record)
    point = curve["points"][1]
    if defect == "coefficients":
        curve["coefficients"][1] += 1
    elif defect == "activity":
        point["activity"] = "1/2"
    elif defect == "exact_mean":
        point["exact"]["mean"]["numerator"] += 1
    elif defect == "residue":
        point["direct_query"]["residue_rows"][0][0] += 1
    elif defect == "direct_mean":
        point["direct_query"]["mean"]["numerator"] += 1
    elif defect == "timing":
        point["timing"]["direct_query"]["samples"][0] = -1
    else:
        point["timing"]["direct_query"]["median_seconds"] += 1
    with pytest.raises(ValueError):
        study.validate_application(curve, benchmark, 1)
