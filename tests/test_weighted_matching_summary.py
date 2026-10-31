"""CPU-only audit tampering and timing/observable interpretation checks."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1]/"notes/benchmarks/summarize_weighted_matching_study.py"
if not SOURCE.is_file():
    pytest.skip("weighted matching summary is a source-checkout tool", allow_module_level=True)
if sys.version_info < (3, 10):
    pytest.skip("research campaign helpers require Python 3.10+", allow_module_level=True)
SPEC = importlib.util.spec_from_file_location("weighted_matching_summary", SOURCE)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
study, ref = summary.study, summary.reference
from notes.benchmarks.weighted_matching_statistics import (
    exact_occupancy_statistics, belief_propagation_statistics,
    independent_edge_statistics, total_variation_distance,
)


def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return ref.file_digest(path)


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    """Tiny synthetic transcript, not a performance measurement or native run."""
    root, run = tmp_path/"source", tmp_path/"run"
    monkeypatch.setattr(summary, "ROOT", root)
    monkeypatch.setattr(study, "METHODS", ("native", "frontier"))
    sources = {}
    for name in study.SOURCES:
        contents = ("fixture source: "+name).encode()
        sources[name] = write_bytes(root/name, contents)
        write_bytes(run/"sources"/name, contents)
    native = {}
    for name in ("perm_polynomial_ext", "perm_c_ext"):
        path = root/(name+".so")
        native[name] = {"path": str(path), "sha256": write_bytes(path, name.encode())}
    baseline_hash = write_bytes(run/"build/polynomial_baselines.so", b"fixture baseline")
    partition_hash = write_bytes(run/"partition-build/weighted_matching_partition.so", b"fixture query")
    build = {"identity": {"source_sha256": sources["notes/benchmarks/polynomial_baselines.cpp"]},
             "binary_sha256": baseline_hash}
    ref.atomic_record(run/"build/build.json", build)
    partition = {"library_sha256": partition_hash,
        "source_sha256": sources["notes/benchmarks/weighted_matching_partition.cpp"],
        "wrapper_sha256": sources["notes/benchmarks/weighted_matching_partition.py"],
        "crt_helper_sha256": sources["notes/benchmarks/polynomial_compare.py"],
        "stdout_sha256": write_bytes(run/"partition-build/compile.stdout.log", b""),
        "stderr_sha256": write_bytes(run/"partition-build/compile.stderr.log", b"")}
    (run/"partition-build/build.json").write_text(json.dumps(partition))
    config = {"sizes": [1], "families": ["dense"], "seeds": [17], "scales": [1],
              "application_size": 1, "repeats": 1, "warmups": 0, "max_states": 8, "timeout": 90}
    manifest = {"identity": {"schema": study.SCHEMA, "configuration": config,
        "activities": list(study.ACTIVITIES), "methods": list(study.METHODS),
        "sources": sources, "native_binaries": native, "previous_source_sha256": None,
        "baseline_binary_sha256": baseline_hash, "partition_binary_sha256": partition_hash},
        "build": build, "partition_build": partition}
    ref.atomic_record(run/"manifest.json", manifest)
    case = study.make_case("dense", 1, 17, 1)
    coeff = [1, case["matrix"][0][0]]
    methods = {"native": {"status": "COMPLETE", "method": "native", "coefficients": coeff,
        "median_seconds": 2., "baseline_peak_rss_bytes": 100, "peak_process_rss_bytes": 120,
        "samples": [{"phase": "measurement", "seconds": 2., "cpu_seconds": 1.9,
                     "result": {"coefficients": coeff, "component_residue_calls": []}}]},
        "frontier": {"status": "STATE_LIMIT", "limit": 8, "error": "fixture cap"}}
    record = {"manifest_sha256": ref.digest(manifest), "case": case, "coefficients": coeff,
              "methods": methods, "complete": True}
    filename = ref.digest(case)+".json"
    ref.atomic_record(run/"cases"/filename, record)
    points = []
    for activity_text in study.ACTIVITIES:
        activity = Fraction(activity_text)
        p, q = activity.numerator, activity.denominator
        exact = exact_occupancy_statistics(coeff, activity)
        bp = belief_propagation_statistics(case["matrix"], activity)
        independent = independent_edge_statistics(case["matrix"], activity)
        moments = [q+coeff[1]*p, coeff[1]*p, coeff[1]*p]
        bound = moments[0]
        primes = ref.plan_primes(bound, False)
        direct = {"status": "COMPLETE", "exact": True, "order": 1, "activity": activity,
            "modulus": math.prod(primes), "library_sha256": partition_hash,
            "bounds": [bound]*3, "primes": primes, "numerator_moments": moments,
            "residue_rows": [[m % prime for m in moments] for prime in primes], "denominator": q,
            **{key: exact[key] for key in ("partition", "mean", "variance")}}
        log_error = bp["log_partition"]-exact["log_partition"]
        points.append(study.jsonable({"activity": activity_text, "exact": exact, "bp": bp,
            "direct_query": direct, "independent_edges": independent,
            "independent_edges_tv": total_variation_distance(exact["probabilities"], independent["probabilities"]),
            "bp_log_partition_error": log_error, "bp_partition_relative_error": math.expm1(log_error),
            "bp_mean_absolute_error": abs(bp["mean"]-float(exact["mean"])),
            "timing": {name: {"samples": [value], "median_seconds": value} for name, value in
                       (("exact_statistics", .01), ("direct_query", .2), ("bp", .03))}}))
    curve = {"manifest_sha256": ref.digest(manifest), "case": case, "coefficients": coeff,
             "status": "COMPLETE", "native_setup_seconds": 2., "points": points}
    ref.atomic_record(run/"applications"/filename, curve)
    ref.atomic_record(run/"completion.json", {"status": "COMPLETE", "manifest_sha256": ref.digest(manifest),
        "case_paths": ["cases/"+filename], "application_paths": ["applications/"+filename],
        "case_count": 1, "application_count": 1})
    return root, run, filename


def test_complete_arithmetic_audit_and_export(fixture_run, tmp_path):
    _, run, _ = fixture_run
    output = tmp_path/"summary"
    report = summary.export(run, output, make_figures=False)
    assert report["status"] == "AUDITED_COMPLETE"
    assert report["activity_point_count"] == 13
    assert report["native_comparisons"]["frontier"] == {"win": 0, "loss": 0, "tie": 0, "censored": 1}
    assert report["break_even"][0]["estimated_break_even_queries"] == 11
    assert report["coefficient_faster_full_grid_cases"] == 1
    assert ref.read_record(output/"report.json") == report
    for name, digest in report["artifact_sha256"].items():
        assert ref.file_digest(output/name) == digest
    with pytest.raises(FileExistsError):
        summary.export(run, output, make_figures=False)


@pytest.mark.parametrize("target", ["current_source", "snapshot", "native_binary", "baseline_binary", "partition_binary", "build_log"])
def test_byte_mutation_rejected(fixture_run, target):
    root, run, _ = fixture_run
    paths = {"current_source": root/study.SOURCES[0], "snapshot": run/"sources"/study.SOURCES[0],
        "native_binary": root/"perm_c_ext.so", "baseline_binary": run/"build/polynomial_baselines.so",
        "partition_binary": run/"partition-build/weighted_matching_partition.so",
        "build_log": run/"partition-build/compile.stderr.log"}
    paths[target].write_bytes(b"changed")
    with pytest.raises(ValueError):
        summary.audit(run)


@pytest.mark.parametrize("mutation", ["missing_method", "missing_point", "setup", "bp", "direct_binary", "coefficients", "state_cap", "completion", "case_count", "source_inventory"])
def test_rehashed_semantic_mutation_rejected(fixture_run, mutation):
    _, run, filename = fixture_run
    path = run/"cases"/filename
    if mutation in ("missing_point", "setup", "bp", "direct_binary"):
        path = run/"applications"/filename
    elif mutation in ("completion", "case_count"):
        path = run/"completion.json"
    elif mutation == "source_inventory":
        path = run/"manifest.json"
    record = ref.read_record(path)
    if mutation == "missing_method": del record["methods"]["frontier"]
    elif mutation == "missing_point": record["points"].pop()
    elif mutation == "setup": record["native_setup_seconds"] = 1.
    elif mutation == "bp": record["points"][0]["bp"]["mean"] += .1
    elif mutation == "direct_binary": record["points"][0]["direct_query"]["library_sha256"] = "0"*64
    elif mutation == "coefficients": record["coefficients"][1] += 1
    elif mutation == "state_cap": record["methods"]["frontier"]["limit"] += 1
    elif mutation == "completion": record["application_paths"] = []
    elif mutation == "case_count": record["case_count"] = 0
    elif mutation == "source_inventory": record["identity"]["sources"]["../secret"] = "0"*64
    ref.atomic_record(path, record)
    with pytest.raises(ValueError):
        summary.audit(run)


def test_bad_envelope_rejected(fixture_run):
    _, run, filename = fixture_run
    path = run/"cases"/filename
    data = json.loads(path.read_text())
    data["payload"]["complete"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        summary.audit(run)


def test_break_even_direction_and_no_positive_saving(fixture_run):
    _, run, filename = fixture_run
    curve = ref.read_record(run/"applications"/filename)
    row = summary.break_even_row(curve)
    assert row["grid_direct_over_coefficient_ratio"] == pytest.approx(2.6/2.13)
    assert row["first_observed_grid_prefix_break_even"] == 11
    points = summary.accuracy_rows([curve])
    assert points[-1]["cumulative_coefficient_seconds"] == pytest.approx(2.13)
    assert points[-1]["cumulative_direct_seconds"] == pytest.approx(2.6)
    assert points[0]["cumulative_direct_over_coefficient_ratio"] < 1
    assert points[-1]["cumulative_direct_over_coefficient_ratio"] > 1
    for point in curve["points"]:
        point["timing"]["exact_statistics"]["median_seconds"] = .3
    row = summary.break_even_row(curve)
    assert row["estimated_break_even_queries"] is None
    assert not row["grid_coefficient_faster"]
    assert row["first_observed_grid_prefix_break_even"] is None


def test_partition_error_is_not_mean_error(fixture_run):
    _, run, filename = fixture_run
    curve = ref.read_record(run/"applications"/filename)
    point = curve["points"][0]
    exact_mean = float(study.rational(point["exact"]["mean"]))
    point["bp"]["mean"] = exact_mean + .01
    point["bp_mean_absolute_error"] = .01
    point["bp_partition_relative_error"] = -.8
    point["bp_log_partition_error"] = math.log(.2)
    row = summary.accuracy_rows([curve])[0]
    assert row["bp_partition_relative_error"] == -.8
    assert row["bp_mean_absolute_error"] == .01
    assert row["bp_mean_relative_error"] == pytest.approx(.01/exact_mean)


def test_native_speedup_ratio_direction_and_censoring(fixture_run):
    _, run, filename = fixture_run
    case = ref.read_record(run/"cases"/filename)
    comparator = deepcopy(case["methods"]["native"])
    comparator["median_seconds"] = 8.
    case["methods"]["compiled_dp"] = comparator
    rows = {r["method"]: r for r in summary.timing_rows([case])}
    assert rows["compiled_dp"]["native_speedup_vs_method"] == 4.
    assert rows["compiled_dp"]["native_comparison"] == "win"
    assert rows["frontier"]["native_comparison"] == "censored"
    assert rows["frontier"]["median_seconds"] is None
    assert rows["frontier"]["native_speedup_vs_method"] is None
