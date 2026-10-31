#!/usr/bin/env python3
"""Public exact weighted-matching study, with isolated CPU workers and checkpoints.

Run from the checkout; see WEIGHTED_MATCHINGS.md. No private evaluator data is
read. Timing and memory are per method/process, not claims of an idle host.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notes.benchmarks import polynomial_compare as reference

SCHEMA = "weighted-matching-study-v1"
FAMILIES = ("dense", "sparse", "banded", "diagonal", "block", "repeated_rows")
METHODS = ("native", "compiled_dp", "polynomial_ryser", "frontier")
ACTIVITIES = ("1/1000", "1/300", "1/100", "1/30", "1/10", "1/3", "1",
              "3", "10", "30", "100", "300", "1000")
SOURCES = ("perm_polynomial.py", "perm_polynomial_omp.c", "perm_binary.py",
           "perm_c_omp.c", "setup.py", "notes/benchmarks/polynomial_compare.py",
           "notes/benchmarks/polynomial_baselines.cpp",
           "notes/benchmarks/weighted_matching_study.py",
           "notes/benchmarks/weighted_matching_frontier.py",
           "notes/benchmarks/weighted_matching_statistics.py",
           "notes/benchmarks/weighted_matching_partition.py",
           "notes/benchmarks/weighted_matching_partition.cpp",
           "notes/benchmarks/summarize_weighted_matching_study.py")


def available_affinity():
    """Allowed CPU IDs, or None when the OS has no affinity API.

    A machine's CPU count is not an affinity mask. In particular, do not
    manufacture CPU IDs or claim that macOS workers have been pinned.
    """
    if not (hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")):
        return None
    return sorted(os.sched_getaffinity(0))


def default_cpu():
    affinity = available_affinity()
    return max(affinity) if affinity else None


def validate_cpu(cpu):
    if cpu is None:
        return
    if type(cpu) is not int or cpu < 0:
        raise ValueError("CPU must be a nonnegative integer or None for an unpinned worker")
    affinity = available_affinity()
    if affinity is None:
        raise ValueError("CPU pinning is unavailable on this platform; omit --cpu")
    if cpu not in affinity:
        raise ValueError("requested CPU is outside available affinity")


def configure_cpu(cpu):
    validate_cpu(cpu)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    effective = available_affinity()
    if cpu is not None and effective != [cpu]:
        raise RuntimeError("worker CPU affinity was not applied")
    return {"requested_cpu": cpu, "effective_cpus": effective,
            "pinned": cpu is not None, "supported": effective is not None}


def peak_rss_bytes():
    # Darwin reports bytes; Linux and the other supported Unix hosts use KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) * (1 if sys.platform == "darwin" else 1024)


def jsonable(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def rational(value):
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    return Fraction(value)


def make_case(family, n, seed, scale=1):
    if family not in FAMILIES or type(n) is not int or not 1 <= n <= 20:
        raise ValueError("unknown family or order outside 1..20")
    if type(seed) is not int or type(scale) is not int or scale < 1:
        raise ValueError("seed and positive weight scale must be integers")
    # Scale is deliberately excluded: scaling comparisons use IDENTICAL graphs.
    stream = int(reference.digest([SCHEMA, family, n, seed]), 16)
    rng = random.Random(stream)
    types = [[rng.randint(1, 9) for _ in range(n)] for _ in range(min(2, n))]
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            present = (family in ("dense", "repeated_rows") or
                       family == "sparse" and (i == j or rng.random() < .2) or
                       family == "banded" and abs(i-j) <= 1 or
                       family == "diagonal" and i == j or
                       family == "block" and i//3 == j//3)
            value = types[i % len(types)][j] if family == "repeated_rows" else rng.randint(1, 9)
            row.append(scale * value if present else 0)
        matrix.append(row)
    return {"family": family, "n": n, "seed": seed, "scale": scale,
            "matrix": matrix, "matrix_sha256": reference.digest(matrix)}


def reference_case(case):
    return {"contract": "rook", "n": case["n"], "linear": case["matrix"],
            "constant": [[1] * case["n"] for _ in range(case["n"])]}


def load_previous(path):
    spec = importlib.util.spec_from_file_location("weighted_rook_previous", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_method(request):
    method = request["method"]
    if method == "native":
        return importlib.import_module("perm_polynomial")
    if method == "previous_native":
        return load_previous(request["previous_source"])
    if method in ("compiled_dp", "polynomial_ryser"):
        return reference.Baselines(request["library"])
    if method == "frontier":
        from notes.benchmarks.weighted_matching_frontier import weighted_matching_frontier
        return weighted_matching_frontier
    raise ValueError("unknown benchmark method")


def evaluate_method(request, implementation):
    method, case = request["method"], request["case"]
    if method in ("native", "previous_native"):
        with reference.capture_public_residues(implementation) as captures:
            values = implementation.weighted_rook_polynomial(
                case["matrix"], backend="native", num_threads=1)
        # Component dispatch yields shorter residue vectors. Keep each intact;
        # these are component transcripts, not residues of the full polynomial.
        result = {"coefficients": list(values), "component_residue_calls": captures}
        if hasattr(implementation, "_weighted_rook_components_validated"):
            components = implementation._weighted_rook_components_validated(case["matrix"])
            result["support_components"] = [[list(r), list(c)] for r, c in components]
        return result
    if method == "frontier":
        result = implementation(case["matrix"], max_states=request["max_states"])
        return jsonable(result)
    problem = reference_case(case)
    bounds = reference.coefficient_bounds(problem)
    primes = reference.plan_primes(max(bounds), False)
    result = reference.method_result(
        "rook_dp" if method == "compiled_dp" else method,
        problem, primes, bounds, False, implementation)
    result.update(bounds=bounds, primes=primes)
    return result


def benchmark_worker(request):
    from notes.benchmarks.weighted_matching_frontier import FrontierStateLimitError
    implementation = prepare_method(request)
    baseline_rss = peak_rss_bytes()
    samples, expected = [], None
    for index in range(request["warmups"] + request["repeats"]):
        start, cpu_start = time.perf_counter(), time.process_time()
        try:
            value = evaluate_method(request, implementation)
        except FrontierStateLimitError as error:
            return {"status": "STATE_LIMIT", "method": request["method"],
                    "error": str(error), "limit": request["max_states"],
                    "elapsed_seconds": time.perf_counter()-start,
                    "baseline_peak_rss_bytes": baseline_rss,
                    "peak_process_rss_bytes": peak_rss_bytes()}
        seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu_start
        reference.require_vector(value["coefficients"], request["case"]["n"]+1)
        if expected is None:
            expected = value["coefficients"]
        reference.require_match(value["coefficients"], expected, "within-worker repeats")
        samples.append({"phase": "warmup" if index < request["warmups"] else "measurement",
                        "seconds": seconds, "cpu_seconds": cpu_seconds, "result": value})
    return {"status": "COMPLETE", "method": request["method"], "samples": samples,
            "coefficients": expected,
            "median_seconds": statistics.median(s["seconds"] for s in samples if s["phase"] == "measurement"),
            "baseline_peak_rss_bytes": baseline_rss,
            "peak_process_rss_bytes": peak_rss_bytes()}


def timed(repeats, function):
    samples, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter()-start)
    return result, {"samples": samples, "median_seconds": statistics.median(samples)}


def application_worker(request):
    from notes.benchmarks.weighted_matching_statistics import (
        exact_occupancy_statistics, independent_edge_statistics,
        belief_propagation_statistics, total_variation_distance)
    from notes.benchmarks.weighted_matching_partition import PartitionBaseline
    baseline = PartitionBaseline(request["partition_library"])
    matrix, coefficients = request["case"]["matrix"], request["coefficients"]
    points = []
    for text in request["activities"]:
        activity = Fraction(text)
        exact, exact_timing = timed(request["repeats"], lambda: exact_occupancy_statistics(coefficients, activity))
        direct, direct_timing = timed(request["repeats"], lambda: baseline.evaluate(matrix, activity))
        for key in ("partition", "mean", "variance"):
            if exact[key] != direct[key]:
                raise ValueError(f"direct scalar query disagrees on {key} at activity {text}")
        bp, bp_timing = timed(request["repeats"], lambda: belief_propagation_statistics(matrix, activity))
        independent = independent_edge_statistics(matrix, activity)
        point = {"activity": text, "exact": jsonable(exact), "direct_query": jsonable(direct),
                 "bp": jsonable(bp), "independent_edges": jsonable(independent),
                 "independent_edges_tv": total_variation_distance(exact["probabilities"], independent["probabilities"]),
                 "timing": {"exact_statistics": exact_timing, "direct_query": direct_timing, "bp": bp_timing}}
        if bp["converged"]:
            point["bp_mean_absolute_error"] = abs(bp["mean"]-float(exact["mean"]))
            point["bp_log_partition_error"] = bp["log_partition"]-exact["log_partition"]
            point["bp_partition_relative_error"] = math.expm1(point["bp_log_partition_error"])
        points.append(point)
    return {"status": "COMPLETE", "points": points,
            "peak_process_rss_bytes": peak_rss_bytes()}


def execute_worker(request, timeout, cpu):
    validate_cpu(cpu)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    request = {**request, "cpu": cpu}
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(request), text=True, capture_output=True, env=environment,
            cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return {"status": "TIMEOUT", "timeout_seconds": timeout,
                "process_wall_seconds": time.perf_counter()-started,
                "stdout": str(error.stdout or ""), "stderr": str(error.stderr or "")}
    if process.returncode:
        return {"status": "ERROR", "returncode": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr}
    result = json.loads(process.stdout)
    result["process_wall_seconds"] = time.perf_counter()-started
    result["stderr"] = process.stderr
    return result


def validate_benchmark(record, repeats, warmups):
    case = record["case"]
    if case != make_case(case["family"], case["n"], case["seed"], case["scale"]):
        raise ValueError("case generation/hash mismatch")
    required = record.get("required_methods", list(METHODS))
    if (not isinstance(required, list) or len(required) != len(set(required)) or
        any(method not in (*METHODS, "previous_native") for method in required)):
        raise ValueError("invalid required-method inventory")
    if record.get("complete") and not set(required).issubset(record["methods"]):
        raise ValueError("complete case is missing a required method")
    expected = None
    for method, result in record["methods"].items():
        if method not in (*METHODS, "previous_native"):
            raise ValueError("unknown recorded method")
        if result["status"] != "COMPLETE":
            if method != "frontier" or result["status"] not in ("STATE_LIMIT", "TIMEOUT"):
                raise ValueError(f"failed method {method}: {result}")
            continue
        if result["method"] != method:
            raise ValueError("method label mismatch")
        for field in ("baseline_peak_rss_bytes", "peak_process_rss_bytes"):
            if type(result[field]) is not int or result[field] < 0:
                raise ValueError("invalid process RSS")
        if result["peak_process_rss_bytes"] < result["baseline_peak_rss_bytes"]:
            raise ValueError("process RSS high-water mark decreased")
        if len(result["samples"]) != repeats+warmups:
            raise ValueError("wrong repeat count")
        reference.require_vector(result["coefficients"], case["n"]+1)
        if expected is None:
            expected = result["coefficients"]
        reference.require_match(result["coefficients"], expected, "independent method agreement")
        for i, sample in enumerate(result["samples"]):
            phase = "warmup" if i < warmups else "measurement"
            if sample["phase"] != phase or not valid_time(sample["seconds"]):
                raise ValueError("invalid timing sample")
            if not valid_time(sample["cpu_seconds"]):
                raise ValueError("invalid CPU timing")
            item = sample["result"]
            reference.require_vector(item["coefficients"], case["n"]+1)
            reference.require_match(item["coefficients"], expected, "stored sample agreement")
            if method in ("compiled_dp", "polynomial_ryser"):
                problem = reference_case(case)
                bounds = reference.coefficient_bounds(problem)
                primes = reference.plan_primes(max(bounds), False)
                if item["bounds"] != bounds or item["primes"] != primes:
                    raise ValueError("stored CRT planning changed")
                reference.validate_result(item, problem, primes, bounds, False)
            elif method in ("native", "previous_native"):
                reference.validate_public_residues(
                    item["coefficients"], item["component_residue_calls"], reference_case(case))
        median = statistics.median(s["seconds"] for s in result["samples"] if s["phase"] == "measurement")
        if median != result["median_seconds"]:
            raise ValueError("median disagrees with raw samples")
    if "coefficients" in record:
        reference.require_vector(record["coefficients"], case["n"]+1)
        reference.require_match(record["coefficients"], expected, "case coefficient summary")
    return expected


def valid_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_application(curve, benchmark, repeats, *, partition_library_sha256=None):
    """Replay exact probability and scalar-CRT arithmetic, not expensive kernels."""
    from notes.benchmarks.weighted_matching_statistics import (
        exact_occupancy_statistics, independent_edge_statistics,
        belief_propagation_statistics, total_variation_distance)
    case = benchmark["case"]
    if curve["status"] != "COMPLETE" or curve["case"] != case:
        raise ValueError("incomplete or mismatched application")
    reference.require_match(curve["coefficients"], benchmark["coefficients"], "application coefficients")
    if [point["activity"] for point in curve["points"]] != list(ACTIVITIES):
        raise ValueError("application activity inventory changed")
    if curve["native_setup_seconds"] != benchmark["methods"]["native"]["median_seconds"]:
        raise ValueError("application setup time changed")
    for point in curve["points"]:
        activity = Fraction(point["activity"])
        exact = exact_occupancy_statistics(curve["coefficients"], activity)
        if point["exact"] != jsonable(exact):
            raise ValueError("saved exact occupancy statistics do not replay")
        direct = point["direct_query"]
        p, q = activity.numerator, activity.denominator
        bound = math.prod(q+p*sum(row) for row in case["matrix"])
        bounds = [bound, case["n"]*bound, case["n"]**2*bound]
        primes = reference.plan_primes(max(bounds), False)
        if (direct["status"] != "COMPLETE" or direct["exact"] is not True or
            direct["order"] != case["n"] or rational(direct["activity"]) != activity or
            direct["modulus"] != math.prod(primes)):
            raise ValueError("direct-query metadata changed")
        if partition_library_sha256 and direct["library_sha256"] != partition_library_sha256:
            raise ValueError("direct-query compiled-library link changed")
        if direct["bounds"] != bounds or direct["primes"] != primes:
            raise ValueError("direct-query CRT planning changed")
        moments = reference.crt_vectors(direct["residue_rows"], primes, bounds, False)
        if moments != direct["numerator_moments"] or direct["denominator"] != q**case["n"]:
            raise ValueError("direct-query CRT reconstruction changed")
        f0, f1, f2 = moments
        replay = {"partition": Fraction(f0, q**case["n"]), "mean": Fraction(f1, f0),
                  "variance": Fraction(f2, f0)-Fraction(f1, f0)**2}
        for key, value in replay.items():
            if rational(direct[key]) != value or value != exact[key]:
                raise ValueError("direct-query statistics disagree")
        for timing in point["timing"].values():
            if len(timing["samples"]) != repeats or any(not valid_time(t) for t in timing["samples"]):
                raise ValueError("invalid query timing samples")
            if statistics.median(timing["samples"]) != timing["median_seconds"]:
                raise ValueError("query median changed")
        bp = point["bp"]
        if (bp["damping"], bp["tolerance"], bp["max_iterations"]) != (0.5, 1e-10, 10000):
            raise ValueError("BP solver parameters changed")
        if bp != jsonable(belief_propagation_statistics(case["matrix"], activity)):
            raise ValueError("BP iteration/output does not replay")
        independent = independent_edge_statistics(case["matrix"], activity)
        if point["independent_edges"] != jsonable(independent):
            raise ValueError("independent-edge diagnostic does not replay")
        tv = total_variation_distance(exact["probabilities"], independent["probabilities"])
        if point["independent_edges_tv"] != tv:
            raise ValueError("independent-edge TV changed")
        if bp["converged"]:
            if any(not math.isfinite(bp[key]) for key in ("mean", "log_partition", "residual")):
                raise ValueError("nonfinite converged BP result")
            log_error = bp["log_partition"]-exact["log_partition"]
            if (point["bp_log_partition_error"] != log_error or
                point["bp_partition_relative_error"] != math.expm1(log_error) or
                point["bp_mean_absolute_error"] != abs(bp["mean"]-float(exact["mean"]))):
                raise ValueError("BP errors disagree with saved outputs")
        elif bp["mean"] is not None or bp["log_partition"] is not None:
            raise ValueError("nonconverged BP exposes an accepted estimate")


def load_or_build_partition(output):
    from notes.benchmarks.weighted_matching_partition import build_partition_baseline
    path = output/"build.json"
    if not path.exists():
        return build_partition_baseline(output)
    record = json.loads(path.read_text())
    checks = ((record["library"], record["library_sha256"]),
              (record["source"], record["source_sha256"]),
              (ROOT/"notes/benchmarks/weighted_matching_partition.py", record["wrapper_sha256"]),
              (ROOT/"notes/benchmarks/polynomial_compare.py", record["crt_helper_sha256"]),
              (record["compiler"], record["compiler_sha256"]),
              (record["stdout_log"], record["stdout_sha256"]),
              (record["stderr_log"], record["stderr_sha256"]))
    if any(reference.file_digest(file) != expected for file, expected in checks):
        raise ValueError("partition baseline build changed; choose a new output directory")
    return record


def source_identity():
    sources = {name: reference.file_digest(ROOT/name) for name in SOURCES}
    binaries = {}
    for name in ("perm_polynomial_ext", "perm_c_ext"):
        module = importlib.import_module(name)
        binaries[name] = {"path": module.__file__, "sha256": reference.file_digest(module.__file__)}
    return {"sources": sources, "native_binaries": binaries}


def run(args):
    if any(n > 12 for n in args.sizes) and not args.allow_large:
        raise ValueError("orders above 12 require --allow-large")
    if any(not 1 <= n <= 20 for n in args.sizes) or min(args.scales) < 1:
        raise ValueError("invalid size/scale")
    if args.application_size not in args.sizes or 1 not in args.scales:
        raise ValueError("application size and scale 1 must occur in the sweep")
    if args.repeats < 1 or args.warmups < 0 or args.max_states < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("invalid repeat/resource limits")
    if any(len(values) != len(set(values)) for values in (args.sizes, args.seeds, args.scales, args.families)):
        raise ValueError("duplicate sweep dimension")
    validate_cpu(args.cpu)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output/".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    library, build = reference.build_baselines(output/"build")
    partition_build = load_or_build_partition(output/"partition-build")
    previous = None
    if args.previous_source:
        previous = output/"sources"/"previous_perm_polynomial.py"
        previous.parent.mkdir(parents=True, exist_ok=True)
        contents = Path(args.previous_source).read_bytes()
        if previous.exists() and previous.read_bytes() != contents:
            raise ValueError("prechange source differs on resume")
        previous.write_bytes(contents)
    config = {key: getattr(args, key) for key in ("sizes", "seeds", "scales", "families", "repeats", "warmups",
              "max_states", "timeout", "application_size", "cpu")}
    methods = list(METHODS) + (["previous_native"] if previous else [])
    identity = {"schema": SCHEMA, "configuration": config, "methods": methods,
                "activities": list(ACTIVITIES), **source_identity(),
                "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "affinity": available_affinity(),
                "previous_source_sha256": reference.file_digest(previous) if previous else None,
                "baseline_binary_sha256": reference.file_digest(library),
                "partition_binary_sha256": reference.file_digest(partition_build["library"])}
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        manifest = reference.read_record(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("source, binary, machine, or configuration changed; use a new output directory")
    else:
        manifest = {"identity": identity, "command": sys.argv, "build": build,
                    "partition_build": jsonable(partition_build),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "limitations": ["Sage unavailable; frontier is an independent Butera-Pernici-style implementation",
                        "Sequential isolated workers; affinity availability and requested CPU are recorded; external host load is not fully observed",
                        "Process peak RSS includes interpreter/imports and one method's repeats",
                        "Native and independent modular baselines use different certified CRT planners"]}
        for name in SOURCES:
            target = output/"sources"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        reference.atomic_record(manifest_path, manifest)
    manifest_sha = reference.digest(manifest)
    def check_live_inputs():
        if source_identity() != {"sources": identity["sources"], "native_binaries": identity["native_binaries"]}:
            raise ValueError("source/binary changed during the run")
        checks = [(library, identity["baseline_binary_sha256"]),
                  (partition_build["library"], identity["partition_binary_sha256"])]
        if previous:
            checks.append((previous, identity["previous_source_sha256"]))
        if any(reference.file_digest(file) != expected for file, expected in checks):
            raise ValueError("baseline or prechange source changed during the run")
        if any(reference.file_digest(output/"sources"/name) != expected
               for name, expected in identity["sources"].items()):
            raise ValueError("archived source snapshot changed")
    check_live_inputs()
    case_paths = []
    for n in args.sizes:
        for family in args.families:
            for seed in args.seeds:
                for scale in args.scales:
                    case = make_case(family, n, seed, scale)
                    path = output/"cases"/(reference.digest(case)+".json")
                    case_paths.append(path)
                    record = reference.read_record(path) if path.exists() else {
                        "manifest_sha256": manifest_sha, "case": case, "methods": {},
                        "required_methods": methods, "complete": False}
                    if record["manifest_sha256"] != manifest_sha:
                        raise ValueError("checkpoint belongs to a different manifest")
                    if record["case"] != case:
                        raise ValueError("checkpoint case does not match its expected path")
                    if record.get("required_methods", methods) != methods:
                        raise ValueError("checkpoint method inventory differs from manifest")
                    validate_benchmark(record, args.repeats, args.warmups)
                    offset = int(reference.digest([family, n, seed]), 16) % len(methods)
                    order = methods[offset:]+methods[:offset]
                    if seed % 2:
                        order = list(reversed(order))
                    for method in order:
                        if method in record["methods"]:
                            continue
                        check_live_inputs()
                        request = {"kind": "benchmark", "case": case, "method": method,
                            "library": str(library), "previous_source": str(previous) if previous else None,
                            "repeats": args.repeats, "warmups": args.warmups, "max_states": args.max_states}
                        record["methods"][method] = execute_worker(request, args.timeout, args.cpu)
                        reference.atomic_record(path, record)
                        check_live_inputs()
                        validate_benchmark(record, args.repeats, args.warmups)
                    record["coefficients"] = validate_benchmark(record, args.repeats, args.warmups)
                    record["complete"] = True
                    reference.atomic_record(path, record)
                    print(f"COMPLETE n={n} {family} seed={seed} scale={scale}", flush=True)
    curves = []
    for path in case_paths:
        record = reference.read_record(path)
        case = record["case"]
        if case["n"] != args.application_size or case["scale"] != 1:
            continue
        curve_path = output/"applications"/path.name
        if curve_path.exists():
            curve = reference.read_record(curve_path)
            if curve["manifest_sha256"] != manifest_sha or curve["case"] != case:
                raise ValueError("application checkpoint identity mismatch")
        else:
            check_live_inputs()
            request = {"kind": "application", "case": case, "coefficients": record["coefficients"],
                       "partition_library": partition_build["library"], "activities": list(ACTIVITIES),
                       "repeats": args.repeats}
            result = execute_worker(request, max(args.timeout, 300), args.cpu)
            curve = {"manifest_sha256": manifest_sha, "case": case,
                     "coefficients": record["coefficients"], "native_setup_seconds": record["methods"]["native"]["median_seconds"],
                     **result}
            reference.atomic_record(curve_path, curve)
            check_live_inputs()
        if curve["status"] != "COMPLETE":
            raise ValueError(f"application worker failed: {curve}")
        validate_application(curve, record, args.repeats,
                             partition_library_sha256=identity["partition_binary_sha256"])
        curves.append(str(curve_path.relative_to(output)))
        print(f"APPLICATION {case['family']} seed={case['seed']}", flush=True)
    # Scaling is a mathematical control, verified across every retained vector.
    indexed = {(r["case"]["family"], r["case"]["n"], r["case"]["seed"], r["case"]["scale"]): r
               for r in (reference.read_record(p) for p in case_paths)}
    for (family, n, seed, scale), record in indexed.items():
        base = indexed[family, n, seed, 1]["coefficients"]
        reference.require_match(record["coefficients"], [x*scale**k for k, x in enumerate(base)], "controlled weight rescaling")
    summary = {"status": "COMPLETE", "manifest_sha256": manifest_sha,
               "case_paths": [str(p.relative_to(output)) for p in case_paths],
               "application_paths": curves, "case_count": len(case_paths), "application_count": len(curves),
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    check_live_inputs()
    reference.atomic_record(output/"completion.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    if sys.argv[1:] == ["--worker"]:
        request = json.load(sys.stdin)
        affinity = configure_cpu(request["cpu"])
        result = benchmark_worker(request) if request["kind"] == "benchmark" else application_worker(request)
        result["cpu_affinity"] = affinity
        print(json.dumps(jsonable(result), allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026090701, 2026090702, 2026090703])
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 100])
    parser.add_argument("--families", choices=FAMILIES, nargs="+", default=list(FAMILIES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--application-size", type=int, default=12)
    parser.add_argument("--cpu", type=int, default=default_cpu(),
                        help="pin workers to this allowed CPU (default: last allowed CPU when supported, otherwise unpinned)")
    parser.add_argument("--previous-source")
    parser.add_argument("--allow-large", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
