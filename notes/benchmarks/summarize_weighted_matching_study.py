#!/usr/bin/env python3
"""Audit a complete weighted-matching study and export tables and static figures.

No native kernel, compiler, or GPU is invoked. Recorded exact CRT arithmetic,
statistics and BP are replayed on the CPU. Hashes link the retained files; they
are provenance checks, not independently authenticated timing certificates.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from notes.benchmarks import polynomial_compare as reference
from notes.benchmarks import weighted_matching_study as study

SCHEMA = "weighted-matching-summary-v1"
METHOD_LABELS = {"native": "Current native", "previous_native": "Previous native",
                 "compiled_dp": "Compiled subset DP", "polynomial_ryser": "Polynomial Ryser",
                 "frontier": "Frontier DP"}


def checked_file(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or reference.file_digest(path) != expected:
        raise ValueError(f"file digest or regular-file mismatch: {path}")


def read_record(path, files, root):
    payload = reference.read_record(path)
    files[str(Path(path).relative_to(root))] = reference.file_digest(path)
    return payload


def audit(run):
    """Return audited manifest, cases, curves and exact input-file hash inventory.

    Current sources and every retained binary must match the recorded pins.
    Frozen Python sources are never executed. Numerical replay uses this checkout.
    """
    run = Path(run).resolve(strict=True)
    files = {}
    manifest = read_record(run/"manifest.json", files, run)
    identity = manifest["identity"]
    if identity["schema"] != study.SCHEMA:
        raise ValueError("unknown study schema")
    config = identity["configuration"]
    for key in ("sizes", "seeds", "scales", "families"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"empty or duplicate configuration: {key}")
    if (config["application_size"] not in config["sizes"] or 1 not in config["scales"] or
            type(config["repeats"]) is not int or config["repeats"] < 1 or
            type(config["warmups"]) is not int or config["warmups"] < 0):
        raise ValueError("invalid activity/repeat configuration")
    if identity["activities"] != list(study.ACTIVITIES):
        raise ValueError("activity protocol changed")
    methods = list(study.METHODS) + (["previous_native"] if identity["previous_source_sha256"] else [])
    if identity["methods"] != methods:
        raise ValueError("method inventory changed")
    required = set(study.SOURCES)
    if set(identity["sources"]) != required:
        raise ValueError("source inventory changed")
    for name, expected in identity["sources"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or name not in required:
            raise ValueError("unexpected source path")
        checked_file(run/"sources"/name, expected)
        files["sources/"+name] = expected
        checked_file(ROOT/name, expected)
    previous = identity["previous_source_sha256"]
    if previous:
        checked_file(run/"sources/previous_perm_polynomial.py", previous)
        files["sources/previous_perm_polynomial.py"] = previous
    if set(identity["native_binaries"]) != {"perm_polynomial_ext", "perm_c_ext"}:
        raise ValueError("native binary inventory changed")
    for binary in identity["native_binaries"].values():
        checked_file(binary["path"], binary["sha256"])
    build = read_record(run/"build/build.json", files, run)
    if build != manifest["build"] or build["binary_sha256"] != identity["baseline_binary_sha256"]:
        raise ValueError("baseline build linkage changed")
    if build["identity"]["source_sha256"] != identity["sources"]["notes/benchmarks/polynomial_baselines.cpp"]:
        raise ValueError("baseline source linkage changed")
    checked_file(run/"build/polynomial_baselines.so", identity["baseline_binary_sha256"])
    files["build/polynomial_baselines.so"] = identity["baseline_binary_sha256"]
    partition_path = run/"partition-build/build.json"
    partition = json.loads(partition_path.read_text())
    files["partition-build/build.json"] = reference.file_digest(partition_path)
    if partition != manifest["partition_build"] or partition["library_sha256"] != identity["partition_binary_sha256"]:
        raise ValueError("partition build linkage changed")
    for field, source in (("source_sha256", "weighted_matching_partition.cpp"),
                          ("wrapper_sha256", "weighted_matching_partition.py"),
                          ("crt_helper_sha256", "polynomial_compare.py")):
        if partition[field] != identity["sources"]["notes/benchmarks/"+source]:
            raise ValueError("partition build source linkage changed")
    for relative, expected in (("partition-build/weighted_matching_partition.so", identity["partition_binary_sha256"]),
                               ("partition-build/compile.stdout.log", partition["stdout_sha256"]),
                               ("partition-build/compile.stderr.log", partition["stderr_sha256"])):
        checked_file(run/relative, expected)
        files[relative] = expected
    manifest_sha = reference.digest(manifest)
    completion = read_record(run/"completion.json", files, run)
    if completion["status"] != "COMPLETE" or completion["manifest_sha256"] != manifest_sha:
        raise ValueError("study is incomplete or completion belongs to another manifest")
    cases, curves, case_paths, curve_paths = [], [], [], []
    indexed = {}
    for n in config["sizes"]:
        for family in config["families"]:
            for seed in config["seeds"]:
                for scale in config["scales"]:
                    case = study.make_case(family, n, seed, scale)
                    filename = reference.digest(case)+".json"
                    relative = "cases/"+filename
                    record = read_record(run/relative, files, run)
                    if (record["case"] != case or record["manifest_sha256"] != manifest_sha or
                            record["complete"] is not True or set(record["methods"]) != set(methods)):
                        raise ValueError("incomplete or mismatched benchmark case")
                    study.validate_benchmark(record, config["repeats"], config["warmups"])
                    for method, result in record["methods"].items():
                        if result["status"] == "STATE_LIMIT" and result["limit"] != config["max_states"]:
                            raise ValueError("frontier state limit changed")
                        if result["status"] == "TIMEOUT" and result["timeout_seconds"] != config["timeout"]:
                            raise ValueError("frontier timeout changed")
                    cases.append(record)
                    case_paths.append(relative)
                    indexed[family, n, seed, scale] = record
                    if n != config["application_size"] or scale != 1:
                        continue
                    relative = "applications/"+filename
                    curve = read_record(run/relative, files, run)
                    if curve["manifest_sha256"] != manifest_sha:
                        raise ValueError("application manifest linkage changed")
                    study.validate_application(curve, record, config["repeats"],
                        partition_library_sha256=identity["partition_binary_sha256"])
                    for point in curve["points"]:
                        if set(point["timing"]) != {"exact_statistics", "direct_query", "bp"}:
                            raise ValueError("query timing inventory changed")
                    curves.append(curve)
                    curve_paths.append(relative)
    if (completion["case_paths"] != case_paths or completion["application_paths"] != curve_paths or
            completion["case_count"] != len(cases) or completion["application_count"] != len(curves)):
        raise ValueError("completion inventory differs from the configured full sweep")
    for (family, n, seed, scale), record in indexed.items():
        base = indexed[family, n, seed, 1]["coefficients"]
        if record["coefficients"] != [v*scale**k for k, v in enumerate(base)]:
            raise ValueError("controlled weight scaling identity failed")
    return {"manifest": manifest, "completion": completion, "cases": cases, "curves": curves,
            "input_sha256": files, "current_sources_checked": True,
            "native_binaries": identity["native_binaries"]}


def case_columns(case):
    return {key: case[key] for key in ("family", "n", "seed", "scale", "matrix_sha256")}


def timing_rows(cases):
    rows = []
    for case in cases:
        native = case["methods"]["native"]["median_seconds"]
        for method, result in case["methods"].items():
            complete = result["status"] == "COMPLETE"
            times = [s["seconds"] for s in result.get("samples", []) if s["phase"] == "measurement"]
            seconds = result.get("median_seconds")
            rows.append({**case_columns(case["case"]), "method": method, "status": result["status"],
                "median_seconds": seconds, "minimum_seconds": min(times) if times else None,
                "maximum_seconds": max(times) if times else None,
                "measurement_seconds": json.dumps(times),
                "warmup_seconds": json.dumps([s["seconds"] for s in result.get("samples", []) if s["phase"] == "warmup"]),
                "median_cpu_seconds": statistics.median(s["cpu_seconds"] for s in result["samples"] if s["phase"] == "measurement") if complete else None,
                "process_wall_seconds": result.get("process_wall_seconds"),
                "peak_process_rss_bytes": result.get("peak_process_rss_bytes"),
                "baseline_peak_rss_bytes": result.get("baseline_peak_rss_bytes"),
                "state_limit": result.get("limit"), "timeout_seconds": result.get("timeout_seconds"),
                "native_speedup_vs_method": seconds/native if complete and native > 0 else None,
                "native_comparison": ("self" if method == "native" else
                    ("win" if native < seconds else "loss" if native > seconds else "tie")) if complete else "censored",
                "error": result.get("error", "")})
    return rows


def accuracy_rows(curves):
    rows = []
    for curve in curves:
        cumulative_coefficient = curve["native_setup_seconds"]
        cumulative_direct = 0.
        for activity_index, point in enumerate(curve["points"], 1):
            cumulative_coefficient += point["timing"]["exact_statistics"]["median_seconds"]
            cumulative_direct += point["timing"]["direct_query"]["median_seconds"]
            exact, bp, independent = point["exact"], point["bp"], point["independent_edges"]
            mean = float(study.rational(exact["mean"]))
            row = {**case_columns(curve["case"]), "activity_index": activity_index,
                "activity": point["activity"], "activity_float": float(Fraction(point["activity"])),
                "cumulative_coefficient_seconds": cumulative_coefficient,
                "cumulative_direct_seconds": cumulative_direct,
                "cumulative_direct_over_coefficient_ratio": cumulative_direct/cumulative_coefficient if cumulative_coefficient else None,
                **{"exact_"+key: str(study.rational(exact[key])) for key in
                   ("partition", "mean", "variance", "empty_probability", "max_size_probability")},
                "exact_mean_float": mean, "exact_log_partition": exact["log_partition"],
                "exact_probabilities": json.dumps([str(study.rational(p)) for p in exact["probabilities"]]),
                "bp_status": bp["status"], "bp_iterations": bp["iterations"], "bp_residual": bp["residual"],
                "bp_mean": bp["mean"], "bp_log_partition": bp["log_partition"],
                "bp_mean_signed_error": bp["mean"]-mean if bp["converged"] else None,
                "bp_mean_absolute_error": point.get("bp_mean_absolute_error"),
                "bp_mean_relative_error": (bp["mean"]-mean)/mean if bp["converged"] and mean else None,
                "bp_log_partition_error": point.get("bp_log_partition_error"),
                "bp_partition_relative_error": point.get("bp_partition_relative_error"),
                "independent_edges_log_partition": independent["log_partition"],
                "independent_edges_mean": independent["mean"],
                "independent_edges_invalid_size_probability": independent["invalid_size_probability"],
                "independent_edges_cardinality_tv": point["independent_edges_tv"],
                **{key+"_seconds": value["median_seconds"] for key, value in point["timing"].items()}}
            rows.append(row)
    return rows


def break_even_row(curve):
    """Use arithmetic means of the per-activity measurement medians.

    The threshold models queries uniformly distributed over the frozen grid.
    The full-grid comparison sums those medians; it is not a measured batch run.
    """
    query = [p["timing"]["exact_statistics"]["median_seconds"] for p in curve["points"]]
    direct = [p["timing"]["direct_query"]["median_seconds"] for p in curve["points"]]
    setup = curve["native_setup_seconds"]
    query_mean, direct_mean = statistics.mean(query), statistics.mean(direct)
    saving = direct_mean-query_mean
    threshold = max(1, math.ceil(setup/saving)) if saving > 0 else None
    prefix_saving = [sum(direct[:i])-sum(query[:i])-setup for i in range(1, len(query)+1)]
    return {**case_columns(curve["case"]), "query_count": len(query), "native_setup_seconds": setup,
            "mean_coefficient_query_seconds": query_mean, "mean_direct_query_seconds": direct_mean,
            "estimated_break_even_queries": threshold,
            "grid_coefficient_total_seconds": setup+sum(query), "grid_direct_total_seconds": sum(direct),
            "grid_direct_over_coefficient_ratio": sum(direct)/(setup+sum(query)) if setup+sum(query) else None,
            "first_observed_grid_prefix_break_even": next((i for i, value in enumerate(prefix_saving, 1) if value >= 0), None),
            "grid_coefficient_faster": setup+sum(query) < sum(direct)}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def occupancy_families(curves):
    available = list(dict.fromkeys(c["case"]["family"] for c in curves))
    chosen = [name for name in ("dense", "banded", "diagonal") if name in available]
    return (chosen + [name for name in available if name not in chosen])[:3]


def figures(output, cases, curves):
    # Keep font-cache writes out of a read-only HOME and the exported artifacts.
    with tempfile.TemporaryDirectory(prefix="weighted-matching-mpl-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            _figures(output, cases, curves)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def _figures(output, cases, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = list(dict.fromkeys(c["case"]["family"] for c in cases))
    methods = list(cases[0]["methods"])
    colors = {method: plt.get_cmap("tab10")(i) for i, method in enumerate(sorted(methods))}
    fig, axes = plt.subplots(math.ceil(len(families)/2), 2, figsize=(12, 3.2*math.ceil(len(families)/2)), squeeze=False)
    for ax, family in zip(axes.flat, families):
        group = [c for c in cases if c["case"]["family"] == family]
        scales = sorted({c["case"]["scale"] for c in group})
        for method in methods:
            for scale_i, scale in enumerate(scales):
                points = []
                for n in sorted({c["case"]["n"] for c in group}):
                    selected = [c["methods"][method] for c in group if c["case"]["n"] == n and c["case"]["scale"] == scale]
                    values = [r["median_seconds"] for r in selected if r["status"] == "COMPLETE"]
                    if values:
                        points.append((n, statistics.median(values)))
                if points:
                    ax.plot(*zip(*points), marker="o", linestyle="-" if scale_i == 0 else "--",
                        color=colors[method], label=f"{METHOD_LABELS[method]}, ×{scale}", markersize=3)
        ax.set(title=family.replace("_", " "), xlabel="Matrix order", ylabel="Median wall time (s)", yscale="log")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(families):]:
        ax.set_visible(False)
    handles, labels = [], []
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels()
        for item, label in zip(h, lab):
            if label not in labels:
                handles.append(item); labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Coefficient setup: median of per-seed medians; capped cases omitted only from lines", y=1.01, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .90))
    save_figure(fig, output, "method-timings", plt)

    families = occupancy_families(curves)
    fig, axes = plt.subplots(len(families), 2, figsize=(12, 3*len(families)), squeeze=False)
    for row, family in enumerate(families):
        selected = [c for c in curves if c["case"]["family"] == family]
        for index, curve in enumerate(selected):
            point = next(p for p in curve["points"] if p["activity"] == "1")
            probs = [float(study.rational(p)) for p in point["exact"]["probabilities"]]
            axes[row, 0].plot(range(len(probs)), probs, marker="o", markersize=3, label=f"seed {curve['case']['seed']}")
            activities = [float(Fraction(p["activity"])) for p in curve["points"]]
            exact = [float(study.rational(p["exact"]["mean"])) for p in curve["points"]]
            bp = [p["bp"]["mean"] if p["bp"]["converged"] else math.nan for p in curve["points"]]
            axes[row, 1].plot(activities, exact, color="black", alpha=.6, label="Exact" if index == 0 else None)
            axes[row, 1].plot(activities, bp, "--", color="tab:orange", alpha=.8, label="BP" if index == 0 else None)
        axes[row, 0].set(title=f"{family.replace('_', ' ')}: exact size distribution, activity 1", xlabel="Occupied edges k", ylabel="Probability")
        axes[row, 1].set(title="Mean occupancy, all retained seeds", xlabel="Activity", ylabel="Mean occupied edges", xscale="log")
        for ax in axes[row]:
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output, "occupancy", plt)

    rows = accuracy_rows(curves)
    families = list(dict.fromkeys(c["case"]["family"] for c in curves))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for family in families:
        selected = [r for r in rows if r["family"] == family and r["bp_status"] == "CONVERGED"]
        x = [r["activity_float"] for r in selected]
        axes[0].scatter(x, [abs(r["bp_partition_relative_error"]) for r in selected], s=13, alpha=.7, label=family.replace("_", " "))
        axes[1].scatter(x, [r["bp_mean_absolute_error"] for r in selected], s=13, alpha=.7)
    error_values = ([abs(r["bp_partition_relative_error"]) for r in rows if r["bp_status"] == "CONVERGED"],
                    [r["bp_mean_absolute_error"] for r in rows if r["bp_status"] == "CONVERGED"])
    for ax, values in zip(axes, error_values):
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_ylim(0, max(1e-12, 1.5*max(values, default=0)))
        ax.set_xlabel("Activity"); ax.grid(alpha=.2)
    axes[0].set(title="Partition-function accuracy", ylabel="Absolute relative error |Z_BP / Z − 1|")
    axes[1].set(title="Mean-occupancy accuracy", ylabel="Absolute error |E_BP[k] − E[k]|")
    axes[0].legend(fontsize=8)
    fig.suptitle("Different observables have different errors; all seeds shown")
    fig.tight_layout()
    save_figure(fig, output, "bp-errors", plt)


def save_figure(fig, output, name, plt):
    fig.savefig(output/(name+".png"), dpi=170, bbox_inches="tight")
    fig.savefig(output/(name+".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def export(run, output, *, make_figures=True):
    audited = audit(run)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("summary output must be a new directory")
    output.mkdir(parents=True)
    cases, curves = audited["cases"], audited["curves"]
    timings, accuracy = timing_rows(cases), accuracy_rows(curves)
    break_even = [break_even_row(curve) for curve in curves]
    write_csv(output/"timings.csv", timings)
    write_csv(output/"accuracy.csv", accuracy)
    write_csv(output/"break-even.csv", break_even)
    comparisons = {method: {label: sum(r["native_comparison"] == label for r in timings if r["method"] == method)
                           for label in ("win", "loss", "tie", "censored")}
                   for method in audited["manifest"]["identity"]["methods"] if method != "native"}
    converged = [r for r in accuracy if r["bp_status"] == "CONVERGED"]
    report = {"schema": SCHEMA, "status": "AUDITED_COMPLETE", "input": str(Path(run).resolve()),
        "current_sources_checked": True, "binary_bytes_checked": True,
        "exporter_sha256": reference.file_digest(__file__), "input_sha256": audited["input_sha256"],
        "native_binaries": audited["native_binaries"], "configuration": audited["manifest"]["identity"]["configuration"],
        "case_count": len(cases), "curve_count": len(curves), "activity_point_count": len(accuracy),
        "native_comparisons": comparisons, "bp_converged_points": len(converged),
        "bp_nonconverged_points": len(accuracy)-len(converged),
        "bp_max_absolute_partition_relative_error": max((abs(r["bp_partition_relative_error"]) for r in converged), default=None),
        "bp_max_mean_absolute_error": max((r["bp_mean_absolute_error"] for r in converged), default=None),
        "coefficient_faster_full_grid_cases": sum(r["grid_coefficient_faster"] for r in break_even),
        "break_even": break_even}
    if make_figures:
        figures(output, cases, curves)
    lines = ["# Weighted-matching study: audited results", "",
        f"{len(cases)} coefficient cases, {len(curves)} application curves and {len(accuracy)} activity points passed replay.", "",
        "Hashes, generated inputs, complete inventories, modular reconstruction, coefficient agreement, exact query moments, BP and independent-edge calculations were checked. No expensive native kernel was rerun. "
        +"Current source and retained binary bytes match their recorded pins.", "",
        "## Timing results", "", "Current-native comparisons use the same matrix, seed and weight scale. A win means its recorded median is smaller; it is not a statistical significance claim. Capped frontier runs remain censored, with no inferred speed ratio.", "",
        "| Comparator | Native wins | Native losses | Ties | Censored |", "|---|---:|---:|---:|---:|"]
    for method, counts in comparisons.items():
        lines.append(f"| {METHOD_LABELS[method]} | {counts['win']} | {counts['loss']} | {counts['tie']} | {counts['censored']} |")
    lines += ["", "[All method timings, raw measurement arrays, memory and resource caps](timings.csv). The setup plot takes the median across per-seed medians; it does not pool raw samples or infer missing timings.", "",
        "## Application accuracy", "",
        f"BP converged at {len(converged)}/{len(accuracy)} points. Its maximum absolute relative partition error was {report['bp_max_absolute_partition_relative_error']}; its maximum absolute mean-occupancy error was {report['bp_max_mean_absolute_error']}. These extrema can occur at different points.", "",
        "Exact probabilities and moments use rational arithmetic; displayed logs, BP and independent-edge probabilities use binary64. BP fixed-point convergence is not an error certificate. The signed partition error in the CSV is Z_BP/Z−1; the figure uses its absolute value. Mean error is in occupied-edge units, not relative partition error. Nonconverged BP estimates are omitted from error plots and counted explicitly.", "",
        "The independent-edge model deliberately ignores matching exclusion. Its partition function is an upper bound, and its cardinality TV is a diagnostic, not a competitive matching approximation. Mass above min(rows, columns) detects only some invalid configurations.", "",
        "[All activity points and exact probabilities](accuracy.csv).", "",
        "## Repeated-query cost", "",
        f"Coefficient setup plus all {len(study.ACTIVITIES)} exact-statistics queries is faster in {report['coefficient_faster_full_grid_cases']}/{len(curves)} curves by sums of retained timing medians.", "",
        "For each activity, take the median query wall time. Average those medians equally over the frozen activity grid to obtain q (coefficient statistics) and d (direct scalar subset DP). With native setup s, the estimated threshold is max(1, ceil(s/(d−q))) when d>q; otherwise no finite threshold is reported. This model assumes future activities follow this grid equally. The full-grid totals are s+sum(q_i) and sum(d_i), not separately measured batch executions. The accuracy CSV also retains both cumulative costs at every grid prefix. The first grid-prefix crossing depends on increasing-activity order and need not persist at every later prefix.", "",
        "[Break-even inputs, thresholds and full-grid totals](break-even.csv).", "",
        "## Scope and provenance", "",
        "These are deterministic synthetic bipartite graph families and retained seeds, not a population sample of real applications. Weight scaling preserves graph structure and multiplies coefficient k by scale^k. Current/previous-native differences include structural shortcuts and prime planning. Independent methods have different exact CRT bounds and implementations; no speed ratio isolates a single optimization. Frontier is an independent implementation, not a Sage timing. Resource limits, interpreter/import memory and external host load limit generalization. Query preparation excludes library construction/loading; per-query execution includes validation, planning, reductions and CRT.", "",
        "The manifest and original raw records remain in the input run directory. [Machine-readable audit and exact input hashes](report.json) link every retained measurement record, source snapshot, build log and baseline binary checked. Hash consistency does not authenticate the machine clock or establish a formal proof of source semantics.", ""]
    if make_figures:
        lines += ["## Figures", "",
                  "The occupancy panel shows the predefined representative families "
                  +", ".join(occupancy_families(curves))+" (all their retained seeds); full distributions for every family remain in accuracy.csv. Timing and BP-error figures include all families.", "",
                  "![Method timings by family](method-timings.png)", "",
                  "![Exact occupancy and BP mean](occupancy.png)", "", "![Separate BP errors](bp-errors.png)", "",
                  "PDF versions: [timings](method-timings.pdf), [occupancy](occupancy.pdf), [errors](bp-errors.pdf).", ""]
    (output/"README.md").write_text("\n".join(lines))
    report["artifact_sha256"] = {p.name: reference.file_digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    reference.atomic_record(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Complete campaign directory")
    parser.add_argument("--output", required=True, help="New summary directory")
    args = parser.parse_args()
    report = export(args.input, args.output)
    print(json.dumps({k: report[k] for k in ("status", "case_count", "curve_count", "activity_point_count", "native_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
