#!/usr/bin/env python3
"""Export every completed seeded CPU comparison, including losses, to CSV/MD."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

try:
    from . import polynomial_compare as benchmark
except ImportError:
    import polynomial_compare as benchmark


def comparison_rows(campaign):
    campaign = Path(campaign)
    manifest = benchmark.read_record(campaign / "manifest.json")
    for relative, expected in manifest["identity"]["sources"].items():
        if benchmark.file_digest(campaign / "sources" / relative) != expected:
            raise ValueError(f"archived source digest mismatch: {relative}")
    rows = []
    for path in sorted((campaign / "cases").glob("*.json")):
        record = benchmark.read_record(path)
        identity = record["identity"]
        if identity["manifest_sha256"] != benchmark.digest(manifest):
            raise ValueError("case manifest digest mismatch")
        if not record["complete"]:
            continue
        case, expected, measured, public_prime_counts = identity["case"], None, {}, set()
        for sample in record["samples"]:
            benchmark.validate_result(sample["result"], case, identity["primes"],
                                      identity["bounds"], identity["signed"])
            coefficients = sample["result"]["coefficients"]
            if expected is None:
                expected = coefficients
            benchmark.require_match(coefficients, expected, "summary vector agreement")
            if sample["phase"] == "measurement":
                measured.setdefault(sample["method"], []).append(sample["result"]["total_seconds"])
            if sample["method"] == "public_exact":
                public_prime_counts.add(sample["result"]["prime_count"])
        if len(public_prime_counts) != 1:
            raise ValueError("public exact prime count changed between repeats")
        required = benchmark.methods_for(case)
        if sorted(measured) != sorted(required) or any(
                len(times) != identity["repeats"] for times in measured.values()):
            raise ValueError("incomplete measurement set")
        timings = {method: statistics.median(times) for method, times in measured.items()}
        alternatives = [method for method in required if method not in ("native_polynomial", "public_exact")]
        closest = min(alternatives, key=timings.get)
        common_winner = min(["native_polynomial", *alternatives], key=timings.get)
        alternative_e2e = timings[closest] + record["shared_planning_seconds"]
        row = {"campaign": str(campaign), "case_sha256": benchmark.digest(case),
               "contract": case["contract"], "family": case["family"],
               "n": case["n"], "scale": case["scale"], "seed": case["seed"],
               "common_prime_count": len(identity["primes"]),
               "public_prime_count": next(iter(public_prime_counts)),
               "common_winner": common_winner, "closest_alternative": closest,
               "native_over_best_alternative": timings["native_polynomial"] / timings[closest],
               "public_over_best_alternative_with_shared_planning": timings["public_exact"] / alternative_e2e,
               "shared_planning_seconds": record["shared_planning_seconds"],
               "alternative_with_shared_planning_seconds": alternative_e2e}
        for method in ("native_polynomial", "polynomial_ryser", "rook_dp", "scalar_interpolation", "public_exact"):
            row[method + "_seconds"] = timings.get(method)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["contract"], row["n"], row["scale"], row["family"]))


def export(campaigns, output):
    rows = [row for campaign in campaigns for row in comparison_rows(campaign)]
    if not rows:
        raise ValueError("no complete cases to summarize")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Seeded CPU polynomial comparison results", "",
             f"{len(rows)} complete cases, one CPU thread. All stored modular/CRT vectors replayed and agree.", "",
             "These single-seed pilot comparisons used a shared host. They are not isolated publication timings,",
             "a statistical estimate over matrix ensembles, or an exhaustive comparison with structure-specific algorithms.", "",
             "Times are medians in milliseconds for a complete exact coefficient vector. Common-prime methods",
             "include all selected modular calls and CRT. Public exact has its own planner and prime choices.",
             "The end-to-end ratio adds shared bound/prime planning time to the closest alternative.", "",
             "A ratio below one favors the native/public method; above one favors the named alternative.", "",
             "| Contract | n | Scale | Family | Common/public primes | Native ms | Ryser ms | DP ms | Interp. ms | Public ms | Closest alternative | Native / alternative | Public / alternative + planning |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in rows:
        times = []
        for method in ("native_polynomial", "polynomial_ryser", "rook_dp", "scalar_interpolation", "public_exact"):
            value = row[method + "_seconds"]
            times.append("—" if value is None else f"{value * 1000:.3f}")
        values = [row["contract"], str(row["n"]), str(row["scale"]), row["family"],
                  f"{row['common_prime_count']}/{row['public_prime_count']}", *times,
                  row["closest_alternative"], f"{row['native_over_best_alternative']:.3f}",
                  f"{row['public_over_best_alternative_with_shared_planning']:.3f}"]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Every case is retained, including structured losses. Sources, raw repeats, all coefficients,",
              "prime counts, scalar evaluations, and timing/memory limitations are in the campaign manifests and cases.",
              "See notes/benchmarks/README.md for algorithm contracts and prior work.", ""]
    (output / "comparisons.md").write_text("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = export(args.campaigns, args.output)
    print(f"Exported {len(rows)} verified cases to {args.output}")


if __name__ == "__main__":
    main()
