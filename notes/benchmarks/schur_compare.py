#!/usr/bin/env python3
"""Frozen, exact matched comparisons of the current CUDA circulant kernels.

prepare and verify never initialize CUDA. run is the only execution command;
run --compile-only compiles without invoking the resulting executable.
See the generated PROTOCOL.md before approving a GPU campaign.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "schur-current-source-comparison-v1"
PROFILES = {"pilot": (4, 6, 10, 12, 16, 18),
            "main": (22, 28, 30, 36, 40), "frontier": (46,)}
SEEDS = (2026090501, 2026090502)
METHODS = ("fixed_sign", "necklace")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_record(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump({"sha256": digest(payload), "payload": payload}, stream,
                  indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_record(path):
    data = json.loads(Path(path).read_text())
    if data.get("sha256") != digest(data.get("payload")):
        raise ValueError(f"record digest mismatch: {path}")
    return data["payload"]


def is_prime(n):
    """Deterministic trial division for the limited <=32-bit preparation inputs."""
    if not 2 <= n < 1 << 32:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


def plan_prime(n):
    for complement in range(1, 65536, 2):
        prime = (1 << 32) - complement
        if (prime - 1) % n == 0 and is_prime(prime):
            return prime
    raise ValueError("no permitted prime for this Schur order")


def primitive_root(n):
    if not is_prime(n):
        raise ValueError("Schur order must be prime")
    for candidate in range(2, n):
        if len({pow(candidate, k, n) for k in range(n - 1)}) == n - 1:
            return candidate
    raise ValueError("primitive root not found")


def root_of_unity(n, prime):
    for candidate in range(2, prime):
        omega = pow(candidate, (prime - 1) // n, prime)
        if omega != 1 and pow(omega, n, prime) == 1:
            return omega
    raise ValueError("root of unity not found")


def matrix_from_base(base):
    m = len(base)
    return [[base[(j - i) % m] for j in range(m)] for i in range(m)]


def ryser_reference(matrix, p):
    """Independent subset Ryser using unbounded Python arithmetic, n<=12 only."""
    m = len(matrix)
    if m > 12:
        raise ValueError("CPU reference intentionally bounded at order 12")
    if m == 0:
        return 1 % p
    total, previous, rows = 0, 0, [0] * m
    for step in range(1, 1 << m):
        subset = step ^ (step >> 1)
        changed = previous ^ subset
        column = changed.bit_length() - 1
        direction = 1 if changed & subset else -1
        for i in range(m):
            rows[i] += direction * matrix[i][column]
        term = math.prod(rows) % p
        total += -term if (m - subset.bit_count()) & 1 else term
        previous = subset
    return total % p


def periodic_reference(base, p):
    """Two equal row/column classes, counted by the number of within-class edges."""
    m, (a, b) = len(base), base[:2]
    if m % 2 or base != [a, b] * (m // 2):
        raise ValueError("not an even period-two circulant")
    k = m // 2
    return (math.factorial(k) ** 2 * sum(
        math.comb(k, j) ** 2 * pow(a, 2 * j, p) * pow(b, 2 * (k - j), p)
        for j in range(k + 1))) % p


def table_values(text):
    return {int(n): int(value) for n, value in
            re.findall(r"^n=(\d+):\s+(-?\d+)\s*$", text, re.MULTILINE)}


def make_cases(profile, values):
    cases = []
    for m in PROFILES[profile]:
        n, prime = m + 1, plan_prime(m + 1)
        g, omega = primitive_root(n), root_of_unity(n, prime)
        for family, seed in [("schur", None)] + [
                (family, seed) for family in ("dense", "periodic") for seed in SEEDS]:
            case_id = f"m{m:02d}-{family}" + (f"-s{seed}" if seed is not None else "")
            rng = random.Random(int(digest([SCHEMA, m, family, seed]), 16))
            if family == "schur":
                base = [pow(omega, pow(g, j, n), prime) for j in range(m)]
                expected = values[n] * pow(n, -1, prime) % prime
                reference = "public Schur integer divided by n modulo p"
            elif family == "periodic":
                a, b = rng.sample(range(1, prime), 2)
                base = [a, b] * (m // 2)
                expected = periodic_reference(base, prime)
                reference = "independent exact two-class combinatorial formula"
            else:
                base = [rng.randrange(1, prime) for _ in range(m)]
                expected, reference = None, "cross-method agreement only"
            matrix = matrix_from_base(base)
            if m <= 12:
                independent = ryser_reference(matrix, prime)
                if expected is not None and expected != independent:
                    raise ValueError(f"independent reference mismatch for {case_id}")
                expected = independent
                reference += "; independent Python subset Ryser" if family != "dense" else ""
                if family == "dense":
                    reference = "independent Python subset Ryser"
            case = {"id": case_id, "m": m, "family": family, "seed": seed,
                    "prime": prime, "base": base, "matrix_sha256": digest(matrix),
                    "orientation": "A[i,j] = base[(j-i) % M]",
                    "expected_residue": expected, "reference": reference,
                    "schur_order": n if family == "schur" else None,
                    "omega": omega if family == "schur" else None,
                    "generator": g if family == "schur" else None,
                    "necklace_depth": min(m, 24) if m != 46 else 28,
                    "production_necklace_size_supported": m in (40, 46),
                    "production_fixed_pm32_size_supported": m in range(28, 45, 2)}
            validate_case(case, values)
            cases.append(case)
    return cases


def validate_case(case, values):
    m, p, base = case["m"], case["prime"], case["base"]
    if m not in sum(PROFILES.values(), ()) or not is_prime(m + 1):
        raise ValueError("unsupported case order")
    complement = (1 << 32) - p
    if not is_prime(p) or not 1 <= complement <= 65535 or (p - 1) % (m + 1):
        raise ValueError("invalid prime or root divisibility")
    # Exact conservative two-fold bound from the frozen production arithmetic.
    if ((1 << 32) - 1 + complement ** 2 >= 2 * p or
            ((complement + 1) << 32) >= 1 << 64):
        raise ValueError("pseudo-Mersenne fold bounds fail")
    if len(base) != m or any(type(x) is not int or not 0 <= x < p for x in base):
        raise ValueError("invalid canonical base residues")
    if case["matrix_sha256"] != digest(matrix_from_base(base)):
        raise ValueError("input matrix hash mismatch")
    if case["orientation"] != "A[i,j] = base[(j-i) % M]":
        raise ValueError("matrix orientation mismatch")
    expected = case["expected_residue"]
    if expected is not None and (type(expected) is not int or not 0 <= expected < p):
        raise ValueError("invalid expected residue")
    if case["family"] == "schur":
        n, omega, g = m + 1, case["omega"], case["generator"]
        if (case["schur_order"] != n or omega == 1 or pow(omega, n, p) != 1
                or len({pow(g, k, n) for k in range(m)}) != m):
            raise ValueError("invalid Schur root/order/generator")
        if base != [pow(omega, pow(g, j, n), p) for j in range(m)]:
            raise ValueError("Schur base mismatch")
        if expected != values[n] * pow(n, -1, p) % p:
            raise ValueError("Schur table reference mismatch")
    elif case["family"] == "periodic":
        if expected != periodic_reference(base, p):
            raise ValueError("periodic reference mismatch")
    elif case["family"] != "dense":
        raise ValueError("unknown family")
    if m <= 12 and expected != ryser_reference(matrix_from_base(base), p):
        raise ValueError("independent small-case reference mismatch")


def schedule(warmups, repeats):
    rows = []
    for round_index in range(warmups + repeats):
        order = METHODS if round_index % 2 == 0 else METHODS[::-1]
        for method in order:
            rows.append({"ordinal": len(rows), "method": method,
                         "warmup": round_index < warmups,
                         "iteration": round_index if round_index < warmups
                         else round_index - warmups})
    return rows


PROTOCOL = """# Frozen Schur/circulant CUDA comparison

This is a matched kernel-template comparison, not public API dispatch timing.
Both paths include exactly the archived production perm_modp_cuda.cu; the wrapper
adds template instantiations without editing kernel definitions. Both use the
same near-2^32 prime and unsigned 32-bit modular arithmetic, with exact equality.
The input is the complete circulant A[i,j] = base[(j-i) mod M], matching the
unchanged production necklace kernel's column-minus-row convention.
Schur cases return the minor permanent; multiplication by n recovers the full
Schur permanent and is checked algebraically during preparation.

Fixed profiles: pilot M=4,6,10,12,16,18; main M=22,28,30,36,40; optional frontier
M=46. Each size has one Schur minor, two public seeded dense circulants and two
public seeded period-two circulants. No cases are selected by timing. The
largest valid prime below 2^32 congruent to 1 modulo M+1 is shared by all five
cases at that size. Prime/root orders and two-fold arithmetic bounds are checked.

One warmup pair and five timed pairs alternate method order. Full cold CUDA
context initialization, parsing, compilation and archive I/O are excluded.
Primary wall time is the full wrapper method call, including host matrix/frontier
construction, allocation, transfers, kernel execution, result normalization and
cleanup. Secondary CUDA-event time is the interval spanning kernel/reduction
launches; it includes host scheduling gaps and is not a sum of pure kernel times.
Keep every timing, warmup, residue, launch configuration and failure. Setup and
finalization wall components are also reported; they are not an exact additive
decomposition of the event interval. Explicit GPU buffer bytes and host payload
bytes are accounting measures, not measured peak RAM or total CUDA device usage.

Fixed-sign scheduling follows production: 256 threads/block, 2^18 outputs for
M>=28, 2^14 for 20<=M<28, otherwise one output/sign state. Necklace scheduling:
256 threads/block, four blocks/SM; prefix depth min(M,24), except M46 depth28.
These are fixed before measurement. M40/depth24 and M46/depth28 match production;
other instantiations are explicitly experimental benchmark configurations.
The current production fixed-sign launcher lacks pm32 at M46; this benchmark
uses the same pm32 template at M46 to avoid an arithmetic-width confound.

Small M<=12 cases have an independent exact Python Ryser reference. Period-two
cases have an independent exact combinatorial formula at every size. Schur
cases check the archived public integer table modulo p after division by n.
Larger dense cases establish cross-method agreement only, not an independent
proof. No tolerance or approximate arithmetic is permitted. Exact mismatches
stop the run with the failing result retained. Timeouts retain completed calls
and can resume only uncompleted slots; no successful call is remeasured.

Per-case speedup is median of the five paired fixed-sign/necklace wall ratios;
all ratios and method samples are retained. Larger values favor necklace.
Use separate family/size results; the suite is not a claim of uniform advantage.
Record host isolation and competing GPU activity separately when running.
No full-integer CRT performance or order47 production speed claim follows from
this modular benchmark. Historical frontier times remain historical.
"""


def prepare(output, profile, root=ROOT):
    output, root = Path(output), Path(root)
    if output.exists():
        raise ValueError("prepare requires a fresh output directory")
    source_paths = {"perm_modp_cuda.cu": root / "perm_modp_cuda.cu",
                    "schur_permanents.txt": root / "schur_permanents.txt",
                    "schur_compare_wrapper.cu": Path(__file__).with_name("schur_compare_wrapper.cu"),
                    "schur_compare.py": Path(__file__)}
    values = table_values(source_paths["schur_permanents.txt"].read_text())
    cases = make_cases(profile, values)
    frozen = output / "frozen"
    frozen.mkdir(parents=True)
    sources = {}
    for name, original in source_paths.items():
        shutil.copyfile(original, frozen / name)
        sources[name] = {"sha256": file_hash(frozen / name),
                         "original_path": str(original.resolve())}
    (output / "PROTOCOL.md").write_text(PROTOCOL)
    manifest = {"schema": SCHEMA, "created": now(), "profile": profile,
                "sources": sources, "cases": cases, "case_sha256": digest(cases),
                "protocol_sha256": file_hash(output / "PROTOCOL.md"),
                "warmups": 1, "repeats": 5, "schedule": schedule(1, 5),
                "preparation_host": platform.uname()._asdict(),
                "invocation": sys.argv}
    write_record(output / "manifest.json", manifest)
    return manifest


def validate_archive(output, check_runner=False):
    output = Path(output)
    manifest = read_record(output / "manifest.json")
    if manifest["schema"] != SCHEMA or digest(manifest["cases"]) != manifest["case_sha256"]:
        raise ValueError("manifest identity mismatch")
    for name, info in manifest["sources"].items():
        if Path(name).name != name or file_hash(output / "frozen" / name) != info["sha256"]:
            raise ValueError(f"frozen source hash mismatch: {name}")
    if check_runner and file_hash(__file__) != manifest["sources"]["schur_compare.py"]["sha256"]:
        raise ValueError("running script differs from frozen runner; use the archived runner")
    if file_hash(output / "PROTOCOL.md") != manifest["protocol_sha256"]:
        raise ValueError("protocol hash mismatch")
    if manifest["warmups"] != 1 or manifest["repeats"] != 5 or manifest["schedule"] != schedule(1, 5):
        raise ValueError("frozen schedule mismatch")
    values = table_values((output / "frozen" / "schur_permanents.txt").read_text())
    # Recreate the public instances, rather than trusting only self-hashes.
    if make_cases(manifest["profile"], values) != manifest["cases"]:
        raise ValueError("case regeneration mismatch")
    return manifest


def execute(command, *, cwd, stdout_path, stderr_path, timeout):
    started = time.monotonic()
    with Path(stdout_path).open("wb") as stdout, Path(stderr_path).open("wb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            status = "COMPLETED" if code == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code, status = process.returncode, "TIMEOUT"
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return {"command": command, "returncode": code, "status": status,
            "elapsed_seconds": time.monotonic() - started}


def build(output, manifest, nvcc, arch, timeout):
    output = Path(output).resolve()
    directory = output / "build"
    record_path = directory / "build.json"
    if record_path.exists():
        record = read_record(record_path)
        if (record["manifest_sha256"] != digest(manifest) or record["arch"] != arch
                or record["binary_sha256"] != file_hash(directory / "schur_compare")):
            raise ValueError("build identity mismatch")
        return directory / "schur_compare"
    if not re.fullmatch(r"sm_[0-9]+[a-z]?", arch):
        raise ValueError("architecture must be explicit, e.g. sm_90")
    compiler = shutil.which(nvcc)
    if compiler is None:
        raise ValueError(f"CUDA compiler unavailable: {nvcc}")
    directory.mkdir(exist_ok=True)
    command = [compiler, "-O3", "-std=c++14", f"-arch={arch}", "-Xptxas=-v",
               str(output / "frozen" / "schur_compare_wrapper.cu"),
               "-o", str(directory / "schur_compare")]
    version = subprocess.run([compiler, "--version"], capture_output=True,
                             text=True, timeout=30, check=True)
    result = execute(command, cwd=output / "frozen", stdout_path=directory / "compile.stdout",
                     stderr_path=directory / "compile.stderr", timeout=timeout)
    write_record(directory / "compile-attempt.json", result)
    if result["status"] != "COMPLETED":
        raise RuntimeError("CUDA compilation failed; retained build/compile.stderr")
    record = {**result, "arch": arch, "manifest_sha256": digest(manifest),
              "binary_sha256": file_hash(directory / "schur_compare"),
              "compiler_sha256": file_hash(compiler), "compiler_version": version.stdout,
              "host": platform.uname()._asdict(), "completed": now()}
    write_record(record_path, record)
    return directory / "schur_compare"


def read_attempt(path):
    """A killed final JSON line is retained but cannot count as a completed call."""
    rows = []
    raw = Path(path).read_bytes()
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        rows.append(json.loads(line))
    return rows


def check_samples(case, rows, expected_schedule):
    by_ordinal = {}
    reference = case["expected_residue"]
    devices = []
    for row in rows:
        if row.get("type") == "device":
            if type(row.get("multiprocessors")) is not int or row["multiprocessors"] <= 0:
                raise ValueError("invalid device multiprocessor count")
            devices.append(row)
            continue
        if row.get("type") != "sample":
            raise ValueError("unknown raw result record")
        ordinal = row.get("ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < len(expected_schedule):
            raise ValueError("invalid sample slot")
        if ordinal in by_ordinal:
            raise ValueError("duplicate completed sample; successful calls must not be rerun")
        for key, value in expected_schedule[ordinal].items():
            if row.get(key) != value:
                raise ValueError("sample schedule mismatch")
        residue = row.get("residue")
        if type(residue) is not int or not 0 <= residue < case["prime"]:
            raise ValueError("invalid modular residue")
        if reference is not None and residue != reference:
            raise ArithmeticError(f"EXACT_MISMATCH: {case['id']} slot {ordinal}: {residue} != {reference}")
        reference = residue
        for key in ("wall_ms", "setup_ms", "device_interval_ms", "finalize_ms"):
            if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("invalid timing value")
        if row["wall_ms"] <= 0 or row["device_interval_ms"] <= 0:
            raise ValueError("zero primary timing")
        if not devices:
            raise ValueError("sample precedes CUDA device identity")
        m = case["m"]
        fixed = row["method"] == "fixed_sign"
        outputs = (1 << 18) if m >= 28 else (1 << 14) if m >= 20 else (1 << (m - 1))
        config = {"represented_sign_states": 1 << (m - int(fixed)),
                  "threads": 256, "depth": -1 if fixed else case["necklace_depth"],
                  "blocks": (outputs + 255) // 256 if fixed else 4 * devices[-1]["multiprocessors"]}
        if any(type(row.get(key)) is not int or row[key] != value for key, value in config.items()):
            raise ValueError("reported kernel launch configuration mismatch")
        tasks = row.get("frontier_tasks")
        if type(tasks) is not int or (tasks != 0 if fixed else not 0 < tasks < 1 << 32):
            raise ValueError("invalid necklace frontier count")
        device_bytes = m * m * 4 + 2 * outputs * 8 if fixed else tasks * 16 + m * 4 + 12
        host_bytes = m * m * 4 if fixed else tasks * 16
        if row.get("explicit_device_bytes") != device_bytes or row.get("host_payload_bytes") != host_bytes:
            raise ValueError("explicit memory accounting mismatch")
        by_ordinal[ordinal] = row
    if by_ordinal and not devices:
        raise ValueError("samples lack CUDA device identity")
    if devices and any(device != devices[0] for device in devices):
        raise ValueError("device identity changed across attempts")
    return by_ordinal, devices[0] if devices else None


def case_summary(case, samples, repeats):
    timed = [r for r in samples.values() if not r["warmup"]]
    per_method = {method: [r for r in timed if r["method"] == method] for method in METHODS}
    paired = []
    for iteration in range(repeats):
        pair = {r["method"]: r for r in timed if r["iteration"] == iteration}
        if len(pair) == 2:
            paired.append({"iteration": iteration,
                           "wall_speedup": pair["fixed_sign"]["wall_ms"] / pair["necklace"]["wall_ms"],
                           "device_interval_speedup": pair["fixed_sign"]["device_interval_ms"] /
                           pair["necklace"]["device_interval_ms"]})
    return {"id": case["id"], "m": case["m"], "family": case["family"], "seed": case["seed"],
            "case_sha256": digest(case), "prime": case["prime"],
            "reference": case["reference"], "completed_samples": len(samples),
            "residue": next(iter(samples.values()))["residue"] if samples else None,
            "paired_speedups": paired,
            "all_samples": sorted(samples.values(), key=lambda r: r["ordinal"]),
            "median_paired_wall_speedup": statistics.median([p["wall_speedup"] for p in paired]) if paired else None,
            "methods": {method: {"samples": sorted(rows, key=lambda r: r["ordinal"]),
                                    "median_wall_ms": statistics.median([r["wall_ms"] for r in rows]) if rows else None}
                        for method, rows in per_method.items()}}


def collect_case(output, case, manifest):
    directory = Path(output) / "cases" / case["id"]
    rows = []
    attempts = []
    compiled_path = Path(output) / "build" / "build.json"
    compiled = read_record(compiled_path) if compiled_path.exists() else None
    for index, attempt_directory in enumerate(sorted(directory.glob("attempt-*"))):
        if not attempt_directory.is_dir() or attempt_directory.name != f"attempt-{index:03d}":
            raise ValueError("noncanonical or missing attempt directory")
        path = attempt_directory / "stdout.jsonl"
        if not path.exists():
            raise ValueError("incomplete attempt directory lacks retained stdout")
        metadata_path = path.parent / "attempt.json"
        if metadata_path.exists():
            metadata = read_record(metadata_path)
            if (metadata["case_sha256"] != digest(case) or
                    metadata["manifest_sha256"] != digest(manifest)):
                raise ValueError("attempt identity mismatch")
            if (compiled is None or compiled["status"] != "COMPLETED" or compiled["returncode"] != 0
                    or metadata["binary_sha256"] != compiled["binary_sha256"]):
                raise ValueError("attempt binary identity mismatch")
            if (metadata["status"] not in ("COMPLETED", "FAILED", "TIMEOUT", "INTERRUPTED") or
                    (metadata["status"] == "COMPLETED" and metadata["returncode"] != 0) or
                    (metadata["status"] in ("FAILED", "TIMEOUT") and
                     (type(metadata["returncode"]) is not int or metadata["returncode"] == 0))):
                raise ValueError("attempt status/exit code mismatch")
            if set(metadata["files"]) != {"input.txt", "schedule.txt", "stdout.jsonl", "stderr.txt"}:
                raise ValueError("attempt file inventory mismatch")
            for name, value in metadata["files"].items():
                if Path(name).name != name or file_hash(path.parent / name) != value:
                    raise ValueError("attempt file hash mismatch")
            attempts.append(metadata)
        else:
            # Interrupted parent: raw calls still validate, but provenance is
            # incomplete and execution must not silently continue this archive.
            raise ValueError(f"unfinished attempt metadata: {path.parent}")
        input_values = list(map(int, (path.parent / "input.txt").read_text().split()))
        if (len(input_values) != case["m"] + 3 or
                input_values[:2] != [case["m"], case["prime"]] or input_values[3:] != case["base"]):
            raise ValueError("attempt input does not match frozen matrix")
        previous_samples, _ = check_samples(case, rows, manifest["schedule"])
        expected = (next(iter(previous_samples.values()))["residue"] if previous_samples
                    else case["expected_residue"])
        if input_values[2] != (expected if expected is not None else -1):
            raise ValueError("attempt reference anchor mismatch")
        remaining = [s for s in manifest["schedule"] if s["ordinal"] not in previous_samples]
        expected_text = "".join(f"{s['ordinal']} {s['method']} {int(s['warmup'])} {s['iteration']}\n"
                                for s in remaining)
        if (path.parent / "schedule.txt").read_text() != expected_text:
            raise ValueError("attempt schedule must contain exactly the remaining slots")
        attempt_rows = read_attempt(path)
        if ((attempt_rows and (attempt_rows[0].get("type") != "device" or
                sum(row.get("type") == "device" for row in attempt_rows) != 1)) or
                (not attempt_rows and metadata["status"] == "COMPLETED")):
            raise ValueError("each attempt requires exactly one initial device record")
        if metadata["status"] == "COMPLETED" and not path.read_bytes().endswith(b"\n"):
            raise ValueError("successful attempt has an incomplete final record")
        attempt_samples = [row for row in attempt_rows if row.get("type") == "sample"]
        if [r.get("ordinal") for r in attempt_samples] != [s["ordinal"] for s in remaining[:len(attempt_samples)]]:
            raise ValueError("attempt sample order mismatch")
        if metadata["status"] == "COMPLETED" and len(attempt_samples) != len(remaining):
            raise ValueError("successful process omitted scheduled samples")
        rows.extend(attempt_rows)
    samples, device = check_samples(case, rows, manifest["schedule"])
    return samples, device, attempts


def verify(output, *, allow_partial=False, allow_missing_binary=False):
    output = Path(output)
    manifest = validate_archive(output)
    build_path = output / "build" / "build.json"
    if build_path.exists():
        compiled = read_record(build_path)
        binary = output / "build" / "schur_compare"
        if (compiled["status"] != "COMPLETED" or compiled["manifest_sha256"] != digest(manifest) or
                (binary.exists() and compiled["binary_sha256"] != file_hash(binary)) or
                (not binary.exists() and not allow_missing_binary)):
            raise ValueError("compiled binary provenance mismatch")
    cases, devices, issues = [], [], []
    for case in manifest["cases"]:
        samples, device, attempts = collect_case(output, case, manifest)
        summary = case_summary(case, samples, manifest["repeats"])
        summary["attempts"] = attempts
        summary["noncompleted_attempts"] = [a for a in attempts if a["status"] != "COMPLETED"]
        summary["timing_run_contiguous"] = len(attempts) == 1 and not summary["noncompleted_attempts"]
        summary["complete"] = len(samples) == len(manifest["schedule"])
        cases.append(summary)
        if device is not None:
            devices.append(device)
    if devices and any(device != devices[0] for device in devices):
        issues.append("device identity differs between cases")
    complete = all(case["complete"] for case in cases)
    if not complete and not allow_partial:
        raise ValueError("campaign incomplete (use --allow-partial for a partial audit)")
    if issues:
        raise ValueError("; ".join(issues))
    return {"schema": SCHEMA, "status": "COMPLETE" if complete else "PARTIAL",
            "manifest_sha256": digest(manifest), "verified_at": now(),
            "complete_cases": sum(case["complete"] for case in cases),
            "total_cases": len(cases), "device": devices[0] if devices else None,
            "binary_hash_checked": build_path.exists() and (output / "build" / "schur_compare").exists(),
            "noncompleted_attempts": sum(len(case["noncompleted_attempts"]) for case in cases),
            "all_attempts_completed_normally": all(not case["noncompleted_attempts"] for case in cases),
            "cases": cases, "issues": issues}


def _run(output, *, nvcc="nvcc", arch="sm_90", compile_only=False,
        compile_timeout=1200, timeout=1800, max_cases=None, max_seconds=None,
        allow_large=False, resume=False):
    output = Path(output).resolve()
    if min(compile_timeout, timeout) <= 0 or (max_cases is not None and max_cases < 1) or (max_seconds is not None and max_seconds <= 0):
        raise ValueError("timeouts and execution limits must be positive")
    manifest = validate_archive(output, check_runner=True)
    if manifest["profile"] == "frontier" and not allow_large:
        raise ValueError("frontier execution requires explicit --allow-large")
    binary = build(output, manifest, nvcc, arch, compile_timeout)
    if compile_only:
        return {"status": "COMPILED", "binary": str(binary)}
    if (output / "cases").exists() and not resume:
        raise ValueError("existing case results require --resume")
    started, count = time.monotonic(), 0
    for case in manifest["cases"]:
        samples, _, prior_attempts = collect_case(output, case, manifest)
        missing = [s for s in manifest["schedule"] if s["ordinal"] not in samples]
        if not missing:
            continue
        if max_cases is not None and count >= max_cases:
            break
        remaining = None if max_seconds is None else max_seconds - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            break
        case_dir = output / "cases" / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = case_dir / f"attempt-{len(prior_attempts):03d}"
        attempt_dir.mkdir()
        expected = next(iter(samples.values()))["residue"] if samples else case["expected_residue"]
        (attempt_dir / "input.txt").write_text(
            f"{case['m']} {case['prime']} {expected if expected is not None else -1}\n" +
            " ".join(map(str, case["base"])) + "\n")
        (attempt_dir / "schedule.txt").write_text("".join(
            f"{s['ordinal']} {s['method']} {int(s['warmup'])} {s['iteration']}\n" for s in missing))
        command = [str(binary), str(attempt_dir / "input.txt"), str(attempt_dir / "schedule.txt")]
        attempt = {"case_sha256": digest(case), "manifest_sha256": digest(manifest),
                   "started": now(), "binary_sha256": file_hash(binary),
                   "host": platform.uname()._asdict(), "runner_invocation": sys.argv}
        try:
            execution = execute(command, cwd=output, stdout_path=attempt_dir / "stdout.jsonl",
                                stderr_path=attempt_dir / "stderr.txt",
                                timeout=min(timeout, remaining) if remaining is not None else timeout)
        except BaseException:
            execution = {"status": "INTERRUPTED", "command": command, "returncode": None}
            attempt.update(execution)
            attempt["files"] = {p.name: file_hash(p) for p in attempt_dir.iterdir() if p.is_file()}
            write_record(attempt_dir / "attempt.json", attempt)
            raise
        attempt.update(execution)
        attempt["files"] = {p.name: file_hash(p) for p in attempt_dir.iterdir() if p.is_file()}
        write_record(attempt_dir / "attempt.json", attempt)
        try:
            # This validates and retains mismatches before any result is called successful.
            samples, _, _ = collect_case(output, case, manifest)
        except (ValueError, ArithmeticError) as error:
            write_record(case_dir / "failure.json", {"error": str(error), "attempt": attempt})
            raise
        write_record(case_dir / "result.json", case_summary(case, samples, manifest["repeats"]))
        count += 1
        print(f"{case['id']}: {len(samples)}/{len(manifest['schedule'])} exact samples; {execution['status']}", flush=True)
        if execution["status"] != "COMPLETED":
            raise RuntimeError(f"case {case['id']} {execution['status']}; completed calls retained for --resume")
    report = verify(output, allow_partial=True)
    write_record(output / "report.json", report)
    return report


def run(output, **kwargs):
    """One writer per archive; concurrent GPU runs cannot reuse the same slots."""
    with (Path(output) / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another runner holds this archive lock") from error
        def terminate(signum, frame):
            raise KeyboardInterrupt(f"received signal {signum}; terminating worker group")
        previous_handler = signal.signal(signal.SIGTERM, terminate)
        try:
            return _run(output, **kwargs)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    p = subparsers.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=PROFILES, default="pilot")
    p.add_argument("--root", type=Path, default=ROOT)
    p = subparsers.add_parser("run")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--arch", default="sm_90")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--compile-timeout", type=float, default=1200)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--max-cases", type=int)
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--allow-large", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = subparsers.add_parser("verify")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--allow-missing-binary", action="store_true",
                   help="audit a source/raw-data release that intentionally omits the compiled binary")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            result = prepare(**args)
            print(json.dumps({"status": "PREPARED", "cases": len(result["cases"]), "profile": result["profile"]}))
        elif command == "run":
            result = run(**args)
            print(json.dumps({"status": result["status"]}))
        else:
            result = verify(**args)
            print(json.dumps({"status": result["status"], "complete_cases": result["complete_cases"],
                              "total_cases": result["total_cases"]}))
    except (ValueError, RuntimeError, ArithmeticError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
