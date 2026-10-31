"""CPU-only independent arithmetic and archive/protocol checks. No CUDA calls."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import signal
import sys

import pytest


if sys.version_info < (3, 10):
    pytest.skip("standalone Schur benchmark requires Python >=3.10", allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "schur_benchmark", ROOT / "notes/benchmarks/schur_compare.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def permutation_reference(matrix, p):
    result = 0
    for columns in itertools.permutations(range(len(matrix))):
        value = 1
        for i, j in enumerate(columns):
            value = value * matrix[i][j] % p
        result = (result + value) % p
    return result


@pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 1), (3, 7),
                                  (4294967290, 4294967289)])
def test_periodic_formula_and_ryser_against_literal_permutations(a, b):
    p = 4294967291
    for m in (2, 4, 6):
        base = [a, b] * (m // 2)
        matrix = bench.matrix_from_base(base)
        expected = permutation_reference(matrix, p)
        assert bench.periodic_reference(base, p) == expected
        assert bench.ryser_reference(matrix, p) == expected


def test_arbitrary_circulant_orientation_and_independent_reference():
    base = [0, 1, 4294967290, 65536]
    matrix = bench.matrix_from_base(base)
    assert matrix[0] == base
    assert matrix[1] == [65536, 0, 1, 4294967290]
    assert bench.ryser_reference(matrix, 4294967291) == permutation_reference(matrix, 4294967291)


def test_prime_root_safety_and_frozen_public_cases(tmp_path):
    manifest = bench.prepare(tmp_path / "run", "pilot")
    assert len(manifest["cases"]) == 30
    assert bench.validate_archive(tmp_path / "run") == manifest
    for m in sum(bench.PROFILES.values(), ()):
        n, p = m + 1, bench.plan_prime(m + 1)
        c = (1 << 32) - p
        assert bench.is_prime(p) and (p - 1) % n == 0
        assert (1 << 32) - 1 + c * c < 2 * p
        omega = bench.root_of_unity(n, p)
        assert omega != 1 and pow(omega, n, p) == 1
    broken = copy.deepcopy(manifest["cases"][0])
    broken["omega"] = 1
    values = bench.table_values((ROOT / "schur_permanents.txt").read_text())
    with pytest.raises(ValueError, match="root/order/generator"):
        bench.validate_case(broken, values)


def fake_build(output, manifest, nvcc, arch, timeout):
    directory = Path(output) / "build"
    directory.mkdir(exist_ok=True)
    binary = directory / "schur_compare"
    binary.write_bytes(b"CPU test stand-in; never executable")
    bench.write_record(directory / "build.json", {
        "status": "COMPLETED", "returncode": 0, "manifest_sha256": bench.digest(manifest),
        "binary_sha256": bench.file_hash(binary), "arch": arch})
    return binary


def fake_execution_factory(*, first_limit=None, mismatch=False):
    calls = []

    def execute(command, *, cwd, stdout_path, stderr_path, timeout):
        values = list(map(int, Path(command[1]).read_text().split()))
        expected = values[2]
        rows = []
        for line in Path(command[2]).read_text().splitlines():
            ordinal, method, warmup, iteration = line.split()
            m = values[0]
            fixed = method == "fixed_sign"
            outputs = (1 << 18) if m >= 28 else (1 << 14) if m >= 20 else (1 << (m - 1))
            rows.append({"type": "sample", "ordinal": int(ordinal), "method": method,
                         "warmup": bool(int(warmup)), "iteration": int(iteration),
                         "residue": (expected + int(mismatch)) % values[1], "wall_ms": 2 if method == "fixed_sign" else 1,
                         "device_interval_ms": 1 if method == "fixed_sign" else .5,
                         "setup_ms": .1, "finalize_ms": .1,
                         "represented_sign_states": 1 << (m - int(fixed)),
                         "depth": -1 if fixed else min(m, 24), "threads": 256,
                         "blocks": (outputs + 255) // 256 if fixed else 4,
                         "frontier_tasks": 0 if fixed else 1,
                         "explicit_device_bytes": m * m * 4 + 2 * outputs * 8 if fixed else 16 + m * 4 + 12,
                         "host_payload_bytes": m * m * 4 if fixed else 16})
        emitted = rows[:first_limit] if not calls and first_limit is not None else rows
        calls.append([r["ordinal"] for r in emitted])
        device = {"type": "device", "name": "CPU-only test fixture", "uuid_hex": "none", "multiprocessors": 1}
        Path(stdout_path).write_text("".join(json.dumps(r) + "\n" for r in [device] + emitted))
        Path(stderr_path).write_text("")
        return {"status": "COMPLETED" if len(emitted) == len(rows) else "TIMEOUT",
                "command": command, "returncode": 0 if len(emitted) == len(rows) else -9,
                "elapsed_seconds": 0.01}

    return calls, execute


def fake_run(tmp_path, monkeypatch, *, first_limit=None, mismatch=False):
    output = tmp_path / "run"
    bench.prepare(output, "pilot")
    calls, execute = fake_execution_factory(first_limit=first_limit, mismatch=mismatch)
    monkeypatch.setattr(bench, "build", fake_build)
    monkeypatch.setattr(bench, "execute", execute)
    return output, calls


def test_timeout_resume_retains_calls_and_pairing(tmp_path, monkeypatch):
    output, calls = fake_run(tmp_path, monkeypatch, first_limit=5)
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        bench.run(output, max_cases=1)
    partial = bench.verify(output, allow_partial=True)
    assert partial["cases"][0]["completed_samples"] == 5
    report = bench.run(output, max_cases=1, resume=True)
    assert calls == [list(range(5)), list(range(5, 12))]
    first = report["cases"][0]
    assert first["complete"] and first["median_paired_wall_speedup"] == 2
    assert len(first["all_samples"]) == 12
    assert len(first["paired_speedups"]) == 5
    assert report["noncompleted_attempts"] == 1
    assert not first["timing_run_contiguous"]
    with pytest.raises(ValueError, match="incomplete"):
        bench.verify(output)


def test_exact_mismatch_is_archived_and_not_resumable(tmp_path, monkeypatch):
    output, _ = fake_run(tmp_path, monkeypatch, mismatch=True)
    with pytest.raises(ArithmeticError, match="EXACT_MISMATCH"):
        bench.run(output, max_cases=1)
    failure = output / "cases/m04-schur/failure.json"
    assert "EXACT_MISMATCH" in bench.read_record(failure)["error"]
    assert (failure.parent / "attempt-000/stdout.jsonl").is_file()
    with pytest.raises(ArithmeticError, match="EXACT_MISMATCH"):
        bench.run(output, max_cases=1, resume=True)


def completed_one(tmp_path, monkeypatch):
    output, _ = fake_run(tmp_path, monkeypatch)
    bench.run(output, max_cases=1)
    return output


@pytest.mark.parametrize("name", ["input.txt", "schedule.txt"])
def test_rehashed_input_or_schedule_substitution_is_rejected(tmp_path, monkeypatch, name):
    output = completed_one(tmp_path, monkeypatch)
    attempt = output / "cases/m04-schur/attempt-000"
    path = attempt / name
    if name == "input.txt":
        data = path.read_text().split()
        data[-1] = str(int(data[-1]) + 1)
        path.write_text(" ".join(data) + "\n")
    else:
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    metadata = bench.read_record(attempt / "attempt.json")
    metadata["files"][name] = bench.file_hash(path)
    bench.write_record(attempt / "attempt.json", metadata)
    with pytest.raises(ValueError, match="input does not match|schedule must contain"):
        bench.verify(output, allow_partial=True)


def test_binary_linkage_and_explicit_source_archive_audit(tmp_path, monkeypatch):
    output = completed_one(tmp_path, monkeypatch)
    binary = output / "build/schur_compare"
    binary.unlink()
    with pytest.raises(ValueError, match="binary provenance"):
        bench.verify(output, allow_partial=True)
    report = bench.verify(output, allow_partial=True, allow_missing_binary=True)
    assert not report["binary_hash_checked"]
    attempt = output / "cases/m04-schur/attempt-000/attempt.json"
    metadata = bench.read_record(attempt)
    metadata["binary_sha256"] = "0" * 64
    bench.write_record(attempt, metadata)
    with pytest.raises(ValueError, match="attempt binary identity"):
        bench.verify(output, allow_partial=True, allow_missing_binary=True)


def test_frozen_source_mutation_fails_before_execution(tmp_path, monkeypatch):
    output, calls = fake_run(tmp_path, monkeypatch)
    with (output / "frozen/perm_modp_cuda.cu").open("a") as stream:
        stream.write("\n// changed\n")
    with pytest.raises(ValueError, match="frozen source hash"):
        bench.run(output, max_cases=1)
    assert not calls


def test_rehashed_raw_order_and_duplicate_calls_are_rejected(tmp_path, monkeypatch):
    output = completed_one(tmp_path, monkeypatch)
    directory = output / "cases/m04-schur/attempt-000"
    path = directory / "stdout.jsonl"
    rows = bench.read_attempt(path)
    rows[1], rows[2] = rows[2], rows[1]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metadata = bench.read_record(directory / "attempt.json")
    metadata["files"][path.name] = bench.file_hash(path)
    bench.write_record(directory / "attempt.json", metadata)
    with pytest.raises(ValueError, match="sample order mismatch"):
        bench.verify(output, allow_partial=True)
    manifest = bench.read_record(output / "manifest.json")
    with pytest.raises(ValueError, match="duplicate"):
        bench.check_samples(manifest["cases"][0], [rows[0], rows[1], rows[1]], manifest["schedule"])


def test_sigterm_interrupt_is_archived_and_handler_restored(tmp_path, monkeypatch):
    output, _ = fake_run(tmp_path, monkeypatch)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(command, **kwargs):
        Path(kwargs["stdout_path"]).write_text("")
        Path(kwargs["stderr_path"]).write_text("interrupted before device initialization\n")
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(bench, "execute", interrupted)
    with pytest.raises(KeyboardInterrupt, match="terminating worker group"):
        bench.run(output, max_cases=1)
    assert signal.getsignal(signal.SIGTERM) == previous
    report = bench.verify(output, allow_partial=True)
    assert report["noncompleted_attempts"] == 1
    assert report["cases"][0]["completed_samples"] == 0
    assert report["cases"][0]["attempts"][0]["status"] == "INTERRUPTED"


@pytest.mark.parametrize("alteration", ["missing", "device", "launch", "status"])
def test_rehashed_execution_integrity_failures(tmp_path, monkeypatch, alteration):
    output = completed_one(tmp_path, monkeypatch)
    directory = output / "cases/m04-schur/attempt-000"
    path = directory / "stdout.jsonl"
    rows = bench.read_attempt(path)
    metadata = bench.read_record(directory / "attempt.json")
    if alteration == "missing":
        rows.pop()
    elif alteration == "device":
        rows.pop(0)
    elif alteration == "launch":
        rows[1]["blocks"] += 1
    else:
        metadata["returncode"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metadata["files"][path.name] = bench.file_hash(path)
    bench.write_record(directory / "attempt.json", metadata)
    with pytest.raises(ValueError, match="omitted scheduled|initial device|launch configuration|status/exit"):
        bench.verify(output, allow_partial=True)
