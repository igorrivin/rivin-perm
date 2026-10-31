"""Exact compiled query baseline versus direct weighted edge-set enumeration."""
import ctypes
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import shutil

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "notes/benchmarks/weighted_matching_partition.py"
if not SOURCE.is_file():
    pytest.skip("checkout-only weighted partition baseline", allow_module_level=True)
SPEC = importlib.util.spec_from_file_location("weighted_partition_test", SOURCE)
partition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partition)


def edge_set_coefficients(matrix):
    n = len(matrix)
    edges = [(i, j, w) for i, row in enumerate(matrix) for j, w in enumerate(row) if w]
    result = [1] + [0] * n
    for k in range(1, n + 1):
        for chosen in itertools.combinations(edges, k):
            if len({i for i, _, _ in chosen}) == k and len({j for _, j, _ in chosen}) == k:
                result[k] += math.prod(w for _, _, w in chosen)
    return result


def oracle(matrix, activity):
    activity = Fraction(activity)
    p, q = activity.numerator, activity.denominator
    coefficients = edge_set_coefficients(matrix)
    terms = [value * p**k * q**(len(matrix) - k) for k, value in enumerate(coefficients)]
    moments = tuple(sum((1 if power == 0 else k**power) * value
                        for k, value in enumerate(terms)) for power in range(3))
    f0, f1, f2 = moments
    mean = Fraction(f1, f0)
    return moments, Fraction(f0, q**len(matrix)), mean, Fraction(f2, f0) - mean**2


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ is required for compiled research query baseline tests")
    return partition.build_partition_baseline(tmp_path_factory.mktemp("weighted-partition-build"))


@pytest.fixture(scope="module")
def baseline(build):
    return partition.PartitionBaseline(build["library"])


def test_build_links_real_source_binary_command_and_logs(build):
    assert Path(build["library"]).is_absolute()
    for key, checksum in [("library", "library_sha256"), ("source", "source_sha256"),
                           ("compiler", "compiler_sha256"), ("stdout_log", "stdout_sha256"),
                           ("stderr_log", "stderr_sha256")]:
        assert hashlib.sha256(Path(build[key]).read_bytes()).hexdigest() == build[checksum]
    assert build["command"][-2:] == ["-o", build["library"]]
    assert json.loads((Path(build["library"]).parent / "build.json").read_text()) == build
    with pytest.raises(FileExistsError):
        partition.build_partition_baseline(Path(build["library"]).parent)


@pytest.mark.parametrize("matrix", [[], [[0]], [[7]], [[0, 0], [0, 0]],
    [[2, 0], [0, 3]], [[2, 5], [7, 11]], [[1, 1, 0], [0, 1, 0], [0, 0, 3]]])
@pytest.mark.parametrize("activity", [0, 1, Fraction(1, 7), Fraction(13, 2)])
def test_exact_queries_match_edge_set_oracle(baseline, matrix, activity):
    original = [row[:] for row in matrix]
    actual = baseline.evaluate(matrix, activity)
    moments, z, mean, variance = oracle(matrix, activity)
    assert actual["numerator_moments"] == moments
    assert (actual["partition"], actual["mean"], actual["variance"]) == (z, mean, variance)
    assert actual["denominator"] == Fraction(activity).denominator**len(matrix)
    assert actual["modulus"] > max(actual["bounds"])
    assert actual["status"] == "COMPLETE" and actual["exact"] is True
    assert matrix == original
    for prime, residues in zip(actual["primes"], actual["residue_rows"]):
        assert residues == tuple(value % prime for value in moments)
    components = sum(actual[name] for name in ("setup_seconds", "reduction_seconds", "call_seconds", "crt_seconds"))
    assert 0 <= components <= actual["total_seconds"]


def test_activity_denominator_divisible_by_a_crt_prime(baseline):
    prime = 2147483647
    actual = baseline.evaluate([[2, 3], [5, 7]], Fraction(1, prime))
    assert actual["primes"][0] == prime
    assert actual["residue_rows"][0] == (29, 58, 116)
    assert actual["numerator_moments"] == (prime**2 + 17*prime + 29, 17*prime + 58, 17*prime + 116)


def test_seeded_dense_and_sparse_queries_against_exhaustive_oracle(baseline):
    rng = random.Random(2026090703)
    for n in range(1, 5):
        for _ in range(5):
            matrix = [[rng.randrange(6) if rng.random() < 0.7 else 0 for _ in range(n)] for _ in range(n)]
            activity = Fraction(rng.randrange(11), rng.randrange(1, 12))
            actual = baseline.evaluate(matrix, activity)
            moments, z, mean, variance = oracle(matrix, activity)
            assert actual["numerator_moments"] == moments
            assert (actual["partition"], actual["mean"], actual["variance"]) == (z, mean, variance)


def test_arbitrary_weights_and_rational_numerator_reduce_before_ctypes(baseline):
    matrix = [[10**150 + 7, 2**409], [True, 10**120 + 3]]
    activity = Fraction(10**45 + 1, 10**31 + 7)
    actual = baseline.evaluate(matrix, activity)
    assert actual["numerator_moments"] == oracle(matrix, activity)[0]
    assert len(actual["primes"]) > 20


@pytest.mark.parametrize("modulus", [3, 101, 2147483647, 2147483629, 2147450881, 2147450879])
def test_c_abi_canonical_boundaries_folded_and_generic_reduction(baseline, modulus):
    # The arithmetic also works for composite odd moduli: this test exercises
    # both sides of the c<=32768 folding cutoff without pretending primality.
    matrix = [[modulus - 1, modulus - 2], [modulus - 3, modulus - 1]]
    p, q = modulus - 1, modulus - 2
    data = (ctypes.c_uint32 * 4)(*(v for row in matrix for v in row))
    out = (ctypes.c_uint32 * 3)()
    assert baseline._function(data, 2, modulus, p, q, out) == 0
    coefficients = edge_set_coefficients(matrix)
    terms = [value * p**k * q**(2-k) for k, value in enumerate(coefficients)]
    expected = [sum((1 if j == 0 else k**j) * value for k, value in enumerate(terms)) % modulus
                for j in range(3)]
    assert list(out) == expected


@pytest.mark.parametrize("matrix, activity, error", [
    ([[1, 2]], 1, ValueError), ([[]], 1, ValueError), ([[1]*21 for _ in range(21)], 1, ValueError),
    ([[1.0]], 1, TypeError), ([[-1]], 1, ValueError), ([["3"]], 1, TypeError),
    (None, 1, TypeError), ("1", 1, TypeError), ([[1]], -1, ValueError),
    ([[1]], Fraction(-1, 3), ValueError), ([[1]], 0.5, TypeError), ([[1]], "1/2", TypeError),
])
def test_invalid_input_rejected_before_c_call(baseline, matrix, activity, error, monkeypatch):
    def fail(*args):
        pytest.fail("invalid Python input reached C++")
    monkeypatch.setattr(baseline, "_function", fail)
    with pytest.raises(error):
        baseline.evaluate(matrix, activity)


def test_c_abi_rejects_unsafe_inputs(baseline):
    out = (ctypes.c_uint32 * 3)(9, 9, 9)
    value = (ctypes.c_uint32 * 1)(101)
    assert baseline._function(value, 21, 101, 1, 1, out) == 1
    assert baseline._function(value, 1, 101, 1, 1, out) == 1
    assert baseline._function(None, 1, 101, 1, 1, out) == 1
    assert baseline._function(None, 0, 2**31, 1, 1, out) == 1
    assert baseline._function(None, 0, 100, 1, 1, out) == 1
    assert baseline._function(None, 0, 101, 101, 1, out) == 1
    assert baseline._function(None, 0, 101, 1, 101, out) == 1
    assert list(out) == [9, 9, 9]


def test_compiled_failure_or_corrupt_residue_is_not_an_exact_result(baseline, monkeypatch):
    monkeypatch.setattr(baseline, "_function", lambda *args: 2)
    with pytest.raises(RuntimeError, match="status 2"):
        baseline.evaluate([[1]], 1)
    def corrupt(matrix, n, prime, p, q, out):
        out[0] = prime
        return 0
    monkeypatch.setattr(baseline, "_function", corrupt)
    with pytest.raises(ValueError, match="noncanonical"):
        baseline.evaluate([[1]], 1)
