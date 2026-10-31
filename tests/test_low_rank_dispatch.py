"""Definition-level oracles and production low-rank dispatch boundaries."""

import itertools
import math
import random

import numpy as np
import pytest

import perm


def definition(matrix):
    return sum(math.prod(matrix[i][j] for i, j in enumerate(columns))
               for columns in itertools.permutations(range(len(matrix))))


def factored_matrix(order, rank, seed):
    rng = random.Random(seed)
    entries = (-3, -2, -1, 1, 2, 3)
    u = [[rng.choice(entries) for _ in range(rank)] for _ in range(order)]
    v = [[rng.choice(entries) for _ in range(order)] for _ in range(rank)]
    return [[sum(u[i][k] * v[k][j] for k in range(rank))
             for j in range(order)] for i in range(order)]


def test_all_small_binary_matrices_and_small_characteristics():
    for n in range(4):
        for entries in itertools.product((0, 1), repeat=n * n):
            matrix = [list(entries[i * n:(i + 1) * n]) for i in range(n)]
            expected = definition(matrix)
            assert perm.matrix_permanent(matrix, parallel=False, method="low_rank") == expected
            for prime in (2, 3, 101):
                factors = perm._low_rank_factor_mod(matrix, prime, 4)
                assert all(sum(factors[0][i][k] * factors[1][k][j]
                               for k in range(len(factors[1]))) % prime == matrix[i][j] % prime
                           for i in range(n) for j in range(n))
                assert perm._low_rank_permanent_from_factors(n, prime, factors) == expected % prime


def test_signed_random_factors_and_row_column_symmetries():
    for n in range(1, 8):
        for rank in range(1, min(n, 4) + 1):
            matrix = factored_matrix(n, rank, 7000 + n * 10 + rank)
            expected = definition(matrix)
            transposed = list(map(list, zip(*matrix)))
            permuted = [row[1:] + row[:1] for row in reversed(matrix)]
            for argument in (matrix, transposed, permuted):
                assert perm.matrix_permanent(argument, parallel=False, method="low_rank") == expected
            assert perm.matrix_permanent(matrix, parallel=False, method="generic") == expected


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_numpy_extremes_stay_in_arbitrary_precision(dtype):
    extreme = np.iinfo(dtype).max
    matrix = np.array([[extreme, extreme - 1], [extreme - 2, extreme - 3]], dtype=dtype)
    with np.errstate(over="raise", invalid="raise"):
        assert perm.matrix_permanent(matrix, parallel=False, method="low_rank") == definition(matrix.tolist())


def test_exact_large_rank_one_selects_low_rank_without_generic_or_pool(monkeypatch, capsys):
    n = 40
    u = [-2] + [i % 3 + 1 for i in range(1, n)]
    v = [i % 5 + 1 for i in range(n)]
    matrix = [[x * y for y in v] for x in u]
    expected = math.factorial(n) * math.prod(u) * math.prod(v)
    monkeypatch.setattr(perm, "_ryser_permanent_with_backend", lambda *a, **k: pytest.fail("unexpected generic call"))
    monkeypatch.setattr(perm, "HAS_C_EXT", False)
    import multiprocessing
    monkeypatch.setattr(multiprocessing, "Pool", lambda *a, **k: pytest.fail("unnecessary process startup"))
    assert perm.matrix_permanent(matrix, verbose=True) == expected
    assert "low-rank" in capsys.readouterr().out


def test_crossover_skips_probe_for_small_and_parallel_expensive_ranks(monkeypatch):
    assert perm._low_rank_limit(18, 1) == 0
    assert perm._low_rank_limit(22, 1) == 4
    assert perm._low_rank_limit(22, 8) == 3
    assert perm._low_rank_limit(40, 64) == 4
    assert perm._low_rank_limit(100, 1) == 3  # Rank-4 coefficient layer exceeds the cap.
    monkeypatch.setattr(perm, "_low_rank_residues", lambda *a, **k: pytest.fail("small input must not be probed"))
    assert perm.matrix_permanent([[1, 2], [3, 4]], parallel=False) == 10


def test_generic_opt_out_and_high_rank_fallback(monkeypatch):
    calls = []
    def generic(matrix, prime, num_threads=0):
        calls.append(prime)
        return 1
    monkeypatch.setattr(perm, "_ryser_permanent_with_backend", generic)
    identity = [[int(i == j) for j in range(24)] for i in range(24)]
    assert perm.matrix_permanent(identity, parallel=False) == 1
    assert calls
    calls.clear()
    monkeypatch.setattr(perm, "_low_rank_residues", lambda *a, **k: pytest.fail("generic must bypass low-rank machinery"))
    assert perm.matrix_permanent(identity, parallel=False, method="generic") == 1
    assert calls


def test_exceptional_first_prime_does_not_certify_low_integer_rank(monkeypatch):
    n, first_prime = 20, 1_073_741_827
    # A=sI+J has rank one at p=s, but full rank at the other CRT primes.
    matrix = [[1 + first_prime * (i == j) for j in range(n)] for i in range(n)]
    expected = sum(math.comb(n, k) * math.factorial(n - k) * first_prime ** k
                   for k in range(n + 1))
    assert len(perm._low_rank_factor_mod(matrix, first_prime, 4)[1]) == 1
    calls = []
    def generic(argument, prime, num_threads=0):
        calls.append(prime)
        return expected % prime
    monkeypatch.setattr(perm, "_ryser_permanent_with_backend", generic)
    monkeypatch.setattr(perm, "_low_rank_permanent_from_factors", lambda *a: pytest.fail("preflight must finish before summation"))
    assert perm.matrix_permanent(matrix, parallel=False) == expected
    assert calls[0] == first_prime and len(calls) > 1


def test_rank_drop_at_a_later_prime_is_valid():
    prime = 1_073_741_827
    matrix = [[prime, 0], [0, -7]]
    assert perm.matrix_permanent(matrix, method="low_rank") == -7 * prime


def test_only_budget_exceptions_trigger_auto_fallback(monkeypatch):
    matrix = [[1] * 20 for _ in range(20)]
    monkeypatch.setattr(perm, "_low_rank_factor_mod", lambda *a: (_ for _ in ()).throw(RuntimeError("factor failed")))
    with pytest.raises(RuntimeError, match="factor failed"):
        perm.matrix_permanent(matrix, parallel=False)


def test_explicit_limits_and_backend_independence(monkeypatch):
    class Native:
        @staticmethod
        def get_max_threads():
            pytest.fail("explicit low rank must not call native controls")
    monkeypatch.setattr(perm, "HAS_C_EXT", True)
    monkeypatch.setattr(perm, "perm_c_ext", Native())
    identity = [[int(i == j) for j in range(5)] for i in range(5)]
    with pytest.raises(perm.LowRankLimitExceeded, match="rank is at least 5"):
        perm.matrix_permanent(identity, method="low_rank")
    assert perm.matrix_permanent([[1, 2], [3, 4]], method="low_rank") == 10
    monkeypatch.setattr(perm, "_LOW_RANK_MAX_STATES", 2)
    with pytest.raises(perm.LowRankLimitExceeded, match="states"):
        perm.matrix_permanent([[1, 0], [0, 1]], method="low_rank")


@pytest.mark.parametrize("method,error", [(None, TypeError), (True, TypeError), ("cuda", ValueError)])
def test_method_validation_precedes_base_cases(method, error):
    with pytest.raises(error, match="method"):
        perm.matrix_permanent([], method=method)


def test_existing_positional_and_integer_contracts_preserved():
    assert perm.matrix_permanent([[True]], False, False, 0) == 1
    assert perm.matrix_permanent([], method="low_rank") == 1
    with pytest.raises(ValueError, match="square"):
        perm.matrix_permanent([[1, 2]], method="low_rank")
    with pytest.raises(TypeError, match="entries must be integers"):
        perm.matrix_permanent([[1.5]], method="low_rank")
