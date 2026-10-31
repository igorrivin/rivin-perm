"""CPU-only tests for CUDA Schur CRT modulus and root planning."""

import math
from pathlib import Path
import re

import pytest

import perm_cuda


EXPECTED_GPU_PRIMES = {
    41: [4294966153, 4294965251, 4294964923],
    43: [4294966163, 4294965733, 4294965131, 4294964959],
    45: [4294966591, 4294966231, 4294965691],
    47: [4294967161, 4294964341, 4294962367, 4294961897],
}


THEOREM4_DIVISORS = {
    3: 3,
    9: 81,
    15: 30375,
    21: 750141,
    25: 15625,
    27: 1594323,
    33: 235782657,
    35: 1313046875,
    39: 3502727631,
    45: 3 ** 20 * 5 ** 9,
}


def _checked_in_schur_values():
    text = (Path(__file__).resolve().parents[1] / "schur_permanents.txt").read_text(
        encoding="utf-8"
    )
    return {
        int(size): int(value)
        for size, value in re.findall(
            r"^n=(\d+):\s+(-?\d+)\s*$", text, re.MULTILINE
        )
    }


@pytest.mark.parametrize("n", [41, 43, 45, 47])
def test_large_prime_plan_is_deterministic_valid_and_covers_bound(n):
    bound = perm_cuda.schur_permanent_bound(n)

    primes = perm_cuda._plan_schur_crt_primes(n, bound)

    congruence = perm_cuda._graham_lehmer_schur_congruence(n)
    theorem_modulus = congruence[1] if congruence is not None else 1

    assert primes == EXPECTED_GPU_PRIMES[n]
    assert primes == perm_cuda._plan_schur_crt_primes(n, bound)
    assert all(1 < prime < 2**32 for prime in primes)
    assert all((prime - 1) % n == 0 for prime in primes)
    assert all(perm_cuda._is_prime(prime) for prime in primes)
    assert all((prime - 1) ** 2 < 2**64 for prime in primes)
    assert all(math.gcd(prime, theorem_modulus) == 1 for prime in primes)
    assert theorem_modulus * math.prod(primes) >= 2 * bound + 1
    assert theorem_modulus * math.prod(primes[:-1]) < 2 * bound + 1


@pytest.mark.parametrize("n", [41, 43, 45, 47])
def test_roots_have_exact_requested_order(n):
    for prime in EXPECTED_GPU_PRIMES[n]:
        omega = perm_cuda._compute_omega(n, prime)

        assert pow(omega, n, prime) == 1
        assert all(
            pow(omega, n // factor, prime) != 1
            for factor in perm_cuda._distinct_prime_factors(n)
        )


def test_planner_stops_on_the_exact_integer_coverage_condition():
    first, second = perm_cuda._find_primes_for_schur(41, 2)
    theorem_modulus = 41 ** 3
    first_combined_modulus = theorem_modulus * first

    assert perm_cuda._plan_schur_crt_primes(
        41, (first_combined_modulus - 1) // 2
    ) == [first]
    assert perm_cuda._plan_schur_crt_primes(
        41, (first_combined_modulus + 1) // 2
    ) == [
        first,
        second,
    ]


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (n, value)
        for n, value in _checked_in_schur_values().items()
        if n >= 5 and perm_cuda._is_prime(n)
    ],
)
def test_graham_lehmer_congruence_matches_checked_in_prime_orders(n, expected):
    residue, modulus = perm_cuda._graham_lehmer_schur_congruence(n)

    assert modulus == n ** 3
    assert residue == math.factorial(n) % modulus
    assert expected % modulus == residue


@pytest.mark.parametrize("n", [0, 1, 2, 4, 6, 8, 10, 46])
def test_graham_lehmer_congruence_is_not_needed_for_trivial_orders(n):
    assert perm_cuda._graham_lehmer_schur_congruence(n) is None


@pytest.mark.parametrize(("n", "divisor"), THEOREM4_DIVISORS.items())
def test_graham_lehmer_prime_power_divisor_matches_known_orders(
    n, divisor
):
    assert perm_cuda._graham_lehmer_schur_congruence(n) == (0, divisor)

    expected = _checked_in_schur_values().get(n)
    if expected is not None:
        assert expected % divisor == 0


def test_order_45_uses_three_gpu_primes_with_theorem4_seed():
    divisor = 3 ** 20 * 5 ** 9
    bound = perm_cuda.schur_permanent_bound(45)

    assert perm_cuda._graham_lehmer_schur_congruence(45) == (0, divisor)
    assert perm_cuda._plan_schur_crt_primes(45, bound) == (
        perm_cuda._find_primes_for_schur(45, 3)
    )


def test_order_47_keeps_four_gpu_primes_and_factorial_seed():
    assert perm_cuda._graham_lehmer_schur_congruence(47) == (
        math.factorial(47) % (47 ** 3),
        47 ** 3,
    )
    assert perm_cuda._plan_schur_crt_primes(
        47, perm_cuda.schur_permanent_bound(47)
    ) == EXPECTED_GPU_PRIMES[47]


def test_primality_and_root_inputs_are_checked():
    assert not perm_cuda._is_prime(2**32 - 1)
    with pytest.raises(ValueError, match="prime"):
        perm_cuda._compute_omega(3, 10)
    with pytest.raises(ValueError, match="mod"):
        perm_cuda._compute_omega(5, 13)


def test_permanent_cuda_uses_large_prime_plan_without_a_device(monkeypatch):
    used_primes = []

    monkeypatch.setattr(perm_cuda, "is_cuda_available", lambda: True)

    def residue(n, omega, prime):
        used_primes.append(prime)
        return (-3) % prime

    monkeypatch.setattr(perm_cuda, "schur_permanent_modp_cuda", residue)

    assert perm_cuda.permanent_cuda(3, verbose=False) == -3
    assert used_primes == perm_cuda._find_primes_for_schur(3, 1)


def test_prime_order_reconstruction_keeps_theorem_modulus_off_gpu(monkeypatch):
    expected = _checked_in_schur_values()[41]
    used_moduli = []

    monkeypatch.setattr(perm_cuda, "is_cuda_available", lambda: True)

    def residue(n, omega, modulus):
        used_moduli.append(modulus)
        return expected % modulus

    monkeypatch.setattr(perm_cuda, "schur_permanent_modp_cuda", residue)

    assert perm_cuda.permanent_cuda(41, verbose=False) == expected
    assert used_moduli == EXPECTED_GPU_PRIMES[41]
    assert 41 ** 3 not in used_moduli


def test_composite_order_reconstruction_keeps_theorem_modulus_off_gpu(
    monkeypatch,
):
    order = 45
    divisor = THEOREM4_DIVISORS[order]
    expected = divisor
    used_moduli = []

    monkeypatch.setattr(perm_cuda, "is_cuda_available", lambda: True)

    def residue(n, omega, modulus):
        assert n == order
        used_moduli.append(modulus)
        return expected % modulus

    monkeypatch.setattr(perm_cuda, "schur_permanent_modp_cuda", residue)

    assert perm_cuda.permanent_cuda(order, verbose=False) == expected
    assert used_moduli == EXPECTED_GPU_PRIMES[order]
    assert divisor not in used_moduli


def test_theorem_can_cover_small_prime_order_without_a_gpu(monkeypatch):
    monkeypatch.setattr(
        perm_cuda,
        "is_cuda_available",
        lambda: pytest.fail("no CUDA availability check should be needed"),
    )
    monkeypatch.setattr(
        perm_cuda,
        "schur_permanent_modp_cuda",
        lambda *args: pytest.fail("the theorem modulus is not a GPU modulus"),
    )

    assert perm_cuda.permanent_cuda(5, verbose=False) == -5
