"""Focused checks for the CPU Schur cofactor/Glynn implementation."""

import math
from pathlib import Path
import random
import re

import pytest

import perm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _checked_in_schur_values():
    text = (REPOSITORY_ROOT / "schur_permanents.txt").read_text(
        encoding="utf-8"
    )
    return {
        int(order): int(value)
        for order, value in re.findall(
            r"^n=(\d+):\s+(-?\d+)\s*$", text, re.MULTILINE
        )
    }


ELIGIBLE_GRAHAM_LEHMER_VALUES = [
    (order, value)
    for order, value in _checked_in_schur_values().items()
    if order >= 5 and perm.is_prime(order)
]


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


def test_python_glynn_matches_ryser_on_random_matrices():
    rng = random.Random(20260814)

    for order in range(2, 7):
        for modulus in (3, 5, 11, 101):
            for _ in range(5):
                matrix = [
                    [rng.randrange(-modulus, 2 * modulus) for _ in range(order)]
                    for _ in range(order)
                ]
                assert perm._glynn_permanent_mod_p(
                    matrix, modulus
                ) == perm.ryser_permanent_mod_p(matrix, modulus)


@pytest.mark.parametrize(
    "order,modulus",
    [(3, 7), (5, 11), (7, 29), (9, 19)],
)
def test_python_schur_cofactor_matches_retained_ryser(order, modulus):
    omega = perm.nth_root_of_unity_mod_p(order, modulus)
    assert perm._schur_permanent_mod_p_glynn(
        order, omega, modulus
    ) == perm._schur_permanent_mod_p_ryser(order, omega, modulus)


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize(
    "order,modulus",
    [(3, 7), (5, 11), (7, 29), (11, 23), (13, 53)],
)
def test_native_prime_order_path_matches_ryser_and_current_glynn(
    order, modulus
):
    omega = perm.nth_root_of_unity_mod_p(order, modulus)
    matrix = perm.build_schur_matrix_mod_p(order, omega, modulus)
    expected_ryser = perm.perm_c_ext.ryser_permanent_mod_p(
        matrix, modulus, 1
    )
    expected_glynn = perm._schur_permanent_mod_p_glynn(
        order, omega, modulus
    )

    assert expected_glynn == expected_ryser
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, 1
    ) == expected_ryser
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, 4
    ) == expected_ryser


def _rotation_complement_orbit_count(length):
    divisors = [divisor for divisor in range(1, length + 1)
                if length % divisor == 0]

    def phi(value):
        return sum(
            math.gcd(candidate, value) == 1
            for candidate in range(1, value + 1)
        )

    rotation_fixed = sum(
        phi(divisor) * 2 ** (length // divisor) for divisor in divisors
    )
    complement_fixed = sum(
        phi(divisor) * 2 ** (length // divisor)
        for divisor in divisors if divisor % 2 == 0
    )
    return (rotation_fixed + complement_fixed) // (2 * length)


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize(
    "order", [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
)
def test_native_necklace_orbit_weights_cover_every_sign_word(order):
    representatives, represented_words = (
        perm.perm_c_ext.schur_necklace_orbit_stats(order)
    )
    assert representatives == _rotation_complement_orbit_count(order - 1)
    assert represented_words == 1 << (order - 1)


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize("order", [0, 2, 9, 25, 64])
def test_native_necklace_stats_reject_unsupported_orders(order):
    with pytest.raises(ValueError, match="odd prime"):
        perm.perm_c_ext.schur_necklace_orbit_stats(order)


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize("order", [19, 23, 29])
def test_native_necklace_serial_and_openmp_results_are_identical(order):
    modulus = perm.find_primes_congruent_to_1_mod_n(
        order, 1, perm._SCHUR_CRT_PRIME_FLOOR
    )[0]
    omega = perm.nth_root_of_unity_mod_p(order, modulus)
    serial = perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, 1
    )
    max_threads = perm.perm_c_ext.get_max_threads()
    thread_counts = {
        1,
        min(2, max_threads),
        min(3, max_threads),
        min(8, max_threads),
        min(64, max_threads),
    }
    for threads in thread_counts:
        assert perm.perm_c_ext.schur_permanent_mod_p(
            order, omega, modulus, threads
        ) == serial


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize(
    "order,expected",
    [
        (order, value)
        for order, value in _checked_in_schur_values().items()
        if order <= 31 and perm.is_prime(order)
    ],
)
def test_native_prime_order_residues_match_known_values_through_31(
    order, expected
):
    modulus = perm.find_primes_congruent_to_1_mod_n(
        order, 1, perm._SCHUR_CRT_PRIME_FLOOR
    )[0]
    omega = perm.nth_root_of_unity_mod_p(order, modulus)
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, min(8, perm.perm_c_ext.get_max_threads())
    ) == expected % modulus


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize(
    "order,modulus,root_exponent,threads",
    [(11, 67, 7, 3), (19, 191, 11, 8), (23, 47, 13, 3)],
)
def test_native_necklace_accepts_alternate_primitive_roots(
    order, modulus, root_exponent, threads
):
    expected = _checked_in_schur_values()[order]
    root = perm.nth_root_of_unity_mod_p(order, modulus)
    omega = pow(root, root_exponent, modulus)
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, threads
    ) == expected % modulus


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize("order", [9, 27])
def test_composite_orders_retain_glynn_fallback(order):
    expected = _checked_in_schur_values()[order]
    modulus = perm.find_primes_congruent_to_1_mod_n(
        order, 1, perm._SCHUR_CRT_PRIME_FLOOR
    )[0]
    omega = perm.nth_root_of_unity_mod_p(order, modulus)
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, min(8, perm.perm_c_ext.get_max_threads())
    ) == expected % modulus


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize("order,modulus,threads", [(9, 19, 1), (19, 191, 2)])
def test_native_schur_serial_and_openmp_match_full_ryser(
    order, modulus, threads
):
    omega = perm.nth_root_of_unity_mod_p(order, modulus)
    matrix = perm.build_schur_matrix_mod_p(order, omega, modulus)
    expected = perm.perm_c_ext.ryser_permanent_mod_p(
        matrix, modulus, threads
    )
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, threads
    ) == expected


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize(
    "order,omega,modulus",
    [
        (4, 2, 5),       # even order
        (3, 1, 2),       # even modulus
        (3, 2, 5),       # omega is not an order-dividing root
    ],
)
def test_native_schur_precondition_fallback_matches_ryser(
    order, omega, modulus
):
    matrix = perm.build_schur_matrix_mod_p(order, omega, modulus)
    expected = perm.perm_c_ext.ryser_permanent_mod_p(matrix, modulus, 1)
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, 1
    ) == expected


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
def test_native_schur_glynn_handles_signed_64_bit_modulus():
    modulus = (1 << 63) - 1
    matrix = perm.build_schur_matrix_mod_p(3, 1, modulus)
    expected = perm.perm_c_ext.ryser_permanent_mod_p(matrix, modulus, 2)
    assert perm.perm_c_ext.schur_permanent_mod_p(3, 1, modulus, 2) == expected


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize("order", [7, 11])
def test_native_necklace_handles_signed_64_bit_modulus(order):
    modulus = (1 << 63) - 1
    expected = math.factorial(order) % modulus
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, 1, modulus, 1
    ) == expected
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, 1, modulus, 4
    ) == expected


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
def test_native_parallel_necklace_handles_degenerate_root():
    order = 19
    modulus = 341
    expected = math.factorial(order) % modulus
    for threads in (1, 2, 3, min(8, perm.perm_c_ext.get_max_threads())):
        assert perm.perm_c_ext.schur_permanent_mod_p(
            order, 1, modulus, threads
        ) == expected


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
def test_native_necklace_handles_overflow_safe_prime_modulus():
    order = 23
    expected = _checked_in_schur_values()[order]
    modulus = perm.find_primes_congruent_to_1_mod_n(order, 1, 1 << 32)[0]
    omega = perm.nth_root_of_unity_mod_p(order, modulus)
    assert perm.perm_c_ext.schur_permanent_mod_p(
        order, omega, modulus, 8
    ) == expected % modulus


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
@pytest.mark.parametrize(
    "modulus",
    [3_037_000_499, 3_037_000_501, (1 << 63) - 1],
)
def test_native_row_product_dispatch_is_exact_across_safe_boundary(modulus):
    matrix = [[modulus - 1, 0], [0, modulus - 1]]
    assert perm.perm_c_ext.ryser_permanent_mod_p(matrix, modulus, 1) == 1


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
def test_native_row_product_dispatch_matches_randomized_bigint_reference():
    rng = random.Random(20260815)

    for modulus in (3_037_000_499, 3_037_000_501, (1 << 63) - 1):
        for order in range(2, 6):
            matrix = [
                [rng.randrange(-modulus, modulus) for _ in range(order)]
                for _ in range(order)
            ]
            expected = perm.ryser_permanent_mod_p(matrix, modulus)
            assert perm.perm_c_ext.ryser_permanent_mod_p(
                matrix, modulus, min(order, 2)
            ) == expected


@pytest.mark.parametrize(
    "order,expected_count",
    [(31, 3), (37, 3), (39, 3), (41, 4), (43, 4), (45, 3), (47, 4)],
)
def test_fast_crt_plan_has_exact_coverage_and_minimal_pass_count(
    order, expected_count
):
    target = 2 * perm.schur_permanent_bound(order) + 1
    primes, prime_product, congruence = perm._plan_schur_exact_crt(
        order, target, perm._SCHUR_CRT_PRIME_FLOOR
    )
    theorem_modulus = congruence[1] if congruence is not None else 1

    assert len(primes) == expected_count
    assert min(primes) >= 1 << 30
    assert all(prime % order == 1 and perm.is_prime(prime) for prime in primes)
    assert all(math.gcd(prime, theorem_modulus) == 1 for prime in primes)
    assert theorem_modulus * prime_product >= target
    assert theorem_modulus * math.prod(primes[:-1]) < target


def test_order_41_plan_stays_on_measured_faster_modulus_tier():
    target = 2 * perm.schur_permanent_bound(41) + 1
    primes, _, congruence = perm._plan_schur_exact_crt(
        41, target, perm._SCHUR_CRT_PRIME_FLOOR
    )

    assert congruence is not None
    assert len(primes) == 4
    assert all(prime <= 3_037_000_499 for prime in primes)


@pytest.mark.parametrize("order", [0, 1, 2, 4, 6, 8, 10, 46])
def test_graham_lehmer_congruence_is_not_needed_for_trivial_orders(order):
    assert perm._graham_lehmer_schur_congruence(order) is None


@pytest.mark.parametrize("order,expected", ELIGIBLE_GRAHAM_LEHMER_VALUES)
def test_graham_lehmer_congruence_matches_all_checked_prime_orders(
    order, expected
):
    residue, modulus = perm._graham_lehmer_schur_congruence(order)
    assert modulus == order ** 3
    assert residue == math.factorial(order) % modulus
    assert expected % modulus == residue


@pytest.mark.parametrize(("order", "divisor"), THEOREM4_DIVISORS.items())
def test_graham_lehmer_prime_power_divisor_matches_known_orders(
    order, divisor
):
    assert perm._graham_lehmer_schur_congruence(order) == (0, divisor)

    expected = _checked_in_schur_values().get(order)
    if expected is not None:
        assert expected % divisor == 0


@pytest.mark.parametrize("order", [3, 9, 39, 45])
def test_composite_orders_use_theorem_seed_in_prime_plan(order):
    target = 2 * perm.schur_permanent_bound(order) + 1
    congruence = perm._graham_lehmer_schur_congruence(order)
    assert congruence is not None
    remaining_target = (target + congruence[1] - 1) // congruence[1]
    expected_primes, expected_product = perm._plan_schur_crt_primes(
        order, remaining_target, perm._SCHUR_CRT_PRIME_FLOOR
    )
    primes, prime_product, planned_congruence = perm._plan_schur_exact_crt(
        order, target, perm._SCHUR_CRT_PRIME_FLOOR
    )

    assert planned_congruence == congruence
    assert primes == expected_primes
    assert prime_product == expected_product


def test_order_45_cpu_plan_uses_theorem4_seed_and_three_primes():
    target = 2 * perm.schur_permanent_bound(45) + 1
    primes, prime_product, congruence = perm._plan_schur_exact_crt(
        45, target, perm._SCHUR_CRT_PRIME_FLOOR
    )

    assert congruence == (0, 3 ** 20 * 5 ** 9)
    assert len(primes) == 3
    assert congruence[1] * prime_product >= target
    assert congruence[1] * math.prod(primes[:-1]) < target


@pytest.mark.parametrize("order,expected", ELIGIBLE_GRAHAM_LEHMER_VALUES)
def test_fast_theorem_reconstructs_checked_values_without_kernel_modulus(
    monkeypatch, order, expected
):
    kernel_moduli = []
    reconstruction_moduli = []
    original_crt = perm.chinese_remainder_theorem

    def fake_residue(actual_order, modulus, num_threads=0):
        assert actual_order == order
        kernel_moduli.append(modulus)
        return expected % modulus

    def recording_crt(residues, moduli):
        reconstruction_moduli.extend(moduli)
        return original_crt(residues, moduli)

    monkeypatch.setattr(perm, "schur_permanent_mod_p_fast", fake_residue)
    monkeypatch.setattr(perm, "chinese_remainder_theorem", recording_crt)

    assert perm.schur_permanent_fast(order, parallel=False) == expected
    theorem_modulus = order ** 3
    assert theorem_modulus not in kernel_moduli
    assert reconstruction_moduli[-1] == theorem_modulus


@pytest.mark.parametrize(
    "order,expected",
    [
        (order, value)
        for order, value in _checked_in_schur_values().items()
        if order in THEOREM4_DIVISORS
    ],
)
def test_fast_composite_seed_reconstructs_checked_values_without_kernel_modulus(
    monkeypatch, order, expected
):
    kernel_moduli = []
    reconstruction_moduli = []
    original_crt = perm.chinese_remainder_theorem

    def fake_residue(actual_order, modulus, num_threads=0):
        assert actual_order == order
        kernel_moduli.append(modulus)
        return expected % modulus

    def recording_crt(residues, moduli):
        reconstruction_moduli.extend(moduli)
        return original_crt(residues, moduli)

    monkeypatch.setattr(perm, "schur_permanent_mod_p_fast", fake_residue)
    monkeypatch.setattr(perm, "chinese_remainder_theorem", recording_crt)

    assert perm.schur_permanent_fast(order, parallel=False) == expected
    theorem_modulus = THEOREM4_DIVISORS[order]
    assert theorem_modulus not in kernel_moduli
    assert reconstruction_moduli[-1] == theorem_modulus


@pytest.mark.parametrize("order", [15, 37])
def test_reference_exact_path_also_appends_theorem_only_at_reconstruction(
    monkeypatch, order
):
    expected = _checked_in_schur_values()[order]
    kernel_moduli = []
    reconstruction_moduli = []
    original_crt = perm.chinese_remainder_theorem

    def fake_residue(actual_order, modulus):
        assert actual_order == order
        kernel_moduli.append(modulus)
        return expected % modulus

    def recording_crt(residues, moduli):
        reconstruction_moduli.extend(moduli)
        return original_crt(residues, moduli)

    monkeypatch.setattr(perm, "schur_permanent_mod_p", fake_residue)
    monkeypatch.setattr(perm, "chinese_remainder_theorem", recording_crt)

    assert perm.schur_permanent(order) == expected
    theorem_modulus = perm._graham_lehmer_schur_congruence(order)[1]
    assert theorem_modulus not in kernel_moduli
    assert reconstruction_moduli[-1] == theorem_modulus


def test_fast_exact_path_uses_wide_prime_floor(monkeypatch):
    calls = []
    original_planner = perm._plan_schur_crt_primes

    def recording_planner(order, target, min_prime):
        calls.append(min_prime)
        return original_planner(order, target, min_prime)

    monkeypatch.setattr(perm, "_plan_schur_crt_primes", recording_planner)
    assert perm.schur_permanent_fast(3, parallel=False) == -3
    assert calls == [perm._SCHUR_CRT_PRIME_FLOOR]


@pytest.mark.skipif(not perm.HAS_C_EXT, reason="OpenMP C extension is unavailable")
def test_native_schur_rejects_unsafe_dimensions_and_moduli():
    with pytest.raises(ValueError, match="0 <= n < 64"):
        perm.perm_c_ext.schur_permanent_mod_p(64, 1, 3, 1)
    with pytest.raises(ValueError, match="positive"):
        perm.perm_c_ext.schur_permanent_mod_p(3, 1, 0, 1)


@pytest.mark.parametrize(
    "order,expected",
    [(1, 1), (3, -3), (5, -5), (7, -105), (9, 81), (11, 6765),
     (13, 175747), (15, 30375)],
)
def test_fast_wide_crt_preserves_known_schur_values(order, expected):
    assert perm.schur_permanent_fast(order, parallel=False) == expected
