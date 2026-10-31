"""Exact application cases for the binary permanent API."""

import itertools
import math

import pytest

import perm_binary
import perm_polynomial
from perm_binary_applications import (
    anisotropic_grid_partition,
    axis_dimer_profile,
    axis_dimer_profile_mod,
    bregman_extremal_matrix,
    cyclic_latin_square,
    derangement_matrix,
    evaluate_polynomial,
    grid_biadjacency,
    grid_dimer_count,
    latin_transversal_count,
    weighted_grid_biadjacency,
)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 2), 2),
        ((2, 4), 5),
        ((4, 4), 36),
        ((4, 6), 281),
        ((2, 2, 2), 9),
        ((2, 2, 3), 32),
        ((2, 2, 4), 121),
        ((2, 3, 4), 1845),
    ],
)
def test_open_grid_dimer_counts(shape, expected):
    assert grid_dimer_count(shape, backend="python") == expected


def test_periodic_grid_dimer_counts():
    assert grid_dimer_count(
        (4, 4), periodic_axes=(0, 1), backend="python"
    ) == 272
    assert grid_dimer_count(
        (4, 4, 2), periodic_axes=(0, 1), backend="python"
    ) == 589185


def test_odd_vertex_grid_has_no_close_packed_dimer_covering():
    assert grid_dimer_count((3, 3, 3), backend="python") == 0
    assert anisotropic_grid_partition(
        (3, 3, 3), 7, parallel=False
    ) == 0
    assert axis_dimer_profile((3, 3, 3)) == (0,)


def test_odd_volume_shortcuts_still_validate_execution_options():
    with pytest.raises(ValueError, match="backend"):
        grid_dimer_count((3, 3, 3), backend="nonsense")
    with pytest.raises(TypeError, match="parallel"):
        grid_dimer_count((3, 3, 3), parallel=0)
    with pytest.raises(ValueError, match="num_threads"):
        grid_dimer_count((3, 3, 3), num_threads=-1)
    with pytest.raises(TypeError, match="parallel"):
        anisotropic_grid_partition((3, 3, 3), 7, parallel=0)
    with pytest.raises(ValueError, match="num_threads"):
        anisotropic_grid_partition(
            (3, 3, 3), 7, num_threads=-1
        )


@pytest.mark.parametrize(
    "shape,periodic_axes,message",
    [
        ((3, 4), (0,), "even length"),
        ((2, 4), (0,), "parallel bonds"),
        ((4, 4), (2,), "outside"),
        ((4, 4), (0, 0), "duplicates"),
    ],
)
def test_periodic_grid_geometry_is_validated(shape, periodic_axes, message):
    with pytest.raises(ValueError, match=message):
        grid_biadjacency(shape, periodic_axes=periodic_axes)


def test_gemini_4_by_4_by_5_application_matrix_is_well_formed():
    matrix = grid_biadjacency((4, 4, 5), periodic_axes=(0, 1))
    assert len(matrix) == 40
    assert all(len(row) == 40 for row in matrix)
    assert {entry for row in matrix for entry in row} == {0, 1}
    assert sorted(map(sum, matrix)) == [5] * 16 + [6] * 24
    column_degrees = [sum(row[column] for row in matrix) for column in range(40)]
    assert sorted(column_degrees) == [5] * 16 + [6] * 24


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 2, 2), (4, 0, 4, 0, 1)),
        ((2, 2, 3), (8, 0, 16, 0, 8, 0, 0)),
        ((2, 2, 4), (16, 0, 48, 0, 44, 0, 12, 0, 1)),
        (
            (2, 3, 4),
            (81, 0, 405, 0, 690, 0, 493, 0, 154, 0, 21, 0, 1),
        ),
    ],
)
def test_vertical_dimer_profiles(shape, expected):
    coefficients = axis_dimer_profile(shape)
    assert coefficients == expected
    assert sum(coefficients) == grid_dimer_count(shape, backend="python")


def test_anisotropic_cube_polynomial_matches_weighted_permanents():
    coefficients = axis_dimer_profile((2, 2, 2))
    # Z(w) = 4 + 4 w^2 + w^4 for the open 2x2x2 cube.
    for weight in (-2, 0, 1, 2, 5):
        expected = evaluate_polynomial(coefficients, weight)
        assert anisotropic_grid_partition(
            (2, 2, 2), weight, parallel=False
        ) == expected
        weighted = weighted_grid_biadjacency(
            (2, 2, 2), (1, 1, weight)
        )
        assert all(isinstance(entry, int) for row in weighted for entry in row)


def test_periodic_axis_profile_tracks_wrapped_edge_directions():
    coefficients = axis_dimer_profile(
        (4, 4), axis=0, periodic_axes=(0, 1)
    )
    assert coefficients == (16, 0, 64, 0, 112, 0, 64, 0, 16)
    assert sum(coefficients) == 272
    assert anisotropic_grid_partition(
        (4, 4), 2, axis=0, periodic_axes=(0, 1), parallel=False
    ) == evaluate_polynomial(coefficients, 2)


def test_profile_guard_points_large_cases_to_modular_backend():
    with pytest.raises(ValueError, match="axis_dimer_profile_mod"):
        axis_dimer_profile(
            (4, 4, 5), periodic_axes=(0, 1)
        )


@pytest.mark.parametrize("modulus", [9, 97, 2_147_483_647])
def test_modular_axis_profiles_match_exact_oracles(modulus):
    for shape in ((2, 2, 2), (2, 2, 3), (2, 2, 4)):
        exact = axis_dimer_profile(shape)
        assert axis_dimer_profile_mod(
            shape, modulus, backend="python"
        ) == tuple(coefficient % modulus for coefficient in exact)


def test_modular_periodic_profile_and_weighted_evaluations():
    modulus = 97
    coefficients = axis_dimer_profile_mod(
        (4, 4),
        modulus,
        axis=0,
        periodic_axes=(0, 1),
        backend="python",
    )
    assert coefficients == tuple(
        coefficient % modulus
        for coefficient in (16, 0, 64, 0, 112, 0, 64, 0, 16)
    )
    for weight in (-2, 0, 1, 2, 5):
        assert evaluate_polynomial(coefficients, weight) % modulus == (
            anisotropic_grid_partition(
                (4, 4),
                weight,
                axis=0,
                periodic_axes=(0, 1),
                parallel=False,
            )
            % modulus
        )


@pytest.mark.skipif(
    not perm_polynomial.HAS_POLYNOMIAL_EXT,
    reason="perm_polynomial_ext is not built",
)
def test_modular_axis_profile_native_integration():
    expected = axis_dimer_profile((4, 4), axis=0, periodic_axes=(0, 1))
    assert axis_dimer_profile_mod(
        (4, 4),
        2_147_483_647,
        axis=0,
        periodic_axes=(0, 1),
        backend="native",
        num_threads=3,
    ) == tuple(value % 2_147_483_647 for value in expected)


def test_modular_profile_odd_shortcut_still_validates_options():
    assert axis_dimer_profile_mod(
        (3, 3, 3), 9, backend="python"
    ) == (0,)
    with pytest.raises(ValueError, match="modulus"):
        axis_dimer_profile_mod((3, 3, 3), 10)
    with pytest.raises(ValueError, match="backend"):
        axis_dimer_profile_mod((3, 3, 3), 9, backend="nonsense")
    with pytest.raises(TypeError, match="parallel"):
        axis_dimer_profile_mod((3, 3, 3), 9, parallel=0)
    with pytest.raises(ValueError, match="num_threads"):
        axis_dimer_profile_mod((3, 3, 3), 9, num_threads=-1)


def test_derangement_matrices_have_subfactorial_permanents():
    derangements = [1, 0]
    for order in range(2, 10):
        derangements.append(
            (order - 1) * (derangements[-1] + derangements[-2])
        )
    for order, expected in enumerate(derangements):
        matrix = derangement_matrix(order)
        assert (
            perm_binary.binary_permanent(matrix, backend="python")
            == expected
        )


@pytest.mark.parametrize("block_size,block_count", [(2, 5), (3, 4), (5, 3)])
def test_bregman_extremal_blocks(block_size, block_count):
    matrix = bregman_extremal_matrix(block_size, block_count)
    expected = math.factorial(block_size) ** block_count
    assert perm_binary.binary_permanent(matrix, backend="python") == expected
    assert perm_binary.binary_permanent_bound(matrix) == expected


def _latin_transversal_oracle(square):
    order = len(square)
    count = 0
    for columns in itertools.permutations(range(order)):
        symbols = {
            square[row][columns[row]] for row in range(order)
        }
        count += int(len(symbols) == order)
    return count


@pytest.mark.parametrize(
    "order,expected", [(0, 1), (1, 1), (2, 0), (3, 3), (4, 0),
                       (5, 15), (6, 0), (7, 133), (8, 0), (9, 2025)]
)
def test_cyclic_latin_transversals_via_binary_inclusion_exclusion(
    order, expected
):
    square = cyclic_latin_square(order)
    assert _latin_transversal_oracle(square) == expected
    assert latin_transversal_count(
        square, backend="python"
    ) == expected


def test_latin_application_validates_structure_and_cost_guard():
    relabeled = [[10, 20, 30], [20, 30, 10], [30, 10, 20]]
    assert latin_transversal_count(
        relabeled, backend="python"
    ) == 3
    klein_four = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    # Cyclic order four has none, but even-order Latin squares need not.
    assert _latin_transversal_oracle(klein_four) == 8
    assert latin_transversal_count(
        klein_four, backend="python"
    ) == 8
    with pytest.raises(ValueError, match="row"):
        latin_transversal_count([[0, 0], [1, 1]], backend="python")
    with pytest.raises(ValueError, match="column"):
        latin_transversal_count([[0, 1], [0, 1]], backend="python")
    with pytest.raises(ValueError, match="inclusion-exclusion"):
        latin_transversal_count(
            cyclic_latin_square(11), backend="python"
        )


@pytest.mark.parametrize(
    "function,args,error",
    [
        (grid_biadjacency, ((),), ValueError),
        (grid_biadjacency, ((2, 0),), ValueError),
        (grid_biadjacency, ((3, 3),), ValueError),
        (weighted_grid_biadjacency, ((2, 2), (1,)), ValueError),
        (derangement_matrix, (-1,), ValueError),
        (bregman_extremal_matrix, (0, 2), ValueError),
        (bregman_extremal_matrix, (2, -1), ValueError),
        (cyclic_latin_square, (-1,), ValueError),
    ],
)
def test_application_input_validation(function, args, error):
    with pytest.raises(error):
        function(*args)
