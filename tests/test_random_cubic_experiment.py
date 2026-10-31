"""Reproducibility checks for the random cubic-graph experiment."""

from fractions import Fraction
from math import factorial

import numpy as np

import experiment_random_cubic as experiment


def test_first_conditioned_configuration_is_reproducible():
    rng = np.random.default_rng(experiment.DEFAULT_SEED)
    matrix, row_masks, proposals = (
        experiment.draw_simple_cubic_biadjacency(rng)
    )
    assert proposals == 4
    assert row_masks.tolist() == [
        388, 4136, 41984, 655616, 6176, 131, 786433, 131, 2082, 49216,
        139272, 66112, 268, 102400, 16912, 263184, 532484, 395264,
        16464, 67072,
    ]
    assert np.all(matrix.sum(axis=0) == 3)
    assert np.all(matrix.sum(axis=1) == 3)
    assert experiment.count_four_cycles(row_masks) == 9
    assert experiment.component_count(row_masks) == 1


def test_exact_conditioning_probability_and_ensemble_mean():
    graph_count, acceptance, fixed_count, mean = (
        experiment.exact_ensemble_theory()
    )
    assert graph_count == int(
        "77705104689340239554388061645133412621507133440000"
    )
    assert fixed_count == int(
        "57068107225598546939327028600928761"
    )
    assert acceptance == Fraction(
        graph_count * 6 ** 40, factorial(60)
    )
    assert mean == Fraction(
        243290351856499068530815227193433139,
        136162134522625612217034046746779,
    )


def test_tied_average_ranks():
    ranks = experiment.average_ranks(np.array([30, 10, 20, 20]))
    assert ranks.tolist() == [4.0, 1.0, 2.5, 2.5]


def test_single_sample_summary_has_defined_degenerate_statistics():
    summary = experiment.summarize(
        np.array([2388]), np.array([9]), np.array([1])
    )
    assert summary["mean"] == 2388
    assert summary["skewness"] is None
    assert summary["four_cycles"]["pearson_r"] is None
    assert summary["four_cycles"]["variance_explained_by_group_means"] is None
