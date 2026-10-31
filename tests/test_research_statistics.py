"""Guard research normalizations, fair comparisons, and retained observations."""

import importlib
import json
from math import comb, factorial

import numpy as np
import pytest

from statistics_helpers import (
    haar_permanent_mean_square, ks_distance, load_permanent_samples,
    sample_key, save_permanent_samples, split_fit_evaluation,
)


def research_module(name):
    # Research drivers are deliberately outside the installed package/sdist.
    if importlib.util.find_spec(name) is None:
        pytest.skip(f'{name} is available only in a research checkout')
    pytest.importorskip('scipy')
    return importlib.import_module(name)


@pytest.mark.parametrize('n', [0, 1, 2, 7, 13, 17])
def test_full_haar_mean_uses_representation_dimension(n):
    expected = 1 if n == 0 else 1 / comb(2 * n - 1, n)
    assert haar_permanent_mean_square(n) == expected
    assert haar_permanent_mean_square(np.int64(n)) == expected
    if n > 1:
        assert expected != factorial(n) / n**n


def test_haar_two_by_two_second_moment_matches_direct_integral():
    # Up to row/column phases, Haar U(2) has entries sqrt(u), sqrt(1-u),
    # -sqrt(1-u), sqrt(u), where u is uniform. Its permanent is 2u-1.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    integrated_square = np.dot(weights, nodes**2) / 2
    assert integrated_square == pytest.approx(haar_permanent_mean_square(2))


@pytest.mark.parametrize('n', [-1, 1.5, True, np.bool_(True), '3'])
def test_haar_mean_rejects_invalid_order(n):
    with pytest.raises(ValueError):
        haar_permanent_mean_square(n)


def test_held_out_split_is_disjoint_deterministic_and_does_not_use_global_rng():
    data = np.arange(2000, dtype=float)
    training, evaluation = split_fit_evaluation(data, seed=19, max_evaluation=31)
    np.random.seed(963)
    repeated = split_fit_evaluation(data, seed=19, max_evaluation=31)
    assert len(evaluation) == 31
    assert len(training) == 1969
    assert set(training).isdisjoint(evaluation)
    np.testing.assert_array_equal(np.sort(np.concatenate([training, evaluation])), data)
    np.testing.assert_array_equal(training, repeated[0])
    np.testing.assert_array_equal(evaluation, repeated[1])
    assert not np.array_equal(evaluation, split_fit_evaluation(data, seed=20)[1])


@pytest.mark.parametrize('data', [np.arange(7), np.ones((8, 1)), np.ones(8) * 1j,
                                  np.array([np.nan] + list(range(8)))])
def test_split_rejects_unusable_observations(data):
    with pytest.raises(ValueError):
        split_fit_evaluation(data)


@pytest.mark.parametrize('limit', [0, -1, True, np.bool_(True), 1.5])
def test_split_rejects_invalid_evaluation_limit(limit):
    with pytest.raises(ValueError):
        split_fit_evaluation(np.arange(12), max_evaluation=limit)


def test_ks_distance_counts_both_sides_of_empirical_jumps():
    assert ks_distance([0.1, 0.4, 0.9], lambda x: x) == pytest.approx(4 / 15)
    assert ks_distance([0.1, 0.1], lambda x: x) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        ks_distance([1j], lambda x: x)
    with pytest.raises(ValueError):
        ks_distance([0.1, 0.2], lambda x: x + 1)


def test_archive_round_trip_preserves_complex_samples_and_provenance(tmp_path):
    path = tmp_path / 'samples.npz'
    values = {sample_key('Haar', 2): np.array([1 + 2j, 3 - 4j]),
              sample_key('GOE', 2): np.array([2.0, -3.0])}
    metadata = {'seed': 42, 'normalization': 'full Haar U(n)', 'schema_version': 2}
    save_permanent_samples(path, values, metadata)
    actual, source = load_permanent_samples(path)
    assert source == metadata
    for key in values:
        np.testing.assert_array_equal(actual[key], values[key])
    with pytest.raises(ValueError):
        save_permanent_samples(path, {sample_key('Haar', 2): [np.nan]}, metadata)
    # Invalid checkpoints do not damage the last valid archive.
    assert load_permanent_samples(path)[1] == metadata
    np.savez(tmp_path / 'missing-metadata.npz', values=np.arange(8))
    with pytest.raises(ValueError, match='provenance'):
        load_permanent_samples(tmp_path / 'missing-metadata.npz')


def test_threshold_reanalysis_changes_the_actual_observation_classification():
    module = research_module('test_anticoncentration')
    # For n=3 the corrected mean is 0.1, while the former value is 2/9.
    samples = np.sqrt([0.005, 0.015, 0.02, 0.04])
    corrected = module.threshold_statistics(samples, 3, haar_permanent_mean_square)
    legacy = module.threshold_statistics(samples, 3, lambda n: factorial(n) / n**n)
    assert corrected['mean/n²'] == 0.75
    assert legacy['mean/n²'] == 0.25
    assert corrected['mean_source'] == 'theoretical'
    assert module.threshold_statistics(samples, 3)['mean_source'] == 'same-sample estimate'


def test_anticoncentration_archive_path_never_calls_gpu(tmp_path, monkeypatch):
    module = research_module('test_anticoncentration')
    source = tmp_path / 'source.npz'
    ensembles = ['Haar', 'Ginibre-C', 'Ginibre-R', 'GUE', 'GOE']
    arrays = {sample_key(ens, 3): np.linspace(0.01, 0.5, 16) for ens in ensembles}
    save_permanent_samples(source, arrays, {'seed': 51, 'origin': 'synthetic regression fixture'})
    monkeypatch.setattr(module, 'perm_gpu', lambda *a, **k: pytest.fail('reanalysis must not generate samples'))
    output = tmp_path / 'analysis'
    module.main(['--input-samples', str(source), '--output-dir', str(output), '--no-figures'])
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['by_ensemble']['Haar']['3']['mean'] == pytest.approx(0.1)
    assert summary['metadata']['seed'] == 51
    assert summary['metadata']['source_sha256']
    assert summary['metadata']['script_sha256']


def test_anticoncentration_accepts_a_haar_only_archive(tmp_path, monkeypatch):
    module = research_module('test_anticoncentration')
    source = tmp_path / 'haar.npz'
    save_permanent_samples(source, {sample_key('Haar', 3): np.linspace(0.01, 0.5, 16)}, {'seed': 9})
    monkeypatch.setattr(module, 'perm_gpu', lambda *a, **k: pytest.fail('retained data must be reused'))
    module.main(['--input-samples', str(source), '--output-dir', str(tmp_path / 'analysis'), '--no-figures'])
    result = json.loads((tmp_path / 'analysis' / 'summary.json').read_text())
    assert list(result['by_ensemble']) == ['Haar']


def test_stable_models_share_training_and_evaluation_observations(monkeypatch):
    module = research_module('fit_stable')
    data = np.linspace(-4, 5, 40)
    expected_train, expected_eval = split_fit_evaluation(data, seed=7, max_evaluation=5)
    fitted = []
    evaluated = []

    def student_fit(values):
        fitted.append(values.copy())
        return 5, 0, 1

    def stable_fit(values):
        fitted.append(values.copy())
        return 2, 0, 0, 1

    def record_distance(values, cdf):
        evaluated.append(values.copy())
        return ks_distance(values, cdf)

    monkeypatch.setattr(module.stats.t, 'fit', student_fit)
    monkeypatch.setattr(module.stats.levy_stable, '_fitstart', stable_fit)
    monkeypatch.setattr(module, 'ks_distance', record_distance)
    result = module.fit_all(data, 'fixture', seed=7, max_evaluation=5)
    assert len(fitted) == 2
    assert len(evaluated) == 3
    for values in fitted:
        np.testing.assert_array_equal(values, expected_train)
    for values in evaluated:
        np.testing.assert_array_equal(values, expected_eval)
    assert result['mu'] == pytest.approx(expected_train.mean())
    assert result['training_count'] == 35 and result['evaluation_count'] == 5
    assert all('pvalue' not in key and not key.endswith('_p') for key in result)
    with pytest.raises(ValueError):
        module.fit_all(data + 1j, 'must select real part explicitly')


def test_haar_distribution_models_share_the_same_split(monkeypatch):
    module = research_module('unitary_perm_stats')
    data = np.linspace(0.001, 0.3, 40)
    expected_train, expected_eval = split_fit_evaluation(data, seed=31, max_evaluation=7)
    fitted = []
    evaluated = []

    class Model:
        def fit(self, values, *, floc):
            assert floc == 0
            fitted.append(values.copy())
            return 0, 1

        def cdf(self, values, *params):
            evaluated.append(values.copy())
            return values

    for name in ('lognorm', 'gamma', 'weibull_min', 'expon', 'rayleigh'):
        monkeypatch.setattr(module.stats, name, Model())
    fits = module.fit_distributions(data, seed=31, max_evaluation=7)
    assert len(fitted) == len(evaluated) == 5
    for values in fitted:
        np.testing.assert_array_equal(values, expected_train)
    for values in evaluated:
        np.testing.assert_array_equal(values, np.sort(expected_eval))
    assert all(fit['evaluation_count'] == 7 and 'ks_pvalue' not in fit for fit in fits.values())


def test_haar_study_retained_samples_and_old_summary_guard(tmp_path, monkeypatch):
    module = research_module('unitary_perm_stats')
    source = tmp_path / 'source.npz'
    values = np.linspace(0.01, 0.6, 40) * np.exp(0.4j)
    save_permanent_samples(source, {sample_key('Haar', 3): values}, {'seed': 63})
    monkeypatch.setattr(module, 'permanent_batch', lambda *a, **k: pytest.fail('raw samples must be reused'))
    monkeypatch.setattr(module, 'permanent_fast', lambda *a, **k: pytest.fail('reanalysis must not recompute exponential DFT reference'))
    output = tmp_path / 'new.json'
    results = module.run_study(3, 40, output, input_samples=source)
    assert results['by_n']['3']['stats_perm_squared']['theoretical_haar_mean_perm_sq'] == pytest.approx(0.1)
    assert 'not recomputed' in results['by_n']['3']['dft']['status']
    raw, _ = load_permanent_samples(output.with_suffix('.npz'))
    np.testing.assert_array_equal(raw[sample_key('Haar', 3)], values)
    legacy = tmp_path / 'old.json'
    legacy.write_text(json.dumps({'metadata': {}, 'by_n': {'3': {}}}))
    with pytest.raises(ValueError, match='raw-sample provenance'):
        module.run_study(3, 40, legacy)


def test_explicit_cpu_backend_ignores_a_loaded_cuda_library(monkeypatch):
    module = research_module('unitary_perm_stats')
    monkeypatch.setattr(module, 'HAS_CUDA', True)
    monkeypatch.setattr(module, 'permanent_batch_gpu', lambda *a, **k: pytest.fail('CPU backend must not launch CUDA'))
    monkeypatch.setattr(module, 'permanent_fast', module.permanent_ryser_complex)
    matrices = np.array([[[1, 2j], [3, 4]], [[2, -1], [5, 6]]], dtype=complex)
    np.testing.assert_allclose(module.permanent_batch(matrices, backend='cpu'), [4 + 6j, 7])
