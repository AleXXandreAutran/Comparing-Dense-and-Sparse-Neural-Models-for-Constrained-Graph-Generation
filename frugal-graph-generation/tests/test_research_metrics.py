import json
import random

import networkx as nx
import numpy as np
import pytest

from frugal_graphs.research_metrics import (algebraic_connectivity, dynamic_metrics, evaluate_samples,
                              failure_resilience, fit_baseline, generate_baseline,
                              import_external_samples, profile_operation)


def record(graph, family=0):
    return {'n': len(graph), 'edges': np.asarray(list(graph.edges()), dtype=np.int64).reshape(-1, 2), 'family': family}


@pytest.mark.parametrize('graph,expected', [(nx.empty_graph(0), 0), (nx.empty_graph(1), 0),
                                          (nx.empty_graph(3), 0), (nx.path_graph(2), 2),
                                          (nx.path_graph(3), 1), (nx.complete_graph(5), 5)])
def test_algebraic_connectivity_known_spectra(graph, expected):
    assert algebraic_connectivity(graph) == pytest.approx(expected)


def test_failure_resilience_extremes_and_seed():
    graph = nx.cycle_graph(6)
    intact = failure_resilience(graph, probability=0)
    destroyed = failure_resilience(graph, probability=1)
    assert intact['connected_rate'] == 1
    assert intact['largest_component_fraction'] == 1
    assert destroyed['connected_rate'] == 0
    assert destroyed['largest_component_fraction'] == pytest.approx(1 / 6)
    assert failure_resilience(graph, seed=91) == failure_resilience(graph, seed=91)


def test_identical_distributions_have_zero_distance_with_variable_sizes():
    records = [record(nx.path_graph(4)), record(nx.cycle_graph(6)), record(nx.complete_graph(3))]
    result = evaluate_samples(records, records, records, budget={3: 3, 4: 3, 6: 6})
    for statistic in ['degree', 'clustering', 'spectral']:
        assert result[f'{statistic}_mmd2'] == pytest.approx(0, abs=1e-12)
        assert result[f'{statistic}_hist_wasserstein'] == pytest.approx(0, abs=1e-12)
    assert result['valid_rate'] == 1
    assert result['wl_novel_fraction_proxy'] == 0


def test_statistics_invariant_under_node_permutation():
    graph = nx.barbell_graph(4, 2)
    mapping = dict(zip(range(len(graph)), np.random.default_rng(9).permutation(len(graph))))
    permuted = nx.relabel_nodes(graph, mapping)
    first, second = record(graph), record(permuted)
    result = evaluate_samples([first], [second], [first], 100)
    for statistic in ['degree', 'clustering', 'spectral']:
        assert result[f'{statistic}_mmd2'] == pytest.approx(0, abs=1e-12)


def test_bandwidth_uses_only_training():
    training = [record(nx.path_graph(5)), record(nx.complete_graph(5))]
    first = evaluate_samples([record(nx.cycle_graph(5))], [record(nx.path_graph(5))], training, 10)
    second = evaluate_samples([record(nx.star_graph(4))], [record(nx.complete_graph(5))], training, 10)
    for statistic in ['degree', 'clustering', 'spectral']:
        assert first[f'{statistic}_bandwidth'] == second[f'{statistic}_bandwidth']


def test_invalid_samples_are_included_in_validity_denominator():
    path = record(nx.path_graph(4))
    duplicate = {**path, 'edges': [[0, 1], [1, 0], [1, 2], [2, 3]]}
    loop = {**path, 'edges': [[0, 0], [0, 1], [1, 2], [2, 3]]}
    complete = record(nx.complete_graph(4))
    result = evaluate_samples([path, duplicate, loop, complete], [path], [path], 3, max_degree=2)
    assert result['sample_count'] == 4
    assert result['simple_rate'] == 0.5
    assert result['valid_rate'] == 0.25
    assert result['budget_violation_rate'] == 0.25
    assert result['degree_violation_rate'] == 0.25


def test_planarity_is_reported_separately():
    result = evaluate_samples([record(nx.complete_graph(5))], [record(nx.path_graph(5))],
                              [record(nx.path_graph(5))], 10)
    assert result['valid_rate'] == 1
    assert result['reference_is_planar']
    assert result['planarity_constraint_valid_rate'] == 0


def test_family_metrics_and_per_record_budgets():
    samples = [record(nx.path_graph(4), family=0), record(nx.cycle_graph(6), family=1)]
    result = evaluate_samples(samples, samples, samples, [3, 6], [2, 2])
    assert result['by_family']['0']['valid_rate'] == 1
    assert result['by_family']['1']['valid_rate'] == 1


def test_dynamic_churn_geometry_and_connectivity():
    previous = [[0, 1], [1, 2]]
    current = [[0, 1], [0, 2]]
    positions = [[0, 0], [1, 0], [2, 0]]
    result = dynamic_metrics(previous, current, positions, radius=1.1, budget=2, max_degree=2)
    assert result['churn_edges'] == 2
    assert result['churn_fraction'] == pytest.approx(2 / 3)
    assert result['range_violations'] == 1
    assert result['connected']
    assert result['lambda2'] == pytest.approx(1)
    assert not result['valid']


def test_empty_singleton_and_malformed_inputs():
    empty = record(nx.empty_graph(0))
    singleton = record(nx.empty_graph(1))
    assert evaluate_samples([empty], [empty], [empty], 0)['valid_rate'] == 0
    assert evaluate_samples([singleton], [singleton], [singleton], 0)['valid_rate'] == 1
    with pytest.raises(ValueError):
        evaluate_samples([], [singleton], [singleton], 0)
    with pytest.raises(ValueError):
        evaluate_samples([{'n': 2, 'edges': [[0, float('nan')]]}], [singleton], [singleton], 0)


def test_profile_separates_passes_and_exposes_same_payload():
    outputs = []
    def operation():
        output = (random.random(), float(np.random.random()))
        outputs.append(output)
        return output
    result = profile_operation(operation, repetitions=3, warmup=1, return_result=True)
    assert len(outputs) == 5
    assert len(set(outputs)) == 1
    assert result['result'] == outputs[-1]
    assert result['latency_ms_p25'] <= result['latency_ms_median'] <= result['latency_ms_p75']
    assert result['rss_sampled_peak_bytes'] >= result['rss_baseline_bytes'] > 0
    assert result['rss_peak_delta_bytes'] >= 0
    assert result['cuda_peak_allocated_bytes'] is None
    assert result['cuda_peak_reserved_bytes'] is None


def test_profile_caller_reset_for_private_generator():
    state = {}
    def reset():
        state['rng'] = np.random.default_rng(7)
    result = profile_operation(lambda: state['rng'].integers(10000), repetitions=2,
                               warmup=0, reset=reset, return_result=True)
    assert result['result'] == np.random.default_rng(7).integers(10000)


def test_profile_rejects_unknown_device_and_invalid_counts():
    with pytest.raises(ValueError, match='device'):
        profile_operation(lambda: None, device='unknown')
    with pytest.raises(ValueError, match='counts'):
        profile_operation(lambda: None, repetitions=1.5)


def test_profile_failure_cleans_polling_thread():
    import threading
    before = len(threading.enumerate())
    calls = []
    def operation():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('sample failed')
    with pytest.raises(RuntimeError, match='sample failed'):
        profile_operation(operation, repetitions=1, warmup=0)
    assert len(threading.enumerate()) == before


def test_fitted_baselines_depend_only_on_training():
    training = [record(nx.cycle_graph(8)), record(nx.path_graph(8))]
    fitted = fit_baseline('degree_prior', training, family=0, n=12)
    assert len(fitted['degree_profile']) == 12
    assert fitted == fit_baseline('degree_prior', training, family=0, n=12)
    block = fit_baseline('fitted_sbm', training, family=0, n=12)
    assert 0 < block['p_in'] < 1 and 0 < block['p_out'] < 1
    with pytest.raises(ValueError):
        fit_baseline('fitted_sbm', training, family=1, n=12)


@pytest.mark.parametrize('name', ['random', 'degree_prior', 'fitted_sbm'])
def test_baselines_are_reproducible_and_obey_constraints(name):
    training = [record(nx.cycle_graph(8)), record(nx.path_graph(8))]
    fitted = fit_baseline(name, training, family=0, n=8)
    edges = generate_baseline(name, n=8, budget=10, seed=8, family=0, max_degree=3, training=fitted)
    repeated = generate_baseline(name, n=8, budget=10, seed=8, family=0, max_degree=3, training=fitted)
    np.testing.assert_array_equal(edges, repeated)
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from(edges)
    assert nx.is_connected(graph)
    assert graph.number_of_edges() <= 10
    assert max(dict(graph.degree()).values()) <= 3


def external_document():
    return {'metadata': {'method': 'DiGress', 'upstream_url': 'https://github.com/cvignac/DiGress',
                         'repo_commit': 'a' * 40, 'config': {'name': 'sbm'}, 'split_sha256': 'b' * 64,
                         'seeds': [11, 23, 37], 'timing_scope': 'complete sampling including correction',
                         'hardware': 'test CPU', 'command': 'python main.py'},
            'samples': [{'n': 3, 'edges': [[0, 1], [1, 2]], 'family': 0}]}


def test_external_adapter_checks_provenance_split_and_counts(tmp_path):
    path = tmp_path / 'external.json'
    document = external_document()
    path.write_text(json.dumps(document))
    imported = import_external_samples(path, 'DiGress', expected_counts={0: 1}, expected_split_sha256='b' * 64)
    assert len(imported['samples']) == 1
    assert len(imported['artifact_sha256']) == 64
    with pytest.raises(ValueError, match='split hash'):
        import_external_samples(path, 'DiGress', expected_split_sha256='c' * 64)
    with pytest.raises(ValueError, match='count'):
        import_external_samples(path, 'DiGress', expected_counts=2)
    del document['metadata']['repo_commit']
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='metadata'):
        import_external_samples(path, 'DiGress')


def test_external_adapter_preserves_invalid_outputs_for_evaluation(tmp_path):
    path = tmp_path / 'external.json'
    document = external_document()
    document['samples'][0]['edges'].append([0, 0])
    path.write_text(json.dumps(document))
    imported = import_external_samples(path, 'DiGress')
    assert len(imported['samples'][0]['edges']) == 3
    valid = [record(nx.path_graph(3))]
    result = evaluate_samples(imported['samples'], valid, valid, budget=2)
    assert result['simple_rate'] == 0
    assert result['valid_rate'] == 0
