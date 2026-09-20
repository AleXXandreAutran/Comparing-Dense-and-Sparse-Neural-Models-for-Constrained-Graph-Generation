from __future__ import annotations

import numpy as np
import pytest
import torch
from frugal_graphs.cvae import benchmark_generator
from frugal_graphs.models import GraphCVAE

@pytest.mark.parametrize('kind', ['neural', 'empirical', 'random'])
def test_generator_benchmark_smoke(kind):
    torch.set_num_threads(1)
    model = GraphCVAE(n=6, hidden=8, latent=3, layers=1) if kind == 'neural' else None
    empirical = np.zeros((2, 15)) if kind == 'empirical' else None
    records = benchmark_generator(model, empirical, 6, 8, family=1, batch_sizes=(1, 4), repetitions=3)
    assert [record['batch_size'] for record in records] == [1, 4]
    for record in records:
        for scope in ('forward', 'generation'):
            assert 0 < record[f'{scope}_ms_p25'] <= record[f'{scope}_ms_median'] <= record[f'{scope}_ms_p75']
        assert record['ms_per_graph'] == pytest.approx(record['generation_ms_median'] / record['batch_size'])
        assert record['graphs_per_second'] * record['ms_per_graph'] == pytest.approx(1000)
    if model is not None:
        assert model.training

def test_model_evaluation_state_restored():
    model = GraphCVAE(n=6, hidden=8, latent=3, layers=1).eval()
    benchmark_generator(model, None, 6, 8, batch_sizes=(1,), repetitions=1)
    assert not model.training

@pytest.mark.parametrize('kwargs', [{'n': 0}, {'m': 4}, {'family': 2}, {'family': True}, {'temperature': -1}, {'temperature': np.nan}, {'batch_sizes': (0,)}, {'batch_sizes': ()}, {'batch_sizes': (True,)}, {'repetitions': 0}, {'repetitions': 1.5}, {'empirical_logits_or_None': np.zeros((2, 14))}, {'empirical_logits_or_None': np.full((2, 15), np.inf)}])
def test_invalid_benchmark_arguments(kwargs):
    arguments = dict(model_or_None=None, empirical_logits_or_None=None, n=6, m=8, batch_sizes=(1,), repetitions=1)
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        benchmark_generator(**arguments)

def test_mutually_exclusive_generators_and_node_count():
    model = GraphCVAE(n=6, hidden=8, latent=3, layers=1)
    with pytest.raises(ValueError, match='choose'):
        benchmark_generator(model, np.zeros((2, 15)), 6, 8)
    with pytest.raises(ValueError, match='node count'):
        benchmark_generator(model, None, 7, 8)
from itertools import combinations, product
from frugal_graphs.graphs import perturb_scores, project_connected, project_topk

def is_connected(adjacency):
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbour in np.flatnonzero(adjacency[node]):
            if int(neighbour) not in seen:
                seen.add(int(neighbour))
                frontier.append(int(neighbour))
    return len(seen) == len(adjacency)

def feasible_masks(n, m):
    edges = list(combinations(range(n), 2))
    masks = []
    for chosen in combinations(range(len(edges)), m):
        adjacency = np.zeros((n, n), dtype=np.float32)
        for edge in chosen:
            a, b = edges[edge]
            adjacency[a, b] = adjacency[b, a] = 1
        if is_connected(adjacency):
            mask = np.zeros(len(edges), dtype=float)
            mask[list(chosen)] = 1
            masks.append(mask)
    return np.asarray(masks)

@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_exhaustive_graph_search_matches_global_optimum(n):
    edge_count = n * (n - 1) // 2
    rng = np.random.default_rng(1000 + n)
    score_cases = [rng.normal(size=edge_count) for _ in range(12)]
    score_cases += [np.zeros(edge_count), -np.ones(edge_count)]
    score_cases += [rng.integers(-2, 3, size=edge_count) for _ in range(12)]
    if n <= 4:
        score_cases += [np.asarray(bits, dtype=float) for bits in product((0, 1), repeat=edge_count)]
    cases = np.stack(score_cases)
    indices = np.triu_indices(n, 1)
    for budget in range(n - 1, edge_count + 1):
        feasible = feasible_masks(n, budget)
        answers = project_connected(cases, n, budget)
        observed = np.sum(answers[:, indices[0], indices[1]] * cases, axis=1)
        optimum = np.max(cases @ feasible.T, axis=1)
        np.testing.assert_allclose(observed, optimum, rtol=1e-12, atol=1e-12)
        for answer in answers:
            assert is_connected(answer)
            assert int(answer.sum()) == 2 * budget

@pytest.mark.parametrize('n', [1, 2, 6, 20])
def test_every_budget_feasible(n):
    rng = np.random.default_rng(7)
    scores = rng.normal(size=n * (n - 1) // 2)
    for m in range(n - 1, len(scores) + 1):
        answer = project_connected(scores, n, m)
        assert answer.dtype == np.float32
        assert answer.shape == (n, n)
        assert is_connected(answer)
        assert np.array_equal(answer, answer.T)
        assert np.all(np.diag(answer) == 0)
        assert set(np.unique(answer)) <= {0, 1}
        assert int(answer.sum()) == 2 * m

def test_topk_can_disconnect_and_connected_decoder_repairs():
    scores = np.asarray([10, 9, 1, 8, 0, -1], dtype=float)
    unconstrained = project_topk(scores, 4, 3)
    constrained = project_connected(scores, 4, 3)
    assert not is_connected(unconstrained)
    assert is_connected(constrained)
    assert int(unconstrained.sum()) == int(constrained.sum()) == 6
    assert constrained[0, 3] == 1

@pytest.mark.parametrize('decoder', [project_connected, project_topk])
def test_distinct_scores_are_permutation_equivariant(decoder):
    n = 7
    rng = np.random.default_rng(24)
    scores = rng.normal(size=n * (n - 1) // 2)
    indices = np.triu_indices(n, 1)
    weights = np.zeros((n, n))
    weights[indices] = scores
    weights += weights.T
    for _ in range(20):
        permutation = rng.permutation(n)
        relabelled = weights[np.ix_(permutation, permutation)][indices]
        for m in (n - 1, n + 2, len(scores)):
            original = decoder(scores, n, m)
            changed = decoder(relabelled, n, m)
            np.testing.assert_array_equal(changed, original[np.ix_(permutation, permutation)])

@pytest.mark.parametrize('decoder', [project_connected, project_topk])
def test_ties_are_deterministic_but_not_permutation_equivariant(decoder):
    scores = np.zeros(6)
    answer = decoder(scores, 4, 3)
    np.testing.assert_array_equal(answer, decoder(scores, 4, 3))
    permutation = np.asarray([1, 0, 2, 3])
    assert not np.array_equal(answer, answer[np.ix_(permutation, permutation)])

@pytest.mark.parametrize('decoder', [project_connected, project_topk])
def test_batch_matches_individual_and_supports_empty_batch(decoder):
    scores = np.random.default_rng(13).normal(size=(8, 15))
    batched = decoder(scores, 6, 8)
    assert batched.dtype == np.float32
    assert batched.shape == (8, 6, 6)
    np.testing.assert_array_equal(batched, np.stack([decoder(row, 6, 8) for row in scores]))
    assert decoder(np.empty((0, 15)), 6, 8).shape == (0, 6, 6)

@pytest.mark.parametrize('scores,n,m,error', [(np.zeros(6), 0, 0, ValueError), (np.zeros(6), 4.0, 3, TypeError), (np.zeros(6), True, 3, TypeError), (np.zeros(6), 4, 3.0, TypeError), (np.zeros(6), 4, True, TypeError), (np.zeros(6), 4, np.asarray([3]), TypeError), (np.zeros(6), 4, 2, ValueError), (np.zeros(6), 4, 7, ValueError), (np.zeros(5), 4, 3, ValueError), (np.zeros((2, 3, 6)), 4, 3, ValueError), (np.asarray(0.0), 4, 3, ValueError), (np.asarray([0, 1, 2, 3, 4, np.nan]), 4, 3, ValueError), (np.asarray([0, 1, 2, 3, 4, np.inf]), 4, 3, ValueError), (np.zeros(6, dtype=complex), 4, 3, TypeError), (np.asarray(['a'] * 6), 4, 3, TypeError)])
def test_invalid_connected_arguments(scores, n, m, error):
    with pytest.raises(error):
        project_connected(scores, n, m)

def test_topk_budget_can_be_zero_but_not_negative():
    assert not project_topk(np.zeros(6), 4, 0).any()
    with pytest.raises(ValueError):
        project_topk(np.zeros(6), 4, -1)

def test_gumbel_seed_and_temperature_semantics():
    scores = np.arange(12).reshape(2, 6)
    actual = perturb_scores(scores, np.random.default_rng(7), 0.25)
    expected = scores + 0.25 * np.random.default_rng(7).gumbel(size=scores.shape)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(perturb_scores(scores, np.random.default_rng(7), 0), scores)
    assert not np.array_equal(actual, scores)

@pytest.mark.parametrize('temperature', [-1, np.nan, np.inf])
def test_gumbel_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        perturb_scores(np.zeros(3), np.random.default_rng(7), temperature)

def test_gumbel_invalid_values():
    with pytest.raises(ValueError):
        perturb_scores([np.nan], np.random.default_rng(7))
    with pytest.raises(TypeError):
        perturb_scores([0], np.random.default_rng(7), True)
import networkx as nx
from frugal_graphs.graphs import canonical_order, generate_dataset, load_dataset, save_dataset
from frugal_graphs.graphs import evaluate_graphs, fit_metric_reference, graph_statistics

@pytest.fixture(scope='module')
def tiny_dataset():
    return generate_dataset(n=10, m=18, train_per_family=5, val_per_family=3, test_per_family=3, reference_per_family=3, seed=73)

def test_dataset_shapes_constraints_reproducibility(tiny_dataset):
    repeated = generate_dataset(n=10, m=18, train_per_family=5, val_per_family=3, test_per_family=3, reference_per_family=3, seed=73)
    for split, sample in tiny_dataset.items():
        assert sample['adj'].shape[1:] == (10, 10)
        assert sample['adj'].dtype == np.float32
        assert sample['family'].dtype == np.int64
        assert set(sample['family']) == {0, 1}
        np.testing.assert_array_equal(sample['adj'], repeated[split]['adj'])
        for adjacency in sample['adj']:
            graph = nx.from_numpy_array(adjacency)
            assert graph.number_of_edges() == 18
            assert nx.is_connected(graph)
            np.testing.assert_array_equal(adjacency, adjacency.T)
            assert not np.diag(adjacency).any()

def test_no_isomorphic_duplicates_across_splits(tiny_dataset):
    previous = []
    for sample in tiny_dataset.values():
        graphs = [nx.from_numpy_array(adjacency) for adjacency in sample['adj']]
        for graph in graphs:
            assert not any((nx.is_isomorphic(graph, other) for other in previous))
        previous.extend(graphs)

def test_permutation_invariant_statistics(tiny_dataset):
    adjacency = tiny_dataset['test']['adj']
    permutation = np.random.default_rng(12).permutation(adjacency.shape[1])
    original = graph_statistics(adjacency)
    reordered = graph_statistics(adjacency[:, permutation][:, :, permutation])
    for key in original:
        np.testing.assert_allclose(original[key], reordered[key], atol=1e-12)

def test_statistics_against_networkx(tiny_dataset):
    adjacency = tiny_dataset['test']['adj']
    statistics = graph_statistics(adjacency)
    for index, array in enumerate(adjacency):
        graph = nx.from_numpy_array(array)
        assert statistics['triangles'][index] == sum(nx.triangles(graph).values()) / 3
        assert statistics['efficiency'][index] == pytest.approx(nx.global_efficiency(graph))

def test_metric_self_distance_and_validity(tiny_dataset):
    train = tiny_dataset['train']['adj']
    test = tiny_dataset['test']['adj']
    reference = fit_metric_reference(train)
    result = evaluate_graphs(test, test, train, reference, m=18)
    for feature in ('degree', 'clustering', 'spectral'):
        assert result[f'{feature}_mmd2'] == pytest.approx(0.0, abs=1e-12)
    assert result['valid_rate'] == 1.0
    assert result['triangles_mean_abs_error'] == 0.0
    assert result['efficiency_mean_abs_error'] == 0.0
    assert result['wl_unique_proxy_rate'] <= 1.0
    assert result['wl_novel_proxy_rate'] <= 1.0

def test_disconnected_graphs_are_included():
    adjacency = np.zeros((2, 4, 4))
    adjacency[1, 0, 1] = adjacency[1, 1, 0] = 1
    result = graph_statistics(adjacency)
    np.testing.assert_array_equal(result['connected'], [False, False])
    np.testing.assert_allclose(result['efficiency'], [0.0, 1.0 / 6])
    assert np.isfinite(result['spectral']).all()

def test_validity_detects_self_loops():
    adjacency = nx.to_numpy_array(nx.complete_graph(4))[None]
    adjacency[0, 0, 0] = 1.0
    assert not graph_statistics(adjacency)['simple'][0]

def test_dataset_roundtrip(tiny_dataset, tmp_path):
    path = tmp_path / 'dataset.npz'
    save_dataset(tiny_dataset, path)
    recovered = load_dataset(path)
    for split in tiny_dataset:
        for key in tiny_dataset[split]:
            np.testing.assert_array_equal(recovered[split][key], tiny_dataset[split][key])

def test_bfs_order_is_deterministic_and_preserves_graph():
    adjacency = nx.to_numpy_array(nx.path_graph(5))
    first = canonical_order(adjacency)
    np.testing.assert_array_equal(first, canonical_order(adjacency))
    assert nx.is_isomorphic(nx.from_numpy_array(first), nx.from_numpy_array(adjacency))

def test_invalid_sizes_rejected():
    with pytest.raises(ValueError):
        generate_dataset(n=5, m=3)
    with pytest.raises(ValueError):
        generate_dataset(train_per_family=-1)
from frugal_graphs.models import EdgeDenoiser

def symmetric_random(batch=4, n=7, soft=False):
    if soft:
        upper = torch.rand(batch, n, n).triu(1)
    else:
        upper = torch.randint(0, 2, (batch, n, n)).float().triu(1)
    return upper + upper.transpose(-1, -2)

def full_edge_scores(scores, n):
    indices = torch.triu_indices(n, n, offset=1)
    matrix = torch.zeros((scores.shape[0], n, n), dtype=scores.dtype)
    matrix[:, indices[0], indices[1]] = scores
    return matrix + matrix.transpose(-1, -2)

@pytest.mark.parametrize('layers,common', [(3, True), (0, True), (3, False)])
@pytest.mark.parametrize('soft', [False, True])
def test_denoiser_equivariant_under_relabelling(layers, common, soft):
    torch.set_num_threads(1)
    torch.manual_seed(59)
    model = EdgeDenoiser(n=7, hidden=12, layers=layers, use_common_neighbors=common).eval()
    adjacency = symmetric_random(soft=soft)
    family = torch.tensor([0, 1, 1, 0])
    time = torch.tensor([0.0, 0.25, 0.7, 1.0])
    with torch.inference_mode():
        original = full_edge_scores(model(adjacency, family, time), 7)
        for _ in range(5):
            permutation = torch.randperm(7)
            relabelled_adj = adjacency[:, permutation][:, :, permutation]
            relabelled = full_edge_scores(model(relabelled_adj, family, time), 7)
            expected = original[:, permutation][:, :, permutation]
            torch.testing.assert_close(relabelled, expected, atol=2e-06, rtol=2e-06)

@pytest.mark.parametrize('layers,common', [(3, True), (0, True), (3, False)])
def test_gradients_flow_to_every_parameter(layers, common):
    torch.manual_seed(19)
    model = EdgeDenoiser(n=7, hidden=12, layers=layers, use_common_neighbors=common)
    clean = symmetric_random()
    noisy = 0.7 * clean + 0.3 * symmetric_random()
    family = torch.tensor([0, 1, 1, 0])
    logits = model(noisy, family, torch.tensor([0.05, 0.25, 0.6, 0.9]))
    assert logits.shape == (4, 21)
    left, right = model.edge_index
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, clean[:, left, right])
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name

def test_checkpoint_config_roundtrip_and_singleton_batch():
    model = EdgeDenoiser(n=6, hidden=8, families=3, layers=2, use_common_neighbors=False)
    reconstructed = EdgeDenoiser(**model.config())
    reconstructed.load_state_dict(model.state_dict())
    inputs = symmetric_random(batch=1, n=6)
    family, time = (torch.tensor([2]), torch.tensor([1.0]))
    logits = model(inputs, family, time)
    assert logits.shape == (1, 15)
    torch.testing.assert_close(logits, reconstructed(inputs, family, time))

def test_condition_changes_predictions():
    torch.manual_seed(151)
    model = EdgeDenoiser(n=7, hidden=12)
    adjacency = symmetric_random(batch=1)
    zero = torch.tensor([0])
    first = model(adjacency, zero, torch.tensor([0.0]))
    other_family = model(adjacency, torch.tensor([1]), torch.tensor([0.0]))
    other_time = model(adjacency, zero, torch.tensor([1.0]))
    assert not torch.allclose(first, other_family)
    assert not torch.allclose(first, other_time)

@pytest.mark.parametrize('kwargs', [{'n': 1}, {'hidden': 0}, {'families': 0}, {'layers': -1}, {'layers': 1.5}, {'n': True}])
def test_invalid_model_config(kwargs):
    with pytest.raises(ValueError):
        EdgeDenoiser(**kwargs)

def test_invalid_input_shape():
    model = EdgeDenoiser(n=7, hidden=8)
    with pytest.raises(ValueError, match='noisy_adj'):
        model(torch.zeros(2, 6, 6), torch.zeros(2, dtype=torch.long), torch.zeros(2))
    with pytest.raises(ValueError, match='family and time'):
        model(torch.zeros(2, 7, 7), torch.zeros(2, dtype=torch.long), torch.zeros(2, 1))
from frugal_graphs.models import alpha_bar, corrupt_edges, reverse_probability

def transition(retained, prior=0.21):
    stationary = torch.tensor([1 - prior, prior], dtype=torch.float64)
    return retained * torch.eye(2, dtype=torch.float64) + (1 - retained) * stationary.repeat(2, 1)

def test_schedule_exact_endpoints_and_monotonicity():
    time = torch.arange(33, dtype=torch.float64)
    schedule = alpha_bar(time)
    assert schedule.dtype == torch.float64
    assert schedule[0] == 1
    assert schedule[-1] == 0
    assert torch.all(schedule[:-1] > schedule[1:])
    assert alpha_bar(torch.tensor([0, 32])).dtype == torch.float32

@pytest.mark.parametrize('alpha_s,alpha_t', [(1.0, 0.9), (0.9, 0.7), (0.9, 0.1), (0.2, 0.0)])
@pytest.mark.parametrize('clean_state', [0, 1])
@pytest.mark.parametrize('observed_state', [0, 1])
def test_posterior_matches_bayes_enumeration(alpha_s, alpha_t, clean_state, observed_state):
    prior = 0.21
    previous = transition(alpha_s, prior)[clean_state]
    step = transition(alpha_t / alpha_s, prior)
    numerator = previous * step[:, observed_state]
    enumerated = numerator / numerator.sum()
    actual = reverse_probability(torch.tensor([[float(clean_state)]], dtype=torch.float64), torch.tensor([[float(observed_state)]], dtype=torch.float64), alpha_s, alpha_t, prior)
    assert actual.item() == pytest.approx(enumerated[1].item(), abs=1e-14)

def test_each_clean_component_is_normalized_before_mixture():
    retained_s, retained_t, prior, prediction = (0.85, 0.3, 0.21, 0.6)
    step = transition(retained_t / retained_s, prior)
    marginal = transition(retained_s, prior)
    for observed in (0, 1):
        posterior_components = []
        for clean in (0, 1):
            joint = marginal[clean] * step[:, observed]
            posterior_components.append((joint / joint.sum())[1])
        expected = (1 - prediction) * posterior_components[0] + prediction * posterior_components[1]
        actual = reverse_probability(torch.tensor([[prediction]], dtype=torch.float64), torch.tensor([[observed]]), retained_s, retained_t, prior)
        assert actual.item() == pytest.approx(expected.item(), abs=1e-14)

def test_final_step_returns_clean_prediction_exactly():
    prediction = torch.tensor([[0.0, 0.13, 0.5, 0.999, 1.0]], dtype=torch.float64)
    observations = torch.tensor([[1, 0, 1, 0, 1]])
    for retained_t in (0.99, 0.3, 0.0):
        torch.testing.assert_close(reverse_probability(prediction, observations, 1.0, retained_t, 0.21), prediction, rtol=0, atol=0)

def test_skipped_transition_composition():
    retained_s, retained_t = (0.8, 0.15)
    composed = transition(retained_s) @ transition(retained_t / retained_s)
    torch.testing.assert_close(composed, transition(retained_t), atol=1e-15, rtol=0)

def test_true_forward_prior_is_recovered_by_reverse_mixture():
    clean_prior = 0.63
    stationary_prior = 0.21
    retained_s, retained_t = (0.73, 0.24)
    initial = torch.tensor([1 - clean_prior, clean_prior], dtype=torch.float64)
    forward_t = transition(retained_t, stationary_prior)
    marginal_t = initial @ forward_t
    recovered_s_one = 0.0
    for observed in (0, 1):
        clean_posterior = initial * forward_t[:, observed] / marginal_t[observed]
        reverse = reverse_probability(clean_posterior[1].reshape(1, 1), torch.tensor([[observed]]), retained_s, retained_t, stationary_prior)
        recovered_s_one += marginal_t[observed] * reverse.item()
    expected_s = initial @ transition(retained_s, stationary_prior)
    assert recovered_s_one.item() == pytest.approx(expected_s[1].item(), abs=1e-14)

def test_corruption_clean_and_terminal_endpoints():
    clean = torch.zeros((3, 200), dtype=torch.float64)
    clean[1] = 1
    clean[2, ::2] = 1
    unchanged = corrupt_edges(clean, torch.zeros(3, dtype=torch.int64), 32, 0.21, torch.Generator().manual_seed(5))
    torch.testing.assert_close(unchanged, clean, rtol=0, atol=0)
    at_terminal = corrupt_edges(clean, torch.full((3,), 32), 32, 0.21, torch.Generator().manual_seed(8))
    independently_sampled = torch.bernoulli(torch.full_like(clean, 0.21), generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(at_terminal, independently_sampled, rtol=0, atol=0)
    opposite = corrupt_edges(1 - clean, torch.full((3,), 32), 32, 0.21, torch.Generator().manual_seed(8))
    torch.testing.assert_close(at_terminal, opposite, rtol=0, atol=0)

def test_all_reverse_probabilities_are_finite_and_bounded():
    generator = torch.Generator().manual_seed(74)
    prediction = torch.rand((8, 190), generator=generator)
    observations = torch.randint(2, (8, 190), generator=generator)
    schedule = alpha_bar(torch.arange(33, dtype=torch.float64))
    for time in range(1, 33):
        for previous_time in (0, time // 2, time - 1):
            probabilities = reverse_probability(prediction, observations, float(schedule[previous_time]), float(schedule[time]), 40 / 190)
            assert torch.isfinite(probabilities).all()
            assert ((probabilities >= 0) & (probabilities <= 1)).all()

@pytest.mark.parametrize('prior', [0.0, 1.0, -0.1, float('nan')])
def test_invalid_stationary_prior_is_rejected(prior):
    with pytest.raises(ValueError):
        reverse_probability(torch.ones((1, 1)), torch.ones((1, 1)), 0.5, 0.1, prior)

def test_invalid_schedule_and_times_are_rejected():
    with pytest.raises(ValueError):
        alpha_bar(torch.tensor([33]), 32)
    with pytest.raises(ValueError):
        alpha_bar(torch.tensor([1]), 0)
    with pytest.raises(ValueError):
        corrupt_edges(torch.ones((1, 3)), torch.tensor([1.5]), 32, 0.21)
    with pytest.raises(ValueError):
        reverse_probability(torch.ones((1, 1)), torch.ones((1, 1)), 0.2, 0.5, 0.21)
import json
from pathlib import Path
import subprocess
import sys
from frugal_graphs.diffusion import generate, load_denoiser

@pytest.fixture
def small_model_and_config():
    torch.set_num_threads(1)
    torch.manual_seed(16)
    model = EdgeDenoiser(n=6, hidden=8, layers=1).eval()
    config = {'n': 6, 'm': 7, 'diffusion_steps': 4}
    return (model, config)

@pytest.mark.parametrize('steps', [1, 4])
@pytest.mark.parametrize('family', [0, 1])
def test_connected_sampler_is_valid_and_reproducible(small_model_and_config, steps, family):
    model, config = small_model_and_config
    actual = generate(model, 9, family, config, steps, 0.5, 42)
    repeated = generate(model, 9, family, config, steps, 0.5, 42)
    assert actual.shape == (9, 6, 6)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert set(np.unique(actual)) <= {0.0, 1.0}
    np.testing.assert_array_equal(actual, actual.transpose(0, 2, 1))
    np.testing.assert_array_equal(actual, repeated)
    assert (np.diagonal(actual, axis1=1, axis2=2) == 0).all()
    np.testing.assert_array_equal(actual.sum(axis=(1, 2)), np.full(9, 14))
    assert all((nx.is_connected(nx.from_numpy_array(adjacency)) for adjacency in actual))

@pytest.mark.parametrize('steps', [1, 4])
def test_raw_sampler_simple_undirected_without_budget_assumption(small_model_and_config, steps):
    model, config = small_model_and_config
    actual = generate(model, 5, 0, config, steps, 0.5, 124, mode='raw')
    repeated = generate(model, 5, 0, config, steps, 0.5, 124, mode='raw')
    assert actual.shape == (5, 6, 6)
    assert np.isfinite(actual).all()
    assert set(np.unique(actual)) <= {0.0, 1.0}
    assert (np.diagonal(actual, axis1=1, axis2=2) == 0).all()
    np.testing.assert_array_equal(actual, actual.transpose(0, 2, 1))
    np.testing.assert_array_equal(actual, repeated)

def test_all_mode_matches_separate_decoders(small_model_and_config):
    model, config = small_model_and_config
    together = generate(model, 4, 1, config, 4, 0.25, 196, mode='all')
    assert set(together) == {'raw', 'topk', 'connected'}
    for mode in together:
        independent = generate(model, 4, 1, config, 4, 0.25, 196, mode=mode)
        np.testing.assert_array_equal(together[mode], independent)

def test_checkpoint_roundtrip_preserves_logits_and_samples(tmp_path, small_model_and_config):
    model, config = small_model_and_config
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model_config': model.config(), 'state_dict': model.state_dict()}, checkpoint)
    loaded = load_denoiser(checkpoint)
    assert not loaded.training
    assert loaded.config() == model.config()
    adjacency = torch.randint(0, 2, (3, 6, 6)).float().triu(1)
    adjacency += adjacency.transpose(1, 2).clone()
    family = torch.tensor([0, 1, 0])
    times = torch.tensor([0.0, 0.5, 1.0])
    with torch.inference_mode():
        torch.testing.assert_close(model(adjacency, family, times), loaded(adjacency, family, times), rtol=0, atol=0)
    for steps in (1, 4):
        np.testing.assert_array_equal(generate(model, 5, 1, config, steps, 0.5, 201), generate(loaded, 5, 1, config, steps, 0.5, 201))

def test_sampling_cli_saves_actual_samples_metadata_and_plot(tmp_path, small_model_and_config):
    model, config = small_model_and_config
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model_config': model.config(), 'state_dict': model.state_dict()}, checkpoint)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    output = tmp_path / 'samples' / 'example.npz'
    plot = output.with_suffix('.png')
    result = subprocess.run([sys.executable, '-m', 'frugal_graphs.project', 'sample', '--checkpoint', str(checkpoint), '--config', str(config_path), '--count', '3', '--steps', '4', '--family', 'geometric', '--temperature', '.5', '--seed', '42', '--output', str(output), '--plot', str(plot)], check=True, capture_output=True, text=True)
    status = json.loads(result.stdout)
    assert status['connected_count'] == status['exact_edge_budget_count'] == 3
    with np.load(output) as samples:
        assert samples['adj'].shape == (3, 6, 6)
        np.testing.assert_array_equal(samples['family'], [1, 1, 1])
        np.testing.assert_array_equal(samples['adj'], generate(model, 3, 1, config, 4, 0.5, 42))
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['edge_counts'] == [7, 7, 7]
    assert metadata['connected_count'] == metadata['symmetric_count'] == metadata['zero_diagonal_count'] == 3
    assert plot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
from frugal_graphs.cvae import sample_graphs

class ZeroDecoder:
    latent = 7

    def decode(self, z, labels):
        return torch.zeros(len(z), 28)

def test_baseline_and_neural_use_common_edge_noise():
    config = {'n': 8, 'm': 12}
    baseline = sample_graphs(None, None, 0, 10, config, 1.0, 19)
    neural = sample_graphs(ZeroDecoder(), None, 0, 10, config, 1.0, 19)
    np.testing.assert_array_equal(baseline, neural)
    again = sample_graphs(None, None, 0, 10, config, 1.0, 19)
    np.testing.assert_array_equal(baseline, again)
    assert np.all(baseline.sum((1, 2)) == 24)
from frugal_graphs.models import GraphCVAE, parameter_counts

def test_encoder_is_permutation_invariant_and_gradients_flow():
    torch.manual_seed(3)
    model = GraphCVAE(n=8, hidden=12, latent=4)
    a = torch.randint(0, 2, (3, 8, 8)).float().triu(1)
    a = a + a.transpose(1, 2)
    f = torch.tensor([0, 1, 0])
    p = torch.randperm(8)
    mean, var = model.encode(a, f)
    mean_p, var_p = model.encode(a[:, p][:, :, p], f)
    torch.testing.assert_close(mean, mean_p, atol=2e-06, rtol=2e-06)
    torch.testing.assert_close(var, var_p, atol=2e-06, rtol=2e-06)
    loss, _ = model.loss(a, f)
    loss.backward()
    assert torch.isfinite(loss)
    assert all((p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    assert model.message_layers[0].mlp[0].weight.grad.norm() > 0

def test_generation_uses_prior_without_input_graph():
    model = GraphCVAE(n=8, hidden=12, latent=4)
    logits = model.decode(torch.randn(5, 4), torch.zeros(5, dtype=torch.long))
    assert logits.shape == (5, 28)
    counts = parameter_counts(model)
    assert 0 < counts['parameters_generation'] < counts['parameters_total']
