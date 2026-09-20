import math

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from frugal_graphs.sparse_model import SparseDenoiser, _example, _training_candidates, load_sparse, quantize_sparse, sample_sparse, score_candidates, train_sparse


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example_graph(n=8):
    edges = np.asarray([(i, i + 1) for i in range(n - 1)] + [(0, 3), (2, 5)], dtype=np.int64)
    candidates = np.asarray([(i, j) for i in range(n) for j in range(i + 1, min(n, i + 4))], dtype=np.int64)
    return edges, candidates


@pytest.mark.parametrize('with_positions', [False, True])
@pytest.mark.parametrize('layers', [0, 2])
def test_permutation_equivariance(with_positions, layers):
    torch.manual_seed(4)
    model = SparseDenoiser(12, layers).eval()
    n = 8
    edges, candidates = example_graph(n)
    rng = np.random.default_rng(3)
    permutation = rng.permutation(n)
    positions = rng.random((n, 2)) if with_positions else None
    permuted_positions = None
    if positions is not None:
        permuted_positions = np.empty_like(positions)
        permuted_positions[permutation] = positions
    actual = score_candidates(model, n, edges, candidates, 0.3, 1, positions)
    permuted = score_candidates(model, n, permutation[edges], permutation[candidates][:, ::-1], 0.3, 1, permuted_positions)
    np.testing.assert_allclose(actual, permuted, atol=3e-6, rtol=3e-6)


def test_gradients_reach_message_passing():
    torch.manual_seed(3)
    model = SparseDenoiser(12, 2)
    edges, candidates = example_graph()
    scores = model(8, torch.from_numpy(edges.T), torch.from_numpy(candidates.T), 0.7, 0)
    target = torch.linspace(0, 1, len(candidates))
    torch.nn.functional.binary_cross_entropy_with_logits(scores, target).backward()
    for layer in model.message_layers:
        assert layer.neighbor_map.weight.grad is not None
        assert torch.isfinite(layer.neighbor_map.weight.grad).all()
        assert layer.neighbor_map.weight.grad.abs().sum() > 0


class RejectSquareGraphTensors(TorchDispatchMode):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        values = result if isinstance(result, (tuple, list)) else (result,)
        for value in values:
            if isinstance(value, torch.Tensor):
                assert value.shape.count(self.n) < 2, f'dense graph tensor from {func}: {value.shape}'
        return result


def test_forward_never_allocates_dense_adjacency():
    n = 101
    model = SparseDenoiser(13, 2)
    edges = np.column_stack((np.arange(n - 1), np.arange(1, n)))
    candidates = _training_candidates(n, edges, 3, np.random.default_rng(5))
    assert len(candidates) <= len(edges) + 3 * n
    with RejectSquareGraphTensors(n):
        logits = model(n, torch.from_numpy(edges.T), torch.from_numpy(candidates.T), 0.5, 0)
        logits.square().mean().backward()
    assert logits.shape == (len(candidates),)


@pytest.mark.parametrize('n', [8, 37, 128])
def test_variable_graph_size_and_empty_edges(n):
    model = SparseDenoiser(8, 1)
    logits = score_candidates(model, n, np.empty((0, 2), dtype=np.int64), np.asarray([[0, n - 1]]), 1, 0)
    assert logits.shape == (1,)
    assert np.isfinite(logits).all()
    empty = score_candidates(model, n, [], np.empty((0, 2), dtype=np.int64), 0, 1)
    assert empty.shape == (0,)


@pytest.mark.parametrize('time', [-0.1, 1.1, math.nan])
def test_invalid_time(time):
    with pytest.raises(ValueError, match='time'):
        SparseDenoiser()(3, torch.empty((2, 0), dtype=torch.long), torch.tensor([[0], [1]]), time, 0)


def test_candidate_order_and_orientation():
    model = SparseDenoiser(8, 1)
    edges, candidates = example_graph()
    a = score_candidates(model, 8, edges, candidates, 0.4, 1)
    b = score_candidates(model, 8, edges[::-1, ::-1], candidates[::-1, ::-1], 0.4, 1)
    np.testing.assert_allclose(a, b[::-1], atol=1e-6)


def test_candidate_training_targets_and_frozen_noise():
    n = 20
    clean = np.column_stack((np.arange(n - 1), np.arange(1, n)))
    record = {'n': n, 'edges': clean, 'family': 0}
    a = _example(record, 2, 16, np.random.default_rng(4))
    b = _example(record, 2, 16, np.random.default_rng(4))
    assert a[6].sum() == len(clean)
    assert len(a[6]) <= len(clean) + 2 * n
    for index in (1, 2, 6):
        assert torch.equal(a[index], b[index])
    assert a[3] == b[3]
    complete = _training_candidates(n, clean, 1, np.random.default_rng(4), complete=True)
    assert len(complete) == n * (n - 1) // 2


def test_training_checkpoint_and_reproduction(tmp_path):
    edges, _ = example_graph()
    dataset = {'train': [{'n': 8, 'edges': edges, 'family': i % 2} for i in range(4)], 'val': [{'n': 8, 'edges': edges, 'family': 0}]}
    config = {'epochs': 3, 'hidden': 8, 'layers': 1, 'batch_size': 2, 'k': 1, 'diffusion_steps': 4}
    a, summary = train_sparse(config, dataset, 12, tmp_path / 'first')
    b, summary_b = train_sparse(config, dataset, 12, tmp_path / 'second')
    loaded = load_sparse(tmp_path / 'first/checkpoint.pt')
    assert summary['history'] == summary_b['history']
    assert summary['val_bce'] == min(row['val_bce'] for row in summary['history'])
    assert a.config() == {'hidden': 8, 'layers': 1}
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
        assert torch.equal(value, loaded.state_dict()[key])


@pytest.mark.parametrize('mode', ['admissible', 'projected'])
@pytest.mark.parametrize('candidate_mode', ['sparse', 'dense'])
def test_sampling_constraints_at_each_step(mode, candidate_mode):
    from frugal_graphs.constrained import validate_graph
    model = SparseDenoiser(8, 1)
    result = sample_sparse(model, 12, 18, 3, 2, 42, 0, max_degree=4, mode=mode, candidates=candidate_mode, trace=True)
    assert len(result['trajectory']) == 4
    for edges in result['trajectory']:
        assert validate_graph(12, edges, 18, max_degree=4)['valid']
    if candidate_mode == 'sparse':
        assert result['candidate_max'] <= min(66, 18 + 2 * 12)
    else:
        assert result['candidate_max'] == 66
    assert result['timing']['total_seconds'] > 0


def test_sampling_reproduction_and_continuation():
    model = SparseDenoiser(8, 1)
    a = sample_sparse(model, 12, 18, 2, 2, 42, 0)
    b = sample_sparse(model, 12, 18, 2, 2, 42, 0)
    np.testing.assert_array_equal(a['edges'], b['edges'])
    resumed = sample_sparse(model, 12, 18, 1, 2, 42, 0, initial_edges=a['edges'], refresh=False)
    np.testing.assert_array_equal(resumed['initial_edges'], a['edges'])


def test_dynamic_quantization_reports_supported_backend():
    model = SparseDenoiser(8, 1)
    quantized, metadata = quantize_sparse(model)
    assert metadata['status'] in ('executed', 'not_executed')
    if quantized is not None:
        scores = score_candidates(quantized, 3, np.asarray([[0, 1], [1, 2]]), np.asarray([[0, 2]]), 0.5, 0)
        assert np.isfinite(scores).all()
        assert 'engine' in metadata
    else:
        assert metadata['reason']
