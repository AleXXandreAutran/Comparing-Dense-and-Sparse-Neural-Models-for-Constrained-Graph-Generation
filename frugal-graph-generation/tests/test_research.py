from itertools import combinations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from frugal_graphs.research import as_graph, dataset_digest, load_records, make_dataset, save_records, to_dense_dataset, verify_research, write_json
from frugal_graphs.research_metrics import evaluate_samples
from frugal_graphs.sparse_model import SparseDenoiser, sample_sparse, score_candidates, train_sparse
from frugal_graphs import research


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def dataset_config():
    return {'n': 16, 'budget': 32, 'dataset_seed': 832, 'train_per_family': 3,
            'val_per_family': 2, 'test_per_family': 2, 'reference_per_family': 2}


@pytest.fixture(scope='module')
def dataset(dataset_config):
    return make_dataset(dataset_config)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_dataset_reproducibility_and_seed_changes(dataset_config, dataset):
    repeated = make_dataset(dataset_config)
    changed = make_dataset({**dataset_config, 'dataset_seed': dataset_config['dataset_seed'] + 1})
    for split in dataset:
        assert dataset_digest(dataset[split]) == dataset_digest(repeated[split])
        assert dataset_digest(dataset[split]) != dataset_digest(changed[split])


def test_dataset_constraints_and_exact_isomorphism_separation(dataset_config, dataset):
    graphs = []
    for split, records in dataset.items():
        assert len(records) == 2 * dataset_config[f'{split}_per_family']
        for record in records:
            graph = as_graph(record)
            assert len(graph) == dataset_config['n']
            assert graph.number_of_edges() == dataset_config['budget']
            assert nx.is_connected(graph)
            if record['family'] == 1:
                assert nx.check_planarity(graph)[0]
            assert set(record) == {'n', 'edges', 'family'}
            graphs.append(graph)
    for left, right in combinations(graphs, 2):
        assert not nx.is_isomorphic(left, right)


def test_record_archive_roundtrip_with_positions_and_trace(tmp_path, dataset):
    original = dict(dataset['train'][0])
    original['positions'] = np.random.default_rng(8).random((original['n'], 2))
    original['sampling'] = {'trajectory': [original['edges']], 'elapsed': np.float64(0.25)}
    path = tmp_path / 'samples.npz'
    save_records(path, [original])
    restored = load_records(path)[0]
    np.testing.assert_array_equal(restored['edges'], original['edges'])
    np.testing.assert_array_equal(restored['positions'], original['positions'])
    np.testing.assert_array_equal(restored['sampling']['trajectory'][0], original['edges'])
    assert restored['sampling']['elapsed'] == 0.25
    assert dataset_digest([restored]) == dataset_digest([original])
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != 'O' for key in archive.files)


def test_dense_conversion_preserves_graphs(dataset):
    dense = to_dense_dataset(dataset)
    for split, records in dataset.items():
        matrices = dense[split]['adj']
        np.testing.assert_array_equal(matrices, matrices.transpose(0, 2, 1))
        np.testing.assert_array_equal(np.diagonal(matrices, axis1=1, axis2=2), 0)
        for index, record in enumerate(records):
            np.testing.assert_array_equal(np.column_stack(np.where(np.triu(matrices[index], 1))), record['edges'])
            assert dense[split]['family'][index] == record['family']


def test_train_sample_verify_and_reject_changed_metrics(tmp_path, dataset, dataset_config):
    model, summary = train_sparse({'epochs': 2, 'hidden': 8, 'layers': 1, 'k': 2, 'diffusion_steps': 4,
                                   'batch_size': 2}, dataset, 5, tmp_path / 'runs/sparse_5')
    assert summary['best_epoch'] in (1, 2)
    records = []
    for seed in (13, 17):
        sample = sample_sparse(model, 16, 32, 2, 2, seed, 0, trace=True)
        edges = sample.pop('edges')
        records.append({'n': 16, 'edges': edges, 'family': 0, 'sampling': sample})
    for split, split_records in dataset.items():
        save_records(tmp_path / f'dataset-{split}.npz', split_records)
    write_json(tmp_path / 'config.json', dataset_config)
    write_json(tmp_path / 'dataset_manifest.json', {split: {'graphs': len(values), 'sha256': dataset_digest(values)}
                                                   for split, values in dataset.items()})
    metrics = evaluate_samples(records, [r for r in dataset['test'] if r['family'] == 0],
                               [r for r in dataset['train'] if r['family'] == 0], 32)
    row = {'method': 'sparse_s2', 'seed': 5, 'family': 'sbm', **metrics}
    write_json(tmp_path / 'metrics.json', [row])
    save_records(tmp_path / 'samples/sparse_s2_sbm_5.npz', records)
    verified = verify_research(tmp_path, partial=True)
    assert verified['scope'] == 'partial'
    assert verified['quality_rows_recomputed'] == 1
    assert verified['valid_intermediate_graphs_checked'] == 6
    row['degree_mmd2'] += 0.1
    write_json(tmp_path / 'metrics.json', [row])
    with pytest.raises(AssertionError):
        verify_research(tmp_path, partial=True)


def test_candidate_ablation_preserves_scores_with_shared_weights():
    from frugal_graphs.constrained import candidate_edges, feasible_seed
    n = 31
    model = SparseDenoiser(12, 2).eval()
    current = feasible_seed(n, 2 * n, seed=11)
    sparse = candidate_edges(n, current, 2, np.random.default_rng(12))
    dense = np.column_stack(np.triu_indices(n, 1))
    sparse_scores = score_candidates(model, n, current, sparse, 0.5, 0)
    dense_scores = score_candidates(model, n, current, dense, 0.5, 0)
    lookup = {tuple(pair): score for pair, score in zip(dense, dense_scores)}
    np.testing.assert_allclose(sparse_scores, [lookup[tuple(pair)] for pair in sparse], atol=1e-6, rtol=1e-6)
    assert len(sparse) <= 4 * n < len(dense)


def test_isolated_profile_worker_cli(tmp_path):
    model = SparseDenoiser(8, 1).eval()
    checkpoint = tmp_path / 'runs/sparse_11/checkpoint.pt'
    checkpoint.parent.mkdir(parents=True)
    torch.save({'format': 'sparse-edit-v1', 'model_config': model.config(), 'state_dict': model.state_dict()}, checkpoint)
    rows = []
    for method in ('sparse_score', 'dense_candidate_score'):
        spec = {'method': method, 'n': 24, 'trained_n': 16, 'seed': 11, 'run': str(tmp_path),
                'k': 2, 'steps': 1, 'diffusion_steps': 4, 'repetitions': 2, 'warmup': 1}
        spec_path, result_path = tmp_path / f'{method}-spec.json', tmp_path / f'{method}.json'
        write_json(spec_path, spec)
        result = subprocess.run([sys.executable, '-m', 'frugal_graphs.research', 'profile-worker',
                                 '--spec', str(spec_path), '--output', str(result_path)],
                                capture_output=True, text=True, timeout=60, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        row = json.loads(result_path.read_text())
        assert row['status'] == 'executed'
        assert row['latency_ms_median'] > 0
        assert row['rss_sampled_peak_bytes'] > 0
        assert row['inference_only_size_extrapolation'] is True
        assert row['checkpoint_serialized_bytes'] > 0
        rows.append(row)
    assert rows[0]['candidate_count'] <= 4 * 24
    assert rows[1]['candidate_count'] == 24 * 23 // 2
    assert rows[0]['checkpoint_serialized_bytes'] == rows[1]['checkpoint_serialized_bytes']


@pytest.fixture(scope='module')
def completed_experiment(tmp_path_factory):
    output = tmp_path_factory.mktemp('complete-experiment')
    config = json.loads((ROOT / 'configs/research.json').read_text())['smoke']
    research.train_and_evaluate(config, output)
    summary = verify_research(output)
    assert summary['scope'] == 'complete'
    assert summary['checkpoints_checked'] == 3
    return output, config


@pytest.fixture
def experiment_copy(tmp_path, completed_experiment):
    original, config = completed_experiment
    output = tmp_path / 'experiment'
    shutil.copytree(original, output)
    return output, config


def file_hashes(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob('*') if path.is_file()}


@pytest.mark.parametrize('split', ['train', 'val', 'test', 'reference'])
@pytest.mark.parametrize('change', ['label', 'edge', 'count', 'missing'])
def test_resume_rejects_changed_datasets_without_writes(experiment_copy, split, change):
    output, config = experiment_copy
    path = output / f'dataset-{split}.npz'
    records = load_records(path)
    if change == 'missing':
        path.unlink()
    else:
        if change == 'label':
            records[0]['family'] = 1 - records[0]['family']
        elif change == 'edge':
            records[0]['edges'][0, 0] = records[0]['edges'][0, 1]
        else:
            records.pop()
        save_records(path, records)
    before = file_hashes(output)
    with pytest.raises(ValueError, match=split):
        research.train_and_evaluate(config, output, resume=True)
    assert file_hashes(output) == before


@pytest.mark.parametrize('change', ['missing', 'count'])
def test_resume_rejects_missing_or_changed_manifest(experiment_copy, change):
    output, config = experiment_copy
    path = output / 'dataset_manifest.json'
    if change == 'missing':
        path.unlink()
    else:
        manifest = json.loads(path.read_text())
        manifest['train']['graphs'] -= 1
        write_json(path, manifest)
    before = file_hashes(output)
    with pytest.raises(ValueError, match='manifest'):
        research.train_and_evaluate(config, output, resume=True)
    assert file_hashes(output) == before


def test_valid_resume_preserves_data_and_checkpoints(experiment_copy):
    output, config = experiment_copy
    before = file_hashes(output)
    rows = json.loads((output / 'metrics.json').read_text())
    research.train_and_evaluate(config, output, resume=True)
    after = file_hashes(output)
    protected = [name for name in before if name.startswith('dataset') or name.endswith('checkpoint.pt')]
    assert all(before[name] == after[name] for name in protected)
    assert not (output / 'verification.json').exists()
    actual = json.loads((output / 'metrics.json').read_text())
    for left, right in zip(rows, actual, strict=True):
        for key in ('method', 'seed', 'family', 'degree_mmd2', 'clustering_mmd2', 'spectral_mmd2'):
            assert left[key] == right[key]
    assert verify_research(output)['scope'] == 'complete'


def test_interrupted_resume_invalidates_completion(experiment_copy, monkeypatch):
    from frugal_graphs import diffusion
    output, config = experiment_copy
    def interrupted(*args, **kwargs):
        raise RuntimeError('interrupted')
    monkeypatch.setattr(diffusion, 'load_denoiser', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        research.train_and_evaluate(config, output, resume=True)
    assert not (output / 'quality_complete.json').exists()
    assert not (output / 'verification.json').exists()
    with pytest.raises(ValueError, match='completion'):
        verify_research(output)


@pytest.mark.parametrize('change', [
    'missing_row', 'duplicate_row', 'wrong_seed', 'sample_count', 'missing_sample',
    'sample_family', 'sample_nodes', 'completion_rows', 'missing_training',
    'duplicate_training', 'missing_checkpoint', 'missing_quantization', 'invalid_quantization',
])
def test_full_verification_rejects_incomplete_runs(experiment_copy, change):
    output, _ = experiment_copy
    rows = json.loads((output / 'metrics.json').read_text())
    if change in ('missing_row', 'duplicate_row', 'wrong_seed', 'sample_count'):
        if change == 'missing_row':
            rows.pop()
        elif change == 'duplicate_row':
            rows[-1] = rows[0]
        elif change == 'wrong_seed':
            rows[0]['seed'] += 100
        else:
            rows[0]['count'] -= 1
        write_json(output / 'metrics.json', rows)
    elif change in ('missing_sample', 'sample_family', 'sample_nodes'):
        row = rows[0]
        path = output / 'samples' / f'{row["method"]}_{row["family"]}_{row["seed"]}.npz'
        samples = load_records(path)
        if change == 'missing_sample':
            samples.pop()
        elif change == 'sample_family':
            samples[0]['family'] = 1 - samples[0]['family']
        else:
            samples[0]['n'] += 1
        save_records(path, samples)
    elif change == 'completion_rows':
        path = output / 'quality_complete.json'
        completion = json.loads(path.read_text())
        completion['rows'] += 1
        write_json(path, completion)
    elif change in ('missing_training', 'duplicate_training'):
        path = output / 'training.json'
        training = json.loads(path.read_text())
        if change == 'missing_training':
            training.pop()
        else:
            training[-1] = training[0]
        write_json(path, training)
    elif change == 'missing_checkpoint':
        (output / 'runs/dense_11/checkpoint.pt').unlink()
    elif change == 'missing_quantization':
        (output / 'runs/sparse_11/quantization.json').unlink()
    else:
        write_json(output / 'runs/sparse_11/quantization.json', {'status': 'unknown'})
    with pytest.raises(ValueError):
        verify_research(output)
    assert not (output / 'verification.json').exists()


def test_full_verification_supports_unavailable_quantization(experiment_copy):
    output, _ = experiment_copy
    rows = json.loads((output / 'metrics.json').read_text())
    rows = [row for row in rows if row['method'] != 'sparse_int8']
    write_json(output / 'metrics.json', rows)
    write_json(output / 'runs/sparse_11/quantization.json', {'status': 'not_executed', 'reason': 'unavailable backend'})
    completion = json.loads((output / 'quality_complete.json').read_text())
    completion['rows'] = len(rows)
    write_json(output / 'quality_complete.json', completion)
    assert verify_research(output)['quality_rows_recomputed'] == len(rows)


def test_partial_verification_cannot_be_packed(experiment_copy, monkeypatch, tmp_path):
    output, _ = experiment_copy
    rows = json.loads((output / 'metrics.json').read_text())[:-1]
    write_json(output / 'metrics.json', rows)
    result = verify_research(output, partial=True)
    assert result['scope'] == 'partial'
    assert (output / 'partial_verification.json').is_file()
    assert not (output / 'verification.json').exists()
    destination = tmp_path / 'bundle'
    monkeypatch.setattr(research, 'ROOT', destination)
    with pytest.raises(ValueError, match='Metric rows'):
        research.pack_results(output)
    assert not destination.exists()


@pytest.mark.parametrize('change', ['missing_row', 'changed_metric'])
def test_pack_rechecks_results_before_writing(experiment_copy, monkeypatch, tmp_path, change):
    output, _ = experiment_copy
    assert (output / 'verification.json').exists()
    rows = json.loads((output / 'metrics.json').read_text())
    if change == 'missing_row':
        rows.pop()
    else:
        rows[0]['degree_mmd2'] += 1
    write_json(output / 'metrics.json', rows)
    destination = tmp_path / 'bundle'
    monkeypatch.setattr(research, 'ROOT', destination)
    with pytest.raises((ValueError, AssertionError)):
        research.pack_results(output)
    assert not destination.exists()


def test_verified_pack_roundtrip(experiment_copy, monkeypatch, tmp_path):
    output, _ = experiment_copy
    (output / 'verification.json').unlink()
    monkeypatch.setattr(research, 'ROOT', tmp_path / 'bundle')
    research.pack_results(output)
    restored = tmp_path / 'restored'
    research.restore_results(restored)
    before = verify_research(output)
    assert verify_research(restored) == before
    assert json.loads((restored / 'metrics.json').read_text()) == json.loads((output / 'metrics.json').read_text())
    for original in (output / 'runs').glob('*/checkpoint.pt'):
        relative = original.relative_to(output)
        left = torch.load(original, map_location='cpu', weights_only=True)['state_dict']
        right = torch.load(restored / relative, map_location='cpu', weights_only=True)['state_dict']
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
