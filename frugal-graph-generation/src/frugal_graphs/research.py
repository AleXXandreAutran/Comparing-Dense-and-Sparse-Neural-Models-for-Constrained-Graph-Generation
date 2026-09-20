from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
import torch

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('sbm', 'planar')


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + '\n')


def environment():
    return {
        'python': sys.version, 'platform': platform.platform(),
        'machine': platform.machine(), 'logical_cpus': os.cpu_count(),
        'device': 'cpu', 'torch_threads': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'numpy', 'scipy', 'networkx', 'matplotlib', 'psutil')},
    }


def source_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / 'src/frugal_graphs').glob('*.py')) if not path.name.startswith('test_')}


def as_graph(record):
    graph = nx.Graph()
    graph.add_nodes_from(range(record['n']))
    graph.add_edges_from(record['edges'])
    return graph


def draw_graph(n, m, family, rng):
    if not n - 1 <= m <= min(n * (n - 1) // 2, 3 * n - 6):
        raise ValueError('The shared planar benchmark requires n-1 <= m <= 3n-6.')
    if family == 0:
        blocks = np.arange(n) % 4
        left, right = np.triu_indices(n, 1)
        within = blocks[left] == blocks[right]
        ratio = 6.0
        outside = m / (within.sum() * ratio + (~within).sum())
        probabilities = np.where(within, min(0.95, ratio * outside), outside)
        for _ in range(100000):
            edges = np.column_stack((left, right))[rng.random(len(left)) < probabilities]
            if len(edges) != m:
                continue
            graph = nx.Graph()
            graph.add_nodes_from(range(n))
            graph.add_edges_from(edges)
            if nx.is_connected(graph):
                return graph
        raise RuntimeError('SBM rejection budget exhausted.')
    positions = rng.random((n, 2))
    triangulation = Delaunay(positions)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for triangle in triangulation.simplices:
        graph.add_edges_from(((int(triangle[i]), int(triangle[(i + 1) % 3])) for i in range(3)))
    if graph.number_of_edges() < m:
        return draw_graph(n, m, family, rng)
    for u, v in graph.edges:
        graph[u][v]['weight'] = float(rng.random())
    tree = nx.minimum_spanning_tree(graph)
    rest = sorted(set(tuple(sorted(edge)) for edge in graph.edges) - set(tuple(sorted(edge)) for edge in tree.edges))
    for index in rng.permutation(len(rest))[:m - (n - 1)]:
        tree.add_edge(*rest[index])
    return nx.Graph(tree)


def make_dataset(config):
    rng = np.random.default_rng(config['dataset_seed'])
    buckets = defaultdict(list)
    dataset = {}
    for split in ('train', 'val', 'test', 'reference'):
        records = []
        count = config[f'{split}_per_family']
        for family in range(2):
            accepted = 0
            for _ in range(max(10000, count * 100)):
                graph = draw_graph(config['n'], config['budget'], family, rng)
                digest = nx.weisfeiler_lehman_graph_hash(graph, iterations=3)
                if any(nx.is_isomorphic(graph, previous) for previous in buckets[digest]):
                    continue
                buckets[digest].append(graph.copy())
                permutation = rng.permutation(config['n'])
                edges = np.asarray(sorted(tuple(sorted((int(permutation[u]), int(permutation[v]))))
                                          for u, v in graph.edges), dtype=np.int64).reshape(-1, 2)
                records.append({'n': config['n'], 'edges': edges, 'family': family})
                accepted += 1
                if accepted == count:
                    break
            if accepted != count:
                raise RuntimeError('Could not create enough distinct graphs.')
        dataset[split] = records
    return dataset


def save_records(path, records):
    arrays = {}
    manifest = []
    for index, record in enumerate(records):
        key = f'edges_{index}'
        arrays[key] = np.asarray(record['edges'], dtype=np.int64)
        metadata = {name: value for name, value in record.items() if name not in ('edges', 'positions')}
        metadata['edge_key'] = key
        if 'positions' in record:
            arrays[f'positions_{index}'] = record['positions']
            metadata['position_key'] = f'positions_{index}'
        manifest.append(metadata)
    arrays['metadata'] = np.asarray(json.dumps(manifest, default=json_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_records(path):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['metadata']))
        records = []
        for row in manifest:
            row['edges'] = archive[row.pop('edge_key')]
            if 'position_key' in row:
                row['positions'] = archive[row.pop('position_key')]
            records.append(row)
    return records


def dataset_digest(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(np.asarray([record['n'], record['family']], dtype='<i8').tobytes())
        digest.update(np.asarray(record['edges'], dtype='<i8').tobytes())
    return digest.hexdigest()


def checked_dataset(output, config):
    manifest_path = output / 'dataset_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing dataset manifest. Use a fresh output directory.')
    manifest = json.loads(manifest_path.read_text())
    dataset = {}
    for split in ('train', 'val', 'test', 'reference'):
        path = output / f'dataset-{split}.npz'
        if not path.is_file():
            raise ValueError(f'Missing {split} dataset. Use a fresh output directory.')
        records = load_records(path)
        expected = manifest.get(split, {})
        count = config[f'{split}_per_family']
        if (len(records) != len(FAMILIES) * count or expected.get('graphs') != len(records)
                or dataset_digest(records) != expected.get('sha256')
                or any(record['n'] != config['n'] for record in records)
                or any(sum(record['family'] == family for record in records) != count
                       for family in range(len(FAMILIES)))):
            raise ValueError(f'The {split} dataset differs from its manifest or configuration.')
        dataset[split] = records
    return dataset


def to_dense_dataset(dataset):
    result = {}
    for split, records in dataset.items():
        adjacency = np.zeros((len(records), records[0]['n'], records[0]['n']), dtype=np.float32)
        for index, record in enumerate(records):
            edges = record['edges']
            adjacency[index, edges[:, 0], edges[:, 1]] = 1
            adjacency[index, edges[:, 1], edges[:, 0]] = 1
        result[split] = {'adj': adjacency, 'family': np.array([r['family'] for r in records], dtype=np.int64)}
    return result


def dense_config(config):
    return {**config, 'm': config['budget'], 'epochs': config['dense_epochs'],
            'models': {'dense': {'hidden': config['hidden'], 'layers': config['layers']}},
            'diffusion_steps': config['diffusion_steps']}


def settings(config):
    variants = [{'method': name, 'kind': 'baseline'} for name in ('random', 'fitted_sbm', 'degree_prior')]
    for steps in config['sampling_steps']:
        variants.extend([
            {'method': f'dense_s{steps}', 'kind': 'dense', 'steps': steps},
            {'method': f'sparse_s{steps}', 'kind': 'sparse', 'steps': steps},
        ])
    maximum = max(config['sampling_steps'])
    variants.extend([
        {'method': 'sparse_projected', 'kind': 'sparse', 'steps': maximum, 'mode': 'projected'},
        {'method': 'sparse_fixed_candidates', 'kind': 'sparse', 'steps': maximum, 'refresh': False},
        {'method': 'sparse_no_mp', 'kind': 'no_mp', 'steps': maximum},
        {'method': 'sparse_dense_candidates', 'kind': 'sparse', 'steps': maximum, 'candidates': 'dense'},
        {'method': 'sparse_degree6', 'kind': 'sparse', 'steps': maximum, 'max_degree': 6},
    ])
    return variants


def generate_one(spec, models, config, family, seed, temperature, training):
    from .sparse_model import sample_sparse
    from .research_metrics import generate_baseline

    n, budget = config['n'], config['budget']
    start = time.perf_counter_ns()
    if spec['kind'] == 'baseline':
        edges = generate_baseline(spec['method'], n, budget, seed, family, training=training)
        info = {'candidate_count': n * (n - 1) // 2}
    elif spec['kind'] == 'dense':
        from .diffusion import generate
        adjacency = generate(models['dense'], 1, family, dense_config(config), spec['steps'], temperature, seed)[0]
        edges = np.column_stack(np.where(np.triu(adjacency, 1) > 0))
        info = {'candidate_count': n * (n - 1) // 2}
    else:
        model = models[spec['kind'] if spec['kind'] in ('no_mp', 'quantized') else 'sparse']
        info = sample_sparse(model, n, budget, spec['steps'], config['k'], seed, family,
                             max_degree=spec.get('max_degree'), mode=spec.get('mode', 'admissible'),
                             refresh=spec.get('refresh', True), temperature=temperature,
                             candidates=spec.get('candidates', 'sparse'), trace=True)
        edges = info.pop('edges')
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return {'n': n, 'edges': edges, 'family': family, 'sampling_ms': elapsed, 'sampling': info}


def train_and_evaluate(config, output, resume=False):
    from .diffusion import train as train_dense, load_denoiser
    from .sparse_model import train_sparse, load_sparse, quantize_sparse
    from .research_metrics import evaluate_samples, fit_baseline, profile_operation

    torch.set_num_threads(config['threads'])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    existing = output / 'config.json'
    if existing.exists():
        if not resume or json.loads(existing.read_text()) != config:
            raise ValueError('Choose a new output or resume the identical configuration.')
        if json.loads((output / 'source.json').read_text()) != source_hashes():
            raise ValueError('Sources changed. Use a fresh output directory.')
        dataset = checked_dataset(output, config)
        for name in ('quality_complete.json', 'verification.json', 'partial_verification.json'):
            (output / name).unlink(missing_ok=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Choose an empty output directory.')
        output.mkdir(parents=True, exist_ok=True)
        write_json(existing, config)
        write_json(output / 'source.json', source_hashes())
        write_json(output / 'environment.json', environment())
        dataset = make_dataset(config)
        for split, records in dataset.items():
            save_records(output / f'dataset-{split}.npz', records)
        write_json(output / 'dataset_manifest.json', {split: {'graphs': len(records), 'sha256': dataset_digest(records)} for split, records in dataset.items()})
    dense_dataset = to_dense_dataset(dataset)
    all_metrics, calibration, training_rows = [], [], []
    for seed in config['seeds']:
        models = {}
        for variant in ('dense', 'sparse', 'no_mp'):
            folder = output / 'runs' / f'{variant}_{seed}'
            if resume and (folder / 'summary.json').exists():
                models[variant] = load_denoiser(folder / 'checkpoint.pt') if variant == 'dense' else load_sparse(folder / 'checkpoint.pt')
                summary = json.loads((folder / 'summary.json').read_text())
            elif variant == 'dense':
                models[variant], summary = train_dense(dense_config(config), dense_dataset, 'dense', seed, folder)
            else:
                train_config = {**config, 'layers': 0 if variant == 'no_mp' else config['layers']}
                models[variant], summary = train_sparse(train_config, dataset, seed, folder)
            write_json(folder / 'summary.json', summary)
            training_rows.append({'method': variant, 'seed': seed, **summary})
        write_json(output / 'training.json', training_rows)
        model_int8, quantization = quantize_sparse(models['sparse'])
        write_json(output / 'runs' / f'sparse_{seed}' / 'quantization.json', quantization)
        variants = settings(config)
        if model_int8 is not None:
            models['quantized'] = model_int8
            variants.append({'method': 'sparse_int8', 'kind': 'quantized', 'steps': max(config['sampling_steps'])})
        for spec in variants:
            for family, name in enumerate(FAMILIES):
                validation = [record for record in dataset['val'] if record['family'] == family]
                test = [record for record in dataset['test'] if record['family'] == family]
                train = [record for record in dataset['train'] if record['family'] == family]
                sampling_train = fit_baseline(spec['method'], train, family, config['n']) if spec['kind'] == 'baseline' else train
                temperatures = [1.0] if spec['kind'] == 'baseline' else config['temperatures']
                best, selected = float('inf'), temperatures[0]
                for temperature in temperatures:
                    generated = [generate_one(spec, models, config, family, seed * 100000 + family * 1000 + i,
                                              temperature, sampling_train) for i in range(config['validation_samples'])]
                    values = evaluate_samples(generated, validation, train, config['budget'], max_degree=spec.get('max_degree'))
                    objective = np.mean([values[f'{key}_mmd2'] for key in ('degree', 'clustering', 'spectral')])
                    calibration.append({'method': spec['method'], 'seed': seed, 'family': name,
                                        'temperature': temperature, 'objective': float(objective)})
                    if objective < best:
                        best, selected = objective, temperature
                operation = lambda: [generate_one(spec, models, config, family, seed * 100000 + family * 1000 + i + 50000,
                                                  selected, sampling_train) for i in range(len(test))]
                paired = profile_operation(operation, repetitions=1, warmup=0, return_result=True)
                generated = paired.pop('result')
                metrics = evaluate_samples(generated, test, train, config['budget'], max_degree=spec.get('max_degree'))
                save_records(output / 'samples' / f'{spec["method"]}_{name}_{seed}.npz', generated)
                row = {'method': spec['method'], 'seed': seed, 'family': name, 'temperature': selected,
                       'count': len(generated), 'steps': spec.get('steps', 1), **metrics,
                       'paired_profile': paired,
                       'paired_sampling_ms_median': float(np.median([g['sampling_ms'] for g in generated]))}
                all_metrics.append(row)
                write_json(output / 'metrics.json', all_metrics)
                write_json(output / 'calibration.json', calibration)
                print(json.dumps({'stage': 'quality', 'method': spec['method'], 'seed': seed, 'family': name,
                                  'degree_mmd2': row['degree_mmd2']}), flush=True)
    for family, name in enumerate(FAMILIES):
        reference = [record for record in dataset['reference'] if record['family'] == family]
        measured = evaluate_samples(reference, [record for record in dataset['test'] if record['family'] == family],
                                    [record for record in dataset['train'] if record['family'] == family], config['budget'])
        save_records(output / 'samples' / f'reference_sample_{name}_{config["dataset_seed"]}.npz', reference)
        all_metrics.append({'method': 'reference_sample', 'seed': config['dataset_seed'], 'family': name,
                            'temperature': None, 'count': len(reference), 'steps': 0,
                            'paired_sampling_ms_median': None, **measured})
    write_json(output / 'metrics.json', all_metrics)
    write_json(output / 'quality_complete.json', {'rows': len(all_metrics), 'models': len(training_rows), 'status': 'complete'})
    return all_metrics


def run_dynamic(config, output):
    from .constrained import dynamic_repair, radius_edges, projection, InfeasibleGraphError, FeasibilitySearchError
    from .sparse_model import load_sparse, score_candidates
    from .research_metrics import dynamic_metrics

    torch.set_num_threads(config['threads'])
    model = load_sparse(output / 'runs' / f'sparse_{config["seeds"][0]}' / 'checkpoint.pt')
    n, budget, cap = config['dynamic_n'], config['dynamic_budget'], config['dynamic_max_degree']
    radius = config['dynamic_radius']
    rows, snapshots = [], []
    for seed in config['seeds']:
        rng = np.random.default_rng(900000 + seed)
        side = int(np.ceil(np.sqrt(n)))
        grid = np.column_stack(np.unravel_index(np.arange(n), (side, side))) / max(side - 1, 1)
        positions = 0.15 + 0.7 * grid + rng.normal(0, 0.005, (n, 2))
        previous = {name: np.empty((0, 2), dtype=np.int64) for name in ('rebuild', 'retain', 'learned_retain')}
        for frame in range(config['dynamic_frames']):
            if frame:
                positions = np.clip(positions + rng.normal(0, config['dynamic_motion'], (n, 2)), 0, 1)
            for method in previous:
                started = time.perf_counter_ns()
                old = previous[method]
                try:
                    if method == 'rebuild':
                        candidates = radius_edges(positions, radius)
                        scores = -np.linalg.norm(positions[candidates[:, 0]] - positions[candidates[:, 1]], axis=1)
                        edges = projection(n, candidates, scores, budget, max_degree=cap)
                    else:
                        edges, repair = dynamic_repair(n, old, positions, radius, budget, max_degree=cap, seed=seed + frame)
                        from .constrained import constrained_update
                        candidates = radius_edges(positions, radius)
                        scores = -np.linalg.norm(positions[candidates[:, 0]] - positions[candidates[:, 1]], axis=1)
                        edges, filling = constrained_update(n, edges, candidates, scores, budget, max_degree=cap, max_swaps=0)
                        if method == 'learned_retain':
                            scores = score_candidates(model, n, edges, candidates, 0.25, 1)
                            scores -= np.linalg.norm(positions[candidates[:, 0]] - positions[candidates[:, 1]], axis=1)
                            old_set = {tuple(edge) for edge in old}
                            scores += np.array([1.0 if tuple(edge) in old_set else 0.0 for edge in candidates])
                            edges, update = constrained_update(n, edges, candidates, scores, budget, max_degree=cap, max_swaps=max(1, n // 8))
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    measured = dynamic_metrics(old, edges, positions, radius, budget, cap)
                    rows.append({'method': method, 'seed': seed, 'frame': frame, 'status': 'feasible',
                                 'sampling_ms': elapsed, **measured})
                    snapshots.append({'n': n, 'family': 1, 'method': method, 'seed': seed, 'frame': frame,
                                      'edges': edges, 'positions': positions.copy(), 'radius': radius})
                    previous[method] = edges
                except (InfeasibleGraphError, FeasibilitySearchError) as error:
                    rows.append({'method': method, 'seed': seed, 'frame': frame, 'status': error.status,
                                 'reason': error.reason, 'valid': False})
    impossible = np.column_stack((np.arange(n) * (radius + 1), np.zeros(n)))
    try:
        dynamic_repair(n, np.empty((0, 2), dtype=np.int64), impossible, radius, budget, cap)
        raise AssertionError('Disconnected range graph was accepted.')
    except InfeasibleGraphError as error:
        infeasible_check = {'status': error.status, 'reason': error.reason}
    save_records(output / 'dynamic_samples.npz', snapshots)
    write_json(output / 'dynamic.json', {'rows': rows, 'infeasible_case': infeasible_check,
                                       'learned_model_scope': 'Static topology model transferred without coordinate conditioning. Distance and retention scores are hand-designed.'})
    print(json.dumps({'stage': 'dynamic', 'frames': len(rows), 'infeasible_check': infeasible_check}), flush=True)


def profile_worker(spec, destination):
    from .constrained import candidate_edges, feasible_seed
    from .sparse_model import load_sparse, sample_sparse, score_candidates, quantize_sparse
    from .research_metrics import profile_operation
    from .models import EdgeDenoiser
    from .diffusion import generate

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    n, budget, seed = spec['n'], 2 * spec['n'], spec['seed']
    folder = Path(spec['run'])
    current = feasible_seed(n, budget, seed=seed)
    mode = spec['method']
    sparse = load_sparse(folder / 'runs' / f'sparse_{seed}' / 'checkpoint.pt')
    candidates = candidate_edges(n, current, spec['k'], np.random.default_rng(seed))
    quantization = None
    if mode == 'sparse_int8':
        sparse, quantization = quantize_sparse(sparse)
        if sparse is None:
            write_json(destination, {**spec, 'status': 'not_executed', 'quantization': quantization})
            return
    if mode in ('sparse_score', 'dense_candidate_score'):
        if mode == 'dense_candidate_score':
            candidates = np.column_stack(np.triu_indices(n, 1))
        operation = lambda: score_candidates(sparse, n, current, candidates, 0.5, 0)
        scope = 'Denoiser latency only; candidate preparation excluded from latency but included in process RSS. Same weights and active graph in both candidate modes.'
    elif mode == 'dense_generation':
        saved = torch.load(folder / 'runs' / f'dense_{seed}' / 'checkpoint.pt', map_location='cpu', weights_only=True)
        model = EdgeDenoiser(**{**saved['model_config'], 'n': n})
        state = {key: value for key, value in saved['state_dict'].items() if key != 'edge_index'}
        model.load_state_dict(state, strict=False)
        model.eval()
        dense = {'n': n, 'm': budget, 'diffusion_steps': spec['diffusion_steps']}
        operation = lambda: generate(model, 1, 0, dense, spec['steps'], 0.5, seed)
        scope = 'Full dense sampling, random initialization and final connectivity projection; model loading excluded.'
    else:
        operation = lambda: sample_sparse(sparse, n, budget, spec['steps'], spec['k'], seed, 0, temperature=0.5)
        scope = 'Full sparse sampling, candidates, neural scoring, admissible updates and invariant checks; model loading excluded.'
    measured = profile_operation(operation, repetitions=spec['repetitions'], warmup=spec['warmup'])
    if mode == 'dense_generation':
        candidate_count = n * (n - 1) // 2
    elif mode in ('sparse_score', 'dense_candidate_score'):
        candidate_count = len(candidates)
    else:
        payload = operation()
        candidate_count = payload['candidate_max']
    import io
    stream = io.BytesIO()
    torch.save((model if mode == 'dense_generation' else sparse).state_dict(), stream)
    row = {**spec, **measured, 'status': 'executed', 'scope': scope, 'candidate_count': candidate_count,
           'checkpoint_serialized_bytes': stream.tell(), 'quantization': quantization,
           'inference_only_size_extrapolation': n != spec['trained_n']}
    write_json(destination, row)


def run_scaling(config, output):
    rows = []
    methods = ('sparse_score', 'dense_candidate_score', 'sparse_generation', 'dense_generation', 'sparse_int8')
    folder = output / 'profiles'
    folder.mkdir(parents=True, exist_ok=True)
    for n in config['scaling_sizes']:
        for method in methods:
            spec = {'method': method, 'n': n, 'trained_n': config['n'], 'seed': config['seeds'][0],
                    'run': str(output.resolve()), 'k': config['k'], 'steps': config['scaling_steps'],
                    'diffusion_steps': config['diffusion_steps'], 'repetitions': config['profile_repetitions'],
                    'warmup': config['profile_warmup']}
            spec_path = folder / f'{method}_{n}_spec.json'
            result_path = folder / f'{method}_{n}.json'
            write_json(spec_path, spec)
            command = [sys.executable, '-m', 'frugal_graphs.research', 'profile-worker', '--spec', str(spec_path), '--output', str(result_path)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if result.returncode:
                raise RuntimeError(f'{method} n={n}: {result.stderr[-3000:]}')
            rows.append(json.loads(result_path.read_text()))
            write_json(output / 'scaling.json', rows)
            print(json.dumps({'stage': 'scaling', 'method': method, 'n': n, 'status': rows[-1]['status']}), flush=True)
    return rows


def check_complete_experiment(output, config, rows):
    required = [output / name for name in ('quality_complete.json', 'training.json')]
    if not all(path.is_file() for path in required):
        raise ValueError('Missing completion or training records. Full verification requires a completed experiment.')
    completion = json.loads(required[0].read_text())
    training = json.loads(required[1].read_text())
    seeds = config['seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Training seeds must be nonempty and distinct.')
    expected_models = {(method, seed) for seed in seeds for method in ('dense', 'sparse', 'no_mp')}
    actual_models = [(row['method'], row['seed']) for row in training]
    if len(actual_models) != len(expected_models) or set(actual_models) != expected_models:
        raise ValueError('Training records are missing, duplicated or unexpected.')
    expected_paths = {f'{method}_{seed}/checkpoint.pt' for method, seed in expected_models}
    actual_paths = {str(path.relative_to(output / 'runs')) for path in (output / 'runs').glob('*/checkpoint.pt')}
    if actual_paths != expected_paths:
        raise ValueError('Model checkpoints are missing or unexpected.')
    expected_rows = {}
    for seed in seeds:
        path = output / 'runs' / f'sparse_{seed}' / 'quantization.json'
        if not path.is_file():
            raise ValueError(f'Missing quantization status for seed {seed}.')
        status = json.loads(path.read_text()).get('status')
        if status not in ('executed', 'not_executed'):
            raise ValueError(f'Invalid quantization status for seed {seed}.')
        methods = [spec['method'] for spec in settings(config)]
        if status == 'executed':
            methods.append('sparse_int8')
        for method in methods:
            for family in FAMILIES:
                expected_rows[method, seed, family] = config['test_per_family']
    for family in FAMILIES:
        expected_rows['reference_sample', config['dataset_seed'], family] = config['reference_per_family']
    actual_rows = [(row['method'], row['seed'], row['family']) for row in rows]
    if len(actual_rows) != len(expected_rows) or set(actual_rows) != set(expected_rows):
        raise ValueError('Metric rows are missing, duplicated or unexpected.')
    if (completion.get('status') != 'complete' or completion.get('rows') != len(rows)
            or completion.get('models') != len(expected_models)):
        raise ValueError('The completion record does not match the experiment.')
    for row, key in zip(rows, actual_rows):
        if row.get('count') != expected_rows[key] or row.get('sample_count') != expected_rows[key]:
            raise ValueError(f'Incorrect sample count for {key}.')


def verify_research(output, *, partial=False):
    from .constrained import validate_graph
    from .research_metrics import evaluate_samples, dynamic_metrics

    verification_path = output / ('partial_verification.json' if partial else 'verification.json')
    (output / 'verification.json').unlink(missing_ok=True)
    verification_path.unlink(missing_ok=True)
    config = json.loads((output / 'config.json').read_text())
    dataset = checked_dataset(output, config)
    if (output / 'source_snapshots.json').exists():
        snapshots = json.loads((output / 'source_snapshots.json').read_text())
        hashes = json.loads((output / 'source.json').read_text())
        for name, digest in hashes.items():
            assert hashlib.sha256(snapshots[name].encode()).hexdigest() == digest, name
    rows = json.loads((output / 'metrics.json').read_text())
    if not partial:
        check_complete_experiment(output, config, rows)
    steps_checked = 0
    for row in rows:
        family = FAMILIES.index(row['family'])
        samples = load_records(output / 'samples' / f'{row["method"]}_{row["family"]}_{row["seed"]}.npz')
        if not partial:
            expected_count = config['reference_per_family'] if row['method'] == 'reference_sample' else config['test_per_family']
            if len(samples) != expected_count or any(sample['n'] != config['n'] or sample['family'] != family for sample in samples):
                raise ValueError(f'Invalid sample batch for {row["method"]}, seed {row["seed"]}, family {row["family"]}.')
        cap = 6 if row['method'] == 'sparse_degree6' else None
        metrics = evaluate_samples(samples, [r for r in dataset['test'] if r['family'] == family],
                                   [r for r in dataset['train'] if r['family'] == family], config['budget'], cap)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert np.isclose(value, row[key], rtol=1e-9, atol=1e-12), (row['method'], key)
        for sample in samples:
            for edges in sample.get('sampling', {}).get('trajectory', []):
                assert validate_graph(sample['n'], edges, config['budget'], cap)['valid']
                steps_checked += 1
    if (output / 'dynamic.json').exists():
        recorded = json.loads((output / 'dynamic.json').read_text())
        previous = {}
        for sample in load_records(output / 'dynamic_samples.npz'):
            key = sample['seed'], sample['method']
            old = previous.get(key, np.empty((0, 2), dtype=np.int64))
            measured = dynamic_metrics(old, sample['edges'], sample['positions'], sample['radius'], config['dynamic_budget'], config['dynamic_max_degree'])
            row = next(r for r in recorded['rows'] if (r['seed'], r['method'], r['frame']) == (*key, sample['frame']))
            assert measured['valid'] == row['valid']
            assert measured['churn_edges'] == row['churn_edges']
            assert np.isclose(measured['lambda2'], row['lambda2'])
            previous[key] = sample['edges']
    checkpoints_checked = 0
    if (output / 'training.json').exists():
        from .diffusion import load_denoiser
        from .sparse_model import load_sparse
        for record in json.loads((output / 'training.json').read_text()):
            folder = output / 'runs' / f'{record["method"]}_{record["seed"]}'
            checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
            if record['method'] == 'dense':
                history = json.loads((folder / 'history.json').read_text())
                best = min(history, key=lambda row: row['val']['loss'])
                assert best['epoch'] == checkpoint['epoch'] == record['best_epoch']
                assert np.isclose(best['val']['loss'], checkpoint['val_bce'])
                model = load_denoiser(folder / 'checkpoint.pt')
                expected_parameters = record['parameters_total']
            else:
                history = checkpoint['summary']['history']
                best = min(history, key=lambda row: row['val_bce'])
                assert best['epoch'] == record['best_epoch'] == checkpoint['summary']['best_epoch']
                assert np.isclose(best['val_bce'], record['val_bce'])
                model = load_sparse(folder / 'checkpoint.pt')
                expected_parameters = record['parameters']
            assert sum(value.numel() for value in model.parameters()) == expected_parameters
            assert all(torch.isfinite(value).all() for value in model.parameters())
            checkpoints_checked += 1
    summary = {'scope': 'partial' if partial else 'complete',
               'quality_rows_recomputed': len(rows), 'valid_intermediate_graphs_checked': steps_checked,
               'checkpoints_checked': checkpoints_checked}
    write_json(verification_path, summary)
    print(json.dumps(summary), flush=True)
    return summary


def evaluate_external(path, output):
    from .research_metrics import import_external_samples, evaluate_samples

    document = json.loads(path.read_text())
    config = json.loads((output / 'config.json').read_text())
    manifest = json.loads((output / 'dataset_manifest.json').read_text())
    method = document['metadata']['method']
    expected = {family: config['test_per_family'] * len(document['metadata']['seeds']) for family in range(2)}
    imported = import_external_samples(path, method, expected_counts=expected, expected_split_sha256=manifest['test']['sha256'])
    seeds = document['metadata']['seeds']
    if len(set(seeds)) != len(seeds):
        raise ValueError('External seeds must be distinct.')
    for record in imported['samples']:
        if record.get('seed') not in seeds or record['n'] != config['n']:
            raise ValueError('Each external sample needs its declared seed and benchmark node count.')
    train, test = (load_records(output / f'dataset-{name}.npz') for name in ('train', 'test'))
    rows = []
    for seed in seeds:
        for family, name in enumerate(FAMILIES):
            selected = [record for record in imported['samples'] if record['family'] == family and record['seed'] == seed]
            if len(selected) != config['test_per_family']:
                raise ValueError('Every external seed/family must contain the full test sample count.')
            measured = evaluate_samples(selected, [r for r in test if r['family'] == family],
                                        [r for r in train if r['family'] == family], config['budget'])
            rows.append({'method': method, 'seed': seed, 'family': name, **measured})
    write_json(output / f'external-{method}.json', {'results': rows, 'provenance': imported['metadata']})


def make_report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter

    rows = json.loads((output / 'metrics.json').read_text())
    methods = list(dict.fromkeys(row['method'] for row in rows))
    config = json.loads((output / 'config.json').read_text())
    lines = ['# Recorded research experiments', '',
             f'{config["n"]} nodes, budget {config["budget"]}, seeds {config["seeds"]}. Values below are means across seeds.', '',
             '| Family | Method | Degree MMD² | Clustering MMD² | Spectral MMD² | Valid | Planar |',
             '|---|---|---:|---:|---:|---:|---:|']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for family_index, family in enumerate(FAMILIES):
        for method in methods:
            selection = [row for row in rows if row['method'] == method and row['family'] == family]
            values = [np.mean([row[key] for row in selection]) for key in ('degree_mmd2', 'clustering_mmd2', 'spectral_mmd2', 'valid_rate', 'planar_rate')]
            lines.append('| ' + ' | '.join([family, method, *[f'{value:.4f}' for value in values]]) + ' |')
        for column, metric in enumerate(('degree_mmd2', 'clustering_mmd2', 'spectral_mmd2')):
            axis = axes[family_index, column]
            averages, errors = [], []
            for method in methods:
                data = [row[metric] for row in rows if row['method'] == method and row['family'] == family]
                averages.append(np.mean(data))
                errors.append(np.std(data, ddof=1) if len(data) > 1 else 0)
            axis.barh(methods, averages, xerr=errors, color='#267b88', alpha=0.85)
            axis.set_title(f'{family}: {metric.replace("_mmd2", "")} MMD²')
            axis.tick_params(labelsize=8)
            axis.invert_yaxis()
            axis.grid(axis='x', alpha=0.2)
    fig.savefig(output / 'quality.png', dpi=160)
    plt.close(fig)
    if (output / 'scaling.json').exists():
        scaling = json.loads((output / 'scaling.json').read_text())
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
        for method in dict.fromkeys(row['method'] for row in scaling):
            group = [row for row in scaling if row['method'] == method and row['status'] == 'executed']
            if not group:
                continue
            for axis, key, divisor, label in zip(axes, ('latency_ms_median', 'rss_sampled_peak_bytes', 'candidate_count'),
                                                (1, 1024 ** 2, 1), ('Latency (ms)', 'Sampled process RSS (MiB)', 'Candidate edges')):
                axis.plot([r['n'] for r in group], [r[key] / divisor for r in group], marker='o', label=method)
                axis.set(xlabel='Nodes', ylabel=label, xscale='log', yscale='log')
                axis.set_xticks(sorted({row['n'] for row in scaling}))
                axis.xaxis.set_major_formatter(ScalarFormatter())
                axis.xaxis.set_minor_locator(NullLocator())
                axis.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        fig.savefig(output / 'scaling.png', dpi=160)
        plt.close(fig)
        lines.extend(['', '## Scaling', '', 'Each setting runs in a fresh CPU process. Score-only and complete-generation scopes are separate.', '',
                      '| Method | Nodes | ms median | RSS MiB | RSS delta MiB | Candidates |', '|---|---:|---:|---:|---:|---:|'])
        for row in scaling:
            if row['status'] == 'executed':
                lines.append(f'| {row["method"]} | {row["n"]} | {row["latency_ms_median"]:.3f} | {row["rss_sampled_peak_bytes"] / 1024**2:.2f} | {row["rss_peak_delta_bytes"] / 1024**2:.2f} | {row["candidate_count"]} |')
    if (output / 'dynamic.json').exists():
        dynamic = json.loads((output / 'dynamic.json').read_text())['rows']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
        lines.extend(['', '## Moving agents', '', '| Method | Feasible / attempted | Mean edges | Mean churn after frame 0 | Mean λ₂ |', '|---|---:|---:|---:|---:|'])
        for method in ('rebuild', 'retain', 'learned_retain'):
            group = [r for r in dynamic if r['method'] == method and r['status'] == 'feasible']
            if not group:
                continue
            frames = sorted({r['frame'] for r in group if r['frame'] > 0})
            for axis, key in zip(axes, ('churn_edges', 'lambda2', 'sampling_ms')):
                axis.plot(frames, [np.mean([r[key] for r in group if r['frame'] == frame]) for frame in frames], label=method)
                axis.set(xlabel='Frame', ylabel=key)
                axis.grid(alpha=0.2)
            after = [r['churn_edges'] for r in group if r['frame'] > 0]
            attempted = sum(r['method'] == method for r in dynamic)
            lines.append(f'| {method} | {len(group)}/{attempted} | {np.mean([r["edge_count"] for r in group]):.2f} | {np.mean(after):.3f} | {np.mean([r["lambda2"] for r in group]):.4f} |')
        axes[0].legend(fontsize=8)
        fig.savefig(output / 'dynamic.png', dpi=160)
        plt.close(fig)
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def pack_results(folder):
    import shutil

    verify_research(folder)
    index = {'format': 1, 'files': {}, 'arrays': {}, 'sha256': {}, 'packaged_source_sha256': source_hashes()}
    arrays, checkpoints = {}, {}
    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        name = str(path.relative_to(folder))
        if path.suffix == '.json':
            index['files'][name] = json.loads(path.read_text())
        elif path.suffix == '.npz':
            mapping = {}
            with np.load(path, allow_pickle=False) as saved:
                for key in saved.files:
                    stored = f'array_{len(arrays):05d}'
                    arrays[stored] = saved[key]
                    mapping[key] = stored
                    index['sha256'][stored] = hashlib.sha256(saved[key].tobytes()).hexdigest()
            index['arrays'][name] = mapping
        elif path.suffix == '.pt':
            checkpoints[name] = torch.load(path, map_location='cpu', weights_only=True)
    for name in ('data/research', 'checkpoints/research', 'results/research', 'figures/research', 'docs'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / 'data/research/graphs.npz', **arrays)
    torch.save(checkpoints, ROOT / 'checkpoints/research/checkpoints.pt')
    index['checkpoint_sha256'] = hashlib.sha256((ROOT / 'checkpoints/research/checkpoints.pt').read_bytes()).hexdigest()
    write_json(ROOT / 'results/research/results.json', index)
    for name in ('quality', 'scaling', 'dynamic'):
        if (folder / f'{name}.png').exists():
            shutil.copyfile(folder / f'{name}.png', ROOT / f'figures/research/{name}.png')
    if (folder / 'report.md').exists():
        shutil.copyfile(folder / 'report.md', ROOT / 'docs/RESULTS.md')


def restore_results(folder):
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Choose an empty output directory.')
    index = json.loads((ROOT / 'results/research/results.json').read_text())
    if hashlib.sha256((ROOT / 'checkpoints/research/checkpoints.pt').read_bytes()).hexdigest() != index['checkpoint_sha256']:
        raise ValueError('Checkpoint archive hash mismatch.')
    for name, value in index['files'].items():
        write_json(folder / name, value)
    with np.load(ROOT / 'data/research/graphs.npz', allow_pickle=False) as saved:
        for name, mapping in index['arrays'].items():
            values = {}
            for key, stored in mapping.items():
                values[key] = saved[stored]
                if hashlib.sha256(values[key].tobytes()).hexdigest() != index['sha256'][stored]:
                    raise ValueError(f'Array hash mismatch: {name}/{key}')
            destination = folder / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **values)
    checkpoints = torch.load(ROOT / 'checkpoints/research/checkpoints.pt', map_location='cpu', weights_only=True)
    for name, saved in checkpoints.items():
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, destination)


def sample_from_bundle(args):
    from .sparse_model import SparseDenoiser, sample_sparse
    from .project import plot_samples

    torch.set_num_threads(1)
    checkpoints = torch.load(ROOT / 'checkpoints/research/checkpoints.pt', map_location='cpu', weights_only=True)
    index = json.loads((ROOT / 'results/research/results.json').read_text())
    if hashlib.sha256((ROOT / 'checkpoints/research/checkpoints.pt').read_bytes()).hexdigest() != index['checkpoint_sha256']:
        raise ValueError('Checkpoint archive hash mismatch.')
    config = index['files']['config.json']
    checkpoint_name = f'runs/sparse_{config["seeds"][0]}/checkpoint.pt'
    saved = checkpoints[checkpoint_name]
    model = SparseDenoiser(**saved['model_config'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    budget = args.budget if args.budget is not None else 2 * args.nodes
    samples = []
    for index in range(args.count):
        result = sample_sparse(model, args.nodes, budget, args.steps, 4, args.seed + index,
                               FAMILIES.index(args.family), max_degree=args.degree_cap, temperature=0.5, trace=True)
        samples.append({'n': args.nodes, 'edges': result.pop('edges'), 'family': FAMILIES.index(args.family), 'sampling': result,
                        'seed': args.seed + index, 'budget': budget, 'degree_cap': args.degree_cap, 'k': 4,
                        'steps': args.steps, 'temperature': 0.5, 'checkpoint': checkpoint_name,
                        'checkpoint_archive_sha256': hashlib.sha256((ROOT / 'checkpoints/research/checkpoints.pt').read_bytes()).hexdigest(),
                        'trained_n': config['n']})
    save_records(args.output / 'samples.npz', samples)
    adjacency = to_dense_dataset({'sample': samples})['sample']['adj']
    plot_samples(adjacency, args.family, args.steps, 0.5, args.seed, args.output / 'samples.png')
    print(json.dumps({'samples': len(samples), 'output': str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description='Constraint-aware graph generation experiments.')
    parser.add_argument('command', choices=('run', 'scale', 'dynamic', 'verify', 'report', 'profile-worker', 'external', 'pack', 'restore', 'sample', 'export'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/research.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated/research')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-scaling', action='store_true')
    parser.add_argument('--partial', action='store_true', help='Verify available results only; valid only with verify.')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--family', choices=FAMILIES, default='sbm')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--degree-cap', type=int)
    args = parser.parse_args()
    if args.partial and args.command != 'verify':
        parser.error('--partial is only valid with verify')
    config = json.loads(args.config.read_text())['smoke' if args.smoke else 'main']
    if args.command in ('scale', 'dynamic'):
        config = json.loads((args.output / 'config.json').read_text())
    if args.command == 'run':
        train_and_evaluate(config, args.output, args.resume)
        run_dynamic(config, args.output)
        if not args.skip_scaling:
            run_scaling(config, args.output)
        make_report(args.output)
    elif args.command == 'scale':
        run_scaling(config, args.output)
    elif args.command == 'dynamic':
        run_dynamic(config, args.output)
    elif args.command == 'verify':
        verify_research(args.output, partial=args.partial)
    elif args.command == 'report':
        make_report(args.output)
    elif args.command == 'profile-worker':
        profile_worker(json.loads(args.spec.read_text()), args.output)
    elif args.command == 'external':
        evaluate_external(args.spec, args.output)
    elif args.command == 'pack':
        pack_results(args.output)
    elif args.command == 'restore':
        restore_results(args.output)
    elif args.command == 'sample':
        if args.count < 1:
            parser.error('--count must be positive')
        sample_from_bundle(args)
    elif args.command == 'export':
        if args.spec is None:
            parser.error('export needs --spec for its destination JSON')
        dataset = {split: load_records(args.output / f'dataset-{split}.npz') for split in ('train', 'val', 'test', 'reference')}
        write_json(args.spec, {'dataset': dataset, 'split_sha256': {split: dataset_digest(rows) for split, rows in dataset.items()}})


if __name__ == '__main__':
    main()
