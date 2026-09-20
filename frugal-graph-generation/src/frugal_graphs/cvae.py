from __future__ import annotations

from numbers import Integral, Real
from time import perf_counter_ns
import numpy as np
import torch
from .graphs import perturb_scores, project_connected

def _timed_calls(operation, repetitions):
    for _ in range(5):
        operation()
    timings = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        start = perf_counter_ns()
        operation()
        timings[index] = (perf_counter_ns() - start) * 1e-06
    return {'median': float(np.median(timings)), 'p25': float(np.percentile(timings, 25)), 'p75': float(np.percentile(timings, 75))}

def benchmark_generator(model_or_None, empirical_logits_or_None, n, m, family=0, temperature=1.0, batch_sizes=(1, 64), repetitions=40, seed=771):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 1:
        raise ValueError('n must be a positive integer')
    edge_count = n * (n - 1) // 2
    project_connected(np.zeros(edge_count), n, m)
    if not isinstance(family, Integral) or isinstance(family, (bool, np.bool_)) or family not in (0, 1):
        raise ValueError('family must be 0 or 1')
    if not isinstance(repetitions, Integral) or isinstance(repetitions, (bool, np.bool_)) or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    if not isinstance(temperature, Real) or isinstance(temperature, (bool, np.bool_)) or (not np.isfinite(temperature)) or (temperature < 0):
        raise ValueError('temperature must be finite and nonnegative')
    batches = tuple(batch_sizes)
    if not batches or any((not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1 for value in batches)):
        raise ValueError('batch_sizes must contain positive integers')
    model = model_or_None
    empirical = empirical_logits_or_None
    if model is not None and empirical is not None:
        raise ValueError('choose a neural model or empirical logits, not both')
    if model is not None:
        if model.n != n:
            raise ValueError('model node count differs from benchmark n')
        if any((value.device.type != 'cpu' for value in (*model.parameters(), *model.buffers()))):
            raise ValueError('this latency benchmark supports CPU models only')
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError('the reference latency benchmark requires a float32 model')
        if family >= model.families:
            raise ValueError('family is unavailable in this model')
    elif empirical is not None:
        empirical = np.asarray(empirical)
        if empirical.shape != (2, edge_count) or empirical.dtype.kind not in 'fiu' or (not np.isfinite(empirical).all()):
            raise ValueError('empirical logits must be finite real values of shape [2,E]')
        empirical = empirical.astype(np.float64, copy=False)
    row_scores = np.zeros(edge_count, dtype=np.float64) if empirical is None else empirical[family]
    original_training = model.training if model is not None else None
    results = []
    if model is not None:
        model.eval()
    try:
        with torch.inference_mode():
            for batch_size in batches:
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(batch_size)]))
                if model is not None:
                    z = torch.from_numpy(rng.normal(size=(batch_size, model.latent)).astype(np.float32))
                    conditions = torch.full((batch_size,), family, dtype=torch.long)

                    def forward():
                        return model.decode(z, conditions)

                    def generate():
                        latent = torch.from_numpy(rng.normal(size=(batch_size, model.latent)).astype(np.float32))
                        labels = torch.full((batch_size,), family, dtype=torch.long)
                        logits = model.decode(latent, labels).numpy()
                        scores = perturb_scores(logits, rng, temperature)
                        return project_connected(scores, n, m)
                else:

                    def forward():
                        return np.broadcast_to(row_scores, (batch_size, edge_count)).copy()

                    def generate():
                        logits = np.broadcast_to(row_scores, (batch_size, edge_count)).copy()
                        scores = perturb_scores(logits, rng, temperature)
                        return project_connected(scores, n, m)
                forward_timings = _timed_calls(forward, repetitions)
                generation_timings = _timed_calls(generate, repetitions)
                median = generation_timings['median']
                record = {'batch_size': int(batch_size)}
                record.update({f'forward_ms_{key}': value for key, value in forward_timings.items()})
                record.update({f'generation_ms_{key}': value for key, value in generation_timings.items()})
                record['ms_per_graph'] = median / batch_size
                record['graphs_per_second'] = 1000.0 * batch_size / median
                results.append(record)
    finally:
        if model is not None:
            model.train(original_training)
    return results
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from .graphs import project_connected, project_topk, perturb_scores
from .graphs import FAMILY_NAMES, generate_dataset, save_dataset, load_dataset
from .graphs import evaluate_graphs, fit_metric_reference, graph_statistics, squared_mmd
from .models import GraphCVAE, parameter_counts

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def train_model(config, dataset, variant, seed, folder):
    seed_everything(seed)
    model = GraphCVAE(n=config['n'], **config['models'][variant])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    train_a = torch.from_numpy(dataset['train']['adj'])
    train_f = torch.from_numpy(dataset['train']['family'])
    val_a = torch.from_numpy(dataset['val']['adj'])
    val_f = torch.from_numpy(dataset['val']['family'])
    history, best_loss, best_epoch = ([], float('inf'), 0)
    start = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        order = torch.randperm(len(train_a))
        train_stats = {key: 0.0 for key in ('loss', 'bce', 'kl_per_dim', 'degree_loss')}
        beta = config['beta'] * min(epoch / 20, 1.0)
        for indices in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.loss(train_a[indices], train_f[indices], beta, config['structural_weight'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key in train_stats:
                train_stats[key] += stats[key] * len(indices) / len(train_a)
        model.eval()
        val_stats = {key: 0.0 for key in train_stats}
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(100000 + seed)
            for start_index in range(0, len(val_a), config['batch_size']):
                end = start_index + config['batch_size']
                _, stats = model.loss(val_a[start_index:end], val_f[start_index:end], config['beta'], config['structural_weight'])
                weight = len(val_a[start_index:end]) / len(val_a)
                for key in val_stats:
                    val_stats[key] += stats[key] * weight
        history.append({'epoch': epoch, 'train': train_stats, 'val': val_stats, 'train_beta': beta})
        if val_stats['loss'] < best_loss:
            best_loss, best_epoch = (val_stats['loss'], epoch)
            torch.save({'model_config': model.config(), 'state_dict': model.state_dict(), 'seed': seed, 'variant': variant, 'epoch': epoch, 'val': val_stats}, folder / 'checkpoint.pt')
        if epoch == 1 or epoch % 20 == 0 or epoch == config['epochs']:
            print(json.dumps({'stage': 'train', 'variant': variant, 'seed': seed, 'epoch': epoch, 'val_bce': round(val_stats['bce'], 5), 'val_kl_per_dim': round(val_stats['kl_per_dim'], 5)}), flush=True)
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        posterior_means, _ = model.encode(val_a, val_f)
        within_family_variance = torch.stack([posterior_means[val_f == f].var(0, unbiased=False) for f in range(2)]).mean(0)
        diagnostic_rng = np.random.default_rng(500000 + seed)
        prior = torch.from_numpy(diagnostic_rng.standard_normal((256, model.latent)).astype(np.float32))
        prior_stdev = [float(model.decode(prior, torch.full((256,), f, dtype=torch.long)).std(0).mean()) for f in range(2)]
    summary = {'variant': variant, 'seed': seed, 'best_epoch': best_epoch, 'training_seconds': elapsed, 'best_validation': checkpoint['val'], 'active_latent_dimensions_within_family': int((within_family_variance > 0.01).sum()), 'posterior_mean_variance_within_family': within_family_variance.tolist(), 'prior_edge_logit_std_by_family': prior_stdev, **parameter_counts(model)}
    write_json(folder / 'history.json', history)
    write_json(folder / 'training.json', summary)
    return (model, summary)

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = GraphCVAE(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def sample_graphs(model, empirical_logits, family, count, config, temperature, seed, connected=True):
    latent_stream, edge_stream = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(latent_stream)
    edge_rng = np.random.default_rng(edge_stream)
    if model is not None:
        z = torch.from_numpy(rng.standard_normal((count, model.latent)).astype(np.float32))
        labels = torch.full((count,), family, dtype=torch.long)
        with torch.inference_mode():
            logits = model.decode(z, labels).numpy()
    elif empirical_logits is not None:
        logits = np.broadcast_to(empirical_logits[family], (count, empirical_logits.shape[1]))
    else:
        logits = np.zeros((count, config['n'] * (config['n'] - 1) // 2))
    scores = perturb_scores(logits, edge_rng, temperature)
    projection = project_connected if connected else project_topk
    return projection(scores, config['n'], config['m'])

def calibrate(model, empirical_logits, dataset, references, config, seed):
    selected, records = ({}, [])
    for family, name in enumerate(FAMILY_NAMES):
        val = dataset['val']['adj'][dataset['val']['family'] == family]
        target = graph_statistics(val)
        best_score = float('inf')
        for temperature in config['temperatures']:
            samples = sample_graphs(model, empirical_logits, family, config['validation_samples'], config, temperature, 200000 + seed + family * 1000)
            statistics = graph_statistics(samples)
            scores = {feature + '_mmd2': squared_mmd(statistics[feature], target[feature], references[family][feature + '_bandwidth']) for feature in ('degree', 'clustering', 'spectral')}
            objective = sum(scores.values()) / 3
            records.append({'family': name, 'temperature': temperature, 'selection_score': objective, **scores})
            if objective < best_score:
                best_score, selected[family] = (objective, temperature)
    return (selected, records)

def save_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys((key for row in rows for key in row)))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def run(config, output, resume=False, skip_benchmarks=False):
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists():
        previous = json.loads((output / 'config.json').read_text())
        if previous != config:
            raise ValueError('Output contains a different configuration; use a fresh output directory.')
        if not resume:
            raise ValueError('Output exists; pass --resume to reuse exact matching artifacts.')
    write_json(output / 'config.json', config)
    source_manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob('*.py'))}
    if resume and (output / 'source_manifest.json').exists() and (json.loads((output / 'source_manifest.json').read_text()) != source_manifest):
        raise ValueError('Source code differs from the recorded run; use a fresh output directory.')
    write_json(output / 'source_manifest.json', source_manifest)
    versions = {package: importlib.metadata.version(package) for package in ('torch', 'numpy', 'scipy', 'networkx', 'matplotlib', 'psutil')}
    environment = {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(), 'device': 'cpu', 'packages': versions, 'timing_note': 'CPU wall time; no energy measurement. Parameter storage is not peak activation/process memory.'}
    write_json(output / 'environment.json', environment)
    data_path = output / 'dataset.npz'
    if resume and data_path.exists():
        dataset = load_dataset(data_path)
    else:
        fields = ('n', 'm', 'train_per_family', 'val_per_family', 'test_per_family', 'reference_per_family')
        dataset = generate_dataset(**{key: config[key] for key in fields}, seed=config['dataset_seed'])
        save_dataset(dataset, data_path)
    manifest = {split: {'graphs': len(values['adj']), 'adjacency_sha256': hashlib.sha256(values['adj'].tobytes()).hexdigest(), 'labels_sha256': hashlib.sha256(values['family'].tobytes()).hexdigest()} for split, values in dataset.items()}
    write_json(output / 'dataset_manifest.json', manifest)
    references = {family: fit_metric_reference(dataset['train']['adj'][dataset['train']['family'] == family]) for family in range(2)}
    write_json(output / 'metric_reference.json', references)
    print(json.dumps({'stage': 'dataset_ready', 'manifest': manifest}), flush=True)
    edges = np.triu_indices(config['n'], 1)
    probabilities = np.stack([(dataset['train']['adj'][dataset['train']['family'] == family][:, edges[0], edges[1]].sum(0) + 0.5) / (config['train_per_family'] + 1.0) for family in range(2)])
    empirical_logits = np.log(probabilities / (1 - probabilities))
    np.save(output / 'empirical_logits.npy', empirical_logits)
    results, training, latency, selections, models = ([], [], [], [], {})
    for seed in config['seeds']:
        for variant in config['models']:
            folder = output / 'runs' / f'{variant}_seed{seed}'
            if resume and (folder / 'training.json').exists():
                model = load_model(folder / 'checkpoint.pt')
                summary = json.loads((folder / 'training.json').read_text())
            else:
                model, summary = train_model(config, dataset, variant, seed, folder)
            models[variant, seed] = model
            training.append(summary)
    for seed in config['seeds']:
        for method in ['random_projected', 'empirical_projected', *config['models']]:
            model = models.get((method, seed))
            empirical = empirical_logits if method == 'empirical_projected' else None
            if method == 'random_projected':
                temperatures, selection = ({0: 1.0, 1: 1.0}, [])
            else:
                temperatures, selection = calibrate(model, empirical, dataset, references, config, seed)
            selections.extend(({'method': method, 'seed': seed, **row} for row in selection))
            for family, family_name in enumerate(FAMILY_NAMES):
                heldout = dataset['test']['adj'][dataset['test']['family'] == family]
                train_adj = dataset['train']['adj']
                for connected in [True, False] if model is not None else [True]:
                    name = method if connected else method + '_topk'
                    samples = sample_graphs(model, empirical, family, len(heldout), config, temperatures[family], 300000 + seed + family * 1000, connected)
                    metrics = evaluate_graphs(samples, heldout, train_adj, references[family], config['m'])
                    row = {'method': name, 'seed': seed, 'family': family_name, 'sample_count': len(samples), 'temperature': temperatures[family], **metrics}
                    results.append(row)
                    sample_path = output / 'samples' / f'{name}_{family_name}_seed{seed}.npz'
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sample_path, adj=samples)
                    print(json.dumps({'stage': 'evaluation', **{k: row[k] for k in ('method', 'seed', 'family', 'temperature', 'valid_rate', 'degree_mmd2', 'clustering_mmd2', 'spectral_mmd2')}}), flush=True)
    for family, family_name in enumerate(FAMILY_NAMES):
        ref = dataset['reference']['adj'][dataset['reference']['family'] == family]
        heldout = dataset['test']['adj'][dataset['test']['family'] == family]
        train_adj = dataset['train']['adj']
        results.append({'method': 'reference_sample', 'seed': config['dataset_seed'], 'family': family_name, 'sample_count': len(ref), 'temperature': 0.0, **evaluate_graphs(ref, heldout, train_adj, references[family], config['m'])})
    save_rows(output / 'metrics.csv', results)
    write_json(output / 'metrics.json', results)
    write_json(output / 'training.json', training)
    save_rows(output / 'calibration.csv', selections)
    if not skip_benchmarks:
        for seed in config['seeds']:
            for method in ['random_projected', 'empirical_projected', *config['models']]:
                model = models.get((method, seed))
                empirical = empirical_logits if method == 'empirical_projected' else None
                temperature = next((row['temperature'] for row in results if row['seed'] == seed and row['method'] == method and (row['family'] == 'community')))
                measurements = benchmark_generator(model, empirical, config['n'], config['m'], family=0, temperature=temperature, batch_sizes=tuple(config['latency_batch_sizes']), repetitions=config['latency_repetitions'], seed=400000 + seed)
                latency.extend(({'method': method, 'seed': seed, **row} for row in measurements))
                print(json.dumps({'stage': 'benchmark', 'method': method, 'seed': seed}), flush=True)
        save_rows(output / 'latency.csv', latency)
        write_json(output / 'latency.json', latency)
    write_json(output / 'completion.json', {'status': 'complete', 'quality_rows': len(results), 'training_runs': len(training), 'latency_rows': len(latency)})
    print(json.dumps({'stage': 'complete', 'output': str(output)}), flush=True)
