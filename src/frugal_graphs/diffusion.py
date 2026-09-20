from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .graphs import project_connected, project_topk, perturb_scores
from .graphs import FAMILY_NAMES, load_dataset, save_dataset
from .models import EdgeDenoiser
from .models import alpha_bar, reverse_probability
from .cvae import write_json, save_rows, seed_everything
from .graphs import graph_statistics, squared_mmd, evaluate_graphs, fit_metric_reference

def to_adjacency(edges, n):
    index = torch.triu_indices(n, n, 1, device=edges.device)
    result = torch.zeros((len(edges), n, n), dtype=edges.dtype, device=edges.device)
    result[:, index[0], index[1]] = edges
    return result + result.transpose(1, 2)

def noisy_batch(clean_edges, steps, prior, generator=None):
    t = torch.randint(1, steps + 1, (len(clean_edges),), generator=generator)
    a = alpha_bar(t, steps).unsqueeze(-1)
    probabilities = a * clean_edges + (1 - a) * prior
    noisy = (torch.rand(probabilities.shape, generator=generator) < probabilities).float()
    return (noisy, t)

def train(config, dataset, variant, seed, folder):
    seed_everything(seed)
    model = EdgeDenoiser(n=config['n'], **config['models'][variant])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    index = torch.triu_indices(config['n'], config['n'], 1)
    train_edges = torch.from_numpy(dataset['train']['adj'])[:, index[0], index[1]]
    train_family = torch.from_numpy(dataset['train']['family'])
    val_edges = torch.from_numpy(dataset['val']['adj'])[:, index[0], index[1]]
    val_family = torch.from_numpy(dataset['val']['family'])
    prior = config['m'] / len(index[0])
    val_generator = torch.Generator().manual_seed(600000 + seed)
    val_noisy, val_time = noisy_batch(val_edges, config['diffusion_steps'], prior, val_generator)
    val_adjacency = to_adjacency(val_noisy, config['n'])
    history, best_loss, best_epoch = ([], float('inf'), 0)
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        order = torch.randperm(len(train_edges))
        training_loss = 0.0
        for indices in order.split(config['batch_size']):
            noisy, t = noisy_batch(train_edges[indices], config['diffusion_steps'], prior)
            optimizer.zero_grad(set_to_none=True)
            logits = model(to_adjacency(noisy, config['n']), train_family[indices], t.float() / config['diffusion_steps'])
            loss = F.binary_cross_entropy_with_logits(logits, train_edges[indices])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite denoising loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_loss += float(loss.detach()) * len(indices) / len(train_edges)
        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for start_index in range(0, len(val_edges), config['batch_size']):
                end = start_index + config['batch_size']
                logits = model(val_adjacency[start_index:end], val_family[start_index:end], val_time[start_index:end].float() / config['diffusion_steps'])
                loss = F.binary_cross_entropy_with_logits(logits, val_edges[start_index:end])
                validation_loss += float(loss) * len(val_edges[start_index:end]) / len(val_edges)
        history.append({'epoch': epoch, 'train': {'bce': training_loss, 'loss': training_loss}, 'val': {'bce': validation_loss, 'loss': validation_loss}})
        if validation_loss < best_loss:
            best_loss, best_epoch = (validation_loss, epoch)
            torch.save({'model_config': model.config(), 'state_dict': model.state_dict(), 'seed': seed, 'variant': variant, 'epoch': epoch, 'val_bce': validation_loss}, folder / 'checkpoint.pt')
        if epoch == 1 or epoch % 20 == 0:
            print(json.dumps({'stage': 'diffusion_train', 'variant': variant, 'seed': seed, 'epoch': epoch, 'val_bce': round(validation_loss, 6)}), flush=True)
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    parameters = sum((p.numel() for p in model.parameters()))
    summary = {'variant': variant, 'seed': seed, 'best_epoch': best_epoch, 'training_seconds': elapsed, 'best_validation': {'bce': best_loss, 'loss': best_loss}, 'parameters_total': parameters, 'parameters_generation': parameters, 'parameter_bytes_fp32': 4 * parameters, 'generation_parameter_bytes_fp32': 4 * parameters}
    write_json(folder / 'training.json', summary)
    write_json(folder / 'history.json', history)
    return (model, summary)

def load_denoiser(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = EdgeDenoiser(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()

@torch.inference_mode()
def generate(model, count, family, config, sampling_steps, temperature, seed, mode='connected'):
    n, m, steps = (config['n'], config['m'], config['diffusion_steps'])
    edge_count = n * (n - 1) // 2
    prior = m / edge_count
    generator = torch.Generator().manual_seed(seed)
    edges = (torch.rand((count, edge_count), generator=generator) < prior).float()
    labels = torch.full((count,), family, dtype=torch.long)
    grid = np.linspace(steps, 0, sampling_steps + 1).round().astype(int)
    if len(np.unique(grid)) != len(grid):
        raise ValueError('sampling_steps must not exceed diffusion_steps')
    logits = None
    for t, s in zip(grid[:-1], grid[1:]):
        logits = model(to_adjacency(edges, n), labels, torch.full((count,), float(t) / steps))
        if s > 0:
            probability = reverse_probability(logits.sigmoid(), edges, float(alpha_bar(torch.tensor(s), steps)), float(alpha_bar(torch.tensor(t), steps)), prior)
            edges = (torch.rand(probability.shape, generator=generator) < probability).float()
    probabilities = logits.sigmoid()
    if mode == 'raw':
        final_edges = (torch.rand(probabilities.shape, generator=generator) < probabilities).float()
        return to_adjacency(final_edges, n).numpy()
    scores = perturb_scores(logits.numpy(), np.random.default_rng(seed + 900000), temperature)
    if mode == 'connected':
        return project_connected(scores, n, m)
    if mode == 'topk':
        return project_topk(scores, n, m)
    if mode == 'all':
        final_edges = (torch.rand(probabilities.shape, generator=generator) < probabilities).float()
        return {'connected': project_connected(scores, n, m), 'topk': project_topk(scores, n, m), 'raw': to_adjacency(final_edges, n).numpy()}
    raise ValueError('unknown generation mode')

def calibrate(model, dataset, references, config, seed, sampling_steps):
    temperatures, records = ({}, [])
    for family, name in enumerate(FAMILY_NAMES):
        val = dataset['val']['adj'][dataset['val']['family'] == family]
        target = graph_statistics(val)
        best = float('inf')
        for temperature in config['temperatures']:
            samples = generate(model, config['validation_samples'], family, config, sampling_steps, temperature, 700000 + seed + family * 1000)
            stats = graph_statistics(samples)
            metrics = {key + '_mmd2': squared_mmd(stats[key], target[key], references[family][key + '_bandwidth']) for key in ('degree', 'clustering', 'spectral')}
            score = sum(metrics.values()) / 3
            records.append({'family': name, 'sampling_steps': sampling_steps, 'temperature': temperature, 'selection_score': score, **metrics})
            if score < best:
                best, temperatures[family] = (score, temperature)
    return (temperatures, records)

def benchmark(model, config, steps, temperature, seed):
    rows = []
    for batch in config['latency_batch_sizes']:
        generator = torch.Generator().manual_seed(seed)
        noisy = to_adjacency((torch.rand((batch, config['n'] * (config['n'] - 1) // 2), generator=generator) < config['m'] / (config['n'] * (config['n'] - 1) // 2)).float(), config['n'])
        labels = torch.zeros(batch, dtype=torch.long)
        times = torch.full((batch,), 0.5)
        measured = {}
        for scope in ('forward', 'generation'):
            durations = []
            for repetition in range(config['latency_repetitions'] + 5):
                start = time.perf_counter_ns()
                with torch.inference_mode():
                    if scope == 'forward':
                        model(noisy, labels, times)
                    else:
                        generate(model, batch, 0, config, steps, temperature, seed + repetition)
                duration = (time.perf_counter_ns() - start) / 1000000.0
                if repetition >= 5:
                    durations.append(duration)
            for key, value in zip(('p25', 'median', 'p75'), np.percentile(durations, [25, 50, 75])):
                measured[f'{scope}_ms_{key}'] = float(value)
        rows.append({'batch_size': batch, 'sampling_steps': steps, **measured, 'ms_per_graph': measured['generation_ms_median'] / batch, 'graphs_per_second': 1000 * batch / measured['generation_ms_median']})
    return rows

def run(config, output, resume=False):
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists():
        if not resume or json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('Use a new output directory or --resume with identical config.')
    write_json(output / 'config.json', config)
    source = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob('*.py'))}
    if resume and (output / 'source_manifest.json').exists() and (json.loads((output / 'source_manifest.json').read_text()) != source):
        raise ValueError('Source changed since previous run.')
    write_json(output / 'source_manifest.json', source)
    dataset = load_dataset(config['dataset'])
    save_dataset(dataset, output / 'dataset.npz')
    source_folder = Path(config['dataset']).parent
    for name in ['dataset_manifest.json', 'environment.json']:
        if (source_folder / name).exists():
            (output / name).write_bytes((source_folder / name).read_bytes())
    references = {f: fit_metric_reference(dataset['train']['adj'][dataset['train']['family'] == f]) for f in range(2)}
    write_json(output / 'metric_reference.json', references)
    models, training = ({}, [])
    for seed in config['seeds']:
        for variant in config['models']:
            folder = output / 'runs' / f'{variant}_seed{seed}'
            if resume and (folder / 'training.json').exists():
                model = load_denoiser(folder / 'checkpoint.pt')
                summary = json.loads((folder / 'training.json').read_text())
            else:
                model, summary = train(config, dataset, variant, seed, folder)
            models[variant, seed] = model
            training.append(summary)
    write_json(output / 'training.json', training)
    results, selections, latencies = ([], [], [])
    for seed in config['seeds']:
        for variant in config['models']:
            model = models[variant, seed]
            for steps in config['sampling_steps']:
                method = f'{variant}_s{steps}'
                temperatures, records = calibrate(model, dataset, references, config, seed, steps)
                selections.extend(({'method': method, 'seed': seed, **r} for r in records))
                for family, name in enumerate(FAMILY_NAMES):
                    heldout = dataset['test']['adj'][dataset['test']['family'] == family]
                    samples = generate(model, len(heldout), family, config, steps, temperatures[family], 800000 + seed + family * 1000, mode='all')
                    for mode, adj in samples.items():
                        label = method if mode == 'connected' else f'{method}_{mode}'
                        metrics = evaluate_graphs(adj, heldout, dataset['train']['adj'], references[family], config['m'])
                        row = {'method': label, 'seed': seed, 'family': name, 'sampling_steps': steps, 'sample_count': len(adj), 'temperature': temperatures[family], **metrics}
                        results.append(row)
                        path = output / 'samples' / f'{label}_{name}_seed{seed}.npz'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, adj=adj)
                    print(json.dumps({'stage': 'diffusion_eval', 'method': method, 'seed': seed, 'family': name, 'temperature': temperatures[family], 'degree_mmd2': results[-3]['degree_mmd2'], 'clustering_mmd2': results[-3]['clustering_mmd2'], 'spectral_mmd2': results[-3]['spectral_mmd2']}), flush=True)
                write_json(output / 'metrics.json', results)
                save_rows(output / 'metrics.csv', results)
                save_rows(output / 'calibration.csv', selections)
    for seed in config['seeds']:
        for variant in config['models']:
            for steps in config['sampling_steps']:
                method = f'{variant}_s{steps}'
                temperature = next((r['temperature'] for r in results if r['method'] == method and r['seed'] == seed and (r['family'] == 'community')))
                latencies.extend(({'method': method, 'seed': seed, **r} for r in benchmark(models[variant, seed], config, steps, temperature, 900000 + seed)))
                print(json.dumps({'stage': 'diffusion_benchmark', 'method': method, 'seed': seed}), flush=True)
    write_json(output / 'latency.json', latencies)
    save_rows(output / 'latency.csv', latencies)
    write_json(output / 'completion.json', {'status': 'complete', 'quality_rows': len(results), 'training_runs': len(training), 'latency_rows': len(latencies)})
    print(json.dumps({'stage': 'complete', 'output': str(output)}), flush=True)
