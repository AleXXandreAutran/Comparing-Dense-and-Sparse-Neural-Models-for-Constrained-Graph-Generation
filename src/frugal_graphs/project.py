from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from .graphs import FAMILY_NAMES
from .diffusion import generate, load_denoiser
from .models import EdgeDenoiser

def plot_samples(adjacencies, family, steps, temperature, seed, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    shown = min(16, len(adjacencies))
    columns = min(4, shown)
    rows = math.ceil(shown / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.7 * rows + 0.7), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis('off')
        if index >= shown:
            continue
        graph = nx.from_numpy_array(adjacencies[index])
        positions = nx.spring_layout(graph, seed=seed + index, iterations=70)
        nx.draw_networkx_edges(graph, positions, ax=axis, edge_color='#94a3b8', alpha=0.6, width=1.0)
        nx.draw_networkx_nodes(graph, positions, ax=axis, node_color='#1763a6', node_size=45, linewidths=0)
        axis.set_title(f'Graph {index + 1:02d}', color='#344154', fontsize=10)
        axis.margins(0.12)
    figure.suptitle(f'Generated {family} graphs', fontsize=15, color='#152d45', y=0.99)
    figure.text(0.5, 0.02, f'{steps} denoising steps · temperature {temperature:g} · seed {seed} · showing {shown}/{len(adjacencies)}', ha='center', fontsize=10, color='#526477')
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor='white')
    plt.close(figure)

def sample_main(argv=None):
    parser = argparse.ArgumentParser(description='Generate graphs from a trained model.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model', default='diffusion_compact_seed11')
    parser.add_argument('--config', type=Path, help='Experiment config; by default infer results directory from checkpoint.')
    parser.add_argument('--family', choices=FAMILY_NAMES, default='community')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--steps', type=int, default=8, help='Number of reverse denoising steps.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Final edgewise Gumbel noise scale; inspect validation calibration for measured settings.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('generated/example.npz'))
    parser.add_argument('--plot', type=Path, help='Optional PNG montage of up to sixteen samples.')
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error('--count must be at least 1')
    if args.steps < 1:
        parser.error('--steps must be at least 1')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if not np.isfinite(args.temperature) or args.temperature < 0:
        parser.error('--temperature must be finite and nonnegative')
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.plot and args.plot.suffix.lower() != '.png':
        parser.error('--plot must end in .png')
    if args.checkpoint:
        if not args.checkpoint.is_file():
            parser.error(f'checkpoint does not exist: {args.checkpoint}')
        config_path = args.config or args.checkpoint.resolve().parents[2] / 'config.json'
        if not config_path.is_file():
            parser.error('provide --config for this checkpoint')
        config = json.loads(config_path.read_text())
        model = load_denoiser(args.checkpoint)
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        checkpoint_run = args.checkpoint.parent.name
    else:
        records = read_records()
        key = f'results/diffusion/runs/{args.model}/checkpoint.pt'
        saved = torch.load(ROOT / 'checkpoints/legacy/checkpoints.pt', map_location='cpu', weights_only=True)
        if key not in saved:
            parser.error('unknown model')
        checkpoint = saved[key]
        model = EdgeDenoiser(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        config = json.loads(args.config.read_text()) if args.config else records['files']['results/diffusion/config.json']
        checkpoint_hash = hashlib.sha256((ROOT / 'checkpoints/legacy/checkpoints.pt').read_bytes()).hexdigest()
        checkpoint_run = args.model
    if args.steps > config['diffusion_steps']:
        parser.error("--steps cannot exceed the experiment's diffusion_steps")
    torch.set_num_threads(1)
    if model.n != config['n']:
        parser.error('checkpoint node count does not match the experiment config')
    family = FAMILY_NAMES.index(args.family)
    adjacencies = generate(model, args.count, family, config, args.steps, args.temperature, args.seed, mode='connected')
    edge_counts = (adjacencies.sum(axis=(1, 2)) / 2).astype(int)
    connected = [nx.is_connected(nx.from_numpy_array(adjacency)) for adjacency in adjacencies]
    metadata = {'checkpoint_run': checkpoint_run, 'checkpoint_file_sha256': checkpoint_hash, 'model_config': model.config(), 'family': args.family, 'count': args.count, 'n': config['n'], 'm': config['m'], 'diffusion_steps': config['diffusion_steps'], 'sampling_steps': args.steps, 'temperature': args.temperature, 'temperature_note': 'User-chosen final Gumbel noise scale; inspect results/legacy/results.json for validated settings.', 'seed': args.seed, 'mode': 'connected', 'connected_count': int(sum(connected)), 'exact_edge_budget_count': int(np.count_nonzero(edge_counts == config['m'])), 'symmetric_count': int(np.count_nonzero((adjacencies == adjacencies.transpose(0, 2, 1)).all(axis=(1, 2)))), 'zero_diagonal_count': int(np.count_nonzero((np.diagonal(adjacencies, axis1=1, axis2=2) == 0).all(axis=1))), 'edge_counts': edge_counts.tolist(), 'validity_note': 'Connectivity and exact edge count are enforced by the combinatorial projection; they are not learned guarantees.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, adj=adjacencies, family=np.full(args.count, family, dtype=np.int64))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.plot:
        plot_samples(adjacencies, args.family, args.steps, args.temperature, args.seed, args.plot)
    print(json.dumps({'output': str(args.output), 'metadata': str(args.output.with_suffix('.json')), 'connected_count': metadata['connected_count'], 'exact_edge_budget_count': metadata['exact_edge_budget_count'], 'count': args.count}))
    return metadata


ROOT = Path(__file__).resolve().parents[2]


def read_records():
    return json.loads((ROOT / 'results/legacy/results.json').read_text())


def extract(output):
    import csv

    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory.')
    output.mkdir(parents=True, exist_ok=True)
    records = read_records()
    for name, value in records['files'].items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, indent=2) + '\n')
    for name, rows in records['csv'].items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with np.load(ROOT / 'data/legacy/graphs.npz', allow_pickle=False) as arrays:
        for name, mapping in records['arrays'].items():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            values = {key: arrays[value] for key, value in mapping.items()}
            if destination.suffix == '.npz':
                np.savez_compressed(destination, **values)
            else:
                np.save(destination, values['value'])
    checkpoints = torch.load(ROOT / 'checkpoints/legacy/checkpoints.pt', map_location='cpu', weights_only=True)
    for name, checkpoint in checkpoints.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)
    return records


def verify():
    from .graphs import load_dataset, evaluate_graphs
    from tempfile import TemporaryDirectory

    records = read_records()
    for name, expected in records['packaging']['source_sha256'].items():
        snapshot = records['reorganization']['original_source_snapshots'][name]
        assert hashlib.sha256(snapshot.encode()).hexdigest() == expected, name
    for name, expected in records['reorganization']['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with np.load(ROOT / 'data/legacy/graphs.npz', allow_pickle=False) as arrays:
        for name, mapping in records['arrays'].items():
            for key, stored in mapping.items():
                digest = hashlib.sha256(arrays[stored].tobytes()).hexdigest()
                assert digest == records['original_array_sha256'][name][key], (name, key)
    checkpoints = torch.load(ROOT / 'checkpoints/legacy/checkpoints.pt', map_location='cpu', weights_only=True)
    for name, checkpoint in checkpoints.items():
        for key, value in checkpoint['state_dict'].items():
            digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
            assert digest == records['original_checkpoint_sha256'][name][key], (name, key)
    metric_count = 0
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / 'recorded'
        extract(root)
        for stage in ('main', 'diffusion'):
            folder = root / 'results' / stage
            dataset = load_dataset(folder / 'dataset.npz')
            manifest = json.loads((folder / 'dataset_manifest.json').read_text())
            for split, arrays in dataset.items():
                assert len(arrays['adj']) == manifest[split]['graphs']
                assert hashlib.sha256(arrays['adj'].tobytes()).hexdigest() == manifest[split]['adjacency_sha256']
                assert hashlib.sha256(arrays['family'].tobytes()).hexdigest() == manifest[split]['labels_sha256']
            references = json.loads((folder / 'metric_reference.json').read_text())
            config = json.loads((folder / 'config.json').read_text())
            rows = json.loads((folder / 'metrics.json').read_text())
            for row in rows:
                family = FAMILY_NAMES.index(row['family'])
                if row['method'] == 'reference_sample':
                    adjacency = dataset['reference']['adj'][dataset['reference']['family'] == family]
                else:
                    path = folder / 'samples' / f"{row['method']}_{row['family']}_seed{row['seed']}.npz"
                    with np.load(path, allow_pickle=False) as archive:
                        adjacency = archive['adj']
                assert len(adjacency) == row['sample_count']
                expected = evaluate_graphs(adjacency, dataset['test']['adj'][dataset['test']['family'] == family], dataset['train']['adj'], references[str(family)], config['m'])
                for key, value in expected.items():
                    assert np.isclose(value, row[key], atol=1e-12, rtol=1e-10), (row['method'], row['seed'], key)
                metric_count += 1
            training = json.loads((folder / 'training.json').read_text())
            for row in training:
                run = folder / 'runs' / f"{row['variant']}_seed{row['seed']}"
                checkpoint = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=True)
                history = json.loads((run / 'history.json').read_text())
                assert checkpoint['epoch'] == row['best_epoch']
                assert min(history, key=lambda epoch: epoch['val']['loss'])['epoch'] == row['best_epoch']
                recorded = checkpoint.get('val', {}).get('loss', checkpoint.get('val_bce'))
                assert np.isclose(recorded, row['best_validation']['loss'])
            completion = json.loads((folder / 'completion.json').read_text())
            assert completion['quality_rows'] == len(rows)
            assert completion['training_runs'] == len(training)
    summary = {'metric_rows_verified': metric_count, 'checkpoints_verified': len(checkpoints), 'array_files_verified': len(records['arrays'])}
    print(json.dumps(summary), flush=True)
    return summary


def train_main(argv):
    import importlib.metadata
    import os
    import platform
    import sys
    from .graphs import generate_dataset, save_dataset
    from .cvae import write_json

    parser = argparse.ArgumentParser(description='Train and evaluate the recorded protocol.')
    parser.add_argument('--stage', choices=('main', 'diffusion'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    records = read_records()
    name = ('smoke' if args.stage == 'main' else 'diffusion_smoke') if args.smoke else args.stage
    config = json.loads(args.config.read_text()) if args.config else records['files'][f'configs/{name}.json']
    if args.stage == 'main':
        if args.dataset:
            parser.error('--dataset is only used for diffusion')
        from .cvae import run
    else:
        from .diffusion import run
        if args.dataset:
            config['dataset'] = str(args.dataset.resolve())
        elif not args.config:
            folder = args.output.resolve() / 'input'
            folder.mkdir(parents=True, exist_ok=True)
            if args.smoke:
                data_config = records['files']['configs/smoke.json']
                fields = ('n', 'm', 'train_per_family', 'val_per_family', 'test_per_family', 'reference_per_family')
                dataset = generate_dataset(**{key: data_config[key] for key in fields}, seed=data_config['dataset_seed'])
                save_dataset(dataset, folder / 'dataset.npz')
                manifest = {split: {'graphs': len(values['adj']), 'adjacency_sha256': hashlib.sha256(values['adj'].tobytes()).hexdigest(), 'labels_sha256': hashlib.sha256(values['family'].tobytes()).hexdigest()} for split, values in dataset.items()}
            else:
                with np.load(ROOT / 'data/legacy/graphs.npz', allow_pickle=False) as arrays:
                    mapping = records['arrays']['results/main/dataset.npz']
                    np.savez_compressed(folder / 'dataset.npz', **{key: arrays[value] for key, value in mapping.items()})
                manifest = records['files']['results/main/dataset_manifest.json']
            write_json(folder / 'dataset_manifest.json', manifest)
            environment = {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'device': 'cpu', 'torch_threads': config['threads'], 'torch_interop_threads': 1, 'packages': {package: importlib.metadata.version(package) for package in ('torch', 'numpy', 'scipy', 'networkx', 'matplotlib', 'psutil')}}
            write_json(folder / 'environment.json', environment)
            config['dataset'] = str(folder / 'dataset.npz')
    run(config, args.output.resolve(), resume=args.resume)


def main():
    import sys

    parser = argparse.ArgumentParser(description='Train, sample or check the graph generation experiments.')
    parser.add_argument('command', choices=('sample', 'train', 'verify', 'extract'))
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    if args.command == 'sample':
        sample_main(remaining)
    elif args.command == 'train':
        train_main(remaining)
    elif args.command == 'verify':
        if remaining:
            parser.error('verify takes no arguments')
        verify()
    else:
        extraction = argparse.ArgumentParser(description='Expand the recorded experiment files.')
        extraction.add_argument('--output', type=Path, default=Path('generated/recorded'))
        destination = extraction.parse_args(remaining).output
        extract(destination)
        print(json.dumps({'output': str(destination)}))


if __name__ == '__main__':
    main()
