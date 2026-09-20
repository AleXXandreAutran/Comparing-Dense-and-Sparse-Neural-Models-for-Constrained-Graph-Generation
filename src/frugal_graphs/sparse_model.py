from __future__ import annotations

import copy
import math
import time as timer
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SparseMessageLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_map = nn.Linear(hidden, hidden)
        self.neighbor_map = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes, left, right, degree):
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, left, nodes[right])
        aggregate.index_add_(0, right, nodes[left])
        aggregate = aggregate / degree.clamp_min(1).unsqueeze(1)
        return nodes + F.silu(self.norm(self.self_map(nodes) + self.neighbor_map(aggregate)))


class SparseDenoiser(nn.Module):
    def __init__(self, hidden=32, layers=2):
        super().__init__()
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 0:
            raise ValueError('layers must be a nonnegative integer')
        self.hidden, self.layers_count = hidden, layers
        self.register_buffer('_anchor', torch.empty(0))
        self.node_input = nn.Linear(10, hidden)
        self.message_layers = nn.ModuleList([SparseMessageLayer(hidden) for _ in range(layers)])
        self.edge_head = nn.Sequential(nn.Linear(3 * hidden + 5, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def config(self):
        return {'hidden': self.hidden, 'layers': self.layers_count}

    def forward(self, n, edge_index, candidate_index, time, family, positions=None):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError('n must be an integer >= 2')
        if family not in (0, 1):
            raise ValueError('family must be 0 or 1')
        t = float(time)
        if not math.isfinite(t) or not 0 <= t <= 1:
            raise ValueError('time must be between zero and one')
        device, dtype = self._anchor.device, self._anchor.dtype
        indices = []
        for value in (edge_index, candidate_index):
            index = torch.as_tensor(value, device=device)
            if index.ndim != 2 or index.shape[0] != 2:
                raise ValueError('edge indices must have shape (2, E)')
            if index.is_floating_point() or index.is_complex() or index.dtype == torch.bool:
                raise ValueError('edge indices must contain integers')
            index = index.to(torch.long)
            if index.numel() and ((index < 0).any() or (index >= n).any() or (index[0] == index[1]).any()):
                raise ValueError('edge indices contain invalid vertices or loops')
            indices.append(index)
        edge_index, candidate_index = indices
        keys = torch.unique(torch.minimum(edge_index[0], edge_index[1]) * n + torch.maximum(edge_index[0], edge_index[1]), sorted=True)
        left, right = keys // n, keys % n
        degree = torch.zeros(n, device=device, dtype=dtype)
        degree.index_add_(0, left, torch.ones_like(left, dtype=dtype))
        degree.index_add_(0, right, torch.ones_like(right, dtype=dtype))
        has_positions = positions is not None
        coordinates = torch.zeros((n, 2), device=device, dtype=dtype)
        if has_positions:
            coordinates = torch.as_tensor(positions, device=device, dtype=dtype)
            if coordinates.shape != (n, 2) or not torch.isfinite(coordinates).all():
                raise ValueError('positions must be a finite (n, 2) array')
            coordinates = coordinates - coordinates.mean(0, keepdim=True)
        ones = torch.ones(n, device=device, dtype=dtype)
        features = torch.stack((ones, degree / (n - 1), degree / degree.mean().clamp_min(1), ones * t, ones * t * t, ones * (family == 0), ones * (family == 1), ones * has_positions, coordinates[:, 0], coordinates[:, 1]), dim=1)
        nodes = F.silu(self.node_input(features))
        for layer in self.message_layers:
            nodes = layer(nodes, left, right, degree)
        u, v = candidate_index
        pair_keys = torch.minimum(u, v) * n + torch.maximum(u, v)
        present = torch.zeros(len(u), device=device, dtype=dtype)
        if len(keys):
            location = torch.searchsorted(keys, pair_keys).clamp_max(len(keys) - 1)
            present = (keys[location] == pair_keys).to(dtype)
        distance = (coordinates[u] - coordinates[v]).square().sum(1).sqrt()
        scalar = torch.stack((present, (degree[u] + degree[v]) / (n - 1), (degree[u] - degree[v]).abs() / (n - 1), distance, torch.full_like(distance, float(has_positions))), dim=1)
        pair = torch.cat((nodes[u] + nodes[v], (nodes[u] - nodes[v]).abs(), nodes[u] * nodes[v], scalar), dim=1)
        return self.edge_head(pair).squeeze(-1)


def _pairs(value, n):
    edges = np.asarray(value)
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('edges must be an integer (E, 2) array')
    if np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError('invalid edges')
    return np.unique(np.sort(edges.astype(np.int64), axis=1), axis=0)


def _training_candidates(n, clean, k, rng, complete=False):
    if k < 1:
        raise ValueError('k must be positive')
    total = n * (n - 1) // 2
    required = {int(u) * n + int(v) for u, v in clean}
    target = total if complete else min(total, len(required) + int(k) * n)
    if target == total:
        return np.asarray([(u, v) for u in range(n) for v in range(u + 1, n)], dtype=np.int64)
    while len(required) < target:
        pairs = rng.integers(0, n, size=(min(4096, 2 * (target - len(required)) + 8), 2))
        for u, v in pairs:
            if u != v:
                required.add(int(min(u, v)) * n + int(max(u, v)))
                if len(required) == target:
                    break
    keys = np.asarray(sorted(required), dtype=np.int64)
    return np.column_stack((keys // n, keys % n))


def _example(record, k, steps, rng, complete=False):
    n = int(record['n'])
    clean = _pairs(record['edges'], n)
    candidates = _training_candidates(n, clean, k, rng, complete)
    clean_keys = clean[:, 0] * n + clean[:, 1]
    candidate_keys = candidates[:, 0] * n + candidates[:, 1]
    target = np.isin(candidate_keys, clean_keys).astype(np.float32)
    t = int(rng.integers(1, steps + 1)) / steps
    retention = 0.0 if t == 1 else math.cos(math.pi * t / 2) ** 2
    prior = float(target.mean())
    noisy = candidates[rng.random(len(candidates)) < retention * target + (1 - retention) * prior]
    return (n, torch.from_numpy(noisy.T.copy()), torch.from_numpy(candidates.T.copy()), t, int(record['family']), record.get('positions'), torch.from_numpy(target))


def _example_loss(model, example):
    n, current, candidates, t, family, positions, target = example
    return F.binary_cross_entropy_with_logits(model(n, current, candidates, t, family, positions), target)


def train_sparse(config, dataset, seed, output):
    options = dict(config)
    epochs, batch_size = int(options.get('epochs', 30)), int(options.get('batch_size', 16))
    steps, k = int(options.get('diffusion_steps', 16)), int(options.get('k', 6))
    if min(epochs, batch_size, steps, k) < 1:
        raise ValueError('epochs, batch_size, diffusion_steps and k must be positive')
    if not dataset.get('train') or not dataset.get('val'):
        raise ValueError('nonempty train and val splits are required')
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = SparseDenoiser(int(options.get('hidden', 32)), int(options.get('layers', 2)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get('lr', options.get('learning_rate', 1e-3))))
    complete = bool(options.get('complete_candidates', False))
    val_rng = np.random.default_rng(int(options.get('val_noise_seed', 271828)))
    validation = [_example(row, k, steps, val_rng, complete) for row in dataset['val']]
    history, best_loss, best_state, best_epoch = [], math.inf, None, 0
    started = timer.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(dataset['train']))
        train_loss = 0.0
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                example = _example(dataset['train'][int(index)], k, steps, rng, complete)
                loss = _example_loss(model, example)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                (loss / len(indices)).backward()
                train_loss += float(loss.detach())
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([float(_example_loss(model, example)) for example in validation]))
        history.append({'epoch': epoch, 'train_bce': train_loss / len(order), 'val_bce': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    output = Path(output)
    checkpoint = output if output.suffix == '.pt' else output / 'checkpoint.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    summary = {'seed': int(seed), 'best_epoch': best_epoch, 'val_bce': best_loss, 'epochs': epochs, 'training_seconds': timer.perf_counter() - started, 'parameters': sum(p.numel() for p in model.parameters()), 'history': history, 'checkpoint': str(checkpoint), 'objective': 'BCE on sampled candidates; clean edges always included, so this is not an unbiased all-pairs likelihood', 'process': 'learned constrained editing; no exact diffusion posterior is claimed'}
    torch.save({'format': 'sparse-edit-v1', 'model_config': model.config(), 'state_dict': best_state, 'training_config': options, 'summary': summary}, checkpoint)
    return model, summary


def load_sparse(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != 'sparse-edit-v1':
        raise ValueError('unsupported sparse checkpoint format')
    model = SparseDenoiser(**saved['model_config'])
    model.load_state_dict(saved['state_dict'])
    return model.eval()


def score_candidates(model, n, current, candidates, time, family, positions=None):
    current, candidates = _pairs(current, n), np.asarray(candidates)
    if candidates.size == 0:
        candidates = np.empty((0, 2), dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError('candidates must have shape (C, 2)')
    model.eval()
    with torch.inference_mode():
        values = model(n, torch.as_tensor(current.T.copy()), torch.as_tensor(candidates.T.copy()), time, family, positions)
    return values.cpu().numpy()


def sample_sparse(model, n, budget, steps, k, seed, family, max_degree=None, positions=None, radius=None, mode='admissible', refresh=True, trace=False, temperature=1.0, candidates='sparse', initial_edges=None, max_swaps=None, condition_positions=False):
    from .constrained import candidate_edges, canonical_edges, constrained_update, feasible_seed, projection, radius_edges, validate_graph
    if mode not in ('admissible', 'projected'):
        raise ValueError('mode must be admissible or projected')
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('k must be a positive integer')
    if radius is not None and positions is None:
        raise ValueError('radius requires positions')
    if candidates not in ('sparse', 'dense'):
        raise ValueError('candidates must be sparse or dense')
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    candidate_mode = candidates
    proposal_k = n - 1 if candidate_mode == 'dense' else k
    swap_limit = n if max_swaps is None else max_swaps
    rng = np.random.default_rng(seed)
    started = timer.perf_counter()
    allowed = radius_edges(positions, radius) if radius is not None else None
    current = feasible_seed(n, budget, max_degree=max_degree, allowed=allowed, seed=seed) if initial_edges is None else canonical_edges(initial_edges, n)
    if initial_edges is not None:
        initial_status = validate_graph(n, current, budget, max_degree=max_degree, positions=positions, radius=radius)
        if not initial_status.get('valid', False):
            raise ValueError('initial_edges must satisfy all requested constraints')
    initial = current.copy()
    fixed_candidates = candidate_edges(n, current, proposal_k, rng, allowed=allowed) if not refresh else None
    metadata, trajectory = [], [current.copy()] if trace else []
    phase = {'initialization_seconds': timer.perf_counter() - started, 'candidate_seconds': 0.0, 'scoring_seconds': 0.0, 'update_seconds': 0.0, 'validation_seconds': 0.0}
    for step in range(steps):
        time = (steps - step) / steps
        tick = timer.perf_counter()
        candidates = candidate_edges(n, current, proposal_k, rng, allowed=allowed) if refresh else fixed_candidates
        phase['candidate_seconds'] += timer.perf_counter() - tick
        tick = timer.perf_counter()
        scores = score_candidates(model, n, current, candidates, time, family, positions if condition_positions else None)
        scores = scores + temperature * max(0.05, time) * rng.gumbel(size=len(scores))
        phase['scoring_seconds'] += timer.perf_counter() - tick
        tick = timer.perf_counter()
        if mode == 'admissible':
            current, info = constrained_update(n, current, candidates, scores, budget, max_degree=max_degree, max_swaps=swap_limit)
        else:
            current = projection(n, candidates, scores, budget, max_degree=max_degree)
            info = {'mode': 'projected'}
        phase['update_seconds'] += timer.perf_counter() - tick
        tick = timer.perf_counter()
        valid = validate_graph(n, current, budget, max_degree=max_degree, positions=positions, radius=radius)
        phase['validation_seconds'] += timer.perf_counter() - tick
        metadata.append({'step': step + 1, 'time': time, 'candidates': len(candidates), 'edges': len(current), 'constraints': valid, 'update': info})
        if trace:
            trajectory.append(current.copy())
    return {'edges': current, 'initial_edges': initial, 'steps': metadata, 'trajectory': trajectory, 'timing': phase | {'total_seconds': timer.perf_counter() - started}, 'candidate_max': max(row['candidates'] for row in metadata), 'mode': mode, 'refresh': bool(refresh), 'candidate_mode': candidate_mode, 'temperature': float(temperature), 'condition_positions': bool(condition_positions)}


def quantize_sparse(model):
    engines = [name for name in torch.backends.quantized.supported_engines if name != 'none']
    if not engines:
        return None, {'status': 'not_executed', 'reason': 'no supported quantized CPU backend'}
    previous_engine = torch.backends.quantized.engine
    engine = previous_engine if previous_engine in engines else engines[0]
    try:
        torch.backends.quantized.engine = engine
        quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
        score_candidates(quantized, 3, np.asarray([[0, 1], [1, 2]]), np.asarray([[0, 2]]), 0.5, 0)
    except (RuntimeError, NotImplementedError, AttributeError) as error:
        return None, {'status': 'not_executed', 'reason': str(error)}
    finally:
        if previous_engine in engines:
            torch.backends.quantized.engine = previous_engine
    return quantized, {'status': 'executed', 'engine': engine, 'method': 'dynamic int8 Linear'}
