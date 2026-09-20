from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import networkx as nx
import numpy as np
import psutil
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance


def _graph(record):
    n = record.get('n')
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    edges = np.asarray(record.get('edges', []))
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('edges must have shape [E, 2]')
    if edges.dtype.kind not in 'iuf' or not np.isfinite(edges).all():
        raise ValueError('edge endpoints must be finite numbers')
    integers = np.equal(edges, np.floor(edges)).all(axis=1)
    inside = ((edges >= 0) & (edges < n)).all(axis=1)
    nonloops = edges[:, 0] != edges[:, 1]
    accepted = edges[integers & inside & nonloops].astype(np.int64)
    accepted = np.unique(np.sort(accepted, axis=1), axis=0)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(accepted.tolist())
    return graph, bool(len(accepted) == len(edges))


def _histogram(values, upper=1.0):
    values = np.asarray(values, dtype=float)
    if not len(values):
        histogram = np.zeros(32, dtype=float)
        histogram[0] = 1
        return histogram
    return np.histogram(np.clip(values, 0, upper), bins=32, range=(0, upper))[0] / len(values)


def algebraic_connectivity(graph):
    n = len(graph)
    if n < 2 or not nx.is_connected(graph):
        return 0.0
    adjacency = nx.to_numpy_array(graph, dtype=float)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    return float(max(0.0, np.linalg.eigvalsh(laplacian)[1]))


def failure_resilience(graph, probability=0.1, trials=20, seed=0):
    if not 0 <= probability <= 1 or trials < 1:
        raise ValueError('probability must be in [0, 1] and trials must be positive')
    edges = list(graph.edges())
    rng = np.random.default_rng(seed)
    connected, largest = [], []
    for _ in range(trials):
        damaged = nx.Graph()
        damaged.add_nodes_from(graph.nodes())
        damaged.add_edges_from(edge for edge, keep in zip(edges, rng.random(len(edges)) >= probability) if keep)
        components = list(nx.connected_components(damaged))
        connected.append(len(components) == 1)
        largest.append(max(map(len, components), default=0) / max(len(graph), 1))
    return {'connected_rate': float(np.mean(connected)),
            'largest_component_fraction': float(np.mean(largest)),
            'edge_failure_probability': float(probability), 'trials': int(trials), 'seed': int(seed)}


def _features(record):
    graph, simple = _graph(record)
    n = len(graph)
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    adjacency = nx.to_numpy_array(graph, dtype=float)
    inverse = np.zeros(n, dtype=float)
    np.divide(1, np.sqrt(degrees), out=inverse, where=degrees > 0)
    laplacian = np.diag((degrees > 0).astype(float)) - inverse[:, None] * adjacency * inverse[None, :]
    eigenvalues = np.clip(np.linalg.eigvalsh(laplacian), 0, 2)
    rounded_eigenvalues = np.round(eigenvalues, 12)
    fingerprint = graph.copy()
    nx.set_node_attributes(fingerprint, {node: str(degree) for node, degree in graph.degree()}, 'degree')
    return {'graph': graph, 'simple': simple, 'n': n, 'edges': graph.number_of_edges(),
            'degree': _histogram(degrees / max(n - 1, 1)),
            'clustering': _histogram(list(nx.clustering(graph).values())),
            'spectral': _histogram(rounded_eigenvalues, upper=2),
            'max_degree': int(degrees.max()) if n else 0,
            'connected': n > 0 and nx.is_connected(graph),
            'planar': nx.check_planarity(graph)[0],
            'lambda2': algebraic_connectivity(graph),
            'triangles': sum(nx.triangles(graph).values()) / 3,
            'wl': (n, nx.weisfeiler_lehman_graph_hash(fingerprint, node_attr='degree', iterations=3))}


def _kernel_mean(left, right, bandwidth):
    total = 0.0
    for start in range(0, len(left), 256):
        distance = cdist(left[start:start + 256], right, metric='sqeuclidean')
        total += np.exp(-distance / (2 * bandwidth ** 2)).sum()
    return total / (len(left) * len(right))


def _mmd2(left, right, bandwidth):
    return float(max(0.0, _kernel_mean(left, left, bandwidth)
                     + _kernel_mean(right, right, bandwidth) - 2 * _kernel_mean(left, right, bandwidth)))


def _bound(value, record, index):
    if value is None:
        return None
    if callable(value):
        value = value(record)
    elif isinstance(value, Mapping):
        value = value[record['n']] if record['n'] in value else value[str(record['n'])]
    elif not np.isscalar(value):
        value = value[index]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('constraint bounds must be nonnegative integers')
    return int(value)


def _summary(samples, references, training, budget, max_degree):
    sample_features = [_features(record) for record in samples]
    reference_features = [_features(record) for record in references]
    training_features = [_features(record) for record in training]
    all_planar = all(item['planar'] for item in reference_features)
    budget_ok, exact_budget, degree_ok, range_ok = [], [], [], []
    for index, (record, item) in enumerate(zip(samples, sample_features)):
        limit, cap = _bound(budget, record, index), _bound(max_degree, record, index)
        budget_ok.append(limit is None or item['edges'] <= limit)
        exact_budget.append(limit is None or item['edges'] == limit)
        degree_ok.append(cap is None or item['max_degree'] <= cap)
        if record.get('positions') is not None and record.get('radius') is not None:
            positions = np.asarray(record['positions'], dtype=float)
            if positions.shape != (item['n'], 2) or not np.isfinite(positions).all():
                raise ValueError('positions must be finite with shape [n, 2]')
            radius = float(record['radius'])
            if not math.isfinite(radius) or radius < 0:
                raise ValueError('radius must be finite and nonnegative')
            range_ok.append(all(np.linalg.norm(positions[u] - positions[v]) <= radius + 1e-12
                                for u, v in item['graph'].edges()))
        else:
            range_ok.append(True)
    count = len(samples)
    result = {'sample_count': count, 'reference_count': len(references), 'training_count': len(training),
              'simple_rate': float(np.mean([item['simple'] for item in sample_features])),
              'connected_rate': float(np.mean([item['connected'] for item in sample_features])),
              'budget_valid_rate': float(np.mean(budget_ok)), 'budget_violation_rate': float(1 - np.mean(budget_ok)),
              'exact_budget_rate': float(np.mean(exact_budget)),
              'degree_valid_rate': float(np.mean(degree_ok)), 'degree_violation_rate': float(1 - np.mean(degree_ok)),
              'range_valid_rate': float(np.mean(range_ok)),
              'planar_rate': float(np.mean([item['planar'] for item in sample_features])),
              'reference_is_planar': bool(all_planar),
              'valid_rate': float(np.mean([item['simple'] and item['connected'] and b and d and r
                                         for item, b, d, r in zip(sample_features, budget_ok, degree_ok, range_ok)])),
              'planarity_constraint_valid_rate': float(np.mean([item['planar'] for item in sample_features])) if all_planar else None,
              'wl_unique_fraction_proxy': len({item['wl'] for item in sample_features}) / count,
              'wl_novel_fraction_proxy': float(np.mean([item['wl'] not in {train['wl'] for train in training_features}
                                                       for item in sample_features])),
              'lambda2_mean': float(np.mean([item['lambda2'] for item in sample_features])),
              'lambda2_reference_mean': float(np.mean([item['lambda2'] for item in reference_features])),
              'nodes_wasserstein': float(wasserstein_distance([item['n'] for item in sample_features], [item['n'] for item in reference_features])),
              'edges_wasserstein': float(wasserstein_distance([item['edges'] for item in sample_features], [item['edges'] for item in reference_features])),
              'triangles_wasserstein': float(wasserstein_distance([item['triangles'] for item in sample_features], [item['triangles'] for item in reference_features]))}
    for name, upper in [('degree', 1), ('clustering', 1), ('spectral', 2)]:
        left = np.stack([item[name] for item in sample_features])
        right = np.stack([item[name] for item in reference_features])
        train = np.stack([item[name] for item in training_features])
        distances = pdist(train[:512])
        positive = distances[distances > 1e-12]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
        result[f'{name}_bandwidth'] = bandwidth
        result[f'{name}_mmd2'] = _mmd2(left, right, bandwidth)
        result[f'{name}_hist_wasserstein'] = float(np.abs(np.cumsum(left.mean(0) - right.mean(0))).sum() * upper / 32)
        result[f'{name}_hist_total_variation'] = float(np.abs(left.mean(0) - right.mean(0)).sum() / 2)
    resilience = [failure_resilience(item['graph'], seed=2718 + index) for index, item in enumerate(sample_features)]
    result['failure_connected_rate'] = float(np.mean([item['connected_rate'] for item in resilience]))
    result['failure_largest_component_fraction'] = float(np.mean([item['largest_component_fraction'] for item in resilience]))
    result['failure_probability'] = 0.1
    result['failure_trials_per_graph'] = 20
    result['failure_seed_base'] = 2718
    return result


def evaluate_samples(samples, reference, train, budget, max_degree=None):
    samples, reference, train = list(samples), list(reference), list(train)
    if not samples or not reference or not train:
        raise ValueError('samples, reference and train must all be nonempty')
    result = _summary(samples, reference, train, budget, max_degree)
    families = sorted({str(record.get('family', 0)) for record in samples})
    if len(families) > 1:
        result['by_family'] = {}
        for family in families:
            chosen = [i for i, record in enumerate(samples) if str(record.get('family', 0)) == family]
            group_ref = [record for record in reference if str(record.get('family', 0)) == family]
            group_train = [record for record in train if str(record.get('family', 0)) == family]
            if not group_ref or not group_train:
                raise ValueError(f'missing reference or training samples for family {family}')
            group_budget = [_bound(budget, samples[i], i) for i in chosen] if budget is not None else None
            group_degree = [_bound(max_degree, samples[i], i) for i in chosen] if max_degree is not None else None
            result['by_family'][family] = _summary([samples[i] for i in chosen], group_ref, group_train, group_budget, group_degree)
    result['metric_scope'] = 'All supplied samples, including invalid samples; dense eigensolvers and NetworkX run outside sampling timers.'
    result['bandwidth_scope'] = 'Median positive training histogram distance, first 512 supplied training graphs; never fit on reference or samples.'
    result['validity_scope'] = 'Simple, connected, edge count <= budget, optional degree and range constraints; planarity reported separately.'
    result['wl_scope'] = 'Weisfeiler-Lehman hashes are diversity and novelty proxies, not exact isomorphism tests.'
    return result


def dynamic_metrics(previous, current, positions, radius, budget, max_degree=None):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise ValueError('positions must have shape [n, 2] and finite values')
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('radius must be finite and nonnegative')
    n = len(positions)
    previous_record = previous if isinstance(previous, Mapping) else {'n': n, 'edges': previous}
    current_record = current if isinstance(current, Mapping) else {'n': n, 'edges': current}
    old_graph, _ = _graph(previous_record)
    graph, simple = _graph(current_record)
    if len(old_graph) != n or len(graph) != n:
        raise ValueError('both graphs and positions must contain the same vertices')
    old_edges = {tuple(sorted(edge)) for edge in old_graph.edges()}
    new_edges = {tuple(sorted(edge)) for edge in graph.edges()}
    violations = sum(np.linalg.norm(positions[u] - positions[v]) > radius + 1e-12 for u, v in new_edges)
    degree = max(dict(graph.degree()).values(), default=0)
    connected = n > 0 and nx.is_connected(graph)
    return {'churn_edges': len(old_edges ^ new_edges),
            'churn_fraction': len(old_edges ^ new_edges) / max(len(old_edges | new_edges), 1),
            'added_edges': len(new_edges - old_edges), 'removed_edges': len(old_edges - new_edges),
            'range_violations': int(violations), 'range_violation_rate': violations / max(len(new_edges), 1),
            'connected': bool(connected), 'edge_count': len(new_edges), 'max_degree': degree,
            'valid': bool(simple and connected and len(new_edges) <= budget and violations == 0
                          and (max_degree is None or degree <= max_degree)),
            'lambda2': algebraic_connectivity(graph), 'failure_resilience': failure_resilience(graph, seed=2718)}


def _rng_restorer(reset, torch):
    if reset is not None:
        return reset
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state() if torch is not None else None
    cuda_state = torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_available() else None
    def restore():
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return restore


def profile_operation(operation, repetitions=7, warmup=2, device='cpu', *, reset=None, return_result=False, poll_seconds=0.001):
    if type(repetitions) is not int or type(warmup) is not int or repetitions < 1 or warmup < 0 or not np.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError('invalid profiling counts or polling period')
    if str(device) not in {'cpu', 'mps'} and not re.fullmatch(r'cuda(?::\d+)?', str(device)):
        raise ValueError('device must be cpu, mps or cuda[:index]')
    torch = sys.modules.get('torch')
    cuda = str(device).startswith('cuda')
    if cuda and (torch is None or not torch.cuda.is_available()):
        raise ValueError('CUDA was requested but is unavailable')
    if str(device).startswith('mps'):
        if torch is None or not torch.backends.mps.is_available():
            raise ValueError('MPS was requested but is unavailable')
        synchronize = torch.mps.synchronize
    elif cuda:
        synchronize = lambda: torch.cuda.synchronize(device)
    else:
        synchronize = lambda: None
    restore = _rng_restorer(reset, torch)
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    for _ in range(warmup):
        restore()
        discarded = operation()
        synchronize()
        del discarded
    times = []
    for _ in range(repetitions):
        restore()
        synchronize()
        started = time.perf_counter_ns()
        discarded = operation()
        synchronize()
        times.append((time.perf_counter_ns() - started) / 1e6)
        del discarded
    synchronize()
    restore()
    baseline = process.memory_info().rss
    peaks = [baseline]
    stopped = threading.Event()
    def poll():
        while not stopped.wait(poll_seconds):
            peaks.append(process.memory_info().rss)
    worker = threading.Thread(target=poll, daemon=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    worker.start()
    try:
        payload = operation()
        synchronize()
        peaks.append(process.memory_info().rss)
    finally:
        stopped.set()
        worker.join()
    try:
        import resource
        highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        highwater = int(highwater if sys.platform == 'darwin' else highwater * 1024)
    except ImportError:
        highwater = None
    result = {'latency_ms_median': float(np.median(times)),
              'latency_ms_p25': float(np.percentile(times, 25)), 'latency_ms_p75': float(np.percentile(times, 75)),
              'latency_ms_samples': times, 'repetitions': repetitions, 'warmup': warmup,
              'rss_before_warmup_bytes': initial_rss, 'rss_baseline_bytes': baseline,
              'rss_sampled_peak_bytes': int(max(peaks)), 'rss_peak_delta_bytes': int(max(peaks) - baseline),
              'rss_poll_seconds': poll_seconds, 'rss_observation_count': len(peaks),
              'process_lifetime_maxrss_bytes': highwater,
              'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)) if cuda else None,
              'cuda_peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)) if cuda else None,
              'device': str(device), 'platform': platform.platform(), 'machine': platform.machine(),
              'cpu_count_logical': psutil.cpu_count(), 'python': platform.python_version(),
              'torch_version': torch.__version__ if torch is not None else None,
              'torch_threads': torch.get_num_threads() if torch is not None else None,
              'measurement_scope': 'Latency and RSS use separate passes of the same operation; no metric calculation included unless supplied by caller.',
              'rss_scope': 'Sampled process RSS after warm-up. Short peaks may be missed; allocator caches and other process allocations remain included. Delta is not total model memory.',
              'maxrss_scope': 'Process lifetime high-water mark, including imports, warm-up and all earlier operations; not operation-only peak.',
              'rng_scope': 'Caller reset before each pass.' if reset else 'Global Python, NumPy and loaded PyTorch RNG restored; private generators require caller reset.'}
    if return_result:
        result['result'] = payload
    return result


def fit_baseline(name, training, family, n):
    chosen = [record for record in (training or []) if record.get('family', family) == family]
    result = {'name': name, 'family': family, 'n': n}
    if name == 'degree_prior':
        if not chosen:
            raise ValueError('degree_prior needs training graphs for this family')
        profiles = []
        target = (np.arange(n) + 0.5) / n
        for record in chosen:
            graph, simple = _graph(record)
            if not simple or len(graph) < 1:
                raise ValueError('baseline training graphs must be nonempty and simple')
            degree = np.sort([d for _, d in graph.degree()]) / max(len(graph) - 1, 1)
            profiles.append(np.interp(target, (np.arange(len(graph)) + 0.5) / len(graph), degree))
        result['degree_profile'] = np.mean(profiles, axis=0).tolist()
    elif name == 'fitted_sbm':
        if not chosen:
            raise ValueError('fitted_sbm needs training graphs for this family')
        within_edges, within_pairs, between_edges, between_pairs, block_counts = 0, 0, 0, 0, []
        for record in chosen:
            graph, simple = _graph(record)
            if not simple or len(graph) < 1:
                raise ValueError('baseline training graphs must be nonempty and simple')
            blocks = list(nx.community.greedy_modularity_communities(graph)) if graph.number_of_edges() else [{node} for node in graph]
            membership = {node: index for index, block in enumerate(blocks) for node in block}
            total_pairs = len(graph) * (len(graph) - 1) // 2
            pairs = sum(len(block) * (len(block) - 1) // 2 for block in blocks)
            edges = sum(membership[u] == membership[v] for u, v in graph.edges())
            within_edges += edges
            within_pairs += pairs
            between_edges += graph.number_of_edges() - edges
            between_pairs += total_pairs - pairs
            block_counts.append(len(blocks))
        result['blocks'] = max(1, min(n, int(round(np.mean(block_counts)))))
        result['p_in'] = (within_edges + 1) / (within_pairs + 2)
        result['p_out'] = (between_edges + 1) / (between_pairs + 2)
    elif name != 'random':
        raise ValueError('baseline must be random, degree_prior or fitted_sbm')
    return result


def generate_baseline(name, n, budget, seed, family, max_degree=None, training=None, positions=None, radius=None):
    from .constrained import projection, radius_edges
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer')
    rng = np.random.default_rng(seed)
    if radius is not None:
        if positions is None or len(positions) != n:
            raise ValueError('radius needs positions for all vertices')
        candidates = radius_edges(positions, radius)
    else:
        candidates = np.column_stack(np.triu_indices(n, 1))
    rows, cols = candidates.T
    fitted = training if isinstance(training, Mapping) else fit_baseline(name, training, family, n)
    if fitted.get('name') != name or fitted.get('family') != family or fitted.get('n') != n:
        raise ValueError('fitted baseline metadata does not match the sampling request')
    if name == 'random':
        scores = rng.gumbel(size=len(candidates))
    elif name == 'degree_prior':
        expected = np.asarray(fitted['degree_profile'], dtype=float).copy()
        rng.shuffle(expected)
        probability = np.clip(expected[rows] * expected[cols] / max(expected.mean(), 1e-8), 1e-5, 1 - 1e-5)
        scores = np.log(probability / (1 - probability)) + rng.gumbel(size=len(candidates))
    elif name == 'fitted_sbm':
        blocks = fitted['blocks']
        membership = np.arange(n) % blocks
        rng.shuffle(membership)
        probability = np.clip(np.where(membership[rows] == membership[cols], fitted['p_in'], fitted['p_out']), 1e-5, 1 - 1e-5)
        scores = np.log(probability / (1 - probability)) + rng.gumbel(size=len(candidates))
    else:
        raise ValueError('baseline must be random, degree_prior or fitted_sbm')
    return projection(n, candidates, scores, budget, max_degree=max_degree)


def import_external_samples(path, method, expected_counts=None, expected_split_sha256=None):
    path = Path(path)
    document = json.loads(path.read_text())
    metadata = document.get('metadata', {})
    required = {'upstream_url', 'repo_commit', 'config', 'split_sha256', 'seeds', 'timing_scope', 'hardware', 'command'}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f'missing external metadata: {sorted(missing)}')
    if metadata.get('method') != method or method not in {'DiGress', 'SparseDiff', 'SPECTRE', 'GDSS'}:
        raise ValueError('method must match an explicitly identified upstream implementation')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', str(metadata['repo_commit'])):
        raise ValueError('repo_commit must be a full 40-character commit hash')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', str(metadata['split_sha256'])):
        raise ValueError('split_sha256 must be a SHA-256 hash')
    if not isinstance(metadata['config'], Mapping) or not metadata['config']:
        raise ValueError('external config must be a nonempty object')
    if not isinstance(metadata['seeds'], list) or not metadata['seeds'] or any(type(seed) is not int for seed in metadata['seeds']):
        raise ValueError('external seeds must be a nonempty integer list')
    for field in ['upstream_url', 'timing_scope', 'hardware', 'command']:
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f'{field} must be a nonempty string')
    if not metadata['upstream_url'].startswith('https://'):
        raise ValueError('upstream_url must identify an HTTPS source')
    if expected_split_sha256 is not None and metadata['split_sha256'].lower() != expected_split_sha256.lower():
        raise ValueError('external split hash does not match this benchmark')
    records = document.get('samples')
    if not isinstance(records, list) or not records:
        raise ValueError('external samples must be a nonempty list')
    samples = []
    for record in records:
        _graph(record)
        sample = dict(record)
        sample['edges'] = np.asarray(record.get('edges', [])).reshape(-1, 2)
        samples.append(sample)
    if expected_counts is not None:
        if isinstance(expected_counts, Mapping):
            actual = Counter(str(record.get('family', 0)) for record in samples)
            if dict(actual) != {str(key): value for key, value in expected_counts.items()}:
                raise ValueError('external sample counts do not match the required families')
        elif len(samples) != expected_counts:
            raise ValueError('external sample count does not match')
    return {'samples': samples, 'metadata': metadata, 'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'verification_scope': 'Format and declared provenance checked; upstream execution is not independently certified.'}
