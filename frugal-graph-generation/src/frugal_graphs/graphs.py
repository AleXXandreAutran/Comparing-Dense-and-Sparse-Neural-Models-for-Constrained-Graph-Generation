from __future__ import annotations

from functools import lru_cache
from numbers import Integral, Real
import numpy as np

@lru_cache(maxsize=32)
def _edge_indices(n):
    return np.triu_indices(n, k=1)

def _validate(scores, n, m, *, connected):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral):
        raise TypeError('n must be an integer')
    if n < 1:
        raise ValueError('n must be at least 1')
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise TypeError('m must be a scalar integer')
    edge_count = n * (n - 1) // 2
    minimum = n - 1 if connected else 0
    if not minimum <= m <= edge_count:
        raise ValueError(f'm must satisfy {minimum} <= m <= {edge_count}')
    values = np.asarray(scores)
    if values.ndim not in (1, 2):
        raise ValueError('scores must have shape [E] or [B, E]')
    if values.shape[-1] != edge_count:
        raise ValueError(f'scores must contain exactly {edge_count} edge scores')
    if values.dtype.kind not in 'fiu':
        raise TypeError('scores must contain real numeric values')
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('scores must all be finite')
    single = values.ndim == 1
    return (values[None, :] if single else values, single)

def _adjacency(selected, n, single):
    result = np.zeros((len(selected), n, n), dtype=np.float32)
    rows, cols = _edge_indices(n)
    result[:, rows, cols] = selected
    result[:, cols, rows] = selected
    return result[0] if single else result

def project_topk(scores, n, m):
    values, single = _validate(scores, n, m, connected=False)
    order = np.argsort(-values, axis=1, kind='stable')
    selected = np.zeros(values.shape, dtype=np.float32)
    np.put_along_axis(selected, order[:, :m], 1.0, axis=1)
    return _adjacency(selected, n, single)

def project_connected(scores, n, m):
    values, single = _validate(scores, n, m, connected=True)
    order = np.argsort(-values, axis=1, kind='stable')
    selected = np.zeros(values.shape, dtype=np.bool_)
    rows, cols = _edge_indices(n)
    for batch_index, ranked in enumerate(order):
        parent = list(range(n))
        size = [1] * n

        def find(vertex):
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex
        tree_edges = 0
        for edge in ranked:
            left, right = (find(int(rows[edge])), find(int(cols[edge])))
            if left == right:
                continue
            if size[left] < size[right]:
                left, right = (right, left)
            parent[right] = left
            size[left] += size[right]
            selected[batch_index, edge] = True
            tree_edges += 1
            if tree_edges == n - 1:
                break
        remaining = ranked[~selected[batch_index, ranked]]
        selected[batch_index, remaining[:m - (n - 1)]] = True
    return _adjacency(selected, n, single)

def perturb_scores(logits, rng, temperature=1.0):
    values = np.asarray(logits)
    if values.dtype.kind not in 'fiu':
        raise TypeError('logits must contain real numeric values')
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('logits must all be finite')
    if isinstance(temperature, (bool, np.bool_)) or not isinstance(temperature, Real):
        raise TypeError('temperature must be a real scalar')
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if temperature == 0:
        return values.copy()
    return values + float(temperature) * rng.gumbel(size=values.shape)
from collections import defaultdict, deque
from pathlib import Path
import networkx as nx
FAMILY_NAMES = ('community', 'geometric')

def canonical_order(adj: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adj)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError('adj must be a square adjacency matrix')
    n = len(adjacency)
    degrees = (adjacency > 0.5).sum(axis=1)
    remaining = set(range(n))
    ordering: list[int] = []
    while remaining:
        root = min(remaining, key=lambda node: (-degrees[node], node))
        queue = deque([root])
        remaining.remove(root)
        while queue:
            node = queue.popleft()
            ordering.append(node)
            neighbors = [int(neighbor) for neighbor in np.flatnonzero(adjacency[node] > 0.5) if neighbor in remaining]
            neighbors.sort(key=lambda neighbor: (-degrees[neighbor], neighbor))
            for neighbor in neighbors:
                remaining.remove(neighbor)
                queue.append(neighbor)
    return np.asarray(adjacency[np.ix_(ordering, ordering)], dtype=np.float32)

def wl_graph_hash(graph: nx.Graph) -> str:
    attributed = graph.copy()
    nx.set_node_attributes(attributed, {node: str(degree) for node, degree in graph.degree}, 'degree')
    return nx.weisfeiler_lehman_graph_hash(attributed, node_attr='degree', iterations=3, digest_size=16)

def _latent_scores(n: int, family: int, rng: np.random.Generator) -> np.ndarray:
    rows, columns = np.triu_indices(n, 1)
    if family == 0:
        blocks = int(rng.integers(2, min(3, n) + 1))
        memberships = np.arange(n) % blocks
        rng.shuffle(memberships)
        within = memberships[rows] == memberships[columns]
        separation = rng.uniform(0.85, 1.15)
        noise = rng.uniform(0.3, 0.45)
        return separation * within + rng.normal(0.0, noise, len(rows))
    if family == 1:
        positions = rng.uniform(0.0, 1.0, size=(n, 2))
        positions[:, 0] *= rng.uniform(0.8, 1.25)
        distances = np.linalg.norm(positions[rows] - positions[columns], axis=1)
        return -distances + rng.normal(0.0, rng.uniform(0.02, 0.06), len(rows))
    raise ValueError(f'unknown family: {family}')

def generate_dataset(n: int=20, m: int=40, train_per_family: int=600, val_per_family: int=150, test_per_family: int=200, reference_per_family: int=200, seed: int=20260920) -> dict[str, dict[str, np.ndarray]]:
    if n < 2 or not n - 1 <= m <= n * (n - 1) // 2:
        raise ValueError('require n >= 2 and n-1 <= m <= n(n-1)/2')
    sizes = (train_per_family, val_per_family, test_per_family, reference_per_family)
    if any((not isinstance(size, (int, np.integer)) or size < 0 for size in sizes)):
        raise ValueError('split sizes must be nonnegative integers')
    splits = ('train', 'val', 'test', 'reference')
    streams = np.random.SeedSequence(seed).spawn(len(splits) * len(FAMILY_NAMES))
    seen: dict[str, list[nx.Graph]] = defaultdict(list)
    dataset: dict[str, dict[str, np.ndarray]] = {}
    for split_index, (split, size) in enumerate(zip(splits, sizes, strict=True)):
        samples: list[np.ndarray] = []
        labels: list[int] = []
        for family in range(len(FAMILY_NAMES)):
            rng = np.random.default_rng(streams[split_index * len(FAMILY_NAMES) + family])
            accepted = 0
            attempts = 0
            while accepted < size:
                attempts += 1
                if attempts > max(1000, size * 100):
                    raise RuntimeError('Too few distinct graphs for requested split sizes; increase n or reduce samples/change the edge budget.')
                raw = project_connected(_latent_scores(n, family, rng), n, m)
                adjacency = canonical_order(raw)
                graph = nx.from_numpy_array(adjacency)
                fingerprint = wl_graph_hash(graph)
                if any((nx.is_isomorphic(graph, previous) for previous in seen[fingerprint])):
                    continue
                seen[fingerprint].append(graph)
                samples.append(adjacency)
                labels.append(family)
                accepted += 1
        dataset[split] = {'adj': np.stack(samples).astype(np.float32) if samples else np.empty((0, n, n), dtype=np.float32), 'family': np.asarray(labels, dtype=np.int64)}
    return dataset

def save_dataset(dataset: dict[str, dict[str, np.ndarray]], path: str | Path) -> None:
    arrays = {f'{split}_{key}': value for split, data in dataset.items() for key, value in data.items()}
    np.savez_compressed(path, **arrays)

def load_dataset(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        return {split: {key: archive[f'{split}_{key}'] for key in ('adj', 'family')} for split in ('train', 'val', 'test', 'reference')}
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

def _as_batch(adjacency: np.ndarray) -> np.ndarray:
    array = np.asarray(adjacency, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != array.shape[2] or len(array) == 0:
        raise ValueError('expected a nonempty [graphs,nodes,nodes] adjacency batch')
    if not np.isfinite(array).all():
        raise ValueError('adjacency entries must be finite')
    return array

def graph_statistics(adjacency: np.ndarray) -> dict[str, np.ndarray]:
    raw = _as_batch(adjacency)
    batch, n, _ = raw.shape
    binary_entries = np.isclose(raw, 0.0, atol=1e-06) | np.isclose(raw, 1.0, atol=1e-06)
    simple = binary_entries.all(axis=(1, 2)) & np.isclose(raw, raw.transpose(0, 2, 1), atol=1e-06).all(axis=(1, 2)) & np.isclose(np.diagonal(raw, axis1=1, axis2=2), 0.0, atol=1e-06).all(axis=1)
    upper = np.triu(raw > 0.5, k=1)
    arrays = (upper | upper.transpose(0, 2, 1)).astype(np.float64)
    degree = arrays.sum(axis=2)
    squared = arrays @ arrays
    triangle_incidence = (squared * arrays).sum(axis=2)
    denominator = degree * (degree - 1)
    clustering = np.divide(triangle_incidence, denominator, out=np.zeros_like(degree), where=denominator > 0)
    degree_histogram = np.stack([np.bincount(row.astype(int), minlength=n) / n for row in degree])
    clustering_histogram = np.stack([np.bincount(np.minimum((row * 10).astype(int), 9), minlength=10) / n for row in clustering])
    normalizer = np.sqrt(degree[:, :, None] * degree[:, None, :])
    laplacian = -np.divide(arrays, normalizer, out=np.zeros_like(arrays), where=normalizer > 0)
    diagonal = np.arange(n)
    laplacian[:, diagonal, diagonal] = degree > 0
    eigenvalues = np.clip(np.linalg.eigvalsh(laplacian), 0, 2)
    spectral_histogram = np.stack([np.bincount(np.minimum(np.floor(np.round(row * 5, 10)).astype(int), 9), minlength=10) / row.size for row in eigenvalues])
    distances = np.where(arrays > 0, 1.0, np.inf)
    distances[:, diagonal, diagonal] = 0.0
    for node in range(n):
        distances = np.minimum(distances, distances[:, :, node, None] + distances[:, None, node, :])
    inverse_distances = np.divide(1.0, distances, out=np.zeros_like(distances), where=distances > 0)
    return {'degree': degree_histogram, 'clustering': clustering_histogram, 'spectral': spectral_histogram, 'triangles': triangle_incidence.sum(axis=1) / 6, 'efficiency': inverse_distances.sum(axis=(1, 2)) / max(n * (n - 1), 1), 'edges': arrays.sum(axis=(1, 2)) / 2, 'connected': np.isfinite(distances).all(axis=(1, 2)), 'simple': simple}

def fit_metric_reference(train_adj: np.ndarray) -> dict[str, float | int]:
    training = _as_batch(train_adj)
    indices = np.linspace(0, len(training) - 1, min(len(training), 600), dtype=int)
    features = graph_statistics(training[indices])
    reference: dict[str, float | int] = {'n_nodes': int(training.shape[1])}
    for key in ('degree', 'clustering', 'spectral'):
        distances = pdist(features[key], metric='euclidean')
        positive = distances[distances > 1e-12]
        reference[f'{key}_bandwidth'] = float(np.median(positive)) if len(positive) else 1.0
    return reference

def squared_mmd(left: np.ndarray, right: np.ndarray, bandwidth: float) -> float:
    if bandwidth <= 0:
        raise ValueError('bandwidth must be positive')
    scale = 2 * bandwidth ** 2
    kernel_xx = np.exp(-cdist(left, left, 'sqeuclidean') / scale)
    kernel_yy = np.exp(-cdist(right, right, 'sqeuclidean') / scale)
    kernel_xy = np.exp(-cdist(left, right, 'sqeuclidean') / scale)
    return float(max(0.0, kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()))

def evaluate_graphs(generated_adj: np.ndarray, test_adj: np.ndarray, train_adj: np.ndarray, metric_reference: dict[str, float | int], m: int) -> dict[str, float]:
    generated = _as_batch(generated_adj)
    held_out = _as_batch(test_adj)
    training = _as_batch(train_adj)
    if any((array.shape[1] != metric_reference['n_nodes'] for array in (generated, held_out, training))):
        raise ValueError('all batches must have the reference node count')
    samples = graph_statistics(generated)
    target = graph_statistics(held_out)
    budget = samples['edges'] == m
    result = {'connected_rate': float(samples['connected'].mean()), 'exact_budget_rate': float(budget.mean()), 'simple_rate': float(samples['simple'].mean()), 'valid_rate': float((samples['connected'] & samples['simple'] & budget).mean())}
    for feature in ('degree', 'clustering', 'spectral'):
        result[f'{feature}_mmd2'] = squared_mmd(samples[feature], target[feature], float(metric_reference[f'{feature}_bandwidth']))
    for feature in ('triangles', 'efficiency'):
        result[f'{feature}_mean_abs_error'] = float(abs(samples[feature].mean() - target[feature].mean()))
        result[f'{feature}_wasserstein'] = float(wasserstein_distance(samples[feature], target[feature]))
        result[f'generated_mean_{feature}'] = float(samples[feature].mean())
        result[f'test_mean_{feature}'] = float(target[feature].mean())
    generated_hashes = [wl_graph_hash(nx.from_numpy_array(array > 0.5)) for array in generated]
    training_hashes = {wl_graph_hash(nx.from_numpy_array(array > 0.5)) for array in training}
    result['wl_unique_proxy_rate'] = float(len(set(generated_hashes)) / len(generated_hashes))
    result['wl_novel_proxy_rate'] = float(np.mean([fingerprint not in training_hashes for fingerprint in generated_hashes]))
    return result
