from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np
from scipy.spatial import cKDTree


class InfeasibleGraphError(ValueError):
    status = "infeasible"

    def __init__(self, reason):
        self.reason = str(reason)
        super().__init__(self.reason)


class FeasibilitySearchError(RuntimeError):
    status = "search_failed"

    def __init__(self, reason):
        self.reason = str(reason)
        super().__init__(self.reason)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _limits(n, budget, max_degree):
    n = _integer(n, "n", 1)
    budget = _integer(budget, "budget")
    if max_degree is not None:
        max_degree = _integer(max_degree, "max_degree")
    if budget < n - 1:
        raise InfeasibleGraphError("A connected graph requires at least n-1 edges")
    if max_degree is not None and n * max_degree < 2 * (n - 1):
        raise InfeasibleGraphError("The degree cap cannot support a connected graph")
    return n, budget, max_degree


def _raw_edges(edges):
    array = np.asarray(edges)
    if array.ndim == 1 and array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    if array.dtype.kind not in "iuf":
        raise ValueError("edge indices must be integers")
    if not np.isfinite(array).all():
        raise ValueError("edge indices must be finite integers")
    if array.size and (np.any(array < -(2**63)) or np.any(array >= 2**63)):
        raise ValueError("edge indices must fit int64")
    converted = array.astype(np.int64)
    if not np.array_equal(array, converted):
        raise ValueError("edge indices must be integers")
    return converted


def canonical_edges(edges, n):
    n = _integer(n, "n", 1)
    array = _raw_edges(edges)
    if array.size and (np.any(array < 0) or np.any(array >= n)):
        raise ValueError("edge index outside [0, n)")
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError("self-loops are not allowed")
    return np.unique(np.sort(array, axis=1), axis=0)


def _edge_array(edges):
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


class _Components:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, u, v):
        u, v = self.find(u), self.find(v)
        if u == v:
            return False
        if self.size[u] < self.size[v]:
            u, v = v, u
        self.parent[v] = u
        self.size[u] += self.size[v]
        self.count -= 1
        return True


def _component_count(n, edges):
    components = _Components(n)
    for u, v in edges:
        components.join(int(u), int(v))
    return components.count


def _degrees(n, edges):
    return np.bincount(np.asarray(edges, dtype=np.int64).reshape(-1), minlength=n)


def _tree(n, allowed, max_degree, rng, preference=None, attempts=64):
    if n == 1:
        return np.empty((0, 2), dtype=np.int64)
    if _component_count(n, allowed) != 1:
        raise InfeasibleGraphError("The allowed-edge graph is disconnected")
    preference = np.zeros(len(allowed)) if preference is None else np.asarray(preference)
    for attempt in range(attempts if max_degree is not None else 1):
        degrees = np.zeros(n, dtype=np.int64)
        components = _Components(n)
        result = []
        jitter = rng.random(len(allowed))
        order = np.lexsort((jitter, -preference)) if attempt < 8 else np.argsort(jitter)
        for index in order:
            u, v = map(int, allowed[index])
            if max_degree is not None and (degrees[u] >= max_degree or degrees[v] >= max_degree):
                continue
            if components.join(u, v):
                result.append((u, v))
                degrees[u] += 1
                degrees[v] += 1
                if components.count == 1:
                    return _edge_array(result)
    raise FeasibilitySearchError(
        "No degree-bounded spanning tree found in 64 greedy attempts; feasibility is undecided"
    )


def feasible_seed(n, budget, max_degree=None, allowed=None, seed=0):
    n, budget, max_degree = _limits(n, budget, max_degree)
    rng = np.random.default_rng(seed)
    if allowed is None:
        order = rng.permutation(n)
        return canonical_edges(np.column_stack((order[:-1], order[1:])), n)
    allowed = canonical_edges(allowed, n)
    return _tree(n, allowed, max_degree, rng)


def radius_edges(positions, radius):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions must have shape (n, 2), with n >= 1")
    if not np.isfinite(positions).all() or not np.isfinite(radius) or radius < 0:
        raise ValueError("positions and radius must be finite, with radius >= 0")
    pairs = cKDTree(positions).query_pairs(float(radius), output_type="ndarray")
    return canonical_edges(pairs, len(positions))


def validate_graph(n, edges, budget, max_degree=None, positions=None, radius=None):
    n = _integer(n, "n", 1)
    budget = _integer(budget, "budget")
    if max_degree is not None:
        max_degree = _integer(max_degree, "max_degree")
    array = _raw_edges(edges)
    in_range = (array >= 0).all(axis=1) & (array < n).all(axis=1)
    self_loops = int(np.count_nonzero(array[:, 0] == array[:, 1]))
    valid_array = array[in_range & (array[:, 0] != array[:, 1])]
    unique = np.unique(np.sort(valid_array, axis=1), axis=0)
    duplicates = len(valid_array) - len(unique)
    degrees = _degrees(n, unique)
    component_count = _component_count(n, unique)
    radius_violations = 0
    if radius is not None:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (n, 2) or not np.isfinite(positions).all():
            raise ValueError("positions must be a finite (n, 2) array")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        distances = np.linalg.norm(positions[unique[:, 0]] - positions[unique[:, 1]], axis=1)
        radius_violations = int(np.count_nonzero(distances > radius))
    degree_excess = 0 if max_degree is None else int(np.maximum(0, degrees - max_degree).sum())
    counts = {
        "self_loops": self_loops,
        "duplicate_edges": int(duplicates),
        "out_of_bounds": int(np.count_nonzero(~in_range)),
        "budget_excess": max(0, len(array) - budget),
        "degree_excess": degree_excess,
        "radius_violations": radius_violations,
    }
    violations = [name for name, count in counts.items() if count]
    if component_count != 1:
        violations.append("disconnected")
    return {
        "valid": not violations,
        "connected": component_count == 1,
        "simple": not (self_loops or duplicates or counts["out_of_bounds"]),
        "budget_ok": len(array) <= budget,
        "degree_ok": degree_excess == 0,
        "radius_ok": radius_violations == 0,
        "num_nodes": n,
        "edge_count": len(array),
        "num_edges": len(array),
        "max_degree": int(degrees.max()),
        "components": component_count,
        "violations": violations,
        **counts,
    }


def candidate_edges(n, current, k, rng, allowed=None):
    n = _integer(n, "n", 1)
    k = _integer(k, "k")
    current = canonical_edges(current, n)
    result = set(map(tuple, current.tolist()))
    if allowed is None:
        take = min(k, n - 1)
        for u in range(n):
            neighbors = rng.choice(n - 1, size=take, replace=False) if take else ()
            for neighbor in neighbors:
                v = int(neighbor) + (int(neighbor) >= u)
                result.add((min(u, v), max(u, v)))
    else:
        allowed = canonical_edges(allowed, n)
        allowed_set = set(map(tuple, allowed.tolist()))
        if not result.issubset(allowed_set):
            raise ValueError("current contains edges outside the allowed set")
        adjacency = [[] for _ in range(n)]
        for u, v in allowed:
            adjacency[u].append(int(v))
            adjacency[v].append(int(u))
        for u, neighbors in enumerate(adjacency):
            take = min(k, len(neighbors))
            indices = rng.choice(len(neighbors), size=take, replace=False) if take else ()
            for index in indices:
                v = neighbors[index]
                result.add((min(u, v), max(u, v)))
    return _edge_array(result)


def _scored_candidates(n, candidates, scores):
    raw = _raw_edges(candidates)
    canonical = canonical_edges(raw, n)
    if len(canonical) != len(raw):
        raise ValueError("candidates must contain unique undirected edges")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(raw),) or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector aligned with candidates")
    weights = {tuple(sorted(map(int, edge))): float(score) for edge, score in zip(raw, scores)}
    return canonical, weights


def _path(adjacency, start, goal):
    parent = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in sorted(adjacency[u]):
            if v in parent:
                continue
            parent[v] = u
            if v == goal:
                result = []
                while parent[v] is not None:
                    p = parent[v]
                    result.append((min(v, p), max(v, p)))
                    v = p
                return result
            queue.append(v)
    raise ValueError("current graph must be connected")


def constrained_update(n, current, candidates, scores, budget, max_degree=None, max_swaps=None):
    n, budget, max_degree = _limits(n, budget, max_degree)
    current = canonical_edges(current, n)
    if not validate_graph(n, current, budget, max_degree)["valid"]:
        raise ValueError("current must satisfy connectivity, budget and degree constraints")
    if max_swaps is not None:
        max_swaps = _integer(max_swaps, "max_swaps")
    _, weights = _scored_candidates(n, candidates, scores)
    active = set(map(tuple, current.tolist()))
    if not active.issubset(weights):
        raise ValueError("candidates must include all current edges")
    adjacency = [set() for _ in range(n)]
    degrees = _degrees(n, current)
    for u, v in active:
        adjacency[u].add(v)
        adjacency[v].add(u)
    initial_score = sum(weights[edge] for edge in active)
    additions, swaps, rejected = 0, 0, 0
    for edge in sorted(weights, key=lambda edge: (-weights[edge], edge)):
        if edge in active:
            continue
        u, v = edge
        if len(active) < budget and (max_degree is None or max(degrees[u], degrees[v]) < max_degree):
            active.add(edge)
            adjacency[u].add(v)
            adjacency[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            additions += 1
            continue
        if max_swaps is not None and swaps >= max_swaps:
            rejected += 1
            continue
        removable = []
        for old in _path(adjacency, u, v):
            a, b = old
            if max_degree is not None:
                if degrees[u] + 1 - int(u in old) > max_degree:
                    continue
                if degrees[v] + 1 - int(v in old) > max_degree:
                    continue
            if weights[edge] > weights[old]:
                removable.append(old)
        if not removable:
            rejected += 1
            continue
        old = min(removable, key=lambda edge: (weights[edge], edge))
        a, b = old
        active.remove(old)
        active.add(edge)
        adjacency[a].remove(b)
        adjacency[b].remove(a)
        adjacency[u].add(v)
        adjacency[v].add(u)
        degrees[a] -= 1
        degrees[b] -= 1
        degrees[u] += 1
        degrees[v] += 1
        swaps += 1
    result = _edge_array(active)
    return result, {
        "status": "feasible",
        "accepted_additions": additions,
        "accepted_swaps": swaps,
        "accepted": additions + swaps,
        "rejected": rejected,
        "initial_edges": len(current),
        "final_edges": len(result),
        "score_gain": sum(weights[edge] for edge in active) - initial_score,
        "atomic_updates": additions + swaps,
    }


def projection(n, candidates, scores, budget, max_degree=None, initial=None):
    n, budget, max_degree = _limits(n, budget, max_degree)
    candidates, weights = _scored_candidates(n, candidates, scores)
    ordered_scores = np.asarray([weights[tuple(edge)] for edge in candidates])
    if initial is None:
        initial = _tree(n, candidates, max_degree, np.random.default_rng(0), ordered_scores)
    result, _ = constrained_update(n, initial, candidates, ordered_scores, budget, max_degree)
    return result


def dynamic_repair(n, previous, positions, radius, budget, max_degree=None, seed=0):
    n, budget, max_degree = _limits(n, budget, max_degree)
    previous = canonical_edges(previous, n)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (n, 2):
        raise ValueError("positions must have shape (n, 2)")
    allowed = radius_edges(positions, radius)
    if _component_count(n, allowed) != 1:
        raise InfeasibleGraphError("The communication-radius graph is disconnected")
    previous_set = set(map(tuple, previous.tolist()))
    allowed_set = set(map(tuple, allowed.tolist()))
    retained = previous_set & allowed_set
    active = set(retained)
    degrees = _degrees(n, _edge_array(active))
    components = _Components(n)
    for u, v in active:
        components.join(u, v)
    rng = np.random.default_rng(seed)
    if len(active) <= budget and (max_degree is None or np.max(degrees) <= max_degree):
        order = rng.permutation(len(allowed))
        order = sorted(order, key=lambda i: int(degrees[allowed[i, 0]] + degrees[allowed[i, 1]]))
        for index in order:
            if components.count == 1:
                break
            if len(active) == budget:
                break
            u, v = map(int, allowed[index])
            if max_degree is not None and max(degrees[u], degrees[v]) >= max_degree:
                continue
            if components.join(u, v):
                active.add((u, v))
                degrees[u] += 1
                degrees[v] += 1
    if components.count != 1 or len(active) > budget or (max_degree is not None and np.max(degrees) > max_degree):
        preference = np.asarray([float(tuple(edge) in retained) for edge in allowed])
        initial = _tree(n, allowed, max_degree, rng, preference)
        active = set(map(tuple, initial.tolist()))
        degrees = _degrees(n, initial)
        for edge in sorted(retained - active):
            u, v = edge
            if len(active) < budget and (max_degree is None or max(degrees[u], degrees[v]) < max_degree):
                active.add(edge)
                degrees[u] += 1
                degrees[v] += 1
    result = _edge_array(active)
    added = len(active - previous_set)
    removed = len(previous_set - active)
    return result, {
        "status": "feasible",
        "previous_edges": len(previous),
        "final_edges": len(result),
        "retained": len(active & previous_set),
        "removed_out_of_range": len(previous_set - allowed_set),
        "removed": removed,
        "added": added,
        "churn": added + removed,
        "churn_rate": (added + removed) / max(1, len(active | previous_set)),
        "allowed_edges": len(allowed),
    }
