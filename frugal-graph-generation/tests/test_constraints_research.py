import itertools

import numpy as np
import pytest

from frugal_graphs.constrained import (
    FeasibilitySearchError,
    InfeasibleGraphError,
    candidate_edges,
    canonical_edges,
    constrained_update,
    dynamic_repair,
    feasible_seed,
    projection,
    radius_edges,
    validate_graph,
)


def assert_feasible(n, edges, budget, degree=None, positions=None, radius=None):
    assert edges.shape == (len(edges), 2)
    assert len(edges) <= budget
    assert len({tuple(edge) for edge in edges}) == len(edges)
    adjacency = [set() for _ in range(n)]
    for u, v in edges:
        assert 0 <= u < v < n
        adjacency[u].add(v)
        adjacency[v].add(u)
        if radius is not None:
            assert np.linalg.norm(positions[u] - positions[v]) <= radius
    reached = {0}
    stack = [0]
    while stack:
        for neighbor in adjacency[stack.pop()] - reached:
            reached.add(neighbor)
            stack.append(neighbor)
    assert len(reached) == n
    if degree is not None:
        assert all(len(neighbors) <= degree for neighbors in adjacency)
    assert validate_graph(n, edges, budget, degree, positions, radius)["valid"]


def test_canonical_edges_are_sorted_and_unique():
    np.testing.assert_array_equal(
        canonical_edges([[3, 1], [1, 3], [0, 2], [2, 0]], 4), [[0, 2], [1, 3]]
    )
    assert canonical_edges([], 1).shape == (0, 2)


@pytest.mark.parametrize("edges", [[[0, 0]], [[0, 4]], [[-1, 2]], [[0.5, 2]], [[np.nan, 2]], [0, 1]])
def test_canonical_rejects_invalid_edges(edges):
    with pytest.raises(ValueError):
        canonical_edges(edges, 4)


@pytest.mark.parametrize("n", [1, 2, 3, 20, 100])
def test_path_seed_respects_minimal_budget_and_degree(n):
    degree = min(2, n - 1)
    edges = feasible_seed(n, n - 1, max_degree=degree, seed=7)
    assert_feasible(n, edges, n - 1, degree)
    np.testing.assert_array_equal(edges, feasible_seed(n, n - 1, degree, seed=7))


@pytest.mark.parametrize("n,budget,degree", [(4, 2, None), (4, 4, 1), (2, 1, 0)])
def test_simple_infeasibility_has_a_certificate(n, budget, degree):
    with pytest.raises(InfeasibleGraphError) as error:
        feasible_seed(n, budget, degree)
    assert error.value.status == "infeasible"


def test_disconnected_allowed_graph_is_proved_infeasible():
    with pytest.raises(InfeasibleGraphError, match="disconnected"):
        feasible_seed(4, 3, allowed=[[0, 1], [2, 3]])


def test_degree_search_failure_is_not_reported_as_proof():
    with pytest.raises(FeasibilitySearchError) as error:
        feasible_seed(5, 4, max_degree=2, allowed=[[0, 1], [0, 2], [0, 3], [0, 4]])
    assert error.value.status == "search_failed"
    assert "undecided" in str(error.value)


def test_allowed_path_is_found_under_degree_two():
    allowed = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
    edges = feasible_seed(6, 5, 2, allowed=allowed)
    assert_feasible(6, edges, 5, 2)
    assert set(map(tuple, edges)) <= set(map(tuple, canonical_edges(allowed, 6)))


@pytest.mark.parametrize("n,k", [(1, 0), (2, 2), (30, 0), (30, 3), (1001, 7)])
def test_sparse_candidate_bound_and_current_retention(n, k):
    current = feasible_seed(n, n - 1)
    candidates = candidate_edges(n, current, k, np.random.default_rng(11))
    assert len(candidates) <= len(current) + k * n
    assert set(map(tuple, current)) <= set(map(tuple, candidates))
    assert np.all(candidates[:, 0] < candidates[:, 1])


def test_complete_candidates_are_available_explicitly():
    current = feasible_seed(9, 8)
    candidates = candidate_edges(9, current, 8, np.random.default_rng(0))
    assert len(candidates) == 36


def test_candidate_builder_never_requests_a_dense_adjacency(monkeypatch):
    n = 5000
    current = feasible_seed(n, n - 1)
    original_zeros = np.zeros
    original_empty = np.empty
    original_ones = np.ones

    def guard(function):
        def wrapped(shape, *args, **kwargs):
            assert not isinstance(shape, tuple) or shape != (n, n)
            return function(shape, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(np, "zeros", guard(original_zeros))
    monkeypatch.setattr(np, "empty", guard(original_empty))
    monkeypatch.setattr(np, "ones", guard(original_ones))
    monkeypatch.setattr(np, "triu_indices", lambda *args, **kwargs: pytest.fail("dense pair enumeration"))
    candidates = candidate_edges(n, current, 2, np.random.default_rng(0))
    assert len(candidates) <= len(current) + 2 * n


def test_candidates_honor_allowed_edges():
    allowed = canonical_edges([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]], 5)
    current = feasible_seed(5, 4, allowed=allowed)
    candidates = candidate_edges(5, current, 1, np.random.default_rng(0), allowed)
    assert set(map(tuple, candidates)) <= set(map(tuple, allowed))
    with pytest.raises(ValueError, match="outside"):
        candidate_edges(5, [[0, 2]], 1, np.random.default_rng(0), allowed)


def test_exchange_preserves_a_bridge_outside_the_created_cycle():
    current = canonical_edges([[0, 1], [1, 2], [2, 3], [3, 4]], 5)
    candidates = canonical_edges([*current.tolist(), [0, 2]], 5)
    weights = {(0, 1): 3, (1, 2): 2, (2, 3): -100, (3, 4): 1, (0, 2): 10}
    scores = [weights[tuple(edge)] for edge in candidates]
    result, info = constrained_update(5, current, candidates, scores, 4)
    assert_feasible(5, result, 4)
    assert (2, 3) in set(map(tuple, result))
    assert (1, 2) not in set(map(tuple, result))
    assert info["accepted_swaps"] == 1
    assert info["score_gain"] == 8


def test_saturated_endpoints_do_not_break_degree_cap():
    current = canonical_edges([[0, 1], [1, 2], [2, 3], [3, 4]], 5)
    candidates = canonical_edges([*current.tolist(), [1, 3]], 5)
    scores = [100 if tuple(edge) == (1, 3) else 0 for edge in candidates]
    result, info = constrained_update(5, current, candidates, scores, 5, max_degree=2)
    np.testing.assert_array_equal(result, current)
    assert info["accepted_swaps"] == 0


def test_degree_cap_is_checked_after_the_atomic_exchange():
    current = canonical_edges([[0, 1], [1, 2], [2, 3], [3, 4]], 5)
    candidates = canonical_edges([*current.tolist(), [0, 3]], 5)
    scores = [10 if tuple(edge) == (0, 3) else 0 for edge in candidates]
    result, info = constrained_update(5, current, candidates, scores, 4, max_degree=2)
    assert_feasible(5, result, 4, 2)
    assert (0, 3) in set(map(tuple, result))
    assert info["accepted_swaps"] == 1


def test_score_alignment_survives_unsorted_edges():
    current = canonical_edges([[0, 1], [1, 2]], 3)
    result, _ = constrained_update(3, current, [[2, 0], [2, 1], [1, 0]], [9, 1, 2], 2)
    np.testing.assert_array_equal(result, [[0, 1], [0, 2]])


def test_swap_limit_zero_preserves_full_graph():
    current = feasible_seed(10, 9)
    candidates = candidate_edges(10, current, 9, np.random.default_rng(0))
    scores = np.random.default_rng(1).normal(size=len(candidates))
    result, info = constrained_update(10, current, candidates, scores, 9, max_swaps=0)
    np.testing.assert_array_equal(result, current)
    assert info["accepted"] == 0


@pytest.mark.parametrize("seed", range(8))
def test_every_single_exchange_state_satisfies_constraints(seed):
    n, budget, degree = 12, 20, 4
    rng = np.random.default_rng(seed)
    current = feasible_seed(n, budget, degree, seed=seed)
    candidates = candidate_edges(n, current, n - 1, rng)
    current, _ = constrained_update(n, current, candidates, rng.normal(size=len(candidates)), budget, degree, 0)
    assert_feasible(n, current, budget, degree)
    for _ in range(15):
        previous = current.copy()
        current, info = constrained_update(n, current, candidates, rng.normal(size=len(candidates)), budget, degree, 1)
        assert info["accepted_swaps"] <= 1
        assert_feasible(n, current, budget, degree)
        assert len(set(map(tuple, previous)) - set(map(tuple, current))) <= 1


def test_projection_returns_a_feasible_graph_not_an_optimality_claim():
    n = 15
    candidates = canonical_edges(list(itertools.combinations(range(n), 2)), n)
    scores = np.random.default_rng(3).normal(size=len(candidates))
    projected = projection(n, candidates, scores, budget=25, max_degree=4)
    assert_feasible(n, projected, 25, 4)


def test_projection_rejects_duplicate_candidates_and_missing_current_edges():
    with pytest.raises(ValueError, match="unique"):
        projection(2, [[0, 1], [1, 0]], [0, 1], 1)
    with pytest.raises(ValueError, match="include"):
        constrained_update(3, [[0, 1], [1, 2]], [[0, 1], [0, 2]], [0, 1], 2)


def test_validation_counts_independent_violations():
    status = validate_graph(4, [[0, 1], [1, 0], [1, 1], [-1, 2], [2, 4]], 2)
    assert not status["valid"]
    assert status["duplicate_edges"] == 1
    assert status["self_loops"] == 1
    assert status["out_of_bounds"] == 2
    assert status["budget_excess"] == 3
    assert not status["connected"]


def test_radius_edges_include_the_boundary_and_colocated_nodes():
    positions = np.asarray([[0, 0], [1, 0], [2, 0], [0, 0]], dtype=float)
    assert set(map(tuple, radius_edges(positions, 1))) == {(0, 1), (0, 3), (1, 2), (1, 3)}
    assert set(map(tuple, radius_edges(positions, 0))) == {(0, 3)}


def test_dynamic_repair_has_zero_churn_without_motion():
    positions = np.column_stack((np.arange(8), np.zeros(8)))
    previous = canonical_edges([[u, u + 1] for u in range(7)], 8)
    result, info = dynamic_repair(8, previous, positions, 1.1, 10, 3)
    np.testing.assert_array_equal(result, previous)
    assert info["churn"] == 0
    assert info["retained"] == 7


def test_dynamic_repair_replaces_out_of_range_edges_and_keeps_connectivity():
    positions = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    previous = canonical_edges([[0, 1], [1, 2], [0, 2]], 4)
    result, info = dynamic_repair(4, previous, positions, 1.1, 3, 2)
    assert_feasible(4, result, 3, 2, positions, 1.1)
    assert (0, 2) not in set(map(tuple, result))
    assert info["removed_out_of_range"] == 1
    assert info["churn"] == info["added"] + info["removed"]


def test_dynamic_repair_distinguishes_range_disconnect_and_degree_search():
    with pytest.raises(InfeasibleGraphError, match="disconnected"):
        dynamic_repair(3, [[0, 1], [1, 2]], [[0, 0], [1, 0], [10, 0]], 1.1, 2)
    positions = [[0, 0], [1, 0], [-0.5, 0.866], [-0.5, -0.866]]
    with pytest.raises(FeasibilitySearchError):
        dynamic_repair(4, [], positions, 1.01, 3, 2)


def test_dynamic_repair_supports_a_smaller_budget_and_degree_cap():
    positions = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    previous = canonical_edges(list(itertools.combinations(range(4), 2)), 4)
    result, _ = dynamic_repair(4, previous, positions, 2, 3, 2)
    assert_feasible(4, result, 3, 2, positions, 2)


def test_constraint_validator_reports_radius_violations():
    report = validate_graph(3, [[0, 1], [1, 2]], 2, positions=[[0, 0], [1, 0], [3, 0]], radius=1)
    assert report["radius_violations"] == 1
    assert not report["valid"]


@pytest.mark.parametrize("seed", range(5))
def test_moving_agents_preserve_all_constraints_at_each_frame(seed):
    rng = np.random.default_rng(seed)
    positions = np.column_stack((np.arange(12), np.zeros(12))).astype(float)
    current = feasible_seed(12, 18, 4, allowed=radius_edges(positions, 2.2), seed=seed)
    for _ in range(12):
        positions += rng.uniform(-0.02, 0.02, size=positions.shape)
        current, info = dynamic_repair(12, current, positions, 2.2, 18, 4, seed)
        assert_feasible(12, current, 18, 4, positions, 2.2)
        assert info["status"] == "feasible"


def test_all_connected_four_vertex_states_remain_feasible_after_an_edit():
    all_edges = list(itertools.combinations(range(4), 2))
    for mask in range(1 << len(all_edges)):
        current = canonical_edges([edge for i, edge in enumerate(all_edges) if mask & (1 << i)], 4)
        status = validate_graph(4, current, 6)
        if not status["connected"]:
            continue
        degree = status["max_degree"]
        current_set = set(map(tuple, current))
        for edge in set(all_edges) - current_set:
            candidates = canonical_edges([*current.tolist(), edge], 4)
            scores = [float(tuple(candidate) == edge) for candidate in candidates]
            for budget in (len(current), len(current) + 1):
                result, info = constrained_update(4, current, candidates, scores, budget, degree, 1)
                assert_feasible(4, result, budget, degree)
                assert info["accepted"] <= 1
