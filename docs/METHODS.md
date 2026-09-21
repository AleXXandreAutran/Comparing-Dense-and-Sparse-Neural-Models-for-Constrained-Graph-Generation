# Methods

This study compares dense and sparse neural models for constrained graph generation. We evaluate graph quality, validity, runtime, and memory use. Earlier experiments following a different protocol are kept in `results/legacy/`.

## Data

We use two synthetic graph families, both with 32 nodes and 64 edges.

The first is a conditioned stochastic block model (SBM) with four equal communities and stronger within-community connectivity. Only connected graphs with exactly 64 edges are retained.

The second contains planar graphs built from Delaunay triangulations of random points. A spanning tree is kept and extra edges are added until the graph reaches 64 edges. Node coordinates are then discarded.

For each family, we use 96 training, 24 validation, 32 test, and 32 independent reference graphs. Nodes are randomly relabelled. Possible duplicates are detected with WL hashes and confirmed with exact graph isomorphism tests.

The dataset seed is 20260921, and training uses seeds 11, 23, and 37. Full settings are given in [research.json](../configs/research.json).

## Models and training

The dense model scores every possible node pair using a GNN with degree and common-neighbour features. It is trained to reconstruct clean graphs from Bernoulli-corrupted inputs using binary cross-entropy (BCE).

The sparse model instead performs message passing on current edges and scores only a smaller candidate set. Its inputs include degree, relative degree, noise time, and graph family. During training, candidates contain all true edges and sampled nonedges. During generation, new candidates are proposed from the current graph.

| Setting      | Dense | Sparse |
| ------------ | ----: | -----: |
| Hidden width |    24 |     24 |
| GNN layers   |     2 |      2 |
| Parameters   | 8,737 |  4,609 |
| Epochs       |    80 |     40 |

Both models use Adam with learning rate 0.001 and batch size 16. Checkpoints are selected using validation BCE. Generation temperatures of 0.25, 0.5, and 1 are compared using degree, clustering, and spectral MMD².

Ablations study the number of generation steps, candidate refresh, all-pairs scoring, projection strategy, degree constraints, and removal of message passing.

## Constraints and sparse generation

Generated graphs must remain simple, connected, and within an edge budget `B`. Optional constraints impose a maximum degree `D` and a communication radius.

The decoder either adds a feasible edge or adds an edge while removing another edge from the resulting cycle. This exchange preserves connectivity as one atomic operation.

Sparse candidate sets contain at most

$$
m + kn
$$

pairs for a graph with `n` nodes, `m` current edges, and `k` proposals per node, up to the total number of possible pairs. The main configuration uses `k=4` and refreshes candidates at every generation step.

## Baselines and quantization

Three constrained baselines are included: random Gumbel scores, a degree-based prior, and an SBM-style prior fitted on the training data.

Optional dynamic INT8 quantization is applied to linear layers. Execution status, model storage, runtime, and memory are recorded separately.

External graph samples can also be imported with provenance information such as repository commit, configuration, split hash, seeds, hardware, and timing scope.

## Quality and validity

All generated graphs are evaluated, including invalid ones.

Graph fidelity is measured from 32-bin histograms of normalized degree, clustering, and normalized-Laplacian eigenvalues. Generated and test distributions are compared using biased RBF MMD², with kernel bandwidths estimated from training data only.

We also report validity, planarity, uniqueness, novelty, and graph robustness. Overall validity checks simplicity, connectivity, edge budget, degree limits, and geometric constraints.

Robustness is measured using algebraic connectivity and 20 random edge-failure trials per graph, where each edge is removed independently with probability 0.1.

## Runtime and memory

Scaling experiments use graph sizes from 32 to 512 nodes, while keeping models trained only on 32-node graphs.

Timing is measured in fresh CPU processes with one PyTorch thread, two warmups, and seven repetitions. We report median latency and quartiles.

Candidate scoring and complete graph generation are measured separately. CPU memory is tracked through process RSS sampled every millisecond. Model weight storage is reported independently.

GPU measurements are stored as `null` when unavailable.

## Moving agents

A dynamic experiment follows 32 moving agents over 30 frames using three motion seeds. All methods observe the same positions, with edge budget 64, maximum degree six, and communication radius 0.34.

We compare three strategies: rebuilding from distances, repairing the previous topology, and combining static neural scores with distance and edge-retention terms.

We report edge churn, connectivity, resilience, runtime, and failed frames. An explicitly disconnected range graph is also included to test infeasibility handling.

## Reproducibility

Each run stores its configuration, source snapshot, environment information, dataset hashes, checkpoints, generated samples, and trajectories.

Verification recomputes metrics and checks graph constraints. Automated tests cover constraints, metrics, profiling, and imported samples, while CI runs the smoke pipeline.

Resumed runs verify dataset hashes and sizes before reusing checkpoints. Full verification requires all expected models, methods, seeds, graph families, and sample batches to be present.
