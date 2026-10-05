# Methods

We compare dense and sparse neural models for generating graphs under constraints. We look at graph quality, validity, runtime, and memory use. Earlier experiments used a different protocol and are stored in `results/legacy/`.

## Data

We use two families of synthetic graphs. Both have 32 nodes and 64 edges.

The first family comes from a conditioned stochastic block model (SBM). It has four communities of equal size, with connections more likely within each community than between communities. We keep only connected graphs with exactly 64 edges.

The second family contains planar graphs made from Delaunay triangulations of random points. We keep a spanning tree, then add edges until the graph has 64 edges. We then discard the node coordinates.

For each family, we use 96 graphs for training, 24 for validation, 32 for testing, and 32 as an independent reference set. We randomly relabel the nodes. We flag possible duplicates using WL hashes, then check them with exact graph isomorphism tests.

The dataset seed is 20260921. Training uses seeds 11, 23, and 37. The full settings are listed in [research.json](../configs/research.json).

## Models and training

The dense model uses a GNN to score every possible pair of nodes. Its features include node degrees and common neighbours. It learns to recover clean graphs from inputs corrupted with Bernoulli noise, using binary cross-entropy (BCE).

The sparse model passes messages along the current edges and scores a smaller set of candidate edges. Its inputs include degree, relative degree, noise time, and graph family. During training, the candidate set includes all true edges and sampled nonedges. During generation, new candidates are proposed from the current graph.

| Setting | Dense | Sparse |
|---|---:|---:|
| Hidden width | 24 | 24 |
| GNN layers | 2 | 2 |
| Parameters | 8,737 | 4,609 |
| Epochs | 80 | 40 |

Both models use Adam with a learning rate of 0.001 and a batch size of 16. We select checkpoints using the validation BCE. We compare generation temperatures of 0.25, 0.5, and 1 using degree, clustering, and spectral MMD².

The ablation experiments test the number of generation steps, candidate refresh, all-pairs scoring, the projection strategy, and degree constraints. We also test what happens when message passing is removed.

## Constraints and sparse generation

Generated graphs must be simple, connected, and stay within an edge budget `B`. Optional constraints can also set a maximum degree `D` and a communication radius.

The decoder either adds an edge that meets the constraints, or adds an edge and removes another from the cycle it creates. This exchange is treated as a single operation, so the graph stays connected.

The sparse candidate set contains at most

$$
m + kn
$$

pairs, where `n` is the number of nodes, `m` is the number of current edges, and `k` is the number of proposals per node. The set cannot exceed the total number of possible node pairs. The main configuration uses `k=4` and refreshes the candidates at every generation step.

## Baselines and quantization

We include three baselines that follow the same constraints: random Gumbel scores, a degree-based prior, and an SBM-style prior fitted to the training data.

Dynamic INT8 quantization can be applied to the linear layers. We record execution status, model storage, runtime, and memory use separately.

Graph samples from other sources can also be imported. Their provenance information can include the repository commit, configuration, data split hash, seeds, hardware, and which part of the process was timed.

## Quality and validity

We evaluate every generated graph, including invalid ones.

To measure how well the generated graphs match the data, we build 32-bin histograms of normalized degree, clustering, and normalized-Laplacian eigenvalues. We compare the generated and test distributions using biased RBF MMD². Kernel bandwidths are estimated from the training data only.

We also report validity, planarity, uniqueness, novelty, and robustness. The overall validity check covers simplicity, connectivity, the edge budget, degree limits, and geometric constraints.

We measure robustness using algebraic connectivity and 20 random edge-failure trials per graph. In each trial, each edge has an independent 10% chance of being removed.

## Runtime and memory

The scaling experiments use graphs with 32 to 512 nodes. The models are trained only on 32-node graphs.

We measure runtime in fresh CPU processes, using one PyTorch thread. Each measurement includes two warmup runs and seven timed repetitions. We report median latency and quartiles.

We measure candidate scoring and complete graph generation separately. CPU memory is tracked through process RSS, sampled every millisecond. The storage used by model weights is reported separately.

GPU measurements are saved as `null` when unavailable.

## Moving agents

A dynamic experiment tracks 32 moving agents over 30 frames, using three motion seeds. All methods observe the same positions and use an edge budget of 64, a maximum degree of six, and a communication radius of 0.34.

We compare three strategies: rebuilding the graph from distances, repairing the previous graph, and combining static neural scores with distance and edge-retention terms.

We report changes in connections (edge churn), connectivity, resilience, runtime, and failed frames. We also include a disconnected communication-range graph to test how the system handles a case where the agents cannot all be connected.

## Reproducibility

Each run saves its configuration, a snapshot of the source code, environment details, dataset hashes, checkpoints, generated samples, and trajectories.

Verification recomputes the metrics and checks the graph constraints. Automated tests cover constraints, metrics, profiling, and imported samples. CI runs the smoke pipeline.

When a run is resumed, dataset hashes and sizes are checked before checkpoints are reused. Full verification requires all expected models, methods, seeds, graph families, and sample batches to be present.
