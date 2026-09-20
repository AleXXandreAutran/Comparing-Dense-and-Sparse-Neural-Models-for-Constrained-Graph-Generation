# Methods

This study compares dense and sparse neural models for graph generation. It measures graph quality, constraint satisfaction, runtime and memory use. The `results/legacy/` directory stores earlier experiments with their own protocol.

## Data

The main dataset contains two graph families, each with 32 nodes and 64 edges:

- Conditioned SBM: four equal-sized communities, with edges six times more likely within a community than between communities before probability clipping. Only connected graphs with exactly 64 edges are kept.
- Planar graphs: a Delaunay triangulation of random points in the unit square, reduced to a spanning tree and enough extra edges to reach 64. Coordinates are then discarded.

These are custom synthetic datasets. Each family has 96 training, 24 validation, 32 test and 32 independent reference graphs. Nodes are randomly relabelled. WL hashes flag possible duplicates, followed by exact isomorphism checks to reject repeats within and across splits.

The dataset seed is 20260921; training uses seeds 11, 23 and 37. Full settings are in [research.json](../configs/research.json). The smoke configuration checks the pipeline.

## Models and training

The dense model is a local binary-edge diffusion implementation. It uses a GNN, degree and common-neighbour features, and scores every unordered node pair. Training predicts clean edges from Bernoulli-corrupted graphs with binary cross-entropy (BCE). Sampling uses reverse edge probabilities, then a decoder enforces the graph constraints.

The sparse model passes messages along current edges and scores a smaller set of candidate pairs. Its inputs include degree, relative degree, noise time and graph family. The edge head combines endpoint embeddings symmetrically. Static training uses topology features and family labels.

During sparse training, candidates contain every clean edge plus random nonedges. BCE is computed on this support. At generation time, candidates come from the current graph and fresh proposals. Sampling combines neural scores, Gumbel noise and feasible graph edits.

| Setting | Dense | Sparse |
|---|---:|---:|
| Hidden width | 24 | 24 |
| Message-passing layers | 2 | 2 |
| Trainable parameters | 8,737 | 4,609 |
| Training epochs | 80 | 40 |

Both use Adam with learning rate 0.001 and batch size 16. Checkpoints are selected by validation BCE with fixed corruption. Temperatures of 0.25, 0.5 and 1 are compared using mean validation degree, clustering and spectral MMD².

Ablations cover 1, 4 and 8 generation steps, fixed or refreshed candidates, all-pairs scoring, rebuilt projection, a degree cap of six, and removal of message passing. The variant without message passing receives degree and candidate-membership information. Training graphs have no degree cap.

The comparison covers two complete generation pipelines, with the architectures and training budgets listed above. Training uses BCE; runtime and memory are measured during evaluation.

## Constraints and sparse computation

Generated graphs must be simple, connected and contain at most `B` edges. Optional constraints limit node degree to `D` and restrict edges to a communication radius.

Starting from a feasible connected graph, the decoder can:

1. Add an edge if the budget and endpoint degrees allow it.
2. Add an edge and remove an existing edge on the resulting cycle, checking the final degrees. This exchange is one atomic update, preserving connectivity.

Every committed state must satisfy the constraints. The greedy decoder adds admissible edges toward the budget. The projected variant rebuilds a feasible spanning tree before adding edges.

Initialization uses a random path without a range mask, or a bounded greedy spanning-tree search with an allowed edge set. Proven basic infeasibility raises `InfeasibleGraphError`; exhausting the search raises `FeasibilitySearchError`.

With `n` nodes, `m` current edges and `k` proposed neighbours per node, the candidate count is at most `m + kn`, capped by the number of possible pairs. The main configuration uses `k=4` and refreshes candidates at each step.

For fixed model width, sparse denoising storage grows with nodes, edges and candidates. Constraint checks sort candidate scores and search for paths. A geometric mask stores allowed pairs, with quadratic storage for dense range neighbourhoods.

## Baselines and quantization

Three local baselines use the same constrained decoder: random Gumbel scores, a degree prior fitted on training graphs, and an SBM-style score model fitted from training communities. The degree and SBM-style priors are fitted before timing.

Optional post-training dynamic INT8 quantization applies to linear layers. Backend support, execution status, storage and performance are recorded separately.

An import interface accepts external samples with provenance metadata: upstream repository and commit, configuration, test-split hash, seeds, hardware, command and timing scope. The main protocol requires 32 samples per family and seed, with 32 nodes per graph. Import validation checks the schema and split identity.

## Quality and validity

Every generated sample is evaluated, including invalid outputs. The main fidelity metrics use 32-bin histograms of normalized degree, clustering and normalized-Laplacian eigenvalues. Biased RBF MMD² compares generated graphs against the test split; lower values indicate closer distributions for the same metric and protocol.

Kernel bandwidths use the median positive distance between training histograms, using at most 512 graphs, with a fallback of one. Test and generated graphs never set them. The independent reference split is also compared with the test split to show finite-sample variation.

Other metrics cover histogram and graph-count distances, planarity, validity, and WL-based uniqueness and novelty. Overall validity combines simplicity, connectivity, edge budget, degree and any range constraint. Planarity is reported separately.

Robustness uses algebraic connectivity, the second eigenvalue of the unnormalized Laplacian, and 20 edge-failure trials per graph. Each edge is removed independently with probability 0.1, using seed 2718 plus the sample index.

## Runtime and memory

Scaling experiments use a fresh CPU process for each method and size, one PyTorch thread, two warmups and seven timed repetitions. Reports give median latency and quartiles. Inference cost is measured at 32, 64, 128, 256 and 512 nodes, using models trained at 32 nodes, four generation steps and checkpoint seed 11.

Two measurements are kept separate:

- Candidate scoring uses identical sparse weights and the same active graph, with sparse or all-pairs candidates prepared before timing.
- Complete generation includes initialization, candidate construction, neural scoring, constraint handling and invariant checks. Model loading and metric evaluation are excluded.

CPU memory is process RSS sampled every millisecond in a separate pass after timing. It includes Python, PyTorch and allocator caches. Baseline RSS, sampled peak, their difference and process-lifetime maximum are distinct measurements. Weight storage is reported separately.

Quality runs use one timing repetition without warmup inside the main research process. Metrics use the samples returned by the memory pass. Random generators are reset between profiling passes.

Unavailable GPU measurements are recorded as `null`.

## Moving agents

The dynamic experiment moves 32 agents over 30 frames with three motion seeds. All methods see the same positions, with edge budget 64, degree cap six and communication radius 0.34.

Three approaches are compared: rebuilding from distance scores, repairing and retaining the previous topology, and adding static neural scores to distance and retention rules. This experiment transfers the model trained on static topology to moving agents.

Churn counts edges added or removed between frames. Its normalized form divides by the edge-set union; frame zero is excluded from transition averages. Reports also include connectivity, resilience, runtime and failed frames. A disconnected range graph provides an explicit infeasible case.

## Reproducibility

Runs save configurations, source snapshots, environment details, split hashes, checkpoints, calibration, samples and trajectories. Verification recomputes metrics and checks intermediate graph states. Tests cover constraints, metrics, profiling and imported samples; CI also runs the smoke pipeline.

Resume checks all four dataset hashes and counts before reusing checkpoints or writing files. Full verification requires all expected methods, seeds, families, sample batches and models. INT8 results are required only when their saved status is `executed`. Packing reruns full verification. `verify --partial` creates a separate report for unfinished experiments. Resumed runs clear earlier completion and verification records until those stages finish again.
