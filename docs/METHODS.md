# Methods

This study compares dense and sparse neural approaches to graph generation. We evaluate not only the quality of the generated graphs, but also whether they satisfy the required constraints, how long generation takes, and how much memory it uses. Earlier experiments, based on a slightly different protocol, are kept separately in `results/legacy/`.

## Data

We use two families of synthetic graphs. Every graph has 32 nodes and exactly 64 edges.

The first family is a conditioned stochastic block model (SBM). Each graph contains four equally sized communities. Before probabilities are clipped, an edge is six times more likely to appear between two nodes in the same community than between nodes from different communities. We keep only connected graphs with exactly 64 edges.

The second family contains planar graphs. We first sample random points in the unit square and construct their Delaunay triangulation. From this graph, we keep a spanning tree and then add enough remaining edges to obtain exactly 64 edges. The node coordinates are discarded afterwards, so the models only observe graph topology.

Both datasets are generated specifically for this study. For each graph family, we use 96 graphs for training, 24 for validation, 32 for testing, and another 32 as an independent reference set. Node labels are randomly permuted. To reduce the risk of duplicate graphs, we first compare Weisfeiler-Lehman (WL) hashes and then run exact isomorphism tests whenever two hashes match. This procedure is applied both within and across dataset splits.

The dataset is generated with seed 20260921. Training is repeated with seeds 11, 23, and 37. The complete configuration is stored in [research.json](../configs/research.json). A smaller smoke configuration is also provided to check that the full pipeline runs correctly.

## Models and training

We compare two neural graph generators.

The dense model is a local binary-edge diffusion model. It uses a graph neural network (GNN), together with degree and common-neighbour features, and assigns a score to every unordered pair of nodes. During training, clean edges are reconstructed from graphs corrupted with Bernoulli noise. The loss is binary cross-entropy (BCE). During generation, the model predicts reverse edge probabilities and a decoder then enforces the graph constraints.

The sparse model avoids scoring every possible node pair. Message passing is performed only along the edges that are currently present, and the model scores a smaller set of candidate edges. Node features include degree, relative degree, diffusion time, and graph family. Candidate edges are scored from the two endpoint embeddings using a symmetric edge head. In the static experiments, the model is trained using topology features together with the graph-family label.

During sparse training, the candidate set always contains all clean edges as well as a random sample of nonedges. BCE is evaluated only on this candidate support. During generation, candidates are built from the current graph together with newly proposed edges. The sampler combines neural scores, Gumbel noise, and feasible graph edits.

| Setting                | Dense | Sparse |
| ---------------------- | ----: | -----: |
| Hidden width           |    24 |     24 |
| Message-passing layers |     2 |      2 |
| Trainable parameters   | 8,737 |  4,609 |
| Training epochs        |    80 |     40 |

Both models are trained with Adam, using a learning rate of 0.001 and a batch size of 16. Checkpoints are selected using validation BCE under a fixed corruption process.

For generation, we test temperatures of 0.25, 0.5, and 1. The temperature is selected from validation results using the mean MMD² obtained from the degree, clustering, and spectral distributions.

We also run several ablations. These compare 1, 4, and 8 generation steps; fixed and refreshed candidate sets; sparse and all-pairs scoring; standard and rebuilt projection; the presence or absence of a degree cap of six; and models with or without message passing. When message passing is removed, the model still receives degree and candidate-membership information. Importantly, no degree cap is imposed on the training graphs.

The main comparison therefore concerns two complete generation pipelines, rather than two isolated neural networks. Their architectures and training budgets are reported above. BCE is used for training, while runtime and memory are measured during evaluation.

## Constraints and sparse computation

Every generated graph must be simple, connected, and contain no more than `B` edges. Additional experiments may also impose a maximum node degree `D` and a communication-radius constraint.

Starting from a feasible connected graph, the decoder supports two basic operations.

1. It may add a new edge when the edge budget and degree constraints allow it.
2. It may add an edge and remove another edge from the cycle that is created. The final node degrees are checked before accepting the move.

The second operation is treated as one atomic update. This is important because connectivity is preserved throughout the modification rather than being broken temporarily.

Every committed graph state is required to satisfy all active constraints. The greedy decoder keeps adding admissible edges until it reaches the edge budget or no further valid move is available. The projected decoder instead rebuilds a feasible spanning tree before adding the remaining edges.

Without a geometric range constraint, initialization uses a random path. When an allowed-edge mask is present, initialization uses a bounded greedy search for a feasible spanning tree. If the constraints are known to be impossible, the code raises `InfeasibleGraphError`. If feasibility is not ruled out in advance but the search cannot find a solution, it raises `FeasibilitySearchError`.

Suppose the graph has `n` nodes and `m` current edges, and each node proposes at most `k` candidate neighbours. The number of candidate pairs is then at most

$$
m + kn,
$$

up to the total number of possible unordered node pairs. In the main experiments, we use `k=4` and rebuild the candidate set at every generation step.

For a fixed hidden dimension, the memory required for sparse denoising grows with the number of nodes, current edges, and candidate edges. Constraint handling also has a computational cost: candidate scores must be ordered, and some updates require path searches. When geometric constraints are used, the allowed-edge mask itself may require quadratic memory if the range neighbourhood is dense.

## Baselines and quantization

We compare the neural models with three local baselines, all using the same constrained decoder.

The first assigns random Gumbel scores to candidate edges. The second uses a degree prior estimated from the training graphs. The third uses an SBM-like score model fitted from community structure in the training data. The degree and SBM-style models are fitted before runtime measurements begin, so their fitting cost is not included in inference time.

We also support optional post-training dynamic INT8 quantization of linear layers. Backend availability, whether quantization was actually executed, model storage, runtime, and memory measurements are recorded separately.

External graph samples can be evaluated through an import interface. Imported data must include provenance information such as the upstream repository and commit, configuration, test-split hash, random seeds, hardware, execution command, and timing scope. Under the main protocol, each method must provide 32 generated graphs for each family and seed, with 32 nodes per graph. Import validation checks both the data schema and the identity of the test split.

## Quality and validity

All generated samples are evaluated, including samples that violate one or more constraints.

The main fidelity analysis uses 32-bin histograms of three graph statistics: normalized degree, clustering coefficient, and eigenvalues of the normalized graph Laplacian. We compare generated and test distributions using biased RBF MMD². Under the same protocol and for the same metric, smaller values indicate that the generated graph distribution is closer to the test distribution.

Kernel bandwidths are estimated only from the training data. For each metric, we compute the median positive distance between training histograms, using at most 512 graphs. If no valid positive distance is available, the bandwidth defaults to one. Test graphs and generated graphs therefore never influence bandwidth selection.

The independent reference split is also compared against the test split. This gives a useful indication of the amount of MMD² variation that can arise simply from finite sample size, even when both sets come from the same data-generation process.

Additional metrics measure histogram differences, graph-count statistics, planarity, validity, uniqueness, and novelty. Uniqueness and novelty use WL hashes. Overall validity requires simplicity, connectivity, compliance with the edge budget, compliance with the degree limit when it is active, and compliance with any geometric range constraint. Planarity is reported separately and is not part of the general validity definition.

Graph robustness is evaluated in two ways. First, we compute algebraic connectivity, defined as the second-smallest eigenvalue of the unnormalized graph Laplacian. Second, we perform 20 edge-failure trials for every graph. In each trial, every edge is independently removed with probability 0.1. The random seed is 2718 plus the index of the generated sample.

## Runtime and memory

Scaling experiments are performed in a fresh CPU process for every method and graph size. PyTorch is restricted to one thread. Each timing experiment uses two warmup runs followed by seven measured repetitions. We report the median latency together with its quartiles.

Inference is evaluated on graphs with 32, 64, 128, 256, and 512 nodes. The same models are used at every size: they are trained only on 32-node graphs, using four generation steps and the checkpoint obtained with training seed 11.

We separate two different computational measurements.

The first measures candidate scoring alone. Sparse and all-pairs scoring use exactly the same sparse-model weights and the same active graph. Candidate sets are prepared before the timer starts.

The second measures complete graph generation. This includes graph initialization, candidate construction, neural scoring, constraint handling, and invariant checks. Model loading and downstream metric computation are excluded.

CPU memory is measured in a separate profiling pass after runtime measurement. The process resident set size (RSS) is sampled every millisecond. As a result, the measurement includes the Python interpreter, PyTorch, and allocator caches in addition to the graph-generation tensors themselves.

We keep several memory quantities separate: RSS before generation, sampled peak RSS, the increase from baseline to peak, and the maximum RSS observed over the lifetime of the process. Model weight storage is also reported independently.

The main quality experiments use a lighter timing setup: one timed repetition and no warmup, all within the main research process. Quality metrics are computed from the samples produced during the memory-profiling pass. Random-number generators are reset between profiling passes so that the measurements remain comparable.

When GPU measurements are not available, they are stored explicitly as `null`.

## Moving agents

We also evaluate the methods in a dynamic setting with 32 moving agents observed over 30 frames. The experiment is repeated with three motion seeds. Every method receives exactly the same agent positions. The edge budget is 64, the maximum degree is six, and the communication radius is 0.34.

Three approaches are compared.

The first rebuilds the graph at every frame using distance-based scores. The second repairs the previous graph while trying to retain as much of its topology as possible. The third combines static neural scores with both distance and edge-retention terms.

This final approach tests whether a model trained only on static graph topology can still be useful when the nodes themselves move over time.

Topology changes are measured using edge churn, defined as the number of edges added or removed between two consecutive frames. We also report normalized churn, obtained by dividing this value by the size of the union of the two edge sets. Since there is no previous graph at frame zero, the first frame is excluded from transition averages.

The dynamic evaluation also reports connectivity, resilience, runtime, and the number of failed frames. To check that infeasible situations are detected correctly, we include an explicit case in which the range-constrained communication graph is disconnected.

## Reproducibility

Each experimental run saves its configuration, source-code snapshot, environment information, dataset hashes, model checkpoints, calibration results, generated samples, and dynamic trajectories.

The verification stage recomputes the reported metrics and checks intermediate graph states against the required invariants. Automated tests cover constraint handling, evaluation metrics, profiling, and imported samples. Continuous integration also runs the smoke configuration.

When a run is resumed, all four dataset hashes and dataset sizes are checked before any checkpoint is reused or new result is written. Full verification requires all expected graph families, methods, training seeds, sample batches, and trained models to be present.

INT8 results are required only when the stored quantization status indicates that quantization was successfully executed. Packaging the final results triggers full verification again.

For incomplete experiments, `verify --partial` produces a separate partial-verification report rather than treating the run as complete. If a completed run is resumed, its previous completion and verification records are cleared until the corresponding stages have successfully finished again.
