# Methods and evaluation

This document describes the research extension in `src/frugal_graphs/`: `research.py`, `sparse_model.py`, `constrained.py` and `research_metrics.py`. The original experiments in `results/legacy/results.json` use a different dataset and protocol. Their scores must not be mixed with the new benchmark.

## Data and split policy

The main configuration uses 32 vertices and 64 edges. Each family has 96 training, 24 validation, 32 test and 32 independent reference graphs. The dataset seed is 20260921. The training seeds are 11, 23 and 37. `configs/research.json` is the source of truth; the smoke configuration is only a pipeline check.

Two families are generated:

- **Conditioned SBM.** Four balanced blocks, independent Bernoulli edges and a within/between probability ratio of six before clipping. Draws are accepted only when connected and when they contain exactly 64 edges. This is an SBM conditioned on these events, not an unconditioned SBM benchmark.
- **Thinned Delaunay graphs.** Uniform points in the unit square define a Delaunay triangulation. A spanning tree selected using random edge weights is retained, then random remaining triangulation edges fill the edge budget. The result is connected and planar. This is a custom planar family, not an official benchmark split from a cited paper.

Graphs receive random vertex labels. WL hashes identify possible duplicates, then exact graph isomorphism checks reject duplicates across all four splits. The families therefore share neither labelled copies nor isomorphic copies across splits. Sampling without replacement slightly changes the family distribution. The split manifest records graph counts and hashes. The planar coordinates are discarded: static models learn topology and family labels, without positions.

The test split is the fidelity reference for model comparisons. The independent reference split is also compared against the test split in the reference_sample rows. This finite-sample comparison is not a guaranteed metric floor.

## Models and training

The dense reference is the repository's binary-edge diffusion model. It uses a GNN, a symmetric edge head, explicit degree/common-neighbour features and all unordered pairs. Training predicts clean edges from Bernoulli-corrupted graphs with BCE. The cosine-squared retention schedule reaches zero at its terminal step. Sampling uses per-edge reverse probabilities followed by a final connected, budgeted decoder. It is a local implementation, not an execution of DiGress.

The sparse model passes messages on the current edges and scores only candidate pairs. It uses node degree, relative degree, noise time and family features. Its symmetric edge head uses sums, absolute differences and products of endpoint embeddings. Candidate membership and endpoint degrees remain available when message passing is disabled. The no-message-passing ablation therefore still has structural information. Optional coordinate inputs exist in the API, but are absent from the recorded static training protocol.

Sparse training always includes every clean edge and adds random nonedges to its candidate support. Bernoulli corruption acts on that support. BCE predicts clean candidate labels; the prior is the clean-edge fraction within the support. This is a support-conditioned denoising objective, not an unbiased all-pairs likelihood. At generation time the support comes from the current graph and fresh candidates, so there is a training/sampling support mismatch. Feasible edits guided by the network and annealed Gumbel perturbations define the sampler. No exact diffusion posterior or learned sampling distribution with a known density is claimed.

With hidden width 24 and two message-passing layers, the sparse model has 4,609 trainable parameters and the dense model has 8,737. The main training budgets are 40 sparse epochs and 80 dense epochs, with Adam at 0.001 and batch size 16. Checkpoints minimize validation BCE using fixed validation corruption. These models differ in architecture, parameter count, objective and training budget; the full-generation comparison cannot isolate sparsity alone. The same-weight candidate-scoring experiment below provides a narrower comparison.

For each method, seed and family, temperature is selected from 0.25, 0.5 and 1 using the mean validation degree, clustering and spectral MMD². The test split is not used for calibration. The one-shot heuristics use temperature one without tuning. Step counts 1, 4 and 8, fixed versus refreshed candidates, all-pairs candidates, rebuilt projection, no message passing and degree cap six provide separate ablations. A degree cap can restrict the target distribution, since the training data were not conditioned on that cap.

Sampling cost is measured during evaluation. The training losses do not include differentiable time or memory penalties. Quantization is optional post-training dynamic int8 quantization of linear layers; backend support, execution status and measurements are recorded. A smaller representation is not assumed to improve speed, peak memory or fidelity.

## Constraints and their invariant

The admissible set contains simple, connected graphs with at most `B` edges and, when requested, maximum degree at most `D`. Dynamic graphs additionally require every edge to lie inside the communication radius. Exact edge count is measured separately. Planarity is a fidelity/validity statistic for the planar family, not a hard sampling constraint.

The sampler starts from a feasible connected seed. It accepts two types of update:

1. Add an edge only when the budget and both endpoint degrees permit it. Connectivity is preserved.
2. Add a candidate edge `(u, v)` and remove one edge on an existing path from `u` to `v`. The new edge closes a cycle, so removing an old edge on that cycle preserves connectivity. The final degrees must satisfy the cap.

An exchange is one atomic graph transition. Serializing removal and addition as separately observable states would not give the same invariant. By induction from the feasible seed, every committed state remains connected, simple and within the budget and degree cap. Restricting both current edges and candidates to the allowed range graph preserves geometric validity as well.

Accepted exchanges strictly improve the current supplied score sum. Additions prefer filling the budget even when their scores are negative. A degree or range restriction can leave fewer than `B` edges. The decoder is a feasibility heuristic: it does not guarantee the largest attainable edge count, maximum score or minimum number of modifications. The projected ablation rebuilds a score-prioritized feasible spanning tree before completing it, rather than continuing the current graph through admissible edits. It is not a minimum-edit projection or a differentiable constraint operator.

Without a range mask, initialization uses a random path. With an explicit allowed set it uses greedy spanning-tree searches, with a bounded number of retries when a degree cap applies. A disconnected allowed graph or an impossible basic edge/degree bound produces `InfeasibleGraphError`. Exhausting the degree-constrained search produces `FeasibilitySearchError`; search failure is not a mathematical proof of infeasibility. Neither case is repaired by violating a constraint.

## Sparsity and complexity

Let `n` be the number of vertices, `m` the current edge count, `k` the candidate neighbours requested per vertex and `c` the candidate count. Current edges remain in the support, with `c <= m + kn` and `c <= n(n-1)/2`. Setting `k >= n-1` explicitly requests every pair. The main sparse configuration uses `k=4` and refreshes candidates at each edit step.

The sparse representation stores edge arrays, node features and adjacency sets. For fixed hidden width, denoising storage grows with `n + m + c`; no dense adjacency or common-neighbour matrix is needed. Candidate canonicalization sorts edge keys. Constraint updates sort candidate scores and may search for a path for each proposed exchange: a conservative bound is `O(c log c + c(n+m))`, plus deterministic neighbour-ordering costs. Sparse storage therefore does not imply linear end-to-end sampling time. Global feasibility work can dominate neural scoring.

An explicit geometric mask has its own size `a`. Radius neighbours use a spatial tree, but listing and storing them is output-sensitive and can still require quadratic space when almost all agents are within range. Sparse candidate claims do not remove that cost. Dense eigenvalue calculations and NetworkX statistics belong to evaluation, outside the sampling timers.

## Fidelity, validity and robustness

Every method's complete generated batch is evaluated. Invalid samples stay in the denominator. Loops, duplicate undirected edges and invalid endpoints fail simple-graph validity; other statistics use the corresponding simple graph after invalid edges are removed.

Per-graph histograms use 32 bins: degree divided by `max(n-1,1)` on `[0,1]`, local clustering on `[0,1]`, and normalized-Laplacian eigenvalues on `[0,2]`. Empty graphs use a unit mass at zero by convention and still fail connectivity. RBF biased squared MMD compares the histogram distributions. Each bandwidth is the median positive Euclidean distance among at most the first 512 training histograms, with fallback one for a constant training statistic. Neither generated graphs nor test graphs set bandwidths. Scores from different statistics, families or protocols are not interchangeable.

Additional metrics include Wasserstein and total variation distances between mean histograms; node-, edge- and triangle-count Wasserstein distances; connectivity, simplicity, budget and degree violations; planarity; and WL-based uniqueness and novelty. WL values are proxies, not exact isomorphism certificates. Aggregate `valid_rate` combines simplicity, connectivity, budget, degree and applicable physical range. Planarity remains separate. Family-specific results are also computed for mixed batches.

Algebraic connectivity is the second eigenvalue of the **unnormalized** Laplacian. It is zero for disconnected graphs and graphs with fewer than two vertices. Edge-failure resilience independently removes each edge with probability 0.1 in 20 trials per graph. Seeds are 2718 plus the sample index. Reports include connected fraction and largest-component fraction. These empirical topology tests do not establish physical system stability; compare them alongside graph size and edge count.

## Timing and memory

Quality rows obtain their metric batch from the memory pass of `profile_operation(..., return_result=True)`. The same deterministic generation operation is used for the timing pass. Quality profiling uses one timing repetition and no warmup, so its latency is descriptive, not a stable microbenchmark. It also occurs within the research process: allocator history and other loaded models contribute to process RSS. Per-graph generation times are retained alongside the batch measurement.

Scaling uses a fresh CPU process for each method/size pair, one PyTorch thread, two warmups and seven timed repetitions. It reports median, 25th and 75th percentile latency. Two scopes are kept separate:

- **Candidate scoring:** identical sparse weights and active graph, with sparse versus all-pairs candidates prepared before timing. This isolates the neural scoring workload.
- **Complete generation:** initialization, candidate construction, scoring, constraint enforcement and invariant checks. Model loading and metric evaluation remain outside timing.

The scaling sizes are 32, 64, 128, 256 and 512, using four sampling steps and checkpoint seed 11. Weights were trained at 32 vertices. Larger-size measurements test inference cost under size extrapolation; they do not establish fidelity at those sizes or provide variation across training seeds.

CPU memory is sampled process RSS at 1 ms in a separate pass after timing. Reports retain RSS before warmup, the warmed baseline, sampled peak, peak minus baseline and process-lifetime maximum RSS. Polling may miss brief peaks. Native allocator caches and unrelated process allocations remain included; a zero RSS delta does not mean zero working memory. The lifetime high-water mark includes imports, warmup and earlier allocations, and is not an operation-only peak. Model parameter/storage bytes are separate quantities. Fresh workers improve comparability without eliminating these limitations.

Available CUDA devices report synchronized timing and allocated/reserved peaks after resetting CUDA peak counters. Unavailable GPU measurements are `null`, not zero. The local CPU experiments support no GPU speed or energy claim. Global Python, NumPy and loaded PyTorch RNGs are reset before each pass; operations using private generators need an explicit reset or a fixed per-call seed. Baseline fitting is performed before timing.

## Dynamic agents

The main case study moves 32 agents for 30 frames, with three motion seeds, budget 64, degree cap six and radius 0.34. Small Gaussian displacements are clipped to the unit square. All methods see the same positions. Rebuild uses distance-based scores; retain first repairs the previous topology; learned-retain adds the static network's scores to hand-designed distance and retention terms. Retained graphs are filled toward the same budget when admissible edges are available. Budget underfilling is still possible under the constraints.

The learned model was trained on static topology without coordinates. This is a transfer experiment with geometric rules, not a trained motion model or a robotics simulation. No learned minimum-churn or optimal-robustness guarantee is claimed. A disconnected range graph is included as an explicit infeasible case.

Churn is the size of the edge-set symmetric difference; normalized churn divides by the union size. Frame zero is excluded from mean transition churn. λ₂, edge-failure resilience, runtime and failed/infeasible frames accompany churn. Mean statistics over feasible frames must be read with the number of failures, rather than treating missing outputs as valid graphs.

## Local and external baselines

Local one-shot baselines share the constrained decoder:

- `random`: independent Gumbel scores on admissible pairs.
- `degree_prior`: a training-fitted, sorted degree profile, interpolated to the requested size, shuffled across vertices and converted to Chung–Lu-style edge scores.
- `fitted_sbm`: greedy-modularity communities from training graphs, pooled smoothed within/between probabilities, and shuffled approximately equal-size blocks.

These are score heuristics followed by a decoder. They are not uniform connected-graph samplers, exact SBM draws or implementations of SPECTRE. Their fitting uses training graphs only and occurs outside timing.

Official implementations are [DiGress](https://github.com/cvignac/DiGress), [SparseDiff](https://github.com/qym7/SparseDiff), [SPECTRE](https://github.com/KarolisMart/SPECTRE) and [GDSS](https://github.com/harryjo97/GDSS). They have **not been executed as part of the local benchmark**. The repository provides a provenance-checked import path for their exported samples. It does not rename local baselines after these papers.

An external export uses the following JSON structure. The repeated letters are illustrative placeholders: replace them with actual hashes, include the full run configuration, and identify the executed upstream command and hardware. The method belongs inside `metadata`.

```json
{
  "metadata": {
    "method": "DiGress",
    "upstream_url": "https://github.com/cvignac/DiGress",
    "repo_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "config": {"dataset": "sbm", "sampling_steps": 500},
    "split_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "seeds": [11],
    "timing_scope": "Attached batch generation including correction, excluding evaluation",
    "hardware": "Actual CPU or GPU, RAM, operating system and thread count",
    "command": "Actual upstream command"
  },
  "samples": [
    {"n": 3, "edges": [[0, 1], [1, 2]], "family": 0, "seed": 11}
  ]
}
```

The three-node record above illustrates the schema only. For the main benchmark, provide 32-node graphs and exactly 32 samples per family for every declared seed, with a seed field on each record. Use integer family IDs, zero-based undirected edge lists and the exact test split hash from `dataset_manifest.json`. The adapter checks required metadata, full commit/hash formats, requested counts and split identity, and returns the export file's SHA-256. Invalid graph outputs remain available for validity evaluation. Schema validation does not certify that upstream code was run. Timing supplied as extra metadata remains externally reported timing. Export raw and corrected samples separately; include correction cost when comparing corrected methods. Match splits, sample counts, seeds and timing scope before making a performance claim.

## Reproducibility and scope

Saved configurations, sources, environment, split hashes, checkpoints, validation calibration, generated graphs and trajectories identify each run. Verification recomputes metrics from saved samples and checks intermediate feasible states. Tests cover known spectra, permutation-invariant statistics, training-only bandwidths, invalid outputs, RNG reset, profiling cleanup, external provenance, constrained edits and sparse inference. CI also runs the smoke pipeline.

Resume checks all four dataset hashes and counts against the saved manifest before reusing checkpoints or writing files. Full verification requires every expected method/seed/family combination, sample batch and model, including quantized results only when their saved status is `executed`. Packing reruns full verification before writing the bundle. For an unfinished experiment, `verify --partial` writes a separate partial report; it does not allow packing. A resumed run clears its previous completion and verification records until the corresponding stages finish again.

This is a small controlled study with three training seeds and custom synthetic distributions. It does not establish state-of-the-art quality, embedded deployment, measured energy savings, or the benefits of coordinate-conditioned learning. A useful outcome may be a negative one: sparse neural scoring can be cheaper while global constraint handling removes that advantage in complete generation. Interpret the measured tables and their scope rather than assuming a gain from parameter or candidate counts.
