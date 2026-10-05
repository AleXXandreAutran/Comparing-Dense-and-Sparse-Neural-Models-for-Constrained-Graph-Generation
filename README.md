# Comparing Dense and Sparse Graph Generation

This project compares dense and sparse neural models for generating graphs that must stay connected and respect an edge budget, with an optional maximum-degree constraint.

The aim is to understand what sparsity changes in practice: the quality of the generated graphs, memory use, and the time needed to produce a valid result. The experiments reveal a trade-off: sparse generation saves memory, but enforcing the graph constraints remains costly.

The implementation uses PyTorch and includes saved models, recorded results, and commands to run and verify the experiments.

## What the experiments show

The main study trained nine models across three random seeds and evaluated 2,880 generated graphs. The datasets contain conditioned stochastic block models (SBMs) and planar graphs, each with 32 nodes and 64 edges.

All evaluated generated graphs satisfied the connectivity, edge-budget, and degree constraints requested during generation.

### Runtime and memory

To study inference scaling, the models were also evaluated at larger graph sizes. At 512 nodes, using one CPU thread, four generation steps, and a batch size of one, the measurements were:

| Measurement | Dense | Sparse |
|---|---:|---:|
| Complete generation, median time (ms) | 88.22 | 288.89 |
| Process memory, sampled RSS (MiB) | 443.64 | 274.47 |

Sparse generation used **38.1% less process memory**, but took **3.27 times longer** overall.

Looking at candidate scoring alone gives a different picture. With the same sparse model weights, scoring the sparse candidate set was **25.4 times faster** than scoring all possible pairs. The cost of handling the constraints dominated the complete generation time.

![Runtime, memory use, and candidate counts as graph size increases](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/figures/research/scaling.png?raw=true)

### Graph quality

The sparse models reproduced clustering statistics less accurately than dense diffusion, and none of the generators produced planar graphs in the evaluated samples.

Planarity was measured separately from the constraints enforced during generation. A graph could therefore satisfy the requested connectivity, edge-budget, and degree constraints while still being nonplanar.

The figure below compares degree, clustering, and spectral statistics. Lower MMD² values indicate closer agreement with the test distribution.

![Degree, clustering, and spectral MMD² for SBM and planar graph datasets](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/figures/research/quality.png?raw=true)

All models were trained on 32-node graphs. Quality was evaluated at that size; larger graphs were used only to measure how inference time and memory use scale.

### Moving agents

The project also includes an experiment with moving agents. It compares rebuilding connections at each frame, repairing the previous graph while retaining connections, and adding learned scores to the retention strategy.

The figure shows changes in connections between frames, algebraic connectivity, and generation time.

![Connection changes, algebraic connectivity, and runtime in the moving-agent experiment](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/figures/research/dynamic.png?raw=true)

For the complete measurements and experimental protocol, see the [detailed results](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/docs/RESULTS.md) and [methods](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/docs/METHODS.md).

## Getting started

Use Python 3.12 and run the commands below from the repository root.

### Install the project

Create a virtual environment, activate it, and install the dependencies and package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

### Generate graphs with the saved sparse model

This example generates graphs with 32 nodes, an edge budget of 64, and a maximum degree of six. Results are written to `generated/demo`.

```bash
python -m frugal_graphs.research sample --nodes 32 --budget 64 --degree-cap 6 --output generated/demo
```

### Restore and check the recorded experiment

Restore the recorded results, then verify them by recomputing metrics and checking graph constraints:

```bash
python -m frugal_graphs.research restore --output generated/recorded
python -m frugal_graphs.research verify --output generated/recorded
```

### Run the tests and a short experiment

Run the tests, then use the smoke configuration for a short experiment and verify its outputs:

```bash
python -m pytest -q
python -m frugal_graphs.research run --smoke --output generated/smoke
python -m frugal_graphs.research verify --output generated/smoke
```

The full experiment settings and random seeds are available in [configs/research.json](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/configs/research.json).

## License

This project is released under the [MIT License](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/LICENSE).

Copyright © 2026 Alexandre Autran.
