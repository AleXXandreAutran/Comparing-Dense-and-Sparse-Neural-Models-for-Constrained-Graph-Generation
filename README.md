# Comparing Dense and Sparse Graph Generation

This project compares dense and sparse neural models for generating graphs. The graphs must stay connected and respect an edge budget. A limit on the number of connections per node can also be added.

The goal is to see how these two approaches affect graph quality, memory use, and generation time. The results show a trade-off: sparse generation uses less memory, but enforcing the constraints still takes time.

The code uses PyTorch. The repository includes saved models, recorded results, and commands to run and check the experiments.

## What the experiments show

The main study trained nine models across three random seeds and evaluated 2,880 generated graphs. It used two types of data: conditioned stochastic block models (SBMs) and planar graphs. Each dataset graph has 32 nodes and 64 edges.

All evaluated generated graphs met the constraints requested during generation: connectivity, edge budget, and maximum degree where specified.

### Runtime and memory

The models were also tested on larger graphs to measure inference time and memory use. The table below shows the results at 512 nodes, with one CPU thread, four generation steps, and a batch size of one:

| Measurement | Dense | Sparse |
|---|---:|---:|
| Complete generation, median time (ms) | 88.22 | 288.89 |
| Process memory, sampled RSS (MiB) | 443.64 | 274.47 |

Sparse generation used **38.1% less process memory**, but took **3.27 times longer** overall.

Scoring candidate edges on its own was much faster. Using the same sparse model weights, scoring a sparse set of candidates was **25.4 times faster** than scoring every possible pair. Handling the constraints took most of the time in the full generation process.

![Runtime, memory use, and candidate counts as graph size increases](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/figures/research/scaling.png?raw=true)

### Graph quality

The sparse models matched the clustering statistics less well than dense diffusion. None of the generators produced planar graphs in the evaluated samples.

Planarity was checked separately and was not enforced during generation. A graph could therefore meet the requested connectivity, edge-budget, and degree limits without being planar.

The figure below compares degree, clustering, and spectral statistics. Lower MMD² values mean a closer match to the test distribution.

![Degree, clustering, and spectral MMD² for SBM and planar graph datasets](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/figures/research/quality.png?raw=true)

All models were trained on 32-node graphs. Graph quality was measured at this size. Larger graphs were used only to study inference time and memory use.

### Moving agents

The project also includes an experiment with agents that move. It compares three ways to update their connections: rebuilding the graph at each frame, keeping and repairing the previous graph, and adding learned scores to the retention strategy.

The figure shows how many connections change between frames, the graph’s algebraic connectivity, and generation time.

![Connection changes, algebraic connectivity, and runtime in the moving-agent experiment](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/figures/research/dynamic.png?raw=true)

The [detailed results](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/docs/RESULTS.md) and [methods](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/docs/METHODS.md) explain the measurements and how the experiments were run.

## Getting started

Use Python 3.12 and run the commands below from the repository root.

### Install the project

Create and activate a virtual environment, then install the dependencies and the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

### Generate graphs with the saved sparse model

This command generates graphs with 32 nodes, an edge budget of 64, and a maximum degree of six. It saves the results in `generated/demo`.

```bash
python -m frugal_graphs.research sample --nodes 32 --budget 64 --degree-cap 6 --output generated/demo
```

### Restore and check the recorded experiment

Restore the recorded results, then recompute the metrics and check that the graphs meet the constraints:

```bash
python -m frugal_graphs.research restore --output generated/recorded
python -m frugal_graphs.research verify --output generated/recorded
```

### Run the tests and a short experiment

Run the tests, try a short experiment with the smoke configuration, and check the results:

```bash
python -m pytest -q
python -m frugal_graphs.research run --smoke --output generated/smoke
python -m frugal_graphs.research verify --output generated/smoke
```

The full experiment settings and random seeds are listed in [configs/research.json](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/configs/research.json).

## License

This project is released under the [MIT License](https://github.com/AleXXandreAutran/Comparing-Dense-and-Sparse-Neural-Models-for-Constrained-Graph-Generation/blob/main/LICENSE).

Copyright © 2026 Alexandre Autran.
