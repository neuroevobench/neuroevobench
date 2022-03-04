# Benchmark Utilities for `evosax` Strategies 🦕 🦖 🐢

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evosax`](https://github.com/RobertTLange/evosax/).

## Installation

```
pip install evosax
```

## Running the Benchmarks for a Single ES + Problem

```
python train.py -config configs/<es>/<problem>.yaml
```

5. **[Optional]**: Tune hyperparameters using [`mle-hyperopt`](https://github.com/mle-infrastructure/mle-hyperopt).

```
pip install mle-toolbox
```

You can then specify hyperparameter ranges and the search strategy in a yaml file as follows:

```yaml
num_iters: 25
search_type: "Grid"
maximize_objective: true
verbose: true
search_params:
  real:
    es_config/optimizer_config/lrate_init:
      begin: 0.001
      end: 0.1
      bins: 5
    es_config/init_stdev:
      begin: 0.01
      end: 0.1
      bins: 5
```

Afterwards, you can easily execute the search using the `mle-search` CLI. Here is an example for running a grid search for ARS over different learning rates and perturbation standard deviations via:

```
mle-search train.py -base configs/<es>/<problem>.yaml -search configs/<es>/search.yaml -iters 25 -log log/<es>/<problem>/
```

This will sequentially execute 25 ARS-MNIST evolution runs for a grid of different learning rates and standard deviations. After the search has completed, you can access the search log at `log/<es>/<problem>/search_log.yaml`. Finally, we provide some [utilities](viz_grid.ipynb) to visualize the search results.

## Benchmark Results

### PGPE


|   | Benchmarks | Parameters | Results (Avg) |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/PGPE/cartpole_easy.yaml)| 935.4268 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/PGPE/cartpole_hard.yaml)| 631.1020 |
MNIST	| 90.0 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/PGPE/mnist.yaml)| 0.9743 |
Brax Ant |	3000 (max_iter=1200) |[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/PGPE/brax_ant.yaml)| 6054.3887 |
Waterworld	| 6 (max_iter=500)	 | [Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/PGPE/waterworld.yaml)| 11.6400 |
Waterworld (MA)	| 2 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/PGPE/waterworld_ma.yaml) | 2.0625 |


### OpenES


|   | Benchmarks | Parameters | Results (Avg) |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/OpenES/cartpole_easy.yaml)| 929.4153 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/OpenES/cartpole_hard.yaml)| 604.6940 |
MNIST	| 90.0 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/OpenES/mnist.yaml)| 0.9669 |
Brax Ant |	3000 (max_iter=1200) |[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/OpenES/brax_ant.yaml)| 6726.2100 |
Waterworld	| 6 (max_iter=500)	 | - | - |
Waterworld (MA)	| 2 (max_iter=2000)	| - | - |


*Note*: For the brax environment I reduced the population size from 1024 to 256 and increased the search iterations by the same factor (300 to 1200) in the main run. For the grid search I used a population size of 256 but with 500 iterations.

| Cartpole-Easy  | Cartpole-Hard | MNIST | Brax|
|---|---|---|---|
<img src="https://github.com/RobertTLange/evosax-benchmarks/blob/main/figures/OpenES/cartpole_easy.png?raw=true" alt="drawing" width="200" />|<img src="https://github.com/RobertTLange/evosax-benchmarks/blob/main/figures/OpenES/cartpole_hard.png?raw=true" alt="drawing" width="200" />| <img src="https://github.com/RobertTLange/evosax-benchmarks/blob/main/figures/OpenES/mnist.png?raw=true" alt="drawing" width="200" /> | <img src="https://github.com/RobertTLange/evosax-benchmarks/blob/main/figures/OpenES/brax.png?raw=true" alt="drawing" width="200" /> |
### Augmented Random Search


|   | Benchmarks | Parameters | Results |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/ARS/cartpole_easy.yaml)| 902.107 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/ARS/cartpole_hard.yaml)| 666.6442 |
Waterworld	| 6 (max_iter=500)	 |[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/ARS/waterworld.yaml)| 6.1300 |
Waterworld (MA)	| 2 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/ARS/waterworld_ma.yaml)| 1.4831 |
Brax Ant |	3000 (max_iter=300) |[Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/ARS/brax_ant.yaml)| 3298.9746 |
MNIST	| 90.0 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evosax-benchmarks/blob/main/configs/ARS/mnist.yaml)| 0.9610 |


| Cartpole-Easy  | Cartpole-Hard | MNIST | 
|---|---|---|
<img src="https://github.com/RobertTLange/evosax-benchmarks/blob/main/figures/ARS/cartpole_easy.png?raw=true" alt="drawing" width="200" />|<img src="https://github.com/RobertTLange/evosax-benchmarks/blob/main/figures/ARS/cartpole_hard.png?raw=true" alt="drawing" width="200" />| <img src="https://github.com/RobertTLange/evosax-benchmarks/blob/main/figures/ARS/mnist.png?raw=true" alt="drawing" width="200" /> |