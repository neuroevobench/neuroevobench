# `NeuroEvoBench`: Benchmarking Neuroevolution Methods for Machine Learning Applications 🦕 🦖 🐢
<a href="https://github.com/neuroevobench/neuroevobench/blob/main/docs/logo.png"><img src="https://github.com/neuroevobench/neuroevobench/blob/main/docs/logo.png" width="200" align="right" /></a>
This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evosax`](https://github.com/RobertTLange/evosax/) and [`EvoJAX`](https://github.com/google/evojax).

The benchmark is meant to facilitate benchmarking of evolutionary optimization (EO) methods specifically tailored towards neuroevolution methods.

All task evaluations are written in JAX so that population evaluations can be parallelized on accelerators (GPU/TPU). This reduces the evaluation times significantly.

You can get started here 👉 [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuroevobench/neuroevobench/blob/main/examples/neb_introduction.ipynb)

## Installation & Setup

```
# Create a clean conda environment
conda create -n es_bench python=3.9
source activate es_bench
# Install a GPU/TPU compatible jax/jaxlib version
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Install the neuroevobench benchmark from PyPi
pip install neuroevobench
```


## Task Availability

![](https://github.com/neuroevobench/neuroevobench/blob/main/docs/task_overview.png)

## Basic `NeuroEvoBench` Usage

### Individual Benchmark Tasks

- Note that the strategy to be evaluated has to follow the [`evosax`](https://github.com/RobertTLange/evosax/) `ask`-`tell` API.

```python
from evosax import Strategies
from neuroevobench.problems.cifar import CifarPolicy
from neuroevobench.problems.cifar import CifarTask
from neuroevobench.problems.cifar import CifarEvaluator

# 1. Create policy for task (CNN classifier)
policy = CifarPolicy()

# 2. Define train/test task based on configs/eval settings
train_task = CifarTask(config.task_config.batch_size, test=False)
test_task = CifarTask(10000, test=True)

# 3. Setup task evaluator with strategy and policy
evaluator = CifarEvaluator(
    policy=policy,
    train_task=train_task,
    test_task=test_task,
    popsize=config.popsize,
    es_strategy=Strategies[config.strategy_name],
    es_config=config.es_config,
    es_params=config.es_params,
    seed_id=config.seed_id,
    log=log,
)

# 4. Run the ES loop with logging
evaluator.run(config.num_generations, config.eval_every_gen)
```

### Running Parameter Search Sweeps

- Please visit [`neuroevobench-analysis`](https://github.com/neuroevobench/neuroevobench-analysis) for example configurations used for the 10 EO baselines and 9 neuroevolution tasks.

- You can execute random search sweeps, multi-seed evaluations for the best found settings and individual training runs via the following command line shortcuts:

```
neb-search --config_fname ${CONFIG_FNAME} --seed_id ${SEED_ID} --experiment_dir ${EXPERIMENT_DIR}
neb-eval --config_fname ${CONFIG_FNAME} --seed_id ${SEED_ID} --experiment_dir ${EXPERIMENT_DIR}
neb-run --config_fname ${CONFIG_FNAME} --seed_id ${SEED_ID} --experiment_dir ${EXPERIMENT_DIR}
```