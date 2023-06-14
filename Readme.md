# `NeuroEvoBench`: Benchmarking Neuroevolution Methods for Machine Learning Applications 🦕 🦖 🐢
<a href="https://github.com/neuroevobench/neuroevobench/blob/main/docs/logo.png"><img src="https://github.com/neuroevobench/neuroevobench/blob/main/docs/logo.png" width="200" align="right" /></a>
This repository contains the task, experiment protocol and evolutionary optimizer (EO) wrappers for evaluating the performance of new gradient-free optimization methods (e.g. ES 🦎 or GA 🧬). It is powered by [`evosax`](https://github.com/RobertTLange/evosax/) and [`EvoJAX`](https://github.com/google/evojax) for acceleration and distributed rollouts.

**Motivation**: Common black-box optimization benchmarks use synthetic test problems (e.g. BBOB, HPO-B) to test the performance of a method. Furthermore, they do not leverage the recent advances in hardware acceleration. This benchmark, on the other hand, is meant to facilitate benchmarking of evolutionary optimization (EO) methods specifically tailored towards neuroevolution methods. All task evaluations are written in JAX so that population evaluations can be parallelized on accelerators (GPU/TPU). This reduces the evaluation times significantly.

You can get started with the example notebook 👉 [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuroevobench/neuroevobench/blob/main/examples/neb_introduction.ipynb) and check out the documentation page 👉 [<img src="https://img.shields.io/badge/home%20-WWW-white??style=plastic&logo=appveyor"/>](https://sites.google.com/view/neuroevobench) for an overview. Finally, checkout [`neuroevobench-analysis`](https://github.com/neuroevobench/neuroevobench-analysis) to download all experiment data and recreate all paper visualizations.

## Installation & Setup

```
# Create a clean conda environment
conda create -n es_bench python=3.9
source activate es_bench

# Install a GPU/TPU compatible jax/jaxlib version
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install the neuroevobench benchmark from the GitHub repo
pip install -q git+https://github.com/neuroevobench/neuroevobench.git@main

# ... or the latest PyPi release
pip install neuroevobench
```

## Task Availability

<a href="https://github.com/neuroevobench/neuroevobench/blob/main/docs/task_overview.png"><img src="https://github.com/neuroevobench/neuroevobench/blob/main/docs/task_overview.png" width="800" align="center" /></a>

We provide 4 different problem classes with a total of 11 tasks. The majority of tasks are concerned with neuroevolution problems, i.e. the optimization of neural network weights. But we also provide BBOB and HPO-C task wrappers for completeness. You can also add your own tasks, e.g. have a look at the [`addition`](https://github.com/neuroevobench/neuroevobench/tree/main/neuroevobench/problems/addition) task for an easy-to-adapt example.

## Basic `NeuroEvoBench` Usage & Task API

Each specific task requires 3 core ingredients in order to be supported by `NeuroEvoBench`: 

1. `Policy`: Defines the network/BBO substrate to optimize its parameters.
2. `Task`: Defines the task (e.g. rollout of robot policy / loss evaluation of net).
3. `Evaluator`: Ties policy, task evaluation and logging together for EO loop.

Note that the EO strategy to be evaluated has to follow the [`evosax`](https://github.com/RobertTLange/evosax/) `ask`-`tell` API.

### Individual Benchmark Tasks

Below we outline how to put all pieces together and to evaluate a single EO method on a CIFAR-10 classification task using a small All-CNN architecture:

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

### Manually Running a Search Experiment

Below we outline how to run a random search experiment manually for the BBOB task. More specifically, we use the [`SimpleES`](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/simple_es.py) implementation from [`evosax`](https://github.com/RobertTLange/evosax/) and search over different initial mutation strength. The performance is measured as the average fitness over all BBOB functions (negative loss):

```python
import copy
from dotmap import DotMap
from mle_hyperopt import RandomSearch
from neuroevobench.problems import neb_eval_loops

# Setup the random search strategy for sequential evaluation
hyper_strategy = RandomSearch(
    real={
        "sigma_init": {
            "begin": 0.01, "end": 0.5, "prior": "uniform"
    },
    search_config={
        "refine_after": 40,
        "refine_top_k": 10
    },
    maximize_objective=True,
    seed_id=0,
    verbose=True,
)

config = DotMap({
    "strategy_name": "SimpleES",
    "popsize": 10,
    "num_dims": 2,
    "es_config": {},
    "es_params": {},
    "num_eval_runs": 5,
    "seed_id": 42,
    "num_generations": 50
})

# Run the random search hyperparameter optimization loop
for search_iter in range(50):
    # Augment the default params with the proposed parameters
    proposal_params = hyper_strategy.ask()
    eval_config = copy.deepcopy(config)
    for k, v in proposal_params.items():
        eval_config.es_config[k] = v

    # Evaluate the parameter config by running a ES loop
    performance, solution = neb_eval_loops["bbob"](
        eval_config,
        log=None,
        search_iter=search_iter,
    )

    # Update search strategy - Note we minimize!
    hyper_strategy.tell(proposal_params, float(performance))
```


### Benchmarking Your Own Method

You can also benchmark your own EO method. It only has to come with the standard `evosax` ask-tell API. E.g.

```python
from flax import struct
from evosax import Strategy


@struct.dataclass
class EvoState:
    ...

@struct.dataclass
class EvoParams:
    ...

class Your_EO_Method(Strategy):
    def __init__(self, popsize, num_dims, pholder_params, **fitness_kwargs):
        """Your Evolutionary Optimizer"""
        super().__init__(...)

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()

    def initialize_strategy(self, rng, params) -> EvoState:
        """`initialize` the evolution strategy."""
        return EvoState(...)

    def ask_strategy(self, rng, state, params):
        """`ask` for new proposed candidates to evaluate next."""
        x = ...
        return x, state

    def tell_strategy(self, x, fitness, state, params) -> EvoState:
        """`tell` update to ES state."""
        return state
```


You can then pass your custom EO method to the evaluation loop execution function, i.e.:

```Python
performance, solution = neb_eval_loops["bbob"](
    eval_config,
    log=None,
    search_iter=search_iter,
    strategy_class=Your_EO_Method  # Specify EO HERE
)
```

We further provide a full example in the colab notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuroevobench/neuroevobench/blob/main/examples/neb_introduction.ipynb).


### Running Distributed Parameter Search/EO Sweeps

You can execute the random search sweeps, multi-seed evaluations for the best found settings and individual training runs via the following command line shortcuts:

```
neb-search --config_fname ${CONFIG_FNAME} --seed_id ${SEED_ID} --experiment_dir ${EXPERIMENT_DIR}
neb-eval --config_fname ${CONFIG_FNAME} --seed_id ${SEED_ID} --experiment_dir ${EXPERIMENT_DIR}
neb-run --config_fname ${CONFIG_FNAME} --seed_id ${SEED_ID} --experiment_dir ${EXPERIMENT_DIR}
```

For an example of configuration files used for the 10 EO baselines and 9 neuroevolution tasks please visit [`neuroevobench-analysis`](https://github.com/neuroevobench/neuroevobench-analysis).
