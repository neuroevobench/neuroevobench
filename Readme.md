# `NeuroEvoBench`: Benchmarking Neuroevolution Methods for Machine Learning Applications 🦕 🦖 🐢

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evosax`](https://github.com/RobertTLange/evosax/).

## Installation

```
pip install -r requirements.txt
```

## Running the Benchmarks for a Single ES + Problem

### Launching a Single Configuration & Single Seed Run

```
python train.py -config configs/<ES>/<problem>.yaml
```

### Launching a Multi-Seed Grid Search

```
mle run configs/<ES>/search_<problem>.yaml
```

### TODOs

- [ ] Add rliable metrics
- [ ] Write CIFAR-10 evaluator
- [ ] Add storage of full ES state
- [ ] Add mle-hyperopt search and configurations for search
- [ ] Run first brax search (500 Steps)

- [ ] Add iter_id to all evaluators -> write general abstraction
- [ ] Add fitness/solution retrieval to all evaluators