# `NeuroEvoBench`: Benchmarking Neuroevolution 🦕 🦖 🐢

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