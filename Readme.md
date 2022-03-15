# Benchmark Utilities for `evosax` Strategies 🦕 🦖 🐢

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evosax`](https://github.com/RobertTLange/evosax/).

## Installation

```
pip install evosax
```

## Running the Benchmarks for a Single ES + Problem

Can do ca. 3 generations of brax on 8 CPU with 16 pmap devices
-> ca. 3 hours per run (2.5 days for 20x IMP)
-> 14 nodes = 28 parallel runs 
-> Ca. 12 batches for 10x10x3 = 36 hours
-> Ca. 3 batches for 5x5x3 = 9 hours

### Hyperparameter Ranges (10x10 and 5x5 grids)

#### Open ES

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1

#### PGPE

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1


#### ARS

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1


#### Simple Genetic

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1


#### CMA-ES

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1


#### Sep-CMA-ES

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1