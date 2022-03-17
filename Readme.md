# Benchmark Utilities for `evosax` Strategies 🦕 🦖 🐢

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evosax`](https://github.com/RobertTLange/evosax/).

## Installation

```
pip install evosax
```

## Running the Benchmarks for a Single ES + Problem

Brax: Can do ca. 3 generations/min of brax on 8 CPU with 16 pmap devices
-> ca. 3/6 hours per run (2.5/5 days for 20x IMP)
-> 15 nodes = 30 parallel runs 
-> Ca. 10 batches for 10x10x3 = 2.5-4 days
-> Ca. 3 batches for 5x5x3 = 18 hours

MNIST: Can do ca. 40 generations of mnist/3min on 10 CPU with 20 pmap devices
-> ca. 2.5 hours per run (2.5 days for 20x IMP)
-> 15 nodes = 30 parallel runs
-> Ca. 10 batches for 10x10x3 = 1 day
-> Ca 3 batches for 5x5x3 = 7.5 hours

Full grid run for one ES -> 10 min + 2.5 days + 1 day => 2 per week?


CHECK INSTABILITY IN PGPE!!!!!!!!!!!!!!!!

### Hyperparameter Ranges (10x10 and 5x5 grids)

#### Open ES (Adam)

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1

#### PGPE (Adam)

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1
- no fitness reshape + 0.1 elite ratio


#### ARS (Adam)

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1
- no fitness reshape + 0.1 elite ratio

--------------------------------------
#### CMA-ES

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1


#### Sep-CMA-ES

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1

#### Simple Genetic

- lrate_init: begin: 0.001, end: 0.04
- sigma_init: begin: 0.01, end: 0.1