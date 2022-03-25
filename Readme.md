# Benchmark Utilities for `evosax` Strategies 🦕 🦖 🐢

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evosax`](https://github.com/RobertTLange/evosax/).

## Installation

```
pip install -r requirements.txt
```

## Running the Benchmarks for a Single ES + Problem

### Launching a Single Configuration & Single Seed Run

```
python train -config configs/<ES>/<problem>.yaml
```

### Launching a Multi-Seed Grid Search

```
mle run configs/<ES>/search_<problem>.yaml
```

## Experiment Plans

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

RERUN PGPE FOR CART WITHOUT ZSCORING?

- [ ] Brax experiments (90 runs on GPU)
- [ ] Runtimes ask/tell


--------------------------------------
### Brax Large Experiments

- 'ant', 'halfcheetah', 'hopper', 'reacher', 'walker2d'
- 5 envs x 3 seeds x 6 algorithms = 90
- Run on V100S for time elapsed measurements!

--------------------------------------
- Answer Tom paper ideas
- Conferences:
    - AAAI Paper + Code clean up
    - ICLR Paper + Code clean up
- Orga:
    - [x] Email Peich TU for Bescheinigung Visa
    - [x] Book flight Chris Rom
    - Beurlaubung Studium tuPort
    - Passport sticker Berlin
    - Travel expenses TU (Portugal & AAAI fees)
    - Ask Sebastian about Summer School? Answer Matteo
- Twitter/SM:
    - [x] Promote podcast/linkedin/homepage
    - [x] Promote evosax talk
    - Read paper Kirsch Symmetries
    - Prepare ML Collage
- Code ES algos
    - Scalable VkD-CMA-ES
    - Scalable LM-CMA-ES
    - Scalable RmES
    - sNES
- Setup new LTH MNIST Pipeline
    - Restructure configs/train scripts
    - Run base experiments for MNIST/F-MNIST
    - Investigate transfer (tasks, multi-ES, GD-ES)


```python
# Evaluation Function
def eval_fn(seed, params):
	score = ...  # evaluate params on seed
	return score

member_scores = []

# Loop over population members
for params in pop_members:
	seed_scores = []
    # Loop over stochastic evaluations
	for seed_id in num_evals:
    	seed_scores.append(eval_fn(seed_id, params))
    # Store mean performance
    member_scores.append(mean(seed_scores))

# Vectorize over stochastic evaluations
score_eval = jax.vmap(eval_fn, in_axes=(0, None))
# Device-Parallelize over population members
pop_eval = jax.pmap(score_eval, in_axes=(None, 0))
# Execute both map strategies (seed & members)
member_scores = pop_eval(num_evals, pop_members).mean(axis=1)
```