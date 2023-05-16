# `NeuroEvoBench`: Benchmarking Neuroevolution Methods for Machine Learning Applications 🦕 🦖 🐢

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evosax`](https://github.com/RobertTLange/evosax/).

## Installation & Setup

```
conda create -n es_bench python=3.9
source activate es_bench
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

```
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163
```

## Running the Benchmarks for a Single ES + Problem

### Launching a Single Configuration & Single Seed Run

```
cd examples/brax
python train.py -config train.yaml
```

### Launching a Multi-Seed Grid Search

```
cd research/search
neb -config train.yaml
```

### TODOs

- [ ] Add rliable metrics
