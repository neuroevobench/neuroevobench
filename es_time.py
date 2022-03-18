import time
import numpy as np
import pandas as pd

import jax
from evosax import Strategies
from evosax.problems import ClassicFitness


def main(
    strategy_name: str = "CMA_ES",
    popsize: int = 100,
    num_dims: int = 10,
    num_mc_evals: int = 100,
):
    """Compute MC Estimates for Run Time of Different CMA Variants."""
    # Setup: Instantiate + init strategy and toy problem
    # Note: Toy problem simply to generate realistic fitness scores
    rng = jax.random.PRNGKey(0)
    strategy = Strategies[strategy_name](popsize, num_dims)
    es_params = strategy.default_params
    es_state = strategy.initialize(rng, es_params)
    evaluator = ClassicFitness(
        fct_name="rosenbrock", num_dims=num_dims, num_rollouts=1, noise_std=0.0
    )

    # Ask and tell once for compilation
    rng, rng_ask, rng_eval = jax.random.split(rng, 3)
    x, es_state = strategy.ask(rng_ask, es_state, es_params)
    fit = evaluator.rollout(rng_eval, x)
    es_state = strategy.tell(x, fit, es_state, es_params)

    # Run Monte-Carlo Experiment
    ask_times, tell_times = [], []

    for mc in range(num_mc_evals + 1):
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        start_t_ask = time.time()
        x, es_state = strategy.ask(rng_ask, es_state, es_params)
        stop_t_ask = time.time()
        fit = evaluator.rollout(rng_eval, x)
        start_t_tell = time.time()
        es_state = strategy.tell(x, fit, es_state, es_params)
        stop_t_tell = time.time()

        # Store computation times
        ask_times.append(stop_t_ask - start_t_ask)
        tell_times.append(stop_t_tell - start_t_tell)

    # Print and return times (mean, std)
    mean_ask, std_ask = np.mean(ask_times[1:]), np.std(ask_times[1:])
    mean_tell, std_tell = np.mean(tell_times[1:]), np.std(tell_times[1:])
    return {
        "num_dims": num_dims,
        "popsize": popsize,
        "mean_ask": mean_ask,
        "std_ask": std_ask,
        "mean_tell": mean_tell,
        "std_tell": std_tell,
    }


if __name__ == "__main__":
    from mle_toolbox.utils import get_jax_os_ready

    get_jax_os_ready(
        num_devices=1,
        device_type="gpu",
    )

    strategy_name = "Sep_CMA_ES"
    results1 = main(strategy_name, 256, 10, 100)
    print(results1)

    results2 = main(strategy_name, 256, 100, 100)
    print(results2)

    results3 = main(strategy_name, 256, 1000, 100)
    print(results3)

    results4 = main(strategy_name, 256, 10000, 100)
    print(results4)

    results5 = main(strategy_name, 256, 100000, 100)
    print(results5)

    results6 = main(strategy_name, 256, 1000000, 100)
    print(results6)

    df = pd.DataFrame(
        [results1, results2, results3, results4, results5, results6]
    )
    print(df)
