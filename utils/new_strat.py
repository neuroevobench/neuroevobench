import os
import jax
import jax.numpy as jnp
from evosax import Strategies, NetworkMapper
from evosax.problems import ClassicFitness, GymFitness
from evosax.utils import ParameterReshaper, FitnessShaper


def run_strategy(strategy_name: str = "Sep_CMA_ES"):
    print(f"EVALUATE {strategy_name}")
    rng = jax.random.PRNGKey(0)
    # Run Strategy on 2D Rosenbrock Function
    popsize = 20
    num_dims = 2
    evaluator = ClassicFitness("rosenbrock", num_dims)
    strategy = Strategies[strategy_name](popsize=popsize, num_dims=num_dims)
    params = strategy.default_params
    state = strategy.initialize(rng, params)

    fitness_log = []
    print(f"START ROSENBROCK VALIDATION")

    for t in range(50):
        rng, rng_eval, rng_iter = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_iter, state, params)
        fitness = evaluator.rollout(rng, x)
        state = strategy.tell(x, fitness, state, params)
        best_id = jnp.argmin(fitness)
        fitness_log.append(fitness[best_id])
        print(t + 1, fitness.mean())

    # Run Strategy on CartPole MLP
    popsize = 100
    evaluator = GymFitness("CartPole-v1", num_env_steps=200, num_rollouts=16)

    network = NetworkMapper["MLP"](
        num_hidden_units=64,
        num_hidden_layers=2,
        num_output_units=2,
        hidden_activation="relu",
        output_activation="categorical",
    )
    pholder = jnp.zeros((1, evaluator.input_shape[0]))
    params = network.init(
        rng,
        x=pholder,
        rng=rng,
    )

    reshaper = ParameterReshaper(params["params"])
    evaluator.set_apply_fn(reshaper.vmap_dict, network.apply)

    strategy = Strategies[strategy_name](
        popsize=popsize, num_dims=reshaper.total_params
    )
    params = strategy.default_params
    state = strategy.initialize(rng, params)

    fit_shaper = FitnessShaper(maximize=True)

    print(f"START CARTPOLE - MLP VALIDATION - {reshaper.total_params}")
    for t in range(200):
        rng, rng_eval, rng_iter = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_iter, state, params)
        x_re = reshaper.reshape(x)
        fitness = evaluator.rollout(rng_eval, x_re).mean(axis=1)
        fit_re = fit_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state, params)
        best_id = jnp.argmax(fitness)
        print(t + 1, fitness.mean(), fitness.max())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-strategy",
        "--strategy_name",
        type=str,
        default="CMA_ES",
        help="Name of strategy to evaluate.",
    )
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_strategy(args.strategy_name)
