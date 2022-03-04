import jax
import jax.numpy as jnp
from evosax import (
    Strategies,
    FitnessShaper,
    ParameterReshaper,
    NetworkMapper,
    ESLog,
)
from problems import setup_es_problem


def main(config, log):
    """Running an ES loop."""
    # Setup task & network apply function.
    rng, train_evaluator, test_evaluator = setup_es_problem(config)

    # Setup network & generate placeholder params.
    network = NetworkMapper[config.network_name](
        **config.network_config.toDict()
    )
    params = network.init(
        rng,
        x=jnp.zeros(train_evaluator.input_shape),
        rng=rng,
    )

    # Setup Parameter Shaper & the problem apply fn.
    param_reshaper = ParameterReshaper(params["params"])
    train_evaluator.set_apply_fn(param_reshaper.vmap_dict, network.apply)
    test_evaluator.set_apply_fn(param_reshaper.vmap_dict, network.apply)

    # Setup ES.
    strategy = Strategies[config.es_name](
        param_reshaper.total_params, **config.es_config.toDict()
    )
    es_params = strategy.default_params
    for k, v in config.es_params.items():
        es_params[k] = v  # Overwrite default params
    es_state = strategy.initialize(rng, es_params)

    # Setup Fitness Shaping.
    fit_shaper = FitnessShaper(**config.fitness_config.toDict())

    es_logging = ESLog(
        param_reshaper.total_params,
        config.num_generations,
        top_k=5,
        maximize=config.fitness_config.maximize,
    )
    es_log = es_logging.initialize()

    # Run ES Loop.
    for gen in range(config.num_generations):
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        # Ask for new parameter proposals and reshape into paramter trees.
        x, es_state = strategy.ask(rng_ask, es_state, es_params)
        reshaped_params = param_reshaper.reshape(x)

        # Rollout population performance, reshape fitness & update strategy.
        fitness = train_evaluator.rollout(rng_eval, reshaped_params)
        fit_re = fit_shaper.apply(x, fitness.mean(axis=1))
        es_state = strategy.tell(x, fit_re, es_state, es_params)

        # Update the logging instance.
        es_log = es_logging.update(es_log, x, fitness.mean(axis=1))
        log.update(
            {
                "num_gens": gen + 1,
                "num_total_steps": (gen + 1)
                * config.es_config.popsize
                * train_evaluator.steps_per_member,
            },
            {
                "perf_best": es_log["log_top_1"][gen],
                "perf_best_top": es_log["log_top_mean"][gen],
                "perf_gen_mean": es_log["log_gen_mean"][gen],
                "perf_gen_best": es_log["log_gen_1"][gen],
            },
        )
    return


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    mle = MLExperiment(config_fname="configs/OpenES/cartpole.yaml")
    main(mle.train_config, mle.log)
