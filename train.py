import jax
import jax.numpy as jnp
from evosax import (
    Strategies,
    FitnessShaper,
    ParameterReshaper,
    NetworkMapper,
    ProblemMapper,
    ESLog,
)


def setup_es_problem(config):
    rng = jax.random.PRNGKey(config.seed_id)
    train_eval = ProblemMapper[config.problem_name](
        **config.problem_train_config.toDict(), test=False
    )
    test_eval = ProblemMapper[config.problem_name](
        **config.problem_test_config.toDict(), n_devices=1, test=True
    )
    return rng, train_eval, test_eval


def main(config, log):
    """Running an ES loop."""
    # Setup task & network apply function.
    rng, train_evaluator, test_evaluator = setup_es_problem(config)

    # Setup network & generate placeholder params.
    network = NetworkMapper[config.network_name](
        **config.network_config.toDict(),
        num_output_units=train_evaluator.action_shape,
    )

    if config.network_name == "MLP":
        params = network.init(
            rng,
            x=jnp.ones([1, train_evaluator.input_shape[0]]),
            rng=rng,
        )
    elif config.network_name == "CNN":
        params = network.init(
            rng,
            x=jnp.ones(train_evaluator.input_shape),
            rng=rng,
        )
    else:
        params = network.init(
            rng,
            x=jnp.ones([1, train_evaluator.input_shape[0]]),
            carry=network.initialize_carry(),
            rng=rng,
        )

    # Setup Parameter Shaper & the problem apply fn.
    train_param_reshaper = ParameterReshaper(params["params"])
    test_param_reshaper = ParameterReshaper(params["params"], n_devices=1)
    if config.network_name != "LSTM":
        train_evaluator.set_apply_fn(
            train_param_reshaper.vmap_dict, network.apply
        )
        test_evaluator.set_apply_fn(
            test_param_reshaper.vmap_dict, network.apply
        )
    else:
        train_evaluator.set_apply_fn(
            train_param_reshaper.vmap_dict,
            network.apply,
            network.initialize_carry,
        )
        test_evaluator.set_apply_fn(
            test_param_reshaper.vmap_dict,
            network.apply,
            network.initialize_carry,
        )

    # Setup ES.
    strategy = Strategies[config.es_name](
        train_param_reshaper.total_params, **config.es_config.toDict()
    )
    es_params = strategy.default_params
    for k, v in config.es_params.items():
        es_params[k] = v  # Overwrite default params
    es_state = strategy.initialize(rng, es_params)

    # Setup Fitness Shaping.
    fit_shaper = FitnessShaper(**config.fitness_config.toDict())

    es_logging = ESLog(
        train_param_reshaper.total_params,
        config.num_generations,
        top_k=5,
        maximize=config.fitness_config.maximize,
    )
    es_log = es_logging.initialize()
    print(f"START EVOLVING {train_param_reshaper.total_params} PARAMS.")
    print("Config", config.es_params)
    # Run ES Loop.
    for gen in range(config.num_generations):
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        # Ask for new parameter proposals and reshape into paramter trees.
        x, es_state = strategy.ask(rng_ask, es_state, es_params)
        reshaped_params = train_param_reshaper.reshape(x)

        # Rollout population performance, reshape fitness & update strategy.
        fitness = train_evaluator.rollout(rng_eval, reshaped_params)
        # Separate loss/acc when evolving classifier
        if type(fitness) == tuple:
            fitness_to_opt, fitness_to_log = fitness[0], fitness[1]
        else:
            fitness_to_opt, fitness_to_log = fitness, fitness
        fit_re = fit_shaper.apply(x, fitness_to_opt.mean(axis=1))
        es_state = strategy.tell(x, fit_re, es_state, es_params)

        # Update the logging instance.
        es_log = es_logging.update(es_log, x, fitness_to_log.mean(axis=1))

        # Sporadically evaluate the mean & best performance on test evaluator.
        if (gen + 1) % config.evaluate_every_gen == 0:
            rng, rng_test = jax.random.split(rng)
            # Stack best params seen & mean strategy params for eval
            best_params = es_log["top_params"][0]
            mean_params = es_state["mean"]
            x_test = jnp.stack([best_params, mean_params], axis=0)
            reshaped_test_params = test_param_reshaper.reshape(x_test)
            test_fitness = test_evaluator.rollout(
                rng_test, reshaped_test_params
            )

            # Separate loss/acc when evolving classifier
            if type(test_fitness) == tuple:
                test_fitness_to_log = test_fitness[1].mean(axis=1)
            else:
                test_fitness_to_log = test_fitness.mean(axis=1)
            log.update(
                {
                    "num_gens": gen + 1,
                    "num_total_steps": (gen + 1)
                    * config.es_config.popsize
                    * train_evaluator.steps_per_member,
                },
                {
                    "train_perf_best": es_log["log_top_1"][gen],
                    "train_perf_gen_mean": es_log["log_gen_mean"][gen],
                    "test_perf_best": test_fitness_to_log[0],
                    "test_perf_strat_mean": test_fitness_to_log[1],
                },
                save=True,
            )

        # Store the es log as a pickle object in extra/ subdir
        log.save_extra(es_log, f"es_log_{config.seed_id}.pkl")


if __name__ == "__main__":
    from mle_toolbox import MLExperiment
    from mle_toolbox.utils import get_jax_os_ready

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/Open_ES/cartpole.yaml")
    get_jax_os_ready(
        num_devices=mle.train_config.num_devices,
        device_type=mle.train_config.device_type,
    )
    main(mle.train_config, mle.log)
