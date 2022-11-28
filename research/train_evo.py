import numpy as np
from evosax import Strategies
from evojax.obs_norm import ObsNormalizer
from evojax.sim_mgr import SimManager
from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper
from evosax_benchmark.evojax_tasks import get_evojax_task


def main(config, log):
    """Running an ES loop."""
    # Setup task & network apply function & ES.
    train_task, test_task, policy = get_evojax_task(
        config.env_name, config.hidden_layers, config.hidden_dims
    )
    solver = Evosax2JAX_Wrapper(
        Strategies[config.strategy_name],
        param_size=policy.num_params,
        pop_size=config.popsize,
        es_config=config.es_config,
        es_params=config.es_params,
        seed=config.seed_id,
    )
    obs_normalizer = ObsNormalizer(
        obs_shape=train_task.obs_shape, dummy=not config.normalize_obs
    )
    sim_mgr = SimManager(
        policy_net=policy,
        train_vec_task=train_task,
        valid_vec_task=test_task,
        seed=config.seed_id,
        obs_normalizer=obs_normalizer,
        pop_size=config.popsize,
        use_for_loop=False,
        **config.eval_config,
    )

    print(f"START EVOLVING {policy.num_params} PARAMS.")
    # Run ES Loop.
    for gen_counter in range(config.num_generations):
        params = solver.ask()
        scores, _ = sim_mgr.eval_params(params=params, test=False)
        solver.tell(fitness=scores)
        if gen_counter == 0 or (gen_counter + 1) % config.eval_every_gen == 0:
            test_scores, _ = sim_mgr.eval_params(
                params=solver.best_params, test=True
            )
            log.update(
                {
                    "num_gens": gen_counter + 1,
                },
                {
                    "train_perf": float(np.nanmean(scores)),
                    "test_perf": float(np.nanmean(test_scores)),
                },
                model=solver.es_state.mean,
                save=True,
            )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/Sep_CMA_ES/ant.yaml")
    main(mle.train_config, mle.log)
