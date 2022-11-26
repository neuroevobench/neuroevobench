import numpy as np
from evosax import Strategies
from src.hpo_wrapper import Evosax2HPO_Wrapper
from src.hpo_tasks import HPOBHandler


def main(config, log):
    """Running an ES loop on HPO task."""
    # Setup task & network apply function & ES.
    method = Evosax2HPO_Wrapper(
        Strategies[config.strategy_name],
        popsize=config.popsize,
        es_config=config.es_config,
        es_params=config.es_params,
        seed=config.seed_id,
    )

    # Run ES Loop on HPO tasks -> Return mean timeseries across all tasks
    mean_perf, mean_steps = eval_cont_hbo_sweep(method, config.num_generations)

    # Loop over all mean results and return performance
    for g in range(mean_perf.shape[0]):
        log.update(
            {"num_gens": g, "num_evals": mean_steps[g]},
            {"test_perf": mean_perf[g]},
            save=True,
        )


def eval_cont_hbo_sweep(
    method: Evosax2HPO_Wrapper,
    num_generations: int,
    data_dir: str = "data/hpo/hpob-data/",
    surrogates_dir: str = "data/hpo/saved-surrogates/",
):
    """Runs BBO evaluation on all HPO-B tasks."""
    # Instantiate data/task handler
    hpob_hdlr = HPOBHandler(
        root_dir=data_dir,
        mode="v3-test",
        surrogates_dir=surrogates_dir,
    )

    # Get all 'spaces' (model types to opt hyperparams for)
    all_spaces = hpob_hdlr.get_search_spaces()
    all_data, all_steps = [], []
    # Loop over all spaces (models -> hyperparameters)
    for k in all_spaces:
        data = hpob_hdlr.get_datasets(k)
        # Loop over all datasets (model + datasets -> hyperparameters)
        for d in data:
            num_dims = hpob_hdlr.get_search_space_dim(k)
            method.init_bbo(num_dims)
            acc, steps = hpob_hdlr.evaluate_continuous(
                method,
                search_space_id=k,
                dataset_id=d,
                seed="test0",
                n_trials=num_generations,
            )
            all_data.append(acc)
            all_steps.append(steps)

    stacked_perf = np.stack(all_data)
    stacked_steps = np.stack(all_steps)
    mean_perf = stacked_perf.mean(axis=0)
    mean_steps = stacked_steps.mean(axis=0)
    return mean_perf, mean_steps


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/Sep_CMA_ES/hpo.yaml")
    main(mle.train_config, mle.log)
