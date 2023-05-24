import os
import copy
from mle_toolbox import load_result_logs
from mle_hyperopt.utils import load_yaml
from neuroevobench.problems import neb_eval_loops


def neb_best_eval(config, log):
    """Load best search config and run evaluation again."""
    meta_log, hyper_log = load_result_logs(config.results_dir)

    run_id = hyper_log.filter(
        {"strategy_name": config.strategy_name}
    ).run_id.iloc[0]

    # Get all yaml files in the experiment directory
    for file in os.listdir(meta_log[run_id].meta.experiment_dir):
        if file.endswith(".yaml"):
            # Check if "search" is in file name
            if "best_config" in file:
                search_fname = os.path.join(
                    meta_log[run_id].meta.experiment_dir, file
                )

    # Load meta and hyper log - extract best parameters
    loaded_params = load_yaml(search_fname, keys_to_list=False)["config"]

    print(f"Loaded parameters for {config.strategy_name}: {search_fname}")
    print(loaded_params)
    # Augment the default params with the best search parameters
    eval_config = copy.deepcopy(config)
    for k, v in loaded_params.items():
        eval_config.es_config[k] = v
    # Evaluate the parameter config by running a ES loop
    # NOTE: Seed id is passed via eval_config
    performance, solution = neb_eval_loops[config.problem_type](
        eval_config,
        log,
    )
    # Store the model of the best member over all search iterations
    log.save_model(solution)


def mle_neb_eval():
    """Run search via mle interface."""
    # Attempt running experiment using mle-infrastructure
    from mle_toolbox import MLExperiment

    mle = MLExperiment(config_fname="train.yaml")
    neb_best_eval(mle.train_config, mle.log)
