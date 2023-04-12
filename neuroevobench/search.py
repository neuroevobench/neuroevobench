import os
import copy
from mle_hyperopt import RandomSearch
from neuroevobench.problems import neb_eval_loops
from neuroevobench.hyperparams import HyperSpace


def neb_search_loop(config, log):
    """Run a random search over EO algorithm parameters/config."""
    # Setup the strategy search space for sequential evaluation
    hyperspace = HyperSpace(config.strategy_name)
    hyper_strategy = RandomSearch(
        **hyperspace.space,
        search_config=config.search_config,
        maximize_objective=True,
        seed_id=config.seed_id,
        verbose=True,
    )
    log_name = f"search_log_seed_{config.seed_id}"

    # Run the random search hyperparameter optimization loop
    for search_iter in range(config.num_hyper_search_iters):
        # Augment the default params with the proposed parameters
        proposal_params = hyper_strategy.ask()
        eval_config = copy.deepcopy(config)
        for k, v in proposal_params.items():
            eval_config.es_config[k] = v
        # Evaluate the parameter config by running a ES loop
        performance, solution = neb_eval_loops[config.problem_type](
            search_iter, eval_config, log
        )

        # Update search strategy - Note we minimize!
        hyper_strategy.tell(proposal_params, performance)

        # Store the model of the best member over all search iterations
        if hyper_strategy.improvement(performance):
            log.save_model(solution)
        hyper_strategy.save(
            os.path.join(log.experiment_dir, log_name + ".yaml")
        )

    # Create a plot of evolution of best configuration over search
    hyper_strategy.plot_best(
        os.path.join(log.experiment_dir, log_name + ".png")
    )


def mle_neb_search():
    """Run search via mle interface."""
    # Attempt running experiment using mle-infrastructure
    from mle_toolbox import MLExperiment

    mle = MLExperiment(config_fname="train.yaml")
    neb_search_loop(mle.train_config, mle.log)