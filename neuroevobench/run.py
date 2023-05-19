import copy
from neuroevobench.problems import neb_eval_loops


def neb_run_config(config, log):
    """Run config evaluation on NEB problem."""
    eval_config = copy.deepcopy(config)
    # Evaluate the parameter config by running a ES loop
    # NOTE: Seed id is passed via eval_config
    performance, solution = neb_eval_loops[config.problem_type](
        eval_config, log
    )
    # Store the model of the best member over all search iterations
    log.save_model(solution)


def mle_neb_eval():
    """Run search via mle interface."""
    # Attempt running experiment using mle-infrastructure
    from mle_toolbox import MLExperiment

    mle = MLExperiment(config_fname="train.yaml")
    neb_run_config(mle.train_config, mle.log)
