import os
import copy
from brax.envs import create
from evosax import Strategies
from neuroevobench.problems.brax import BraxPolicy
from neuroevobench.problems.brax import BraxTask
from neuroevobench.problems.brax import BraxEvaluator

from mle_hyperopt import RandomSearch
from neuroevobench.hyperparams import HyperSpace


def search(mle, config, log):
    """Run a random search over ES strategy parameters."""
    # Setup the strategy search space for sequential evaluation
    hyperspace = HyperSpace(config.strategy_name, "brax")
    hyper_strategy = RandomSearch(
        **hyperspace.space_dict,
        search_config=config.search_config,
        maximize_objective=True,
        seed_id=config.seed_id,
        verbose=True,
    )

    # Run the random search hyperparameter optimization loop
    for search_iter in range(config.num_hyper_search_iters):
        # Augment the default params with the proposed parameters
        proposal_params = hyper_strategy.ask()
        eval_config = copy.deepcopy(config)
        for k, v in proposal_params.items():
            eval_config.es_config[k] = v
        # Evaluate the parameter config by running a ES loop
        performance, solution = run(search_iter, eval_config, log)

        # Update search strategy - Note we minimize!
        hyper_strategy.tell(proposal_params, performance)

        # Store the model of the best member over all search iterations
        if hyper_strategy.improvement(performance):
            mle.log.save_model(solution)
        log_name = f"search_log_seed_{mle.seed_id}"
        hyper_strategy.save(
            os.path.join(mle.experiment_dir, log_name + ".yaml")
        )

    # Create a plot of evolution of best configuration over search
    hyper_strategy.plot_best(
        os.path.join(mle.experiment_dir, log_name + ".png")
    )


def run(search_iter, config, log):
    """Running an ES loop on Brax task."""
    # 1. Create placeholder env to get number of actions for policy init
    env = create(env_name=config.env_name, legacy_spring=True)
    policy = BraxPolicy(
        input_dim=env.observation_size,
        output_dim=env.action_size,
        hidden_dims=config.model_config.num_hidden_layers
        * [config.model_config.num_hidden_units],
    )

    # 2. Define train/test task based on configs/eval settings
    train_task = BraxTask(
        config.env_name, config.task_config.max_steps, test=False
    )
    test_task = BraxTask(
        config.env_name, config.task_config.max_steps, test=True
    )

    # 3. Setup task evaluator with strategy and policy
    evaluator = BraxEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=config.popsize,
        es_strategy=Strategies[config.strategy_name],
        es_config=config.es_config,
        es_params=config.es_params,
        num_evals_per_member=config.task_config.num_evals_per_member,
        seed_id=config.seed_id,
        log=log,
        iter_id=search_iter,
    )

    # 4. Run the ES loop with logging
    evaluator.run(config.num_generations, config.eval_every_gen)

    # 5. Return mean params and final performance
    return evaluator.fitness_eval, evaluator.solution_eval


if __name__ == "__main__":
    # Attempt running experiment using mle-infrastructure
    from mle_toolbox import MLExperiment

    mle = MLExperiment(config_fname="train.yaml")
    search(mle, mle.train_config, mle.log)
