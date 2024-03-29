from brax.v1.envs import create
from evosax import Strategies
from neuroevobench.problems.brax import BraxPolicy
from neuroevobench.problems.brax import BraxTask
from neuroevobench.problems.brax import BraxEvaluator


def main(config, log):
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
    )

    # 4. Run the ES loop with logging
    evaluator.run(config.num_generations, config.eval_every_gen)


if __name__ == "__main__":
    # Attempt running experiment using mle-infrastructure
    # try:
    #     from mle_toolbox import MLExperiment

    #     mle = MLExperiment(config_fname="configs/train.yaml")
    #     main(mle.train_config, mle.log)
    # # mle-infrastructure is not supported - use default utilities
    # except Exception:
    from neuroevobench.utils import CSV_Logger, load_config

    train_config = load_config()
    main(train_config, CSV_Logger("temp/"))
