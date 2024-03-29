import envpool
from evosax import Strategies
from neuroevobench.problems.atari import AtariPolicy
from neuroevobench.problems.atari import AtariTask
from neuroevobench.problems.atari import AtariEvaluator


def main(config, log):
    """Running an ES loop on ATARI task."""
    # 1. Create placeholder env to get number of actions for policy init
    env = envpool.make_gym(config.task_config.env_name, num_envs=1)
    policy = AtariPolicy(num_actions=env.action_space.n, **config.model_config)

    # 2. Define train/test task based on configs/eval settings
    train_task = AtariTask(
        policy,
        popsize=config.popsize,
        **config.task_config,
    )
    test_task = AtariTask(
        policy,
        popsize=2,
        env_name=config.task_config.env_name,
        max_steps=config.task_config.max_steps,
        num_envs_per_member=20,
    )

    # 3. Setup task evaluator with strategy and policy
    evaluator = AtariEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=config.popsize,
        es_strategy=Strategies[config.strategy_name],
        es_config=config.es_config,
        es_params=config.es_params,
        seed_id=config.seed_id,
        log=log,
    )

    # 4. Run the ES loop with logging
    evaluator.run(
        config.num_generations,
        config.eval_every_gen,
    )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="train.yaml")
    main(mle.train_config, mle.log)
