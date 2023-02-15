import envpool
from evosax import Strategies
from evosax_benchmark.atari import AtariEvaluator
from evosax_benchmark.atari import AtariPolicy


def main(config, log):
    """Running an ES loop on ATARI task."""
    # 1. Create placeholder env to get number of actions for policy init
    env = envpool.make_gym(config.task_config.env_name, num_envs=1)
    policy = AtariPolicy(num_actions=env.action_space.n, **config.model_config)

    # TODO(RobertTLange): Define train/test task outside of evaluator?
    # 2. Setup task evaluator with strategy and policy
    evaluator = AtariEvaluator(
        policy=policy,
        popsize=config.popsize,
        es_strategy=Strategies[config.strategy_name],
        es_config=config.es_config,
        es_params=config.es_params,
        task_config=config.task_config,
    )

    # 3. Run the ES loop with logging
    evaluator.run(
        config.seed_id,
        config.num_generations,
        config.eval_every_gen,
        log=log,
    )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/train.yaml")
    main(mle.train_config, mle.log)
