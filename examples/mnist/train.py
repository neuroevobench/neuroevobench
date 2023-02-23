from evosax import Strategies
from evosax_benchmark.mnist import MNISTPolicy
from evosax_benchmark.mnist import MNISTTask
from evosax_benchmark.mnist import MNISTEvaluator


def main(config, log):
    """Running an ES loop on Brax task."""
    # 1. Create placeholder env to get number of actions for policy init
    policy = MNISTPolicy(
        hidden_dims=config.model_config.num_hidden_layers
        * [config.model_config.num_hidden_units],
    )

    # 2. Define train/test task based on configs/eval settings
    train_task = MNISTTask(
        config.env_name, config.task_config.batch_size, test=False
    )
    test_task = MNISTTask(config.env_name, 0, test=True)

    # 3. Setup task evaluator with strategy and policy
    evaluator = MNISTEvaluator(
        policy=policy,
        train_task=train_task,
        test_task=test_task,
        popsize=config.popsize,
        es_strategy=Strategies[config.strategy_name],
        es_config=config.es_config,
        es_params=config.es_params,
        seed_id=config.seed_id,
    )

    # 4. Run the ES loop with logging
    evaluator.run(
        config.num_generations,
        config.eval_every_gen,
        log=log,
    )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    # Setup experiment run (visible GPUs for JAX parallelism)
    mle = MLExperiment(config_fname="configs/train.yaml")
    main(mle.train_config, mle.log)
