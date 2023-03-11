from evosax import Strategies
from neuroevobench.problems.addition import AdditionPolicy
from neuroevobench.problems.addition import AdditionTask
from neuroevobench.problems.addition import AdditionEvaluator


def main(config, log):
    """Running an ES loop on Brax task."""
    # 1. Create placeholder env to get number of actions for policy init
    policy = AdditionPolicy(hidden_dims=config.model_config.num_hidden_units)

    # 2. Define train/test task based on configs/eval settings
    train_task = AdditionTask(
        config.task_config.batch_size,
        seq_length=config.task_config.seq_length,
        seed_id=config.seed_id,  # Fix seed for data generation
        test=False,
    )
    test_task = AdditionTask(
        10000,
        seq_length=config.task_config.seq_length,
        seed_id=config.seed_id,  # Fix seed for data generation
        test=True,
    )

    # 3. Setup task evaluator with strategy and policy
    evaluator = AdditionEvaluator(
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
