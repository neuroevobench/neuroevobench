from evosax import Strategies
from neuroevobench.problems.mnist_classify import MNIST_Classify_Policy
from neuroevobench.problems.mnist_classify import MNIST_Classify_Task
from neuroevobench.problems.mnist_classify import MNIST_Classify_Evaluator


def main(config, log):
    """Running an ES loop on MNIST classification task."""
    # 1. Create placeholder env to get number of actions for policy init
    policy = MNIST_Classify_Policy(
        hidden_dims=config.model_config.num_hidden_layers
        * [config.model_config.num_hidden_units],
    )

    # 2. Define train/test task based on configs/eval settings
    train_task = MNIST_Classify_Task(
        config.env_name, config.task_config.batch_size, test=False
    )
    test_task = MNIST_Classify_Task(config.env_name, 0, test=True)

    # 3. Setup task evaluator with strategy and policy
    evaluator = MNIST_Classify_Evaluator(
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
